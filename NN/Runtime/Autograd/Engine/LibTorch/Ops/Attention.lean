/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.LibTorch.Ops.Core
public import Mathlib.Algebra.GroupWithZero.Nat
public import NN.Runtime.Autograd.Engine.LibTorch.Convert

/-!
# CUDA Tape Operations: Attention
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace LibTorch

open Spec TorchLean
open TorchLean TorchLean.Tensor

namespace Tape

/-!
## Multi-head self-attention

Forward structure matches `Spec.MultiHeadAttention.forward`:
1. `Q = x @ Wq`, `K = x @ Wk`, `V = x @ Wv`
2. reshape to heads `(numHeads, n, headDim)`
3. attention per head (batched): `softmax(Q Kᵀ / sqrt(headDim)) @ V`
4. combine heads, then output projection `@ Wo`

Masking:
- Blocked entries contribute zero softmax numerator, and fully blocked rows return zero.
- The paired ATen forward/backward calls retain provider-specific state on the forward buffer.
  TorchLean owns the global tape and projection VJPs; LibTorch records no autograd graph.
- The host `Tensor Bool` mask is copied to the device.
-/

namespace Internal

/-- Shared implementation behind the single-sample and batched attention entrypoints. -/
def attention
  {n numHeads dModel headDim : Nat} (_hSeq : n ≠ 0)
  (batch : Nat) (_hBatch : batch ≠ 0) (inputShape outputShape : Shape) (nodeName : String)
  (t : Tape) (wqId wkId wvId woId xId : Nat)
  (mask : Option (Tensor Bool [n, n]) := none) :
  IO (Result (Tape × Nat)) := (do
  let one32 : UInt32 := 1
  let depth1 : UInt32 := 1
  let n32 ← ExceptT.mk (pure <| AnyBuffer.natToU32Checked n)
  let dModel32 ← ExceptT.mk (pure <| AnyBuffer.natToU32Checked dModel)
  let head32 ← ExceptT.mk (pure <| AnyBuffer.natToU32Checked headDim)
  let projDim : Nat := numHeads * headDim
  let proj32 ← ExceptT.mk (pure <| AnyBuffer.natToU32Checked projDim)
  let rows : Nat := batch * n
  let rows32 ← ExceptT.mk (pure <| AnyBuffer.natToU32Checked rows)
  let batchHeads : Nat := batch * numHeads
  let batchHeads32 ← ExceptT.mk (pure <| AnyBuffer.natToU32Checked batchHeads)
  let wq ← ExceptT.mk (pure <| requireValue (t := t) wqId (.dim dModel (.dim projDim .scalar)))
  let wk ← ExceptT.mk (pure <| requireValue (t := t) wkId (.dim dModel (.dim projDim .scalar)))
  let wv ← ExceptT.mk (pure <| requireValue (t := t) wvId (.dim dModel (.dim projDim .scalar)))
  let wo ← ExceptT.mk (pure <| requireValue (t := t) woId (.dim projDim (.dim dModel .scalar)))
  let x ← ExceptT.mk (pure <| requireValue (t := t) xId inputShape)
  -- Flatten the leading batch into the row axis for the shared projections.
  let Q := Buffer.bmm x wq one32 rows32 dModel32 proj32
  let K := Buffer.bmm x wk one32 rows32 dModel32 proj32
  let V := Buffer.bmm x wv one32 rows32 dModel32 proj32
  -- Split heads:
  --   `(batch,n,projDim)` views as `(batch,n,numHeads,headDim)`, then swaps to
  --   `(batch,numHeads,n,headDim)`. The first two axes are folded into the BMM batch axis.
  let dimsView : Array Nat := #[batch, n, numHeads, headDim]
  let dimsHead : Array Nat := #[batch, numHeads, n, headDim]
  let Qh := Buffer.releaseThen Q <| Buffer.swapAdjacentAtDepth Q dimsView depth1
  let Kh := Buffer.releaseThen K <| Buffer.swapAdjacentAtDepth K dimsView depth1
  let Vh := Buffer.releaseThen V <| Buffer.swapAdjacentAtDepth V dimsView depth1
  let scaleDenom : Float := if headDim = 0 then 1.0 else Float.sqrt (Float.ofNat headDim)
  let scale : Float := 1.0 / scaleDenom
  -- Optional mask: `mask[i,j]=true` means allowed.
  let (maskB, hasMask) : Buffer × UInt32 :=
    match mask with
    | none => (Buffer.zeros 0, 0)
    | some m =>
        let mF := Buffer.ofFloatArray (Convert.flattenBoolMask (s := .dim n (.dim n .scalar)) m)
        let inDims : Array Nat := #[n, n]
        let outDims : Array Nat := #[batchHeads, n, n]
        let axisMap : Array Nat := #[0, 1, 2]
        let maskB := Buffer.broadcastTo mF inDims outDims axisMap
        (Buffer.releaseThen mF maskB, 1)
  let result ← liftM <| IO.lazyPure fun _ =>
    Buffer.libTorchAttentionFwd Qh Kh Vh maskB hasMask batchHeads32 n32 head32 scale
  let outHeads ← match result with
    | .ok output => pure output
    | .error message =>
        for buffer in #[Qh, Kh, Vh, maskB] do
          discard <| liftM (Buffer.releaseIO buffer)
        throw message
  -- Combine heads and fold the leading axes back into `rows` for the output projection.
  let swapped := Buffer.swapAdjacentAtDepth outHeads dimsHead depth1
  let concat := swapped
  let y := Buffer.bmm concat wo one32 rows32 proj32 dModel32
  let node : Node :=
    { name := some nodeName
      value := { s := outputShape, buf := y }
      requiresGrad := (t.getNode? wqId).any (·.requiresGrad) ||
        (t.getNode? wkId).any (·.requiresGrad) ||
        (t.getNode? wvId).any (·.requiresGrad) ||
        (t.getNode? woId).any (·.requiresGrad) ||
        (t.getNode? xId).any (·.requiresGrad)
      parents := #[wqId, wkId, wvId, woId, xId]
      cleanup := #[Qh, Kh, Vh, maskB, outHeads, swapped]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outputShape
        -- Backprop through output projection: y = concat @ wo
        let dConcat :=
          Buffer.bmmRightTranspose dLdy.buf wo one32 rows32 dModel32 proj32
        let dWo :=
          Buffer.bmmLeftTranspose concat dLdy.buf one32 proj32 rows32 dModel32
        let dSwapped := dConcat
        let dOutHeads := Buffer.swapAdjacentAtDepth dSwapped dimsView depth1
        let (dQh, dKh, dVh) ←
          match Buffer.libTorchAttentionBwd outHeads dOutHeads with
          | .ok (dq, dk, dv) =>
              pure (Buffer.releaseThen dOutHeads dq, dk, dv)
          | .error message => .error message
        -- Undo the head permutation and view each projection gradient as `(rows, projDim)`.
        let dQ := Buffer.releaseThen dQh <| Buffer.swapAdjacentAtDepth dQh dimsHead depth1
        let dK := Buffer.releaseThen dKh <| Buffer.swapAdjacentAtDepth dKh dimsHead depth1
        let dV := Buffer.releaseThen dVh <| Buffer.swapAdjacentAtDepth dVh dimsHead depth1
        -- Backprop projections Q = x @ wq etc.
        let dxQ := Buffer.bmmRightTranspose dQ wq one32 rows32 proj32 dModel32
        let dxK := Buffer.bmmRightTranspose dK wk one32 rows32 proj32 dModel32
        let dxV := Buffer.bmmRightTranspose dV wv one32 rows32 proj32 dModel32
        let dxQK := Buffer.add dxQ dxK
        let dxRaw := Buffer.add dxQK dxV
        let dx := Buffer.releaseThen dxQ <| Buffer.releaseThen dxK <|
          Buffer.releaseThen dxV <| Buffer.releaseThen dxQK dxRaw
        let dWq := Buffer.bmmLeftTranspose x dQ one32 dModel32 rows32 proj32
        let dWk := Buffer.bmmLeftTranspose x dK one32 dModel32 rows32 proj32
        let dWv := Buffer.bmmLeftTranspose x dV one32 dModel32 rows32 proj32
        let dWv := Buffer.releaseThen dConcat <| Buffer.releaseThen dQ <|
          Buffer.releaseThen dK <| Buffer.releaseThen dV dWv
        pure #[
          (xId,  { s := inputShape, buf := dx }),
          (wqId, { s := .dim dModel (.dim projDim .scalar), buf := dWq }),
          (wkId, { s := .dim dModel (.dim projDim .scalar), buf := dWk }),
          (wvId, { s := .dim dModel (.dim projDim .scalar), buf := dWv }),
          (woId, { s := .dim projDim (.dim dModel .scalar), buf := dWo })
        ] }
  pure (t.addNode node) : ExceptT String IO (Tape × Nat)).run

end Internal

/--
Self-attention with shared projection weights and an optional leading batch dimension.

Without `batch`, the input is `[n, dModel]`; `batch := some b` selects `[b, n, dModel]`.
The head count is `numHeads`, including one for single-head attention.
-/
def attention
  {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (t : Tape) (wqId wkId wvId woId xId : Nat)
  (mask : Option (Tensor Bool [n, n]) := none)
  (batch : Option Nat := none) (hBatch : batch.getD 1 ≠ 0 := by decide) :
  IO (Result (Tape × Nat)) :=
  let shape := match batch with
    | none => Shape.dim n (.dim dModel .scalar)
    | some b => Shape.dim b (.dim n (.dim dModel .scalar))
  Internal.attention
    (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
    h1 (batch.getD 1) hBatch shape shape "attention" t wqId wkId wvId woId xId mask

end Tape

end LibTorch
end Autograd
end Runtime
