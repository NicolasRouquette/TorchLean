/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Ops.Shape

/-!
# CUDA Tape Operations: Matrix, FFT, and Loss Nodes
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

open Spec
open Tensor

namespace Tape

/-!
## Linear algebra
-/

/-- Matrix multiply node for tensors of shape `(m,n)` and `(n,p)`. -/
def matmul {m n p : Nat} (t : Tape) (aId bId : Nat) : Result (Tape × Nat) := do
  let m32 ← u32 m
  let n32 ← u32 n
  let p32 ← u32 p
  let one32 : UInt32 := 1
  let σ₁ : Shape := .dim m (.dim n .scalar)
  let σ₂ : Shape := .dim n (.dim p .scalar)
  let τ : Shape := .dim m (.dim p .scalar)
  binary (t := t) "matmul" aId bId σ₁ σ₂ τ
    (forward := fun a b => Buffer.bmm a b one32 m32 n32 p32)
    (backward := fun a b dLdy =>
      let dA := Buffer.bmmRightTranspose dLdy b one32 m32 p32 n32
      let dB := Buffer.bmmLeftTranspose a dLdy one32 n32 m32 p32
      (dA, dB))

/-- Batched matrix multiply for `(batch,m,n) × (batch,n,p)` CUDA buffers. -/
def bmm {batch m n p : Nat} (t : Tape) (aId bId : Nat) : Result (Tape × Nat) := do
  let b32 ← u32 batch
  let m32 ← u32 m
  let n32 ← u32 n
  let p32 ← u32 p
  let σ₁ : Shape := .dim batch (.dim m (.dim n .scalar))
  let σ₂ : Shape := .dim batch (.dim n (.dim p .scalar))
  let τ : Shape := .dim batch (.dim m (.dim p .scalar))
  binary (t := t) "bmm" aId bId σ₁ σ₂ τ
    (forward := fun a b => Buffer.bmm a b b32 m32 n32 p32)
    (backward := fun a b dLdy =>
      let dA := Buffer.bmmRightTranspose dLdy b b32 m32 p32 n32
      let dB := Buffer.bmmLeftTranspose a dLdy b32 n32 m32 p32
      (dA, dB))

/--
Fused real-FFT spectral convolution used by the CUDA FNO1D path.

Shapes:
- `x : (grid, width)`,
- `wRe, wIm : (modes, width, width)`,
- output `y : (grid, width)`.

The low-level buffer primitive owns the numerical contract and VJP:
`rfft(x)` is unnormalized, the inverse is normalized, and the backward kernels include the
half-spectrum adjoint factors for real FFTs. This tape node records those three parent
dependencies and checks the runtime shapes before calling the native kernels.
-/
def spectralConv1dRfft {grid width modes : Nat}
    (t : Tape) (xId wReId wImId : Nat) : Result (Tape × Nat) := do
  if grid = 0 then
    throw "autograd: spectralConv1dRfft: grid must be positive"
  if width = 0 then
    throw "autograd: spectralConv1dRfft: width must be positive"
  if modes > grid / 2 + 1 then
    throw "autograd: spectralConv1dRfft: modes exceeds rfft frequency count"
  let grid32 ← u32 grid
  let width32 ← u32 width
  let modes32 ← u32 modes
  let xShape : Shape := .dim grid (.dim width .scalar)
  let wShape : Shape := .dim modes (.dim width (.dim width .scalar))
  let x ← requireValue (t := t) xId xShape
  let wRe ← requireValue (t := t) wReId wShape
  let wIm ← requireValue (t := t) wImId wShape
  let y := Buffer.spectralConv1dRfftFwd x wRe wIm grid32 width32 modes32
  let node : Node :=
    { name := some "spectralConv1dRfft"
      value := { s := xShape, buf := y }
      requires_grad := true
      parents := [xId, wReId, wImId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny xShape
        let dx := Buffer.spectralConv1dRfftBwdX x wRe wIm dLdy.buf grid32 width32 modes32
        let dWRe := Buffer.spectralConv1dRfftBwdWRe x wRe wIm dLdy.buf grid32 width32 modes32
        let dWIm := Buffer.spectralConv1dRfftBwdWIm x wRe wIm dLdy.buf grid32 width32 modes32
        pure
          [ (xId, { s := xShape, buf := dx })
          , (wReId, { s := wShape, buf := dWRe })
          , (wImId, { s := wShape, buf := dWIm }) ] }
  pure (t.addNode node)

/-!
## Linear layer / losses
-/

/-- Linear layer: `y = W·x + b` with `W : (outDim,inDim)`, `x : inDim`, `b : outDim`. -/
def linear {outDim inDim : Nat} (t : Tape) (wId bId xId : Nat) : Result (Tape × Nat) := do
  let out32 ← u32 outDim
  let in32 ← u32 inDim
  let one32 : UInt32 := 1
  let wBuf ← requireValue (t := t) wId (.dim outDim (.dim inDim .scalar))
  let bBuf ← requireValue (t := t) bId (.dim outDim .scalar)
  let xBuf ← requireValue (t := t) xId (.dim inDim .scalar)
  let wx := Buffer.bmm wBuf xBuf one32 out32 in32 one32
  let yBuf := Buffer.add wx bBuf
  let node : Node :=
    { name := some "linear"
      value := { s := .dim outDim .scalar, buf := yBuf }
      requires_grad := true
      parents := [wId, bId, xId]
      cleanup := [wx]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny (.dim outDim .scalar)
        let g := dLdy.buf
        let dW := Buffer.bmm g xBuf one32 out32 one32 in32
        let db := Buffer.copy g
        let dx := Buffer.bmmLeftTranspose wBuf g one32 in32 out32 one32
        pure
          [ (wId, { s := .dim outDim (.dim inDim .scalar), buf := dW })
          , (bId, { s := .dim outDim .scalar, buf := db })
          , (xId, { s := .dim inDim .scalar, buf := dx }) ] }
  pure (t.addNode node)

/-- Mean-squared-error loss with `"mean"` reduction (single scalar output). -/
def mseLoss {s : Shape} (t : Tape) (yhatId targetId : Nat) : Result (Tape × Nat) := do
  let yhat ← requireValue (t := t) yhatId s
  let target ← requireValue (t := t) targetId s
  let diff := Buffer.sub yhat target
  let squared := Buffer.mul diff diff
  let sum := Buffer.reduceSum squared
  let denom : Float := Float.ofNat (Spec.Shape.size s)
  let mean := Buffer.scale sum (1.0 / denom)
  let node : Node :=
    { name := some "mse_loss"
      value := { s := Shape.scalar, buf := mean }
      requires_grad := true
      parents := [yhatId, targetId]
      cleanup := [diff, squared, sum]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny Shape.scalar
        let gBroad := broadcastScalarToShape dLdy.buf s
        let scaleConst : Float := 2.0 / denom
        let diffGrad := Buffer.mul diff gBroad
        let dYhat := Buffer.scale diffGrad scaleConst
        let dTarget := Buffer.releaseThen gBroad <|
          Buffer.releaseThen diffGrad <|
            Buffer.scale dYhat (-1.0)
        pure [
          (yhatId, { s := s, buf := dYhat }),
          (targetId, { s := s, buf := dTarget })
        ] }
  pure (t.addNode node)
end Tape

end Cuda
end Autograd
end Runtime
