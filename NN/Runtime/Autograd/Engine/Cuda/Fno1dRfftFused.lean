/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Tape
public import NN.Runtime.Autograd.Engine.Cuda.Convert

public import NN.Runtime.Autograd.Engine.Cuda.Ops

/-!
# CUDA FNO1D (real RFFT fused path)

This file provides a CUDA-only forward + VJP wrapper for a small real-valued FNO1D model whose
spectral convolution is implemented by the fused cuFFT-backed primitive `Tape.spectralConv1dRfft`.

Why this is not a `TorchLean.NN.LayerDef`:
- `LayerDef` is backend-polymorphic and runs through the `Torch.Ops` interface.
- The fused `spectralConv1dRfft` op is implemented only for the CUDA tape backend.

This module is meant to be called by runnable examples that want the performance path, while the
portable reference path lives in `NN.Runtime.Autograd.TorchLean.Fno1d`.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

open Spec
open Tensor

namespace Fno1dRfftFused

/-- Runtime vector shape abbreviation used by the small fused FNO wrapper. -/
abbrev vec (n : Nat) : Shape := .dim n .scalar

/-- Runtime matrix shape abbreviation used by the small fused FNO wrapper. -/
abbrev mat (m n : Nat) : Shape := .dim m (.dim n .scalar)

/--
Trainable parameter plus Adam moment buffers.

All three arrays use the same row-major layout for `shape`. The value array is uploaded to CUDA
when building a tape; the moment arrays stay on the host because this small wrapper performs Adam
updates in Lean after downloading gradients.
-/
structure Param where
  /-- Runtime tensor shape for `value`, `m`, and `v`. -/
  shape : Shape
  /-- Current parameter values in row-major order. -/
  value : FloatArray
  /-- Adam first-moment accumulator. -/
  m : FloatArray
  /-- Adam second-moment accumulator. -/
  v : FloatArray

/-- Output of one fused-FNO tape construction. -/
structure Forward where
  /-- The completed CUDA tape. -/
  tape : Tape
  /-- Node id of the prediction tensor. -/
  predId : Nat
  /-- Optional scalar loss node id, present only when a target was supplied. -/
  lossId? : Option Nat
  /-- Tape node ids for parameters, in the same order as the parameter array. -/
  paramIds : Array Nat

namespace Forward

/-- Number of CUDA buffer handles owned by a completed forward tape. -/
def ownedBufferCount (fw : Forward) : Nat :=
  fw.tape.nodes.foldl (fun n node => n + 1 + node.cleanup.length) 0

/--
Release every forward value and saved workspace owned by a completed tape.

The fused FNO wrapper rebuilds its tape for every sample. Explicit disposal is therefore part of
the wrapper's ownership contract; waiting for external-object finalizers makes long training and
evaluation runs retain one full tape per sample until a later runtime collection.
-/
def dispose (fw : Forward) : IO Unit := do
  let mut released := 0
  for node in fw.tape.nodes do
    released := released + (← Buffer.releaseIO node.value.buf).toNat
    for workspace in node.cleanup do
      released := released + (← Buffer.releaseIO workspace).toNat
  if released > fw.ownedBufferCount then
    throw <| IO.userError "autograd: fused-fno: invalid forward buffer release count"

end Forward

/-- Minimal Adam state carried across fused-FNO training steps. -/
structure AdamState where
  /-- Step counter (1-based in the Adam bias correction formulas). -/
  step : Nat := 0
  /-- Cached `beta1^step` for bias correction (starts at 1). -/
  beta1Pow : Float := 1.0
  /-- Cached `beta2^step` for bias correction (starts at 1). -/
  beta2Pow : Float := 1.0

/-- Allocate a zero-filled `FloatArray` of length `n`. -/
def zerosArray (n : Nat) : FloatArray :=
  FloatArray.mk (Array.mk (List.replicate n 0.0))

/--
One step of the small deterministic LCG used for fused-FNO parameter initialization.

This stays local to the fused CUDA example path so the engine layer does not depend on the
higher-level `Torch.Init` helper namespace.
-/
def lcgNext (s : Nat) : Nat :=
  (1664525 * s + 1013904223) % 4294967296

/-- Deterministic pseudo-random number in `[0, 1)` derived from `seed` and a scalar index. -/
def rand01 (seed idx : Nat) : Float :=
  let rec go : Nat → Nat → Nat
    | 0, s => lcgNext s
    | Nat.succ n, s => go n (lcgNext s)
  (Float.ofNat (go idx seed)) / 4294967296.0

/-- Deterministic uniform sample in `[lo, hi)` for a scalar index. -/
def uniformAt (seed idx : Nat) (lo hi : Float) : Float :=
  lo + rand01 seed idx * (hi - lo)

/-- Initialize a row-major parameter array with deterministic uniform samples. -/
def initFloatArray (shape : Shape) (seed : Nat) (lo hi : Float) : FloatArray :=
  FloatArray.mk <| Array.ofFn (n := Spec.Shape.size shape) (fun i =>
    uniformAt seed i.val lo hi)

/-- Initialize a trainable parameter and zero Adam moments. -/
def initParam (shape : Shape) (seed : Nat) (lo hi : Float) : Param :=
  { shape := shape
    value := initFloatArray shape seed lo hi
    m := zerosArray (Spec.Shape.size shape)
    v := zerosArray (Spec.Shape.size shape) }

/-- Initialize a bias-like parameter at zero with zero Adam moments. -/
def initBias (shape : Shape) : Param :=
  { shape := shape
    value := zerosArray (Spec.Shape.size shape)
    m := zerosArray (Spec.Shape.size shape)
    v := zerosArray (Spec.Shape.size shape) }

/--
Initialize parameters for the fused FNO1D model:

- input lift: `W_in : (1,width)`, `b_in : (width)`
- blocks: `(wRe,wIm) : (modes,width,width)`, `wSkip : (width,width)`, `bSkip : (width)`
- output proj: `W_out : (width,1)`, `b_out : (1)`
-/
def initParams (grid width modes blocks : Nat) (seed : Nat) : Array Param := Id.run do
  let _ := grid
  let spectralShape : Shape := .dim modes (.dim width (.dim width .scalar))
  let wSkipShape : Shape := mat width width
  let bSkipShape : Shape := vec width
  let mut ps : Array Param := #[]
  ps := ps.push (initParam (mat 1 width) (seed + 1) (-0.04) 0.04)
  ps := ps.push (initBias (vec width))
  for b in [0:blocks] do
    let base := seed + 100 + 31 * b
    ps := ps.push (initParam spectralShape (base + 0) (-0.04) 0.04)
    ps := ps.push (initParam spectralShape (base + 1) (-0.04) 0.04)
    ps := ps.push (initParam wSkipShape (base + 2) (-0.04) 0.04)
    ps := ps.push (initBias bSkipShape)
  ps := ps.push (initParam (mat width 1) (seed + 1000) (-0.04) 0.04)
  ps := ps.push (initBias (vec 1))
  pure ps

/-- Fetch a parameter with an error message that points to the fused-FNO wrapper. -/
def getParam (ps : Array Param) (i : Nat) : Result Param :=
  match ps[i]? with
  | some p => pure p
  | none => throw s!"autograd: fused-fno: parameter index out of bounds: {i}"

/--
Validate uploaded parameter buffer `i` against its declared shape, add it as a
gradient-requiring CUDA tape leaf, and record the new node id.
-/
def addParamLeaf (t : Tape) (ps : Array Param) (paramBuffers : Array Buffer)
    (paramIds : Array Nat) (i : Nat) :
    Result (Tape × Array Nat × Nat) := do
  let p ← getParam ps i
  let buf ← match paramBuffers[i]? with
    | some buf => pure buf
    | none => throw s!"autograd: fused-fno: missing uploaded parameter buffer: {i}"
  let checked ← match AnyBuffer.validate { s := p.shape, buf := buf } with
    | .ok checked => pure checked
    | .error msg => throw s!"autograd: fused-fno: invalid uploaded parameter {i}: {msg}"
  let (t', id) := t.leaf
    checked
    (some s!"param{i}")
  pure (t', paramIds.push id, id)

/-- Broadcast a vector of length `cols` across `grid` rows. -/
def broadcastVecToMat (t : Tape) (grid cols : Nat) (xId : Nat) : Result (Tape × Nat) :=
  do
    let _ ← t.requireValue xId (vec cols)
    Tape.broadcastTo (t := t) (s₁ := vec cols) (s₂ := mat grid cols) Shape.BroadcastTo.proof xId

/--
Build a CUDA tape that computes prediction (and optionally MSE loss) for the fused real-RFFT FNO.

Inputs:
- `x : (grid)` (interpreted as `(grid,1)`),
- optional `target : (grid)`.

Every external CUDA buffer is checked against the corresponding logical shape and element count
before an operation can consume it. Input and parameter buffers are checked before their leaf nodes
are added; the optional target is checked by the MSE operation on the loss path.
-/
def forwardWithBuffers (grid width modes blocks : Nat)
    (ps : Array Param)
    (target? : Option (Tensor Float (vec grid)))
    (xBuffer : Buffer) (paramBuffers : Array Buffer) (targetBuffer? : Option Buffer) :
    Result Forward := do
  let xMatShape : Shape := mat grid 1
  let yMatShape : Shape := mat grid 1
  let hiddenShape : Shape := mat grid width
  let paramIds0 : Array Nat := #[]
  let checkedInput ← match AnyBuffer.validate { s := xMatShape, buf := xBuffer } with
    | .ok checked => pure checked
    | .error msg => throw s!"autograd: fused-fno: invalid uploaded input: {msg}"
  let (t0, xId) := Tape.empty.leaf
    checkedInput
    (some "x") false

  let (t1, paramIds1, wInId) ← addParamLeaf t0 ps paramBuffers paramIds0 0
  let (t2, paramIds2, bInId) ← addParamLeaf t1 ps paramBuffers paramIds1 1
  let mut paramIds := paramIds2
  let (t3, h0Id) ← Tape.matmul (t := t2) (m := grid) (n := 1) (p := width) xId wInId
  let (t4, bInBId) ← broadcastVecToMat (grid := grid) (cols := width) t3 bInId
  let (t5, hId0) ← Tape.add (t := t4) (s := hiddenShape) h0Id bInBId

  let mut t := t5
  let mut hId := hId0
  for b in [0:blocks] do
    let base := 2 + 4 * b
    let (tA, idsA, wReId) ← addParamLeaf t ps paramBuffers paramIds base
    t := tA; paramIds := idsA
    let (tB, idsB, wImId) ← addParamLeaf t ps paramBuffers paramIds (base + 1)
    t := tB; paramIds := idsB
    let (tC, idsC, wSkipId) ← addParamLeaf t ps paramBuffers paramIds (base + 2)
    t := tC; paramIds := idsC
    let (tD, idsD, bSkipId) ← addParamLeaf t ps paramBuffers paramIds (base + 3)
    t := tD; paramIds := idsD
    let (tSpec, ySpecId) ← Tape.spectralConv1dRfft (t := t) (grid := grid) (width := width) (modes := modes)
      hId wReId wImId
    let (tSkip0, ySkip0Id) ← Tape.matmul (t := tSpec) (m := grid) (n := width) (p := width) hId wSkipId
    let (tBias, bSkipBId) ← broadcastVecToMat (grid := grid) (cols := width) tSkip0 bSkipId
    let (tSkip, ySkipId) ← Tape.add (t := tBias) (s := hiddenShape) ySkip0Id bSkipBId
    let (tSum, yId) ← Tape.add (t := tSkip) (s := hiddenShape) ySpecId ySkipId
    let (tRelu, yReluId) ← Tape.relu (t := tSum) (s := hiddenShape) yId
    t := tRelu
    hId := yReluId

  let outBase := 2 + 4 * blocks
  let (tOutW, idsOutW, wOutId) ← addParamLeaf t ps paramBuffers paramIds outBase
  t := tOutW; paramIds := idsOutW
  let (tOutB, idsOutB, bOutId) ← addParamLeaf t ps paramBuffers paramIds (outBase + 1)
  t := tOutB; paramIds := idsOutB
  let (tPred0, pred0Id) ← Tape.matmul (t := t) (m := grid) (n := width) (p := 1) hId wOutId
  let (tPredB, bOutBId) ← Tape.broadcastTo (t := tPred0) (s₁ := vec 1) (s₂ := yMatShape)
    Shape.BroadcastTo.proof bOutId
  let (tPred, predId) ← Tape.add (t := tPredB) (s := yMatShape) pred0Id bOutBId
  match target? with
  | none =>
      pure { tape := tPred, predId := predId, lossId? := none, paramIds := paramIds }
  | some _ =>
      let targetBuffer ← match targetBuffer? with
        | some buf => pure buf
        | none => throw "autograd: fused-fno: missing uploaded target buffer"
      let (tTarget, targetId) := tPred.leaf
        { s := yMatShape, buf := targetBuffer }
        (some "target") false
      let (tLoss, lossId) ← Tape.mseLoss (t := tTarget) (s := yMatShape) predId targetId
      pure { tape := tLoss, predId := predId, lossId? := some lossId, paramIds := paramIds }

/--
Build one fused-FNO tape from fresh CUDA uploads.

The uploads are effectful so repeated forwards over identical host arrays cannot share an external
buffer handle. This gives each returned `Forward` exclusive ownership of the buffers it disposes.
-/
def forward (grid width modes blocks : Nat)
    (ps : Array Param)
    (x : Tensor Float (vec grid))
    (target? : Option (Tensor Float (vec grid))) :
    IO (Result Forward) := do
  let xBuffer ← Buffer.ofFloatArrayIO (Convert.flattenFloat (s := vec grid) x)
  let mut paramBuffers := #[]
  for p in ps do
    paramBuffers := paramBuffers.push (← Buffer.ofFloatArrayIO p.value)
  let targetBuffer? ← match target? with
    | none => pure none
    | some y =>
        pure <| some (← Buffer.ofFloatArrayIO (Convert.flattenFloat (s := vec grid) y))
  let result := forwardWithBuffers grid width modes blocks ps target? xBuffer paramBuffers targetBuffer?
  match result with
  | .ok fw => pure <| .ok fw
  | .error msg =>
      discard <| Buffer.releaseIO xBuffer
      for buffer in paramBuffers do
        discard <| Buffer.releaseIO buffer
      match targetBuffer? with
      | some buffer => discard <| Buffer.releaseIO buffer
      | none => pure ()
      pure <| .error msg

/-- Download a scalar CUDA tape value to host `Float`. -/
def scalarFromTape (t : Tape) (id : Nat) : IO (Result Float) := do
  match Tape.requireValue (t := t) id Shape.scalar with
  | .error msg => pure <| .error msg
  | .ok b =>
      let a ← Buffer.toFloatArrayIO b
      pure <| .ok (a.get! 0)

/-- Download a `(grid,1)` prediction matrix as a length-`grid` tensor. -/
def predFromTape (grid : Nat) (t : Tape) (id : Nat) : IO (Result (Tensor Float (vec grid))) := do
  match Tape.requireValue (t := t) id (mat grid 1) with
  | .error msg => pure <| .error msg
  | .ok b =>
      let values ← Buffer.toFloatArrayIO b
      match Convert.unflattenFloat? (s := vec grid) values with
      | some y => pure <| .ok y
      | none => pure <| .error "autograd: fused-fno: prediction shape mismatch"

/-- Mean MSE loss over a host-side list of `(input,target)` samples. -/
def meanLoss (grid width modes blocks : Nat)
    (ps : Array Param) (samples : List (Tensor Float (vec grid) × Tensor Float (vec grid))) :
    IO (Result Float) := do
  if samples.isEmpty then
    pure <| .ok (0.0 / 0.0)
  else
    let mut acc := 0.0
    for (x, y) in samples do
      match ← forward (grid := grid) (width := width) (modes := modes) (blocks := blocks) ps x
          (some y) with
      | .error msg => return .error msg
      | .ok fw =>
          let result ← match fw.lossId? with
            | some lossId => scalarFromTape fw.tape lossId
            | none => pure <| .error "autograd: fused-fno: internal missing loss id"
          fw.dispose
          match result with
          | .error msg => return .error msg
          | .ok loss => acc := acc + loss
    pure <| .ok (acc / Float.ofNat samples.length)

/--
Host-side Adam update for one flattened parameter array.

Bias correction factors are passed in already computed as `1 - beta₁^t` and `1 - beta₂^t`.
-/
def adamUpdateArrayBiasCorrected
    (value m v grad : FloatArray)
    (lr beta1 beta2 eps : Float)
    (biasCorr1 biasCorr2 : Float) :
    FloatArray × FloatArray × FloatArray := Id.run do
  let mut value' : Array Float := #[]
  let mut m' : Array Float := #[]
  let mut v' : Array Float := #[]
  for i in [:value.size] do
    let g := grad.get! i
    let mi := beta1 * m.get! i + (1.0 - beta1) * g
    let vi := beta2 * v.get! i + (1.0 - beta2) * g * g
    let mHat := mi / biasCorr1
    let vHat := vi / biasCorr2
    value' := value'.push (value.get! i - lr * mHat / (Float.sqrt vHat + eps))
    m' := m'.push mi
    v' := v'.push vi
  pure (FloatArray.mk value', FloatArray.mk m', FloatArray.mk v')

/--
Run reverse-mode on the fused-FNO tape and update every recorded parameter with Adam.

Gradients are computed on CUDA buffers and downloaded to host arrays before the update. A
high-throughput optimizer kernel should live in a separate CUDA optimizer layer, not inside this
model helper.
-/
def prepareAdamBackward
    (fw : Forward) (st : AdamState) (beta1 beta2 : Float) :
    IO (Result (Tape.SparseGradMap × AdamState × Float × Float)) := do
  let lossId ← match fw.lossId? with
    | some id => pure id
    | none => return .error "autograd: fused-fno: internal missing loss id"
  let seed : AnyBuffer := { s := Shape.scalar, buf := ← Buffer.fullIO 1 1.0 }
  let grads ← try
      Tape.backwardSparse (t := fw.tape) lossId seed
        (fun id => fw.paramIds.contains id)
    catch e =>
      return .error s!"autograd: fused-fno: sparse backward failed: {e}"

  -- Advance bias correction state.
  let st' : AdamState :=
    { step := st.step + 1
      beta1Pow := st.beta1Pow * beta1
      beta2Pow := st.beta2Pow * beta2 }
  let biasCorr1 := 1.0 - st'.beta1Pow
  let biasCorr2 := 1.0 - st'.beta2Pow

  pure <| .ok (grads, st', biasCorr1, biasCorr2)

/-- Run one Adam update and deterministically release the consumed gradient and forward buffers. -/
def updateParamsAdam
    (ps : Array Param) (fw : Forward) (lr : Float) (st : AdamState)
    (beta1 : Float := 0.9) (beta2 : Float := 0.999) (eps : Float := 1e-8) :
    IO (Result (Array Param × AdamState)) := do
  match ← prepareAdamBackward fw st beta1 beta2 with
  | .error msg =>
      fw.dispose
      pure <| .error msg
  | .ok (grads, st', biasCorr1, biasCorr2) =>
      let mut out := ps
      for i in [:ps.size] do
        let p ← match getParam out i with
          | .ok p => pure p
          | .error msg => Tape.releaseSparseGrads grads; fw.dispose; return .error msg
        let nodeId ← match fw.paramIds[i]? with
          | some id => pure id
          | none =>
              Tape.releaseSparseGrads grads
              fw.dispose
              return .error "autograd: fused-fno: internal missing param id"
        let gAny ← match grads.get? nodeId with
          | some g => pure g
          | none =>
              Tape.releaseSparseGrads grads
              fw.dispose
              return .error "autograd: fused-fno: internal missing grad"
        if _h : gAny.s = p.shape then
          let grad ← Buffer.toFloatArrayIO gAny.buf
          let (value', m', v') := adamUpdateArrayBiasCorrected
            p.value p.m p.v grad lr beta1 beta2 eps biasCorr1 biasCorr2
          if hi : i < out.size then
            out := out.set i { p with value := value', m := m', v := v' } hi
          else
            Tape.releaseSparseGrads grads
            fw.dispose
            return .error "autograd: fused-fno: internal update index invalid"
        else
          Tape.releaseSparseGrads grads
          fw.dispose
          return .error "autograd: fused-fno: gradient shape mismatch"
      Tape.releaseSparseGrads grads
      fw.dispose
      pure <| .ok (out, st')

end Fno1dRfftFused

end Cuda
end Autograd
end Runtime
