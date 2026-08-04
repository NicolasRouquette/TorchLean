/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Ops.ConvPool

/-!
# CUDA Tape Operations: Normalization and Row Softmax
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

open Spec
open Tensor

namespace Tape

/-!
## Normalization
-/

/--
LayerNorm over the last dimension for `(seqLen, embedDim)` buffers.

The tape records one normalization operation and keeps TorchLean's usual VJP. A fused buffer
primitive evaluates the forward formula and that VJP without materializing each reduction,
broadcast, and pointwise intermediate as a separate device buffer.
-/
def layerNorm {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (t : Tape) (xId gammaId betaId : Nat) : Result (Tape × Nat) := do
  have _ := h_seq_pos
  have _ := h_embed_pos
  let rows32 ← u32 seqLen
  let cols32 ← u32 embedDim
  let x ← requireValue (t := t) xId (.dim seqLen (.dim embedDim .scalar))
  let gamma ← requireValue (t := t) gammaId (.dim embedDim .scalar)
  let beta ← requireValue (t := t) betaId (.dim embedDim .scalar)
  let invCols : Float := 1.0 / Float.ofNat embedDim
  let eps : Float := Numbers.epsilon
  let (y, xHat, invStd) :=
    Buffer.layerNormFwd x gamma beta rows32 cols32 invCols eps
  let outShape : Shape := .dim seqLen (.dim embedDim .scalar)
  let node : Node :=
    { name := some "layer_norm"
      value := { s := outShape, buf := y }
      requires_grad := true
      parents := [xId, gammaId, betaId]
      cleanup := [xHat, invStd]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny outShape
        let (dx, dGamma, dBeta) :=
          Buffer.layerNormBwd dLdy.buf xHat invStd gamma rows32 cols32
            (Float.ofNat embedDim) invCols
        pure [
          (xId, { s := outShape, buf := dx }),
          (gammaId, { s := .dim embedDim .scalar, buf := dGamma }),
          (betaId, { s := .dim embedDim .scalar, buf := dBeta })
        ] }
  pure (t.addNode node)

/--
BatchNorm for a single channel-first image `(C,H,W)` (no batch axis).

We normalize per-channel across the spatial dimension `H*W`, reusing the same math as layer-norm
by treating the buffer as a `(channels, height*width)` matrix.
-/
def batchnormChannelFirst
  {channels height width : Nat} (h_c : channels > 0) (h_h : height > 0) (h_w : width > 0)
  (t : Tape) (xId gammaId betaId : Nat) : Result (Tape × Nat) := do
  have _ := h_c
  have _ := h_h
  have _ := h_w
  let rows32 ← u32 channels
  let cols : Nat := height * width
  if cols = 0 then
    throw "autograd: batchnorm_channel_first: height*width = 0"
  let cols32 ← u32 cols
  let xShape : Shape := .dim channels (.dim height (.dim width .scalar))
  let x ← requireValue (t := t) xId xShape
  let gamma ← requireValue (t := t) gammaId (.dim channels .scalar)
  let beta ← requireValue (t := t) betaId (.dim channels .scalar)
  -- Treat as a zero-copy (channels, cols) view; the underlying layout already matches.
  let sum1 := Buffer.reduceSumByRow x rows32 cols32
  let invCols : Float := 1.0 / Float.ofNat cols
  let mean := Buffer.scale sum1 invCols
  let meanB := Buffer.broadcastVecToCols mean rows32 cols32
  let centered := Buffer.sub x meanB
  let centered2 := Buffer.mul centered centered
  let varSum := Buffer.reduceSumByRow centered2 rows32 cols32
  let var := Buffer.scale varSum invCols
  let eps : Float := Numbers.epsilon
  let epsVec := Buffer.full rows32 eps
  let varEps := Buffer.add var epsVec
  let std := Buffer.sqrt varEps
  let stdB := Buffer.broadcastVecToCols std rows32 cols32
  let xHat := Buffer.div centered stdB
  let gammaB := Buffer.broadcastVecToCols gamma rows32 cols32
  let betaB := Buffer.broadcastVecToCols beta rows32 cols32
  let xHatGamma := Buffer.mul xHat gammaB
  let y := Buffer.add xHatGamma betaB
  let node : Node :=
    { name := some "batchnorm_channel_first"
      value := { s := xShape, buf := y }
      requires_grad := true
      parents := [xId, gammaId, betaId]
      cleanup :=
        [ sum1, mean, meanB, centered, centered2, varSum, var, epsVec, varEps
        , std, stdB, xHat, gammaB, betaB, xHatGamma ]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny xShape
        -- dBeta / dGamma sum over spatial dimension (axis=1 of the folded matrix).
        let dBeta := Buffer.reduceSumByRow dLdy.buf rows32 cols32
        let dGammaPointwise := Buffer.mul dLdy.buf xHat
        let dGamma := Buffer.releaseThen dGammaPointwise <|
          Buffer.reduceSumByRow dGammaPointwise rows32 cols32
        -- dX
        let dXhat := Buffer.mul dLdy.buf gammaB
        let sumDXhat := Buffer.reduceSumByRow dXhat rows32 cols32
        let dXhatXhat := Buffer.mul dXhat xHat
        let sumDXhatXhat := Buffer.releaseThen dXhatXhat <|
          Buffer.reduceSumByRow dXhatXhat rows32 cols32
        let sum1B := Buffer.broadcastVecToCols sumDXhat rows32 cols32
        let sum2B := Buffer.broadcastVecToCols sumDXhatXhat rows32 cols32
        let scaledDXhat := Buffer.scale dXhat (Float.ofNat cols)
        let centeredDXhat := Buffer.sub scaledDXhat sum1B
        let xHatSum2 := Buffer.mul xHat sum2B
        let term := Buffer.sub centeredDXhat xHatSum2
        let invStd := Buffer.inv std
        let invStdB := Buffer.broadcastVecToCols invStd rows32 cols32
        let termInv := Buffer.mul term invStdB
        let dxRaw := Buffer.scale termInv invCols
        let dx :=
          Buffer.releaseThen dXhat <| Buffer.releaseThen sumDXhat <|
            Buffer.releaseThen sumDXhatXhat <| Buffer.releaseThen sum1B <|
              Buffer.releaseThen sum2B <| Buffer.releaseThen scaledDXhat <|
                Buffer.releaseThen centeredDXhat <| Buffer.releaseThen xHatSum2 <|
                  Buffer.releaseThen term <| Buffer.releaseThen invStd <|
                    Buffer.releaseThen invStdB <| Buffer.releaseThen termInv dxRaw
        pure [
          (xId, { s := xShape, buf := dx }),
          (gammaId, { s := .dim channels .scalar, buf := dGamma }),
          (betaId, { s := .dim channels .scalar, buf := dBeta })
        ] }
  pure (t.addNode node)

/-!
## Softmax (last axis, row folding)

We implement softmax along the last axis by folding all leading dimensions into one `rows` axis.
This covers:
- 2D softmax (`(rows, cols)`),
- 3D batched softmax (`(batch, rows, cols)`) by folding `batch*rows` into `rows`.
-/

def softmax {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  match s with
  | .scalar =>
      let _x ← requireValue (t := t) xId Shape.scalar
      let one32 : UInt32 := 1
      let one := Buffer.full one32 1.0
      let node : Node :=
        { name := some "softmax"
          value := { s := Shape.scalar, buf := one }
          requires_grad := true
          parents := [xId]
          backward := fun dLdyAny => do
            let _ ← requireGrad dLdyAny Shape.scalar
            let dx := Buffer.zeros one32
            pure [(xId, { s := Shape.scalar, buf := dx })] }
      pure (t.addNode node)
  | _ =>
      let (rows32, cols32) ← foldRowsColsLastAxis s
      let x ← requireValue (t := t) xId s
      let yOwned := rowSoftmaxForward x rows32 cols32
      let node : Node :=
        { name := some "softmax"
          value := { s := s, buf := yOwned.value }
          requires_grad := true
          parents := [xId]
          cleanup := yOwned.workspace
          backward := fun dLdyAny => do
            let dLdy ← requireGrad dLdyAny s
            let dx := rowSoftmaxBwd yOwned.value dLdy.buf rows32 cols32
            pure [(xId, { s := s, buf := dx })] }
      pure (t.addNode node)

/-- Stable log-softmax along the last axis, implemented directly on CUDA buffers. -/
def logSoftmax {s : Shape} (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  match s with
  | .scalar =>
      let _x ← requireValue (t := t) xId Shape.scalar
      let one32 : UInt32 := 1
      let zero := Buffer.zeros one32
      let node : Node :=
        { name := some "log_softmax"
          value := { s := Shape.scalar, buf := zero }
          requires_grad := true
          parents := [xId]
          backward := fun dLdyAny => do
            let _ ← requireGrad dLdyAny Shape.scalar
            let dx := Buffer.zeros one32
            pure [(xId, { s := Shape.scalar, buf := dx })] }
      pure (t.addNode node)
  | _ =>
      let (rows32, cols32) ← foldRowsColsLastAxis s
      let x ← requireValue (t := t) xId s
      let yOwned := rowLogSoftmaxForward x rows32 cols32
      let node : Node :=
        { name := some "log_softmax"
          value := { s := s, buf := yOwned.value }
          requires_grad := true
          parents := [xId]
          cleanup := yOwned.workspace
          backward := fun dLdyAny => do
            let dLdy ← requireGrad dLdyAny s
            let dx := rowLogSoftmaxBwd yOwned.value dLdy.buf rows32 cols32
            pure [(xId, { s := s, buf := dx })] }
      pure (t.addNode node)
end Tape

end Cuda
end Autograd
end Runtime
