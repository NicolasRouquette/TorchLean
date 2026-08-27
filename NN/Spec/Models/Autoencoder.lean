/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Activation
public import NN.Spec.Core.Sequence

/-!
# Autoencoder (spec model)

This file defines a small **fully-connected autoencoder**:

- encoder: `h = act(W_enc x + b_enc)`
- decoder: `x̂ = W_dec h + b_dec`

PyTorch analogue: `nn.Sequential(nn.Linear(inputDim, hiddenDim), act, nn.Linear(hiddenDim,
  inputDim))`
applied to a single vector (no batch dimension).

This is spec-level/reference code. It is written for auditability and differentiation, and it is
intended to be instantiated over multiple scalar backends (`Float`, intervals, proof-level reals,
...).

The activation is represented by `Activation.Kind`, so a misspelled configuration cannot silently
change the model into an identity activation.
-/

@[expose] public section


namespace Spec

open Tensor

variable {α : Type} [Context α]

/-!
## Parameters

We store the encoder and decoder weights explicitly.

Shapes:

- `encoderWeight : (hiddenDim × inputDim)`
- `decoderWeight : (inputDim × hiddenDim)`
- `encoderBias   : (hiddenDim)`
- `decoderBias   : (inputDim)`
-/
/-- Parameters for a 1-hidden-layer fully-connected autoencoder. -/
structure AutoencoderSpec (α : Type) (inputDim hiddenDim : Nat) where
  /-- Encoder weights with shape `(hiddenDim × inputDim)`. -/
  encoderWeight : Tensor α [hiddenDim, inputDim]
  /-- Encoder bias with shape `(hiddenDim)`. -/
  encoderBias : Tensor α [hiddenDim]
  /-- Decoder weights with shape `(inputDim × hiddenDim)`. -/
  decoderWeight : Tensor α [inputDim, hiddenDim]
  /-- Decoder bias with shape `(inputDim)`. -/
  decoderBias : Tensor α [inputDim]
  /-- Pointwise activation between the encoder and decoder. -/
  activation : Activation.Kind := .relu

/-!
## Forward
-/

/-- Encode a vector into a hidden representation:

`h = act(W_enc x + b_enc)`.

PyTorch analogy: `act(linear(x))` for a single `nn.linear`.
-/
def autoencoderEncodeSpec {inputDim hiddenDim : Nat}
  (m : AutoencoderSpec α inputDim hiddenDim)
  (input : Tensor α [inputDim]) :
  Tensor α [hiddenDim] :=
  let linearOut := addSpec (matVecMulSpec m.encoderWeight input) m.encoderBias
  m.activation.applySpec linearOut

/-- Decode a hidden representation back to input space:

`x̂ = W_dec h + b_dec`.

PyTorch analogy: a second `nn.Linear(hiddenDim, inputDim)` without an activation.
-/
def autoencoderDecodeSpec {inputDim hiddenDim : Nat}
  (m : AutoencoderSpec α inputDim hiddenDim)
  (hidden : Tensor α [hiddenDim]) :
  Tensor α [inputDim] :=
  addSpec (matVecMulSpec m.decoderWeight hidden) m.decoderBias

/-- Full autoencoder forward pass: `decode(encode(x))`. -/
def autoencoderForwardSpec {inputDim hiddenDim : Nat}
  (m : AutoencoderSpec α inputDim hiddenDim)
  (input : Tensor α [inputDim]) :
  Tensor α [inputDim] :=
  let hidden := autoencoderEncodeSpec m input
  autoencoderDecodeSpec m hidden

/-- Apply an autoencoder independently at every index of a leading shape. -/
def autoencoderForwardLeadingSpec (leading : Shape) {inputDim hiddenDim : Nat}
  (m : AutoencoderSpec α inputDim hiddenDim)
  (input : Tensor α (leading.concat [inputDim])) :
  Tensor α (leading.concat [inputDim]) :=
  Tensor.mapEach leading (autoencoderForwardSpec m) input

/-!
## Backward (manual VJP)

This file includes a small, explicit backward pass for the autoencoder.

PyTorch analogy: this is what autograd computes, but spelled out as pure functions.
The key linear-algebra identities used are:

- If `y = W x + b`, then `dW = dY ⊗ x`, `db = dY`, and `dX = Wᵀ dY`.
- If `h = act(z)`, then `dZ = dH ⊙ act'(z)`.
-/

/-- Gradient w.r.t. encoder weights: `dW_enc = dZ ⊗ x`. -/
def autoencoderEncoderWeightsDerivSpec {inputDim hiddenDim : Nat}
  (m : AutoencoderSpec α inputDim hiddenDim)
  (input : Tensor α [inputDim])
  (grad_output : Tensor α [inputDim]) :
  Tensor α [hiddenDim, inputDim] :=
  -- `dH = W_decᵀ dOut`.
  let grad_hidden :=
    matVecMulSpec (swapAdjacentAxes m.decoderWeight 0) grad_output
  let linearOut := addSpec (matVecMulSpec m.encoderWeight input) m.encoderBias
  let gradLinear := mulSpec grad_hidden (m.activation.derivSpec linearOut)
  outerProductSpec gradLinear input

/-- Gradient w.r.t. encoder bias: `db_enc = dZ`. -/
def autoencoderEncoderBiasDerivSpec {inputDim hiddenDim : Nat}
  (m : AutoencoderSpec α inputDim hiddenDim)
  (input : Tensor α [inputDim])
  (grad_output : Tensor α [inputDim]) :
  Tensor α [hiddenDim] :=
  let grad_hidden :=
    matVecMulSpec (swapAdjacentAxes m.decoderWeight 0) grad_output
  let linearOut := addSpec (matVecMulSpec m.encoderWeight input) m.encoderBias
  mulSpec grad_hidden (m.activation.derivSpec linearOut)

/-- Gradient w.r.t. decoder weights: `dW_dec = dOut ⊗ h`. -/
def autoencoderDecoderWeightsDerivSpec {inputDim hiddenDim : Nat}
  (m : AutoencoderSpec α inputDim hiddenDim)
  (input : Tensor α [inputDim])
  (grad_output : Tensor α [inputDim]) :
  Tensor α [inputDim, hiddenDim] :=
  let hidden := autoencoderEncodeSpec m input
  outerProductSpec grad_output hidden

/-- Gradient w.r.t. decoder bias: `db_dec = dOut`. -/
def autoencoderDecoderBiasDerivSpec {inputDim hiddenDim : Nat}
  (_m : AutoencoderSpec α inputDim hiddenDim)
  (grad_output : Tensor α [inputDim]) :
  Tensor α [inputDim] :=
  grad_output

/-- Gradient w.r.t. input: `dX = W_encᵀ dZ`. -/
def autoencoderInputDerivSpec {inputDim hiddenDim : Nat}
  (m : AutoencoderSpec α inputDim hiddenDim)
  (input : Tensor α [inputDim])
  (grad_output : Tensor α [inputDim]) :
  Tensor α [inputDim] :=
  let grad_hidden :=
    matVecMulSpec (swapAdjacentAxes m.decoderWeight 0) grad_output
  let linearOut := addSpec (matVecMulSpec m.encoderWeight input) m.encoderBias
  let gradLinear := mulSpec grad_hidden (m.activation.derivSpec linearOut)
  matVecMulSpec (swapAdjacentAxes m.encoderWeight 0) gradLinear

/-- Complete backward pass: returns

`(dW_enc, db_enc, dW_dec, db_dec, dX)`.
-/
def autoencoderBackwardSpec {inputDim hiddenDim : Nat}
  (m : AutoencoderSpec α inputDim hiddenDim)
  (input : Tensor α [inputDim])
  (grad_output : Tensor α [inputDim]) :
  (Tensor α [hiddenDim, inputDim] ×
   Tensor α [hiddenDim] ×
   Tensor α [inputDim, hiddenDim] ×
   Tensor α [inputDim] ×
   Tensor α [inputDim]) :=
  let encoderWeight := autoencoderEncoderWeightsDerivSpec m input grad_output
  let encoderBias := autoencoderEncoderBiasDerivSpec m input grad_output
  let decoderWeight := autoencoderDecoderWeightsDerivSpec m input grad_output
  let decoderBias := autoencoderDecoderBiasDerivSpec m grad_output
  let inputGradient := autoencoderInputDerivSpec m input grad_output
  (encoderWeight, encoderBias, decoderWeight, decoderBias, inputGradient)

/-- Mean-squared reconstruction error (single example).

PyTorch analogy: `F.mse_loss(x_hat, x, reduction="mean")`.
-/
def autoencoderReconstructionErrorSpec {inputDim hiddenDim : Nat}
  (m : AutoencoderSpec α inputDim hiddenDim)
  (input : Tensor α [inputDim]) (h : inputDim ≠ 0) :
  α :=
  let reconstructed := autoencoderForwardSpec m input
  let error := subSpec input reconstructed
  let squared_error := squareSpec error
  have inst : Shape.HasNonemptyAxis 0 (Shape.dim inputDim Shape.scalar) := by
    apply Shape.hasNonemptyAxisZeroOfNe h
  item (reduceSum 0 squared_error inst.proof) / inputDim

/-- A compact helper used by examples: compression ratio as a `Float`.

Note: if `hiddenDim = 0`, this produces `∞`/`NaN` depending on the `Float` backend.
The rest of the spec never needs this number; it is purely for display.
-/
def autoencoderCompressionRatioSpec {inputDim hiddenDim : Nat} :
  Float :=
  inputDim.toFloat / hiddenDim.toFloat

end Spec
