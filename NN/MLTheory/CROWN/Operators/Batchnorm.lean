/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Core
public import NN.Spec.Core.Context
public import NN.Spec.Core.Tensor
public import NN.Spec.Core.TensorOps

/-!
# BatchNorm operator bounds (IBP + affine)

This file bounds inference-time BatchNorm. Since inference-time BatchNorm is an affine
transformation (with frozen statistics), both IBP and affine propagation are exact (componentwise).

At inference time, TorchLean uses
`y = γ * (x - μ) / sqrt(max(σ², 0) + ε) + β`,
so the layer reduces to `y = scale * x + offset`, where
`scale = γ / sqrt(max(σ², 0) + ε)` and
`offset = β - γ * μ / sqrt(max(σ², 0) + ε)`.

The `max` is the same totalization used by `Spec.batchNormInference` and the IR evaluator. It has no
effect on valid nonnegative running variances, while keeping every TorchLean layer aligned on
malformed approximate-runtime inputs.

References:
- Ioffe and Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing
  Internal Covariate Shift", ICML 2015.
- PyTorch analogue: `torch.nn.BatchNorm1d/2d/3d` in evaluation mode.
-/

@[expose] public section


namespace NN.MLTheory.CROWN.Operators.Batchnorm

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN

variable {α : Type} [Context α]

/-- Parameters for BatchNorm layer (frozen at inference). -/
structure BatchNormParams (α : Type) [Context α] where
  /-- Number of channels/features -/
  dim : Nat
  /-- Running mean μ -/
  running_mean : Tensor α (.dim dim .scalar)
  /-- Running variance σ² -/
  running_var : Tensor α (.dim dim .scalar)
  /-- Learnable scale γ -/
  gamma : Tensor α (.dim dim .scalar)
  /-- Learnable bias β -/
  beta : Tensor α (.dim dim .scalar)
  /-- Small constant for numerical stability -/
  eps : α

/-- Compute the equivalent affine scale: `γ / sqrt(max(σ², 0) + ε)`. -/
def computeScale (params : BatchNormParams α) : Tensor α (.dim params.dim .scalar) :=
  match params.running_var, params.gamma with
  | .dim var, .dim gam =>
    Tensor.dim (fun i =>
      match var i, gam i with
      | .scalar v, .scalar g =>
        let denom := MathFunctions.sqrt (max v Numbers.zero + params.eps)
        Tensor.scalar (g / denom))

/-- Compute the equivalent affine offset: `β - γ * μ / sqrt(max(σ², 0) + ε)`. -/
def computeOffset (params : BatchNormParams α) : Tensor α (.dim params.dim .scalar) :=
  match params.running_mean, params.running_var, params.gamma, params.beta with
  | .dim mu, .dim var, .dim gam, .dim bet =>
    Tensor.dim (fun i =>
      match mu i, var i, gam i, bet i with
      | .scalar m, .scalar v, .scalar g, .scalar b =>
        let denom := MathFunctions.sqrt (max v Numbers.zero + params.eps)
        Tensor.scalar (b - g * m / denom))

/-- IBP for BatchNorm: since BN is affine, we can compute exact bounds.
    y = scale * x + offset, so:
    - If scale > 0: y_lo = scale * x_lo + offset, y_hi = scale * x_hi + offset
    - If scale < 0: y_lo = scale * x_hi + offset, y_hi = scale * x_lo + offset
-/
def ibpBatchnorm (params : BatchNormParams α)
    (xB : Box α (.dim params.dim .scalar)) : Box α (.dim params.dim .scalar) :=
  let scale := computeScale params
  let offset := computeOffset params
  match xB.lo, xB.hi, scale, offset with
  | .dim xlo, .dim xhi, .dim sc, .dim off =>
    let outLo := Tensor.dim (fun i =>
      match xlo i, xhi i, sc i, off i with
      | .scalar xl, .scalar xh, .scalar s, .scalar o =>
        -- If scale >= 0, use xl for lo; else use xh
        let lo := if s > Numbers.zero then s * xl + o else s * xh + o
        Tensor.scalar lo)
    let outHi := Tensor.dim (fun i =>
      match xlo i, xhi i, sc i, off i with
      | .scalar xl, .scalar xh, .scalar s, .scalar o =>
        -- If scale >= 0, use xh for hi; else use xl
        let hi := if s > Numbers.zero then s * xh + o else s * xl + o
        Tensor.scalar hi)
    { lo := outLo, hi := outHi }

/-- Affine bounds for BatchNorm propagation.
    Since BN is affine, compose the affine forms:
    If prev = A_prev * x_in + c_prev and BN = scale * · + offset
    Then composed = scale * (A_prev * x_in + c_prev) + offset
                  = diag(scale) * A_prev * x_in + (scale * c_prev + offset)
-/
def affBatchnorm {inDim : Nat} (params : BatchNormParams α)
    (aff : AffineVec α inDim params.dim) : AffineVec α inDim params.dim :=
  let scale := computeScale params
  let offset := computeOffset params
  match aff.A, aff.c, scale, offset with
  | .dim rows, .dim cv, .dim sc, .dim off =>
    -- Scale each row of A by corresponding scale[i]
    let A' := Tensor.dim (fun i =>
      match rows i, sc i with
      | .dim cols, .scalar si =>
        Tensor.dim (fun j =>
          match cols j with
          | .scalar aij => Tensor.scalar (si * aij)))
    -- Scale bias and add offset
    let c' := Tensor.dim (fun i =>
      match cv i, sc i, off i with
      | .scalar ci, .scalar si, .scalar oi =>
        Tensor.scalar (si * ci + oi))
    { A := A', c := c' }

/-- Derivative bounds for BatchNorm: since BN is affine, d(BN)/dx = scale (constant).
    Given input derivative bounds [dlo, dhi], output = scale * [dlo, dhi]. -/
def derivBatchnorm (params : BatchNormParams α)
    (dB : Box α (.dim params.dim .scalar)) : Box α (.dim params.dim .scalar) :=
  let scale := computeScale params
  match dB.lo, dB.hi, scale with
  | .dim dlo, .dim dhi, .dim sc =>
    let outLo := Tensor.dim (fun i =>
      match dlo i, dhi i, sc i with
      | .scalar dl, .scalar dh, .scalar s =>
        Tensor.scalar (if s > Numbers.zero then s * dl else s * dh))
    let outHi := Tensor.dim (fun i =>
      match dlo i, dhi i, sc i with
      | .scalar dl, .scalar dh, .scalar s =>
        Tensor.scalar (if s > Numbers.zero then s * dh else s * dl))
    { lo := outLo, hi := outHi }

/--
Propagate second-derivative bounds through inference-time BatchNorm.

Although the second derivative of the affine map `x ↦ scale * x + offset` with respect to `x` is
zero, composition with a curve `x(t)` gives `d²/dt² BN(x(t)) = scale * x''(t)`. Consequently this
uses the same signed scaling rule as first-derivative propagation.
-/
def secondDerivBatchnorm (params : BatchNormParams α)
    (d2B : Box α (.dim params.dim .scalar)) : Box α (.dim params.dim .scalar) :=
  derivBatchnorm params d2B

end NN.MLTheory.CROWN.Operators.Batchnorm
