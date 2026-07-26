/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Adapters
public import NN.API.Common
public import NN.API.Data
public import NN.API.Macros
public import NN.API.Public.TensorPack
public import NN.API.Rand
public import NN.API.RL
public import NN.API.Runtime
public import NN.API.Samples
public import NN.API.SelfSupervised
public import NN.API.Text
public import NN.API.Text.Bpe
public import NN.Spec.Layers.PositionalEncoding
public import NN.API.Public.NN

import Mathlib.Algebra.Order.Algebra

@[expose] public section

namespace NN
namespace API

/-!
# Public autograd operations

This module contains the public gradient, VJP, Jacobian, JVP, and HVP APIs for models and pure
one-argument tensor functions.
-/

namespace autograd

/-!
Autograd operations (grad/vjp/jacobian) over TorchLean programs.

This namespace is conceptually similar to PyTorch autograd + functorch/`torch.func`:
- gradients of losses w.r.t. parameters and inputs
- VJPs and Jacobians for analysis and verification tooling

PyTorch references:
- Autograd: `https://pytorch.org/docs/stable/autograd.html`
- `torch.func` (jacfwd/jacrev, etc.): `https://pytorch.org/docs/stable/func.html`
-/

namespace model

/-
Model-shaped autograd: a TorchLean `NN.Seq` plus an `OutputLoss` over its output.

This covers the common training use case.
-/

@[inherit_doc TorchLean.Autodiff.Model.Params]
abbrev Params {σ τ : Spec.Shape} (model : TorchLean.LayerCore.Seq σ τ) (α : Type) :=
  TorchLean.Autodiff.Model.Params model α

@[inherit_doc TorchLean.Autodiff.Model.OutputLoss]
abbrev OutputLoss (τ υ : Spec.Shape) :=
  TorchLean.Autodiff.Model.OutputLoss τ υ

@[inherit_doc TorchLean.Autodiff.Model.linearParams]
abbrev linearParams {α : Type} {inDim outDim : Nat} {seedW seedB : Nat}
    (w : Spec.Tensor α (.dim outDim (.dim inDim .scalar)))
    (b : Spec.Tensor α (.dim outDim .scalar)) :
    Params (TorchLean.Layers.linear inDim outDim seedW seedB) α :=
  TorchLean.Autodiff.Model.linearParams
    (α := α) (inDim := inDim) (outDim := outDim) (seedW := seedW) (seedB := seedB) w b

namespace OutputLoss

@[inherit_doc TorchLean.Autodiff.Model.OutputLoss.mse]
abbrev mse {τ : Spec.Shape} (reduction : TorchLean.Loss.Reduction := .mean) :
    model.OutputLoss τ τ :=
  TorchLean.Autodiff.Model.OutputLoss.mse (τ := τ) (reduction := reduction)

@[inherit_doc TorchLean.Autodiff.Model.OutputLoss.crossEntropyOneHot]
abbrev crossEntropyOneHot {τ : Spec.Shape} (reduction : TorchLean.Loss.Reduction := .mean) :
    model.OutputLoss τ τ :=
  TorchLean.Autodiff.Model.OutputLoss.crossEntropyOneHot (τ := τ) (reduction := reduction)

@[inherit_doc TorchLean.Autodiff.Model.OutputLoss.detach]
abbrev detach {τ υ : Spec.Shape} (loss : model.OutputLoss τ υ) :
    model.OutputLoss τ υ :=
  TorchLean.Autodiff.Model.OutputLoss.detach loss

end OutputLoss

/--
Gradient of a model-loss w.r.t. the model parameters.

Common training use case. PyTorch analogue: `loss.backward()` followed by parameter updates.
-/
def gradParams {σ τ υ : Spec.Shape} (model : TorchLean.LayerCore.Seq σ τ) (loss :
  TorchLean.Autodiff.Model.OutputLoss τ υ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (params : TorchLean.Autodiff.Model.Params model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ) :
    IO (TorchLean.Autodiff.Model.Params model α) :=
  TorchLean.Autodiff.Model.gradParams (α := α) model loss params x target

/-- Gradient of the loss w.r.t. the inputs (`x` and `target`). -/
def gradInputs {σ τ υ : Spec.Shape} (model : TorchLean.LayerCore.Seq σ τ) (loss :
  TorchLean.Autodiff.Model.OutputLoss τ υ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (params : TorchLean.Autodiff.Model.Params model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ) :
    IO (TorchLean.TensorPack α [σ, υ]) :=
  TorchLean.Autodiff.Model.gradInputs (α := α) model loss params x target

/-- Convenience: gradient of the loss w.r.t. `x`. -/
def gradX {σ τ υ : Spec.Shape} (model : TorchLean.LayerCore.Seq σ τ) (loss :
  TorchLean.Autodiff.Model.OutputLoss τ υ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (params : TorchLean.Autodiff.Model.Params model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ) :
    IO (Spec.Tensor α σ) := do
  let gxs ← gradInputs (model := model) (loss := loss) (α := α) params x target
  pure (tensorpack.first gxs)

/-- Convenience: gradient of the loss w.r.t. the `target` argument. -/
def gradTarget {σ τ υ : Spec.Shape} (model : TorchLean.LayerCore.Seq σ τ) (loss :
  TorchLean.Autodiff.Model.OutputLoss τ υ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (params : TorchLean.Autodiff.Model.Params model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ) :
    IO (Spec.Tensor α υ) := do
  let gxs ← gradInputs (model := model) (loss := loss) (α := α) params x target
  pure (tensorpack.second gxs)

/--
Forward+backward result for a scalar loss built from a model output.

PyTorch comparison: this is the "compute loss + backward" payload, but with shapes tracked.
-/
structure ValueAndGrads {σ τ υ : Spec.Shape} (model : TorchLean.LayerCore.Seq σ τ) (α : Type) where
  /-- Value at the current point. -/
  value : Spec.Tensor α Spec.Shape.scalar
  /-- Gradients w.r.t. parameters. -/
  dparams : TorchLean.Autodiff.Model.Params model α
  /-- Gradient w.r.t. input. -/
  dx : Spec.Tensor α σ
  /-- Gradient w.r.t. target. -/
  dtarget : Spec.Tensor α υ

/--
Run `loss(model(params, x), target)` and compute gradients w.r.t:

- model parameters,
- `x`,
- `target`.

This hides the `CompiledScalar`/argument-pack boilerplate for the common "one sample" case.
-/
def valueAndGrads {σ τ υ : Spec.Shape} (model : TorchLean.LayerCore.Seq σ τ) (loss :
  TorchLean.Autodiff.Model.OutputLoss τ υ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (params : TorchLean.Autodiff.Model.Params model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ) :
    IO (ValueAndGrads (model := model) (α := α) (σ := σ) (υ := υ)) := do
  let paramShapes := TorchLean.LayerCore.Seq.paramShapes model
  let c ←
    TorchLean.Autodiff.compileLoss (α := α)
      (paramShapes := paramShapes) (inputShapes := [σ, υ])
      (TorchLean.Autodiff.Model.lossProgram (model := model) loss)

  let args : TorchLean.TensorPack α (paramShapes ++ [σ, υ]) :=
    tensorpack.append (ss₁ := paramShapes) (ss₂ := [σ, υ]) params (tensorpack.pair x target)

  let value : Spec.Tensor α Spec.Shape.scalar :=
    _root_.Runtime.Autograd.Torch.CompiledScalar.forward (α := α) (Γ := paramShapes ++ [σ, υ]) c
      args

  let gAll : TorchLean.TensorPack α (paramShapes ++ [σ, υ]) :=
    _root_.Runtime.Autograd.Torch.CompiledScalar.backward (α := α) (Γ := paramShapes ++ [σ, υ]) c
      args

  let (dps, dxys) :=
    tensorpack.split (α := α) (ss₁ := paramShapes) (ss₂ := [σ, υ]) gAll

  pure
    { value := value
      dparams := dps
      dx := tensorpack.first dxys
      dtarget := tensorpack.second dxys }

/-- Return the scalar loss tensor together with gradients for the model parameters. -/
def valueAndGradParams {σ τ υ : Spec.Shape} (model : TorchLean.LayerCore.Seq σ τ) (loss :
  TorchLean.Autodiff.Model.OutputLoss τ υ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (params : TorchLean.Autodiff.Model.Params model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ) :
    IO (Spec.Tensor α Spec.Shape.scalar × TorchLean.Autodiff.Model.Params model α) := do
  let out ← valueAndGrads (model := model) (loss := loss) (α := α) params x target
  pure (out.value, out.dparams)

/-- `valueAndGradParams`, but convert the 0-dim loss tensor to a scalar `α`. -/
def valueAndGradParamsScalar {σ τ υ : Spec.Shape} (model : TorchLean.LayerCore.Seq σ τ) (loss :
  TorchLean.Autodiff.Model.OutputLoss τ υ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (params : TorchLean.Autodiff.Model.Params model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ) :
    IO (α × TorchLean.Autodiff.Model.Params model α) := do
  let (valueT, dps) ← valueAndGradParams (model := model) (loss := loss) (α := α) params x target
  pure (Spec.Tensor.toScalar valueT, dps)

/-- Return `(loss_value, grad_x)`. -/
def valueAndGradX {σ τ υ : Spec.Shape} (model : TorchLean.LayerCore.Seq σ τ) (loss :
  TorchLean.Autodiff.Model.OutputLoss τ υ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (params : TorchLean.Autodiff.Model.Params model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ) :
    IO (Spec.Tensor α Spec.Shape.scalar × Spec.Tensor α σ) := do
  let out ← valueAndGrads (model := model) (loss := loss) (α := α) params x target
  pure (out.value, out.dx)

/-- Return `(loss_value, grad_target)`. -/
def valueAndGradTarget {σ τ υ : Spec.Shape} (model : TorchLean.LayerCore.Seq σ τ) (loss :
  TorchLean.Autodiff.Model.OutputLoss τ υ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (params : TorchLean.Autodiff.Model.Params model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ) :
    IO (Spec.Tensor α Spec.Shape.scalar × Spec.Tensor α υ) := do
  let out ← valueAndGrads (model := model) (loss := loss) (α := α) params x target
  pure (out.value, out.dtarget)

/--
Vector-Jacobian product (VJP) w.r.t. model parameters.

Primitive for sending output cotangents back into parameters. Use it for custom losses or analysis
tooling when you already have a seed tensor `seedOut : τ`.
-/
def vjpParams {σ τ : Spec.Shape} (model : TorchLean.LayerCore.Seq σ τ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (params : TorchLean.Autodiff.Model.Params model α)
    (x : Spec.Tensor α σ) (seedOut : Spec.Tensor α τ) :
    IO (TorchLean.Autodiff.Model.Params model α) :=
  TorchLean.Autodiff.Model.vjpParams (α := α) model params x seedOut

/--
VJP w.r.t. the model input.

Returns a one-element `TorchLean.TensorPack` to match the general "inputs list" API shape. For the common
case, use `vjpInput` to get the tensor directly.
-/
def vjpInputs {σ τ : Spec.Shape} (model : TorchLean.LayerCore.Seq σ τ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (params : TorchLean.Autodiff.Model.Params model α)
    (x : Spec.Tensor α σ) (seedOut : Spec.Tensor α τ) :
    IO (TorchLean.TensorPack α [σ]) :=
  TorchLean.Autodiff.Model.vjpInputs (α := α) model params x seedOut

/-- Vector-Jacobian product with respect to the single model input tensor. -/
def vjpInput {σ τ : Spec.Shape} (model : TorchLean.LayerCore.Seq σ τ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (params : TorchLean.Autodiff.Model.Params model α)
    (x : Spec.Tensor α σ) (seedOut : Spec.Tensor α τ) :
    IO (Spec.Tensor α σ) := do
  let dxs ← vjpInputs (model := model) (α := α) params x seedOut
  pure (tensorpack.unpackSingleton dxs)

/--
Reverse-mode Jacobian (`jacrev`) of the model output w.r.t. parameters.

Returns an array of parameter-structured gradients: one entry per output coordinate.
This mirrors the usual "jacrev returns a stack of per-output gradients" shape.
-/
def jacrevParams {σ τ : Spec.Shape} (model : TorchLean.LayerCore.Seq σ τ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (params : TorchLean.Autodiff.Model.Params model α)
    (x : Spec.Tensor α σ) :
    IO (Array (TorchLean.Autodiff.Model.Params model α)) :=
  TorchLean.Autodiff.Model.jacrevParams (α := α) model params x

/--
Jacobian-vector product (JVP) of a scalar loss w.r.t. parameters.

Directional derivative in the direction `vparams`. Conceptually:

$$
\left.\frac{d}{dt}
\operatorname{loss}(\mathrm{params}+t\,\mathrm{vparams},x,\mathrm{target})
\right|_{t=0}.
$$
-/
def jvpParams {σ τ υ : Spec.Shape} (model : TorchLean.LayerCore.Seq σ τ) (loss :
  TorchLean.Autodiff.Model.OutputLoss τ υ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (params : TorchLean.Autodiff.Model.Params model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ)
    (vparams : TorchLean.Autodiff.Model.Params model α) :
    IO α :=
  TorchLean.Autodiff.Model.jvpParams (α := α) model loss params x target vparams

/--
Hessian-vector product (HVP) of a scalar loss w.r.t. parameters.

Returns a parameter-structured tensor list of the same shape as `params`.
-/
def hvpParams {σ τ υ : Spec.Shape} (model : TorchLean.LayerCore.Seq σ τ) (loss :
  TorchLean.Autodiff.Model.OutputLoss τ υ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (params : TorchLean.Autodiff.Model.Params model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ)
    (vparams : TorchLean.Autodiff.Model.Params model α) :
    IO (TorchLean.Autodiff.Model.Params model α) :=
  TorchLean.Autodiff.Model.hvpParams (α := α) model loss params x target vparams
end model

namespace func

/-
Pure-function autograd: treat a pure function `f : Tensor σ -> Tensor τ` as the object of
differentiation (no parameters).
-/

/-!
In PyTorch terms, this is the "functorch" style: differentiate plain functions, not modules.
-/

@[inherit_doc TorchLean.Autodiff.Function.Fn]
abbrev Fn (σ τ : Spec.Shape) :=
  TorchLean.Autodiff.Function.Fn σ τ

/-- Forward-mode Jacobian (`jacfwd`) for a pure tensor function. -/
def jacfwd {σ τ : Spec.Shape} (f : TorchLean.Autodiff.Function.Fn σ τ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (x : Spec.Tensor α σ) :
    IO (Array (Spec.Tensor α τ)) :=
  TorchLean.Autodiff.Function.jacfwd (α := α) f x

/-- Hessian for a scalar-valued function. -/
def hessian {σ : Spec.Shape} (f : TorchLean.Autodiff.Function.Fn σ Spec.Shape.scalar)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (x : Spec.Tensor α σ) :
    IO (Array (Spec.Tensor α σ)) :=
  TorchLean.Autodiff.Function.hessian (α := α) f x

/-- Vector-Jacobian product (VJP) for a pure function. -/
def vjp {σ τ : Spec.Shape} (f : TorchLean.Autodiff.Function.Fn σ τ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (x : Spec.Tensor α σ) (seedOut : Spec.Tensor α τ) :
    IO (Spec.Tensor α σ) := do
  let params : TorchLean.TensorPack α ([] : List Spec.Shape) := .nil
  let gxs ←
    TorchLean.Autodiff.vjpOutInputs (α := α)
      (paramShapes := ([] : List Spec.Shape)) (inputShapes := [σ]) (τ := τ)
      (TorchLean.Autodiff.Function.forwardProgram (σ := σ) (τ := τ) f)
      params (tensorpack.singleton x) seedOut
  pure (tensorpack.unpackSingleton gxs)

/--
Reverse-mode Jacobian (`jacrev`) of a pure tensor function.

Returns the Jacobian rows as an array of `doutput/dinput` tensors.
-/
def jacrev {σ τ : Spec.Shape} (f : TorchLean.Autodiff.Function.Fn σ τ)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (x : Spec.Tensor α σ) :
    IO (Array (Spec.Tensor α σ)) := do
  let params : TorchLean.TensorPack α ([] : List Spec.Shape) := .nil
  let rows ←
    TorchLean.Autodiff.jacrevOutInputs (α := α)
      (paramShapes := ([] : List Spec.Shape)) (inputShapes := [σ]) (τ := τ)
      (TorchLean.Autodiff.Function.forwardProgram (σ := σ) (τ := τ) f)
      params (tensorpack.singleton x)
  pure <| rows.map tensorpack.unpackSingleton

/-- Gradient of a scalar-valued function w.r.t. its input. -/
def grad {σ : Spec.Shape} (f : TorchLean.Autodiff.Function.Fn σ Spec.Shape.scalar)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (x : Spec.Tensor α σ) :
    IO (Spec.Tensor α σ) := do
  let params : TorchLean.TensorPack α ([] : List Spec.Shape) := .nil
  let gxs ←
    TorchLean.Autodiff.gradInputs (α := α)
      (paramShapes := ([] : List Spec.Shape)) (inputShapes := [σ])
      (TorchLean.Autodiff.Function.forwardProgram (σ := σ) (τ := Spec.Shape.scalar) f)
      params (tensorpack.singleton x)
  pure (tensorpack.unpackSingleton gxs)

/-- Return `(value, grad)` for a scalar-valued function at `x`. -/
def valueAndGrad {σ : Spec.Shape} (f : TorchLean.Autodiff.Function.Fn σ Spec.Shape.scalar)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (x : Spec.Tensor α σ) :
    IO (Spec.Tensor α Spec.Shape.scalar × Spec.Tensor α σ) := do
  let c ←
    TorchLean.Autodiff.compileLoss (α := α)
      (paramShapes := ([] : List Spec.Shape)) (inputShapes := [σ])
      (TorchLean.Autodiff.Function.forwardProgram (σ := σ) (τ := Spec.Shape.scalar) f)
  let args : TorchLean.TensorPack α [σ] := tensorpack.singleton x
  let value : Spec.Tensor α Spec.Shape.scalar :=
    _root_.Runtime.Autograd.Torch.CompiledScalar.forward (α := α) (Γ := [σ]) c args
  let gAll : TorchLean.TensorPack α [σ] :=
    _root_.Runtime.Autograd.Torch.CompiledScalar.backward (α := α) (Γ := [σ]) c args
  pure (value, tensorpack.unpackSingleton gAll)

/-- `valueAndGrad`, but convert the 0-dim value tensor to a scalar `α`. -/
def valueAndGradScalar {σ : Spec.Shape} (f : TorchLean.Autodiff.Function.Fn σ Spec.Shape.scalar)
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (x : Spec.Tensor α σ) :
    IO (α × Spec.Tensor α σ) := do
  let (valueT, g) ← valueAndGrad (f := f) (α := α) x
  pure (Spec.Tensor.toScalar valueT, g)
end func

end autograd

end API
end NN
