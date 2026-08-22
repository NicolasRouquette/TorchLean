/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.TensorPack
public import NN.API.Runtime
public import NN.API.Neural.Builders

import Mathlib.Algebra.Order.Algebra

@[expose] public section

namespace TorchLean


/-!
# Automatic Differentiation

This module contains gradient, VJP, Jacobian, JVP, and HVP operations for models and pure
one-argument tensor functions.
-/

namespace autograd

/-!
Autograd operations (grad/vjp/jacobian) over TorchLean programs.

This namespace is conceptually similar to PyTorch autograd + functorch/`torch.func`:
- gradients of losses with respect to model state and inputs
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

/-- Complete model state, indexed by its statically known tensor shapes. -/
abbrev State {σ τ : Spec.Shape} (model : nn.Sequential σ τ) (α : Type) :=
  _root_.Runtime.Autograd.Torch.TList α
    (_root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes model)

/-- A scalar loss computed from a model output and its target. -/
abbrev OutputLoss (τ υ : Spec.Shape) :=
  ∀ {α : Type}, [Context α] → [DecidableEq Spec.Shape] →
    {m : Type → Type} → [Monad m] →
      [_root_.Runtime.Autograd.Torch.Ops (m := m) (α := α)] →
      _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) τ →
      _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) υ →
      m (_root_.Runtime.Autograd.TorchLean.RefTy
        (m := m) (α := α) Spec.Shape.scalar)

/-- Cast a model's initial `Float` state into another scalar representation. -/
def initStateWith {σ τ : Spec.Shape} (model : nn.Sequential σ τ)
    {α : Type} (cast : Float → α) : State model α :=
  _root_.Runtime.Autograd.TorchLean.Module.castTList cast
    (_root_.Runtime.Autograd.TorchLean.NN.Seq.initState model)

/-- Initialize model state in a scalar representation that accepts host `Float` values. -/
def initState {σ τ : Spec.Shape} (model : nn.Sequential σ τ)
    {α : Type} [Runtime.FromFloat α] : State model α :=
  initStateWith model (Runtime.ofFloat (α := α))

namespace OutputLoss

/-- Mean-squared error between a model output and its target. -/
def mse {τ : Spec.Shape}
    (reduction : _root_.TorchLean.Loss.Reduction := .mean) :
    model.OutputLoss τ τ :=
  fun {α} _ _ => fun {m} _ _ output target =>
    _root_.TorchLean.Loss.mse
      (m := m) (α := α) (s := τ) output target (reduction := reduction)

/-- Cross-entropy between logits and one-hot targets along the selected class dimension. -/
def oneHotCrossEntropy {τ : Spec.Shape}
    (axis : Nat) [Spec.Shape.AxisInBounds axis τ]
    (reduction : _root_.TorchLean.Loss.Reduction := .mean) :
    model.OutputLoss τ τ :=
  fun {α} _ _ => fun {m} _ _ logits target =>
    _root_.TorchLean.Loss.oneHotCrossEntropy
      (m := m) (α := α) (s := τ) axis logits target (reduction := reduction)

/-- Stop gradients through the model output before evaluating `loss`. -/
def detach {τ υ : Spec.Shape} (loss : model.OutputLoss τ υ) : model.OutputLoss τ υ :=
  fun {α} _ _ => fun {m} _ _ output target => do
    let output' ← _root_.Runtime.Autograd.TorchLean.F.detach
      (m := m) (α := α) (s := τ) output
    loss (α := α) (m := m) output' target

end OutputLoss

/-- Lower `loss (model state input) target` to a scalar typed graph. -/
def lossProgram {σ τ υ : Spec.Shape}
    (model : nn.Sequential σ τ) (loss : OutputLoss τ υ) :
    ∀ {α : Type}, [Context α] → [DecidableEq Spec.Shape] →
      _root_.Runtime.Autograd.TorchLean.Program α
        (_root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes model ++ [σ, υ])
        Spec.Shape.scalar :=
  fun {α} _ _ => fun {m} _ _ =>
    _root_.Runtime.Autograd.Torch.CurriedRef.curry
      (Ref := fun s => _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) s)
      (ss := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes model ++ [σ, υ])
      (β := m (_root_.Runtime.Autograd.TorchLean.RefTy
        (m := m) (α := α) Spec.Shape.scalar))
      (fun args => do
        let (state, inputs) :=
          _root_.Runtime.Autograd.Torch.RefList.split
            (Ref := fun s => _root_.Runtime.Autograd.TorchLean.RefTy
              (m := m) (α := α) s)
            (ss₁ := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes model)
            (ss₂ := [σ, υ]) args
        let (input, target) := match inputs with
          | .cons input (.cons target .nil) => (input, target)
        let output ←
          _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
            (Ref := fun s => _root_.Runtime.Autograd.TorchLean.RefTy
              (m := m) (α := α) s)
            (ss := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes model ++ [σ])
            (β := m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) τ))
            (_root_.Runtime.Autograd.TorchLean.NN.Seq.forward model (α := α))
            (_root_.Runtime.Autograd.Torch.RefList.append state (.cons input .nil))
        loss (α := α) (m := m) output target)

/--
Gradient of a model loss with respect to every tensor in the model state.

The result has the same shape-indexed layout as `State model α`. For models with persistent buffers,
this computes mathematical sensitivities for those entries as well; `nn.requiresGrad` separately
controls which state tensors an optimizer updates.
-/
def gradState {σ τ υ : Spec.Shape} (model : nn.Sequential σ τ) (loss :
  OutputLoss τ υ)
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (state : State model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ) :
    IO (State model α) :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.gradParams
    (α := α)
    (paramShapes := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes model)
    (inputShapes := [σ, υ])
    (lossProgram model loss) state (.cons x (.cons target .nil))

/-- Gradient of the loss w.r.t. the inputs (`x` and `target`). -/
def gradInputs {σ τ υ : Spec.Shape} (model : nn.Sequential σ τ) (loss :
  OutputLoss τ υ)
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (state : State model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ) :
    IO (_root_.TorchLean.TensorPack α [σ, υ]) :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.gradInputs
    (α := α)
    (paramShapes := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes model)
    (inputShapes := [σ, υ])
    (lossProgram model loss) state (.cons x (.cons target .nil))

/-- Convenience: gradient of the loss w.r.t. `x`. -/
def gradX {σ τ υ : Spec.Shape} (model : nn.Sequential σ τ) (loss :
  OutputLoss τ υ)
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (state : State model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ) :
    IO (Spec.Tensor α σ) := do
  let gxs ← gradInputs (model := model) (loss := loss) (α := α) state x target
  pure (TensorPack.get gxs ⟨0, by simp⟩)

/-- Convenience: gradient of the loss w.r.t. the `target` argument. -/
def gradTarget {σ τ υ : Spec.Shape} (model : nn.Sequential σ τ) (loss :
  OutputLoss τ υ)
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (state : State model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ) :
    IO (Spec.Tensor α υ) := do
  let gxs ← gradInputs (model := model) (loss := loss) (α := α) state x target
  pure (TensorPack.get gxs ⟨1, by simp⟩)

/--
Forward+backward result for a scalar loss built from a model output.

PyTorch comparison: this is the "compute loss + backward" payload, but with shapes tracked.
-/
structure ValueAndGrads {σ τ υ : Spec.Shape} (model : nn.Sequential σ τ) (α : Type) where
  /-- Value at the current point. -/
  value : Spec.Tensor α Spec.Shape.scalar
  /-- Gradients with respect to the complete model state. -/
  dState : State model α
  /-- Gradient w.r.t. input. -/
  dx : Spec.Tensor α σ
  /-- Gradient w.r.t. target. -/
  dtarget : Spec.Tensor α υ

/--
Run `loss(model(state, x), target)` and compute gradients w.r.t:

- model state,
- `x`,
- `target`.

This hides the `TypedScalarGraph`/argument-pack boilerplate for the common "one sample" case.
-/
def valueAndGrads {σ τ υ : Spec.Shape} (model : nn.Sequential σ τ) (loss :
  OutputLoss τ υ)
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (state : State model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ) :
    IO (ValueAndGrads (model := model) (α := α) (σ := σ) (υ := υ)) := do
  let stateShapes := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes model
  let c ←
    _root_.Runtime.Autograd.TorchLean.Autodiff.lowerScalarToTypedGraph (α := α)
      (paramShapes := stateShapes) (inputShapes := [σ, υ])
      (lossProgram model loss)

  let args : _root_.TorchLean.TensorPack α (stateShapes ++ [σ, υ]) :=
    TensorPack.append (ss₁ := stateShapes) (ss₂ := [σ, υ]) state
      (.cons x (.cons target .nil))

  let value : Spec.Tensor α Spec.Shape.scalar :=
    _root_.Runtime.Autograd.Torch.TypedScalarGraph.forward (α := α) (Γ := stateShapes ++ [σ, υ]) c
      args

  let gAll : _root_.TorchLean.TensorPack α (stateShapes ++ [σ, υ]) :=
    _root_.Runtime.Autograd.Torch.TypedScalarGraph.backward (α := α) (Γ := stateShapes ++ [σ, υ]) c
      args

  let (dps, dxys) :=
    TensorPack.split (α := α) (ss₁ := stateShapes) (ss₂ := [σ, υ]) gAll

  pure
    { value := value
      dState := dps
      dx := TensorPack.get dxys ⟨0, by simp⟩
      dtarget := TensorPack.get dxys ⟨1, by simp⟩ }

/-- Return the scalar loss tensor together with gradients for the complete model state. -/
def valueAndGradState {σ τ υ : Spec.Shape} (model : nn.Sequential σ τ) (loss :
  OutputLoss τ υ)
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (state : State model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ) :
    IO (Spec.Tensor α Spec.Shape.scalar × State model α) := do
  let out ← valueAndGrads (model := model) (loss := loss) (α := α) state x target
  pure (out.value, out.dState)

/-- `valueAndGradState`, but convert the 0-dim loss tensor to a scalar `α`. -/
def valueAndGradStateScalar {σ τ υ : Spec.Shape} (model : nn.Sequential σ τ) (loss :
  OutputLoss τ υ)
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (state : State model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ) :
    IO (α × State model α) := do
  let (valueT, dps) ← valueAndGradState (model := model) (loss := loss) (α := α) state x target
  pure (Spec.Tensor.item valueT, dps)

/-- Return `(loss_value, grad_x)`. -/
def valueAndGradX {σ τ υ : Spec.Shape} (model : nn.Sequential σ τ) (loss :
  OutputLoss τ υ)
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (state : State model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ) :
    IO (Spec.Tensor α Spec.Shape.scalar × Spec.Tensor α σ) := do
  let out ← valueAndGrads (model := model) (loss := loss) (α := α) state x target
  pure (out.value, out.dx)

/-- Return `(loss_value, grad_target)`. -/
def valueAndGradTarget {σ τ υ : Spec.Shape} (model : nn.Sequential σ τ) (loss :
  OutputLoss τ υ)
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (state : State model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ) :
    IO (Spec.Tensor α Spec.Shape.scalar × Spec.Tensor α υ) := do
  let out ← valueAndGrads (model := model) (loss := loss) (α := α) state x target
  pure (out.value, out.dtarget)

/--
Vector-Jacobian product (VJP) with respect to the complete model state.

Use it for custom losses or analysis tooling when you already have an output cotangent
`seedOut : Tensor α τ`.
-/
def vjpState {σ τ : Spec.Shape} (model : nn.Sequential σ τ)
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (state : State model α)
    (x : Spec.Tensor α σ) (seedOut : Spec.Tensor α τ) :
    IO (State model α) :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.vjpOutParams
    (α := α)
    (paramShapes := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes model)
    (inputShapes := [σ]) (τ := τ)
    (fun {β} _ _ =>
      _root_.Runtime.Autograd.TorchLean.NN.Seq.forward model (α := β))
    state (.cons x .nil) seedOut

/--
VJP w.r.t. the model input.

Returns a one-element `_root_.TorchLean.TensorPack` to match the general "inputs list" API shape. For the common
case, use `vjpInput` to get the tensor directly.
-/
def vjpInputs {σ τ : Spec.Shape} (model : nn.Sequential σ τ)
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (state : State model α)
    (x : Spec.Tensor α σ) (seedOut : Spec.Tensor α τ) :
    IO (_root_.TorchLean.TensorPack α [σ]) :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.vjpOutInputs
    (α := α)
    (paramShapes := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes model)
    (inputShapes := [σ]) (τ := τ)
    (fun {β} _ _ =>
      _root_.Runtime.Autograd.TorchLean.NN.Seq.forward model (α := β))
    state (.cons x .nil) seedOut

/-- Vector-Jacobian product with respect to the single model input tensor. -/
def vjpInput {σ τ : Spec.Shape} (model : nn.Sequential σ τ)
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (state : State model α)
    (x : Spec.Tensor α σ) (seedOut : Spec.Tensor α τ) :
    IO (Spec.Tensor α σ) := do
  let dxs ← vjpInputs (model := model) (α := α) state x seedOut
  pure (TensorPack.get dxs ⟨0, by simp⟩)

/--
Reverse-mode Jacobian (`jacrev`) of the model output with respect to model state.

Returns an array of state-structured gradients: one entry per output coordinate.
This mirrors the usual "jacrev returns a stack of per-output gradients" shape.
-/
def jacrevState {σ τ : Spec.Shape} (model : nn.Sequential σ τ)
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (state : State model α)
    (x : Spec.Tensor α σ) :
    IO (Array (State model α)) :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.jacrevOutParams
    (α := α)
    (paramShapes := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes model)
    (inputShapes := [σ]) (τ := τ)
    (fun {β} _ _ =>
      _root_.Runtime.Autograd.TorchLean.NN.Seq.forward model (α := β))
    state (.cons x .nil)

/--
Jacobian-vector product (JVP) of a scalar loss with respect to model state.

Directional derivative in the direction `vState`. Conceptually:

$$
\left.\frac{d}{dt}
\operatorname{loss}(\mathrm{state}+t\,\mathrm{vState},x,\mathrm{target})
\right|_{t=0}.
$$
-/
def jvpState {σ τ υ : Spec.Shape} (model : nn.Sequential σ τ) (loss :
  OutputLoss τ υ)
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (state : State model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ)
    (vState : State model α) :
    IO α :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.jvpLossParams
    (α := α)
    (paramShapes := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes model)
    (inputShapes := [σ, υ])
    (lossProgram model loss) state (.cons x (.cons target .nil)) vState

/--
Hessian-vector product (HVP) of a scalar loss with respect to model state.

Returns a state-structured tensor list with the same layout as `state`.
-/
def hvpState {σ τ υ : Spec.Shape} (model : nn.Sequential σ τ) (loss :
  OutputLoss τ υ)
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (state : State model α)
    (x : Spec.Tensor α σ) (target : Spec.Tensor α υ)
    (vState : State model α) :
    IO (State model α) :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.hvpParams
    (α := α)
    (paramShapes := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes model)
    (inputShapes := [σ, υ])
    (lossProgram model loss) state (.cons x (.cons target .nil)) vState
end model

namespace func

/-
Pure-function autograd: treat a pure function `f : Tensor σ -> Tensor τ` as the object of
differentiation (no parameters).
-/

/-!
In PyTorch terms, this is the "functorch" style: differentiate plain functions, not modules.
-/

/-- A scalar-polymorphic tensor function written against TorchLean's differentiable operations. -/
abbrev TensorFunction (σ τ : Spec.Shape) :=
  ∀ {α : Type}, [Context α] → [DecidableEq Spec.Shape] →
    {m : Type → Type} → [Monad m] →
      [_root_.Runtime.Autograd.Torch.Ops (m := m) (α := α)] →
      _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) σ →
      m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) τ)

/-- Adapt a tensor function to the single-input program representation used by autograd. -/
def forwardProgram {σ τ : Spec.Shape} (f : TensorFunction σ τ) :
    ∀ {α : Type}, [Context α] → [DecidableEq Spec.Shape] →
      _root_.Runtime.Autograd.TorchLean.Program α [σ] τ :=
  fun {α} _ _ => fun {m} _ _ =>
    _root_.Runtime.Autograd.Torch.CurriedRef.curry
      (Ref := fun s => _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) s)
      (ss := [σ])
      (β := m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) τ))
      (fun args =>
        match args with
        | .cons input .nil => f (α := α) (m := m) input)

/-- Forward-mode Jacobian (`jacfwd`) for a pure tensor function. -/
def jacfwd {σ τ : Spec.Shape} (f : TensorFunction σ τ)
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (x : Spec.Tensor α σ) :
    IO (Array (Spec.Tensor α τ)) :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.jacfwdInput
    (α := α) (σ := σ) (τ := τ) (forwardProgram f) x

/-- Hessian for a scalar-valued function. -/
def hessian {σ : Spec.Shape} (f : TensorFunction σ Spec.Shape.scalar)
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (x : Spec.Tensor α σ) :
    IO (Array (Spec.Tensor α σ)) :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.hessianInput
    (α := α) (σ := σ) (forwardProgram f) x

/-- Vector-Jacobian product (VJP) for a pure function. -/
def vjp {σ τ : Spec.Shape} (f : TensorFunction σ τ)
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (x : Spec.Tensor α σ) (seedOut : Spec.Tensor α τ) :
    IO (Spec.Tensor α σ) := do
  let state : _root_.TorchLean.TensorPack α ([] : List Spec.Shape) := .nil
  let gxs ←
    _root_.Runtime.Autograd.TorchLean.Autodiff.vjpOutInputs (α := α)
      (paramShapes := ([] : List Spec.Shape)) (inputShapes := [σ]) (τ := τ)
      (forwardProgram (σ := σ) (τ := τ) f)
      state (.cons x .nil) seedOut
  pure (TensorPack.get gxs ⟨0, by simp⟩)

/--
Reverse-mode Jacobian (`jacrev`) of a pure tensor function.

Returns the Jacobian rows as an array of `doutput/dinput` tensors.
-/
def jacrev {σ τ : Spec.Shape} (f : TensorFunction σ τ)
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (x : Spec.Tensor α σ) :
    IO (Array (Spec.Tensor α σ)) := do
  let state : _root_.TorchLean.TensorPack α ([] : List Spec.Shape) := .nil
  let rows ←
    _root_.Runtime.Autograd.TorchLean.Autodiff.jacrevOutInputs (α := α)
      (paramShapes := ([] : List Spec.Shape)) (inputShapes := [σ]) (τ := τ)
      (forwardProgram (σ := σ) (τ := τ) f)
      state (.cons x .nil)
  pure <| rows.map fun xs => TensorPack.get xs ⟨0, by simp⟩

/-- Gradient of a scalar-valued function w.r.t. its input. -/
def grad {σ : Spec.Shape} (f : TensorFunction σ Spec.Shape.scalar)
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (x : Spec.Tensor α σ) :
    IO (Spec.Tensor α σ) := do
  let state : _root_.TorchLean.TensorPack α ([] : List Spec.Shape) := .nil
  let gxs ←
    _root_.Runtime.Autograd.TorchLean.Autodiff.gradInputs (α := α)
      (paramShapes := ([] : List Spec.Shape)) (inputShapes := [σ])
      (forwardProgram (σ := σ) (τ := Spec.Shape.scalar) f)
      state (.cons x .nil)
  pure (TensorPack.get gxs ⟨0, by simp⟩)

/-- Return `(value, grad)` for a scalar-valued function at `x`. -/
def valueAndGrad {σ : Spec.Shape} (f : TensorFunction σ Spec.Shape.scalar)
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (x : Spec.Tensor α σ) :
    IO (Spec.Tensor α Spec.Shape.scalar × Spec.Tensor α σ) := do
  let c ←
    _root_.Runtime.Autograd.TorchLean.Autodiff.lowerScalarToTypedGraph (α := α)
      (paramShapes := ([] : List Spec.Shape)) (inputShapes := [σ])
      (forwardProgram (σ := σ) (τ := Spec.Shape.scalar) f)
  let args : _root_.TorchLean.TensorPack α [σ] := .cons x .nil
  let value : Spec.Tensor α Spec.Shape.scalar :=
    _root_.Runtime.Autograd.Torch.TypedScalarGraph.forward (α := α) (Γ := [σ]) c args
  let gAll : _root_.TorchLean.TensorPack α [σ] :=
    _root_.Runtime.Autograd.Torch.TypedScalarGraph.backward (α := α) (Γ := [σ]) c args
  pure (value, TensorPack.get gAll ⟨0, by simp⟩)

/-- `valueAndGrad`, but convert the 0-dim value tensor to a scalar `α`. -/
def valueAndGradScalar {σ : Spec.Shape} (f : TensorFunction σ Spec.Shape.scalar)
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (x : Spec.Tensor α σ) :
    IO (α × Spec.Tensor α σ) := do
  let (valueT, g) ← valueAndGrad (f := f) (α := α) x
  pure (Spec.Tensor.item valueT, g)
end func

end autograd

end TorchLean
