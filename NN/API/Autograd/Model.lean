/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Sample
public import NN.API.Neural.Builders

/-!
# Model Automatic Differentiation

Model state, output losses, and differentiation with respect to model state and inputs.
Import `NN.API.Autograd` for the complete public autograd API.
-/

@[expose] public section

namespace TorchLean

namespace autograd

namespace model

/-
Model-shaped autograd: a TorchLean `NN.Seq` plus an `OutputLoss` over its output.

This covers the common training use case.
-/

/-- Complete model state, indexed by its statically known tensor shapes. -/
abbrev State {inputShape outputShape : List Nat}
    (model : nn.Sequential inputShape outputShape) (α : Type) :=
  _root_.TorchLean.TensorPack α
    (_root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes model)

/-- A scalar loss computed from a model output and its target. -/
abbrev OutputLoss (outputShape targetShape : List Nat) :=
  ∀ {α : Type}, [Context α] → [DecidableEq Shape] →
    {m : Type → Type} → [Monad m] →
      [_root_.Runtime.Autograd.Torch.Ops (m := m) (α := α)] →
      _root_.TorchLean.Runtime.ValueRef (m := m) (α := α) outputShape →
      _root_.TorchLean.Runtime.ValueRef (m := m) (α := α) targetShape →
      m (_root_.TorchLean.Runtime.ValueRef
        (m := m) (α := α) ([] : List Nat))

/-- Cast a model's initial `Float` state into another scalar representation. -/
def initStateWith {inputShape outputShape : List Nat}
    (model : nn.Sequential inputShape outputShape)
    {α : Type} (cast : Float → α) : State model α :=
  _root_.Runtime.Autograd.TorchLean.Module.castPack cast
    (_root_.Runtime.Autograd.TorchLean.NN.Seq.initState model)

/-- Initialize model state in a scalar representation that accepts host `Float` values. -/
def initState {inputShape outputShape : List Nat}
    (model : nn.Sequential inputShape outputShape)
    {α : Type} [Runtime.FromFloat α] : State model α :=
  initStateWith model (Runtime.ofFloat (α := α))

namespace OutputLoss

/-- Mean-squared error between a model output and its target. -/
def mse {outputShape : List Nat}
    (reduction : _root_.TorchLean.Loss.Reduction := .mean) :
    model.OutputLoss outputShape outputShape :=
  fun {α} _ _ => fun {m} _ _ output target =>
    _root_.TorchLean.Loss.mse
      (m := m) (α := α) (s := Shape.ofList outputShape) output target
      (reduction := reduction)

/-- Cross-entropy between logits and one-hot targets along the selected class dimension. -/
def oneHotCrossEntropy {outputShape : List Nat}
    (axis : Nat) [Shape.AxisInBounds axis (Shape.ofList outputShape)]
    (reduction : _root_.TorchLean.Loss.Reduction := .mean) :
    model.OutputLoss outputShape outputShape :=
  fun {α} _ _ => fun {m} _ _ logits target =>
    _root_.TorchLean.Loss.oneHotCrossEntropy
      (m := m) (α := α) (s := Shape.ofList outputShape) axis logits target
      (reduction := reduction)

/-- Stop gradients through the model output before evaluating `loss`. -/
def detach {outputShape targetShape : List Nat}
    (loss : model.OutputLoss outputShape targetShape) :
    model.OutputLoss outputShape targetShape :=
  fun {α} _ _ => fun {m} _ _ output target => do
    let output' ← _root_.Runtime.Autograd.TorchLean.F.detach
      (m := m) (α := α) (s := Shape.ofList outputShape) output
    loss (α := α) (m := m) output' target

end OutputLoss

/-- Lower `loss (model state input) target` to a scalar typed graph. -/
def lossProgram {inputShape outputShape targetShape : List Nat}
    (model : nn.Sequential inputShape outputShape)
    (loss : OutputLoss outputShape targetShape) :
    ∀ {α : Type}, [Context α] → [DecidableEq Shape] →
      _root_.Runtime.Autograd.TorchLean.Program α
        (_root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes model ++
          [Shape.ofList inputShape, Shape.ofList targetShape])
        ([] : List Nat) :=
  fun {α} _ _ => fun {m} _ _ =>
    _root_.Runtime.Autograd.Torch.CurriedRef.curry
      (Ref := fun s => _root_.TorchLean.Runtime.ValueRef (m := m) (α := α) s)
      (ss := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes model ++
        [Shape.ofList inputShape, Shape.ofList targetShape])
      (β := m (_root_.TorchLean.Runtime.ValueRef
        (m := m) (α := α) ([] : List Nat)))
      (fun args => do
        let (state, inputs) :=
          _root_.Runtime.Autograd.Torch.RefList.split
            (Ref := fun s => _root_.TorchLean.Runtime.ValueRef
              (m := m) (α := α) s)
            (ss₁ := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes model)
            (ss₂ := [Shape.ofList inputShape, Shape.ofList targetShape]) args
        let (input, target) := match inputs with
          | .cons input (.cons target .nil) => (input, target)
        let output ←
          _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
            (Ref := fun s => _root_.TorchLean.Runtime.ValueRef
              (m := m) (α := α) s)
            (ss := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes model ++
              [Shape.ofList inputShape])
            (β := m (_root_.TorchLean.Runtime.ValueRef
              (m := m) (α := α) (Shape.ofList outputShape)))
            (_root_.Runtime.Autograd.TorchLean.NN.Seq.forward model (α := α))
            (_root_.Runtime.Autograd.Torch.RefList.append state (.cons input .nil))
        loss (α := α) (m := m) output target)

/--
Gradient of a model loss with respect to every tensor in the model state.

The result has the same shape-indexed layout as `State model α`. For models with persistent buffers,
this computes mathematical sensitivities for those entries as well; `nn.requiresGrad` separately
controls which state tensors an optimizer updates.
-/
def gradState {inputShape outputShape targetShape : List Nat}
    (model : nn.Sequential inputShape outputShape) (loss : OutputLoss outputShape targetShape)
    {α : Type} [_root_.Context α] [DecidableEq Shape]
    (state : State model α)
    (x : Tensor α inputShape) (target : Tensor α targetShape) :
    IO (State model α) :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.gradParams
    (α := α)
    (paramShapes := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes model)
    (inputShapes := [Shape.ofList inputShape, Shape.ofList targetShape])
    (lossProgram model loss) state (.cons x (.cons target .nil))

/-- Gradient of the loss w.r.t. the inputs (`x` and `target`). -/
def gradInputs {inputShape outputShape targetShape : List Nat}
    (model : nn.Sequential inputShape outputShape) (loss : OutputLoss outputShape targetShape)
    {α : Type} [_root_.Context α] [DecidableEq Shape]
    (state : State model α)
    (x : Tensor α inputShape) (target : Tensor α targetShape) :
    IO (_root_.TorchLean.TensorPack α
      [Shape.ofList inputShape, Shape.ofList targetShape]) :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.gradInputs
    (α := α)
    (paramShapes := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes model)
    (inputShapes := [Shape.ofList inputShape, Shape.ofList targetShape])
    (lossProgram model loss) state (.cons x (.cons target .nil))

/--
Forward+backward result for a scalar loss built from a model output.

PyTorch comparison: this is the "compute loss + backward" payload, but with shapes tracked.
-/
structure ValueAndGrads {inputShape outputShape targetShape : List Nat}
    (model : nn.Sequential inputShape outputShape) (α : Type) where
  /-- Value at the current point. -/
  value : Tensor α ([] : List Nat)
  /-- Gradients with respect to the complete model state. -/
  dState : State model α
  /-- Gradient w.r.t. input. -/
  dx : Tensor α inputShape
  /-- Gradient w.r.t. target. -/
  dtarget : Tensor α targetShape

/--
Run `loss(model(state, x), target)` and compute gradients w.r.t:

- model state,
- `x`,
- `target`.

This hides the `TypedScalarGraph`/argument-pack boilerplate for the common "one sample" case.
-/
def valueAndGrads {inputShape outputShape targetShape : List Nat}
    (model : nn.Sequential inputShape outputShape) (loss : OutputLoss outputShape targetShape)
    {α : Type} [_root_.Context α] [DecidableEq Shape]
    (state : State model α)
    (x : Tensor α inputShape) (target : Tensor α targetShape) :
    IO (ValueAndGrads (model := model) (α := α)
      (inputShape := inputShape) (targetShape := targetShape)) := do
  let stateShapes := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes model
  let c ←
    _root_.Runtime.Autograd.TorchLean.Autodiff.lowerScalarToTypedGraph (α := α)
      (paramShapes := stateShapes)
      (inputShapes := [Shape.ofList inputShape, Shape.ofList targetShape])
      (lossProgram model loss)

  let args : _root_.TorchLean.TensorPack α
      (stateShapes ++ [Shape.ofList inputShape, Shape.ofList targetShape]) :=
    _root_.TorchLean.TensorPack.append (ss₁ := stateShapes)
      (ss₂ := [Shape.ofList inputShape, Shape.ofList targetShape]) state
      (.cons x (.cons target .nil))

  let value : Tensor α ([] : List Nat) :=
    _root_.Runtime.Autograd.Torch.TypedScalarGraph.forward (α := α)
      (Γ := stateShapes ++ [Shape.ofList inputShape, Shape.ofList targetShape]) c args

  let gAll : _root_.TorchLean.TensorPack α
      (stateShapes ++ [Shape.ofList inputShape, Shape.ofList targetShape]) :=
    _root_.Runtime.Autograd.Torch.TypedScalarGraph.backward (α := α)
      (Γ := stateShapes ++ [Shape.ofList inputShape, Shape.ofList targetShape]) c args

  let (dps, dxys) :=
    _root_.TorchLean.TensorPack.split (α := α) (ss₁ := stateShapes)
      (ss₂ := [Shape.ofList inputShape, Shape.ofList targetShape]) gAll

  pure
    { value := value
      dState := dps
      dx := _root_.TorchLean.TensorPack.get dxys ⟨0, by simp⟩
      dtarget := _root_.TorchLean.TensorPack.get dxys ⟨1, by simp⟩ }

/-- Return the scalar loss and gradients for the complete model state. -/
def valueAndGradStateScalar {inputShape outputShape targetShape : List Nat}
    (model : nn.Sequential inputShape outputShape) (loss : OutputLoss outputShape targetShape)
    {α : Type} [_root_.Context α] [DecidableEq Shape]
    (state : State model α)
    (x : Tensor α inputShape) (target : Tensor α targetShape) :
    IO (α × State model α) := do
  let out ← valueAndGrads (model := model) (loss := loss) (α := α) state x target
  pure (Spec.Tensor.item out.value, out.dState)

/--
Vector-Jacobian product (VJP) with respect to the complete model state.

Use it for custom losses or analysis tooling when you already have an output cotangent
`seedOut : Tensor α τ`.
-/
def vjpState {inputShape outputShape : List Nat} (model : nn.Sequential inputShape outputShape)
    {α : Type} [_root_.Context α] [DecidableEq Shape]
    (state : State model α)
    (x : Tensor α inputShape) (seedOut : Tensor α outputShape) :
    IO (State model α) :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.vjpOutParams
    (α := α)
    (paramShapes := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes model)
    (inputShapes := [Shape.ofList inputShape]) (τ := Shape.ofList outputShape)
    (fun {β} _ _ =>
      _root_.Runtime.Autograd.TorchLean.NN.Seq.forward model (α := β))
    state (.cons x .nil) seedOut

/-- Vector-Jacobian product with respect to the single model input tensor. -/
def vjpInput {inputShape outputShape : List Nat} (model : nn.Sequential inputShape outputShape)
    {α : Type} [_root_.Context α] [DecidableEq Shape]
    (state : State model α)
    (x : Tensor α inputShape) (seedOut : Tensor α outputShape) :
    IO (Tensor α inputShape) := do
  let dxs ← _root_.Runtime.Autograd.TorchLean.Autodiff.vjpOutInputs
    (α := α)
    (paramShapes := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes model)
    (inputShapes := [Shape.ofList inputShape]) (τ := Shape.ofList outputShape)
    (fun {β} _ _ =>
      _root_.Runtime.Autograd.TorchLean.NN.Seq.forward model (α := β))
    state (.cons x .nil) seedOut
  pure (_root_.TorchLean.TensorPack.get dxs ⟨0, by simp⟩)

/--
Reverse-mode Jacobian (`jacrev`) of the model output with respect to model state.

Returns an array of state-structured gradients: one entry per output coordinate.
This mirrors the usual "jacrev returns a stack of per-output gradients" shape.
-/
def jacrevState {inputShape outputShape : List Nat}
    (model : nn.Sequential inputShape outputShape)
    {α : Type} [_root_.Context α] [DecidableEq Shape]
    (state : State model α)
    (x : Tensor α inputShape) :
    IO (Array (State model α)) :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.jacrevOutParams
    (α := α)
    (paramShapes := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes model)
    (inputShapes := [Shape.ofList inputShape]) (τ := Shape.ofList outputShape)
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
def jvpState {inputShape outputShape targetShape : List Nat}
    (model : nn.Sequential inputShape outputShape) (loss : OutputLoss outputShape targetShape)
    {α : Type} [_root_.Context α] [DecidableEq Shape]
    (state : State model α)
    (x : Tensor α inputShape) (target : Tensor α targetShape)
    (vState : State model α) :
    IO α :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.jvpLossParams
    (α := α)
    (paramShapes := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes model)
    (inputShapes := [Shape.ofList inputShape, Shape.ofList targetShape])
    (lossProgram model loss) state (.cons x (.cons target .nil)) vState

/--
Hessian-vector product (HVP) of a scalar loss with respect to model state.

Returns a tensor pack with the same shape layout as `state`.
-/
def hvpState {inputShape outputShape targetShape : List Nat}
    (model : nn.Sequential inputShape outputShape) (loss : OutputLoss outputShape targetShape)
    {α : Type} [_root_.Context α] [DecidableEq Shape]
    (state : State model α)
    (x : Tensor α inputShape) (target : Tensor α targetShape)
    (vState : State model α) :
    IO (State model α) :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.hvpParams
    (α := α)
    (paramShapes := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes model)
    (inputShapes := [Shape.ofList inputShape, Shape.ofList targetShape])
    (lossProgram model loss) state (.cons x (.cons target .nil)) vState
end model

end autograd

end TorchLean
