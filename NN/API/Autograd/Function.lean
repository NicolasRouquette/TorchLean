/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor
public import NN.API.Runtime
public import NN.Runtime.Autograd.TorchLean.Autodiff

/-!
# Function Transforms

Automatic differentiation transforms for pure one-argument tensor functions.
Import `NN.API.Autograd` for the complete public autograd API.
-/

@[expose] public section

namespace TorchLean

namespace autograd

namespace func

/-
Pure-function autograd: treat a pure function `f : Tensor σ -> Tensor τ` as the object of
differentiation (no parameters).
-/

/-!
In PyTorch terms, this is the "functorch" style: differentiate plain functions, not modules.
-/

/-- A scalar-polymorphic tensor function written against TorchLean's differentiable operations. -/
abbrev TensorFunction (inputShape outputShape : List Nat) :=
  ∀ {α : Type}, [Context α] → [DecidableEq Shape] →
    {m : Type → Type} → [Monad m] →
      [_root_.Runtime.Autograd.Torch.Ops (m := m) (α := α)] →
      _root_.TorchLean.Runtime.ValueRef (m := m) (α := α) inputShape →
      m (_root_.TorchLean.Runtime.ValueRef (m := m) (α := α) outputShape)

/-- Adapt a tensor function to the single-input program representation used by autograd. -/
def forwardProgram {inputShape outputShape : List Nat}
    (f : TensorFunction inputShape outputShape) :
    ∀ {α : Type}, [Context α] → [DecidableEq Shape] →
      _root_.Runtime.Autograd.TorchLean.Program α [Shape.ofList inputShape] outputShape :=
  fun {α} _ _ => fun {m} _ _ =>
    _root_.Runtime.Autograd.Torch.CurriedRef.curry
      (Ref := fun s => _root_.TorchLean.Runtime.ValueRef (m := m) (α := α) s)
      (ss := [Shape.ofList inputShape])
      (β := m (_root_.TorchLean.Runtime.ValueRef
        (m := m) (α := α) (Shape.ofList outputShape)))
      (fun args =>
        match args with
        | .cons input .nil => f (α := α) (m := m) input)

/-- Forward-mode Jacobian (`jacfwd`) for a pure tensor function. -/
def jacfwd {inputShape outputShape : List Nat} (f : TensorFunction inputShape outputShape)
    {α : Type} [_root_.Context α] [DecidableEq Shape]
    (x : Tensor α inputShape) :
    IO (Array (Tensor α outputShape)) :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.jacfwdInput
    (α := α) (σ := Shape.ofList inputShape) (τ := Shape.ofList outputShape)
    (forwardProgram f) x

/-- Hessian for a scalar-valued function. -/
def hessian {inputShape : List Nat} (f : TensorFunction inputShape [])
    {α : Type} [_root_.Context α] [DecidableEq Shape]
    (x : Tensor α inputShape) :
    IO (Array (Tensor α inputShape)) :=
  _root_.Runtime.Autograd.TorchLean.Autodiff.hessianInput
    (α := α) (σ := Shape.ofList inputShape) (forwardProgram f) x

/-- Vector-Jacobian product (VJP) for a pure function. -/
def vjp {inputShape outputShape : List Nat} (f : TensorFunction inputShape outputShape)
    {α : Type} [_root_.Context α] [DecidableEq Shape]
    (x : Tensor α inputShape) (seedOut : Tensor α outputShape) :
    IO (Tensor α inputShape) := do
  let state : _root_.TorchLean.TensorPack α ([] : List Shape) := .nil
  let gxs ←
    _root_.Runtime.Autograd.TorchLean.Autodiff.vjpOutInputs (α := α)
      (paramShapes := ([] : List Shape)) (inputShapes := [Shape.ofList inputShape])
      (τ := Shape.ofList outputShape)
      (forwardProgram (inputShape := inputShape) (outputShape := outputShape) f)
      state (.cons x .nil) seedOut
  pure (_root_.TorchLean.TensorPack.get gxs ⟨0, by simp⟩)

/--
Reverse-mode Jacobian (`jacrev`) of a pure tensor function.

Returns the Jacobian rows as an array of `doutput/dinput` tensors.
-/
def jacrev {inputShape outputShape : List Nat} (f : TensorFunction inputShape outputShape)
    {α : Type} [_root_.Context α] [DecidableEq Shape]
    (x : Tensor α inputShape) :
    IO (Array (Tensor α inputShape)) := do
  let state : _root_.TorchLean.TensorPack α ([] : List Shape) := .nil
  let rows ←
    _root_.Runtime.Autograd.TorchLean.Autodiff.jacrevOutInputs (α := α)
      (paramShapes := ([] : List Shape)) (inputShapes := [Shape.ofList inputShape])
      (τ := Shape.ofList outputShape)
      (forwardProgram (inputShape := inputShape) (outputShape := outputShape) f)
      state (.cons x .nil)
  pure <| rows.map fun xs => _root_.TorchLean.TensorPack.get xs ⟨0, by simp⟩

/-- Gradient of a scalar-valued function w.r.t. its input. -/
def grad {inputShape : List Nat} (f : TensorFunction inputShape [])
    {α : Type} [_root_.Context α] [DecidableEq Shape]
    (x : Tensor α inputShape) :
    IO (Tensor α inputShape) := do
  let state : _root_.TorchLean.TensorPack α ([] : List Shape) := .nil
  let gxs ←
    _root_.Runtime.Autograd.TorchLean.Autodiff.gradInputs (α := α)
      (paramShapes := ([] : List Shape)) (inputShapes := [Shape.ofList inputShape])
      (forwardProgram (inputShape := inputShape) (outputShape := []) f)
      state (.cons x .nil)
  pure (_root_.TorchLean.TensorPack.get gxs ⟨0, by simp⟩)

/-- Return `(value, grad)` for a scalar-valued function at `x`. -/
def valueAndGrad {inputShape : List Nat} (f : TensorFunction inputShape [])
    {α : Type} [_root_.Context α] [DecidableEq Shape]
    (x : Tensor α inputShape) :
    IO (Tensor α ([] : List Nat) × Tensor α inputShape) := do
  let c ←
    _root_.Runtime.Autograd.TorchLean.Autodiff.lowerScalarToTypedGraph (α := α)
      (paramShapes := ([] : List Shape)) (inputShapes := [Shape.ofList inputShape])
      (forwardProgram (inputShape := inputShape) (outputShape := []) f)
  let args : _root_.TorchLean.TensorPack α [Shape.ofList inputShape] := .cons x .nil
  let value : Tensor α ([] : List Nat) :=
    _root_.Runtime.Autograd.Torch.TypedScalarGraph.forward (α := α)
      (Γ := [Shape.ofList inputShape]) c args
  let gAll : _root_.TorchLean.TensorPack α [Shape.ofList inputShape] :=
    _root_.Runtime.Autograd.Torch.TypedScalarGraph.backward (α := α)
      (Γ := [Shape.ofList inputShape]) c args
  pure (value, _root_.TorchLean.TensorPack.get gAll ⟨0, by simp⟩)

/-- `valueAndGrad`, but convert the 0-dim value tensor to a scalar `α`. -/
def valueAndGradScalar {inputShape : List Nat} (f : TensorFunction inputShape [])
    {α : Type} [_root_.Context α] [DecidableEq Shape]
    (x : Tensor α inputShape) :
    IO (α × Tensor α inputShape) := do
  let (valueT, g) ← valueAndGrad (f := f) (α := α) x
  pure (Spec.Tensor.item valueT, g)
end func

end autograd

end TorchLean
