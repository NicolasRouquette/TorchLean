/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.NN.Core

/-!
# TorchLean NN: Sequential Models
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor
open Proofs.Autograd.Algebra

namespace NN

/-! ## Sequential models -/

/--
Sequential composition of `Layer`s, indexed by input/output shape.

This is the builder-layer analogue of `torch.nn.Sequential`: a `Seq σ τ` represents a model that
takes an input of shape `σ` and produces an output of shape `τ` by running layers left-to-right.
-/
inductive Seq : Shape → Shape → Type 2 where
  /-- The empty sequence, which leaves a tensor unchanged. -/
  | id (s : Shape) : Seq s s
  /-- Run one layer, then the remaining sequence. -/
  | cons {σ τ υ : Shape} : Layer σ τ → Seq τ υ → Seq σ υ

namespace Seq

/--
Collect the parameter and persistent-buffer shapes owned by a sequential model.

This concatenates each layer's `stateShapes` in order.
-/
def stateShapes : {σ τ : Shape} → Seq σ τ → List Shape
  | _, _, .id _ => []
  | _, _, .cons l rest => l.stateShapes ++ stateShapes rest

/--
Collect the gradient flags for all parameters and buffers in a sequential model.

This concatenates each layer's `requiresGrad` in order. Persistent buffers carry `false`.
-/
def requiresGrad : {σ τ : Shape} → Seq σ τ → Array Bool
  | _, _, .id _ => #[]
  | _, _, .cons l rest => l.requiresGrad ++ requiresGrad rest

/--
Initial parameter and persistent-buffer values for a sequential model.

This concatenates each layer's `initState` into the flat state list expected by `forward` and
the supervised module constructors.
-/
def initState : {σ τ : Shape} → (m : Seq σ τ) → TorchLean.TensorPack Float (stateShapes m)
  | _, _, .id _ => .nil
  | _, _, .cons l rest =>
      let xs := l.initState
      let ys := initState rest
      TorchLean.TensorPack.append (α := Float)
        (ss₁ := l.stateShapes) (ss₂ := stateShapes rest) xs ys

/--
Collect a storage-first initializer plan when every parameterized layer supplies one.

Parameter-free layers need no annotation and contribute the empty plan. If any parameterized layer
has only tensor-valued initializers, the whole model falls back to the ordinary initialization path.
-/
def runtimeInit? : {σ τ : Shape} → (m : Seq σ τ) →
    Option (TorchLean.Module.RuntimeInit.Plan (stateShapes m))
  | _, _, .id _ => some .nil
  | _, _, .cons l rest =>
      match l.runtimeInit, runtimeInit? rest with
      | some xs, some ys => some (TorchLean.Module.RuntimeInit.Plan.append xs ys)
      | _, _ => none

/-- Whether any layer in the sequence owns mode-dependent mutable buffers. -/
def hasBufferUpdates : {σ τ : Shape} → Seq σ τ → Bool
  | _, _, .id _ => false
  | _, _, .cons l rest => l.updateBuffers.isSome || hasBufferUpdates rest

/--
Sequential composition for `Seq` models.

`comp f g` runs `f` then `g`. We also provide the infix `>>>` operator.
-/
def comp {σ τ υ : Shape} : Seq σ τ → Seq τ υ → Seq σ υ
  | .id _, g => g
  | .cons l rest, g => .cons l (comp rest g)

infixr:80 " >>> " => comp

/--
Internal evaluator that splits the flat model state as it walks the model.

This is the reference-level forward pass used to implement `forward`.
-/
def forwardState {σ τ : Shape} (model : Seq σ τ) {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Torch.Ops (m := m) (α := α)]
    (mode : Mode)
    (ps : Torch.RefList (RefTy (m := m) (α := α)) (stateShapes model))
    (x : RefTy (m := m) (α := α) σ) : m (RefTy (m := m) (α := α) τ) :=
  match model with
  | .id _ => pure x
  | .cons l rest =>
      let (psL, psR) :=
        Torch.RefList.split (Ref := RefTy (m := m) (α := α))
          (ss₁ := l.stateShapes) (ss₂ := stateShapes rest) ps
      do
        let y ← l.forwardRef (α := α) (m := m) mode psL x
        forwardState (model := rest) (α := α) (m := m) mode psR y

/--
The differentiable forward computation of a sequential model.

The result is operation-polymorphic: eager execution records an autograd tape, while typed-graph
execution records shape-indexed SSA data. `mode` controls layers such as dropout and batch
normalization; it does not enable or disable gradient tracking.
-/
def forward {σ τ : Shape} (model : Seq σ τ) (mode : Mode := .eval)
    {α : Type} [Context α] [DecidableEq Shape] :
    TorchLean.Program α (stateShapes model ++ [σ]) τ :=
  fun {m} _ _ =>
    Torch.CurriedRef.curry (Ref := RefTy (m := m) (α := α))
      (ss := stateShapes model ++ [σ]) (β := m (RefTy (m := m) (α := α) τ)) (fun args => do
        let (ps, x) := Torch.RefList.splitLast (Ref := RefTy (m := m) (α := α)) (ss := stateShapes
          model) (τ := σ) args
        forwardState (model := model) (α := α) (m := m) mode ps x)

  /-!
  ## Forward and inference helpers

  The naming mirrors the PyTorch split:

  - `Mode.eval` / `Mode.train` choose layer behavior in `forward`,
  - `forwardNoGrad` executes a concrete forward pass without recording gradients,
  - `predict` is eval-mode eager inference from live parameters,
  - `lowerToTypedGraph` records a reusable typed graph.

  In particular, `predict` is not the typed graph runner. Lowering captures the layer mode in the
  graph. Evaluate the result with `TypedGraph.forward`.
  -/

  /-!
  These helpers run a `Seq` directly through the eager runtime, given a *live* `ParamList`.

  Why this exists: several runnable examples want to inspect logits (argmax decoding, probes,
  interactive loops) without re-implementing the `useParams/useInputs` boilerplate.

  Note: this is eager-only. To record once and run many times, use
  `lowerToTypedGraph`, then evaluate the returned graph with `TypedGraph.forward`.
  -/

  /--
  Run an eager forward pass for one concrete input under an explicit mode.

  This uses the eager runtime so CUDA kernels stay available, reads back the concrete output, and
  then releases ephemeral CUDA tape buffers because no backward pass will follow. Use this for
  validation, decoding, diffusion sampling, and other inference loops.
  -/
  def forwardNoGrad {σ τ : Shape}
      (opts : _root_.Runtime.Autograd.Torch.Options)
      (model : Seq σ τ)
      {α : Type} [Context α] [DecidableEq Shape]
      [tensorTransfer : _root_.Runtime.Autograd.Torch.TensorTransfer α]
      (params : _root_.Runtime.Autograd.Torch.ParamList α (stateShapes model))
      (x : Spec.Tensor α σ) (mode : Mode := .eval) : IO (Spec.Tensor α τ) := do
    -- Inference still uses the eager session machinery so it can select native kernels, but its
    -- leaves are deliberately non-differentiable and the transient tape is released before return.
    let opts := { opts with gradEnabled := false }
    let sess ← _root_.Runtime.Autograd.Torch.Internal.EagerSession.new (α := α) opts
    sess.resetTape
    let outRef ← (do
      let pRefs ← _root_.Runtime.Autograd.Torch.Internal.useParams (α := α)
        (ss := stateShapes model) params
      let xRefs ← _root_.Runtime.Autograd.Torch.Internal.useInputs (α := α)
        (ss := [σ]) (.cons x .nil)
      let allRefs := _root_.Runtime.Autograd.Torch.RefList.append
        (ss₁ := stateShapes model) (ss₂ := [σ]) pRefs xRefs
      _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
        (ss := stateShapes model ++ [σ])
        (forward model (mode := mode) (α := α)) allRefs) |>.run sess
    let y ← _root_.Runtime.Autograd.Torch.Internal.EagerSession.getValue (α := α) sess outRef
    if opts.usesCuda then
      _root_.Runtime.Autograd.Torch.Internal.EagerSession.releaseCudaTapeNonParamValues sess
      sess.cudaTape.set _root_.Runtime.Autograd.Cuda.Tape.empty
      sess.paramsByLeaf.set (Std.HashMap.emptyWithCapacity)
      sess.nats.set #[]
      _root_.Runtime.Autograd.Torch.Internal.EagerSession.collectCudaAllocator
    else
      pure ()
    pure y

  /--
  Run eval-mode eager inference for one concrete input.

  This is the eval-mode convenience wrapper around `forwardNoGrad`.
  -/
  def predict {σ τ : Shape}
      (opts : _root_.Runtime.Autograd.Torch.Options)
      (model : Seq σ τ)
      {α : Type} [Context α] [DecidableEq Shape]
      [tensorTransfer : _root_.Runtime.Autograd.Torch.TensorTransfer α]
      (params : _root_.Runtime.Autograd.Torch.ParamList α (stateShapes model))
      (x : Spec.Tensor α σ) : IO (Spec.Tensor α τ) :=
    forwardNoGrad (α := α) (tensorTransfer := tensorTransfer) opts model params x

  /--
  Lower a sequential model into a reusable `TypedGraph`.

  The model is recorded once as a typed SSA graph and can then be evaluated repeatedly.
  -/
  def lowerToTypedGraph {σ τ : Shape}
      (model : Seq σ τ)
      (mode : Mode := .eval)
      {α : Type} [Context α] [DecidableEq Shape] :
      IO (_root_.Runtime.Autograd.Torch.TypedGraph α (stateShapes model ++ [σ]) τ) :=
    _root_.Runtime.Autograd.TorchLean.Autodiff.lowerToTypedGraph (α := α)
      (paramShapes := stateShapes model) (inputShapes := [σ]) (τ := τ)
      (fun {β} _ _ => forward model (mode := mode) (α := β))

  /--
  Update per-layer buffers across a sequential model.

This walks the model left-to-right and, for each layer that defines `Layer.updateBuffers`,
updates that layer’s parameter/buffer slice using the current activation. This is used to implement
BatchNorm-style running statistics (and similar stateful layers) in a pure, explicit way.

PyTorch analogy: updating `running_mean` / `running_var` buffers during a forward pass in train
  mode.
-/
def updateBuffers {σ τ : Shape} (mode : Mode) (model : Seq σ τ)
    {α : Type} [Context α] [DecidableEq Shape]
    (ps : TorchLean.TensorPack α (stateShapes model)) (x : Tensor α σ) :
    IO (TorchLean.TensorPack α (stateShapes model)) :=
  match model with
  | .id _ => pure .nil
  | .cons l rest => do
      let (psL, psR) :=
        TorchLean.TensorPack.split
          (α := α) (ss₁ := l.stateShapes) (ss₂ := stateShapes rest) ps
      let psL' ←
        match l.updateBuffers with
        | some f => f mode psL x
        | none => pure psL
      let y ← Layer.forwardTensor l mode psL' x
      let psR' ← updateBuffers mode rest psR y
      pure <| TorchLean.TensorPack.append
        (α := α) (ss₁ := l.stateShapes) (ss₂ := stateShapes rest) psL' psR'

/-! ## Scalar objectives -/

namespace Objective

/--
Pair an immutable sequential model with a scalar loss in an explicit layer mode.

The resulting definition initializes the model's complete parameter-and-buffer state and computes
the loss from one `(input, target)` pair.
-/
def createWithMode {σ τ : Shape} (mode : Mode) (model : Seq σ τ)
    (loss : ∀ {α : Type}, [Context α] → [DecidableEq Shape] → TorchLean.Program α [τ, τ]
      Shape.scalar) :
    TorchLean.Module.ObjectiveDef Unit (stateShapes model) [σ, τ] :=
  { initState := initState model
    runtimeInit := runtimeInit? model
    requiresGrad := requiresGrad model
    loss := fun {α} => by
      intro _ _; exact
        (fun {m} _ _ =>
          Torch.CurriedRef.curry (Ref := RefTy (m := m) (α := α))
            (ss := stateShapes model ++ [σ, τ])
            (β := m (RefTy (m := m) (α := α) Shape.scalar)) (fun args => do
              let (ps, xy) :=
                Torch.RefList.split (Ref := RefTy (m := m) (α := α))
                  (ss₁ := stateShapes model) (ss₂ := [σ, τ]) args
              let .cons x (.cons y .nil) := xy
              let yhat ← forwardState (model := model) (α := α) (m := m) mode ps x
              Torch.CurriedRef.uncurry (Ref := RefTy (m := m) (α := α)) (ss := [τ, τ])
                (loss (α := α) (m := m)) (.cons yhat (.cons y .nil))
          ))
  }

/-- Pair a sequential model with a scalar loss in training mode. -/
def create {σ τ : Shape} (model : Seq σ τ)
    (loss : ∀ {α : Type}, [Context α] → [DecidableEq Shape] → TorchLean.Program α [τ, τ]
      Shape.scalar) :
    TorchLean.Module.ObjectiveDef Unit (stateShapes model) [σ, τ] :=
  createWithMode .train model loss

/-- Pair a model with mean-squared error in an explicit layer mode. -/
def mseWithMode {σ τ : Shape} (mode : Mode) (model : Seq σ τ)
    (reduction : TorchLean.Loss.Reduction := .mean) :
    TorchLean.Module.ObjectiveDef Unit (stateShapes model) [σ, τ] :=
  createWithMode mode (model := model) (loss := fun {α} _ _ =>
    fun {m} _ _ =>
      fun yhat y => TorchLean.Loss.mse (m := m) (α := α) (s := τ) yhat y (reduction := reduction))

/-- Pair a model with mean-squared error in training mode. -/
def mse {σ τ : Shape} (model : Seq σ τ) (reduction : TorchLean.Loss.Reduction :=
  .mean) :
    TorchLean.Module.ObjectiveDef Unit (stateShapes model) [σ, τ] :=
  mseWithMode .train model reduction

/-- Pair a model with one-hot cross entropy in an explicit layer mode. -/
def oneHotCrossEntropyWithMode {σ τ : Shape} (mode : Mode) (model : Seq σ τ)
    (axis : Nat) [Shape.AxisInBounds axis τ]
    (reduction : TorchLean.Loss.Reduction := .mean) :
    TorchLean.Module.ObjectiveDef Unit (stateShapes model) [σ, τ] :=
  createWithMode mode (model := model) (loss := fun {α} _ _ =>
    fun {m} _ _ =>
      fun logits targetOneHot =>
        TorchLean.Loss.oneHotCrossEntropy (m := m) (α := α) (s := τ) axis
          logits targetOneHot
          (reduction := reduction))

/-- Pair a model with one-hot cross entropy in training mode. -/
def oneHotCrossEntropy {σ τ : Shape} (model : Seq σ τ)
    (axis : Nat) [Shape.AxisInBounds axis τ]
    (reduction : TorchLean.Loss.Reduction := .mean) :
    TorchLean.Module.ObjectiveDef Unit (stateShapes model) [σ, τ] :=
  oneHotCrossEntropyWithMode .train model axis reduction

end Objective

end Seq
end NN

end TorchLean
end Autograd
end Runtime
