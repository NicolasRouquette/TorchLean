/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.Module.RuntimeInit

import Mathlib.Algebra.Order.Algebra

/-!
# Scalar Objectives

Executable scalar-objective definitions and runtime state. This module provides loss and gradient
evaluation, optimizer updates, state access, and optimizers bound to a live module.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor
open Proofs.Autograd.Algebra

namespace Module

/--
An immutable scalar-objective definition:
- `initState` stores initial trainable parameters and persistent buffers as `Float` tensors,
- `loss` is *polymorphic in the scalar backend* (same code works for Float/IEEE32Exec/…).

You can instantiate this definition as an `Objective` under a chosen execution mode and scalar.
-/
structure ObjectiveDef (β : Type) (stateShapes inputShapes : List Shape)
    (dataInputShapes : List Shape := []) where
  /-- Initial parameter-and-buffer state, cast from `Float` at instantiation time. -/
  initState : TorchLean.TensorPack Float stateShapes
  /--
  Optional storage-first initialization plan for executable `Float` runs.

  The ordinary tensors remain the semantic initial values. This plan records how a runtime may
  materialize the same initialization directly in backend storage without traversing a large
  nested Lean tensor first.
  -/
  runtimeInit : Option (RuntimeInit.Plan stateShapes) := none
  /-- Differentiability flags aligned with `stateShapes`; persistent buffers carry `false`. -/
  requiresGrad : Array Bool := Array.replicate stateShapes.length true
  /-- Validate non-differentiable inputs before they reach the runtime program. -/
  validateDataInputs : TorchLean.TensorPack β dataInputShapes → Except String Unit :=
    fun _ => pure ()
  /--
  Scalar loss over differentiable tensors followed by non-differentiable data tensors.

  The second curried input pack can carry labels, bounded token IDs, or gather indices without
  converting them through the model's floating-point scalar type.
  -/
  loss :
    ∀ {α : Type}, [Context α] → [DecidableEq Shape] →
      ∀ {m : Type → Type}, [Monad m] → [Torch.Ops (m := m) (α := α)] →
        Torch.CurriedRef (fun s => Torch.Ops.Ref (m := m) (α := α) s)
          (stateShapes ++ inputShapes)
          (Torch.CurriedRef (fun s => Torch.Ops.DataRef (m := m) (α := α) β s)
            dataInputShapes (m (Torch.Ops.Ref (m := m) (α := α) Shape.scalar)))

/--
Runtime state for a model together with a scalar objective.

This is lower level than PyTorch's loss classes: it owns model state as well as the objective. It
wraps `Torch.ScalarTrainer` and exposes objective evaluation, explicit gradients, and updates.
-/
structure Objective (α β : Type) [Context α] [DecidableEq Shape]
    (stateShapes inputShapes : List Shape) (dataInputShapes : List Shape := []) where
  /-- Trainer that owns trainable parameters and persistent buffers. -/
  trainer : Torch.ScalarTrainer α β stateShapes inputShapes dataInputShapes
  /-- Runtime options used to instantiate the module. -/
  opts : Torch.Options
  /-- Concrete host/device tensor conversion selected when the module was instantiated. -/
  tensorTransfer : _root_.Runtime.Autograd.Torch.TensorTransfer α


namespace Objective

/--
Create a runtime objective from an explicit scalar program and initial model state.

This is the low-level constructor; public training code starts from an `ObjectiveDef` and calls
`ObjectiveDef.instantiate`.
-/
def create {α β : Type} [Context α] [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.TensorTransfer α]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (opts : Torch.Options := {})
    (requiresGrad : Array Bool := Array.replicate stateShapes.length true)
    (validateDataInputs : TorchLean.TensorPack β dataInputShapes → Except String Unit :=
      fun _ => pure ())
    (loss :
      ∀ {m : Type → Type}, [Monad m] → [Torch.Ops (m := m) (α := α)] →
        Torch.CurriedRef (fun s => Torch.Ops.Ref (m := m) (α := α) s) (stateShapes ++ inputShapes)
          (Torch.CurriedRef (fun s => Torch.Ops.DataRef (m := m) (α := α) β s)
            dataInputShapes (m (Torch.Ops.Ref (m := m) (α := α) Shape.scalar))))
    (initState : TorchLean.TensorPack α stateShapes) :
    IO (Objective α β stateShapes inputShapes dataInputShapes) := do
  let mkTr :=
    Torch.scalarTrainer (α := α) (paramShapes := stateShapes) (inputShapes := inputShapes)
      (dataInputShapes := dataInputShapes)
      (opts := opts) (initRequiresGrad := requiresGrad)
      (validateDataInputs := validateDataInputs) (loss := loss)
  let tr ← Torch.Curried.uncurry (α := α) (ss := stateShapes)
    (β := IO (Torch.ScalarTrainer α β stateShapes inputShapes dataInputShapes)) mkTr initState
  pure { trainer := tr, opts := opts, tensorTransfer := inferInstance }

/-- Evaluate the scalar objective. -/
def loss {α β : Type} [Context α] [DecidableEq Shape]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (m : Objective α β stateShapes inputShapes dataInputShapes)
    (xs : TorchLean.TensorPack α inputShapes)
    (dataInputs : TorchLean.TensorPack β dataInputShapes) :
    IO (Tensor α .scalar) :=
  Torch.ScalarTrainer.runLoss (α := α) (paramShapes := stateShapes) (inputShapes := inputShapes)
    m.trainer xs dataInputs

/-- Return state-shaped gradients, using zero for entries that do not require gradients. -/
def gradState {α β : Type} [Context α] [DecidableEq Shape]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (m : Objective α β stateShapes inputShapes dataInputShapes)
    (xs : TorchLean.TensorPack α inputShapes)
    (dataInputs : TorchLean.TensorPack β dataInputShapes) :
    IO (TorchLean.TensorPack α stateShapes) :=
  Torch.ScalarTrainer.runGradState (α := α) (paramShapes := stateShapes)
    (inputShapes := inputShapes) m.trainer xs dataInputs

/-- Return a scalar loss and its state-shaped gradients from one forward tape. -/
def lossAndGradState {α β : Type} [Context α] [DecidableEq Shape]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (m : Objective α β stateShapes inputShapes dataInputShapes)
    (xs : TorchLean.TensorPack α inputShapes)
    (dataInputs : TorchLean.TensorPack β dataInputShapes) :
    IO (Tensor α .scalar × TorchLean.TensorPack α stateShapes) :=
  Torch.ScalarTrainer.runLossAndGradState
    (α := α) (paramShapes := stateShapes) (inputShapes := inputShapes) m.trainer xs dataInputs

/-- Compute gradients and apply one SGD update with learning rate `lr`. -/
def sgdStep {α β : Type} [Context α] [DecidableEq Shape]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (m : Objective α β stateShapes inputShapes dataInputShapes)
    (lr : α) (xs : TorchLean.TensorPack α inputShapes)
    (dataInputs : TorchLean.TensorPack β dataInputShapes) :
    IO Unit :=
  Torch.ScalarTrainer.runStep (α := α) (paramShapes := stateShapes) (inputShapes := inputShapes)
    m.trainer lr xs dataInputs

/-- Apply one SGD update and return the loss from the tape that produced its gradients. -/
def sgdStepWithLoss {α β : Type} [Context α] [DecidableEq Shape]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (m : Objective α β stateShapes inputShapes dataInputShapes)
    (lr : α) (xs : TorchLean.TensorPack α inputShapes)
    (dataInputs : TorchLean.TensorPack β dataInputShapes) :
    IO (Tensor α .scalar) :=
  Torch.ScalarTrainer.runStepWithLoss
    (α := α) (paramShapes := stateShapes) (inputShapes := inputShapes) m.trainer lr xs dataInputs

/-- Initialize optimizer state from this module's current state. -/
def initOptimizer {α β : Type} [Context α] [DecidableEq Shape]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (m : Objective α β stateShapes inputShapes dataInputShapes)
    (opt : TorchLean.Optim.Optimizer α stateShapes) :
    IO opt.State := do
  -- Generic optimizer states are initialized from host tensors. Synchronize device-backed
  -- parameters first so a CUDA module cannot seed those states from stale host mirrors.
  let _ ← m.trainer.getState
  opt.init m.trainer.state

/--
Run one optimizer step using an explicit optimizer + state.

This mirrors a PyTorch training step:
1. compute the explicit state gradient (`ScalarTrainer.runGradState`)
2. update parameters via `opt.step` and return the new optimizer state
-/
def optimizerStep {α β : Type} [Context α] [DecidableEq Shape]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (m : Objective α β stateShapes inputShapes dataInputShapes)
    (opt : TorchLean.Optim.Optimizer α stateShapes) (st : opt.State)
    (xs : TorchLean.TensorPack α inputShapes)
    (dataInputs : TorchLean.TensorPack β dataInputShapes) :
    IO opt.State := do
  match ← opt.trainerStep? m.trainer st xs dataInputs with
  | some st' =>
      pure st'
  | none =>
      let grads ← Torch.ScalarTrainer.runGradState (α := α)
        (paramShapes := stateShapes) (inputShapes := inputShapes) m.trainer xs dataInputs
      let _ ← m.trainer.getState
      opt.step st m.trainer.state grads

/--
Run an explicit optimizer step and return both the next optimizer state and the loss used for it.

Native trainer hooks may keep gradients and optimizer state on the selected device. The fallback
path computes the loss and gradients once, synchronizes parameter mirrors when needed, and applies
the optimizer to those gradients.
-/
def optimizerStepWithLoss {α β : Type} [Context α] [DecidableEq Shape]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (m : Objective α β stateShapes inputShapes dataInputShapes)
    (opt : TorchLean.Optim.Optimizer α stateShapes) (st : opt.State)
    (xs : TorchLean.TensorPack α inputShapes)
    (dataInputs : TorchLean.TensorPack β dataInputShapes) :
    IO (opt.State × Tensor α .scalar) := do
  match ← opt.trainerStepWithLoss? m.trainer st xs dataInputs with
  | some result =>
      pure result
  | none =>
      let (loss, grads) ← Torch.ScalarTrainer.runLossAndGradState (α := α)
        (paramShapes := stateShapes) (inputShapes := inputShapes) m.trainer xs dataInputs
      let _ ← m.trainer.getState
      let st' ← opt.step st m.trainer.state grads
      pure (st', loss)

/-- Read the complete parameter-and-buffer state as a shape-indexed list. -/
def state {α β : Type} [Context α] [DecidableEq Shape]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (m : Objective α β stateShapes inputShapes dataInputShapes) :
    IO (TorchLean.TensorPack α stateShapes) :=
  m.trainer.getState

/-- Replace the complete parameter-and-buffer state. -/
def loadState {α β : Type} [Context α] [DecidableEq Shape]
    {stateShapes inputShapes dataInputShapes : List Shape}
    (m : Objective α β stateShapes inputShapes dataInputShapes)
    (ps : TorchLean.TensorPack α stateShapes) : IO Unit :=
  Torch.ParamList.setValues (α := α) (ss := stateShapes) m.trainer.state ps

/-- Train with vanilla SGD for a fixed number of steps on a fixed array of samples. -/
def trainSGD {α : Type} [Context α] [DecidableEq Shape] [ToString α]
    {stateShapes inputShapes : List Shape}
    (m : Objective α Unit stateShapes inputShapes)
    (lr : α) (steps : Nat) (samples : Array (TorchLean.TensorPack α inputShapes))
    (logEvery : Nat := 1) : IO Unit :=
  match samples[0]? with
  | none => throw <| IO.userError "trainSGD: empty dataset"
  | some first =>
      for step in [0:steps] do
        let inputs := samples.getD (step % samples.size) first
        let loss ← m.loss inputs .nil
        if logEvery != 0 && step % logEvery = 0 then
          IO.println s!"step {step}: loss={loss.item}"
        m.sgdStep lr inputs .nil

/-- Like `trainSGD`, but with an explicit optimizer and mutable optimizer state. -/
def trainWithOptimizer {α : Type} [Context α] [DecidableEq Shape] [ToString α]
    {stateShapes inputShapes : List Shape}
    (m : Objective α Unit stateShapes inputShapes)
    (opt : TorchLean.Optim.Optimizer α stateShapes) (st0 : opt.State)
    (steps : Nat) (samples : Array (TorchLean.TensorPack α inputShapes))
    (logEvery : Nat := 1) : IO opt.State := do
  let first ← match samples[0]? with
    | some sample => pure sample
    | none => throw <| IO.userError "trainWithOptimizer: empty dataset"
  let mut optimizerState := st0
  for step in [0:steps] do
    let inputs := samples.getD (step % samples.size) first
    let (nextState, loss) ← m.optimizerStepWithLoss opt optimizerState inputs .nil
    optimizerState := nextState
    if logEvery != 0 && step % logEvery = 0 then
      IO.println s!"step {step}: loss={loss.item}"
  pure optimizerState

/-- Compute the mean loss over an array of samples (no parameter updates). -/
def meanLoss {α : Type} [Context α] [DecidableEq Shape] [ToString α]
    {stateShapes inputShapes : List Shape}
    (m : Objective α Unit stateShapes inputShapes)
    (samples : Array (TorchLean.TensorPack α inputShapes)) : IO α := do
  if samples.isEmpty then
    throw <| IO.userError "meanLoss: empty dataset"
  let mut total : α := 0
  for inputs in samples do
    let loss ← m.loss inputs .nil
    total := total + loss.item
  pure (total / (samples.size : α))

end Objective

/-- Mutable optimizer state bound to one executable module. -/
structure BoundOptimizer (α : Type) [Context α] [DecidableEq Shape]
    (stateShapes inputShapes : List Shape) (State : Type) where
  module : Objective α Unit stateShapes inputShapes
  state : IO.Ref State
  step : TorchLean.TensorPack α inputShapes → IO Unit

/-- Initialize an optimizer and bind its state and update operation to `module`. -/
def bindOptimizer {α : Type} [Context α] [DecidableEq Shape]
    {stateShapes inputShapes : List Shape}
    (module : Objective α Unit stateShapes inputShapes)
    (optimizer : Optim.Optimizer α stateShapes) :
    IO (BoundOptimizer α stateShapes inputShapes optimizer.State) := do
  let initialState ← Objective.initOptimizer module optimizer
  let state ← IO.mkRef initialState
  let step (sample : TorchLean.TensorPack α inputShapes) : IO Unit := do
    let currentState ← state.get
    let nextState ← Objective.optimizerStep module optimizer currentState sample .nil
    state.set nextState
  pure { module, state, step }

end Module

end TorchLean
end Autograd
end Runtime
