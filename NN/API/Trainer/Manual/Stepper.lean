/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Trainer.Manual.Execution

@[expose] public section

namespace TorchLean
namespace Trainer
namespace Manual

open SeqTask

/-!
# Runtime Supervised Training

Stateful training steps built from the lower-level task and runner API.
-/

/-! ## Stateful training steps -/

/--
Stateful training loop object: a `Runner` plus an optimizer state and a step counter. It packages
the model runner with the state needed to step on successive batches.
-/
structure Stepper (α : Type) [_root_.Context α] [DecidableEq Spec.Shape]
    {σ τ : Spec.Shape} (task : SeqTask σ τ) where
  /-- Underlying task runner (module + lowered typed graphs for forward evaluation and losses). -/
  runner : Runner α task
  /-- Run a single optimization step on one supervised sample, returning the loss value. -/
  stepSample : _root_.TorchLean.TensorPack α [σ, τ] → IO α
  /-- Run an epoch over an explicit sample array, returning the per-step loss values. -/
  epochSamples : Array (_root_.TorchLean.TensorPack α [σ, τ]) → IO (Array α)
  /-- Read the total number of `stepSample` calls performed so far. -/
  stepCount : IO Nat

/--
Construct a `Stepper` for a runner, optimizer config, and optional scheduler.

This is the recommended way to build custom training loops without reimplementing the optimizer
logic: call `stepper`, then choose `stepSample` for single batches or `epochSamples` for explicit
sample lists.
-/
def stepper {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α] [_root_.TorchLean.Runtime.FromFloat α]
    (runner : Runner α task) (optimizer : _root_.TorchLean.optim.Optimizer)
    (scheduler : Option _root_.TorchLean.Trainer.Scheduler.Config := none) :
    IO (Stepper α task) := do
  Runner.train runner
  let stepRef ← IO.mkRef 0
  Internal.withBoundOptimizer runner optimizer scheduler fun opt initialState scheduleState => do
    let stateRef ← IO.mkRef initialState
    let runStep := fun (sample : _root_.TorchLean.TensorPack α [σ, τ]) => do
      Runner.train runner
      let stepIdx ← stepRef.get
      Runner.updateBuffers runner sample
      let state := scheduleState stepIdx (← stateRef.get)
      let (state', loss) ←
        TorchLean.Module.optimizerStepWithLoss runner.module opt state sample .nil
      stateRef.set state'
      stepRef.set (stepIdx + 1)
      pure (Spec.Tensor.item loss)
    pure {
      runner := runner
      stepSample := runStep
      epochSamples := fun samples => samples.mapM runStep
      stepCount := stepRef.get
    }

end Manual
end Trainer
end TorchLean
