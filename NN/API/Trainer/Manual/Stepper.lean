/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Trainer.Manual.Loops

import Mathlib.Algebra.Order.Algebra
import NN.Spec.Autograd.AutogradSpec

@[expose] public section

namespace TorchLean
namespace Trainer
namespace Manual

/-!
# Runtime Supervised Training

Supervised tasks, runners, steppers, optimizer configs, trainer aliases, and the low-level session
exports that back executable examples.
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
  stepSample : _root_.Runtime.Autograd.Torch.TList α [σ, τ] → IO α
  /-- Run an epoch over an explicit list of samples, returning the per-step loss values. -/
  epochSamples : List (_root_.Runtime.Autograd.Torch.TList α [σ, τ]) → IO (List α)
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
    (runner : Runner α task) (optimizer : OptimizerConfig)
    (scheduler : Option _root_.TorchLean.Trainer.Scheduler.Config := none) :
    IO (Stepper α task) := do
  Runner.train runner
  let stepRef ← IO.mkRef 0
  match optimizer with
  | .sgd lr momentum =>
      if momentum == 0.0 then
        let runStep := fun (sample : _root_.Runtime.Autograd.Torch.TList α [σ, τ]) => do
          Runner.train runner
          let stepIdx ← stepRef.get
          Runner.updateBuffers runner sample
          let lrα := _root_.TorchLean.Runtime.ofFloat (stepLR scheduler optimizer stepIdx)
          let loss ← TorchLean.Module.sgdStepWithLoss runner.module lrα sample .nil
          stepRef.set (stepIdx + 1)
          pure (Spec.Tensor.toScalar loss)
        pure {
          runner := runner
          stepSample := runStep
          epochSamples := fun samples => samples.mapM runStep
          stepCount := stepRef.get
        }
      else
        let opt := _root_.Runtime.Autograd.TorchLean.Optim.momentumSGD
          (α := α) (paramShapes := stateShapes task)
          (lr := _root_.TorchLean.Runtime.ofFloat lr) (momentum := _root_.TorchLean.Runtime.ofFloat momentum)
        let st0 : _root_.Runtime.Autograd.TorchLean.Optim.StateList _root_.Optim.MomentumSGD.State α (stateShapes task)
          ←
          TorchLean.Module.initOptimizer runner.module opt
        let stRef ← IO.mkRef st0
        let runStep := fun (sample : _root_.Runtime.Autograd.Torch.TList α [σ, τ]) => do
          Runner.train runner
          let stepIdx ← stepRef.get
          Runner.updateBuffers runner sample
          let lrα := _root_.TorchLean.Runtime.ofFloat (stepLR scheduler optimizer stepIdx)
          let st0 ← stRef.get
          let st := momentumSGDStateWithLR (paramShapes := stateShapes task) lrα st0
          let (st', loss) ←
            TorchLean.Module.optimizerStepWithLoss runner.module opt st sample .nil
          stRef.set st'
          stepRef.set (stepIdx + 1)
          pure (Spec.Tensor.toScalar loss)
        pure {
          runner := runner
          stepSample := runStep
          epochSamples := fun samples => samples.mapM runStep
          stepCount := stepRef.get
        }
  | .adagrad lr epsilon =>
      let opt := _root_.Runtime.Autograd.TorchLean.Optim.adagrad
        (α := α) (paramShapes := stateShapes task)
        (lr := _root_.TorchLean.Runtime.ofFloat lr)
        (epsilon := _root_.TorchLean.Runtime.ofFloat epsilon)
      let st0 : _root_.Runtime.Autograd.TorchLean.Optim.StateList _root_.Optim.AdaGrad.State α (stateShapes task) ←
        TorchLean.Module.initOptimizer runner.module opt
      let stRef ← IO.mkRef st0
      let runStep := fun (sample : _root_.Runtime.Autograd.Torch.TList α [σ, τ]) => do
        Runner.train runner
        let stepIdx ← stepRef.get
        Runner.updateBuffers runner sample
        let lrα := _root_.TorchLean.Runtime.ofFloat (stepLR scheduler optimizer stepIdx)
        let st0 ← stRef.get
        let st := adagradStateWithLR (paramShapes := stateShapes task) lrα st0
        let (st', loss) ←
          TorchLean.Module.optimizerStepWithLoss runner.module opt st sample .nil
        stRef.set st'
        stepRef.set (stepIdx + 1)
        pure (Spec.Tensor.toScalar loss)
      pure {
        runner := runner
        stepSample := runStep
        epochSamples := fun samples => samples.mapM runStep
        stepCount := stepRef.get
      }
  | .rmsprop lr decay epsilon =>
      let opt := _root_.Runtime.Autograd.TorchLean.Optim.rmsprop
        (α := α) (paramShapes := stateShapes task)
        (lr := _root_.TorchLean.Runtime.ofFloat lr)
        (decay := _root_.TorchLean.Runtime.ofFloat decay)
        (epsilon := _root_.TorchLean.Runtime.ofFloat epsilon)
      let st0 : _root_.Runtime.Autograd.TorchLean.Optim.StateList _root_.Optim.RMSProp.State α (stateShapes task) ←
        TorchLean.Module.initOptimizer runner.module opt
      let stRef ← IO.mkRef st0
      let runStep := fun (sample : _root_.Runtime.Autograd.Torch.TList α [σ, τ]) => do
        Runner.train runner
        let stepIdx ← stepRef.get
        Runner.updateBuffers runner sample
        let lrα := _root_.TorchLean.Runtime.ofFloat (stepLR scheduler optimizer stepIdx)
        let st0 ← stRef.get
        let st := rmspropStateWithLR (paramShapes := stateShapes task) lrα st0
        let (st', loss) ←
          TorchLean.Module.optimizerStepWithLoss runner.module opt st sample .nil
        stRef.set st'
        stepRef.set (stepIdx + 1)
        pure (Spec.Tensor.toScalar loss)
      pure {
        runner := runner
        stepSample := runStep
        epochSamples := fun samples => samples.mapM runStep
        stepCount := stepRef.get
      }
  | .adam lr beta1 beta2 epsilon =>
      let opt := _root_.Runtime.Autograd.TorchLean.Optim.adam
        (α := α) (paramShapes := stateShapes task)
        (lr := _root_.TorchLean.Runtime.ofFloat lr)
        (beta1 := _root_.TorchLean.Runtime.ofFloat beta1)
        (beta2 := _root_.TorchLean.Runtime.ofFloat beta2)
        (epsilon := _root_.TorchLean.Runtime.ofFloat epsilon)
      let st0 : _root_.Runtime.Autograd.TorchLean.Optim.StateList _root_.Optim.Adam.State α (stateShapes task) ←
        TorchLean.Module.initOptimizer runner.module opt
      let stRef ← IO.mkRef st0
      let runStep := fun (sample : _root_.Runtime.Autograd.Torch.TList α [σ, τ]) => do
        Runner.train runner
        let stepIdx ← stepRef.get
        Runner.updateBuffers runner sample
        let lrα := _root_.TorchLean.Runtime.ofFloat (stepLR scheduler optimizer stepIdx)
        let st0 ← stRef.get
        let st := adamStateWithLR (paramShapes := stateShapes task) lrα st0
        let (st', loss) ←
          TorchLean.Module.optimizerStepWithLoss runner.module opt st sample .nil
        stRef.set st'
        stepRef.set (stepIdx + 1)
        pure (Spec.Tensor.toScalar loss)
      pure {
        runner := runner
        stepSample := runStep
        epochSamples := fun samples => samples.mapM runStep
        stepCount := stepRef.get
      }
  | .adamw lr weightDecay beta1 beta2 epsilon =>
      let opt := _root_.Runtime.Autograd.TorchLean.Optim.adamw
        (α := α) (paramShapes := stateShapes task)
        (lr := _root_.TorchLean.Runtime.ofFloat lr) (weightDecay := _root_.TorchLean.Runtime.ofFloat weightDecay)
        (beta1 := _root_.TorchLean.Runtime.ofFloat beta1)
        (beta2 := _root_.TorchLean.Runtime.ofFloat beta2)
        (epsilon := _root_.TorchLean.Runtime.ofFloat epsilon)
      let st0 : _root_.Runtime.Autograd.TorchLean.Optim.StateList _root_.Optim.AdamW.State α (stateShapes task) ←
        TorchLean.Module.initOptimizer runner.module opt
      let stRef ← IO.mkRef st0
      let runStep := fun (sample : _root_.Runtime.Autograd.Torch.TList α [σ, τ]) => do
        Runner.train runner
        let stepIdx ← stepRef.get
        Runner.updateBuffers runner sample
        let lrα := _root_.TorchLean.Runtime.ofFloat (stepLR scheduler optimizer stepIdx)
        let st0 ← stRef.get
        let st := adamwStateWithLR (paramShapes := stateShapes task) lrα st0
        let (st', loss) ←
          TorchLean.Module.optimizerStepWithLoss runner.module opt st sample .nil
        stRef.set st'
        stepRef.set (stepIdx + 1)
        pure (Spec.Tensor.toScalar loss)
      pure {
        runner := runner
        stepSample := runStep
        epochSamples := fun samples => samples.mapM runStep
        stepCount := stepRef.get
      }
  | .adadelta lr rho epsilon =>
      let opt := _root_.Runtime.Autograd.TorchLean.Optim.adadelta
        (α := α) (paramShapes := stateShapes task)
        (lr := _root_.TorchLean.Runtime.ofFloat lr)
        (rho := _root_.TorchLean.Runtime.ofFloat rho)
        (epsilon := _root_.TorchLean.Runtime.ofFloat epsilon)
      let st0 : _root_.Runtime.Autograd.TorchLean.Optim.StateList _root_.Optim.Adadelta.State α (stateShapes task) ←
        TorchLean.Module.initOptimizer runner.module opt
      let stRef ← IO.mkRef st0
      let runStep := fun (sample : _root_.Runtime.Autograd.Torch.TList α [σ, τ]) => do
        Runner.train runner
        let stepIdx ← stepRef.get
        Runner.updateBuffers runner sample
        let lrα := _root_.TorchLean.Runtime.ofFloat (stepLR scheduler optimizer stepIdx)
        let st0 ← stRef.get
        let st := adadeltaStateWithLR (paramShapes := stateShapes task) lrα st0
        let (st', loss) ←
          TorchLean.Module.optimizerStepWithLoss runner.module opt st sample .nil
        stRef.set st'
        stepRef.set (stepIdx + 1)
        pure (Spec.Tensor.toScalar loss)
      pure {
        runner := runner
        stepSample := runStep
        epochSamples := fun samples => samples.mapM runStep
        stepCount := stepRef.get
      }

/-- Run one optimization step on a single supervised sample. -/
def step {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (loop : Stepper α task) (sample : _root_.Runtime.Autograd.Torch.TList α [σ, τ]) : IO α :=
  loop.stepSample sample

/-- Run one epoch over a list of supervised samples, returning the per-step losses. -/
def epoch {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (loop : Stepper α task) (samples : List (_root_.Runtime.Autograd.Torch.TList α [σ, τ])) : IO (List α) :=
  loop.epochSamples samples

end Manual
end Trainer
end TorchLean
