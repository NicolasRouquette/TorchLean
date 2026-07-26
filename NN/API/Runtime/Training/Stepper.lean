/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Runtime.Training.Loops

import Mathlib.Algebra.Order.Algebra
import NN.Spec.Autograd.AutogradSpec

@[expose] public section

namespace NN
namespace API
namespace TorchLean

/-!
# Runtime Supervised Training

Supervised tasks, runners, steppers, optimizer configs, trainer aliases, and the low-level session
exports that back executable examples.
-/

namespace Supervised

/-! ## Stateful training steps -/

/--
Stateful training loop object: a `Runner` plus an optimizer state and a step counter. It packages
the model runner with the state needed to step on successive batches.
-/
structure Stepper (α : Type) [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    {σ τ : Spec.Shape} (task : SeqTask σ τ) where
  /-- Underlying task runner (module + compiled forward artifacts/losses). -/
  runner : Runner α task
  /-- Run a single optimization step on one supervised sample, returning the loss value. -/
  stepSample : API.TorchLean.TensorPack α [σ, τ] → IO α
  /-- Run an epoch over an explicit list of samples, returning the per-step loss values. -/
  epochSamples : List (API.TorchLean.TensorPack α [σ, τ]) → IO (List α)
  /-- Read the total number of `stepSample` calls performed so far. -/
  stepCount : IO Nat

/--
Construct a `Stepper` for a runner, optimizer config, and optional scheduler.

This is the recommended way to build custom training loops without reimplementing the optimizer
logic: call `stepper`, then choose `stepSample` for single batches or `epochSamples` for explicit
sample lists.
-/
def stepper {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α] [API.Runtime.Scalar α]
    (runner : Runner α task) (optimizer : OptimizerConfig)
    (scheduler : Option API.TorchLean.Schedulers.Config := none) :
    IO (Stepper α task) := do
  trainMode runner
  let stepRef ← IO.mkRef 0
  match optimizer with
  | .sgd lr momentum =>
      if momentum == 0.0 then
        let runStep := fun (sample : API.TorchLean.TensorPack α [σ, τ]) => do
          trainMode runner
          let stepIdx ← stepRef.get
          updateRunnerBuffers runner sample
          let lrα := API.Runtime.ofFloat (stepLR scheduler optimizer stepIdx)
          let loss ← API.TorchLean.Module.stepWithLoss runner.module lrα sample
          stepRef.set (stepIdx + 1)
          pure (Spec.Tensor.toScalar loss)
        pure {
          runner := runner
          stepSample := runStep
          epochSamples := fun samples => samples.mapM runStep
          stepCount := stepRef.get
        }
      else
        let opt := API.TorchLean.Optim.momentumSGD
          (α := α) (paramShapes := paramShapes task)
          (lr := API.Runtime.ofFloat lr) (momentum := API.Runtime.ofFloat momentum)
        let st0 : API.TorchLean.Optim.StateList _root_.Optim.MomentumSGD.State α (paramShapes task)
          ←
          API.TorchLean.Module.initOptim runner.module opt
        let stRef ← IO.mkRef st0
        let runStep := fun (sample : API.TorchLean.TensorPack α [σ, τ]) => do
          trainMode runner
          let stepIdx ← stepRef.get
          updateRunnerBuffers runner sample
          let lrα := API.Runtime.ofFloat (stepLR scheduler optimizer stepIdx)
          let st0 ← stRef.get
          let st := momentumSGDStateWithLR (paramShapes := paramShapes task) lrα st0
          let (st', loss) ←
            API.TorchLean.Module.stepWithOptimizerAndLoss runner.module opt st sample
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
      let opt := API.TorchLean.Optim.adagrad
        (α := α) (paramShapes := paramShapes task)
        (lr := API.Runtime.ofFloat lr)
        (epsilon := API.Runtime.ofFloat epsilon)
      let st0 : API.TorchLean.Optim.StateList _root_.Optim.AdaGrad.State α (paramShapes task) ←
        API.TorchLean.Module.initOptim runner.module opt
      let stRef ← IO.mkRef st0
      let runStep := fun (sample : API.TorchLean.TensorPack α [σ, τ]) => do
        trainMode runner
        let stepIdx ← stepRef.get
        updateRunnerBuffers runner sample
        let lrα := API.Runtime.ofFloat (stepLR scheduler optimizer stepIdx)
        let st0 ← stRef.get
        let st := adagradStateWithLR (paramShapes := paramShapes task) lrα st0
        let (st', loss) ← API.TorchLean.Module.stepWithOptimizerAndLoss runner.module opt st sample
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
      let opt := API.TorchLean.Optim.rmsprop
        (α := α) (paramShapes := paramShapes task)
        (lr := API.Runtime.ofFloat lr)
        (decay := API.Runtime.ofFloat decay)
        (epsilon := API.Runtime.ofFloat epsilon)
      let st0 : API.TorchLean.Optim.StateList _root_.Optim.RMSProp.State α (paramShapes task) ←
        API.TorchLean.Module.initOptim runner.module opt
      let stRef ← IO.mkRef st0
      let runStep := fun (sample : API.TorchLean.TensorPack α [σ, τ]) => do
        trainMode runner
        let stepIdx ← stepRef.get
        updateRunnerBuffers runner sample
        let lrα := API.Runtime.ofFloat (stepLR scheduler optimizer stepIdx)
        let st0 ← stRef.get
        let st := rmspropStateWithLR (paramShapes := paramShapes task) lrα st0
        let (st', loss) ← API.TorchLean.Module.stepWithOptimizerAndLoss runner.module opt st sample
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
      let opt := API.TorchLean.Optim.adam
        (α := α) (paramShapes := paramShapes task)
        (lr := API.Runtime.ofFloat lr)
        (beta1 := API.Runtime.ofFloat beta1)
        (beta2 := API.Runtime.ofFloat beta2)
        (epsilon := API.Runtime.ofFloat epsilon)
      let st0 : API.TorchLean.Optim.StateList _root_.Optim.Adam.State α (paramShapes task) ←
        API.TorchLean.Module.initOptim runner.module opt
      let stRef ← IO.mkRef st0
      let runStep := fun (sample : API.TorchLean.TensorPack α [σ, τ]) => do
        trainMode runner
        let stepIdx ← stepRef.get
        updateRunnerBuffers runner sample
        let lrα := API.Runtime.ofFloat (stepLR scheduler optimizer stepIdx)
        let st0 ← stRef.get
        let st := adamStateWithLR (paramShapes := paramShapes task) lrα st0
        let (st', loss) ← API.TorchLean.Module.stepWithOptimizerAndLoss runner.module opt st sample
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
      let opt := API.TorchLean.Optim.adamw
        (α := α) (paramShapes := paramShapes task)
        (lr := API.Runtime.ofFloat lr) (weightDecay := API.Runtime.ofFloat weightDecay)
        (beta1 := API.Runtime.ofFloat beta1)
        (beta2 := API.Runtime.ofFloat beta2)
        (epsilon := API.Runtime.ofFloat epsilon)
      let st0 : API.TorchLean.Optim.StateList _root_.Optim.AdamW.State α (paramShapes task) ←
        API.TorchLean.Module.initOptim runner.module opt
      let stRef ← IO.mkRef st0
      let runStep := fun (sample : API.TorchLean.TensorPack α [σ, τ]) => do
        trainMode runner
        let stepIdx ← stepRef.get
        updateRunnerBuffers runner sample
        let lrα := API.Runtime.ofFloat (stepLR scheduler optimizer stepIdx)
        let st0 ← stRef.get
        let st := adamwStateWithLR (paramShapes := paramShapes task) lrα st0
        let (st', loss) ← API.TorchLean.Module.stepWithOptimizerAndLoss runner.module opt st sample
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
      let opt := API.TorchLean.Optim.adadelta
        (α := α) (paramShapes := paramShapes task)
        (lr := API.Runtime.ofFloat lr)
        (rho := API.Runtime.ofFloat rho)
        (epsilon := API.Runtime.ofFloat epsilon)
      let st0 : API.TorchLean.Optim.StateList _root_.Optim.Adadelta.State α (paramShapes task) ←
        API.TorchLean.Module.initOptim runner.module opt
      let stRef ← IO.mkRef st0
      let runStep := fun (sample : API.TorchLean.TensorPack α [σ, τ]) => do
        trainMode runner
        let stepIdx ← stepRef.get
        updateRunnerBuffers runner sample
        let lrα := API.Runtime.ofFloat (stepLR scheduler optimizer stepIdx)
        let st0 ← stRef.get
        let st := adadeltaStateWithLR (paramShapes := paramShapes task) lrα st0
        let (st', loss) ← API.TorchLean.Module.stepWithOptimizerAndLoss runner.module opt st sample
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
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (loop : Stepper α task) (sample : API.TorchLean.TensorPack α [σ, τ]) : IO α :=
  loop.stepSample sample

/-- Run one epoch over a list of supervised samples, returning the per-step losses. -/
def epoch {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (loop : Stepper α task) (samples : List (API.TorchLean.TensorPack α [σ, τ])) : IO (List α) :=
  loop.epochSamples samples

end Supervised

end TorchLean
end API
end NN
