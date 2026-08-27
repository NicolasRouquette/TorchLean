/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Trainer.Manual.Control

/-!
# Stream Training

Step-indexed typed batch sources and direct-module training for generated, replay-buffer, and
external streaming workloads.
-/

@[expose] public section

namespace TorchLean
namespace Trainer
namespace Manual

/-- An effectful source of already-collated module inputs indexed by optimizer step. -/
structure StepBatchStream (α : Type) (inputShapes : List Shape) where
  /-- Produce the input used at the given logical optimizer step. -/
  sample : Nat → IO (_root_.TorchLean.TensorPack α inputShapes)

namespace StepBatchStream

/-- Repeat one batch at every step. -/
def fixed {α : Type} {inputShapes : List Shape}
    (sample : _root_.TorchLean.TensorPack α inputShapes) : StepBatchStream α inputShapes :=
  { sample := fun _ => pure sample }

/-- Lift a pure step-indexed source. -/
def ofFn {α : Type} {inputShapes : List Shape}
    (sample : Nat → _root_.TorchLean.TensorPack α inputShapes) : StepBatchStream α inputShapes :=
  { sample := fun step => pure (sample step) }

/-- Cycle through a nonempty in-memory array. -/
def cycle {α : Type} {inputShapes : List Shape}
    (samples : Array (_root_.TorchLean.TensorPack α inputShapes)) (h : samples.size > 0) :
    StepBatchStream α inputShapes :=
  let stream := _root_.TorchLean.Data.SampleStream.ofArray samples
  { sample := fun step => pure (stream.cycle (by simpa [stream] using h) step) }

end StepBatchStream

/--
Train a scalar objective for a fixed number of updates from a step-indexed batch source.

The source is sampled at indices `0` and `steps` for the report endpoints. Those evaluations are
separate from the samples requested by optimizer updates.
-/
def Objective.trainStream {inputShapes stateShapes : List Shape}
    {α : Type} [_root_.Context α] [DecidableEq Shape]
    (module : TorchLean.Module.Objective α Unit stateShapes inputShapes)
    (optimizer : _root_.Runtime.Autograd.TorchLean.Optim.Optimizer α stateShapes)
    (steps : Nat) (stream : StepBatchStream α inputShapes)
    (callbacks : Callbacks α := {}) : IO (TrainReport α) := do
  let beforeSample ← stream.sample 0
  let before := Spec.Tensor.item (← TorchLean.Module.loss module beforeSample .nil)
  callbacks.onTrainStart
  let mut optimizerState ← TorchLean.Module.initOptimizer module optimizer
  for step in [0:steps] do
    let sample ← stream.sample step
    let loss := Spec.Tensor.item (← TorchLean.Module.loss module sample .nil)
    callbacks.onStep { epoch := 0, step, loss }
    optimizerState ← TorchLean.Module.optimizerStep module optimizer optimizerState sample .nil
  callbacks.onEpochEnd { epoch := 0, steps }
  let afterSample ← stream.sample steps
  let after := Spec.Tensor.item (← TorchLean.Module.loss module afterSample .nil)
  let report := { before, after }
  callbacks.onTrainEnd report
  pure report

end Manual
end Trainer
end TorchLean
