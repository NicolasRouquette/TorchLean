/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Trainer.Manual.Control
public import NN.API.Trainer.Manual.Loops
public import NN.API.Trainer.Manual.Stepper

/-!
# Loader Training

Callback-driven training over typed supervised epoch loaders. Direct objectives support epoch and
optimizer-step limits; runner training additionally applies the configured scheduler.
-/

@[expose] public section

namespace TorchLean
namespace Trainer
namespace Manual

open SeqTask

/--
Train over a finite tensor-pack loader.

Each loader batch contributes one averaged-gradient optimizer update. The scheduler advances once
per epoch.
-/
def trainLoader {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    [_root_.TorchLean.Runtime.FromFloat α]
    (runner : Runner α task) (cfg : LoaderTrainConfig)
    (loader : _root_.TorchLean.Data.EpochLoader
      (_root_.TorchLean.TensorPack α [σ, τ])) :
    IO (TrainReport α ×
      _root_.TorchLean.Data.EpochLoader (_root_.TorchLean.TensorPack α [σ, τ])) := do
  runner.train
  let before ← runner.meanLossStream loader.samples
  let loader' ←
    Internal.withBoundOptimizer runner cfg.optimizer cfg.scheduler fun opt state scheduleState =>
      Internal.runLoaderEpochs runner cfg loader opt state scheduleState
  let after ← runner.meanLossStream loader'.samples
  pure ({ before, after }, loader')

/--
Train a scalar objective over a supervised loader until an epoch or optimizer-step limit is met.

The returned loader contains the shuffle state for the next epoch. A step-limited run rejects an
empty epoch because it could otherwise fail to make progress.
-/
def Objective.trainLoader {inputShape targetShape : List Nat} {n : Nat}
    {stateShapes : List Shape} {α : Type} [_root_.Context α] [DecidableEq Shape]
    (module : TorchLean.Module.Objective α Unit stateShapes
      [Shape.ofList (n :: inputShape), Shape.ofList (n :: targetShape)])
    (optimizer : _root_.Runtime.Autograd.TorchLean.Optim.Optimizer α stateShapes)
    (limit : TrainingLimit)
    (loader : TorchLean.Data.SupervisedEpochs α n inputShape targetShape)
    (callbacks : Callbacks α := {}) :
    IO (TrainReport α × TorchLean.Data.SupervisedEpochs α n inputShape targetShape) := do
  let before ← Objective.meanLoss module loader
  callbacks.onTrainStart
  let active : Nat → Nat → Bool := fun epoch step =>
    match limit with
    | .epochs count => decide (epoch < count)
    | .steps count => decide (step < count)
  let acceptsStep : Nat → Bool := fun step =>
    match limit with
    | .epochs _ => true
    | .steps count => decide (step < count)
  let requiresNonemptyEpoch :=
    match limit with
    | .epochs _ => false
    | .steps _ => true
  let mut optimizerState ← TorchLean.Module.initOptimizer module optimizer
  let mut currentLoader := loader
  let mut epoch := 0
  let mut step := 0
  while active epoch step do
    let (nextLoader, rawBatches) ←
      match TorchLean.Data.EpochLoader.epoch "Objective.trainLoader" currentLoader.loader with
      | .ok result => pure result
      | .error message => throw <| IO.userError s!"Objective.trainLoader: {message}"
    if requiresNonemptyEpoch && rawBatches.isEmpty then
      throw <| IO.userError "Objective.trainLoader: loader produced no batches"
    currentLoader := { loader := nextLoader }
    let epochStart := step
    for rawBatch in rawBatches do
      if acceptsStep step then
        let sample ← TorchLean.CLI.orThrow "Objective.trainLoader" <|
          TorchLean.Data.collateSupervised (α := α) (inputShape := inputShape)
            (targetShape := targetShape) n rawBatch
        let loss := Spec.Tensor.item (← TorchLean.Module.loss module sample .nil)
        callbacks.onStep { epoch, step, loss }
        optimizerState ←
          TorchLean.Module.optimizerStep module optimizer optimizerState sample .nil
        step := step + 1
    callbacks.onEpochEnd { epoch, steps := step - epochStart }
    epoch := epoch + 1
  let after ← Objective.meanLoss module currentLoader
  let report := { before, after }
  callbacks.onTrainEnd report
  pure (report, currentLoader)

/--
Train a task runner for complete loader epochs using its optimizer and scheduler configuration.
-/
def Runner.trainLoader {inputShape targetShape : List Nat} {n : Nat}
    {task : SeqTask (Shape.ofList (n :: inputShape)) (Shape.ofList (n :: targetShape))}
    {α : Type} [_root_.Context α] [DecidableEq Shape] [ToString α]
    [_root_.TorchLean.Runtime.FromFloat α]
    (runner : Runner α task) (config : LoaderTrainConfig)
    (loader : TorchLean.Data.SupervisedEpochs α n inputShape targetShape)
    (callbacks : Callbacks α := {}) :
    IO (TrainReport α × TorchLean.Data.SupervisedEpochs α n inputShape targetShape) := do
  runner.eval
  let before ← Objective.meanLoss runner.module loader
  callbacks.onTrainStart
  runner.train
  let stepper ← Manual.stepper runner config.optimizer (scheduler := config.scheduler)
  let mut currentLoader := loader
  let mut globalStep := 0
  for epoch in [0:config.epochs] do
    let (nextLoader, rawBatches) ←
      match TorchLean.Data.EpochLoader.epoch "Runner.trainLoader" currentLoader.loader with
      | .ok result => pure result
      | .error message => throw <| IO.userError s!"Runner.trainLoader: {message}"
    currentLoader := { loader := nextLoader }
    let epochStart := globalStep
    for rawBatch in rawBatches do
      let sample ← TorchLean.CLI.orThrow "Runner.trainLoader" <|
        TorchLean.Data.collateSupervised (α := α) (inputShape := inputShape)
          (targetShape := targetShape) n rawBatch
      let loss ← stepper.stepSample sample
      callbacks.onStep { epoch, step := globalStep, loss }
      globalStep := globalStep + 1
    callbacks.onEpochEnd { epoch, steps := globalStep - epochStart }
  runner.eval
  let after ← Objective.meanLoss runner.module currentLoader
  let report := { before, after }
  callbacks.onTrainEnd report
  pure (report, currentLoader)

end Manual
end Trainer
end TorchLean
