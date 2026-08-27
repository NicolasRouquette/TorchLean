/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Trainer.Manual.Optimizer
public import NN.Data.SampleStream

/-!
# Manual Training Execution

Gradient accumulation, optimizer-step execution, and reusable finite-loop mechanics.
-/

@[expose] public section

namespace TorchLean
namespace Trainer
namespace Manual

open SeqTask

namespace Runner

/-- Return the state-shaped gradient of one supervised sample. -/
def gradState {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (sample : _root_.TorchLean.TensorPack α [σ, τ]) :
    IO (_root_.TorchLean.TensorPack α (stateShapes task)) := do
  train runner
  updateBuffers runner sample
  TorchLean.Module.gradState runner.module sample .nil

end Runner

namespace Internal

/-- Treat `0` as the conservative single-sample step size. -/
def effectiveTrainBatchSize (n : Nat) : Nat :=
  if n = 0 then 1 else n

/-- Add two shape-aligned gradient packs and materialize every result tensor. -/
def addGradientPacks {α : Type} [Add α] :
    {ss : List Spec.Shape} →
    _root_.TorchLean.TensorPack α ss →
    _root_.TorchLean.TensorPack α ss →
    _root_.TorchLean.TensorPack α ss
  | [], .nil, .nil => .nil
  | _ :: ss, .cons x xs, .cons y ys =>
      .cons
        (Spec.Tensor.materialize (Spec.Tensor.addSpec x y))
        (addGradientPacks (ss := ss) xs ys)

/-- Scale a shape-aligned gradient pack and materialize every result tensor. -/
def scaleGradientPack {α : Type} [Mul α] (c : α) :
    {ss : List Spec.Shape} →
    _root_.TorchLean.TensorPack α ss →
    _root_.TorchLean.TensorPack α ss
  | [], .nil => .nil
  | _ :: ss, .cons x xs =>
      .cons
        (Spec.Tensor.materialize (Spec.Tensor.scaleSpec x c))
        (scaleGradientPack c (ss := ss) xs)

/-- Compute mean parameter gradients for a nonempty batch at one parameter point. -/
def meanGradients {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (runner : Runner α task)
    (batch : Array (_root_.TorchLean.TensorPack α [σ, τ])) :
    IO (_root_.TorchLean.TensorPack α (stateShapes task)) := do
  match batch[0]? with
  | none => throw <| IO.userError "Supervised.meanGradients: empty batch"
  | some first =>
      for sample in batch do
        Runner.updateBuffers runner sample
      let mut gradSum ← TorchLean.Module.gradState runner.module first .nil
      for sample in batch.drop 1 do
        let grads ← TorchLean.Module.gradState runner.module sample .nil
        gradSum := addGradientPacks gradSum grads
      let invCount : α := 1 / (batch.size : α)
      pure (scaleGradientPack invCount gradSum)

/-- Compute mean loss and parameter gradients for a nonempty batch at one parameter point. -/
def meanLossAndGradients {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (runner : Runner α task)
    (batch : Array (_root_.TorchLean.TensorPack α [σ, τ])) :
    IO (α × _root_.TorchLean.TensorPack α (stateShapes task)) := do
  match batch[0]? with
  | none => throw <| IO.userError "Supervised.meanLossAndGradients: empty batch"
  | some first =>
      for sample in batch do
        Runner.updateBuffers runner sample
      let (firstLoss, firstGrads) ←
        TorchLean.Module.lossAndGradState runner.module first .nil
      let mut lossSum := Spec.Tensor.item firstLoss
      let mut gradSum := firstGrads
      for sample in batch.drop 1 do
        let (loss, grads) ← TorchLean.Module.lossAndGradState runner.module sample .nil
        lossSum := lossSum + Spec.Tensor.item loss
        gradSum := addGradientPacks gradSum grads
      let invCount : α := 1 / (batch.size : α)
      pure (lossSum * invCount, scaleGradientPack invCount gradSum)

/-- Apply one optimizer update without reading a loss scalar. -/
def stepBatch {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (runner : Runner α task)
    (opt : _root_.Runtime.Autograd.TorchLean.Optim.Optimizer α (stateShapes task))
    (state : opt.State) (useNativeSingleton : Bool)
    (batch : Array (_root_.TorchLean.TensorPack α [σ, τ])) : IO opt.State := do
  if hEmpty : batch.size = 0 then
    throw <| IO.userError "Supervised.stepBatch: empty batch"
  else if hSingleton : batch.size = 1 then
    let sample := batch[0]'(by simp [hSingleton])
    if useNativeSingleton then
      Runner.updateBuffers runner sample
      TorchLean.Module.optimizerStep runner.module opt state sample .nil
    else
      let meanGrads ← meanGradients runner batch
      let _ ← runner.module.trainer.getState
      opt.step state runner.module.trainer.state meanGrads
  else
    let meanGrads ← meanGradients runner batch
    let _ ← runner.module.trainer.getState
    opt.step state runner.module.trainer.state meanGrads

/-- Apply one optimizer update and return the loss from the gradient-producing tapes. -/
def stepBatchAndLoss {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (runner : Runner α task)
    (opt : _root_.Runtime.Autograd.TorchLean.Optim.Optimizer α (stateShapes task))
    (state : opt.State) (useNativeSingleton : Bool)
    (batch : Array (_root_.TorchLean.TensorPack α [σ, τ])) : IO (opt.State × α) := do
  if hEmpty : batch.size = 0 then
    throw <| IO.userError "Supervised.stepBatchAndLoss: empty batch"
  else if hSingleton : batch.size = 1 then
    let sample := batch[0]'(by simp [hSingleton])
    if useNativeSingleton then
      Runner.updateBuffers runner sample
      let (state', loss) ←
        TorchLean.Module.optimizerStepWithLoss runner.module opt state sample .nil
      pure (state', Spec.Tensor.item loss)
    else
      let (meanLoss, meanGrads) ← meanLossAndGradients runner batch
      let _ ← runner.module.trainer.getState
      let state' ← opt.step state runner.module.trainer.state meanGrads
      pure (state', meanLoss)
  else
    let (meanLoss, meanGrads) ← meanLossAndGradients runner batch
    let _ ← runner.module.trainer.getState
    let state' ← opt.step state runner.module.trainer.state meanGrads
    pure (state', meanLoss)

/-- Run fixed-step training for one concrete optimizer state type. -/
def runSampleSteps {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    (runner : Runner α task) (cfg : TrainConfig)
    (nextBatch : IO (Array (_root_.TorchLean.TensorPack α [σ, τ])))
    (watchMemory : Nat → IO Unit)
    (opt : _root_.Runtime.Autograd.TorchLean.Optim.Optimizer α (stateShapes task))
    (initialState : opt.State) (scheduleState : Nat → opt.State → opt.State) : IO Unit := do
  let mut state := initialState
  let useNativeSingleton := effectiveTrainBatchSize cfg.batchSize == 1
  for stepIdx in [0:cfg.steps] do
    let batch ← nextBatch
    let scheduledState := scheduleState stepIdx state
    if cfg.logEvery > 0 && stepIdx % cfg.logEvery = 0 then
      let (state', loss) ←
        stepBatchAndLoss runner opt scheduledState useNativeSingleton batch
      state := state'
      IO.println s!"step {stepIdx}: loss={loss}"
    else
      state ← stepBatch runner opt scheduledState useNativeSingleton batch
    watchMemory (stepIdx + 1)

/-- Run complete tensor-pack loader epochs for one concrete optimizer state type. -/
def runLoaderEpochs {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    (runner : Runner α task) (cfg : LoaderTrainConfig)
    (initialLoader : _root_.TorchLean.Data.EpochLoader
      (_root_.TorchLean.TensorPack α [σ, τ]))
    (opt : _root_.Runtime.Autograd.TorchLean.Optim.Optimizer α (stateShapes task))
    (initialState : opt.State) (scheduleState : Nat → opt.State → opt.State) :
    IO (_root_.TorchLean.Data.EpochLoader (_root_.TorchLean.TensorPack α [σ, τ])) := do
  let useNativeSingleton := initialLoader.batchSize == 1
  let rec trainBatches (stepIdx : Nat) (state : opt.State)
      (batches : Array (Array (_root_.TorchLean.TensorPack α [σ, τ]))) :
      IO (Nat × opt.State) := do
    let mut nextStep := stepIdx
    let mut nextState := state
    for batch in batches do
      if cfg.logEvery > 0 && nextStep % cfg.logEvery = 0 then
        let (state', loss) ← stepBatchAndLoss runner opt nextState useNativeSingleton batch
        IO.println s!"step {nextStep}: loss={loss}"
        nextState := state'
      else
        nextState ← stepBatch runner opt nextState useNativeSingleton batch
      nextStep := nextStep + 1
    pure (nextStep, nextState)
  let rec trainEpochs (epoch remaining stepIdx : Nat)
      (loader : _root_.TorchLean.Data.EpochLoader
        (_root_.TorchLean.TensorPack α [σ, τ]))
      (state : opt.State) :
      IO (_root_.TorchLean.Data.EpochLoader
        (_root_.TorchLean.TensorPack α [σ, τ])) := do
    match remaining with
    | 0 => pure loader
    | n + 1 =>
        let (loader', batches) ←
          match _root_.TorchLean.Data.EpochLoader.epoch "Manual.trainLoader" loader with
          | .ok out => pure out
          | .error msg => throw <| IO.userError s!"Manual.trainLoader: {msg}"
        let scheduledState := scheduleState epoch state
        let (stepIdx', state') ← trainBatches stepIdx scheduledState batches
        trainEpochs (epoch + 1) n stepIdx' loader' state'
  trainEpochs 0 cfg.epochs 0 initialLoader initialState

end Internal

end Manual
end Trainer
end TorchLean
