/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Runtime.Training.Core

import Mathlib.Algebra.Order.Algebra
import NN.Spec.Autograd.AutogradSpec

@[expose] public section

namespace NN
namespace API
namespace TorchLean

/-!
# Runtime Supervised Training

Training loops for in-memory datasets and `DataLoader`s. A list-valued batch means gradient
accumulation: every dataset item is differentiated at the same parameter point, the typed gradient
packs are averaged, and the optimizer is applied once. An item may itself contain a fixed-size
tensor minibatch; use a list batch of one for one vectorized pass in that case.
-/

namespace Supervised

/-! ## Gradient accumulation -/

namespace Internal

/-- Add two shape-aligned gradient packs and materialize every result tensor. -/
def addGradientPacks {α : Type} [Add α] :
    {ss : List Spec.Shape} →
    API.TorchLean.TensorPack α ss →
    API.TorchLean.TensorPack α ss →
    API.TorchLean.TensorPack α ss
  | [], .nil, .nil => .nil
  | _ :: ss, .cons x xs, .cons y ys =>
      .cons
        (Spec.Tensor.materialize (Spec.Tensor.addSpec x y))
        (addGradientPacks (ss := ss) xs ys)

/-- Scale a shape-aligned gradient pack and materialize every result tensor. -/
def scaleGradientPack {α : Type} [Mul α] (c : α) :
    {ss : List Spec.Shape} →
    API.TorchLean.TensorPack α ss →
    API.TorchLean.TensorPack α ss
  | [], .nil => .nil
  | _ :: ss, .cons x xs =>
      .cons
        (Spec.Tensor.materialize (Spec.Tensor.scaleSpec x c))
        (scaleGradientPack c (ss := ss) xs)

/--
Compute mean parameter gradients for a nonempty sample batch at one parameter point.

Mutable model buffers advance before differentiation. Trainable parameters remain fixed until all
sample gradients have been collected.
-/
def meanGradients {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task)
    (batch : List (API.TorchLean.TensorPack α [σ, τ])) :
    IO (API.TorchLean.TensorPack α (paramShapes task)) := do
  match batch with
  | [] =>
      throw <| IO.userError "Supervised.meanGradients: empty batch"
  | first :: rest =>
      for sample in batch do
        updateRunnerBuffers runner sample
      let mut gradSum ← API.TorchLean.Module.backward runner.module first
      for sample in rest do
        let grads ← API.TorchLean.Module.backward runner.module sample
        gradSum := addGradientPacks gradSum grads
      let invCount : α := 1 / (batch.length : α)
      pure (scaleGradientPack invCount gradSum)

/--
Compute the mean loss and parameter gradients for a nonempty sample batch at one parameter point.

Use this form only when the caller needs the loss value. On CUDA, reading that scalar back to the
host synchronizes the device.
-/
def meanLossAndGradients {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task)
    (batch : List (API.TorchLean.TensorPack α [σ, τ])) :
    IO (α × API.TorchLean.TensorPack α (paramShapes task)) := do
  match batch with
  | [] =>
      throw <| IO.userError "Supervised.meanLossAndGradients: empty batch"
  | first :: rest =>
      -- Buffer updates are state transitions, not optimizer steps. Apply them once before taking
      -- derivatives so every sample sees the same trainable parameter values and buffer state.
      for sample in batch do
        updateRunnerBuffers runner sample
      let (firstLoss, firstGrads) ←
        API.TorchLean.Module.lossAndBackward runner.module first
      let mut lossSum := Spec.Tensor.toScalar firstLoss
      let mut gradSum := firstGrads
      for sample in rest do
        let (loss, grads) ← API.TorchLean.Module.lossAndBackward runner.module sample
        lossSum := lossSum + Spec.Tensor.toScalar loss
        gradSum := addGradientPacks gradSum grads
      let invCount : α := 1 / (batch.length : α)
      pure (lossSum * invCount, scaleGradientPack invCount gradSum)

/--
Apply one optimizer update without reading a loss scalar.

`useNativeSingleton` is chosen once by the surrounding loop. Keeping it fixed prevents a final
partial loader batch from switching between device-resident and host optimizer state.
-/
def stepBatch {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task)
    (opt : API.TorchLean.Optim.Optimizer α (paramShapes task))
    (state : opt.State)
    (useNativeSingleton : Bool)
    (batch : List (API.TorchLean.TensorPack α [σ, τ])) :
    IO opt.State := do
  match batch with
  | [] =>
      throw <| IO.userError "Supervised.stepBatch: empty batch"
  | [sample] =>
      if useNativeSingleton then
        updateRunnerBuffers runner sample
        API.TorchLean.Module.stepWith runner.module opt state sample
      else
        let meanGrads ← meanGradients runner batch
        let _ ← runner.module.trainer.getParams
        opt.step state runner.module.trainer.params meanGrads
  | _ =>
      let meanGrads ← meanGradients runner batch
      let _ ← runner.module.trainer.getParams
      opt.step state runner.module.trainer.params meanGrads

/-- Apply one optimizer update and return the loss from the gradient-producing tapes. -/
def stepBatchAndLoss {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task)
    (opt : API.TorchLean.Optim.Optimizer α (paramShapes task))
    (state : opt.State)
    (useNativeSingleton : Bool)
    (batch : List (API.TorchLean.TensorPack α [σ, τ])) :
    IO (opt.State × α) := do
  match batch with
  | [] =>
      throw <| IO.userError "Supervised.stepBatchAndLoss: empty batch"
  | [sample] =>
      if useNativeSingleton then
        updateRunnerBuffers runner sample
        let (state', loss) ←
          API.TorchLean.Module.stepWithOptimizerAndLoss runner.module opt state sample
        pure (state', Spec.Tensor.toScalar loss)
      else
        let (meanLoss, meanGrads) ← meanLossAndGradients runner batch
        let _ ← runner.module.trainer.getParams
        let state' ← opt.step state runner.module.trainer.params meanGrads
        pure (state', meanLoss)
  | _ =>
      let (meanLoss, meanGrads) ← meanLossAndGradients runner batch
      let _ ← runner.module.trainer.getParams
      let state' ← opt.step state runner.module.trainer.params meanGrads
      pure (state', meanLoss)

/-- Run the fixed-step in-memory loop for one concrete optimizer state type. -/
def runSampleSteps {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    (runner : Runner α task)
    (cfg : TrainConfig)
    (nextBatch : IO (List (API.TorchLean.TensorPack α [σ, τ])))
    (watchMemory : Nat → IO Unit)
    (opt : API.TorchLean.Optim.Optimizer α (paramShapes task))
    (initialState : opt.State)
    (scheduleState : Nat → opt.State → opt.State) :
    IO Unit := do
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

end Internal

/-! ## In-memory datasets -/

/--
Train on an in-memory sample list for a fixed number of optimizer updates.

`cfg.batchSize` controls how many dataset items contribute to each update. It does not multiply the
number of optimizer steps. If an item is already a vectorized tensor minibatch, use `batchSize := 1`
unless you intend to accumulate gradients across several such minibatches.
-/
def trainSamples {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [API.Runtime.Scalar α]
    (runner : Runner α task)
    (cfg : TrainConfig)
    (samples : List (API.TorchLean.TensorPack α [σ, τ])) :
    IO (TrainReport α) := do
  trainMode runner
  let before ← meanLoss runner samples
  let watchEvery := effectiveCudaMemWatch runner.module.opts cfg.steps cfg.cudaMemWatch
  let memWatchRef ← IO.mkRef (none : Option CudaMemWatchState)
  let watchMemory (done : Nat) : IO Unit := do
    let state ← memWatchRef.get
    let state ← reportCudaMemWatch runner.module.opts watchEvery cfg.steps done state
    memWatchRef.set state
  watchMemory 0
  unless samples.isEmpty do
    let restRef ← IO.mkRef samples
    let nextBatch :=
      nextCyclicBatch "Supervised.trainSamples" samples restRef
        (effectiveTrainBatchSize cfg.batchSize)
    match cfg.optimizer with
    | .sgd lr momentum =>
        if momentum == 0.0 then
          let opt := API.TorchLean.Optim.sgd
            (α := α) (paramShapes := paramShapes task) (API.Runtime.ofFloat lr)
          let state ← API.TorchLean.Module.initOptim runner.module opt
          Internal.runSampleSteps runner cfg nextBatch watchMemory opt state (fun step state =>
            sgdStateWithLR (paramShapes := paramShapes task)
              (API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer step)) state)
        else
          let opt := API.TorchLean.Optim.momentumSGD
            (α := α) (paramShapes := paramShapes task)
            (API.Runtime.ofFloat lr) (API.Runtime.ofFloat momentum)
          let state ← API.TorchLean.Module.initOptim runner.module opt
          Internal.runSampleSteps runner cfg nextBatch watchMemory opt state (fun step state =>
            momentumSGDStateWithLR (paramShapes := paramShapes task)
              (API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer step)) state)
    | .adagrad lr epsilon =>
        let opt := API.TorchLean.Optim.adagrad
          (α := α) (paramShapes := paramShapes task)
          (API.Runtime.ofFloat lr) (API.Runtime.ofFloat epsilon)
        let state ← API.TorchLean.Module.initOptim runner.module opt
        Internal.runSampleSteps runner cfg nextBatch watchMemory opt state (fun step state =>
          adagradStateWithLR (paramShapes := paramShapes task)
            (API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer step)) state)
    | .rmsprop lr decay epsilon =>
        let opt := API.TorchLean.Optim.rmsprop
          (α := α) (paramShapes := paramShapes task)
          (API.Runtime.ofFloat lr) (API.Runtime.ofFloat decay) (API.Runtime.ofFloat epsilon)
        let state ← API.TorchLean.Module.initOptim runner.module opt
        Internal.runSampleSteps runner cfg nextBatch watchMemory opt state (fun step state =>
          rmspropStateWithLR (paramShapes := paramShapes task)
            (API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer step)) state)
    | .adam lr beta1 beta2 epsilon =>
        let opt := API.TorchLean.Optim.adam
          (α := α) (paramShapes := paramShapes task)
          (API.Runtime.ofFloat lr) (API.Runtime.ofFloat beta1) (API.Runtime.ofFloat beta2)
          (API.Runtime.ofFloat epsilon)
        let state ← API.TorchLean.Module.initOptim runner.module opt
        Internal.runSampleSteps runner cfg nextBatch watchMemory opt state (fun step state =>
          adamStateWithLR (paramShapes := paramShapes task)
            (API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer step)) state)
    | .adamw lr weightDecay beta1 beta2 epsilon =>
        let opt := API.TorchLean.Optim.adamw
          (α := α) (paramShapes := paramShapes task)
          (API.Runtime.ofFloat lr) (API.Runtime.ofFloat weightDecay)
          (API.Runtime.ofFloat beta1) (API.Runtime.ofFloat beta2) (API.Runtime.ofFloat epsilon)
        let state ← API.TorchLean.Module.initOptim runner.module opt
        Internal.runSampleSteps runner cfg nextBatch watchMemory opt state (fun step state =>
          adamwStateWithLR (paramShapes := paramShapes task)
            (API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer step)) state)
    | .adadelta lr rho epsilon =>
        let opt := API.TorchLean.Optim.adadelta
          (α := α) (paramShapes := paramShapes task)
          (API.Runtime.ofFloat lr) (API.Runtime.ofFloat rho) (API.Runtime.ofFloat epsilon)
        let state ← API.TorchLean.Module.initOptim runner.module opt
        Internal.runSampleSteps runner cfg nextBatch watchMemory opt state (fun step state =>
          adadeltaStateWithLR (paramShapes := paramShapes task)
            (API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer step)) state)
  let after ← meanLoss runner samples
  pure { before := before, after := after }

/-- Train over a dataset by materializing its sample list. -/
def trainDataset {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [API.Runtime.Scalar α]
    (runner : Runner α task)
    (cfg : TrainConfig)
    (dataset : _root_.Runtime.Autograd.Train.Dataset
      (API.TorchLean.TensorPack α [σ, τ])) :
    IO (TrainReport α) :=
  trainSamples runner cfg dataset.toList

/-! ## Data loaders -/

namespace Internal

/-- Run all loader epochs for one concrete optimizer state type. -/
def runLoaderEpochs {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    (runner : Runner α task)
    (cfg : LoaderTrainConfig)
    (initialLoader : _root_.Runtime.Autograd.Train.DataLoader
      (API.TorchLean.TensorPack α [σ, τ]))
    (opt : API.TorchLean.Optim.Optimizer α (paramShapes task))
    (initialState : opt.State)
    (scheduleState : Nat → opt.State → opt.State) :
    IO (_root_.Runtime.Autograd.Train.DataLoader
      (API.TorchLean.TensorPack α [σ, τ])) := do
  let useNativeSingleton := initialLoader.batchSize == 1
  let rec trainBatches (stepIdx : Nat) (state : opt.State)
      (batches : List (List (API.TorchLean.TensorPack α [σ, τ]))) :
      IO (Nat × opt.State) := do
    match batches with
    | [] =>
        pure (stepIdx, state)
    | batch :: rest =>
        if cfg.logEvery > 0 && stepIdx % cfg.logEvery = 0 then
          let (state', loss) ←
            stepBatchAndLoss runner opt state useNativeSingleton batch
          IO.println s!"step {stepIdx}: loss={loss}"
          trainBatches (stepIdx + 1) state' rest
        else
          let state' ← stepBatch runner opt state useNativeSingleton batch
          trainBatches (stepIdx + 1) state' rest
  let rec trainEpochs (epoch remaining stepIdx : Nat)
      (loader : _root_.Runtime.Autograd.Train.DataLoader
        (API.TorchLean.TensorPack α [σ, τ]))
      (state : opt.State) :
      IO (_root_.Runtime.Autograd.Train.DataLoader
        (API.TorchLean.TensorPack α [σ, τ])) := do
    match remaining with
    | 0 =>
        pure loader
    | n + 1 =>
        let (loader', batches) ←
          match _root_.Runtime.Autograd.Train.DataLoader.epoch
              "Supervised.trainLoader" loader with
          | .ok out => pure out
          | .error msg => throw <| IO.userError s!"Supervised.trainLoader: {msg}"
        let scheduledState := scheduleState epoch state
        let (stepIdx', state') ← trainBatches stepIdx scheduledState batches
        trainEpochs (epoch + 1) n stepIdx' loader' state'
  trainEpochs 0 cfg.epochs 0 initialLoader initialState

end Internal

/--
Train over a `DataLoader`.

Every loader batch contributes one averaged-gradient optimizer update. The scheduler advances once
per epoch.
-/
def trainLoader {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [API.Runtime.Scalar α]
    (runner : Runner α task)
    (cfg : LoaderTrainConfig)
    (dl : _root_.Runtime.Autograd.Train.DataLoader
      (API.TorchLean.TensorPack α [σ, τ])) :
    IO (TrainReport α ×
      _root_.Runtime.Autograd.Train.DataLoader
        (API.TorchLean.TensorPack α [σ, τ])) := do
  trainMode runner
  let before ← meanLossDataset runner dl.dataset
  let dl' ←
    match cfg.optimizer with
    | .sgd lr momentum =>
        if momentum == 0.0 then
          let opt := API.TorchLean.Optim.sgd
            (α := α) (paramShapes := paramShapes task) (API.Runtime.ofFloat lr)
          let state ← API.TorchLean.Module.initOptim runner.module opt
          Internal.runLoaderEpochs runner cfg dl opt state (fun step state =>
            sgdStateWithLR (paramShapes := paramShapes task)
              (API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer step)) state)
        else
          let opt := API.TorchLean.Optim.momentumSGD
            (α := α) (paramShapes := paramShapes task)
            (API.Runtime.ofFloat lr) (API.Runtime.ofFloat momentum)
          let state ← API.TorchLean.Module.initOptim runner.module opt
          Internal.runLoaderEpochs runner cfg dl opt state (fun step state =>
            momentumSGDStateWithLR (paramShapes := paramShapes task)
              (API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer step)) state)
    | .adagrad lr epsilon =>
        let opt := API.TorchLean.Optim.adagrad
          (α := α) (paramShapes := paramShapes task)
          (API.Runtime.ofFloat lr) (API.Runtime.ofFloat epsilon)
        let state ← API.TorchLean.Module.initOptim runner.module opt
        Internal.runLoaderEpochs runner cfg dl opt state (fun step state =>
          adagradStateWithLR (paramShapes := paramShapes task)
            (API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer step)) state)
    | .rmsprop lr decay epsilon =>
        let opt := API.TorchLean.Optim.rmsprop
          (α := α) (paramShapes := paramShapes task)
          (API.Runtime.ofFloat lr) (API.Runtime.ofFloat decay) (API.Runtime.ofFloat epsilon)
        let state ← API.TorchLean.Module.initOptim runner.module opt
        Internal.runLoaderEpochs runner cfg dl opt state (fun step state =>
          rmspropStateWithLR (paramShapes := paramShapes task)
            (API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer step)) state)
    | .adam lr beta1 beta2 epsilon =>
        let opt := API.TorchLean.Optim.adam
          (α := α) (paramShapes := paramShapes task)
          (API.Runtime.ofFloat lr) (API.Runtime.ofFloat beta1) (API.Runtime.ofFloat beta2)
          (API.Runtime.ofFloat epsilon)
        let state ← API.TorchLean.Module.initOptim runner.module opt
        Internal.runLoaderEpochs runner cfg dl opt state (fun step state =>
          adamStateWithLR (paramShapes := paramShapes task)
            (API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer step)) state)
    | .adamw lr weightDecay beta1 beta2 epsilon =>
        let opt := API.TorchLean.Optim.adamw
          (α := α) (paramShapes := paramShapes task)
          (API.Runtime.ofFloat lr) (API.Runtime.ofFloat weightDecay)
          (API.Runtime.ofFloat beta1) (API.Runtime.ofFloat beta2) (API.Runtime.ofFloat epsilon)
        let state ← API.TorchLean.Module.initOptim runner.module opt
        Internal.runLoaderEpochs runner cfg dl opt state (fun step state =>
          adamwStateWithLR (paramShapes := paramShapes task)
            (API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer step)) state)
    | .adadelta lr rho epsilon =>
        let opt := API.TorchLean.Optim.adadelta
          (α := α) (paramShapes := paramShapes task)
          (API.Runtime.ofFloat lr) (API.Runtime.ofFloat rho) (API.Runtime.ofFloat epsilon)
        let state ← API.TorchLean.Module.initOptim runner.module opt
        Internal.runLoaderEpochs runner cfg dl opt state (fun step state =>
          adadeltaStateWithLR (paramShapes := paramShapes task)
            (API.Runtime.ofFloat (stepLR cfg.scheduler cfg.optimizer step)) state)
  let after ← meanLossDataset runner dl'.dataset
  pure ({ before := before, after := after }, dl')

end Supervised

end TorchLean
end API
end NN
