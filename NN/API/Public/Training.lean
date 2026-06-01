/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Adapters
public import NN.API.Common
public import NN.API.Data
public import NN.API.Macros
public import NN.API.Rand
public import NN.API.RL
public import NN.API.Runtime
public import NN.API.Samples
public import NN.API.SelfSupervised
public import NN.API.Text
public import NN.API.Text.Bpe
public import NN.Spec.Layers.PositionalEncoding

import Mathlib.Algebra.Order.Algebra

@[expose] public section

namespace NN
namespace API

/-!
# Public optimizers, losses, metrics, and training helpers

This module contains the executable training API: optimizer configs, loss exports, metrics,
callbacks, loaders, and module-level fit helpers.
-/

namespace optim

/-!
Optimizer configs for the high-level training helpers.

These mirror common PyTorch optimizers (by name and default hyperparameters), but they produce a
TorchLean trainer config rather than a mutable optimizer object.

PyTorch references:
- `torch.optim`: `https://pytorch.org/docs/stable/optim.html`
-/

@[inherit_doc TorchLean.Trainer.Optimizer]
abbrev Optimizer := TorchLean.Trainer.Optimizer

@[inherit_doc TorchLean.Trainer.sgd]
abbrev sgd := TorchLean.Trainer.sgd

@[inherit_doc TorchLean.Trainer.momentumSGD]
abbrev momentumSGD := TorchLean.Trainer.momentumSGD

@[inherit_doc TorchLean.Trainer.adam]
abbrev adam := TorchLean.Trainer.adam

@[inherit_doc TorchLean.Trainer.adamw]
abbrev adamw := TorchLean.Trainer.adamw

end optim

namespace loss

/-
Loss functions are re-exported from the TorchLean runtime.

PyTorch references:
- `torch.nn.functional` loss docs: `https://pytorch.org/docs/stable/nn.functional.html`
-/

@[inherit_doc TorchLean.Loss.Reduction]
abbrev Reduction := TorchLean.Loss.Reduction

export TorchLean.Loss
  (mse
   nllOneHot crossEntropyOneHot
   nllIndex nllNat crossEntropyIndex crossEntropyNat
   rowTargetFlatIndices nllRowsNat crossEntropyRowsNat
   bceWithLogits bce)

end loss

namespace metrics

/- Small classification metrics helpers (argmax, one-hot correctness, etc.). -/
export TorchLean.Metrics (argmax? classOfOneHot? correctOneHot?)

end metrics

namespace train

/-!
High-level training helpers.

This namespace is the public training surface: it wires together
- a model (`nn.Sequential`)
- a loss (regression or classification)
- an optimizer config (`API.optim`)
- optional LR schedules

The API exposes a small set of default building blocks, so model commands can share the same training
path while still making the model, loss, optimizer, and logging choices explicit.

### PyTorch Mapping

These helpers correspond to the training loop code you would typically write around:
- `torch.optim.*`
- forward pass + loss
- `loss.backward()` + optimizer step
- batching via `torch.utils.data.DataLoader`
-/

@[inherit_doc TorchLean.Trainer.Task]
abbrev Task := TorchLean.Trainer.Task
@[inherit_doc TorchLean.Trainer.Runner]
abbrev Runner := TorchLean.Trainer.Runner
@[inherit_doc TorchLean.Trainer.Stepper]
abbrev Stepper := TorchLean.Trainer.Stepper

@[inherit_doc TorchLean.Trainer.FitConfig]
abbrev FitConfig := TorchLean.Trainer.FitConfig
@[inherit_doc TorchLean.Trainer.LoaderFitConfig]
abbrev LoaderFitConfig := TorchLean.Trainer.LoaderFitConfig
@[inherit_doc TorchLean.Trainer.FitReport]
abbrev FitReport := TorchLean.Trainer.FitReport

/-!
`API.train.*` exposes the stable training names from `TorchLean.Trainer`.

The declarations are exported directly so the public namespace points at the runtime definitions
instead of duplicating one-line forwarders.
-/

export TorchLean.Trainer
  (regression
   classificationOneHot
   steps epochs
   constantLR stepLR exponentialLR
   constantEpochLR stepEpochLR exponentialEpochLR
   instantiate instantiateWithOptions
   run
   params mode setMode trainMode evalMode isTraining
   backward
   predict predictBatch predictClass?
   accuracyOneHot)

/-!
## Metric Artifacts

The public training API also exposes TorchLean's metric artifact format.  This is
the local equivalent of “log scalars during a run, then inspect them later”: write a JSON
`TrainLog`, view it with the training widgets, or adapt the JSON to an external tracker such as
Weights & Biases.
-/

export _root_.Runtime.Training
  (Series TrainLog Curve MetricHistory ConfigEntry Artifact RunInfo ExperimentLog LogDestination)
export _root_.Runtime.Training.Curve (push toTrainLog)
export _root_.Runtime.Training.MetricHistory (empty push toTrainLog)
export _root_.Runtime.Training.ExperimentLog (init log logRow addArtifact toTrainLog)
export _root_.Runtime.Training.LogDestination (disabled json isEnabled path? writeTrainLog)
export _root_.Runtime.Training.TrainLog (writeJson readJson)

/--
A runner bundled with the task that created it.

This is an ergonomic wrapper around `Runner α task`: it remembers the dependent `task`, so user code
can call `tr.predict x`, `tr.fit cfg samples`, etc. without repeatedly writing
`(task := task)`.
-/
structure TaskRunner (σ τ : Spec.Shape) (α : Type)
    [Semantics.Scalar α] [DecidableEq Spec.Shape] where
  /-- The supervised task: model plus loss. -/
  task : Task σ τ
  /-- The instantiated runner for `task`. -/
  runner : Runner α task

namespace TaskRunner

/-- Bundle an existing runner with its task. -/
def ofRunner {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) : TaskRunner σ τ α :=
  { task := task, runner := runner }

/-- Get the current model parameters from a bundled runner. -/
def params {σ τ : Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (tr : TaskRunner σ τ α) :
    IO (TorchLean.TList α (TorchLean.Supervised.paramShapes tr.task)) :=
  train.params tr.runner

/-- Read the current mode (`.train` or `.eval`) from a bundled runner. -/
def mode {σ τ : Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (tr : TaskRunner σ τ α) : IO TorchLean.NN.Mode :=
  train.mode tr.runner

/-- Set the mode (`.train` or `.eval`) on a bundled runner. -/
def setMode {σ τ : Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (tr : TaskRunner σ τ α) (value : TorchLean.NN.Mode) : IO Unit :=
  train.setMode tr.runner value

/-- Switch a bundled runner to training mode. -/
def trainMode {σ τ : Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (tr : TaskRunner σ τ α) : IO Unit :=
  train.trainMode tr.runner

/-- Switch a bundled runner to evaluation mode. -/
def evalMode {σ τ : Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (tr : TaskRunner σ τ α) : IO Unit :=
  train.evalMode tr.runner

/-- Check whether a bundled runner is in training mode. -/
def isTraining {σ τ : Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (tr : TaskRunner σ τ α) : IO Bool :=
  train.isTraining tr.runner

/-- Predict on one input tensor using the bundled runner's active mode. -/
def predict {σ τ : Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (tr : TaskRunner σ τ α) (x : Spec.Tensor α σ) :
    IO (Spec.Tensor α τ) :=
  train.predict tr.runner x

/-- Predict on a list of inputs using the bundled runner's active mode. -/
def predictBatch {σ τ : Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (tr : TaskRunner σ τ α) (xs : List (Spec.Tensor α σ)) :
    IO (List (Spec.Tensor α τ)) :=
  train.predictBatch tr.runner xs

/-- Mean loss over an entire dataset for a bundled runner. -/
def meanLossDataset {σ τ : Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α]
    (tr : TaskRunner σ τ α) (dataset : _root_.Runtime.Autograd.Train.Dataset
      (sample.Supervised α σ τ)) :
    IO α :=
  TorchLean.Trainer.meanLossDataset (task := tr.task) tr.runner dataset

/-- Fit a bundled runner on an explicit list of samples for a fixed number of steps. -/
def fit {σ τ : Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α] [Runtime.Scalar α]
    (tr : TaskRunner σ τ α) (cfg : FitConfig) (samples : List (sample.Supervised α σ τ)) :
    IO (FitReport α) :=
  TorchLean.Trainer.fit (task := tr.task) tr.runner cfg samples

/-- Fit a bundled runner on a `Dataset` for a fixed number of steps. -/
def fitDataset {σ τ : Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α] [Runtime.Scalar α]
    (tr : TaskRunner σ τ α) (cfg : FitConfig)
    (dataset : _root_.Runtime.Autograd.Train.Dataset (sample.Supervised α σ τ)) :
    IO (FitReport α) :=
  TorchLean.Trainer.fitDataset (task := tr.task) tr.runner cfg dataset

/-- Fit a bundled runner using a `DataLoader` for a fixed number of epochs. -/
def fitLoader {σ τ : Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α] [Runtime.Scalar α]
    (tr : TaskRunner σ τ α) (cfg : LoaderFitConfig)
    (loader : _root_.Runtime.Autograd.Train.DataLoader (sample.Supervised α σ τ)) :
    IO (FitReport α × _root_.Runtime.Autograd.Train.DataLoader (sample.Supervised α σ τ)) :=
  TorchLean.Trainer.fitLoader (task := tr.task) tr.runner cfg loader

end TaskRunner

/--
CLI-oriented runner entry point that passes a bundled `TaskRunner` to the continuation.

This mirrors `train.run`, but removes the need to keep threading `(task := task)` after
instantiation.
-/
def runTask {σ τ : Spec.Shape} (task : Task σ τ) (args : List String)
    (k :
      ∀ {α : Type}, [Semantics.Scalar α] → [DecidableEq Spec.Shape] → [ToString α] →
        [Runtime.Scalar α] → TaskRunner σ τ α → List String → IO Unit) :
    IO Unit :=
  run task args (fun {α} _ _ _ _ runner rest =>
    k (α := α) { task := task, runner := runner } rest)

/--
Count correct predictions in a one-hot labeled **batched** dataset.

This is the minibatch analogue of `accuracyOneHot`: the task already has a leading dim0 batch axis,
so we score each row of the batch independently and accumulate totals.

Returns `(correct, total)` where `total = batch * numBatches`.
-/
def accuracyOneHotBatched
    {σ : Spec.Shape} {classes batch : Nat}
    {task : Task (.dim batch σ) (.dim batch (NN.Tensor.Shape.Vec classes))}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    (runner : Runner α task) (samples : List (sample.Batch α batch σ (NN.Tensor.Shape.Vec classes)))
      :
    IO (Nat × Nat) := do
  let mut correct : Nat := 0
  let mut total : Nat := 0
  for s in samples do
    let xBatch := sample.x s
    let yBatch := sample.y s
    let logitsBatch ← predict (task := task) runner xBatch
    for i in List.finRange batch do
      let logits := Spec.getAtSpec logitsBatch i
      let target := Spec.getAtSpec yBatch i
      if let some true := metrics.correctOneHot? logits target then
        correct := correct + 1
      total := total + 1
  pure (correct, total)

/-- Mean loss over an entire dataset, used by before/after training reports. -/
def meanLossDataset {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α]
    (runner : Runner α task) (dataset : _root_.Runtime.Autograd.Train.Dataset (sample.Supervised α σ
      τ)) :
    IO α :=
  TorchLean.Trainer.meanLossDataset (task := task) runner dataset

/-- Fit on an explicit list of samples for a fixed number of steps. -/
def fit {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α] [Runtime.Scalar α]
    (runner : Runner α task) (cfg : FitConfig) (samples : List (sample.Supervised α σ τ)) :
    IO (FitReport α) :=
  TorchLean.Trainer.fit (task := task) runner cfg samples

/-- Fit on a `Dataset` for a fixed number of steps. -/
def fitDataset {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α] [Runtime.Scalar α]
    (runner : Runner α task) (cfg : FitConfig) (dataset : _root_.Runtime.Autograd.Train.Dataset
      (sample.Supervised α σ τ)) :
    IO (FitReport α) :=
  TorchLean.Trainer.fitDataset (task := task) runner cfg dataset

/-- Fit using a `DataLoader` for a fixed number of epochs. -/
def fitLoader {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α] [Runtime.Scalar α]
    (runner : Runner α task) (cfg : LoaderFitConfig)
    (loader : _root_.Runtime.Autograd.Train.DataLoader (sample.Supervised α σ τ)) :
    IO (FitReport α × _root_.Runtime.Autograd.Train.DataLoader (sample.Supervised α σ τ)) :=
  TorchLean.Trainer.fitLoader (task := task) runner cfg loader

/-- Callback event fired after each training step. -/
structure StepEvent (α : Type) where
  /-- Current epoch number. -/
  epoch : Nat
  /-- Global optimizer-step counter. -/
  step : Nat
  /-- Loss reported for this step. -/
  loss : α

/-- Callback event fired at the end of an epoch (how many steps ran). -/
structure EpochEvent where
  /-- Epoch number that just completed. -/
  epoch : Nat
  /-- Number of steps executed in the epoch. -/
  steps : Nat

/--
Hooks for instrumenting `fitLoaderBatched`-style training loops.

Callbacks are ordinary `IO` hooks. They can print progress, update an in-memory curve, sample CUDA
allocator state, or forward events to a project-specific metrics backend.
-/
structure Callbacks (α : Type) where
  /-- Called once before training starts. -/
  onTrainStart : IO Unit := pure ()
  /-- Called after each training step. -/
  onStep : StepEvent α → IO Unit := fun _ => pure ()
  /-- Called after each epoch. -/
  onEpochEnd : EpochEvent → IO Unit := fun _ => pure ()
  /-- Called once after training finishes. -/
  onTrainEnd : FitReport α → IO Unit := fun _ => pure ()

namespace Callbacks

/-- No-op callbacks. -/
def empty {α : Type} : Callbacks α := {}

/-- Combine two callback collections by running them in sequence. -/
def append {α : Type} (a b : Callbacks α) : Callbacks α :=
  { onTrainStart := do
      a.onTrainStart
      b.onTrainStart
    onStep := fun ev => do
      a.onStep ev
      b.onStep ev
    onEpochEnd := fun ev => do
      a.onEpochEnd ev
      b.onEpochEnd ev
    onTrainEnd := fun report => do
      a.onTrainEnd report
      b.onTrainEnd report
  }

/-- `∅` for callbacks: a no-op callback collection. -/
instance {α : Type} : EmptyCollection (Callbacks α) where
  emptyCollection := empty

/-- `Callbacks` form a monoid under sequential composition. -/
instance {α : Type} : Append (Callbacks α) where
  append := append

end Callbacks

/-- Build callbacks that run `action` once at the start of training. -/
def onTrainStart {α : Type} (action : IO Unit) : Callbacks α :=
  { onTrainStart := action }

/-- Build callbacks that observe every training step. -/
def onStep {α : Type} (f : StepEvent α → IO Unit) : Callbacks α :=
  { onStep := f }

/--
Build a training callback that samples the CUDA allocator at a fixed step cadence.

The callback owns a small `IO.Ref` for the previous sample, so examples can compose it with ordinary
loss-logging callbacks without threading allocator state through their training loops.
-/
def cudaMemWatchCallbacks {α : Type} (opts : _root_.Runtime.Autograd.Torch.Options)
    (watchEvery totalSteps : Nat) : IO (Callbacks α) := do
  let stateRef ← IO.mkRef (none : Option Common.CudaMemWatchState)
  pure <| onStep (α := α) (fun ev => do
    let state ← stateRef.get
    let state ← Common.reportCudaMemWatch opts watchEvery totalSteps (ev.step + 1) state
    stateRef.set state)

/-- Build callbacks that run at the end of each epoch. -/
def onEpochEnd {α : Type} (f : EpochEvent → IO Unit) : Callbacks α :=
  { onEpochEnd := f }

/-- Build callbacks that run once at the end of training, with the final report. -/
def onTrainEnd {α : Type} (f : FitReport α → IO Unit) : Callbacks α :=
  { onTrainEnd := f }

/-- Callback helper: log the loss every `every` steps (if `every > 0`). -/
def logLossEvery {α : Type} [ToString α] (every : Nat := 1) : Callbacks α :=
  onStep (fun ev => do
    if every > 0 && ev.step % every = 0 then
      IO.println s!"step {ev.epoch}:{ev.step}: loss={ev.loss}")

/--
Step-indexed source of already-collated module inputs.

`Data.batchLoader` is the right interface when the data is a finite supervised dataset.  Other
training jobs draw batches from a rule or an external source: replay buffers, collocation samplers,
synthetic scale inputs, or file-backed sequence windows.  `StepBatchStream` is the lower-level
interface for those cases.

The stream is still fully typed: each produced sample is a `TList` matching the module's
`inputShapes`.  The training loop below is model-agnostic and only assumes that the module can run
`forward` and `stepWith` on those samples.
-/
structure StepBatchStream (α : Type) (inputShapes : List Spec.Shape) where
  /-- Produce the input sample used at logical optimizer step `step`. -/
  sample : Nat → IO (TorchLean.TList α inputShapes)

namespace StepBatchStream

/-- Constant stream for fixed-batch overfit runs and fixed-sample training jobs. -/
def fixed {α : Type} {inputShapes : List Spec.Shape}
    (x : TorchLean.TList α inputShapes) : StepBatchStream α inputShapes :=
  { sample := fun _ => pure x }

/-- Build a stream from a pure step-indexed sample function. -/
def ofFn {α : Type} {inputShapes : List Spec.Shape}
    (f : Nat → TorchLean.TList α inputShapes) : StepBatchStream α inputShapes :=
  { sample := fun step => pure (f step) }

/--
Cycle through a nonempty list of samples.

This adapter lets list-backed datasets use the step-stream trainer.  The explicit nonempty proof
keeps empty datasets from turning into silent modulo-by-zero behavior.
-/
def cycle {α : Type} {inputShapes : List Spec.Shape}
    (xs : List (TorchLean.TList α inputShapes)) (h : xs ≠ []) :
    StepBatchStream α inputShapes :=
  match xs with
  | [] => False.elim (h rfl)
  | x :: rest =>
      let ys := x :: rest
      { sample := fun step => pure (ys.getD (step % ys.length) x) }

end StepBatchStream

/--
Run an action with the runner temporarily switched to `value` mode.

This is useful for "evaluate on a validation set during training" in callback-based loops.
-/
def withMode {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    {β : Type} (runner : Runner α task) (value : TorchLean.NN.Mode) (action : IO β) : IO β := do
  let prev ← mode runner
  setMode runner value
  try
    action
  finally
    setMode runner prev

/--
Mean loss for an already-instantiated scalar module over a typed minibatch loader.

This is the general streaming evaluation path used by the runtime examples. It is not
CIFAR-specific: any supervised task whose loss module consumes
`[dim n σ, dim n τ]` can use the same loader.  The loader stores ordinary per-example samples
`(x : σ, y : τ)`; this helper asks `Data.epoch` for raw minibatches and calls
`Data.collateSupervised` to build one shape-typed batch at a time.

Two details are important for larger examples:

- We force `shuffle := false` for evaluation so before/after metrics are deterministic.
- We do not call `Data.BatchLoader.batchDataset`, because that would materialize every collated
  minibatch at once.  Streaming keeps the same API usable for image, sequence, and scientific ML
  examples where the batch tensors are much larger than small tabular datasets.
-/
def meanLossModuleLoader {σ τ : Spec.Shape} {n : Nat} {paramShapes : List Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α]
    (module : TorchLean.Module.ScalarModule α paramShapes [Spec.Shape.dim n σ, Spec.Shape.dim n τ])
    (loader : Data.BatchLoader α n σ τ) : IO α := do
  let evalLoader : Data.RawDataLoader (sample.Supervised α σ τ) :=
    { loader.raw with shuffle := false, dropLast := true }
  let (_dlNext, rawBatches) ←
    match Data.epoch "train.meanLossModuleLoader" evalLoader with
    | Except.ok out => pure out
    | Except.error msg => throw <| IO.userError s!"train.meanLossModuleLoader: {msg}"
  let mut total : α := 0
  let mut count : Nat := 0
  for rawBatch in rawBatches do
    let sample ← Common.orThrow "train.meanLossModuleLoader" <|
      Data.collateSupervised (α := α) (σ := σ) (τ := τ) n rawBatch
    let lossTensor ← TorchLean.Module.forward module sample
    let loss := Spec.Tensor.toScalar lossTensor
    total := total + loss
    count := count + 1
  if count = 0 then
    pure 0
  else
    pure (total / (count : α))

/--
Mean loss over a typed minibatch loader through a `train.Runner`.

This is the runner-facing wrapper around `meanLossModuleLoader`.  Use it when the example is built
around `train.run`, task modes, and the proof-facing trainer abstraction.  Use
`meanLossModuleLoader` directly when the example has already instantiated a runtime
`TorchLean.Module.ScalarModule`, which is the common fast path for CUDA examples.
-/
def meanLossBatchLoader {σ τ : Spec.Shape} {n : Nat} {task : Task (.dim n σ) (.dim n τ)}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α]
    (runner : Runner α task) (loader : Data.BatchLoader α n σ τ) : IO α :=
  meanLossModuleLoader runner.module loader

/-- One-hot accuracy over a typed minibatch loader without materializing all collated batches. -/
def accuracyOneHotBatchLoader
    {σ : Spec.Shape} {classes batch : Nat}
    {task : Task (.dim batch σ) (.dim batch (NN.Tensor.Shape.Vec classes))}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    (runner : Runner α task)
    (loader : Data.BatchLoader α batch σ (NN.Tensor.Shape.Vec classes)) :
    IO (Nat × Nat) := do
  let evalLoader : Data.RawDataLoader (sample.Supervised α σ (NN.Tensor.Shape.Vec classes)) :=
    { loader.raw with shuffle := false, dropLast := true }
  let (_dlNext, rawBatches) ←
    match Data.epoch "train.accuracyOneHotBatchLoader" evalLoader with
    | Except.ok out => pure out
    | Except.error msg => throw <| IO.userError s!"train.accuracyOneHotBatchLoader: {msg}"
  let mut correct : Nat := 0
  let mut total : Nat := 0
  for rawBatch in rawBatches do
    let sample ← Common.orThrow "train.accuracyOneHotBatchLoader" <|
      Data.collateSupervised (α := α) (σ := σ) (τ := NN.Tensor.Shape.Vec classes) batch rawBatch
    let (c, t) ← accuracyOneHotBatched (task := task) runner [sample]
    correct := correct + c
    total := total + t
  pure (correct, total)

/--
Train a runtime scalar module from a typed minibatch loader.

This is the shared "real epoch loop" for model examples that instantiate a module directly with
`TorchLean.Module.instantiateWithOptions`, including CUDA runs.  It mirrors the PyTorch structure:

1. create an optimizer state for the module parameters;
2. for each epoch, ask the general `Data.batchLoader` for shuffled raw batches;
3. collate each raw batch into a shape-typed `(xBatch, yBatch)` sample;
4. report the scalar loss through callbacks;
5. run `forward/backward/optimizer.step` through `TorchLean.Module.stepWith`.

The function is polymorphic in the input shape `σ`, target shape `τ`, batch size `n`, scalar type
`α`, parameter shapes, and optimizer. It is not an image-specific helper. CNN, ResNet, ViT, MLP,
sequence, operator-learning, and future model examples should all be able to use this path whenever
their supervised loss module has input shapes `[dim n σ, dim n τ]`.
-/
def fitModuleLoaderWith {σ τ : Spec.Shape} {n : Nat} {paramShapes : List Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α]
    (module : TorchLean.Module.ScalarModule α paramShapes [Spec.Shape.dim n σ, Spec.Shape.dim n τ])
    (optimizer : TorchLean.Optim.Optimizer α paramShapes)
    (epochs : Nat)
    (loader : Data.BatchLoader α n σ τ)
    (callbacks : Callbacks α := Callbacks.empty) :
    IO (FitReport α × Data.BatchLoader α n σ τ) := do
  let before ← meanLossModuleLoader module loader
  callbacks.onTrainStart

  let mut optState ← TorchLean.Module.initOptim module optimizer
  let mut dl := loader
  let mut globalStep : Nat := 0

  for epochIdx in [0:epochs] do
    let (rawNext, rawBatches) ←
      match Data.epoch "train.fitModuleLoaderWith" dl.raw with
      | Except.ok out => pure out
      | Except.error msg => throw <| IO.userError s!"train.fitModuleLoaderWith: {msg}"
    dl := { raw := rawNext }
    for rawBatch in rawBatches do
      let sample ← Common.orThrow "train.fitModuleLoaderWith" <|
        Data.collateSupervised (α := α) (σ := σ) (τ := τ) n rawBatch
      let lossTensor ← TorchLean.Module.forward module sample
      let loss := Spec.Tensor.toScalar lossTensor
      callbacks.onStep { epoch := epochIdx, step := globalStep, loss := loss }
      optState ← TorchLean.Module.stepWith module optimizer optState sample
      globalStep := globalStep + 1
    callbacks.onEpochEnd { epoch := epochIdx, steps := globalStep }

  let after ← meanLossModuleLoader module dl
  let report := { before := before, after := after }
  callbacks.onTrainEnd report
  pure (report, dl)

/--
Train a runtime scalar module for exactly `steps` optimizer updates.

`fitModuleLoaderWith` above is epoch-based: each unit means one full pass over the loader. This
variant is update-based, which is the convention used by runnable examples that expose a `--steps`
flag.

The loop still draws shuffled minibatches from `Data.batchLoader` epoch by epoch, but it stops as
soon as the requested number of optimizer updates has run. The returned loader is the advanced
loader state, so callers can continue training from the next shuffled epoch if they want to.
-/
def fitModuleLoaderStepsWith {σ τ : Spec.Shape} {n : Nat} {paramShapes : List Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α]
    (module : TorchLean.Module.ScalarModule α paramShapes [Spec.Shape.dim n σ, Spec.Shape.dim n τ])
    (optimizer : TorchLean.Optim.Optimizer α paramShapes)
    (steps : Nat)
    (loader : Data.BatchLoader α n σ τ)
    (callbacks : Callbacks α := Callbacks.empty) :
    IO (FitReport α × Data.BatchLoader α n σ τ) := do
  let before ← meanLossModuleLoader module loader
  callbacks.onTrainStart

  let mut optState ← TorchLean.Module.initOptim module optimizer
  let mut dl := loader
  let mut globalStep : Nat := 0
  let mut epochIdx : Nat := 0

  while globalStep < steps do
    let (rawNext, rawBatches) ←
      match Data.epoch "train.fitModuleLoaderStepsWith" dl.raw with
      | Except.ok out => pure out
      | Except.error msg => throw <| IO.userError s!"train.fitModuleLoaderStepsWith: {msg}"
    if rawBatches.isEmpty then
      throw <| IO.userError "train.fitModuleLoaderStepsWith: loader produced no batches"
    dl := { raw := rawNext }
    let epochStart := globalStep
    for rawBatch in rawBatches do
      if globalStep < steps then
        let sample ← Common.orThrow "train.fitModuleLoaderStepsWith" <|
          Data.collateSupervised (α := α) (σ := σ) (τ := τ) n rawBatch
        let lossTensor ← TorchLean.Module.forward module sample
        let loss := Spec.Tensor.toScalar lossTensor
        callbacks.onStep { epoch := epochIdx, step := globalStep, loss := loss }
        optState ← TorchLean.Module.stepWith module optimizer optState sample
        globalStep := globalStep + 1
    callbacks.onEpochEnd { epoch := epochIdx, steps := globalStep - epochStart }
    epochIdx := epochIdx + 1

  let after ← meanLossModuleLoader module dl
  let report := { before := before, after := after }
  callbacks.onTrainEnd report
  pure (report, dl)

/--
Train a scalar module from a step-indexed batch stream.

This is the shared loop for workloads whose batches are produced step by step rather than by one
finite `Data.batchLoader` epoch:

- RL algorithms can sample replay or rollout batches,
- PDE examples can resample collocation points,
- generated workloads can stream synthetic inputs without storing a dataset.

The function is generic in `inputShapes`. It does not know whether the sample is
`[x, y]`, `[state, action, target]`, or `[]`; it only asks the stream for the next typed input list
and then runs the same `forward/backward/optimizer.step` machinery as the loader-based trainer.
-/
def fitModuleStreamStepsWith {inputShapes : List Spec.Shape} {paramShapes : List Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    (module : TorchLean.Module.ScalarModule α paramShapes inputShapes)
    (optimizer : TorchLean.Optim.Optimizer α paramShapes)
    (steps : Nat)
    (stream : StepBatchStream α inputShapes)
    (callbacks : Callbacks α := Callbacks.empty) :
    IO (FitReport α) := do
  let sample0 ← stream.sample 0
  let beforeTensor ← TorchLean.Module.forward module sample0
  let before := Spec.Tensor.toScalar beforeTensor
  callbacks.onTrainStart

  let mut optState ← TorchLean.Module.initOptim module optimizer

  for step in [0:steps] do
    let sample ← stream.sample step
    let lossTensor ← TorchLean.Module.forward module sample
    let loss := Spec.Tensor.toScalar lossTensor
    callbacks.onStep { epoch := 0, step := step, loss := loss }
    optState ← TorchLean.Module.stepWith module optimizer optState sample

  callbacks.onEpochEnd { epoch := 0, steps := steps }
  let sampleAfter ← stream.sample steps
  let afterTensor ← TorchLean.Module.forward module sampleAfter
  let after := Spec.Tensor.toScalar afterTensor
  let report := { before := before, after := after }
  callbacks.onTrainEnd report
  pure report

/--
Report-oriented wrapper for `fitModuleStreamStepsWith`.

Callers use this exactly like the loader wrappers: pass the module, optimizer, runtime options, step
count, and stream, and get standard before/after reporting plus CUDA memory watching.
-/
def fitModuleStreamStepsReport {inputShapes : List Spec.Shape} {paramShapes : List Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    (module : TorchLean.Module.ScalarModule α paramShapes inputShapes)
    (optimizer : TorchLean.Optim.Optimizer α paramShapes)
    (opts : _root_.Runtime.Autograd.Torch.Options)
    (steps : Nat)
    (stream : StepBatchStream α inputShapes)
    (cudaMemWatch : Nat := 0)
    (extraCallbacks : Callbacks α := Callbacks.empty) :
    IO (FitReport α) := do
  let watchEvery := Common.effectiveCudaMemWatch opts steps cudaMemWatch
  let memHooks ← cudaMemWatchCallbacks (α := α) opts watchEvery steps
  let hooks : Callbacks α :=
    memHooks
    ++ extraCallbacks
    ++ onTrainEnd (α := α) (Common.printFitReport steps)
  fitModuleStreamStepsWith module optimizer steps stream hooks

/--
Float stream trainer that records a per-step loss curve.

Generated and file-backed batches do not always have one finite loader to summarize.  This wrapper
keeps their training curves in the same JSON format as the supervised examples.
-/
def fitModuleStreamStepsCurveFloat {inputShapes : List Spec.Shape} {paramShapes : List Spec.Shape}
    (module : TorchLean.Module.ScalarModule Float paramShapes inputShapes)
    (optimizer : TorchLean.Optim.Optimizer Float paramShapes)
    (opts : _root_.Runtime.Autograd.Torch.Options)
    (steps : Nat)
    (stream : StepBatchStream Float inputShapes)
    (cudaMemWatch : Nat := 0)
    (extraCallbacks : Callbacks Float := Callbacks.empty) :
    IO (FitReport Float × _root_.Runtime.Training.Curve) := do
  let curveRef ← IO.mkRef ({} : _root_.Runtime.Training.Curve)
  let curveHooks : Callbacks Float :=
    onStep (α := Float) (fun ev =>
      curveRef.modify (fun c => c.push ev.step ev.loss))
  let report ← fitModuleStreamStepsReport module optimizer opts steps stream cudaMemWatch
    (extraCallbacks ++ curveHooks)
  let curve ← curveRef.get
  pure (report, curve)

/--
Train from a runner-backed loader with explicit callbacks instead of inline printing in example
code.

This is the runner-facing public path for PyTorch-style custom loops:
- keep the optimizer/scheduler logic in the library,
- inject logging, evaluation, and prediction reporting through callbacks.

This path keeps the `Runner` abstraction, including task modes and scheduler support.  For
CUDA-heavy entrypoints that already have a `TorchLean.Module.ScalarModule`, prefer
`fitModuleLoaderWith`; both paths consume the same general `API.Data.batchLoader`.
-/
def fitLoaderWith {σ τ : Spec.Shape} {n : Nat} {task : Task (.dim n σ) (.dim n τ)}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α] [Runtime.Scalar α]
    (runner : Runner α task) (cfg : LoaderFitConfig)
    (loader : Data.BatchLoader α n σ τ)
    (callbacks : Callbacks α := Callbacks.empty) :
    IO (FitReport α × Data.BatchLoader α n σ τ) := do
  evalMode runner
  let before ← meanLossBatchLoader (task := task) runner loader
  callbacks.onTrainStart

  trainMode runner
  let loop ← TorchLean.Trainer.stepper (task := task) runner cfg.optimizer (scheduler :=
    cfg.scheduler)
  let mut dl := loader
  let mut globalStep : Nat := 0

  for epochIdx in [0:cfg.epochs] do
    let (rawNext, rawBatches) ←
      match Data.epoch "train.fitLoaderWith" dl.raw with
      | Except.ok out => pure out
      | Except.error msg => throw <| IO.userError s!"train.fitLoaderWith: {msg}"
    dl := { raw := rawNext }
    for rawBatch in rawBatches do
      let sample ← Common.orThrow "train.fitLoaderWith" <|
        Data.collateSupervised (α := α) (σ := σ) (τ := τ) n rawBatch
      let loss ← TorchLean.Trainer.step (task := task) loop sample
      callbacks.onStep { epoch := epochIdx, step := globalStep, loss := loss }
      globalStep := globalStep + 1
    callbacks.onEpochEnd { epoch := epochIdx, steps := globalStep }

  evalMode runner
  let after ← meanLossBatchLoader (task := task) runner dl
  let report := { before := before, after := after }
  callbacks.onTrainEnd report
  pure (report, dl)

/--
Public minibatch training path.

`data.batchLoader` produces a typed `BatchLoader` (with a type-level batch size `n`), and this
helper bridges from an untyped runtime loader into the typed training loop.
-/
def fitLoaderBatched {σ τ : Spec.Shape} {n : Nat} {task : Task (.dim n σ) (.dim n τ)}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α] [Runtime.Scalar α]
    (runner : Runner α task) (cfg : LoaderFitConfig)
    (loader : _root_.Runtime.Autograd.Train.DataLoader (sample.Supervised α σ τ)) :
    IO (FitReport α × _root_.Runtime.Autograd.Train.DataLoader (sample.Supervised α σ τ)) := do
  if loader.batchSize != n then
    throw <| IO.userError
      s!"train.fitLoaderBatched: expected loader.batchSize={n}, got {loader.batchSize}"
  if !loader.dropLast then
    throw <| IO.userError "train.fitLoaderBatched: expected loader.dropLast=true"

  -- Prefer the typed-loader implementation: it is the canonical “minibatch” training loop.
  let typed : Data.BatchLoader α n σ τ := { raw := loader }
  let callbacks :=
    if cfg.logEvery > 0 then
      logLossEvery (α := α) cfg.logEvery
    else
      Callbacks.empty
  let (report, typed') ← fitLoaderWith (runner := runner) (cfg := cfg) (loader := typed) (callbacks
    := callbacks)
  pure (report, typed'.raw)

/--
Create a `Stepper` loop for a runner and optimizer (optionally with an LR scheduler).

This corresponds to the “inner training loop” state in typical PyTorch code:
an optimizer state plus (optional) schedule state, ready to step on a batch.
-/
def stepper {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α] [Runtime.Scalar α]
    (runner : Runner α task) (optimizer : optim.Optimizer)
    (scheduler : Option TorchLean.Schedulers.Config := none) :
    IO (Stepper α task) :=
  TorchLean.Trainer.stepper (task := task) runner optimizer scheduler

/-- Run one optimization step on a single supervised sample (one batch). -/
def step {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (loop : Stepper α task) (sample : sample.Supervised α σ τ) : IO α :=
  TorchLean.Trainer.step (task := task) loop sample

/-- Run one epoch over a list of supervised samples, returning the per-step losses. -/
def epoch {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape]
    (loop : Stepper α task) (samples : List (sample.Supervised α σ τ)) : IO (List α) :=
  TorchLean.Trainer.epoch (task := task) loop samples

namespace Report

/-!
### Small Reporting Helpers (IO)

These helpers factor out common "print a loss/accuracy table" patterns for runnable model commands.
They do not affect semantics: they only call the underlying `train.*` functions and print
human-facing summaries.
-/

/-- Print a titled list of named report lines. -/
def reportProbes {β : Type} (title : String) (probes : List β) (lineOf : β → IO String) : IO Unit :=
  do
  IO.println title
  for p in probes do
    IO.println (← lineOf p)

/-- Convenience: mean loss on a dataset, printed with a label. -/
def reportMeanLoss
    {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α]
    (runner : Runner α task)
    (dataset : _root_.Runtime.Autograd.Train.Dataset (sample.Supervised α σ τ))
    (label : String) : IO Unit := do
  let loss ← TorchLean.Trainer.meanLossDataset (task := task) runner dataset
  IO.println s!"mean_loss({label}) = {loss}"

/-- Convenience: mean loss on a typed minibatch loader, streamed batch by batch. -/
def reportMeanLossLoader
    {σ τ : Spec.Shape} {batch : Nat} {task : Task (.dim batch σ) (.dim batch τ)}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α]
    (runner : Runner α task)
    (loader : Data.BatchLoader α batch σ τ)
    (label : String) : IO Unit := do
  let loss ← meanLossBatchLoader (task := task) runner loader
  IO.println s!"mean_loss({label}) = {loss}"

/--
Convenience: mean loss on a typed minibatch loader for an already-instantiated runtime module.

Use this in direct CUDA/runtime examples to avoid building a `Runner` only for logging.  The data
path is still the same public loader path: `Data.batchLoader` plus `Data.collateSupervised`.
-/
def reportMeanLossModuleLoader
    {σ τ : Spec.Shape} {batch : Nat} {paramShapes : List Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α]
    (module : TorchLean.Module.ScalarModule α paramShapes [Spec.Shape.dim batch σ, Spec.Shape.dim
      batch τ])
    (loader : Data.BatchLoader α batch σ τ)
    (label : String) : IO Unit := do
  let loss ← meanLossModuleLoader module loader
  IO.println s!"mean_loss({label}) = {loss}"

/--
Report predicted classes on a list of named inputs.

Each entry is `(name, x, expectedClass)`.
If `includeLogits := true`, also prints the raw model outputs.
-/
def reportClassProbes
    {σ : Spec.Shape} {classes : Nat} {task : Task σ (.dim classes .scalar)}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    (runner : Runner α task)
    (probes : List (String × Spec.Tensor α σ × Nat))
    (title : String := "predictions")
    (includeLogits : Bool := false) : IO Unit := do
  reportProbes title probes (fun (name, x, expected) => do
    let logits ← predict (task := task) runner x
    let pred? := metrics.argmax? logits
    let predStr :=
      match pred? with
      | some k => toString k.val
      | none => "none"
    let logitsStr :=
      if includeLogits then
        s!" logits={Spec.pretty logits}"
      else
        ""
    pure s!"  {name}: expected={expected} predicted={predStr}{logitsStr}")

/--
Report predicted classes on a list of named inputs, for a **batched** model.

This expects inputs of the *unbatched* input shape `σ` and replicates each one across the batch
axis, then reports the prediction for row 0.
-/
def reportClassProbesBatchedFromSingle
    {σ : Spec.Shape} {classes batch : Nat} {task : Task (.dim batch σ) (.dim batch
      (NN.Tensor.Shape.Vec classes))}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    (runner : Runner α task)
    (probes : List (String × Spec.Tensor α σ × Nat))
    (title : String := "predictions")
    (includeLogits : Bool := false) : IO Unit := do
  reportProbes title probes (fun (name, xSingle, expected) => do
    let xBatch : Spec.Tensor α (.dim batch σ) :=
      Spec.Tensor.dim (fun _ => xSingle)
    let logitsBatch ← predict (task := task) runner xBatch
    -- If `batch = 0`, there is no row to display. That case is not meaningful for training anyway.
    match List.finRange batch with
    | [] =>
        pure s!"  {name}: batch=0 (no prediction)"
    | i0 :: _ =>
        let logits0 := Spec.getAtSpec logitsBatch i0
        let pred? := metrics.argmax? logits0
        let predStr :=
          match pred? with
          | some k => toString k.val
          | none => "none"
        let logitsStr :=
          if includeLogits then
            s!" logits={Spec.pretty logits0}"
          else
            ""
        pure s!"  {name}: expected={expected} predicted={predStr}{logitsStr}")

/-- Convenience: mean loss + one-hot accuracy on a dataset, printed with a label. -/
def reportLossAccuracyOneHot
    {σ : Spec.Shape} {classes : Nat} {task : Task σ (.dim classes .scalar)}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    (runner : Runner α task)
    (dataset : _root_.Runtime.Autograd.Train.Dataset (sample.Supervised α σ (.dim classes .scalar)))
    (label : String) : IO Unit := do
  let loss ← TorchLean.Trainer.meanLossDataset (task := task) runner dataset
  let (correct, total) ← accuracyOneHot (task := task) runner dataset.toList
  IO.println s!"mean_loss({label}) = {loss}"
  IO.println s!"accuracy({label}) = {correct}/{total}"

/-- Batched variant of `reportLossAccuracyOneHot`. -/
def reportLossAccuracyOneHotBatched
    {σ : Spec.Shape} {classes batch : Nat}
    {task : Task (.dim batch σ) (.dim batch (NN.Tensor.Shape.Vec classes))}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    (runner : Runner α task)
    (dataset : _root_.Runtime.Autograd.Train.Dataset (sample.Batch α batch σ (NN.Tensor.Shape.Vec
      classes)))
    (label : String) : IO Unit := do
  let loss ← TorchLean.Trainer.meanLossDataset (task := task) runner dataset
  let (correct, total) ← accuracyOneHotBatched (task := task) runner dataset.toList
  IO.println s!"mean_loss({label}) = {loss}"
  IO.println s!"accuracy({label}) = {correct}/{total}"

/-- Loader variant of `reportLossAccuracyOneHotBatched`, streaming through minibatches. -/
def reportLossAccuracyOneHotLoader
    {σ : Spec.Shape} {classes batch : Nat}
    {task : Task (.dim batch σ) (.dim batch (NN.Tensor.Shape.Vec classes))}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α]
    [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    (runner : Runner α task)
    (loader : Data.BatchLoader α batch σ (NN.Tensor.Shape.Vec classes))
    (label : String) : IO Unit := do
  let loss ← meanLossBatchLoader (task := task) runner loader
  let (correct, total) ← accuracyOneHotBatchLoader (task := task) runner loader
  IO.println s!"mean_loss({label}) = {loss}"
  IO.println s!"accuracy({label}) = {correct}/{total}"

end Report

/--
Train a runtime module for a fixed number of optimizer updates with the standard runtime reports.

This is the common path for direct-module training, not an example-only helper.  It composes the
generic step loop with before/after mean-loss reporting and CUDA allocator telemetry, while still
accepting extra callbacks for projects that want their own metrics, validation, or tracing.
-/
def fitModuleLoaderStepsReport {σ τ : Spec.Shape} {n : Nat} {paramShapes : List Spec.Shape}
    {α : Type} [Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    [Add α] [Div α] [Zero α] [Coe Nat α]
    (module : TorchLean.Module.ScalarModule α paramShapes [Spec.Shape.dim n σ, Spec.Shape.dim n τ])
    (optimizer : TorchLean.Optim.Optimizer α paramShapes)
    (opts : _root_.Runtime.Autograd.Torch.Options)
    (steps : Nat)
    (loader : Data.BatchLoader α n σ τ)
    (cudaMemWatch : Nat := 0)
    (extraCallbacks : Callbacks α := Callbacks.empty) :
    IO (FitReport α × Data.BatchLoader α n σ τ) := do
  let watchEvery := Common.effectiveCudaMemWatch opts steps cudaMemWatch
  let memHooks ← cudaMemWatchCallbacks (α := α) opts watchEvery steps
  let hooks : Callbacks α :=
    (onTrainStart (α := α) do
      Report.reportMeanLossModuleLoader module loader "train(before)")
    ++ extraCallbacks
    ++ memHooks
    ++ onTrainEnd (α := α) (fun _ =>
      Report.reportMeanLossModuleLoader module loader "train(after)")
  fitModuleLoaderStepsWith module optimizer steps loader hooks

/--
Float-specialized module training that also records a scalar loss curve.

The training loop itself is the same as `fitModuleLoaderStepsReport`; this wrapper only adds a
`Curve` callback for JSON logs and website widgets.  Keeping it in `train` lets future model files
request a curve without reimplementing callback plumbing.
-/
def fitModuleLoaderStepsCurveFloat {σ τ : Spec.Shape} {n : Nat}
    {paramShapes : List Spec.Shape}
    (module : TorchLean.Module.ScalarModule Float paramShapes [Spec.Shape.dim n σ,
      Spec.Shape.dim n τ])
    (optimizer : TorchLean.Optim.Optimizer Float paramShapes)
    (opts : _root_.Runtime.Autograd.Torch.Options)
    (steps : Nat)
    (loader : Data.BatchLoader Float n σ τ)
    (cudaMemWatch : Nat := 0)
    (extraCallbacks : Callbacks Float := Callbacks.empty) :
    IO (FitReport Float × Data.BatchLoader Float n σ τ × _root_.Runtime.Training.Curve) := do
  let curveRef ← IO.mkRef ({} : _root_.Runtime.Training.Curve)
  let curveHooks : Callbacks Float :=
    onStep (α := Float) (fun ev => curveRef.modify (fun c => c.push ev.step ev.loss))
  let (report, loader') ← fitModuleLoaderStepsReport module optimizer opts steps loader
    cudaMemWatch (extraCallbacks ++ curveHooks)
  let curve ← curveRef.get
  pure (report, loader', curve)

/--
Train a Float runtime module, write a standard scalar-curve log, and return the fit report.

This is the high-level path used by runnable training commands.  The caller provides the model,
optimizer, loader, runtime options, and metadata notes; the library owns the callback composition,
CUDA telemetry, before/after reports, and JSON curve emission.
-/
def fitModuleLoaderStepsLoggedFloat {σ τ : Spec.Shape} {n : Nat}
    {paramShapes : List Spec.Shape}
    (module : TorchLean.Module.ScalarModule Float paramShapes [Spec.Shape.dim n σ,
      Spec.Shape.dim n τ])
    (optimizer : TorchLean.Optim.Optimizer Float paramShapes)
    (opts : _root_.Runtime.Autograd.Torch.Options)
    (steps : Nat)
    (loader : Data.BatchLoader Float n σ τ)
    (log : _root_.Runtime.Training.LogDestination)
    (title : String)
    (notes : Array String := #[])
    (seriesName : String := "loss")
    (cudaMemWatch : Nat := 0) :
    IO (FitReport Float × Data.BatchLoader Float n σ τ) := do
  let (report, loader', curve) ← fitModuleLoaderStepsCurveFloat module optimizer opts steps loader
    cudaMemWatch
  Common.writeCurveLogTo log title curve seriesName notes
  pure (report, loader')

end train

end API
end NN
