/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Runtime.Training.Stepper

import Mathlib.Algebra.Order.Algebra
import NN.Spec.Autograd.AutogradSpec

@[expose] public section

namespace NN
namespace API
namespace TorchLean

namespace Trainer

/-!
`NN.API.TorchLean.Trainer` is the stable public namespace for supervised training machinery.

Usage note:

If you are writing ordinary model code, prefer the `NN` umbrella:
`import NN; open TorchLean; Trainer.new ...`. This namespace is the lower runtime
layer underneath that API, and it exposes lower-level controls for custom training loops.

The intended workflow is:
- pick a `Task` (regression / classification),
- call `instantiate` to get a `Runner` (parameters + buffers + backend state),
- call `trainSamples` / `trainDataset` / `trainLoader`, or build a `Stepper` for custom loops.

This API is backend-agnostic: the same code can run in `.eager` mode or via a compiled backend,
depending on the `backend` argument passed to `instantiate`.
-/

/-- A supervised sequence-model task pairing a model with its loss and gradient computation. -/
abbrev Task := Supervised.SeqTask
/-- Instantiated training state containing parameters, buffers, and the selected execution backend. -/
abbrev Runner := Supervised.Runner
/-- Optimizer selection and hyperparameters accepted by the supervised trainer. -/
abbrev Optimizer := Supervised.OptimizerConfig
/-- Shared epoch, batching, logging, and optimizer settings for in-memory training. -/
abbrev TrainConfig := Supervised.TrainConfig
/-- Mutable CUDA-memory observation state used to report allocator growth during training. -/
abbrev CudaMemWatchState := Supervised.CudaMemWatchState
/-- Training configuration for streaming batches from a data loader. -/
abbrev LoaderTrainConfig := Supervised.LoaderTrainConfig
/-- Aggregate losses, timings, and step counts returned by a completed training run. -/
abbrev TrainReport := Supervised.TrainReport
/-- Stateful single-batch training interface for applications that own the outer loop. -/
abbrev Stepper := Supervised.Stepper

/-- Resolve whether CUDA memory monitoring is enabled after applying configuration defaults. -/
abbrev effectiveCudaMemWatch := Supervised.effectiveCudaMemWatch

/-- Sample CUDA memory use, update the watch state, and emit a report when its policy requests one. -/
abbrev reportCudaMemWatch := Supervised.reportCudaMemWatch

/-- Lower-runtime regression task with mean-squared error loss by default. -/
def runtimeRegressionTask {σ τ : Spec.Shape} (model : API.TorchLean.LayerCore.Seq σ τ)
    (reduction : API.TorchLean.Loss.Reduction := .mean) :
    Task σ τ :=
  Supervised.SeqTask.mse model reduction

/-- Lower-runtime classifier task with cross-entropy loss by default. -/
def runtimeClassifierTask {σ τ : Spec.Shape} (model : API.TorchLean.LayerCore.Seq σ τ)
    (reduction : API.TorchLean.Loss.Reduction := .mean) :
    Task σ τ :=
  Supervised.SeqTask.crossEntropyOneHot model reduction

/--
SGD optimizer config.

PyTorch analogue: `torch.optim.SGD`
  (`https://pytorch.org/docs/stable/generated/torch.optim.SGD.html`).
-/
def sgd (lr : Float) (momentum : Float := 0.0) : Optimizer := .sgd lr momentum

/-- Momentum SGD optimizer config. Pass `lr` and optionally override `momentum`. -/
def momentumSGD (lr : Float) (momentum : Float := 0.9) : Optimizer := .sgd lr momentum

/--
AdaGrad optimizer config. Pass `lr`; override `epsilon` when the run needs it.

PyTorch analogue: `torch.optim.Adagrad`
  (`https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html`).
-/
def adagrad (lr : Float) (epsilon : Float := 1e-10) : Optimizer := .adagrad lr epsilon

/--
RMSProp optimizer config. Pass `lr`; override `decay` or `epsilon` when the run needs it.

PyTorch analogue: `torch.optim.RMSprop`
  (`https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html`).
-/
def rmsprop (lr : Float) (decay : Float := 0.99) (epsilon : Float := 1e-8) : Optimizer :=
  .rmsprop lr decay epsilon

/--
Adam optimizer config. Pass `lr`; override `beta1`, `beta2`, or `epsilon` when the run needs it.

PyTorch analogue: `torch.optim.Adam`
  (`https://pytorch.org/docs/stable/generated/torch.optim.Adam.html`).
-/
def adam (lr : Float) (beta1 : Float := 0.9) (beta2 : Float := 0.999) (epsilon : Float := 1e-8) :
    Optimizer :=
  .adam lr beta1 beta2 epsilon

/--
AdamW optimizer config. Pass `lr`; override weight decay or moment parameters when the run needs it.

PyTorch analogue: `torch.optim.AdamW`
  (`https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html`).
-/
def adamw (lr : Float) (weightDecay : Float := 0.01)
    (beta1 : Float := 0.9) (beta2 : Float := 0.999) (epsilon : Float := 1e-8) :
    Optimizer :=
  .adamw lr weightDecay beta1 beta2 epsilon

/--
Adadelta optimizer config. Override `lr`, `rho`, or `epsilon` when the run needs it.

PyTorch analogue: `torch.optim.Adadelta`
  (`https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html`).
-/
def adadelta (lr : Float := 1.0) (rho : Float := 0.9) (epsilon : Float := 1e-6) : Optimizer :=
  .adadelta lr rho epsilon

/-- Optimizer algorithm accepted by simple CLI commands that expose a `--optim` flag. -/
inductive OptimizerKind where
  /-- Stochastic gradient descent. -/
  | sgd
  /-- AdaGrad with TorchLean's usual public defaults. -/
  | adagrad
  /-- RMSProp with TorchLean's usual public defaults. -/
  | rmsprop
  /-- Adam with TorchLean's usual public defaults. -/
  | adam
  /-- AdamW with decoupled weight decay. -/
  | adamw
  /-- Adadelta with TorchLean's usual public defaults. -/
  | adadelta
deriving DecidableEq, Repr

namespace OptimizerKind

/-- Parse an optimizer name accepted by a command-line `--optim` flag. -/
def parse (s : String) : Except String OptimizerKind :=
  if s == "sgd" then
    pure .sgd
  else if s == "adagrad" then
    pure .adagrad
  else if s == "rmsprop" then
    pure .rmsprop
  else if s == "adam" then
    pure .adam
  else if s == "adamw" then
    pure .adamw
  else if s == "adadelta" then
    pure .adadelta
  else
    throw s!"bad --optim {s}; expected sgd, adagrad, rmsprop, adam, adamw, or adadelta"

/-- Human-readable optimizer name used in logs. -/
def name : OptimizerKind → String
  | .sgd => "SGD"
  | .adagrad => "AdaGrad"
  | .rmsprop => "RMSProp"
  | .adam => "Adam"
  | .adamw => "AdamW"
  | .adadelta => "Adadelta"

/-- Build a public optimizer config for this optimizer kind and learning rate. -/
def toOptimizer (kind : OptimizerKind) (lr : Float) : Optimizer :=
  match kind with
  | .sgd => Trainer.sgd lr
  | .adagrad => Trainer.adagrad (lr := lr)
  | .rmsprop => Trainer.rmsprop (lr := lr)
  | .adam => Trainer.adam (lr := lr) (beta1 := 0.9) (beta2 := 0.95) (epsilon := 1e-8)
  | .adamw =>
      Trainer.adamw (lr := lr) (weightDecay := 0.1) (beta1 := 0.9) (beta2 := 0.95)
        (epsilon := 1e-8)
  | .adadelta => Trainer.adadelta (lr := lr)

end OptimizerKind

/-- Fixed-step training config over an in-memory sample list or dataset. -/
def steps (count : Nat) (optimizer : Optimizer := sgd 0.01) (logEvery : Nat := 1) : TrainConfig :=
  { steps := count, optimizer := optimizer, logEvery := logEvery }

/-- Epoch-based training config over a data loader. -/
def epochs (count : Nat) (optimizer : Optimizer := sgd 0.01) (logEvery : Nat := 1) :
    LoaderTrainConfig :=
  { epochs := count, optimizer := optimizer, logEvery := logEvery }

/-- Attach a scheduler to a step-based training config. -/
def withScheduler (cfg : TrainConfig) (scheduler : API.TorchLean.Schedulers.Config) : TrainConfig :=
  { cfg with scheduler := some scheduler }

/-- Attach a scheduler to an epoch-based loader training config. -/
def withEpochScheduler (cfg : LoaderTrainConfig) (scheduler : API.TorchLean.Schedulers.Config) :
    LoaderTrainConfig :=
  { cfg with scheduler := some scheduler }

/-- Step-based constant learning-rate schedule. -/
def constantLR (cfg : TrainConfig) (lr : Float) : TrainConfig :=
  withScheduler cfg (.constant lr)

/-- Step-based step-decay schedule. -/
def stepLR (cfg : TrainConfig) (base : Float) (stepSize : Nat) (gamma : Float := 0.1) : TrainConfig :=
  withScheduler cfg (.step base stepSize gamma)

/-- Step-based exponential-decay schedule. -/
def exponentialLR (cfg : TrainConfig) (base : Float) (gamma : Float) : TrainConfig :=
  withScheduler cfg (.exponential base gamma)

/-- Epoch-based constant learning-rate schedule. -/
def constantEpochLR (cfg : LoaderTrainConfig) (lr : Float) : LoaderTrainConfig :=
  withEpochScheduler cfg (.constant lr)

/-- Epoch-based step-decay schedule. -/
def stepEpochLR (cfg : LoaderTrainConfig) (base : Float) (stepSize : Nat) (gamma : Float := 0.1) :
    LoaderTrainConfig :=
  withEpochScheduler cfg (.step base stepSize gamma)

/-- Epoch-based exponential-decay schedule. -/
def exponentialEpochLR (cfg : LoaderTrainConfig) (base : Float) (gamma : Float) : LoaderTrainConfig :=
  withEpochScheduler cfg (.exponential base gamma)

/--
Instantiate a runner under explicit Torch options such as `backend` and `device`.

This is the recommended entrypoint when you want CUDA eager execution from the training helpers
without dropping down to `TorchLean.Module` directly.
-/
def instantiateConfigured {σ τ : Spec.Shape} (task : Task σ τ)
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [API.Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (opts : API.TorchLean.Options := {}) :
    IO (Runner α task) :=
  Supervised.instantiateWithRuntimeOptions (task := task) (α := α) opts

/-- Instantiate a `Float` runner with storage-first parameter initialization when available. -/
def instantiateConfiguredFloat {σ τ : Spec.Shape} (task : Task σ τ)
    (opts : API.TorchLean.Options := {}) : IO (Runner Float task) :=
  Supervised.instantiateConfiguredFloat task opts

/--
Instantiate a runner (parameters + buffers + backend state) for the given task.

This allocates and initializes model parameters (via `Seq.initParams`) and sets up the chosen
execution backend (`.eager` vs `.compiled`).
-/
def instantiate {σ τ : Spec.Shape} (task : Task σ τ)
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [API.Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (backend : API.TorchLean.Backend := .eager) :
    IO (Runner α task) :=
  instantiateConfigured (task := task) (α := α) { backend := backend }

/--
CLI-oriented runner entry point.

This parses dtype/backend flags (via `NN.API.DType` / `Module.ExecConfig`) and then calls the
continuation `k` under the selected scalar backend.
-/
def run {σ τ : Spec.Shape} (task : Task σ τ) (args : List String)
    (k :
      ∀ {α : Type}, [API.Semantics.Scalar α] → [DecidableEq Spec.Shape] → [ToString α] →
        [API.Runtime.Scalar α] → Runner α task → List String → IO Unit) :
    IO Unit :=
  Supervised.run task args k

/-- Get the current model parameters from a runner. -/
def params {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) :
    IO (API.TorchLean.TensorPack α (Supervised.paramShapes task)) :=
  Supervised.params runner

/-- Read the current mode (train vs eval). -/
def mode {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) : IO API.TorchLean.LayerCore.Mode :=
  Supervised.mode runner

/-- Set the mode (train vs eval). -/
def setMode {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (value : API.TorchLean.LayerCore.Mode) : IO Unit :=
  Supervised.setMode runner value

/-- Put the runner in training mode so layers such as dropout/batchnorm use training behavior. -/
def trainMode {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) : IO Unit :=
  Supervised.trainMode runner

/-- Put the runner in evaluation mode so stateful layers use inference behavior. -/
def evalMode {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) : IO Unit :=
  Supervised.evalMode runner

/-- Check whether the runner is in training mode. -/
def isTraining {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) : IO Bool :=
  Supervised.isTraining runner

/-- Run forward+backward on one supervised sample and return gradients for all parameters. -/
def backward {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (sample : API.TorchLean.TensorPack α [σ, τ]) :
    IO (API.TorchLean.TensorPack α (Supervised.paramShapes task)) :=
  Supervised.backward runner sample

/--
Predict on a single input tensor.

This runs the forward pass under the runner's current mode.
-/
def predict {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (x : Spec.Tensor α σ) :
    IO (Spec.Tensor α τ) :=
  Supervised.predict runner x

/-- Predict on a list of inputs (runs the forward pass repeatedly). -/
def predictBatch {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (xs : List (Spec.Tensor α σ)) :
    IO (List (Spec.Tensor α τ)) :=
  Supervised.predictBatch runner xs

/--
Predict the argmax class for a classification task, if `argmax` is well-defined for `α`.

It runs `predict` and then applies `Metrics.argmax?` to the output vector.
-/
def predictClass? {σ : Spec.Shape} {n : Nat} {task : Task σ (.dim n .scalar)}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (x : Spec.Tensor α σ) :
    IO (Option (Fin n)) :=
  Supervised.predictClass? (task := task) runner x

/-- Count correct predictions in a one-hot labeled sample list (returns `(correct, total)`). -/
def accuracyOneHot {σ : Spec.Shape} {n : Nat} {task : Task σ (.dim n .scalar)}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (samples : List (API.TorchLean.TensorPack α [σ, .dim n .scalar])) :
    IO (Nat × Nat) :=
  Supervised.accuracyOneHot (task := task) runner samples

/-- Mean loss over an explicit list of samples. -/
def meanLoss {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    (runner : Runner α task) (samples : List (API.TorchLean.TensorPack α [σ, τ])) :
    IO α :=
  Supervised.meanLoss runner samples

/-- Mean loss over an entire `Dataset`. -/
def meanLossDataset {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    (runner : Runner α task)
    (dataset : _root_.Runtime.Autograd.Train.Dataset (API.TorchLean.TensorPack α [σ, τ])) :
    IO α :=
  Supervised.meanLossDataset runner dataset

/--
Train on an explicit list of samples for a fixed number of steps.

Returns a small report with mean loss before/after.
-/
def trainSamples {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α] [API.Runtime.Scalar α]
    (runner : Runner α task) (cfg : TrainConfig)
    (samples : List (API.TorchLean.TensorPack α [σ, τ])) :
    IO (TrainReport α) :=
  Supervised.trainSamples runner cfg samples

/-- Train on a `Dataset` for a fixed number of steps. -/
def trainDataset {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α] [API.Runtime.Scalar α]
    (runner : Runner α task) (cfg : TrainConfig)
    (dataset : _root_.Runtime.Autograd.Train.Dataset (API.TorchLean.TensorPack α [σ, τ])) :
    IO (TrainReport α) :=
  Supervised.trainDataset runner cfg dataset

/-- Train using a `DataLoader` for a fixed number of epochs. -/
def trainLoader {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α] [API.Runtime.Scalar α]
    (runner : Runner α task) (cfg : LoaderTrainConfig)
    (loader : _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ])) :
    IO (TrainReport α ×
      _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ])) :=
  Supervised.trainLoader runner cfg loader

/--
Construct a stateful stepper for custom loops.

This is useful if you want to control:
- evaluation cadence,
- logging,
- validation, early stopping, etc.

The returned `Stepper` still uses TorchLean's optimizer/scheduler implementations.
-/
def stepper {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α] [API.Runtime.Scalar α]
    (runner : Runner α task) (optimizer : Optimizer)
    (scheduler : Option API.TorchLean.Schedulers.Config := none) :
    IO (Stepper α task) :=
  Supervised.stepper runner optimizer scheduler

/-- Run a single training step on one sample using a `Stepper`. -/
def step {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (loop : Stepper α task) (sample : API.TorchLean.TensorPack α [σ, τ]) : IO α :=
  Supervised.step loop sample

/-- Run an epoch over a list of samples using a `Stepper` (returns the per-step losses). -/
def epoch {σ τ : Spec.Shape} {task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (loop : Stepper α task) (samples : List (API.TorchLean.TensorPack α [σ, τ])) : IO (List α) :=
  Supervised.epoch loop samples

/--
Convenience: instantiate + train on a list of samples.

Returns both the `Runner` (so you can keep using the trained parameters) and the train report.
-/
def train {σ τ : Spec.Shape} {_task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α] [API.Runtime.Scalar α]
    (task : Task σ τ) (cfg : TrainConfig)
    (samples : List (API.TorchLean.TensorPack α [σ, τ]))
    (backend : API.TorchLean.Backend := .eager) :
    IO (Runner α task × TrainReport α) := do
  let runner ← instantiate (task := task) (α := α) backend
  let report ← trainSamples runner cfg samples
  pure (runner, report)

/--
Convenience: instantiate a task and train on a `Dataset`.

Returns both the `Runner` and the train report.
-/
def trainDatasetTask {σ τ : Spec.Shape} {_task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α] [API.Runtime.Scalar α]
    (task : Task σ τ) (cfg : TrainConfig)
    (dataset : _root_.Runtime.Autograd.Train.Dataset (API.TorchLean.TensorPack α [σ, τ]))
    (backend : API.TorchLean.Backend := .eager) :
    IO (Runner α task × TrainReport α) := do
  let runner ← instantiate (task := task) (α := α) backend
  let report ← trainDataset runner cfg dataset
  pure (runner, report)

/--
Convenience: instantiate a task and train using a `DataLoader`.

Returns the `Runner`, the train report, and the updated loader state (shuffled epoch cursor).
-/
def trainLoaderTask {σ τ : Spec.Shape} {_task : Task σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α] [API.Runtime.Scalar α]
    (task : Task σ τ) (cfg : LoaderTrainConfig)
    (loader : _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ]))
    (backend : API.TorchLean.Backend := .eager) :
    IO (Runner α task × TrainReport α ×
      _root_.Runtime.Autograd.Train.DataLoader (API.TorchLean.TensorPack α [σ, τ])) := do
  let runner ← instantiate (task := task) (α := α) backend
  let (report, loader') ← trainLoader runner cfg loader
  pure (runner, report, loader')

end Trainer

end TorchLean
end API
end NN
