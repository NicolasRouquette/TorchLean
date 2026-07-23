/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Core
public import NN.API.Rand
public import NN.API.TorchLean.ParamIO
public import NN.API.TorchLean.Schedulers
public import NN.GraphSpec.Models.TorchLean
public import NN.Runtime.Autograd.TorchLean
public import NN.Runtime.Autograd.Train.Dataset
public import NN.Runtime.RL
public import NN.Spec.RL.MDP
public import NN.Spec.RL.MarkovMDP
public import NN.Spec.RL.FiniteStochasticMDP
public import NN.API.Runtime.Core
public import NN.API.Runtime.Layers
public import NN.API.Runtime.Autograd
public import NN.API.Runtime.Module

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

/-
Supervised training helpers built directly on `ScalarModule`.

This is a slightly lower-level layer than `NN.API.Public.train`: it is designed around a
`SeqTask σ τ` (model + loss) and produces a `Runner` + `Stepper` that can be used in scripts.
-/

/-- Built-in loss choices for `SeqTask`. -/
inductive SeqLoss where
  | mse (reduction : API.TorchLean.Loss.Reduction := .mean)
  | crossEntropyOneHot (reduction : API.TorchLean.Loss.Reduction := .mean)

/-- A supervised task is just a model plus a choice of loss. -/
structure SeqTask (σ τ : Spec.Shape) where
  /-- Model to run. -/
  model : API.TorchLean.LayerCore.Seq σ τ
  /-- Loss function. -/
  loss : SeqLoss

/--
Build a `ScalarModuleDef` for a task, choosing an explicit model mode (train/eval).

This is the underlying "instantiate me as a runnable module" step for training.
-/
def SeqTask.moduleDefWithMode {σ τ : Spec.Shape} (task : SeqTask σ τ)
    (mode : API.TorchLean.LayerCore.Mode) :
    API.TorchLean.Module.ScalarModuleDef (API.TorchLean.LayerCore.Seq.paramShapes task.model) [σ, τ] :=
  match task.loss with
  | .mse reduction =>
      API.TorchLean.LayerCore.Seq.mseScalarModuleDefWithMode mode (model := task.model) (reduction :=
        reduction)
  | .crossEntropyOneHot reduction =>
      API.TorchLean.LayerCore.Seq.crossEntropyOneHotScalarModuleDefWithMode mode
        (model := task.model) (reduction := reduction)

/-- Default module definition for a task (training mode). -/
def SeqTask.moduleDef {σ τ : Spec.Shape} (task : SeqTask σ τ) :
    API.TorchLean.Module.ScalarModuleDef (API.TorchLean.LayerCore.Seq.paramShapes task.model) [σ, τ] :=
  task.moduleDefWithMode .train

namespace SeqTask

/-- Constructor: regression task (MSE loss). -/
def mse {σ τ : Spec.Shape} (model : API.TorchLean.LayerCore.Seq σ τ)
    (reduction : API.TorchLean.Loss.Reduction := .mean) :
    SeqTask σ τ :=
  { model := model, loss := .mse reduction }

/-- Constructor: one-hot classification task (cross-entropy loss). -/
def crossEntropyOneHot {σ τ : Spec.Shape} (model : API.TorchLean.LayerCore.Seq σ τ)
    (reduction : API.TorchLean.Loss.Reduction := .mean) :
    SeqTask σ τ :=
  { model := model, loss := .crossEntropyOneHot reduction }

end SeqTask

/-- Parameter shapes for a task (delegates to `Seq.paramShapes`). -/
abbrev paramShapes {σ τ : Spec.Shape} (task : SeqTask σ τ) : List Spec.Shape :=
  API.TorchLean.LayerCore.Seq.paramShapes task.model

/--
Optimizer hyperparameter configuration for the supervised training helpers.

This configuration covers the optimizer choices exposed by the public training helpers. It mirrors
a few common PyTorch optimizers by name/defaults, but it does not try to cover the full option surface of
  `torch.optim.*`.
-/
inductive OptimizerConfig where
  /--
  SGD optimizer config.

  PyTorch analogy: `torch.optim.SGD(..., lr=..., momentum=...)` when `momentum > 0`,
  and plain SGD when `momentum = 0`.
  -/
  | sgd (lr : Float) (momentum : Float := 0.0)
  /-- AdaGrad optimizer config. -/
  | adagrad (lr : Float) (epsilon : Float := 1e-10)
  /-- RMSProp optimizer config. -/
  | rmsprop (lr : Float) (decay : Float := 0.99) (epsilon : Float := 1e-8)
  /-- Adam optimizer config. -/
  | adam (lr : Float) (beta1 : Float := 0.9) (beta2 : Float := 0.999) (epsilon : Float := 1e-8)
  /-- AdamW optimizer config (decoupled weight decay). -/
  | adamw (lr : Float) (weightDecay : Float := 0.01)
      (beta1 : Float := 0.9) (beta2 : Float := 0.999) (epsilon : Float := 1e-8)
  /-- Adadelta optimizer config. -/
  | adadelta (lr : Float := 1.0) (rho : Float := 0.9) (epsilon : Float := 1e-6)
  deriving Repr

/--
Step-based training configuration for `trainSamples` / `trainDataset`.

Fields:
- `steps`: number of parameter updates,
- `batchSize`: number of samples consumed by one public step for in-memory datasets,
- `optimizer`: optimizer hyperparameters,
- `scheduler`: optional learning-rate schedule (applied per step),
- `logEvery`: progress printing frequency (`0` disables logging).
-/
structure TrainConfig where
  /-- Number of training steps. -/
  steps : Nat
  /--
  Number of samples consumed by one public training step.

  The in-memory supervised loop applies one optimizer update per sample in the group, while
  reporting/logging at the outer step cadence. Loader-backed training uses the loader batches
  directly.
  -/
  batchSize : Nat := 1
  /-- Optimizer configuration. -/
  optimizer : OptimizerConfig := .sgd 0.01
  /-- Scheduler configuration. -/
  scheduler : Option API.TorchLean.Schedulers.Config := none
  /-- Log once every this many steps. -/
  logEvery : Nat := 1
  /-- Sample CUDA allocator state every this many completed steps; `0` disables sampling. -/
  cudaMemWatch : Nat := 0
  deriving Repr

/-- State carried by the CUDA-memory drift detector used by sustained training runs. -/
structure CudaMemWatchState where
  firstStep : Nat
  firstFreeBytes : Nat
  warned : Bool
deriving Repr

/-- Resolve an explicit CUDA-memory cadence, or enable periodic sampling for very long runs. -/
def effectiveCudaMemWatch (opts : _root_.Runtime.Autograd.Torch.Options)
    (steps requested : Nat) : Nat :=
  if requested != 0 then
    requested
  else if opts.usesCuda && steps >= 1000 then
    Nat.max 1 (steps / 10)
  else
    0

/--
Sample the CUDA allocator and warn when sustained free-memory loss projects exhaustion before the
requested run completes.
-/
def reportCudaMemWatch (opts : _root_.Runtime.Autograd.Torch.Options)
    (watchEvery totalSteps done : Nat) (state? : Option CudaMemWatchState) :
    IO (Option CudaMemWatchState) := do
  if !opts.usesCuda || watchEvery = 0 || (done != 0 && done % watchEvery != 0) then
    pure state?
  else
    let stats ← _root_.Runtime.Autograd.Cuda.Buffer.allocatorStatsWithToken (UInt32.ofNat done)
    IO.println s!"  cuda_mem step={done}: {stats.format}"
    let freeNow := stats.deviceFreeBytes.toNat
    match state? with
    | none =>
        pure (some { firstStep := done, firstFreeBytes := freeNow, warned := false })
    | some st =>
        if st.warned || done <= st.firstStep || st.firstFreeBytes <= freeNow then
          pure (some st)
        else
          let span := done - st.firstStep
          let drop := st.firstFreeBytes - freeNow
          let dropPerStep := drop / Nat.max 1 span
          if dropPerStep = 0 then
            pure (some st)
          else
            let projectedFailure := done + freeNow / dropPerStep
            if projectedFailure < totalSteps then
              IO.println <|
                s!"  cuda_mem warning: free device memory is dropping by ~{dropPerStep} bytes/step; " ++
                  s!"projected allocation failure before requested step count (around step {projectedFailure})."
              pure (some { st with warned := true })
            else
              pure (some st)

/--
Small summary returned by lower training helpers.

By default, `before` and `after` are mean loss values, but the type is polymorphic so callers can
report other scalars in the same shape.
-/
structure TrainReport (α : Type) where
  /-- Metrics before training. -/
  before : α
  /-- Metrics after training. -/
  after : α

/--
Epoch-based training configuration for `trainLoader` (data-loader training).

Fields:
- `epochs`: number of epochs (each epoch iterates once over the loader),
- `optimizer`: optimizer hyperparameters,
- `scheduler`: optional learning-rate schedule (applied per step/epoch depending on helper),
- `logEvery`: progress printing frequency (`0` disables logging).
-/
structure LoaderTrainConfig where
  /-- Number of epochs to train for. -/
  epochs : Nat
  /-- Optimizer configuration. -/
  optimizer : OptimizerConfig := .sgd 0.01
  /-- Scheduler configuration. -/
  scheduler : Option API.TorchLean.Schedulers.Config := none
  /-- Log once every this many steps. -/
  logEvery : Nat := 1
  deriving Repr

/-- Extract the base learning rate encoded in an optimizer configuration. -/
def optimizerLR : OptimizerConfig → Float
  | .sgd lr _ => lr
  | .adagrad lr _ => lr
  | .rmsprop lr _ _ => lr
  | .adam lr _ _ _ => lr
  | .adamw lr _ _ _ _ => lr
  | .adadelta lr _ _ => lr

/--
Resolve the learning rate to use at a given training step.

If a scheduler is present, it takes precedence over the optimizer's baked-in base learning rate.
Otherwise it returns `optimizerLR cfg`.
-/
def stepLR (scheduler : Option API.TorchLean.Schedulers.Config) (cfg : OptimizerConfig)
    (step : Nat) : Float :=
  match scheduler with
  | some sched => API.TorchLean.Schedulers.lrAt sched step
  | none => optimizerLR cfg

/-- Map a state update over every optimizer-state entry in a shape-indexed parameter list. -/
def mapStateList {State : Type → Spec.Shape → Type} {α : Type} :
    {ss : List Spec.Shape} →
    ({s : Spec.Shape} → State α s → State α s) →
    API.TorchLean.Optim.StateList State α ss →
    API.TorchLean.Optim.StateList State α ss
  | [], _, .nil => .nil
  | _ :: ss, f, .cons st rest => .cons (f st) (mapStateList (ss := ss) f rest)

/-- Set the learning rate field of every Adam optimizer state entry to `lr`. -/
def adamStateWithLR {α : Type} (lr : α) {paramShapes : List Spec.Shape} :
    API.TorchLean.Optim.StateList _root_.Optim.Adam.State α paramShapes →
    API.TorchLean.Optim.StateList _root_.Optim.Adam.State α paramShapes :=
  mapStateList (ss := paramShapes) (fun st => { st with lr := lr })

/-- Set the learning rate field of every momentum-SGD optimizer state entry to `lr`. -/
def momentumSGDStateWithLR {α : Type} (lr : α) {paramShapes : List Spec.Shape} :
    API.TorchLean.Optim.StateList _root_.Optim.MomentumSGD.State α paramShapes →
    API.TorchLean.Optim.StateList _root_.Optim.MomentumSGD.State α paramShapes :=
  mapStateList (ss := paramShapes) (fun st => { st with lr := lr })

/-- Set the learning rate field of every AdaGrad optimizer state entry to `lr`. -/
def adagradStateWithLR {α : Type} (lr : α) {paramShapes : List Spec.Shape} :
    API.TorchLean.Optim.StateList _root_.Optim.AdaGrad.State α paramShapes →
    API.TorchLean.Optim.StateList _root_.Optim.AdaGrad.State α paramShapes :=
  mapStateList (ss := paramShapes) (fun st => { st with lr := lr })

/-- Set the learning rate field of every RMSProp optimizer state entry to `lr`. -/
def rmspropStateWithLR {α : Type} (lr : α) {paramShapes : List Spec.Shape} :
    API.TorchLean.Optim.StateList _root_.Optim.RMSProp.State α paramShapes →
    API.TorchLean.Optim.StateList _root_.Optim.RMSProp.State α paramShapes :=
  mapStateList (ss := paramShapes) (fun st => { st with lr := lr })

/-- Set the learning rate field of every AdamW optimizer state entry to `lr`. -/
def adamwStateWithLR {α : Type} (lr : α) {paramShapes : List Spec.Shape} :
    API.TorchLean.Optim.StateList _root_.Optim.AdamW.State α paramShapes →
    API.TorchLean.Optim.StateList _root_.Optim.AdamW.State α paramShapes :=
  mapStateList (ss := paramShapes) (fun st => { st with lr := lr })

/-- Set the learning rate field of every Adadelta optimizer state entry to `lr`. -/
def adadeltaStateWithLR {α : Type} (lr : α) {paramShapes : List Spec.Shape} :
    API.TorchLean.Optim.StateList _root_.Optim.Adadelta.State α paramShapes →
    API.TorchLean.Optim.StateList _root_.Optim.Adadelta.State α paramShapes :=
  mapStateList (ss := paramShapes) (fun st => { st with lr := lr })

/--
A fully instantiated supervised task runner.

This bundles:
- the imperative `ScalarModule` (parameters/buffers stored in refs),
- compiled forward artifacts and loss functions for both `.train` and `.eval` modes (so switching mode is
  low-overhead),
- and the current mode stored in an `IO.Ref`.

The mode influences both operator behavior (e.g. dropout/batchnorm) and whether buffers are updated
during training.
-/
structure Runner (α : Type) [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    {σ τ : Spec.Shape} (task : SeqTask σ τ) where
  /-- Instantiated scalar module storing parameters/buffers in mutable refs. -/
  module : API.TorchLean.Module.ScalarModule α (paramShapes task) [σ, τ]
  /-- Compiled forward predictor specialized to training-mode behavior. -/
  predictorTrain : _root_.Runtime.Autograd.Torch.CompiledGraph α (paramShapes task ++ [σ]) τ
  /-- Compiled forward predictor specialized to eval-mode behavior. -/
  predictorEval : _root_.Runtime.Autograd.Torch.CompiledGraph α (paramShapes task ++ [σ]) τ
  /-- Compiled loss function for training-mode behavior. -/
  lossTrain : _root_.Runtime.Autograd.Torch.CompiledScalar α (paramShapes task ++ [σ, τ])
  /-- Compiled loss function for eval-mode behavior. -/
  lossEval : _root_.Runtime.Autograd.Torch.CompiledScalar α (paramShapes task ++ [σ, τ])
  /-- Mutable mode flag (`.train` / `.eval`) used by stateful layers (e.g. dropout/batchnorm). -/
  mode : IO.Ref API.TorchLean.LayerCore.Mode

/-- Finish runner construction once parameter storage has been instantiated. -/
def runnerOfModule {σ τ : Spec.Shape} (task : SeqTask σ τ)
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (module : API.TorchLean.Module.ScalarModule α (paramShapes task) [σ, τ]) :
    IO (Runner α task) := do
  let predictorTrain ← API.TorchLean.LayerCore.Seq.compileForwardWithMode .train (α := α) task.model
  let predictorEval ← API.TorchLean.LayerCore.Seq.compileForwardWithMode .eval (α := α) task.model
  let lossTrain ← API.TorchLean.Autodiff.compileLoss (α := α)
    (paramShapes := paramShapes task) (inputShapes := [σ, τ])
    (task.moduleDefWithMode .train).loss
  let lossEval ← API.TorchLean.Autodiff.compileLoss (α := α)
    (paramShapes := paramShapes task) (inputShapes := [σ, τ])
    (task.moduleDefWithMode .eval).loss
  let mode : IO.Ref API.TorchLean.LayerCore.Mode ← IO.mkRef .eval
  pure { module, predictorTrain, predictorEval, lossTrain, lossEval, mode }

/--
Instantiate a `Runner` by explicitly providing a `Float → α` cast and backend.

Use this when you want to run the same task over different numeric backends (e.g. `Float` vs
`IEEE32Exec`) or when you want custom literal injection.
-/
def instantiateConfigured {σ τ : Spec.Shape} (task : SeqTask σ τ)
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (cast : Float → α) (opts : API.TorchLean.Options := {}) :
    IO (Runner α task) := do
  let module ← API.TorchLean.Module.instantiateConfigured (α := α) task.moduleDef cast opts
  runnerOfModule task module

/--
Instantiate a `Float` runner using storage-first parameter initialization when the model provides
it. Models without a runtime plan automatically retain the ordinary tensor initializer path.
-/
def instantiateConfiguredFloat {σ τ : Spec.Shape} (task : SeqTask σ τ)
    (opts : API.TorchLean.Options := {}) : IO (Runner Float task) := do
  let module ← API.TorchLean.Module.instantiateFloat task.moduleDef opts
  runnerOfModule task module

/--
Instantiate a `Runner` by explicitly providing a `Float → α` cast and a backend selector.

Instantiate a module with explicit runtime options.
-/
def instantiateWith {σ τ : Spec.Shape} (task : SeqTask σ τ)
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (cast : Float → α) (backend : API.TorchLean.Backend := .eager) :
    IO (Runner α task) := do
  instantiateConfigured (task := task) (α := α) cast { backend := backend }

/--
Instantiate a `Runner` using the standard runtime literal injection `API.Runtime.ofFloat`.

This is the common entrypoint for executable examples.
-/
def instantiateWithRuntimeOptions {σ τ : Spec.Shape} (task : SeqTask σ τ)
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [API.Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (opts : API.TorchLean.Options := {}) :
    IO (Runner α task) :=
  instantiateConfigured (task := task) (α := α) API.Runtime.ofFloat opts

/--
Instantiate a `Runner` using the standard runtime literal injection `API.Runtime.ofFloat` and a
backend selector.

Instantiate a module after parsing runtime options.
-/
def instantiate {σ τ : Spec.Shape} (task : SeqTask σ τ)
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [API.Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (backend : API.TorchLean.Backend := .eager) :
    IO (Runner α task) :=
  instantiateWithRuntimeOptions (task := task) (α := α) { backend := backend }

/--
Run a TorchLean task with CLI-style dtype/backend selection, then call `k` with a fully constructed
  runner.

This is used by `lake exe` entrypoints: `run` takes care of parsing dtype flags and instantiating
the underlying module/compiled programs.
-/
def run {σ τ : Spec.Shape} (task : SeqTask σ τ) (args : List String)
    (k :
      ∀ {α : Type}, [API.Semantics.Scalar α] → [DecidableEq Spec.Shape] → [ToString α] →
        [API.Runtime.Scalar α] →
        Runner α task → List String → IO Unit) :
    IO Unit := do
  API.TorchLean.Module.withModuleRuntime task.moduleDef args (fun {α} _ _ _ _ module rest => do
    let predictorTrain ← API.TorchLean.LayerCore.Seq.compileForwardWithMode .train (α := α) task.model
    let predictorEval ← API.TorchLean.LayerCore.Seq.compileForwardWithMode .eval (α := α) task.model
    let lossTrain ← API.TorchLean.Autodiff.compileLoss (α := α)
      (paramShapes := paramShapes task) (inputShapes := [σ, τ])
      (task.moduleDefWithMode .train).loss
    let lossEval ← API.TorchLean.Autodiff.compileLoss (α := α)
      (paramShapes := paramShapes task) (inputShapes := [σ, τ])
      (task.moduleDefWithMode .eval).loss
    let mode : IO.Ref API.TorchLean.LayerCore.Mode ← IO.mkRef .eval
    k (α := α)
      { module := module
        predictorTrain := predictorTrain
        predictorEval := predictorEval
        lossTrain := lossTrain
        lossEval := lossEval
        mode := mode }
      rest
  )

/-- Read the current parameter list from a runner. -/
def params {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) :
    IO (API.TorchLean.TensorPack α (paramShapes task)) :=
  API.TorchLean.Module.params runner.module

/-- Read the runner's current mode (`.train` or `.eval`). -/
def mode {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) : IO API.TorchLean.LayerCore.Mode :=
  runner.mode.get

/-- Set the runner mode (`.train` or `.eval`). -/
def setMode {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (value : API.TorchLean.LayerCore.Mode) : IO Unit :=
  runner.mode.set value

/-- Convenience: `setMode runner .train`. -/
def trainMode {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) : IO Unit :=
  setMode runner .train

/-- Convenience: `setMode runner .eval`. -/
def evalMode {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) : IO Unit :=
  setMode runner .eval

/-- Predicate: are we in training mode? -/
def isTraining {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) : IO Bool :=
  do
    pure ((← mode runner) == .train)

/-- Pick the predictor compiled for the runner's current mode (`.train` or `.eval`). -/
def activePredictor {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) :
    IO (_root_.Runtime.Autograd.Torch.CompiledGraph α (paramShapes task ++ [σ]) τ) := do
  match (← mode runner) with
  | .train => pure runner.predictorTrain
  | .eval => pure runner.predictorEval

/-- Pick the loss program compiled for the runner's current mode (`.train` or `.eval`). -/
def activeLoss {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) :
    IO (_root_.Runtime.Autograd.Torch.CompiledScalar α (paramShapes task ++ [σ, τ])) := do
  match (← mode runner) with
  | .train => pure runner.lossTrain
  | .eval => pure runner.lossEval

/--
Refresh mode-dependent runner buffers using one supervised sample.

This mutates the module parameters only in `.train` mode, mirroring PyTorch-style buffer updates
for layers such as normalization. In `.eval` mode it is a no-op.
-/
def updateRunnerBuffers {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (sample : API.TorchLean.TensorPack α [σ, τ]) : IO Unit := do
  let currentMode ← mode runner
  if currentMode == .train && API.TorchLean.LayerCore.Seq.hasBufferUpdates task.model then
    match sample with
    | .cons x (.cons _y .nil) => do
        let ps ← params runner
        let ps' ← API.TorchLean.LayerCore.Seq.updateBuffers currentMode task.model ps x
        API.TorchLean.Module.setParams runner.module ps'
  else
    pure ()

/--
Run one forward/backward pass on a single supervised sample and return gradients for all parameters.
Unlike PyTorch's in-place `.grad` convention, this API returns the gradient pack explicitly.
-/
def backward {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (sample : API.TorchLean.TensorPack α [σ, τ]) :
    IO (API.TorchLean.TensorPack α (paramShapes task)) := do
  -- The instantiated scalar module always uses the training-mode program; keep the runner mode
  -- aligned so `updateRunnerBuffers` is not accidentally skipped.
  trainMode runner
  updateRunnerBuffers runner sample
  API.TorchLean.Module.backward runner.module sample

/-- Predict on one input tensor using the runner's active mode (`.train` or `.eval`). -/
def predict {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (x : Spec.Tensor α σ) :
    IO (Spec.Tensor α τ) := do
  if runner.module.opts.usesCuda then
    let currentMode ← mode runner
    API.TorchLean.LayerCore.Seq.forward (α := α) (tensorConv := runner.module.tensorConv)
      runner.module.opts currentMode task.model runner.module.trainer.params x
  else
    let ps ← params runner
    let predictor ← activePredictor runner
    pure (API.TorchLean.LayerCore.Seq.forwardArtifact task.model predictor ps x)

/-- Predict on a list of inputs by repeatedly calling `predict`. -/
def predictBatch {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (xs : List (Spec.Tensor α σ)) :
    IO (List (Spec.Tensor α τ)) :=
  xs.mapM (predict runner)

/-- For classification heads: run `predict`, then take `argmax` over the logits (if defined). -/
def predictClass? {σ : Spec.Shape} {n : Nat} {task : SeqTask σ (.dim n .scalar)}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (x : Spec.Tensor α σ) :
    IO (Option (Fin n)) := do
  let logits ← predict runner x
  pure <| API.TorchLean.Metrics.argmax? (α := α) (n := n) logits

/-- Compute `(correct, total)` for a one-hot classification dataset. -/
def accuracyOneHot {σ : Spec.Shape} {n : Nat} {task : SeqTask σ (.dim n .scalar)}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (samples : List (API.TorchLean.TensorPack α [σ, .dim n .scalar])) :
    IO (Nat × Nat) := do
  let rec go (correct total : Nat) :
      List (API.TorchLean.TensorPack α [σ, .dim n .scalar]) → IO (Nat × Nat)
    | [] => pure (correct, total)
    | sample :: rest =>
        do
          let (x, y) :=
            match sample with
            | .cons x (.cons y .nil) => (x, y)
          let logits ← predict runner x
          let ok := API.TorchLean.Metrics.correctOneHot? (α := α) (n := n) logits y
          go (if ok = some true then correct + 1 else correct) (total + 1) rest
  go 0 0 samples

/-- Mean scalar loss over a list of supervised samples (uses the runner's active mode). -/
def meanLoss {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    (runner : Runner α task) (samples : List (API.TorchLean.TensorPack α [σ, τ])) :
    IO α := do
  let values ←
    if runner.module.opts.usesCuda then
      samples.mapM (fun sample => do
        let loss ← API.TorchLean.Module.forward runner.module sample
        pure (Spec.Tensor.toScalar loss))
    else do
      let compiled ← activeLoss runner
      let ps ← params runner
      samples.mapM (fun sample => do
        let args : API.TorchLean.TensorPack α (paramShapes task ++ [σ, τ]) :=
          _root_.Proofs.Autograd.Algebra.TList.append
            (α := α) (ss₁ := paramShapes task) (ss₂ := [σ, τ]) ps sample
        pure (Spec.Tensor.toScalar <| _root_.Runtime.Autograd.Torch.CompiledScalar.forward compiled
          args))
  match values with
  | [] => pure 0
  | xs => pure (xs.foldl (· + ·) 0 / (xs.length : α))

/-- Mean scalar loss over a dataset (materialized via `dataset.toList`). -/
def meanLossDataset {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape] [ToString α]
    (runner : Runner α task)
    (dataset : _root_.Runtime.Autograd.Train.Dataset (API.TorchLean.TensorPack α [σ, τ])) :
    IO α :=
  meanLoss runner dataset.toList

/-- Scalar loss for one sample through the instantiated runtime module. -/
def moduleLoss {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (sample : API.TorchLean.TensorPack α [σ, τ]) : IO α := do
  let loss ← API.TorchLean.Module.forward runner.module sample
  pure (Spec.Tensor.toScalar loss)

/-- Treat `0` as the conservative single-sample step size. -/
def effectiveTrainBatchSize (n : Nat) : Nat :=
  if n = 0 then 1 else n

/-- Take the next cyclic group of samples from an in-memory training set. -/
def nextCyclicBatch {α : Type} (context : String)
    (samples : List α) (restRef : IO.Ref (List α)) (batchSize : Nat) : IO (List α) := do
  if samples.isEmpty then
    throw <| IO.userError s!"{context}: empty sample cycle"
  let mut batch : List α := []
  for _ in [0:batchSize] do
    let mut rest ← restRef.get
    if rest.isEmpty then
      rest := samples
    match rest with
    | [] => throw <| IO.userError s!"{context}: empty sample cycle"
    | sample :: rest' =>
        restRef.set rest'
        batch := sample :: batch
  pure batch.reverse

end Supervised

end TorchLean
end API
end NN
