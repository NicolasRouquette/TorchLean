/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Scalar
public import NN.API.Rand
public import NN.API.Trainer.Scheduler
public import NN.GraphSpec.Models.TorchLean
public import NN.Runtime.Autograd.TorchLean
public import NN.Runtime.Autograd.Train.Dataset
public import NN.Runtime.RL
public import NN.Spec.RL.MDP
public import NN.Spec.RL.MarkovMDP
public import NN.Spec.RL.FiniteStochasticMDP
public import NN.API.Module.Command

import Mathlib.Algebra.Order.Algebra
import NN.Spec.Autograd.AutogradSpec

@[expose] public section

namespace TorchLean
namespace Trainer
namespace Manual

/-!
# Supervised Runtime Training

Supervised tasks, runners, steppers, optimizer configs, trainer aliases, and the low-level session
exports that back executable examples.
-/

/-
Supervised training helpers built directly on `Objective`.

This is a lower-level layer than `TorchLean.Trainer`: it is designed around a
`SeqTask σ τ` (model + loss) and produces a `Runner` + `Stepper` that can be used in scripts.
-/

/-- Built-in loss choices for `SeqTask`. -/
inductive SeqLoss (outputShape : Spec.Shape) where
  | mse (reduction : _root_.TorchLean.Loss.Reduction := .mean)
  | oneHotCrossEntropy (axis : Nat)
      [validAxis : Spec.Shape.AxisInBounds axis outputShape]
      (reduction : _root_.TorchLean.Loss.Reduction := .mean)

/-- A supervised task is just a model plus a choice of loss. -/
structure SeqTask (σ τ : Spec.Shape) where
  /-- Model to run. -/
  model : _root_.Runtime.Autograd.TorchLean.NN.Seq σ τ
  /-- Loss function. -/
  loss : SeqLoss τ

/--
Build the scalar objective for a task in an explicit train/eval mode.

This is the underlying "instantiate me as a runnable module" step for training.
-/
def SeqTask.objectiveWithMode {σ τ : Spec.Shape} (task : SeqTask σ τ)
    (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode) :
    TorchLean.Module.ObjectiveDef
      (_root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes task.model) [σ, τ] :=
  match task.loss with
  | .mse reduction =>
      _root_.Runtime.Autograd.TorchLean.NN.Seq.Objective.mseWithMode mode
        (model := task.model) (reduction := reduction)
  | @SeqLoss.oneHotCrossEntropy _ axis validAxis reduction =>
      letI := validAxis
      _root_.Runtime.Autograd.TorchLean.NN.Seq.Objective.oneHotCrossEntropyWithMode mode
        (model := task.model) axis (reduction := reduction)

/-- Build the task's scalar objective in training mode. -/
def SeqTask.objective {σ τ : Spec.Shape} (task : SeqTask σ τ) :
    TorchLean.Module.ObjectiveDef
      (_root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes task.model) [σ, τ] :=
  task.objectiveWithMode .train

namespace SeqTask

/-- Constructor: regression task (MSE loss). -/
def mse {σ τ : Spec.Shape} (model : _root_.Runtime.Autograd.TorchLean.NN.Seq σ τ)
    (reduction : _root_.TorchLean.Loss.Reduction := .mean) :
    SeqTask σ τ :=
  { model := model, loss := .mse reduction }

/-- Constructor: one-hot classification task (cross-entropy loss). -/
def oneHotCrossEntropy {σ τ : Spec.Shape}
    (model : _root_.Runtime.Autograd.TorchLean.NN.Seq σ τ)
    (axis : Nat) [Spec.Shape.AxisInBounds axis τ]
    (reduction : _root_.TorchLean.Loss.Reduction := .mean) :
    SeqTask σ τ :=
  { model := model, loss := .oneHotCrossEntropy axis reduction }

end SeqTask

/-- Shapes of the task model's trainable parameters and persistent buffers. -/
abbrev stateShapes {σ τ : Spec.Shape} (task : SeqTask σ τ) : List Spec.Shape :=
  _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes task.model

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
  /-- Number of optimizer updates. -/
  steps : Nat
  /--
  Number of dataset items consumed by one training step.

  The loop differentiates every item at the same parameter point, averages the resulting gradient
  packs, and performs one optimizer update. If each item is already a fixed-size tensor minibatch,
  keep this value at `1` for one vectorized forward/backward pass per update.
  -/
  batchSize : Nat := 1
  /-- Optimizer configuration. -/
  optimizer : OptimizerConfig := .sgd 0.01
  /-- Scheduler configuration. -/
  scheduler : Option _root_.TorchLean.Trainer.Scheduler.Config := none
  /-- Log once every this many steps. -/
  logEvery : Nat := 1
  /-- Sample CUDA allocator state every this many completed steps; `0` disables sampling. -/
  cudaMemWatch : Nat := 0
  deriving Repr

namespace CUDAMemory

/-- State carried by the CUDA-memory drift detector used by sustained training runs. -/
structure State where
  firstStep : Nat
  firstFreeBytes : Nat
  warned : Bool
deriving Repr

/-- Resolve an explicit CUDA-memory cadence, or enable periodic sampling for very long runs. -/
def cadence (opts : _root_.Runtime.Autograd.Torch.Options)
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
def sample (opts : _root_.Runtime.Autograd.Torch.Options)
    (watchEvery totalSteps done : Nat) (state? : Option State) : IO (Option State) := do
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
                s!"  cuda_mem warning: free device memory is dropping by ~{dropPerStep} " ++
                  s!"bytes/step; projected allocation failure before requested step count " ++
                  s!"(around step {projectedFailure})."
              pure (some { st with warned := true })
            else
              pure (some st)

end CUDAMemory

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
- `scheduler`: optional learning-rate schedule applied once per epoch,
- `logEvery`: progress printing frequency (`0` disables logging).
-/
structure LoaderTrainConfig where
  /-- Number of epochs to train for. -/
  epochs : Nat
  /-- Optimizer configuration. -/
  optimizer : OptimizerConfig := .sgd 0.01
  /-- Scheduler configuration. -/
  scheduler : Option _root_.TorchLean.Trainer.Scheduler.Config := none
  /-- Log once every this many steps. -/
  logEvery : Nat := 1
  deriving Repr

/-- Extract the base learning rate encoded in an optimizer configuration. -/
def optimizerLr : OptimizerConfig → Float
  | .sgd lr _ => lr
  | .adagrad lr _ => lr
  | .rmsprop lr _ _ => lr
  | .adam lr _ _ _ => lr
  | .adamw lr _ _ _ _ => lr
  | .adadelta lr _ _ => lr

/--
Resolve the learning rate to use at a given training step.

If a scheduler is present, it takes precedence over the optimizer's baked-in base learning rate.
Otherwise it returns `optimizerLr cfg`.
-/
def stepLr (scheduler : Option _root_.TorchLean.Trainer.Scheduler.Config) (cfg : OptimizerConfig)
    (step : Nat) : Float :=
  match scheduler with
  | some sched => _root_.TorchLean.Trainer.Scheduler.lrAt sched step
  | none => optimizerLr cfg

/-- Map a state update over every optimizer-state entry in a shape-indexed parameter list. -/
def mapStateList {State : Type → Spec.Shape → Type} {α : Type} :
    {ss : List Spec.Shape} →
    ({s : Spec.Shape} → State α s → State α s) →
    _root_.Runtime.Autograd.TorchLean.Optim.StateList State α ss →
    _root_.Runtime.Autograd.TorchLean.Optim.StateList State α ss
  | [], _, .nil => .nil
  | _ :: ss, f, .cons st rest => .cons (f st) (mapStateList (ss := ss) f rest)

/-- Set the learning rate field of every Adam optimizer state entry to `lr`. -/
def adamStateWithLr {α : Type} (lr : α) {paramShapes : List Spec.Shape} :
    _root_.Runtime.Autograd.TorchLean.Optim.StateList _root_.Optim.Adam.State α paramShapes →
    _root_.Runtime.Autograd.TorchLean.Optim.StateList _root_.Optim.Adam.State α paramShapes :=
  mapStateList (ss := paramShapes) (fun st => { st with lr := lr })

/-- Set the learning rate field of every plain-SGD optimizer state entry to `lr`. -/
def sgdStateWithLr {α : Type} (lr : α) {paramShapes : List Spec.Shape} :
    _root_.Runtime.Autograd.TorchLean.Optim.StateList _root_.Optim.SGD.State α paramShapes →
    _root_.Runtime.Autograd.TorchLean.Optim.StateList _root_.Optim.SGD.State α paramShapes :=
  mapStateList (ss := paramShapes) (fun st => { st with lr := lr })

/-- Set the learning rate field of every momentum-SGD optimizer state entry to `lr`. -/
def momentumSgdStateWithLr {α : Type} (lr : α) {paramShapes : List Spec.Shape} :
    _root_.Runtime.Autograd.TorchLean.Optim.StateList _root_.Optim.MomentumSGD.State α paramShapes →
    _root_.Runtime.Autograd.TorchLean.Optim.StateList _root_.Optim.MomentumSGD.State α paramShapes :=
  mapStateList (ss := paramShapes) (fun st => { st with lr := lr })

/-- Set the learning rate field of every AdaGrad optimizer state entry to `lr`. -/
def adagradStateWithLr {α : Type} (lr : α) {paramShapes : List Spec.Shape} :
    _root_.Runtime.Autograd.TorchLean.Optim.StateList _root_.Optim.AdaGrad.State α paramShapes →
    _root_.Runtime.Autograd.TorchLean.Optim.StateList _root_.Optim.AdaGrad.State α paramShapes :=
  mapStateList (ss := paramShapes) (fun st => { st with lr := lr })

/-- Set the learning rate field of every RMSProp optimizer state entry to `lr`. -/
def rmspropStateWithLr {α : Type} (lr : α) {paramShapes : List Spec.Shape} :
    _root_.Runtime.Autograd.TorchLean.Optim.StateList _root_.Optim.RMSProp.State α paramShapes →
    _root_.Runtime.Autograd.TorchLean.Optim.StateList _root_.Optim.RMSProp.State α paramShapes :=
  mapStateList (ss := paramShapes) (fun st => { st with lr := lr })

/-- Set the learning rate field of every AdamW optimizer state entry to `lr`. -/
def adamwStateWithLr {α : Type} (lr : α) {paramShapes : List Spec.Shape} :
    _root_.Runtime.Autograd.TorchLean.Optim.StateList _root_.Optim.AdamW.State α paramShapes →
    _root_.Runtime.Autograd.TorchLean.Optim.StateList _root_.Optim.AdamW.State α paramShapes :=
  mapStateList (ss := paramShapes) (fun st => { st with lr := lr })

/-- Set the learning rate field of every Adadelta optimizer state entry to `lr`. -/
def adadeltaStateWithLr {α : Type} (lr : α) {paramShapes : List Spec.Shape} :
    _root_.Runtime.Autograd.TorchLean.Optim.StateList _root_.Optim.Adadelta.State α paramShapes →
    _root_.Runtime.Autograd.TorchLean.Optim.StateList _root_.Optim.Adadelta.State α paramShapes :=
  mapStateList (ss := paramShapes) (fun st => { st with lr := lr })

/--
A fully instantiated supervised task runner.

This bundles:
- the imperative `Objective` (parameters/buffers stored in refs),
- lowered typed graphs for forward evaluation and loss in both `.train` and `.eval` modes (so switching mode is
  low-overhead),
- and the current mode stored in an `IO.Ref`.

The mode influences both operator behavior (e.g. dropout/batchnorm) and whether buffers are updated
during training.
-/
structure Runner (α : Type) [_root_.Context α] [DecidableEq Spec.Shape]
    {σ τ : Spec.Shape} (task : SeqTask σ τ) where
  /-- Instantiated scalar module storing parameters/buffers in mutable refs. -/
  module : TorchLean.Module.Objective α (stateShapes task) [σ, τ]
  /-- Typed forward graph specialized to training-mode behavior. -/
  predictorTrain : _root_.Runtime.Autograd.Torch.TypedGraph α (stateShapes task ++ [σ]) τ
  /-- Typed forward graph specialized to eval-mode behavior. -/
  predictorEval : _root_.Runtime.Autograd.Torch.TypedGraph α (stateShapes task ++ [σ]) τ
  /-- Typed scalar loss graph specialized to training-mode behavior. -/
  lossTrain : _root_.Runtime.Autograd.Torch.TypedScalarGraph α (stateShapes task ++ [σ, τ])
  /-- Typed scalar loss graph specialized to eval-mode behavior. -/
  lossEval : _root_.Runtime.Autograd.Torch.TypedScalarGraph α (stateShapes task ++ [σ, τ])
  /-- Mutable mode flag (`.train` / `.eval`) used by stateful layers (e.g. dropout/batchnorm). -/
  mode : IO.Ref _root_.Runtime.Autograd.TorchLean.NN.Mode

namespace Runner

/-- Finish runner construction once parameter storage has been instantiated. -/
def ofModule {σ τ : Spec.Shape} (task : SeqTask σ τ)
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (module : TorchLean.Module.Objective α (stateShapes task) [σ, τ]) :
    IO (Runner α task) := do
  let predictorTrain ← _root_.Runtime.Autograd.TorchLean.NN.Seq.lowerToTypedGraph
    (α := α) task.model (mode := .train)
  let predictorEval ← _root_.Runtime.Autograd.TorchLean.NN.Seq.lowerToTypedGraph
    (α := α) task.model (mode := .eval)
  let lossTrain ← _root_.Runtime.Autograd.TorchLean.Autodiff.lowerScalarToTypedGraph (α := α)
    (paramShapes := stateShapes task) (inputShapes := [σ, τ])
    (task.objectiveWithMode .train).loss
  let lossEval ← _root_.Runtime.Autograd.TorchLean.Autodiff.lowerScalarToTypedGraph (α := α)
    (paramShapes := stateShapes task) (inputShapes := [σ, τ])
    (task.objectiveWithMode .eval).loss
  let mode : IO.Ref _root_.Runtime.Autograd.TorchLean.NN.Mode ← IO.mkRef .train
  pure { module, predictorTrain, predictorEval, lossTrain, lossEval, mode }

/--
Instantiate a `Runner` by explicitly providing a `Float → α` cast and backend.

Use this when you want to run the same task over different scalar semantics (for example native
`Float32` and `IEEE32Exec`) or when you want custom literal injection.
-/
def instantiateAs {σ τ : Spec.Shape} (task : SeqTask σ τ)
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (cast : Float → α) (opts : _root_.Runtime.Autograd.Torch.Options := {}) :
    IO (Runner α task) := do
  let module ← TorchLean.Module.instantiateAs (α := α) task.objective cast opts
  ofModule task module

/--
Instantiate a runner over Lean's binary64 `Float` type.

This explicit binary64 path is useful for numerical comparisons and low-level tests. Public
training defaults to native `Float32`. Storage-first initialization is used when the model provides
a plan; other models retain their tensor initializer.
-/
def instantiateFloat64 {σ τ : Spec.Shape} (task : SeqTask σ τ)
    (opts : _root_.Runtime.Autograd.Torch.Options := {}) : IO (Runner Float task) := do
  let module ← TorchLean.Module.instantiateFloat64 task.objective opts
  ofModule task module

/--
Instantiate a `Runner` using the standard runtime literal injection `_root_.TorchLean.Runtime.ofFloat`.

This is the common entrypoint for executable examples.
-/
def instantiate {σ τ : Spec.Shape} (task : SeqTask σ τ)
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [_root_.TorchLean.Runtime.FromFloat α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (opts : _root_.Runtime.Autograd.Torch.Options := {}) :
    IO (Runner α task) :=
  instantiateAs (task := task) (α := α) _root_.TorchLean.Runtime.ofFloat opts

end Runner

/--
Run a TorchLean task with CLI-style scalar/execution selection, then call `k` with a fully constructed
  runner.

This is used by `lake exe` entrypoints: `run` takes care of parsing scalar flags and instantiating
the underlying module and typed graphs.
-/
def run {σ τ : Spec.Shape} (task : SeqTask σ τ) (args : List String)
    (k :
      ∀ {α : Type}, [_root_.Context α] → [DecidableEq Spec.Shape] → [ToString α] →
        [_root_.TorchLean.Runtime.FromFloat α] →
        Runner α task → List String → IO Unit) :
    IO Unit := do
  TorchLean.Module.withModule task.objective args (fun {α} _ _ _ _ _cast module rest => do
    let predictorTrain ← _root_.Runtime.Autograd.TorchLean.NN.Seq.lowerToTypedGraph
      (α := α) task.model (mode := .train)
    let predictorEval ← _root_.Runtime.Autograd.TorchLean.NN.Seq.lowerToTypedGraph
      (α := α) task.model (mode := .eval)
    let lossTrain ←
      _root_.Runtime.Autograd.TorchLean.Autodiff.lowerScalarToTypedGraph (α := α)
      (paramShapes := stateShapes task) (inputShapes := [σ, τ])
      (task.objectiveWithMode .train).loss
    let lossEval ←
      _root_.Runtime.Autograd.TorchLean.Autodiff.lowerScalarToTypedGraph (α := α)
      (paramShapes := stateShapes task) (inputShapes := [σ, τ])
      (task.objectiveWithMode .eval).loss
    let mode : IO.Ref _root_.Runtime.Autograd.TorchLean.NN.Mode ← IO.mkRef .train
    k (α := α)
      { module := module
        predictorTrain := predictorTrain
        predictorEval := predictorEval
        lossTrain := lossTrain
        lossEval := lossEval
        mode := mode }
      rest
  )

namespace Runner

/-- Read the complete parameter-and-buffer state from a runner. -/
def state {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (runner : Runner α task) :
    IO (_root_.Runtime.Autograd.Torch.TList α (stateShapes task)) :=
  TorchLean.Module.state runner.module

/-- Read the runner's current mode (`.train` or `.eval`). -/
def currentMode {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (runner : Runner α task) : IO _root_.Runtime.Autograd.TorchLean.NN.Mode :=
  runner.mode.get

/-- Set the runner mode (`.train` or `.eval`). -/
def setMode {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (value : _root_.Runtime.Autograd.TorchLean.NN.Mode) : IO Unit :=
  runner.mode.set value

/-- Select training behavior for stateful layers. -/
def train {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (runner : Runner α task) : IO Unit :=
  setMode runner .train

/-- Select evaluation behavior for stateful layers. -/
def eval {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (runner : Runner α task) : IO Unit :=
  setMode runner .eval

/-- Predicate: are we in training mode? -/
def isTraining {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (runner : Runner α task) : IO Bool :=
  do
    pure ((← currentMode runner) == .train)

/--
Refresh mode-dependent runner buffers using one supervised sample.

This mutates the module parameters only in `.train` mode, mirroring PyTorch-style buffer updates
for layers such as normalization. In `.eval` mode it is a no-op.
-/
def updateBuffers {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (sample : _root_.Runtime.Autograd.Torch.TList α [σ, τ]) : IO Unit := do
  let selectedMode ← currentMode runner
  if selectedMode == .train &&
      _root_.Runtime.Autograd.TorchLean.NN.Seq.hasBufferUpdates task.model then
    match sample with
    | .cons x (.cons _y .nil) => do
        let ps ← state runner
        let ps' ←
          _root_.Runtime.Autograd.TorchLean.NN.Seq.updateBuffers selectedMode task.model ps x
        TorchLean.Module.loadState runner.module ps'
  else
    pure ()

/--
Return the state-shaped gradient of one supervised sample.

Unlike PyTorch's mutating `.backward()` operation, this function returns the gradient pack.
-/
def gradState {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (sample : _root_.Runtime.Autograd.Torch.TList α [σ, τ]) :
    IO (_root_.Runtime.Autograd.Torch.TList α (stateShapes task)) := do
  -- The instantiated scalar module always uses the training-mode program; keep the runner mode
  -- aligned so `updateBuffers` is not accidentally skipped.
  train runner
  updateBuffers runner sample
  TorchLean.Module.gradState runner.module sample .nil

/-- Run one input tensor using the active mode (`.train` or `.eval`). -/
def run {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (x : Spec.Tensor α σ) :
    IO (Spec.Tensor α τ) := do
  if runner.module.opts.usesCuda then
    let selectedMode ← currentMode runner
    _root_.Runtime.Autograd.TorchLean.NN.Seq.forwardNoGrad
      (α := α) (tensorConv := runner.module.tensorConv)
      runner.module.opts task.model runner.module.trainer.state x (mode := selectedMode)
  else
    let ps ← state runner
    let predictor :=
      match ← currentMode runner with
      | .train => runner.predictorTrain
      | .eval => runner.predictorEval
    let args : _root_.Runtime.Autograd.Torch.TList α
        (_root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes task.model ++ [σ]) :=
      _root_.Proofs.Autograd.Algebra.TList.append
        (ss₁ := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes task.model)
        (ss₂ := [σ]) ps (.cons x .nil)
    pure (_root_.Runtime.Autograd.Torch.TypedGraph.forward predictor args)

/-- Run a list of inputs using the active mode. -/
def runBatch {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (xs : List (Spec.Tensor α σ)) :
    IO (List (Spec.Tensor α τ)) :=
  xs.mapM (run runner)

/-- Compute `(correct, total)` along a class axis for a one-hot classification dataset. -/
def accuracyOneHot {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (axis : Nat) [Spec.Shape.AxisInBounds axis τ]
    (samples : List (_root_.Runtime.Autograd.Torch.TList α [σ, τ])) :
    IO (Nat × Nat) := do
  let rec go (correct total : Nat) :
      List (_root_.Runtime.Autograd.Torch.TList α [σ, τ]) → IO (Nat × Nat)
    | [] => pure (correct, total)
    | sample :: rest =>
        do
          let (x, y) :=
            match sample with
            | .cons x (.cons y .nil) => (x, y)
          let logits ← run runner x
          let (sampleCorrect, sampleTotal) :=
            _root_.TorchLean.Metrics.accuracyOneHotAxis (α := α) axis logits y
          go (correct + sampleCorrect) (total + sampleTotal) rest
  go 0 0 samples

/-- Mean scalar loss over a list of supervised samples (uses the runner's active mode). -/
def meanLoss {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    (runner : Runner α task) (samples : List (_root_.Runtime.Autograd.Torch.TList α [σ, τ])) :
    IO α := do
  let values ←
    if runner.module.opts.usesCuda then
      samples.mapM (fun sample => do
        let loss ← TorchLean.Module.loss runner.module sample .nil
        pure (Spec.Tensor.item loss))
    else do
      let lossGraph :=
        match ← currentMode runner with
        | .train => runner.lossTrain
        | .eval => runner.lossEval
      let ps ← state runner
      samples.mapM (fun sample => do
        let args : _root_.Runtime.Autograd.Torch.TList α (stateShapes task ++ [σ, τ]) :=
          _root_.Proofs.Autograd.Algebra.TList.append
            (α := α) (ss₁ := stateShapes task) (ss₂ := [σ, τ]) ps sample
        pure (Spec.Tensor.item <| _root_.Runtime.Autograd.Torch.TypedScalarGraph.forward lossGraph
          args))
  match values with
  | [] => pure 0
  | xs => pure (xs.foldl (· + ·) 0 / (xs.length : α))

/-- Mean scalar loss over a dataset (materialized via `dataset.toList`). -/
def meanLossDataset {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    (runner : Runner α task)
    (dataset : _root_.Runtime.Autograd.Train.Dataset (_root_.Runtime.Autograd.Torch.TList α [σ, τ])) :
    IO α :=
  meanLoss runner dataset.toList

/-- Scalar loss for one sample through the instantiated runtime module. -/
def moduleLoss {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (sample : _root_.Runtime.Autograd.Torch.TList α [σ, τ]) : IO α := do
  let loss ← TorchLean.Module.loss runner.module sample .nil
  pure (Spec.Tensor.item loss)

end Runner

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

end Manual
end Trainer
end TorchLean
