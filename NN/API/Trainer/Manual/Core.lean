/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Scalar
public import NN.API.Optim.Config
public import NN.API.Trainer.Scheduler
public import NN.GraphSpec.Models.TorchLean
public import NN.Runtime.Autograd.TorchLean
public import NN.API.Module.Command

@[expose] public section

namespace TorchLean
namespace Trainer
namespace Manual

/-!
# Supervised Runtime Training

Supervised tasks, runners, training configurations, and the low-level session objects used by
custom executable training loops.
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
    TorchLean.Module.ObjectiveDef Unit
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
    TorchLean.Module.ObjectiveDef Unit
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
abbrev SeqTask.stateShapes {σ τ : Spec.Shape} (task : SeqTask σ τ) : List Spec.Shape :=
  _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes task.model

open SeqTask

/-- Fixed-step training configuration for finite sample streams. -/
structure TrainConfig where
  /-- Number of optimizer updates. -/
  steps : Nat
  /-- Number of stream items accumulated into one optimizer update. -/
  batchSize : Nat := 1
  /-- Optimizer configuration. -/
  optimizer : optim.Optimizer := .sgd 0.01
  /-- Optional learning-rate schedule applied per optimizer step. -/
  scheduler : Option _root_.TorchLean.Trainer.Scheduler.Config := none
  /-- Log once every this many steps; `0` disables logging. -/
  logEvery : Nat := 1
  /-- Sample CUDA allocator state at this cadence; `0` disables sampling. -/
  cudaMemWatch : Nat := 0
  deriving Repr

/-- Epoch-based training configuration for finite loaders. -/
structure LoaderTrainConfig where
  /-- Number of complete epochs. -/
  epochs : Nat
  /-- Optimizer configuration. -/
  optimizer : optim.Optimizer := .sgd 0.01
  /-- Optional learning-rate schedule applied once per epoch. -/
  scheduler : Option _root_.TorchLean.Trainer.Scheduler.Config := none
  /-- Log once every this many steps; `0` disables logging. -/
  logEvery : Nat := 1
  deriving Repr

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
  module : TorchLean.Module.Objective α Unit (stateShapes task) [σ, τ]
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
    (module : TorchLean.Module.Objective α Unit (stateShapes task) [σ, τ]) :
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
    [Runtime.TensorTransfer α]
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
    [Runtime.TensorTransfer α]
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
    IO (_root_.TorchLean.TensorPack α (stateShapes task)) :=
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

/-- Refresh mode-dependent runner buffers using one supervised sample. -/
def updateBuffers {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (sample : _root_.TorchLean.TensorPack α [σ, τ]) : IO Unit := do
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

/-- Run an action with a runner temporarily switched to the requested mode. -/
def withMode {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    {β : Type} (runner : Runner α task) (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
    (action : IO β) : IO β := do
  let previous ← runner.currentMode
  runner.setMode mode
  try
    action
  finally
    runner.setMode previous

end Runner

end Manual
end Trainer
end TorchLean
