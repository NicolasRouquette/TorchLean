/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Data.Synthetic
public import NN.API.Trainer.Core
public import NN.API.CLI

/-!
# Training Options

Datasets, probes, runtime flag parsing, and per-training options for the trainer API.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

/-- Supervised data source that can be materialized at the trainer's selected scalar type. -/
structure DataSource (σ τ : Shape) where
  /-- Materialize the dataset at the runtime-selected scalar type. -/
  build :
    {α : Type} →
    [_root_.Context α] →
    [Runtime.FromFloat α] →
    IO (Training.Dataset (SupervisedSample α σ τ))

/-- A small input probe printed before and after training. -/
structure Probe (σ : Shape) where
  /-- Human-facing probe name. -/
  name : String
  /-- Human-facing input description. -/
  inputText : String := ""
  /-- Runtime-polymorphic input tensor. -/
  input : {α : Type} → [_root_.Context α] → [Runtime.FromFloat α] → Tensor α σ
  /-- Optional expected value shown beside the prediction. -/
  expected : Option String := none

namespace Probe

/-- Two-coordinate vector probe for small tabular regression examples. -/
def point (name : String) (x y : Float) (expected : Option String := none) :
    Probe (.dim 2 .scalar) :=
  { name := name
    inputText := s!"x=({x},{y})"
    input := fun {α} _ _ =>
      TorchLean.Data.Synthetic.pointVector (α := α) _root_.TorchLean.Runtime.ofFloat x y
    expected := expected }

/-- Probe built from a concrete `Float` tensor. -/
def ofFloatTensor {σ : Shape} (name : String) (x : Tensor Float σ)
    (inputText : String := "") (expected : Option String := none) :
    Probe σ :=
  { name := name
    inputText := inputText
    input := fun {α} _ _ => TorchLean.Tensor.castFloat (_root_.TorchLean.Runtime.ofFloat (α := α)) x
    expected := expected }

end Probe

/-- Runtime and optimizer settings stored with a trainer. -/
structure RunConfig extends RuntimeSettings where

namespace RunConfig

/-- Override the scalar semantics for this run configuration. -/
def withScalar (run : RunConfig) (scalar : Runtime.ScalarMode) : RunConfig :=
  { run with scalar := scalar }

/-- Override the execution mode for this run configuration. -/
def withExecution (run : RunConfig) (execution : Runtime.ExecutionMode) : RunConfig :=
  { run with execution := execution }

/-- Override the execution device using a maintained backend profile. -/
def withDevice (run : RunConfig) (device : Runtime.Device) : Except String RunConfig := do
  match _root_.NN.Backend.BackendProfile.maintainedForDevice? device with
  | some _ => pure { run with device := device, backendProfile? := none }
  | none =>
      throw s!"device `{device.cliName}` has no maintained runtime profile; provide an explicit backend profile"

/--
Select a complete backend contract profile.

The profile carries the device, provider preference, assurance policy, VJP ownership, and capsule
registry together. It can select, for example, LibTorch forward execution with a TorchLean-owned
backward pass.
-/
def withBackendProfile (run : RunConfig) (profile : _root_.NN.Backend.BackendProfile) : RunConfig :=
  { run with device := profile.policy.device, backendProfile? := some profile }

/-- Enable or disable first-use backend capsule reporting. -/
def withBackendReport (run : RunConfig) (enabled : Bool := true) : RunConfig :=
  { run with showBackend := enabled }

/-- Execute operations eagerly while building a dynamic tape. -/
def eager (run : RunConfig) : RunConfig :=
  run.withExecution .eager

/-- Record once and reuse the shape-indexed typed SSA graph. -/
def typedGraph (run : RunConfig) : RunConfig :=
  run.withExecution .typedGraph

/-- Run on CPU. -/
def cpu (run : RunConfig) : RunConfig :=
  { run with device := .cpu, backendProfile? := none }

/-- Run on CUDA. -/
def cuda (run : RunConfig) : RunConfig :=
  { run with device := .cuda, backendProfile? := none }

/-- Apply parsed runtime/device options to a persistent trainer run configuration. -/
def withRuntimeOptions (run : RunConfig) (opts : Options) : RunConfig :=
  { run with
      execution := opts.execution
      device := opts.device
      backendProfile? := opts.backendProfile?
      showBackend := opts.showBackend }

/-- Build a run configuration from parsed runtime flags and trainer choices. -/
def ofRuntimeOptions (opts : Options) (base : RunConfig := {}) : RunConfig :=
  base.withRuntimeOptions opts

/-- Convert a run configuration to the runtime `Options` record. -/
def toRuntimeOptions (run : RunConfig) : Options :=
  { execution := run.execution
    device := run.device
    backendProfile? := run.backendProfile?
    showBackend := run.showBackend }

/-- CLI arguments that reproduce a scalar-semantics choice. -/
def scalarArgs : Runtime.ScalarMode → List String
  | .float32 => ["--scalar", "float32"]
  | .ieee32Exec => ["--scalar", "ieee32-exec"]
  | .complex64 => ["--scalar", "complex64"]

/-- CLI arguments that reproduce an execution-mode choice. -/
def executionArgs : Runtime.ExecutionMode → List String
  | .eager => ["--execution", "eager"]
  | .typedGraph => ["--execution", "typed-graph"]

/-- CLI arguments that reproduce a device choice. -/
def deviceArgs : Runtime.Device → List String
  | .cpu => ["--device", "cpu"]
  | .cuda => ["--device", "cuda"]
  | .rocm => ["--device", "rocm"]
  | .metal => ["--device", "metal"]
  | .wasm => ["--device", "wasm"]
  | .tpu => ["--device", "tpu"]
  | .trainium => ["--device", "trainium"]
  | .custom => ["--device", "custom"]
  | .external => ["--device", "external"]

/-- Parse CLI runtime flags into persistent trainer run settings. -/
def parseRuntimeArgs (args : List String) (base : RunConfig := {}) :
    Except String (RunConfig × List String) := do
  let (exec, rest) ←
    TorchLean.Module.ExecConfig.parseWithScalar args base.scalar
  let _ ← match _root_.NN.Backend.BackendProfile.maintainedForDevice? exec.device with
    | some profile => pure profile
    | none =>
        throw s!"device `{exec.device.cliName}` has no maintained runtime profile; use a programmatic backend profile"
  pure
    ({ base with
        scalar := exec.scalar
        execution := exec.execution
        device := exec.device
        backendProfile? := none
        showBackend := exec.showBackend },
      rest)

/-- Resolve runtime flags into a `Trainer.RunConfig` and reject unused trailing arguments. -/
def parseRuntimeArgsOrThrow
    (exeName : String) (args : List String) (base : RunConfig := {}) :
    IO RunConfig := do
  let (cfg, rest) ←
    match parseRuntimeArgs args base with
    | .ok out => pure out
    | .error msg => throw <| IO.userError msg
  CLI.requireNoArgs exeName rest
  pure cfg

/-- Render this persistent run configuration as standard runtime CLI flags. -/
def toArgs (run : RunConfig) : List String :=
  scalarArgs run.scalar ++
  executionArgs run.execution ++
  deviceArgs run.device ++
  (if run.showBackend then ["--show-backend"] else [])

end RunConfig

namespace Config

/-- Build trainer options from an already parsed runtime configuration. -/
def fromRunConfig {σ τ : Shape}
    (run : RunConfig) (task : Task σ τ := .regression) (seed : Nat := 0) :
    Config σ τ :=
  { task := task
    seed := seed
    optimizer := run.optimizer
    scalar := run.scalar
    execution := run.execution
    device := run.device
    backendProfile? := run.backendProfile?
    showBackend := run.showBackend }

end Config

namespace Implementation

/--
Run a callback under runtime scalar semantics that can also be read back to host `Float` tensors.

Trainer methods return ordinary `Float` predictions for display and downstream scripts, even
when the model itself runs under an executable scalar such as `IEEE32Exec`. This dispatcher carries
the extra scalar-readback evidence that `ScalarMode.withRuntime` intentionally does not require.
-/
def withReadableRuntime {β : Type}
    (scalar : Runtime.ScalarMode)
    (k : ∀ {α : Type}, [_root_.Context α] → [DecidableEq Shape] → [ToString α] →
      [Runtime.FromFloat α] →
      [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α] → IO β) :
    IO (Except String β) := do
  match scalar with
  | .float32 =>
      let out ← k (α := Float32)
      pure (.ok out)
  | .ieee32Exec =>
      let out ← k (α := TorchLean.Floats.IEEE32Exec)
      pure (.ok out)
  | .complex64 =>
      pure (.error
        "complex runtime trainer predictions do not yet have a public Float readback path")

namespace Regression

/-- Runtime configuration carried by this trainer. -/
def runConfig {σ τ : Shape} (trainer : Regression σ τ) : Trainer.RunConfig :=
  { toRuntimeSettings := trainer.runtime }

end Regression

namespace OneHotCrossEntropy

/-- Runtime configuration carried by this trainer. -/
def runConfig {σ τ : Shape} (trainer : OneHotCrossEntropy σ τ) : Trainer.RunConfig :=
  { toRuntimeSettings := trainer.runtime }

end OneHotCrossEntropy

namespace Custom

/-- Runtime configuration carried by this trainer. -/
def runConfig {σ τ : Shape} (trainer : Custom σ τ) : Trainer.RunConfig :=
  { toRuntimeSettings := trainer.runtime }

end Custom

end Implementation

/-- Per-training-call options for the trainer API. -/
structure TrainOptions where
  /-- Number of optimizer updates. -/
  steps : Nat := 1
  /--
  Number of dataset items included in one optimizer update.

  For an ordinary sample dataset, each item is one example. A dataset made by `Data.batchDataset`
  already stores fixed-size tensor minibatches as its items; the usual vectorized path therefore
  keeps this option at `1`. Values above one accumulate gradients from several such items before
  updating.
  -/
  batchSize : Nat := 1
  /-- Optional learning-rate schedule, indexed by completed optimizer updates. -/
  scheduler : Option _root_.TorchLean.Trainer.Scheduler.Config := none
  /-- Print step losses every `logEvery` updates; `0` disables stdout step logging. -/
  logEvery : Nat := 0
  /-- Sample CUDA allocator state every this many completed updates; `0` disables sampling. -/
  cudaMemWatch : Nat := 0
  /-- Optional TrainLog artifact destination. Use `.disabled` for stdout-only runs. -/
  log : Training.LogDestination := .disabled
  /-- Title used when writing a TrainLog artifact. -/
  title : String := "Training"
  /-- Free-form notes attached to the TrainLog artifact. -/
  notes : Array String := #[]
  /-- Optional model-state checkpoint loaded before training. -/
  loadCheckpoint? : Option System.FilePath := none
  /-- Optional model-state checkpoint written after training. -/
  saveCheckpoint? : Option System.FilePath := none

namespace TrainOptions

/-- Start training options with a fixed number of optimizer steps. -/
def forSteps (count : Nat) : TrainOptions :=
  { steps := count }

/-- Override stdout step logging cadence. -/
def withLogEvery (opts : TrainOptions) (logEvery : Nat) : TrainOptions :=
  { opts with logEvery := logEvery }

/-- Override the CUDA allocator sampling cadence. -/
def withCudaMemWatch (opts : TrainOptions) (cudaMemWatch : Nat) : TrainOptions :=
  { opts with cudaMemWatch := cudaMemWatch }

/-- Override the requested minibatch size. -/
def withBatchSize (opts : TrainOptions) (batchSize : Nat) : TrainOptions :=
  { opts with batchSize := batchSize }

/-- Apply a learning-rate schedule during this training call. -/
def withScheduler
    (opts : TrainOptions) (scheduler : _root_.TorchLean.Trainer.Scheduler.Config) : TrainOptions :=
  { opts with scheduler := some scheduler }

/-- Run with the optimizer's fixed learning rate. -/
def withoutScheduler (opts : TrainOptions) : TrainOptions :=
  { opts with scheduler := none }

/-- Override the training-log destination. -/
def withLog (opts : TrainOptions) (log : Training.LogDestination) : TrainOptions :=
  { opts with log := log }

/-- Disable TrainLog artifact writing for a training call that will write a richer custom artifact later. -/
def disableLog (opts : TrainOptions) : TrainOptions :=
  { opts with log := .disabled }

/-- Override the training-log title. -/
def withTitle (opts : TrainOptions) (title : String) : TrainOptions :=
  { opts with title := title }

/-- Override the training-log notes. -/
def withNotes (opts : TrainOptions) (notes : Array String) : TrainOptions :=
  { opts with notes := notes }

/-- Load a model-state checkpoint before training. -/
def withLoadCheckpoint (opts : TrainOptions) (path : System.FilePath) : TrainOptions :=
  { opts with loadCheckpoint? := some path }

/-- Save a model-state checkpoint after training. -/
def withSaveCheckpoint (opts : TrainOptions) (path : System.FilePath) : TrainOptions :=
  { opts with saveCheckpoint? := some path }

/-- Lower the public training options to the manual runtime training config. -/
def toTrainConfig (opts : TrainOptions) (optimizer : optim.Optimizer) :
    TorchLean.Trainer.Manual.TrainConfig :=
  { steps := opts.steps
    batchSize := opts.batchSize
    optimizer := optimizer
    scheduler := opts.scheduler
    logEvery := opts.logEvery
    cudaMemWatch := opts.cudaMemWatch }

end TrainOptions

/-- A named classification input used for before/after prediction reporting. -/
structure ClassProbe (σ : Shape) where
  /-- Human-facing probe name. -/
  name : String
  /-- Runtime-polymorphic input tensor. -/
  input : {α : Type} → [_root_.Context α] → [Runtime.FromFloat α] → Tensor α σ
  /-- Expected class index, printed beside the prediction. -/
  expected : Nat

namespace ClassProbe

/-- Convert a single-example class probe into the batched tensor probe used by `trainer.train`. -/
def toBatchedProbe {σ : Shape} (batch : Nat) (probe : ClassProbe σ) :
    Probe (.dim batch σ) :=
  { name := probe.name
    inputText := s!"expected={probe.expected}"
    input := fun {α} _ _ => Tensor.repeatBatch batch (probe.input (α := α))
    expected := some (toString probe.expected) }

end ClassProbe

end Trainer

end TorchLean
