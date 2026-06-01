/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

-- shake: keep-all

public import NN.API.Core
public import NN.API.Runtime
public import NN.Runtime.Training.Log
public import NN.Tensor.API

import Mathlib.Algebra.Order.Algebra

/-!
# `NN.API.Common`

Shared helpers used across TorchLean workflows and entrypoints: typed tensor constructors, casting
between scalar backends, and `Except`/`IO` glue.
-/

@[expose] public section

namespace NN
namespace API

namespace Common

open Spec

/-!
## Common Helpers For Workflows And Entry Points

This module contains helper functions that show up repeatedly in executable workflows.

What belongs here:
- casting `Spec.Tensor Float` into an arbitrary scalar backend `α` (given `Float → α`)
- building typed tensors from raw lists (`tensorF`, `vecF`, `matF`)
- bridging `Except String` into `IO` (`orThrow`)
- running model commands under a chosen scalar backend via `--dtype` (see `NN.API.DType`)

What does *not* belong here:
- core semantics/proofs (those live under `NN` / `NN.Proofs`)
- command-specific CLI parsing that belongs with a model or tool entrypoint
-/

/-!
### PyTorch Mapping Notes

If you're coming from PyTorch, this module plays the role of the shared utility layer often written
around:
- `torch.tensor([...], dtype=..., device=...)`
- choosing a dtype/backend based on CLI flags
- `try/except` wrappers for dataset loading and runnable workflow code

The main difference is that TorchLean tracks shapes in types, so tensor constructors here return
`Except String ...` rather than silently reshaping.
-/

/-- Cast a tensor elementwise while preserving its type-level shape. -/
def castTensor {α : Type} (cast : Float → α) {s : Spec.Shape} (t : Spec.Tensor Float s) :
    Spec.Tensor α s :=
  Spec.mapTensor cast t

/--
Convert an `Except String α` into an `IO α`, raising a tagged `userError` on failure.

This is handy in `main` functions where we want the process to exit with a readable message.
-/
def orThrow {α : Type} (tag : String := "Example") : Except String α → IO α
  | .ok a => pure a
  | .error msg => throw <| IO.userError s!"{tag}: {msg}"

/-- Standard location for model-example training logs under `data/model_zoo`. -/
def modelZooTrainLog (stem : String) : System.FilePath :=
  System.FilePath.mk s!"data/model_zoo/{stem}_trainlog.json"

/--
Fail with a tagged `userError` if a boolean condition is false.

Executable workflows use this for named precondition checks such as
`Common.check exeName "loss finite" (loss == loss)`.
-/
def check (tag msg : String) (b : Bool) : IO Unit :=
  unless b do throw <| IO.userError s!"{tag}: {msg}"

/-- Validate that a natural-number CLI flag is positive. -/
def requirePositiveNatFlag (exeName flag : String) (value : Nat) : Except String Unit := do
  if value = 0 then
    throw s!"{exeName}: --{flag} must be > 0"

/-- Write a prepared `TrainLog` JSON artifact and report the file path. -/
def writeTrainLog (path : System.FilePath) (log : _root_.Runtime.Training.TrainLog) : IO Unit := do
  _root_.Runtime.Training.TrainLog.writeJson path log
  IO.println s!"  wrote TrainLog JSON: {path}"

/-- Write a prepared `TrainLog` to a destination that may be disabled. -/
def writeTrainLogTo (dest : _root_.Runtime.Training.LogDestination)
    (log : _root_.Runtime.Training.TrainLog) : IO Unit := do
  _root_.Runtime.Training.LogDestination.writeTrainLog dest log
  match dest.path? with
  | some path => IO.println s!"  wrote TrainLog JSON: {path}"
  | none => IO.println "  TrainLog JSON disabled"

/--
Write a standard JSON training artifact for routines that record an initial and final loss.

The function uses `Runtime.Training.TrainLog.beforeAfterLoss` and the stable TrainLog JSON format.
The output schema is independent of the model, dataset, and runtime backend.
-/
def writeBeforeAfterLossLog (path : System.FilePath) (title : String) (steps : Nat)
    (loss0 loss1 : Float) (notes : Array String := #[]) : IO Unit := do
  let log := _root_.Runtime.Training.TrainLog.beforeAfterLoss title steps loss0 loss1 notes
  writeTrainLog path log

/--
Write a before/after loss log to an explicit logging destination.

`LogDestination.disabled` is a no-op, mirroring `wandb disabled` for runs where metrics should stay
on stdout only.
-/
def writeBeforeAfterLossLogTo (dest : _root_.Runtime.Training.LogDestination)
    (title : String) (steps : Nat) (loss0 loss1 : Float) (notes : Array String := #[]) :
    IO Unit := do
  let log := _root_.Runtime.Training.TrainLog.beforeAfterLoss title steps loss0 loss1 notes
  writeTrainLogTo dest log

/-- Print the standard before/after loss summary returned by `fit` helpers. -/
def printFitReport {α : Type} [ToString α] (steps : Nat)
    (report : TorchLean.Trainer.FitReport α) : IO Unit :=
  IO.println s!"  steps={steps} loss0={report.before} loss1={report.after}"

/-- Print the standard first/last loss summary for a scalar training curve. -/
def printCurveLossSummary (steps : Nat) (curve : _root_.Runtime.Training.Curve) : IO Unit := do
  let loss0 := curve.values.getD 0 0.0
  let lossN := curve.values.getD (curve.values.size - 1) loss0
  IO.println s!"  steps={steps} loss0={loss0} loss{steps}={lossN}"

/-- Write a one-series scalar curve as a standard `TrainLog` JSON artifact. -/
def writeCurveLog (path : System.FilePath) (title : String)
    (curve : _root_.Runtime.Training.Curve) (seriesName : String := "loss")
    (notes : Array String := #[]) : IO Unit := do
  let log := curve.toTrainLog title seriesName (notes := notes)
  writeTrainLog path log

/-- Write a one-series scalar curve to an explicit logging destination. -/
def writeCurveLogTo (dest : _root_.Runtime.Training.LogDestination) (title : String)
    (curve : _root_.Runtime.Training.Curve) (seriesName : String := "loss")
    (notes : Array String := #[]) : IO Unit := do
  let log := curve.toTrainLog title seriesName (notes := notes)
  writeTrainLogTo dest log

/-- Common CLI result for training commands that accept `--steps`/`--epochs` and `--log`. -/
structure LoggedTrainFlags where
  /-- Number of optimizer updates. -/
  steps : Nat
  /-- Logging destination. Use `--log false` / `off` / `none` to disable. -/
  log : _root_.Runtime.Training.LogDestination
  /-- Path where the JSON `TrainLog` should be written. -/
  logPath : System.FilePath
  /-- CUDA allocator telemetry cadence shared by fixed-sample and custom training loops. -/
  cudaMemWatch : Nat := 0
deriving Repr

/--
Parse common training flags: positive `--steps`/`--epochs` plus optional `--log`.

`--log <path>` writes the standard local JSON artifact. `--log false`, `--log off`, or
`--log none` disables artifact writing while preserving the parsed step count.
-/
def parseLoggedTrainFlags (exeName : String) (args : List String)
    (defaultLogPath : System.FilePath) (defaultSteps : Nat := 1)
    (allowZeroSteps : Bool := false) :
    Except String (LoggedTrainFlags × List String) := do
  let (logRaw?, args) ← CLI.takeFlagValueOnce args "log"
  let (steps, args) ← CLI.takeStepsOrEpochs args defaultSteps
  let (cudaMemWatch?, args) ← CLI.takeNatFlagOnce args "cuda-mem-watch"
  unless allowZeroSteps do
    requirePositiveNatFlag exeName "steps/--epochs" steps
  let log := _root_.Runtime.Training.LogDestination.parse? defaultLogPath logRaw?
  pure ({ steps := steps, log := log, logPath := log.pathD defaultLogPath,
          cudaMemWatch := cudaMemWatch?.getD 0 }, args)

/--
Training flags shared by the runnable model examples.

Most model commands should not each define their own parser for the same knobs.  They can parse
model-specific data flags first, then call `parseModelTrainFlags` for the common optimizer loop:

- `--steps` / `--epochs`: how many optimizer passes the example should run;
- `--log`: where to write a TrainLog JSON artifact;
- `--lr`: Adam learning rate.

Special examples can still extend this record with extra fields, but the default path stays one
shared parser rather than one local `TrainOptions` clone per model.
-/
structure ModelTrainFlags where
  /-- Shared step/epoch count and logging destination. -/
  train : LoggedTrainFlags
  /-- Learning rate for the default Adam optimizer used by examples. -/
  lr : Float
  /--
  Print CUDA allocator telemetry every `N` training steps when running on CUDA.

  `0` disables reporting. The flag lives with the common model-training options because allocator
  telemetry is a runtime concern shared by all CUDA examples.
  -/
  cudaMemWatch : Nat := 0
deriving Repr

/-- Parse the standard model-training flags: `--steps`/`--epochs`, `--log`, `--lr`, and CUDA telemetry. -/
def parseModelTrainFlags (exeName : String) (args : List String)
    (defaultLogPath : System.FilePath) (defaultSteps : Nat := 1) (defaultLr : Float := 1e-3)
    (allowZeroSteps : Bool := false) :
    Except String (ModelTrainFlags × List String) := do
  let (train, args) ← parseLoggedTrainFlags exeName args defaultLogPath defaultSteps allowZeroSteps
  let (lr?, args) ← CLI.takeFloatFlagOnce args "lr"
  let lr := lr?.getD defaultLr
  if lr <= 0.0 then
    throw s!"{exeName}: --lr must be > 0"
  pure ({ train := train, lr := lr, cudaMemWatch := train.cudaMemWatch }, args)

/-! ### Progress Cadence Helpers -/

/--
Return whether a completed training step should emit a progress report.

The convention is shared across example trainers: `logEvery = 0` disables progress output; otherwise
we log at exact multiples of the completed-step count.
-/
def shouldLogStep (logEvery done : Nat) : Bool :=
  logEvery != 0 && done % logEvery == 0

/-! ### CUDA Memory Watch Helpers -/

/--
Choose a CUDA memory-watch cadence for public examples.

Users can pass `--cuda-mem-watch N` to choose an exact cadence. When no cadence is supplied, long
CUDA runs sample about ten times over the requested training horizon. Short runs and CPU runs stay
quiet by default, so the examples do not print allocator telemetry unless it is likely to be useful.
-/
def effectiveCudaMemWatch (opts : _root_.Runtime.Autograd.Torch.Options)
    (steps requested : Nat) : Nat :=
  if requested != 0 then
    requested
  else if opts.useGpu && steps >= 1000 then
    Nat.max 1 (steps / 10)
  else
    0

/--
State for a simple CUDA-memory drift detector.

The first reported sample becomes the baseline.  Later samples compare current CUDA free memory
against that baseline and warn once if the observed per-step drop projects failure before the
requested run length.
-/
structure CudaMemWatchState where
  firstStep : Nat
  firstFreeBytes : Nat
  warned : Bool
deriving Repr

/--
Maybe print a one-line CUDA allocator report.

The report samples the native allocator at a fixed cadence and warns if the observed free-memory
slope would cross zero before the requested training horizon.
-/
def reportCudaMemWatch (opts : _root_.Runtime.Autograd.Torch.Options)
    (watchEvery totalSteps done : Nat) (state? : Option CudaMemWatchState) :
    IO (Option CudaMemWatchState) := do
  if !opts.useGpu || watchEvery = 0 || (done != 0 && done % watchEvery != 0) then
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
Default Adam optimizer constructor used by supervised and vision examples.

The reusable part is the optimizer convention, not the model.  Individual examples still own their
architecture and loss, while this helper keeps the Adam hyperparameter spelling identical across
MLP, CNN, ResNet, ViT, and similar model commands.
-/
def adamOptimizer {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    (cast : Float → α) (ps : List Shape) (lr : Float) :
    TorchLean.Optim.Optimizer α ps :=
  TorchLean.Optim.adam (α := α) (paramShapes := ps)
    (lr := cast lr) (beta1 := cast 0.9) (beta2 := cast 0.999) (epsilon := cast 1e-8)

/--
Run an executable with the standard TorchLean runtime parser, using the polymorphic scalar path by
default and switching to the `Float` path when requested.

This is the common shape for public examples that support all executable scalar backends, but need
the `Float` path for CUDA bridges, decoded reports, or JSON artifacts whose metrics are stored as
`Float`.
-/
def runAnyOrFloat
    (exeName : String) (args : List String)
    (preferFloat : List String → Bool)
    (banner : TorchLean.Options → String)
    (anyK :
      ∀ {α : Type}, [Semantics.Scalar α] → [DecidableEq Spec.Shape] → [ToString α] →
        [Runtime.Scalar α] →
        (cast : Float → α) → (opts : TorchLean.Options) → (rest : List String) → IO Unit)
    (floatK : (opts : TorchLean.Options) → (rest : List String) → IO Unit)
    (printOk : Bool := true) : IO UInt32 := do
  let runOpts : TorchLean.Module.RunOptions :=
    { banner? := some banner, printOk := printOk }
  if preferFloat args then
    TorchLean.Module.run exeName args (.float floatK) runOpts
  else
    TorchLean.Module.run exeName args (.any anyK) runOpts

/--
Run an executable on the concrete `Float` runtime path.

We use this for runnable training commands that produce Float-valued artifacts: CPU/CUDA eager
execution, native kernels, and JSON loss curves. Commands that need to expose another scalar backend
can use `runAnyOrFloat`.
-/
def runFloat
    (exeName : String) (args : List String)
    (banner : TorchLean.Options → String)
    (k : (opts : TorchLean.Options) → (rest : List String) → IO Unit)
    (printOk : Bool := true) : IO UInt32 :=
  TorchLean.Module.run exeName args (.float k)
    { banner? := some banner, printOk := printOk }

/-! ### Common model-run parsers -/

/-- Common arguments for a model command that reads one supervised CSV. -/
structure CsvModelTrainFlags where
  /-- CSV file containing model inputs and targets. -/
  csvPath : System.FilePath
  /-- Seed used for model initialization and data shuffling. -/
  seed : Nat
  /-- Shared training flags: steps, log destination, learning rate, and CUDA telemetry. -/
  train : ModelTrainFlags
deriving Repr

/--
Parse common flags for a supervised CSV model runner.

This keeps the parser shape out of individual models.  Model files still choose their default CSV,
default log path, default step count, and default learning rate.
-/
def parseCsvModelTrainFlags (exeName : String) (args : List String)
    (defaultCsv : System.FilePath) (defaultLogPath : System.FilePath)
    (defaultSteps : Nat := 1) (defaultLr : Float := 1e-3)
    (allowZeroSteps : Bool := false) :
    Except String (CsvModelTrainFlags × List String) := do
  let (csv?, args) ← CLI.takePathFlagOnce args "csv"
  let csvPath := csv?.getD defaultCsv
  let (seed, args) ← CLI.takeSeed args 0
  let (train, args) ← parseModelTrainFlags exeName args defaultLogPath defaultSteps defaultLr
    allowZeroSteps
  pure ({ csvPath := csvPath, seed := seed, train := train }, args)

/-- List generator: `[0, 1, ..., n-1]` mapped through `f`. -/
def listGen {α : Type} (n : Nat) (f : Nat → α) : List α :=
  (List.range n).map f

/--
Build an `N`-D tensor from a raw list of floats and cast it into the chosen scalar backend.

Fails if `xs.length ≠ numel(dims)`.
-/
def tensorF {α : Type} [Context α] (cast : Float → α) (dims : List Nat) (xs : List Float) :
    Except String (Spec.Tensor α (NN.Tensor.shapeOfDims dims)) := do
  let tF ← NN.Tensor.tensorND (α := Float) dims xs
  pure (Spec.mapTensor cast tF)

/--
Generate an `N`-D tensor by calling `f` for each element index, then cast into the chosen backend.

The function `f` is indexed by the *flat* element index `0..numel-1`.
-/
def tensorFGen {α : Type} [Context α] (cast : Float → α) (dims : List Nat) (f : Nat → Float) :
    Except String (Spec.Tensor α (NN.Tensor.shapeOfDims dims)) :=
  let n := NN.Tensor.numelDims dims
  tensorF (α := α) cast dims (listGen n f)

/--
Generate an `N`-D tensor by calling `f` for each flat element index, with no failure case.

This is the “total” sibling of `tensorFGen`: since we generate exactly `numel(dims)` values,
the reshape cannot fail, so we avoid an `Except`.
When you want to build a deterministic constant tensor for an example, this is usually the
right tool.
-/
def tensorFGen! {α : Type} [Context α] (cast : Float → α) (dims : List Nat) (f : Nat → Float) :
    Spec.Tensor α (NN.Tensor.shapeOfDims dims) :=
  let xs : List Float := listGen (NN.Tensor.numelDims dims) f
  have hLen : xs.length = NN.Tensor.numelDims dims := by
    simp [xs, listGen]
  let tF : Spec.Tensor Float (NN.Tensor.shapeOfDims dims) :=
    NN.Tensor.tensorNDOfLenEq (α := Float) (dims := dims) (xs := xs) hLen
  Spec.mapTensor cast tF

/--
Generate a tensor of a known shape `s` by calling `f` for each flat element index, then cast into
the chosen backend.

This packages the standard shape-cast pattern used when a tensor is generated from flat indices:

```lean
let xDyn : Tensor α (shapeOfDims s.toList) := ...
let x : Tensor α s := by simpa using xDyn
```
-/
def tensorFGenShape! {α : Type} [Context α] (cast : Float → α) (s : Spec.Shape) (f : Nat → Float) :
    Spec.Tensor α s := by
  simpa [NN.Tensor.shapeOfDims_toList] using
    (tensorFGen! (α := α) cast (_root_.Spec.Shape.toList s) f)

/--
1D vector tensor constructor specialized to shape `Vec n`.

Fails if `xs.length ≠ n`.
-/
def vecF {α : Type} [Context α] (cast : Float → α) (n : Nat) (xs : List Float) :
    Except String (Spec.Tensor α (.dim n .scalar)) := do
  let t ← tensorF (α := α) cast [n] xs
  pure (by simpa [NN.Tensor.shapeOfDims] using t)

/-- Generator variant of `vecF`. -/
def vecFGen {α : Type} [Context α] (cast : Float → α) (n : Nat) (f : Nat → Float) :
    Except String (Spec.Tensor α (.dim n .scalar)) := do
  let t ← tensorFGen (α := α) cast [n] f
  pure (by simpa [NN.Tensor.shapeOfDims] using t)

/--
2D matrix tensor constructor specialized to shape `Mat rows cols`.

Fails if `xs.length ≠ rows * cols`.
-/
def matF {α : Type} [Context α] (cast : Float → α) (rows cols : Nat) (xs : List Float) :
    Except String (Spec.Tensor α (.dim rows (.dim cols .scalar))) := do
  let t ← tensorF (α := α) cast [rows, cols] xs
  pure (by simpa [NN.Tensor.shapeOfDims] using t)

/-- Generator variant of `matF`. -/
def matFGen {α : Type} [Context α] (cast : Float → α) (rows cols : Nat) (f : Nat → Float) :
    Except String (Spec.Tensor α (.dim rows (.dim cols .scalar))) := do
  let t ← tensorFGen (α := α) cast [rows, cols] f
  pure (by simpa [NN.Tensor.shapeOfDims] using t)

/--
Run a workflow once under a dtype selected from `args` (via `--dtype` / `--float32-mode`).

This logs the chosen dtype and then calls `k` with a cast function `Float → α` for the selected
scalar backend.

In particular:
- `--dtype=float` selects Lean's builtin `Float` (trusted semantics, executable),
- `--dtype=float32` selects TorchLean's verified IEEE32 executable semantics,
- `--dtype=complex` selects TorchLean's parametric complex scalar over Float32,
- `--dtype=real` selects `ℝ` (proof-only; errors at runtime).
-/
def runWithDType (title : String) (args : List String)
    (k :
      ∀ {α : Type},
        [API.Semantics.Scalar α] →
        [DecidableEq Spec.Shape] →
        [ToString α] →
        (Float → α) → IO Unit) :
    IO Unit := do
  IO.println s!"=== {title} workflow ==="
  let (dtype, _args') ←
    match NN.API.DType.parseAndStrip args with
    | .ok v => pure v
    | .error msg => throw <| IO.userError msg
  NN.API.DType.log dtype
  match (← NN.API.DType.withExec dtype (fun {α} _ _ _ cast => k (α := α) cast)) with
  | .ok () => pure ()
  | .error msg => throw <| IO.userError msg

/--
Like `runWithDType`, but also provides an `API.Runtime.Scalar α` instance.

Use this when your workflow uses numeric literals (`1.0`, `-3.5`, etc.) at runtime.
-/
def runWithRuntimeDType (title : String) (args : List String)
    (k : ∀ {α : Type}, [API.Semantics.Scalar α] → [DecidableEq Spec.Shape] → [ToString α] →
      [API.Runtime.Scalar α] → IO Unit) :
    IO Unit := do
  IO.println s!"=== {title} workflow ==="
  let (dtype, _args') ←
    match NN.API.DType.parseAndStrip args with
    | .ok v => pure v
    | .error msg => throw <| IO.userError msg
  NN.API.DType.log dtype
  match (← NN.API.DType.withRuntime dtype (fun {α} _ _ _ _ => k (α := α))) with
  | .ok () => pure ()
  | .error msg => throw <| IO.userError msg

/-- Entry-point alias for runnable binaries that need runtime float-literal injection. -/
abbrev mainWithRuntimeDType := runWithRuntimeDType

end Common

end API
end NN
