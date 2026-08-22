/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Scalar
public import NN.API.CLI.Training
public import NN.API.Module.Command
public import NN.API.Trainer.Run

/-!
# Model-Zoo Commands

Command-line records used by TorchLean's runnable model examples.

These declarations live with the model zoo because they describe repository
commands and local artifact formats, not the tensor, trainer, or runtime API. Applications can use
the public trainer directly without importing diffusion schedules, PPM paths, or prepared-dataset
defaults.
-/

@[expose] public section

namespace NN.Examples.ModelZoo

open TorchLean

/-! ## Generative-model artifacts -/

/-- Diffusion schedule parameters exposed by the model-zoo diffusion command. -/
structure DiffusionScheduleFlags where
  /-- Number of diffusion timesteps in the schedule. -/
  T : Nat
  /-- First beta value in the schedule. -/
  betaStart : Float
  /-- Final beta value in the schedule. -/
  betaEnd : Float
deriving Repr

namespace DiffusionScheduleFlags

/-- Parse `--T`, `--beta-start`, and `--beta-end`. -/
def parse
    (args : List String)
    (defaultT : Nat := 100)
    (defaultBetaStart : Float := 1e-4)
    (defaultBetaEnd : Float := 0.12) :
    Except String (DiffusionScheduleFlags × List String) := do
  let (T, args) ← CLI.takeNatFlagDefault args "T" defaultT
  let (betaStart, args) ← CLI.takeFloatFlagDefault args "beta-start" defaultBetaStart
  let (betaEnd, args) ← CLI.takeFloatFlagDefault args "beta-end" defaultBetaEnd
  pure ({ T, betaStart, betaEnd }, args)

/-- Stable TrainLog metadata for a diffusion schedule. -/
def trainLogNotes (cfg : DiffusionScheduleFlags) : Array String :=
  #[s!"T={cfg.T}", s!"betaStart={cfg.betaStart}", s!"betaEnd={cfg.betaEnd}"]

end DiffusionScheduleFlags

/-- Optional image artifacts emitted by generation and reconstruction examples. -/
structure ImageArtifactFlags where
  /-- Optional timestep used for reconstruction-from-noise artifacts. -/
  reconstructStep? : Option Nat
  /-- Optional path for an unconditional sample image. -/
  samplePpm? : Option System.FilePath
  /-- Optional path for the clean reference image. -/
  referencePpm? : Option System.FilePath
  /-- Optional path for the noised or intermediate image. -/
  noisyPpm? : Option System.FilePath
  /-- Optional path for the reconstructed image. -/
  reconstructPpm? : Option System.FilePath
deriving Repr

namespace ImageArtifactFlags

/-- Parse image-artifact paths and the optional reconstruction timestep. -/
def parse (args : List String) : Except String (ImageArtifactFlags × List String) := do
  let (reconstructStep?, args) ← CLI.takeNatFlagOnce args "reconstruct-step"
  let (samplePpm?, args) ← CLI.takePathFlagOnce args "sample-ppm"
  let (referencePpm?, args) ← CLI.takePathFlagOnce args "reference-ppm"
  let (noisyPpm?, args) ← CLI.takePathFlagOnce args "noisy-ppm"
  let (reconstructPpm?, args) ← CLI.takePathFlagOnce args "reconstruct-ppm"
  pure
    ({ reconstructStep?, samplePpm?, referencePpm?, noisyPpm?, reconstructPpm? }, args)

/-- Stable TrainLog metadata for requested image artifacts. -/
def trainLogNotes (cfg : ImageArtifactFlags) : Array String :=
  (match cfg.reconstructStep? with | none => #[] | some t => #[s!"reconstructStep={t}"]) ++
  (match cfg.samplePpm? with | none => #[] | some p => #[s!"samplePpm={p}"]) ++
  (match cfg.referencePpm? with | none => #[] | some p => #[s!"referencePpm={p}"]) ++
  (match cfg.noisyPpm? with | none => #[] | some p => #[s!"noisyPpm={p}"]) ++
  (match cfg.reconstructPpm? with | none => #[] | some p => #[s!"reconstructPpm={p}"])

end ImageArtifactFlags

/-! ## Prepared-data and diagnostic artifacts -/

/-- Train/test tensor paths and row counts for paired-NPY scientific examples. -/
structure PairedNpyEvalFlags where
  /-- Number of rows loaded from the prepared training tensors. -/
  trainRows : Nat
  /-- Number of rows loaded from the prepared held-out tensors. -/
  testRows : Nat
  /-- Prefix length used for deterministic train/test loss reports. -/
  evalRows : Nat
  /-- Training input `.npy` path. -/
  trainX : System.FilePath
  /-- Training target `.npy` path. -/
  trainY : System.FilePath
  /-- Held-out input `.npy` path. -/
  testX : System.FilePath
  /-- Held-out target `.npy` path. -/
  testY : System.FilePath
deriving Repr

namespace PairedNpyEvalFlags

/-- Parse train/test paths and row counts for paired NPY tensors. -/
def parse
    (args : List String)
    (defaultTrainX defaultTrainY defaultTestX defaultTestY : System.FilePath)
    (defaultTrainRows defaultTestRows : Nat)
    (defaultEvalRows : Nat := 16) :
    Except String (PairedNpyEvalFlags × List String) := do
  let (trainRows, args) ← CLI.takeNatFlagDefault args "train-rows" defaultTrainRows
  let (testRows, args) ← CLI.takeNatFlagDefault args "test-rows" defaultTestRows
  let (evalRows, args) ← CLI.takeNatFlagDefault args "eval-rows" defaultEvalRows
  let (trainX, args) ← CLI.takePathFlagDefault args "x" defaultTrainX
  let (trainY, args) ← CLI.takePathFlagDefault args "y" defaultTrainY
  let (testX, args) ← CLI.takePathFlagDefault args "test-x" defaultTestX
  let (testY, args) ← CLI.takePathFlagDefault args "test-y" defaultTestY
  pure ({ trainRows, testRows, evalRows, trainX, trainY, testX, testY }, args)

/-- Stable TrainLog metadata for paired train/test NPY tensors. -/
def trainLogNotes (cfg : PairedNpyEvalFlags) : Array String :=
  #[
    s!"train_rows={cfg.trainRows}",
    s!"test_rows={cfg.testRows}",
    s!"eval_rows={cfg.evalRows}",
    s!"train_x={cfg.trainX}",
    s!"train_y={cfg.trainY}",
    s!"test_x={cfg.testX}",
    s!"test_y={cfg.testY}"
  ]

end PairedNpyEvalFlags

/-- Optional CSV artifact path for commands that emit one tabular diagnostic. -/
structure CsvArtifactFlags where
  /-- CSV path for the diagnostic artifact. -/
  plotCsv : System.FilePath
deriving Repr

namespace CsvArtifactFlags

/-- Parse the optional `--plot-csv` artifact path. -/
def parse
    (args : List String)
    (defaultPlotCsv : System.FilePath) :
    Except String (CsvArtifactFlags × List String) := do
  let (plotCsv, args) ← CLI.takePathFlagDefault args "plot-csv" defaultPlotCsv
  pure ({ plotCsv }, args)

end CsvArtifactFlags

/-- Prepared NPY feature/target paths and their row budget. -/
structure NpyDataFlags where
  /-- Prepared feature or image tensor path. -/
  xPath : System.FilePath
  /-- Prepared label or target tensor path. -/
  yPath : System.FilePath
  /-- Number of rows to read from the arrays. -/
  nRows : Nat
  /-- Data-loader seed. -/
  seed : Nat
deriving Repr

namespace NpyDataFlags

/-- Parse `--seed`, `--n-total`, `--x`, and `--y` for an NPY-backed example. -/
def parse
    (args : List String)
    (defaultX defaultY : System.FilePath)
    (defaultRows : Nat) :
    Except String (NpyDataFlags × List String) := do
  let (seed, args) ← CLI.takeSeed args (default := 0)
  let (nRows, args) ← CLI.takeNatFlagDefault args "n-total" defaultRows
  let (xPath, args) ← CLI.takePathFlagDefault args "x" defaultX
  let (yPath, args) ← CLI.takePathFlagDefault args "y" defaultY
  pure ({ xPath, yPath, nRows, seed }, args)

/-- Stable TrainLog metadata for an NPY-backed dataset branch. -/
def trainLogNotes (cfg : NpyDataFlags) (datasetName : String) : Array String :=
  #[s!"data={datasetName}", s!"x={cfg.xPath}", s!"y={cfg.yPath}", s!"nRows={cfg.nRows}"]

end NpyDataFlags

/-- Prepared image datasets understood by the built-in image-model commands. -/
inductive ImageDatasetChoice where
  /-- Prepared 64x64 RGB image tensors. -/
  | imagenet64
  /-- Prepared CIFAR-10 32x32 RGB tensors. -/
  | cifar10
deriving Repr, BEq

namespace ImageDatasetChoice

/-- Parse the dataset selector and reject ambiguous combinations. -/
def parse (args : List String) : Except String (ImageDatasetChoice × List String) := do
  let (dataset?, args) ← CLI.takeFlagValueOnce args "dataset"
  let (cifarFlag, args) ← CLI.takeBoolFlagOnce args "cifar10"
  let (imagenetFlag, args) ← CLI.takeBoolFlagOnce args "imagenet64"
  match dataset?, cifarFlag, imagenetFlag with
  | some _, true, _ | some _, _, true | none, true, true =>
      throw "choose only one dataset selector: --dataset, --cifar10, or --imagenet64"
  | some raw, false, false =>
      match raw with
      | "imagenet64" | "imagenet" | "imagenette64" => pure (.imagenet64, args)
      | "cifar10" | "cifar" => pure (.cifar10, args)
      | _ => throw s!"unknown --dataset {raw}; expected imagenet64 or cifar10"
  | none, true, false => pure (.cifar10, args)
  | none, false, true => pure (.imagenet64, args)
  | none, false, false => pure (.imagenet64, args)

end ImageDatasetChoice

/-- Prepared forecasting-window paths and report controls. -/
structure ForecastWindowDataFlags where
  /-- Prepared input-window tensor path. -/
  xPath : System.FilePath
  /-- Prepared target-window tensor path. -/
  yPath : System.FilePath
  /-- Number of forecasting windows to use. -/
  windows : Nat
  /-- Window index used for before/after forecast display. -/
  reportOffset : Nat
  /-- Data-loader seed. -/
  seed : Nat
deriving Repr

namespace ForecastWindowDataFlags

/-- Stable TrainLog metadata for forecasting-window datasets. -/
def trainLogNotes (cfg : ForecastWindowDataFlags) : Array String :=
  #[
    s!"windows={cfg.windows}",
    s!"report_index={cfg.reportOffset}",
    s!"x={cfg.xPath}",
    s!"y={cfg.yPath}"
  ]

end ForecastWindowDataFlags

/-! ## Data plus training controls -/

/-- Standard model-training flags plus a reproducibility seed. -/
structure SeededTrainFlags extends TorchLean.CLI.Training.OptimizerOptions where
  /-- Seed used for initialization, synthetic data, or row order. -/
  seed : Nat
deriving Repr

/-- Parse the standard model-training flags together with `--seed`. -/
def parseSeededTrainFlags
    (exeName : String)
    (args : List String)
    (defaultLogPath : System.FilePath)
    (defaultSteps : Nat := 1)
    (defaultLr : Float := 1e-3)
    (allowZeroSteps : Bool := false) :
    Except String (SeededTrainFlags × List String) := do
  let (seed, args) ← CLI.takeSeed args 0
  let (train, args) ←
    TorchLean.CLI.Training.OptimizerOptions.parse exeName args defaultLogPath defaultSteps defaultLr
      allowZeroSteps
  pure ({ toOptimizerOptions := train, seed }, args)

/-- Fixed-step training flags paired with an NPY dataset. -/
structure NpyLoggedTrainFlags extends TorchLean.CLI.Training.RunOptions, NpyDataFlags where
deriving Repr

namespace NpyLoggedTrainFlags

/-- Parse NPY data and logged-training flags, then reject unused command arguments. -/
def parse
    (exeName : String)
    (args : List String)
    (defaultLogPath : System.FilePath)
    (defaultSteps : Nat)
    (parseData : List String → Except String (NpyDataFlags × List String)) :
    Except String NpyLoggedTrainFlags := do
  let (data, rest) ← parseData args
  let (train, rest) ←
    TorchLean.CLI.Training.RunOptions.parse exeName rest defaultLogPath defaultSteps
  CLI.checkNoArgs rest
  pure { toRunOptions := train, toNpyDataFlags := data }

end NpyLoggedTrainFlags

/-- Optimizer/training flags paired with an NPY dataset. -/
structure NpyModelTrainFlags extends TorchLean.CLI.Training.OptimizerOptions, NpyDataFlags where
deriving Repr

namespace NpyModelTrainFlags

/-- Parse NPY data and the standard model-training flags. -/
def parse
    (exeName : String)
    (args : List String)
    (defaultLogPath : System.FilePath)
    (defaultSteps : Nat := 1)
    (defaultLr : Float := 1e-3)
    (parseData : List String → Except String (NpyDataFlags × List String)) :
    Except String (NpyModelTrainFlags × List String) := do
  let (data, rest) ← parseData args
  let (train, rest) ←
    TorchLean.CLI.Training.OptimizerOptions.parse exeName rest defaultLogPath defaultSteps defaultLr
  pure ({ toOptimizerOptions := train, toNpyDataFlags := data }, rest)

end NpyModelTrainFlags

/-- Optimizer/training flags paired with a forecasting-window dataset. -/
structure ForecastWindowModelTrainFlags
    extends TorchLean.CLI.Training.OptimizerOptions, ForecastWindowDataFlags where
deriving Repr

namespace ForecastWindowModelTrainFlags

/-- Parse forecasting data and the standard model-training flags. -/
def parse
    (exeName : String)
    (args : List String)
    (defaultLogPath : System.FilePath)
    (defaultSteps : Nat := 100)
    (defaultLr : Float := 0.01)
    (parseData : List String → Except String (ForecastWindowDataFlags × List String)) :
    Except String (ForecastWindowModelTrainFlags × List String) := do
  let (data, rest) ← parseData args
  let (train, rest) ←
    TorchLean.CLI.Training.OptimizerOptions.parse exeName rest defaultLogPath defaultSteps defaultLr
  pure ({ toOptimizerOptions := train, toForecastWindowDataFlags := data }, rest)

end ForecastWindowModelTrainFlags

/-- Optimizer/training flags for a model command that reads one supervised CSV. -/
structure CsvTrainFlags extends TorchLean.CLI.Training.OptimizerOptions where
  /-- CSV file containing model inputs and targets. -/
  csvPath : System.FilePath
  /-- Seed used for model initialization and data shuffling. -/
  seed : Nat
deriving Repr

/-- Parse a CSV path, seed, and the standard model-training flags. -/
def parseCsvTrainFlags
    (exeName : String)
    (args : List String)
    (defaultCsv defaultLogPath : System.FilePath)
    (defaultSteps : Nat := 1)
    (defaultLr : Float := 1e-3)
    (allowZeroSteps : Bool := false) :
    Except String (CsvTrainFlags × List String) := do
  let (csv?, args) ← CLI.takePathFlagOnce args "csv"
  let csvPath := csv?.getD defaultCsv
  let (seed, args) ← CLI.takeSeed args 0
  let (train, args) ←
    TorchLean.CLI.Training.OptimizerOptions.parse exeName args defaultLogPath defaultSteps defaultLr
      allowZeroSteps
  pure ({ toOptimizerOptions := train, csvPath, seed }, args)

end NN.Examples.ModelZoo

namespace TorchLean.CLI.Training.RunOptions

/-- Convert parsed command options to the options consumed by `Trainer.train`. -/
def toTrainerOptions
    (opts : CLI.Training.RunOptions)
    (enableLog : Bool := true)
    (logEvery : Nat := 0)
    (title : String := "Training")
    (notes : Array String := #[]) :
    Trainer.TrainOptions :=
  { steps := opts.steps
    batchSize := opts.batchSize
    log := if enableLog then opts.log else .disabled
    logEvery
    cudaMemWatch := opts.cudaMemWatch
    title
    notes }

/-- Enable logging only when the command line explicitly contains `--log`. -/
def toTrainerOptionsWhenRequested
    (opts : CLI.Training.RunOptions)
    (args : List String)
    (logEvery : Nat := 0)
    (title : String := "Training")
    (notes : Array String := #[]) :
    Trainer.TrainOptions :=
  opts.toTrainerOptions (enableLog := CLI.hasFlagValue args "log")
    (logEvery := logEvery) (title := title) (notes := notes)

end TorchLean.CLI.Training.RunOptions

namespace TorchLean.CLI.Training.OptimizerOptions

/-- Convert parsed optimizer and command options to the options consumed by `Trainer.train`. -/
def toTrainerOptions
    (opts : CLI.Training.OptimizerOptions)
    (logEvery : Nat := 0)
    (title : String := "Training")
    (notes : Array String := #[]) :
    Trainer.TrainOptions :=
  opts.toRunOptions.toTrainerOptions (logEvery := logEvery) (title := title) (notes := notes)

end TorchLean.CLI.Training.OptimizerOptions
