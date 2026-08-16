/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.Examples.ModelZoo

/-!
# Shared Model Training Commands

Example-side command runners for built-in model-zoo entries.

The trainer API provides `Trainer.new`, `trainer.train`, and trained prediction handles. This
file owns repository command plumbing: parse model-zoo flags, check local files, run training, and
print the standard summary.
-/

@[expose] public section

namespace NN.Examples.Models.TrainCommand

open TorchLean

/-- Run one parsed model-training command and finish with access to runtime flags and result. -/
def runParsedWith {φ ρ : Type}
    (exeName : String)
    (args : List String)
    (parseFlags : List String → Except String (φ × List String))
    (banner : Options → String)
    (train : Options → φ → IO ρ)
    (finish : Options → φ → ρ → IO Unit) :
    IO UInt32 :=
  Module.Command.runFloat32 exeName args
    (banner := banner)
    (k := fun opts rest => do
      let (flags, rest) ← ModelZoo.orThrow exeName <| parseFlags rest
      CLI.requireNoArgs exeName rest
      let result ← train opts flags
      finish opts flags result)

/-- Run one parsed model-training command when the final printer only needs the trained result. -/
def runParsed {φ ρ : Type}
    (exeName : String)
    (args : List String)
    (parseFlags : List String → Except String (φ × List String))
    (banner : Options → String)
    (train : Options → φ → IO ρ)
    (print : ρ → IO Unit) :
    IO UInt32 :=
  runParsedWith exeName args parseFlags banner train (fun _ _ trained => print trained)

/-- CSV-backed regression command using the public trainer API. -/
def regressionCsv {σ τ : Shape}
    (exeName : String)
    (args : List String)
    (defaultCsv : System.FilePath)
    (defaultLogPath : System.FilePath)
    (defaultSteps : Nat := 1)
    (defaultLr : Float := 1e-3)
    (banner : Options → String)
    (train : Options → ModelZoo.CsvTrainFlags →
      IO (Trainer.TrainResult σ τ)) :
    IO UInt32 :=
  runParsed exeName args
    (fun rest =>
      ModelZoo.parseCsvTrainFlags exeName rest defaultCsv defaultLogPath defaultSteps defaultLr)
    banner train (fun result => result.printSummary)

/-- NPY-backed classifier command using the public trainer API. -/
def classificationNpy
    (exeName : String)
    (args : List String)
    (parseFlags : List String → Except String (ModelZoo.NpyModelTrainFlags × List String))
    (banner : Options → String)
    (train : Options → ModelZoo.NpyModelTrainFlags →
      IO Trainer.TrainSummary) :
    IO UInt32 :=
  runParsed exeName args parseFlags banner train (fun report => report.printSummary)

/-- NPY-backed regression command using the public trainer API. -/
def regressionNpy {σ τ : Shape}
    (exeName : String)
    (args : List String)
    (parseFlags : List String → Except String (ModelZoo.NpyModelTrainFlags × List String))
    (banner : Options → String)
    (train : Options → ModelZoo.NpyModelTrainFlags →
      IO (Trainer.TrainResult σ τ)) :
    IO UInt32 :=
  runParsed exeName args parseFlags banner train (fun result => result.printSummary)

/-- Forecast-window regression command using the public trainer API. -/
def forecastWindow {σ τ : Shape}
    (exeName : String)
    (args : List String)
    (parseFlags :
      List String → Except String (ModelZoo.ForecastWindowModelTrainFlags × List String))
    (banner : Options → String)
    (train : Options → ModelZoo.ForecastWindowModelTrainFlags →
      IO (Trainer.TrainResult σ τ)) :
    IO UInt32 :=
  runParsed exeName args parseFlags banner train (fun result => result.printSummary)

/--
Shared runner for the normal `lake exe torchlean ...` training commands.

This is the common path for examples that load data, train once, and print the standard report. A
model file should only define its own option record when it genuinely does more than training, for
example text generation, probe evaluation, or a custom curriculum.
-/
structure Config (δ : Type) where
  /-- CLI subcommand name, for example `torchlean rnn`. -/
  exeName : String
  /-- Default JSON log path used when `--log` is omitted. -/
  defaultLogJson : System.FilePath
  /-- Default number of optimizer steps when `--steps` is omitted. -/
  defaultSteps : Nat
  /-- Model description used in banners. -/
  description : String
  /-- Command-specific data flags, rendered by `--help`. -/
  dataOptions : List String := []
  /-- Parse data flags, then leave device/training flags for the shared parser. -/
  parseData : List String → Except String (δ × List String)
  /-- Run the actual training body after data, device, and training flags have been parsed. -/
  train : Options → δ → CLI.Training.RunOptions → IO Unit

/-- Usage text for model examples using the shared runner. -/
def usage {δ : Type} (cfg : Config δ) : String :=
  let dataSection :=
    if cfg.dataOptions.isEmpty then []
    else ["", "Data:"] ++ cfg.dataOptions
  String.intercalate "\n" <|
    [ s!"{cfg.exeName}: {cfg.description}"
    , ""
    , "Usage:"
    , s!"  lake exe torchlean {cfg.exeName.drop 10} [options]"
    ] ++ dataSection ++
    [ ""
    , "Training:"
    , s!"  --steps N          optimizer updates (default: {cfg.defaultSteps})"
    , "  --lr X             learning rate"
    , "  --log PATH|false   write a TrainLog JSON, or disable logging"
    , ""
    , "Runtime:"
    , "  --device auto|cpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external"
    , "  --execution eager|typed-graph"
    , "  --scalar float32|ieee32-exec|complex64"
    , "  --show-backend     print backend capsules as they execute"
    ]

/-- Run a public model-training command. -/
def run {δ : Type} (cfg : Config δ) (args : List String) : IO UInt32 := do
  if args.contains "--help" || args.contains "-h" then
    IO.println (usage cfg)
    return 0
  Module.Command.runFloat32 cfg.exeName args
    (banner := ModelZoo.bannerWithDevice cfg.exeName cfg.description)
    (k := fun opts rest => do
      let (dataArgs, rest) ← ModelZoo.orThrow cfg.exeName <| cfg.parseData rest
      let (train, rest) ← ModelZoo.orThrow cfg.exeName <|
        CLI.Training.RunOptions.parse cfg.exeName rest cfg.defaultLogJson cfg.defaultSteps
      CLI.requireNoArgs cfg.exeName rest
      cfg.train opts dataArgs train)

end NN.Examples.Models.TrainCommand
