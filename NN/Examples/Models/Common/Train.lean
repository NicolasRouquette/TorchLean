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

/-- Help text for model commands with caller-supplied data and training options. -/
def modelUsage
    (exeName : String)
    (dataOptions trainingOptions : Array String)
    (extraSections : Array String := #[]) : String :=
  String.intercalate "\n" <| (#[
    s!"Usage: lake exe torchlean {exeName.drop 10} [options]",
    "",
    "Data:"
  ] ++ dataOptions ++ #[
    "",
    "Training:"
  ] ++ trainingOptions ++ extraSections ++ #[
    "",
    "Runtime:",
    "  --device auto|cpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external",
    "  --execution eager|typed-graph",
    "  --scalar float32",
    "  --seed N --show-backend"
  ]).toList

/-- Standard optimizer and logging flags accepted by most model commands. -/
def optimizerUsage (exeName : String) (dataOptions : Array String) : String :=
  modelUsage exeName dataOptions #[
    "  --steps N          optimizer updates",
    "  --batch-size N     dataset items accumulated per update",
    "  --lr X             learning rate",
    "  --log PATH|false   write a TrainLog JSON, or disable logging",
    "  --cuda-mem-watch N sample CUDA allocator state every N updates"
  ]

/-- Fixed-optimizer training flags accepted by custom-curve commands such as the GAN example. -/
def runOptionsUsage (exeName : String) (dataOptions : Array String) : String :=
  modelUsage exeName dataOptions #[
    "  --steps N          optimizer updates",
    "  --batch-size N     dataset items accumulated per update",
    "  --log PATH|false   write a TrainLog JSON, or disable logging",
    "  --cuda-mem-watch N sample CUDA allocator state every N updates"
  ]

/-- Run one parsed model-training command and finish with access to runtime flags and result. -/
def runParsedWith {φ ρ : Type}
    (exeName : String)
    (args : List String)
    (parseFlags : List String → Except String (φ × List String))
    (banner : Options → String)
    (train : Options → φ → IO ρ)
    (finish : Options → φ → ρ → IO Unit)
    (usage? : Option String := none) :
    IO UInt32 :=
  Module.Command.runFloat32 exeName args
    (banner := banner)
    (usage? := usage?)
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
    (print : ρ → IO Unit)
    (usage? : Option String := none) :
    IO UInt32 :=
  runParsedWith exeName args parseFlags banner train (fun _ _ trained => print trained) usage?

/-- CSV-backed regression command using the public trainer API. -/
def regressionCsv {inputShape targetShape : List Nat}
    (exeName : String)
    (args : List String)
    (defaultCsv : System.FilePath)
    (defaultLogPath : System.FilePath)
    (defaultSteps : Nat := 1)
    (defaultLr : Float := 1e-3)
    (banner : Options → String)
    (train : Options → ModelZoo.CsvTrainFlags →
      IO (Trainer.TrainResult inputShape targetShape)) :
    IO UInt32 :=
  runParsed exeName args
    (fun rest =>
      ModelZoo.parseCsvTrainFlags exeName rest defaultCsv defaultLogPath defaultSteps defaultLr)
    banner train (fun result => result.printSummary)
    (usage? := some <| optimizerUsage exeName #["  --csv PATH         supervised CSV file"])

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
    (usage? := some <| optimizerUsage exeName #[
      "  --x PATH           feature/image NPY file",
      "  --y PATH           class-label NPY file",
      "  --n-total N        rows to load"
    ])

/-- NPY-backed regression command using the public trainer API. -/
def regressionNpy {inputShape targetShape : List Nat}
    (exeName : String)
    (args : List String)
    (parseFlags : List String → Except String (ModelZoo.NpyModelTrainFlags × List String))
    (banner : Options → String)
    (train : Options → ModelZoo.NpyModelTrainFlags →
      IO (Trainer.TrainResult inputShape targetShape)) :
    IO UInt32 :=
  runParsed exeName args parseFlags banner train (fun result => result.printSummary)
    (usage? := some <| optimizerUsage exeName #[
      "  --x PATH           feature/image NPY file",
      "  --y PATH           target NPY file",
      "  --n-total N        rows to load"
    ])

/-- Forecast-window regression command using the public trainer API. -/
def forecastWindow {inputShape targetShape : List Nat}
    (exeName : String)
    (args : List String)
    (parseFlags :
      List String → Except String (ModelZoo.ForecastWindowModelTrainFlags × List String))
    (banner : Options → String)
    (train : Options → ModelZoo.ForecastWindowModelTrainFlags →
      IO (Trainer.TrainResult inputShape targetShape)) :
    IO UInt32 :=
  runParsed exeName args parseFlags banner train (fun result => result.printSummary)
    (usage? := some <| optimizerUsage exeName #[
      "  --x PATH           input-window NPY file",
      "  --y PATH           target-window NPY file",
      "  --windows N        windows to load",
      "  --report-offset N  window shown before and after training"
    ])

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
  /-- Learning rate used when `--lr` is omitted. -/
  defaultLr : Float
  /-- Model description used in banners. -/
  description : String
  /-- Command-specific data flags, rendered by `--help`. -/
  dataOptions : Array String := #[]
  /-- Parse data flags, then leave device/training flags for the shared parser. -/
  parseData : List String → Except String (δ × List String)
  /-- Run the actual training body after data, device, and training flags have been parsed. -/
  train : Options → δ → CLI.Training.OptimizerOptions → IO Unit

/-- Usage text for model examples using the shared runner. -/
def usage {δ : Type} (cfg : Config δ) : String :=
  let dataSection :=
    if cfg.dataOptions.isEmpty then #[]
    else #["", "Data:"] ++ cfg.dataOptions
  String.intercalate "\n" <|
    (#[ s!"{cfg.exeName}: {cfg.description}"
    , ""
    , "Usage:"
    , s!"  lake exe torchlean {cfg.exeName.drop 10} [options]"
    ] ++ dataSection ++
    #[ ""
    , "Training:"
    , s!"  --steps N          optimizer updates (default: {cfg.defaultSteps})"
    , s!"  --lr X             learning rate (default: {cfg.defaultLr})"
    , "  --batch-size N     dataset items accumulated per optimizer update (default: 1)"
    , "  --log PATH|false   write a TrainLog JSON, or disable logging"
    , ""
    , "Runtime:"
    , "  --device auto|cpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external"
    , "  --execution eager|typed-graph"
    , "  --scalar float32"
    , "  --show-backend     print backend capsules as they execute"
    ]).toList

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
        CLI.Training.OptimizerOptions.parse cfg.exeName rest cfg.defaultLogJson cfg.defaultSteps
          cfg.defaultLr
      CLI.requireNoArgs cfg.exeName rest
      cfg.train opts dataArgs train)

end NN.Examples.Models.TrainCommand
