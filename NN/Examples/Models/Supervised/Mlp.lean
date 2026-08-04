/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Device-agnostic real-data example:
  python3 scripts/datasets/download_example_data.py --auto-mpg
  lake exe torchlean mlp --device cpu
  lake -R -K cuda=true exe torchlean mlp --device cuda
-/

module


public import NN.API
public import NN.Examples.Models.Common

/-!
# MLP Tabular Regression

This example trains an MLP on the UCI Auto MPG regression task. The prepared CSV has seven
normalized numeric car features and one normalized target column for miles per gallon, so the model
is just ordinary supervised tabular regression:

`x1..x7 -> Linear -> ReLU -> Linear -> y`.

Prepare the CSV once:

```bash
python3 scripts/datasets/download_example_data.py --auto-mpg
lake exe torchlean mlp --device cpu --steps 1
```

The downloader writes normalized columns `x1..x7,y`. If you want to try your own tabular regression
CSV, pass `--csv PATH` with the same columns.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Supervised.Mlp

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "torchlean mlp"

/-- Default JSON loss-curve path for this command. -/
def defaultLogJson : System.FilePath := ModelZoo.trainLogPath "mlp"

/-- Static minibatch size for the Auto MPG tabular loader. -/
def batch : Nat := 5

/-- Auto MPG has seven numeric predictors after dropping `car_name`. -/
def inDim : Nat := 7

/-- Hidden width of the one-hidden-layer MLP. -/
def hidDim : Nat := 32

/-- Regression target width: normalized miles-per-gallon. -/
def outDim : Nat := 1

/-- Shared MLP configuration used by shapes and the constructor. -/
def cfg : nn.models.MlpConfig :=
  { batch := batch, inDim := inDim, hidDim := hidDim, outDim := outDim }

/-- Input shape: a minibatch of Auto MPG feature vectors. -/
abbrev σ : Shape := .dim batch (.dim inDim .scalar)

/-- Output shape: one scalar regression prediction per row. -/
abbrev τ : Shape := .dim batch (.dim outDim .scalar)

/-- One-hidden-layer ReLU MLP from the public model API. -/
def model : nn.M (nn.Sequential σ τ) :=
  nn.models.mlpRelu cfg

/--
Auto MPG as a public TorchLean dataset.

The only dataset-specific details here are the CSV path, header convention, batch size, and feature
count. Runtime scalar selection stays inside `Trainer`, so the same dataset works for CPU, CUDA,
compiled, eager, and checked scalar modes.
-/
def data (path : System.FilePath) (seed : Nat) :
    Trainer.Dataset σ τ :=
  Data.tabularCsvDataset path batch inDim outDim
    (csvOptions := { skipHeader := true }) (shuffle := true) (seed := seed)

/-- Train the Auto MPG MLP with the public `Trainer` surface. -/
def train (opts : Options) (flags : ModelZoo.CsvTrainFlags) :
    IO (Trainer.TrainResult σ τ) := do
  Data.requireFile exeName "CSV dataset" flags.csvPath RealData.missingAutoMpgHint
  let trainer :=
    Trainer.new model <|
      Trainer.Config.fromRunConfig
        (Trainer.runConfig opts { optimizer := optim.adam { lr := flags.lr } })
        .regression
        (seed := flags.seed)
  trainer.train
    (data flags.csvPath flags.seed)
    (CLI.Training.OptimizerOptions.toTrainerOptions flags.toOptimizerOptions
      (title := "MLP tabular training")
      (notes := #[s!"dataset={flags.csvPath}", s!"lr={flags.lr}",
        s!"steps={flags.steps}", s!"batch={batch}"]))

/-- CLI entrypoint for Auto MPG regression on CPU or CUDA. -/
def main (args : List String) : IO UInt32 :=
  TrainCommand.regressionCsv exeName args
    _root_.NN.Examples.Data.RealPaths.autoMpgCsv defaultLogJson 1 1e-3
    (ModelZoo.bannerWithDevice exeName "Auto MPG MLP regression")
    train

end NN.Examples.Models.Supervised.Mlp
