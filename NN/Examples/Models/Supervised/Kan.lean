/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Device-agnostic real-data example:
  python3 scripts/datasets/download_example_data.py --auto-mpg
  lake exe torchlean kan --device cpu --steps 20
  lake -R -K cuda=true exe torchlean kan --device cuda --steps 20
-/

module

public import NN.API
public import NN.Examples.Models.Common

/-!
# KAN Regression

This example trains a small Kolmogorov-Arnold Network on the prepared Auto MPG tabular-regression
CSV. The downloader normalizes the columns to $[0,1]$, so the piecewise-linear KAN basis uses
$\mathtt{inputScale}=\mathtt{gridSize}-1$ to spread its knots across the data interval.

`KAN` is a model constructor. The task is chosen by the general trainer API:

```lean
let trainer := Trainer.new model { task := .regression, optimizer := optim.adam { lr := 1e-3 } }
```

The edge basis is a normal config field. This example uses triangular piecewise-linear hats; a
spline, polynomial, or rational edge family would plug in through the same
`nn.models.KanEdgeFamily` slot, while the trainer continues to choose the task.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Supervised.Kan

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "torchlean kan"

/-- Default JSON loss-curve path for this command. -/
def defaultLogJson : System.FilePath := ModelZoo.trainLogPath "kan"

/-- Static minibatch size for the Auto MPG tabular loader. -/
def batch : Nat := 5

/-- Auto MPG has seven normalized numeric predictors after dropping the car name. -/
def inDim : Nat := 7

/-- Scalar regression target. -/
def outDim : Nat := 1

/-- KAN configuration using triangular edge bases over normalized tabular features. -/
def cfg : nn.models.KanConfig :=
  { inDim := inDim
    hidden := []
    outDim := outDim
    edge := nn.models.KanPiecewiseLinear.edgeFamily { gridSize := 4, inputScale := 3 } }

abbrev σ := cfg.inputShape (.dim batch .scalar)
abbrev τ := cfg.outputShape (.dim batch .scalar)

/-- Generic KAN model. Regression/classification is selected by `Trainer`, not by the model name. -/
def model : nn.Builder (nn.Sequential σ τ) :=
  nn.models.kan cfg (.dim batch .scalar)

/-- Prepared Auto MPG CSV as a public trainer dataset. -/
def data (path : System.FilePath) (seed : Nat) : Trainer.DataSource σ τ :=
  Data.tabularCsvDataset path batch inDim outDim
    (csvOptions := { skipHeader := true }) (shuffle := true) (seed := seed)

/-- Train the Auto MPG KAN with the public `Trainer` surface. -/
def train (opts : Options) (flags : ModelZoo.CsvTrainFlags) :
    IO (Trainer.TrainResult σ τ) := do
  Data.requireFile exeName "CSV dataset" flags.csvPath RealData.missingAutoMpgHint
  let trainer :=
    Trainer.new model <|
      Trainer.Config.fromRunConfig
        (Trainer.RunConfig.ofRuntimeOptions opts { optimizer := optim.adam { lr := flags.lr } })
        .regression
        (seed := flags.seed)
  trainer.train
    (data flags.csvPath flags.seed)
    (CLI.Training.OptimizerOptions.toTrainerOptions flags.toOptimizerOptions
      (title := "KAN Auto MPG regression")
      (notes := #[ModelZoo.deviceNote opts, s!"data={flags.csvPath}", s!"lr={flags.lr}",
        s!"steps={flags.steps}", s!"batch={batch}", s!"edge={cfg.edge.name}"]))

/-- CLI entrypoint for Auto MPG regression with a KAN model. -/
def main (args : List String) : IO UInt32 :=
  TrainCommand.regressionCsv exeName args
    _root_.NN.Examples.Data.RealPaths.autoMpgCsv defaultLogJson 20 1e-2
    (ModelZoo.bannerWithDevice exeName "Auto MPG KAN regression")
    train

end NN.Examples.Models.Supervised.Kan
