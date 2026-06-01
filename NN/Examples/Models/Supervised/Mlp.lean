/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Device-agnostic real-data example:
  python3 scripts/datasets/download_example_data.py --auto-mpg
  lake exe torchlean mlp --cpu
  lake build -R -K cuda=true && lake exe torchlean mlp --cuda
-/

module


public import NN
public import NN.API.Models.Mlp
public import NN.Examples.Data.RealPaths

/-!
# MLP Tabular Regression

This example trains an MLP on the UCI Auto MPG regression task. The prepared CSV has seven
normalized numeric car features and one normalized target column for miles per gallon, so the model
is just ordinary supervised tabular regression:

`x1..x7 -> Linear -> ReLU -> Linear -> y`.

Prepare the CSV once:

```bash
python3 scripts/datasets/download_example_data.py --auto-mpg
lake exe torchlean mlp --cpu --steps 1
```

The downloader writes normalized columns `x1..x7,y`. If you want to try your own tabular regression
CSV, pass `--csv PATH` with the same columns.
-/

@[expose] public section

open Spec Tensor
open NN.API

namespace NN.Examples.Models.Supervised.Mlp

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "torchlean mlp"

/-- Default JSON loss-curve path for this command. -/
def defaultLogJson : System.FilePath := Common.modelZooTrainLog "mlp"

/-- Static minibatch size for the Auto MPG tabular loader. -/
def batch : Nat := 5

/-- Auto MPG has seven numeric predictors after dropping `car_name`. -/
def inDim : Nat := 7

/-- Hidden width of the one-hidden-layer MLP. -/
def hidDim : Nat := 32

/-- Regression target width: normalized miles-per-gallon. -/
def outDim : Nat := 1

/-- Shared MLP configuration used by shapes and the constructor. -/
def cfg : nn.models.Mlp1Config :=
  { batch := batch, inDim := inDim, hidDim := hidDim, outDim := outDim }

/-- Input shape: a minibatch of Auto MPG feature vectors. -/
abbrev σ : Shape :=
  nn.models.mlp1InShape cfg

/-- Output shape: one scalar regression prediction per row. -/
abbrev τ : Shape :=
  nn.models.mlp1OutShape cfg

/-- One-hidden-layer ReLU MLP from the public model API. -/
def mkModel : nn.M (nn.Sequential σ τ) :=
  nn.models.mlp1Relu cfg

/--
Load the prepared Auto MPG CSV as a typed minibatch loader.

The dataset-specific facts stay here: the default preparation script writes columns `x1..x7,y`,
the first seven columns are inputs, the last column is the regression target, and we use full
minibatches of size `batch`.  The generic CSV-to-loader mechanics live in `Data.tabularCsvLoader`.
-/
def loadAutoMpg {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    (path : System.FilePath) (seed : Nat) : IO (Data.BatchLoader α batch (Shape.Vec inDim)
      (Shape.Vec outDim)) := do
  unless (← path.pathExists) do
    throw <| IO.userError
      s!"{exeName}: missing CSV dataset: {path}\nRun: python3 scripts/datasets/download_example_data.py --auto-mpg"
  let loaderE ← Data.tabularCsvLoader (α := α) path batch inDim outDim
    (csvOptions := { skipHeader := true }) (shuffle := true) (seed := seed) (dropLast := true)
  Common.orThrow exeName loaderE

/--
Instantiate the MLP, attach Adam, train for the requested number of updates, and write the standard
training log if logging is enabled.

CPU and CUDA both use the same Float runtime module; `opts` selects the device and backend.
-/
def fitAutoMpg (opts : Runtime.Autograd.Torch.Options)
    (flags : Common.CsvModelTrainFlags) : IO (train.FitReport Float) := do
  let loader ← loadAutoMpg (α := Float) flags.csvPath flags.seed
  nn.withModel mkModel fun model => do
    let modDef := nn.mseScalarModuleDef model
    let module ← TorchLean.Module.instantiateWithOptions (α := Float) modDef id opts
    let opt := Common.adamOptimizer (α := Float) id (nn.paramShapes model) flags.train.lr
    let cudaMemWatch :=
      Common.effectiveCudaMemWatch opts flags.train.train.steps flags.train.cudaMemWatch
    let (report, _loader') ← train.fitModuleLoaderStepsLoggedFloat module opt opts
      flags.train.train.steps loader flags.train.train.log "MLP tabular training"
      #[s!"dataset={flags.csvPath}", s!"device={if opts.useGpu then "cuda" else "cpu"}",
        s!"lr={flags.train.lr}", s!"steps={flags.train.train.steps}", s!"batch={batch}",
        s!"cuda_mem_watch={cudaMemWatch}"]
      "loss" flags.train.cudaMemWatch
    pure report

/-- CLI entrypoint for Auto MPG regression on CPU or CUDA. -/
def main (args : List String) : IO UInt32 := do
  Common.runFloat exeName args
    (banner := fun opts =>
      s!"{exeName}: Auto MPG MLP regression (device={if opts.useGpu then "cuda" else "cpu"})")
    (k := fun opts rest => do
      let (flags, rest) ← Common.orThrow exeName <|
        Common.parseCsvModelTrainFlags exeName rest
          _root_.NN.Examples.Data.RealPaths.autoMpgCsv defaultLogJson 1 1e-3
      Common.orThrow exeName <| CLI.requireNoArgs rest
      let report ← fitAutoMpg opts flags
      Common.printFitReport flags.train.train.steps report)

end NN.Examples.Models.Supervised.Mlp
