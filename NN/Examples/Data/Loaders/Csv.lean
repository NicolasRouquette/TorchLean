/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.Examples.Data.SamplePaths

/-!
# CSV loader tutorial

This tutorial mirrors the "data first" workflow people expect from PyTorch:

1. Load a dataset from disk (CSV).
2. Turn it into a fixed-size batched dataset.
3. Train through the public `Trainer` API.

Generate a small deterministic regression dataset with
`python3 NN/Examples/Data/generate_small_data.py`:

- `NN/Examples/Data/small_regression.csv` with rows `x1,x2,y` (25 samples).

Build:

- `lake build NN.Examples.Data.Loaders.Csv`

The tutorial code is compiled with the rest of TorchLean and is directly runnable as
`lake exe torchlean data_csv`. It keeps data loading and training together so the parsed CSV shape
can be checked before the first optimizer step.

Optional flags (tutorial-specific):

- `--data-dir PATH` (default: `NN/Examples/Data`)
- `--csv PATH` (override the CSV file)
- `--seed S` (controls shuffling and model initialization)
- `--batch N`
- `--steps N`

Public API used here:

- `Data.tabularCsvDataset`
- `Trainer.new`
- `Trainer.RunConfig`
- `Trainer.TrainOptions`
- `trainer.train`
-/

@[expose] public section


namespace NN.Examples.Data.Loaders.Csv

open TorchLean

def missingCsvHint : String :=
  "Generate the small regression CSV with:\n" ++
  "  python3 NN/Examples/Data/generate_small_data.py"

def inDim : Nat := 2
def outDim : Nat := 1

/-- A small 2-layer batched MLP `2 -> 8 -> 1`. -/
def mkModel {batch : Nat} : nn.Builder (nn.Sequential [batch, inDim] [batch, outDim]) :=
  nn.blocks.mlp inDim outDim { hidden := [8] } [batch]

/-- Command-line help for the CSV loader tutorial. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean CSV loader tutorial"
    , ""
    , "Usage:"
    , "  lake exe torchlean data_csv [options]"
    , ""
    , "Options:"
    , "  --data-dir PATH"
    , "  --csv PATH"
    , "  --seed N"
    , "  --batch N"
    , "  --steps N"
    , "  --scalar float32|ieee32-exec|complex64"
    , "  --execution eager|typed-graph"
    , "  --device auto|cpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external"
    , "  --show-backend                    print backend capsules as they execute"
    ]

def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return

  let label := "Data.Loaders.Csv"
  let (dataDir, args) ← CLI.orThrow label <| _root_.NN.Examples.Data.SamplePaths.takeDataDir args
  let (seed, args) ← CLI.orThrow label <| CLI.takeSeed args 0
  let (steps, args) ← CLI.orThrow label <| CLI.takeStepsFlagDefault args 30
  let (batch, args) ← CLI.orThrow label <| CLI.takePositiveNatFlag args label "batch" 5
  let (csvPath, args) ← CLI.orThrow label <|
    CLI.takePathFlagDefault args "csv" (_root_.NN.Examples.Data.SamplePaths.regressionCsv dataDir)

  let model := mkModel (batch := batch)
  let run ← Trainer.RunConfig.parseRuntimeArgsOrThrow label args
    { optimizer := optim.adam { lr := 0.05 } }
  let trainer := Trainer.new model <|
    Trainer.Config.fromRunConfig run .regression (seed := seed)

  IO.println "== CSV loader training tutorial =="
  trainer.printInfo
  IO.println s!"data_dir = {dataDir}"
  IO.println s!"csv_path  = {csvPath}"
  IO.println s!"seed      = {seed}"
  IO.println (s!"train     = Adam(lr=0.05), steps={steps}, batch_size={batch}, " ++
    s!"shuffle=true, drop_last=true")

  let csvOptions : Data.CsvOptions := { skipHeader := true }
  let data :=
    Data.tabularCsvDataset csvPath batch inDim outDim
      (csvOptions := csvOptions) (shuffle := true) (seed := seed)
  Data.requireFile label "CSV dataset" csvPath missingCsvHint
  let trained ← trainer.train data { steps := steps }
  trained.printSummary
  let heldout : Tensor Float [batch, inDim] :=
    Tensor.full [batch, inDim] 0.25
  trained.printPrediction "predict(batch=heldout)" heldout

end NN.Examples.Data.Loaders.Csv
