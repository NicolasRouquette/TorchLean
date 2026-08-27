/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.Examples.Quickstart.Common
public import NN.Examples.Data.SamplePaths

/-!
# Minibatch MLP training (batching is implicit)

This next-step file shows the intended minibatch path in TorchLean:

1. load a CSV dataset through `Data`,
2. choose persistent runtime settings with `Trainer.RunConfig`,
3. write the model over the minibatch shape `[batch, inDim]`,
4. call `trainer.train data trainOptions`,
5. reuse the training result for one follow-up prediction batch.

Run the loader tutorial instead when possible:

- `lake exe torchlean data_csv --steps 1 --batch 5 --scalar float32 --execution eager`

Build this comparison module directly with:

- `lake build NN.Examples.Quickstart.MinibatchMlpTrain`

Optional flags (tutorial-specific):

- `--data-dir PATH` (default: `NN/Examples/Data`)
- `--csv PATH` (override the CSV file)
- `--seed S`
- `--steps N`
- `--batch N`
-/

@[expose] public section


namespace NN.Examples.Quickstart.MinibatchMLPTrain

open TorchLean

/-- Default JSON log path used only when the user explicitly passes `--log`. -/
def defaultLogJson : System.FilePath := ModelZoo.trainLogPath "quickstart_minibatch_mlp"

def missingCsvHint : String :=
  "Generate the small regression CSV with:\n" ++
  "  python3 NN/Examples/Data/generate_small_data.py"

def inDim : Nat := 2
def hidDim : Nat := 8
def outDim : Nat := 1

/-- Batched MLP `2 -> 8 -> 1` built from the public model constructor. -/
def mkModel {batch : Nat} : nn.Builder (nn.Sequential [batch, inDim] [batch, outDim]) :=
  nn.blocks.mlp inDim outDim { hidden := [hidDim] } [batch]

/-- Command-line help for the minibatch MLP quickstart. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean minibatch MLP quickstart"
    , ""
    , "Usage:"
    , "  lake exe torchlean quickstart_minibatch_mlp [options]"
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
    , "  --log PATH"
    ]

def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return
  let (dataDir, args) ← CLI.orThrow "MinibatchMLPTrain" <|
    _root_.NN.Examples.Data.SamplePaths.takeDataDir args
  let (seed, args) ← CLI.seed "MinibatchMLPTrain" args
  let (batch, args) ← CLI.positiveNatFlag "MinibatchMLPTrain" args "batch" 5
  let (csvPath, args) ← CLI.orThrow "MinibatchMLPTrain" <|
    CLI.takePathFlagDefault args "csv" (_root_.NN.Examples.Data.SamplePaths.regressionCsv dataDir)
  let parsed ←
    _root_.NN.Examples.Quickstart.parseRuntimeTrain
      "MinibatchMLPTrain" args defaultLogJson 30 (optim.adam { lr := 0.05 })
  let trainer := Trainer.new (mkModel (batch := batch)) <|
    Trainer.Config.fromRunConfig parsed.run .regression (seed := seed)

  IO.println "== Quickstart next step: minibatch MLP training =="
  IO.println s!"data_dir = {dataDir}"
  IO.println s!"csv_path  = {csvPath}"
  IO.println s!"seed      = {seed}"
  trainer.printInfo
  IO.println (s!"train     = Adam(lr=0.05), steps={parsed.train.steps}, batch_size={batch}, " ++
    s!"shuffle=true, drop_last=true")

  let csvOptions : Data.CsvOptions := { skipHeader := true }
  let data :=
    Data.tabularCsvDataset csvPath batch inDim outDim
      (csvOptions := csvOptions) (shuffle := true) (seed := seed)
  Data.requireFile "MinibatchMLPTrain" "CSV dataset" csvPath missingCsvHint
  let trained ← trainer.train data parsed.trainOptions
  trained.printSummary
  let heldout : Tensor Float [batch, inDim] :=
    Tensor.full [batch, inDim] 0.25
  trained.printPrediction "predict(batch=heldout)" heldout

end NN.Examples.Quickstart.MinibatchMLPTrain
