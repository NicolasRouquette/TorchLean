/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.Examples.Data.SamplePaths

/-!
# NPY loader tutorial (NumPy/PyTorch interop)

This tutorial shows how to train from `.npy` files (NumPy arrays), similar to a common PyTorch
workflow where you:

1. prepare `X.npy` / `y.npy` in Python (NumPy / PyTorch),
2. then train a model in TorchLean by loading those files.

Generate small deterministic `.npy` files with
`python3 NN/Examples/Data/generate_small_data.py`:

- `NN/Examples/Data/small_regression_X.npy`  (shape 25×2, dtype float32)
- `NN/Examples/Data/small_regression_y.npy`  (shape 25×1, dtype float32)

Build:

- `lake build NN.Examples.Data.Loaders.Npy`

The tutorial code is compiled with the rest of TorchLean. For command-line model training, use the
`torchlean` executable examples in `NN/Examples/Models`.

Optional flags (tutorial-specific):

- `--data-dir PATH` (default: `NN/Examples/Data`)
- `--x PATH`, `--y PATH` (override the `.npy` files)
- `--seed S` (controls shuffling and model initialization)
- `--batch N`
- `--steps N`

Public API used here:

- `Data.readNpy` (metadata)
- `Data.supervisedDataset`
- `Data.batchDataset`
- `Trainer.new`
- `Trainer.RunConfig`
- `Trainer.TrainOptions`
- `trainer.train`
-/

@[expose] public section


namespace NN.Examples.Data.Loaders.Npy

open TorchLean

def nSamples : Nat := 25
def inDim : Nat := 2
def outDim : Nat := 1

/-- A small 2-layer batched MLP `2 -> 8 -> 1`. -/
def mkModel {batch : Nat} : nn.Builder (nn.Sequential (.dim batch (.dim inDim .scalar)) (.dim batch (.dim outDim .scalar))) :=
  nn.blocks.mlp inDim outDim { hidden := [8] } (.dim batch .scalar)

/-- Command-line help for the NPY loader tutorial. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean NPY loader tutorial"
    , ""
    , "Usage:"
    , "  lake exe torchlean data_npy [options]"
    , ""
    , "Options:"
    , "  --data-dir PATH"
    , "  --x PATH"
    , "  --y PATH"
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

  let label := "Data.Loaders.Npy"
  let (dataDir, args) ← CLI.orThrow label <| _root_.NN.Examples.Data.SamplePaths.takeDataDir args
  let (seed, args) ← CLI.orThrow label <| CLI.takeSeed args 0
  let (steps, args) ← CLI.orThrow label <| CLI.takeStepsFlagDefault args 20
  let (batch, args) ← CLI.orThrow label <| CLI.takePositiveNatFlag args label "batch" 5
  let (paths, args) ← CLI.orThrow label <|
    _root_.NN.Examples.Data.SamplePaths.takeXyPaths args
      (_root_.NN.Examples.Data.SamplePaths.regressionXNpy dataDir)
      (_root_.NN.Examples.Data.SamplePaths.regressionYNpy dataDir)
  let xPath := paths.xPath
  let yPath := paths.yPath

  let model := mkModel (batch := batch)
  let run ← Trainer.RunConfig.parseRuntimeArgsOrThrow label args
    { optimizer := optim.adam { lr := 0.05 } }
  let trainer := Trainer.new model <|
    Trainer.Config.fromRunConfig run .regression (seed := seed)

  IO.println "== NPY loader training tutorial =="
  trainer.printInfo
  IO.println s!"data_dir = {dataDir}"
  IO.println s!"x_path   = {xPath}"
  IO.println s!"y_path   = {yPath}"
  IO.println s!"seed     = {seed}"
  IO.println (s!"train    = Adam(lr=0.05), steps={steps}, batch_size={batch}, " ++
    s!"shuffle=true, drop_last=true")
  let xMeta ← CLI.orThrow label <| (← Data.readNpy xPath)
  let yMeta ← CLI.orThrow label <| (← Data.readNpy yPath)
  IO.println s!"X.npy dtype={xMeta.dtype} shape={xMeta.shape}"
  IO.println s!"y.npy dtype={yMeta.dtype} shape={yMeta.shape}"

  let src : Data.SupervisedSource :=
    Data.SupervisedSource.ofPaths .npy xPath yPath nSamples [inDim] [outDim]
  let data0 := Data.supervisedDataset src
  let data := Data.batchDataset batch data0 (shuffle := true) (seed := seed)
  let trained ← trainer.train data { steps := steps }
  trained.printSummary
  let heldout : Tensor Float (.dim batch (.dim inDim .scalar)) :=
    Tensor.fill 0.25 (.dim batch (.dim inDim .scalar))
  trained.printPrediction "predict(batch=heldout)" heldout

end NN.Examples.Data.Loaders.Npy
