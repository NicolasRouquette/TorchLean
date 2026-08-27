/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.Examples.Data.RealPaths
public import NN.Examples.Data.SamplePaths

/-!
# CIFAR10-style image loader tutorial (NPY, offline)

This tutorial mirrors a classic PyTorch recipe:

1. Load a labeled image dataset from disk (`.npy` exported from NumPy/PyTorch).
2. Split into train/test.
3. Build a small CNN by explicitly stacking layers.
4. Train for multiple epochs over shuffled minibatches through the public `Trainer` API.

To keep this runnable without network downloads, generate a small deterministic
"CIFAR10-shaped" dataset locally:

- `NN/Examples/Data/small_cifar10like_X.npy`: shape `(200, 3, 32, 32)`, dtype `float32`
- `NN/Examples/Data/small_cifar10like_y.npy`: shape `(200,)`, dtype `float32` labels `0..9`

Generate it with:

`python3 NN/Examples/Data/generate_small_data.py`

Build:

- `lake build NN.Examples.Data.Loaders.Cifar10Images`

For command-line CIFAR training, use `torchlean cnn` or `torchlean vit` with
`--x`, `--y`, and `--n-total`.

Optional flags (tutorial-specific):

- `--data-dir PATH` (default: `NN/Examples/Data`)
- `--real-cifar10` (use `data/real/cifar10/cifar10_train_*.npy`, as prepared by
  `scripts/datasets/download_example_data.py --cifar10`)
- `--x PATH`, `--y PATH` (override the `.npy` files)
- `--n-total N` (number of rows in the selected `.npy` files; default `200`)
- `--seed S` (controls split + shuffling + model initialization)
- `--batch N`
- `--epochs E`
- `--lr LR` (default: `0.001`)
- `--train-size N` (default: 160)
- `--check-only` (validate paths, tensor shapes, and dataset splitting without training)

Why this tutorial matters:

- it shows the public `Data` file-loading path rather than only in-memory tensors;
- it keeps the model architecture familiar (Conv/ReLU/Pool stack + classifier head);
 - it shows the "offline artifact" workflow many PyTorch users already have, where arrays
  have been pre-exported to `.npy` and training happens without any dataset download step.
- it stays on the same public `Trainer` surface as the model examples instead of dropping to the
  callback runner API.
- it shows that the trained result is still usable for immediate inference, not only for a terminal
  loss summary.
-/

@[expose] public section


namespace NN.Examples.Data.Loaders.Cifar10Images

open TorchLean

abbrev classes : Nat := 10
abbrev channels : Nat := 3
abbrev height : Nat := 32
abbrev width : Nat := 32
abbrev nTotal : Nat := 200

/-- Small CNN (no BatchNorm): Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Linear(10). -/
def mkModel {batch : Nat} :
    nn.Builder (nn.Sequential [batch, channels, height, width] [batch, classes]) :=
  let outC1 : Nat := 16
  let outC2 : Nat := 32
  let spatial0 : Tensor Nat [2] := tensor! [height, width]
  let conv1 : nn.Conv 2 :=
    { outChannels := outC1
      kernel := tensor! [3, 3]
      padding := tensor! [1, 1]
      kernelNonzero := by intro i; fin_cases i <;> decide
      strideNonzero := by intro i; fin_cases i <;> decide }
  let conv2 : nn.Conv 2 :=
    { outChannels := outC2
      kernel := tensor! [3, 3]
      padding := tensor! [1, 1]
      kernelNonzero := by intro i; fin_cases i <;> decide
      strideNonzero := by intro i; fin_cases i <;> decide }
  let pool : nn.Pool 2 :=
    { kernel := tensor! [2, 2]
      stride := tensor! [2, 2]
      kernelNonzero := by intro i; fin_cases i <;> decide
      strideNonzero := by intro i; fin_cases i <;> decide }
  let spatial1 := Spec.convOutSpatial spatial0 conv1.kernel conv1.stride conv1.padding
  let spatial2 := Spec.poolOutSpatialPad spatial1 pool.kernel pool.stride pool.padding
  let spatial3 := Spec.convOutSpatial spatial2 conv2.kernel conv2.stride conv2.padding
  let spatial4 := Spec.poolOutSpatialPad spatial3 pool.kernel pool.stride pool.padding
  nn.Sequential![
    nn.conv (leading := [batch]) (inChannels := channels) spatial0 conv1,
    nn.relu,
    nn.maxPool (leading := [batch]) (channels := outC1) spatial1 pool,
    nn.conv (leading := [batch]) (inChannels := outC1) spatial2 conv2,
    nn.relu,
    nn.maxPool (leading := [batch]) (channels := outC2) spatial3 pool,
    pure (nn.heads.classifier (leading := [batch])
      (shape := outC2 :: spatial4.toList) classes)
  ]

/-- Shared offline CIFAR10-like tensor source used by this tutorial. -/
def source (xPath yPath : System.FilePath) (nRows : Nat) : Data.LabeledSource :=
  Data.LabeledSource.ofPaths .npy xPath yPath nRows [channels, height, width] classes

def trainDataset (xPath yPath : System.FilePath) (nRows trainSize seed : Nat) :
    Trainer.Dataset [channels, height, width] [classes] :=
  (Data.randomSplitDataset trainSize (Data.labeledDataset (source xPath yPath nRows)) seed).1

/-- Runtime-polymorphic test split used for `--check-only` reporting. -/
def testDataset (xPath yPath : System.FilePath) (nRows trainSize seed : Nat) :
    Trainer.Dataset [channels, height, width] [classes] :=
  (Data.randomSplitDataset trainSize (Data.labeledDataset (source xPath yPath nRows)) seed).2

/-- Command-line help for the CIFAR10-style NPY loader tutorial. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean CIFAR10-style NPY loader tutorial"
    , ""
    , "Usage:"
    , "  lake exe torchlean data_cifar10 [options]"
    , ""
    , "Options:"
    , "  --data-dir PATH"
    , "  --real-cifar10"
    , "  --x PATH"
    , "  --y PATH"
    , "  --n-total N"
    , "  --train-size N"
    , "  --seed N"
    , "  --epochs N"
    , "  --batch N"
    , "  --lr LR"
    , "  --check-only"
    , "  --scalar float32|ieee32-exec"
    , "  --execution eager|typed-graph"
    , "  --device auto|cpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external"
    , "  --show-backend                    print backend capsules as they execute"
    ]

def main (args : List String) : IO Unit := do
  let args0 := CLI.dropDashDash args
  if CLI.hasHelp args0 then
    IO.println usage
    return
  let checkOnly := args0.contains "--check-only"
  let realCifar10 := args0.contains "--real-cifar10"
  let args := args0.filter (fun a => a != "--check-only" && a != "--real-cifar10")

  let label := "Data.Loaders.Cifar10Images"
  let (dataDir, args) ← CLI.orThrow label <| _root_.NN.Examples.Data.SamplePaths.takeDataDir
    args
  let (seed, args) ← CLI.orThrow label <| CLI.takeSeed args 0
  let (eb, args) ← CLI.orThrow label <| CLI.takePositiveEpochBatch args label 5 20
  let (trainSize0, args) ← CLI.orThrow label <| CLI.takeNatFlagDefault args
    "train-size" 0
  let (nRows0, args) ← CLI.orThrow label <| CLI.takeNatFlagDefault args
    "n-total" nTotal
  let (lr, args) ← CLI.orThrow label <| CLI.takeFloatFlagDefault args "lr" 0.001
  let defaultX :=
    if realCifar10 then
      _root_.NN.Examples.Data.RealPaths.cifar10TrainX
    else
      _root_.NN.Examples.Data.SamplePaths.cifar10likeXNpy dataDir
  let defaultY :=
    if realCifar10 then
      _root_.NN.Examples.Data.RealPaths.cifar10TrainY
    else
      _root_.NN.Examples.Data.SamplePaths.cifar10likeYNpy dataDir
  let (paths, args) ← CLI.orThrow label <|
    _root_.NN.Examples.Data.SamplePaths.takeXyPaths args defaultX defaultY
  let xPath := paths.xPath
  let yPath := paths.yPath
  let nRows := nRows0
  let trainSize :=
    if trainSize0 = 0 then
      (if checkOnly then Nat.min 16 nRows else Nat.min 160 nRows)
    else
      trainSize0
  let trainSteps : Nat := eb.epochs * (trainSize / eb.batch)
  let run ← Trainer.RunConfig.parseRuntimeArgsOrThrow label args
    { optimizer := optim.adam { lr := lr } }
  let trainer := Trainer.new (mkModel (batch := eb.batch)) <|
    Trainer.Config.fromRunConfig run (.oneHotCrossEntropy 1) (seed := seed)

  IO.println "== CIFAR10-style NPY CNN tutorial =="
  IO.println s!"data_dir   = {dataDir}"
  IO.println s!"x_path     = {xPath}"
  IO.println s!"y_path     = {yPath}"
  IO.println s!"rows       = {nRows}"
  IO.println s!"seed       = {seed}"
  IO.println s!"train_size = {trainSize} / {nRows}"
  trainer.printInfo
  IO.println <|
    (s!"train      = Adam(lr={lr}), epochs={eb.epochs}, " ++
      s!"batch_size={eb.batch}, shuffle=true, drop_last=true, steps={trainSteps}")
  if checkOnly then
    IO.println "mode       = --check-only (validate paths, tensor shapes, and dataset split)"
  (← IO.getStdout).flush
  let dsAll ←
    CLI.orThrow label <|
      (← Data.LabeledSource.load (α := Float) (source xPath yPath nRows))

  if trainSize > dsAll.size then
    throw <| IO.userError
      s!"{label}: --train-size {trainSize} exceeds dataset size {dsAll.size}"

  let (_seed', (dsTrain, dsTest)) := Data.SampleStream.randomSplitAt seed trainSize dsAll

  if checkOnly then
    IO.println s!"loaded     = {dsAll.size} image rows"
    IO.println s!"split      = train {dsTrain.size}, test {dsTest.size}"
    IO.println "check      = dataset shape/path runtime check passed"
    pure ()
  else
    let trainData :=
      Data.batchDataset eb.batch (trainDataset xPath yPath nRows trainSize seed)
        (shuffle := true) (seed := seed) (dropLast := true)
    let trained ← trainer.train trainData
      { steps := trainSteps
        title := "CIFAR10-style NPY CNN tutorial"
        notes :=
          #[s!"x={xPath}", s!"y={yPath}", s!"rows={nRows}",
            s!"train_size={trainSize}", s!"test_size={dsTest.size}",
            s!"epochs={eb.epochs}", s!"batch={eb.batch}", s!"lr={lr}"] }
    trained.printSummary
    let blank : Tensor Float [channels, height, width] :=
      Tensor.full [channels, height, width] 0.0
    trained.printPrediction "blank" (Tensor.repeatAxis 0 eb.batch blank)

end NN.Examples.Data.Loaders.Cifar10Images
