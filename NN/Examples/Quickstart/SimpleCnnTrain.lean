/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.Examples.Data.Bands
public import NN.Examples.Quickstart.Common

/-!
# Simple CNN training example

This is the image-classification companion to the small MLP tutorial. The file shows the public
training path in its simplest CNN form: define the layers, choose a trainer, batch the small dataset,
and call `trainer.train`. Shape-indexed tensors and the checked training task remain present, but
the first read follows the model code rather than subsystem plumbing.

Check this tutorial module directly:

- `lake build NN.Examples.Quickstart.SimpleCnnTrain`

For the maintained command-line CNN trainer, use `NN/Examples/Models/Vision/Cnn.lean`:

- `python3 scripts/datasets/download_example_data.py --cifar10`
- `lake -R -K cuda=true exe torchlean cnn --device cuda --n-total 1 --steps 1`

Optional flags:

- `--steps N`
- `--batch N`

See `NN/Examples/Quickstart/README.md` for the shared conventions in this folder.
-/

@[expose] public section


namespace NN.Examples.Quickstart.SimpleCNNTrain

open TorchLean

/-- Default JSON log path used only when the user explicitly passes `--log`. -/
def defaultLogJson : System.FilePath := ModelZoo.trainLogPath "quickstart_simple_cnn"

def mkModel {batch : Nat} :
    nn.Builder (nn.Sequential [batch, 1, 4, 4] [batch, 2]) :=
  let cfg : nn.models.CnnConfig 2 :=
    { inChannels := 1
      spatial := tensor! [4, 4]
      outDim := 2
      conv :=
        { outChannels := 3
          kernel := tensor! [2, 2]
          kernelNonzero := by intro i; fin_cases i <;> decide
          strideNonzero := by intro i; fin_cases i <;> decide }
      pool :=
        { kernel := tensor! [1, 1]
          kernelNonzero := by intro i; fin_cases i <;> decide
          strideNonzero := by intro i; fin_cases i <;> decide } }
  by
    simpa [cfg, nn.models.CnnConfig.inputShape, nn.models.CnnConfig.outputShape,
      Spec.Shape.ofList, Spec.Shape.concat, Spec.Shape.appendDim] using
      nn.models.cnn cfg [batch] (hInChannels := by simp [cfg])

/-- Command-line help for the simple CNN quickstart. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean simple CNN quickstart"
    , ""
    , "Usage:"
    , "  lake exe torchlean quickstart_cnn [options]"
    , ""
    , "Options:"
    , "  --seed N"
    , "  --batch N"
    , "  --steps N"
    , "  --scalar float32|ieee32-exec"
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
  IO.println "== Quickstart next step: simple CNN training =="
  let (seed, args) ← CLI.seed "SimpleCNNTrain" args 0
  let (batch, args) ← CLI.positiveNatFlag "SimpleCNNTrain" args "batch" 2
  let parsed ←
    _root_.NN.Examples.Quickstart.parseRuntimeTrain
      "SimpleCNNTrain" args defaultLogJson 1 (optim.adam { lr := 0.03 })
  let trainer := Trainer.new (mkModel (batch := batch)) <|
    Trainer.Config.fromRunConfig parsed.run (.oneHotCrossEntropy 1) (seed := seed)

  let trainData :=
    Data.batchDataset batch Data.Bands.dataset
      (shuffle := true) (seed := seed) (dropLast := true)
  let trained ← trainer.train trainData parsed.trainOptions
  trained.printSummary
  match Data.Bands.probeSamples[0]? with
  | none => pure ()
  | some (name, x, expected) =>
      let xBatch := Tensor.repeatAxis 0 batch x
      trained.printPrediction s!"{name} expected={expected}" xBatch

end NN.Examples.Quickstart.SimpleCNNTrain
