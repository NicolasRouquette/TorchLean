/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Real-data CUDA example:
  python3 scripts/datasets/download_example_data.py --cifar10
  lake -R -K cuda=true exe torchlean cnn --device cuda --n-total 1 --steps 1
-/

module


public import NN.API
public import NN.Examples.Models.Common.RealData

/-!
# CNN Training Example

Runnable `torchlean cnn` example. It trains a small convolutional classifier on a prepared CIFAR-10
minibatch.

The reusable model wiring lives behind the public `TorchLean.nn.models.cnn` constructor. The command
adds the pieces around it: CLI parsing, dataset selection, step-limited loader training, and TrainLog
artifact writing.

```bash
python3 scripts/datasets/download_example_data.py --cifar10
lake -R -K cuda=true exe torchlean cnn --device cuda --n-total 1 --steps 1
```
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Vision.Cnn

/-- CLI subcommand name used in terminal banners and parser errors. -/
def exeName : String := "torchlean cnn"

/-- Default JSON loss-curve path for this command. -/
def defaultLogJson : System.FilePath := ModelZoo.trainLogPath "cnn"

/--
Static minibatch size for the compact CIFAR run.

The model owns the batch axis, so this value appears in both the input/output shapes and the
classifier trainer type.
-/
def batch : Nat := 1

/-- CIFAR image channels. -/
def inC : Nat := 3

/-- Height of the CIFAR crop used by this runnable CNN command. -/
def inH : Nat := 8

/-- Width of the CIFAR crop used by this runnable CNN command. -/
def inW : Nat := 8

/-- CIFAR class count, hence the output-logit width. -/
def outDim : Nat := RealData.cifarClasses

/-- Shared CNN configuration used by shapes and the reusable public model constructor. -/
def cfg : nn.models.CnnConfig 2 :=
  { inChannels := inC
    spatial := tensor! [inH, inW]
    outDim := outDim
    conv :=
      { outChannels := 4
        kernel := tensor! [3, 3]
        stride := tensor! [2, 2]
        padding := tensor! [1, 1]
        kernelNonzero := by intro i; fin_cases i <;> decide
        strideNonzero := by intro i; fin_cases i <;> decide }
    pool :=
      { kernel := tensor! [2, 2]
        stride := tensor! [2, 2]
        kernelNonzero := by intro i; fin_cases i <;> decide
        strideNonzero := by intro i; fin_cases i <;> decide } }

/-- Input shape: a minibatch of CIFAR images in channel-first layout. -/
abbrev σ : List Nat := [batch, inC, inH, inW]

/-- Output shape: one row of class logits per image. -/
abbrev τ : List Nat := [batch, outDim]

/--
Small convolutional classifier from the public model API.

The command chooses the CIFAR paths and runtime flags; the model itself stays an ordinary
`nn.Sequential` value built from the public API.
-/
def model : nn.Builder (nn.Sequential σ τ) :=
  by
    simpa [σ, τ, cfg, nn.models.CnnConfig.inputShape, nn.models.CnnConfig.outputShape,
      Spec.Shape.ofList, Spec.Shape.concat, Spec.Shape.appendDim] using
      nn.models.cnn cfg [batch]

/-- Train the CIFAR CNN with the public `Trainer` surface. -/
def train (opts : Options) (flags : RealData.CifarModelTrainFlags) :
    IO Trainer.TrainSummary := do
  let batches ←
    RealData.loadCifarBatches exeName batch flags.nRows flags.seed flags.xPath flags.yPath
  let batches := batches.map (RealData.cropCifarBatch batch inH inW (by decide) (by decide))
  let trainer :=
    Trainer.new model <|
      Trainer.Config.fromRunConfig
        (Trainer.RunConfig.ofRuntimeOptions opts { optimizer := optim.adam { lr := flags.lr } })
        (.oneHotCrossEntropy 1)
        (seed := flags.seed)
  let trained ← trainer.train
    (Data.floatSamples batches)
    (CLI.Training.OptimizerOptions.toTrainerOptions flags.toOptimizerOptions
      (title := "CNN training")
      (notes := RealData.cifarClassifierNotes batch flags))
  pure trained.report

/-- CLI entrypoint for CIFAR CNN training; CUDA is the maintained validation path. -/
def main (args : List String) : IO UInt32 :=
  TrainCommand.classificationNpy exeName args
    (fun rest => RealData.CifarModelTrainFlags.parse exeName rest defaultLogJson 1 1e-3)
    (ModelZoo.bannerWithDevice exeName "CNN training")
    train

end NN.Examples.Models.Vision.Cnn
