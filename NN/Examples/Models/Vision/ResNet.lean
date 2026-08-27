/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Run:
  python3 scripts/datasets/download_example_data.py --cifar10
  lake -R -K cuda=true exe torchlean resnet --device cuda --n-total 1 --steps 1
-/

module

public import NN.API
public import NN.Examples.Models.Common.RealData

/-!
# Residual Classifier Training Example

This command trains the public rank-polymorphic residual classifier on a small CIFAR-10 crop. The
model contains a convolutional stem, two residual branches, global average pooling over all spatial
axes, and a linear classifier. The compact crop keeps the command useful as a runtime check while
still exercising the same residual composition and pooling code used by larger configurations.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Vision.ResNet

/-- CLI subcommand name used in terminal banners and parser errors. -/
def exeName : String := "torchlean resnet"

/-- Default JSON training-log path. -/
def defaultLogJson : System.FilePath := ModelZoo.trainLogPath "resnet"

/-- Static minibatch size carried by the checked model shape. -/
def batch : Nat := 1

/-- CIFAR input channels. -/
def inChannels : Nat := RealData.cifarChannels

/-- Height of the compact CIFAR crop. -/
def height : Nat := 8

/-- Width of the compact CIFAR crop. -/
def width : Nat := 8

/-- Channel width of the residual trunk. -/
def hiddenChannels : Nat := 4

/-- A three-point stencil along each spatial axis, with unit stride and same padding. -/
def blockGeometry : nn.ConvGeometry 2 :=
  nn.ConvGeometry.samePadding tensor! [1, 1]

/-- Configuration shared by the model constructor and its typed input/output shapes. -/
def cfg : nn.models.ResNetConfig 2 :=
  { inChannels := inChannels
    spatial := tensor! [height, width]
    spatialNonzero := by intro i; fin_cases i <;> decide
    hiddenChannels := hiddenChannels
    block := blockGeometry
    blockPreservesSpatial := by
      simpa [nn.ConvGeometry.outSpatial, blockGeometry] using
        nn.ConvGeometry.outSpatial_samePadding tensor! [height, width] tensor! [1, 1]
          (by intro i; fin_cases i <;> decide)
    numClasses := RealData.cifarClasses }

/-- Batched channel-first input shape. -/
abbrev σ : List Nat := [batch, inChannels, height, width]

/-- One row of class logits per input sample. -/
abbrev τ : List Nat := [batch, RealData.cifarClasses]

/-- Residual classifier from the public model API. -/
def model : nn.Builder (nn.Sequential σ τ) :=
  by
    simpa [σ, τ, cfg, nn.models.ResNetConfig.inputShape,
      nn.models.ResNetConfig.outputShape, Spec.Shape.ofList, Spec.Shape.concat,
      Spec.Shape.appendDim] using
      nn.models.resnet cfg [batch]
        (hInChannels := by decide) (hHiddenChannels := by decide)

/-- Train the residual classifier with the public classification trainer. -/
def train (opts : Options) (flags : RealData.CifarModelTrainFlags) :
    IO Trainer.TrainSummary := do
  let batches ←
    RealData.loadCifarBatches exeName batch flags.nRows flags.seed flags.xPath flags.yPath
  let batches := batches.map (RealData.cropCifarBatch batch height width (by decide) (by decide))
  let trainer :=
    Trainer.new model <|
      Trainer.Config.fromRunConfig
        (Trainer.RunConfig.ofRuntimeOptions opts { optimizer := optim.adam { lr := flags.lr } })
        (.oneHotCrossEntropy 1)
        (seed := flags.seed)
  let trained ← trainer.train
    (Data.floatSamples batches)
    (CLI.Training.OptimizerOptions.toTrainerOptions flags.toOptimizerOptions
      (title := "ResNet CIFAR training")
      (notes := RealData.cifarClassifierNotes batch flags
        #[s!"spatial={height}x{width}", s!"hiddenChannels={hiddenChannels}"]))
  pure trained.report

/-- CLI entrypoint for the CIFAR residual-classifier training path. -/
def main (args : List String) : IO UInt32 :=
  TrainCommand.classificationNpy exeName args
    (fun rest => RealData.CifarModelTrainFlags.parse exeName rest defaultLogJson 1 1e-3)
    (ModelZoo.bannerWithDevice exeName "ResNet CIFAR training")
    train

end NN.Examples.Models.Vision.ResNet
