/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Run:
  python3 scripts/datasets/download_example_data.py --cifar10
  lake -R -K cuda=true exe torchlean tanh_autoencoder --device cuda --steps 1 --n-total 1
-/

module

public import NN.API
public import NN.Examples.Models.Common.RealData

/-!
# Tanh-Bottleneck Autoencoder CIFAR Example

Trains a compact vector reconstruction model with a narrow continuous `tanh` bottleneck. It is not
a vector-quantized autoencoder: there is no codebook, nearest-code lookup, or commitment objective.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Generative.TanhBottleneckAutoencoder

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "torchlean tanh_autoencoder"

/-- Default JSON loss-curve path for this command. -/
def defaultLogJson : System.FilePath := ModelZoo.trainLogPath "tanh_autoencoder"

/--
Shared vector-image configuration.

This command uses the same compact flattened-CIFAR boundary as the ordinary autoencoder, changing
only the bottleneck activation while keeping data handling fixed.
-/
def cfg : nn.models.DenseGenerative.Config :=
  { dataDim := 16, hiddenDim := 8, latentDim := 4 }

/-- Number of image vectors loaded for each training sample. -/
def batch : Nat := 1

/-- Input shape: a batch of flattened CIFAR image vectors. -/
abbrev σ := cfg.dataShape [batch]

/-- Target shape: reconstructed flattened CIFAR image vectors. -/
abbrev τ := cfg.dataShape [batch]

/-- Trainable vector autoencoder with a narrow continuous `tanh` bottleneck. -/
def model : nn.Builder (nn.Sequential σ τ) :=
  nn.models.DenseGenerative.tanhBottleneckAutoencoder cfg [batch]

/-- Public singleton dataset for compact CIFAR reconstruction. -/
def data (flags : RealData.CifarModelTrainFlags) : Trainer.Dataset σ τ :=
  RealData.cifarFeatureDataset batch cfg (by decide) exeName (fun x ↦ Sample.mk x x)
    flags.xPath flags.yPath flags.nRows flags.seed

/-- Train the tanh-bottleneck autoencoder with the public `Trainer` surface. -/
def train (opts : Options) (flags : RealData.CifarModelTrainFlags) :
    IO (Trainer.TrainResult σ τ) := do
  Data.requirePairedFiles exeName
    "CIFAR-10 images" flags.xPath
    "CIFAR-10 labels" flags.yPath
    RealData.missingCifarHint
  let trainer :=
    Trainer.new model <|
      Trainer.Config.fromRunConfig
        (Trainer.RunConfig.ofRuntimeOptions opts { optimizer := optim.adam { lr := flags.lr } })
        .regression
        (seed := flags.seed)
  trainer.train
    (data flags)
    (CLI.Training.OptimizerOptions.toTrainerOptions flags.toOptimizerOptions
      (title := "Tanh-bottleneck CIFAR reconstruction")
      (notes := RealData.cifarClassifierNotes batch flags #[s!"latentDim={cfg.latentDim}"]))

/--
Executable entrypoint for the tanh-bottleneck autoencoder run.

The command loads a real CIFAR minibatch, trains the reconstruction objective, and records the same
summary/log artifact format as the other public trainer commands.
-/
def main (args : List String) : IO UInt32 :=
  TrainCommand.regressionNpy exeName args
    (fun rest => RealData.CifarModelTrainFlags.parse exeName rest defaultLogJson 10 1e-3)
    (ModelZoo.bannerWithDevice exeName "CIFAR tanh-bottleneck autoencoder")
    train

end NN.Examples.Models.Generative.TanhBottleneckAutoencoder
