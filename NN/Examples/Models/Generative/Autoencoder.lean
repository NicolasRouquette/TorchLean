/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Run:
  python3 scripts/datasets/download_example_data.py --cifar10
  lake -R -K cuda=true exe torchlean autoencoder --device cuda --steps 1 --n-total 1
-/

module

public import NN.API
public import NN.Examples.Models.Common.RealData

/-!
# Autoencoder CIFAR Example

Trains a compact vector autoencoder on a real CIFAR-10 minibatch.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Generative.Autoencoder

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "torchlean autoencoder"

/-- Default JSON loss-curve path for this command. -/
def defaultLogJson : System.FilePath := ModelZoo.trainLogPath "autoencoder"

/--
Shared vector-image configuration.

The compact config fixes the CIFAR batch size, flattened image dimension, and latent width used by
the vector generative examples, so autoencoder/VAE/VQ-VAE/GAN runs use the same data boundary.
-/
def cfg : nn.models.VectorGenerativeConfig :=
  nn.models.vectorGenerativeConfig 1 16 8 4

/-- Input shape: a batch of flattened CIFAR image vectors. -/
abbrev σ := nn.models.vectorDataShape cfg

/-- Target shape: the same flattened image-vector batch, because this is reconstruction. -/
abbrev τ := nn.models.vectorDataShape cfg

/--
Trainable vector autoencoder.

The architecture is defined in the public model API. The command chooses the dataset, optimizer,
runtime options, and logging path.
-/
def model : nn.Builder (nn.Sequential σ τ) :=
  nn.models.vectorAutoencoder cfg

/-- Public singleton dataset for compact CIFAR reconstruction. -/
def data (flags : RealData.CifarModelTrainFlags) : Trainer.DataSource σ τ :=
  RealData.cifarVectorDataset cfg (by decide) exeName (nn.models.reconstructionSample cfg)
    flags.xPath flags.yPath flags.nRows flags.seed

/-- Train the compact autoencoder with the public `Trainer` surface. -/
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
      (title := "Autoencoder CIFAR reconstruction")
      (notes := RealData.cifarClassifierNotes cfg.batch flags))

/--
Executable entrypoint for CIFAR reconstruction.

The command loads one real CIFAR minibatch, builds the supervised reconstruction sample `x -> x`,
trains the autoencoder for `--steps`, and writes the standard TorchLean training summary/log.
-/
def main (args : List String) : IO UInt32 :=
  TrainCommand.regressionNpy exeName args
    (fun rest => RealData.CifarModelTrainFlags.parse exeName rest defaultLogJson 10 1e-3)
    (ModelZoo.bannerWithDevice exeName "CIFAR vector reconstruction")
    train

end NN.Examples.Models.Generative.Autoencoder
