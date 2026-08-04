/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Run:
  python3 scripts/datasets/download_example_data.py --cifar10
  lake -R -K cuda=true exe torchlean vqvae --device cuda --steps 1 --n-total 1
-/

module

public import NN.API
public import NN.Examples.Models.Common.RealData
public import NN.Spec.Models.VqVae
public import NN.MLTheory.Generative.Latent.VQVAE

/-!
# VQ-VAE-Style CIFAR Example

Trains a compact vector reconstruction model with a narrow `tanh` bottleneck, paired with the
VQ-VAE spec/theory modules. The codebook objective is stated in `NN.Spec.Models.VqVae`; this runtime
example is the executable reconstruction path.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Generative.VqVae

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "torchlean vqvae"

/-- Default JSON loss-curve path for this command. -/
def defaultLogJson : System.FilePath := ModelZoo.trainLogPath "vqvae"

/--
Shared vector-image configuration.

The VQ-VAE runtime path uses the same compact flattened-CIFAR boundary as the autoencoder and VAE
commands, so the model comparison changes the bottleneck while keeping data handling fixed.
-/
def cfg : nn.models.VectorGenerativeConfig := nn.models.compactImageConfig

/-- Input shape: a batch of flattened CIFAR image vectors. -/
abbrev σ := nn.models.vectorDataShape cfg

/-- Target shape: reconstructed flattened CIFAR image vectors. -/
abbrev τ := nn.models.vectorDataShape cfg

/--
Trainable VQ-VAE-style vector model.

The codebook-facing objective is handled in the imported spec/theory modules; this command exercises
the executable reconstruction path with a narrow quantization-style bottleneck.
-/
def model : nn.M (nn.Sequential σ τ) :=
  nn.models.vectorVqVae cfg

/-- Public singleton dataset for compact CIFAR reconstruction. -/
def data (flags : RealData.CifarModelTrainFlags) : Trainer.Dataset σ τ :=
  RealData.cifarVectorDataset cfg (by decide) exeName (nn.models.reconstructionSample cfg)
    flags.xPath flags.yPath flags.nRows flags.seed

/-- Train the compact VQ-VAE-style model with the public `Trainer` surface. -/
def train (opts : Options) (flags : RealData.CifarModelTrainFlags) :
    IO (Trainer.TrainResult σ τ) := do
  Data.requirePairedFiles exeName
    "CIFAR-10 images" flags.xPath
    "CIFAR-10 labels" flags.yPath
    RealData.missingCifarHint
  let trainer :=
    Trainer.new model <|
      Trainer.Config.fromRunConfig
        (Trainer.runConfig opts { optimizer := optim.adam { lr := flags.lr } })
        .regression
        (seed := flags.seed)
  trainer.train
    (data flags)
    (CLI.Training.OptimizerOptions.toTrainerOptions flags.toOptimizerOptions
      (title := "VQ-VAE-style CIFAR reconstruction")
      (notes := RealData.cifarClassifierNotes cfg.batch flags #[s!"latentDim={cfg.latentDim}"]))

/--
Executable entrypoint for the compact VQ-VAE-style run.

The command loads a real CIFAR minibatch, trains the reconstruction objective, and records the same
summary/log artifact format as the other public trainer commands.
-/
def main (args : List String) : IO UInt32 :=
  TrainCommand.regressionNpy exeName args
    (fun rest => RealData.CifarModelTrainFlags.parse exeName rest defaultLogJson 10 1e-3)
    (ModelZoo.bannerWithDevice exeName "CIFAR VQ-VAE-style training")
    train

end NN.Examples.Models.Generative.VqVae
