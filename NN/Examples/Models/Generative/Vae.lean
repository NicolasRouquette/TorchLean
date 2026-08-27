/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Run:
  python3 scripts/datasets/download_example_data.py --cifar10
  lake -R -K cuda=true exe torchlean latent_stats --device cuda --steps 1 --n-total 1
-/

module

public import NN.API
public import NN.Examples.Models.Common.RealData

/-!
# Supervised Latent-Statistics CIFAR Example

This compact supervised model reconstructs flattened CIFAR images while driving two auxiliary
latent-statistic vectors toward zero. It is not a variational autoencoder: it has no latent
distribution, reparameterized sampling, or ELBO objective.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Generative.LatentStatistics

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "torchlean latent_stats"

/-- Default JSON loss-curve path for this command. -/
def defaultLogJson : System.FilePath := ModelZoo.trainLogPath "latent_stats"

/--
Shared vector-image configuration.

The runtime example uses the same flattened CIFAR data boundary as the other vector generative
commands, with two auxiliary latent-statistic vectors appended to the reconstruction.
-/
def cfg : nn.models.DenseGenerative.Config :=
  { dataDim := 16, hiddenDim := 8, latentDim := 4 }

/-- Number of image vectors loaded for each training sample. -/
def batch : Nat := 1

/-- Input shape: a batch of flattened CIFAR image vectors. -/
abbrev σ := cfg.dataShape [batch]

/-- Output shape: reconstruction plus latent regularization proxy channels. -/
abbrev τ := cfg.reconstructionStatisticsShape [batch]

namespace Internal

/-- Reconstruction target followed by zero latent-statistic proxy channels. -/
def target (x : Tensor Float σ) : Tensor Float τ :=
  Tensor.stack 0 fun bi =>
    let row := Tensor.get x bi
    Tensor.stack 0 fun j =>
      let value :=
        if h : j.val < cfg.dataDim then
          Tensor.item (Tensor.get row ⟨j.val, h⟩)
        else
          0.0
      Tensor.full [] value

/-- Supervised reconstruction sample with zero-valued auxiliary latent-statistic targets. -/
def sample (x : Tensor Float σ) : Sample.Supervised Float σ τ :=
  Sample.mk x (target x)

end Internal

/-- Trainable reconstruction model with two supervised auxiliary latent vectors. -/
def model : nn.Builder (nn.Sequential σ τ) :=
  nn.models.DenseGenerative.reconstructionWithLatentStatistics cfg [batch]

/-- Public singleton dataset for compact CIFAR reconstruction plus latent-stat targets. -/
def data (flags : RealData.CifarModelTrainFlags) : Trainer.Dataset σ τ :=
  RealData.cifarFeatureDataset batch cfg (by decide) exeName Internal.sample
    flags.xPath flags.yPath flags.nRows flags.seed

/-- Train the latent-statistics reconstruction model with the public `Trainer` surface. -/
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
      (title := "CIFAR reconstruction with latent statistics")
      (notes := RealData.cifarClassifierNotes batch flags #[s!"latentDim={cfg.latentDim}"]))

/--
Executable entrypoint for supervised reconstruction with auxiliary latent statistics.

The command loads CIFAR vectors, constructs the reconstruction/latent-proxy target, trains with
Adam, and writes the standard TorchLean training summary/log.
-/
def main (args : List String) : IO UInt32 :=
  TrainCommand.regressionNpy exeName args
    (fun rest => RealData.CifarModelTrainFlags.parse exeName rest defaultLogJson 10 1e-3)
    (ModelZoo.bannerWithDevice exeName "CIFAR latent-statistics reconstruction")
    train

end NN.Examples.Models.Generative.LatentStatistics
