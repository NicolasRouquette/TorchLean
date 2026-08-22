/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Run:
  python3 scripts/datasets/download_example_data.py --cifar10
  lake -R -K cuda=true exe torchlean vae --device cuda --steps 1 --n-total 1
-/

module

public import NN.API
public import NN.Examples.Models.Common.RealData
public import NN.Spec.Models.Vae
public import NN.MLTheory.Generative.Latent.VAE

/-!
# β-VAE-Style CIFAR Example

Runnable compact VAE path over flattened CIFAR images.

The formal VAE objective and decomposition theorems live in `NN.Spec.Models.Vae` and
`NN.MLTheory.Generative.Latent.VAE`. This executable uses a compact supervised runtime target:
reconstruct the image while keeping latent mean/log-variance proxy channels near zero.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Generative.Vae

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "torchlean vae"

/-- Default JSON loss-curve path for this command. -/
def defaultLogJson : System.FilePath := ModelZoo.trainLogPath "vae"

/--
Shared vector-image configuration.

The runtime example uses the same flattened CIFAR data boundary as the other vector generative
commands, while the VAE-specific output shape adds latent mean/log-variance proxy channels.
-/
def cfg : nn.models.DenseGenerative.Config :=
  { dataDim := 16, hiddenDim := 8, latentDim := 4 }

/-- Number of image vectors loaded for each training sample. -/
def batch : Nat := 1

/-- Input shape: a batch of flattened CIFAR image vectors. -/
abbrev σ := cfg.dataShape (.dim batch .scalar)

/-- Output shape: reconstruction plus latent regularization proxy channels. -/
abbrev τ := cfg.vaeOutputShape (.dim batch .scalar)

namespace Internal

/-- Reconstruction target followed by zero latent-statistic proxy channels. -/
def target (x : Tensor Float σ) : Tensor Float τ :=
  .dim fun bi =>
    let row := Spec.getAtSpec x bi
    .dim fun j =>
      let value :=
        if h : j.val < cfg.dataDim then
          Tensor.item (Spec.get row ⟨j.val, h⟩)
        else
          0.0
      .scalar value

/-- Supervised sample used by this compact executable VAE path. -/
def sample (x : Tensor Float σ) : Sample.Supervised Float σ τ :=
  Sample.mk x (target x)

end Internal

/--
Trainable VAE-style vector model.

The executable target is still an MSE-style supervised sample; the imported spec/theory files state
the VAE objective separately.
-/
def model : nn.Builder (nn.Sequential σ τ) :=
  nn.models.DenseGenerative.vae cfg (.dim batch .scalar)

/-- Public singleton dataset for compact CIFAR reconstruction plus latent-stat targets. -/
def data (flags : RealData.CifarModelTrainFlags) : Trainer.DataSource σ τ :=
  RealData.cifarFeatureDataset batch cfg (by decide) exeName Internal.sample
    flags.xPath flags.yPath flags.nRows flags.seed

/-- Train the compact VAE-style model with the public `Trainer` surface. -/
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
      (title := "VAE-style CIFAR reconstruction")
      (notes := RealData.cifarClassifierNotes batch flags #[s!"latentDim={cfg.latentDim}"]))

/--
Executable entrypoint for the compact VAE-style run.

The command loads CIFAR vectors, constructs the reconstruction/latent-proxy target, trains with
Adam, and writes the standard TorchLean training summary/log.
-/
def main (args : List String) : IO UInt32 :=
  TrainCommand.regressionNpy exeName args
    (fun rest => RealData.CifarModelTrainFlags.parse exeName rest defaultLogJson 10 1e-3)
    (ModelZoo.bannerWithDevice exeName "CIFAR beta-VAE-style training")
    train

end NN.Examples.Models.Generative.Vae
