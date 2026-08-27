/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Run:
  python3 scripts/datasets/download_example_data.py --cifar10
  lake -R -K cuda=true exe torchlean mae --device cuda --steps 1 --n-total 1
-/

module

public import NN.API
public import NN.Examples.Models.Common.RealData

/-!
# Masked Autoencoder CIFAR Example

This is the compact ViT-MAE-style training path in TorchLean.

The data path is explicit:

1. load real CIFAR-10 `.npy` arrays through `Data`;
2. take a typed image batch with shape `[batch, channels, height, width]`;
3. hide deterministic spatial blocks with `ssl.BlockMAE.sample`;
4. run a ViT encoder over patch tokens;
5. train a decoder head to reconstruct the original image vector.

The architecture uses one transformer encoder block and a linear pixel decoder rather than a large
asymmetric MAE decoder. It exercises image patch masking, patch embedding, transformer tokens, and
reconstruction of the original image.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Generative.Mae

/-- Command name used in error messages and CLI output. -/
def exeName : String := "torchlean mae"

/-- Default JSON loss-curve path for this command. -/
def defaultLogJson : System.FilePath := ModelZoo.trainLogPath "mae"

/-- CIFAR minibatch size used by the typed MAE command. -/
def batch : Nat := 1

/-- Number of CIFAR image channels. -/
def inC : Nat := RealData.cifarChannels

/-- Cropped CIFAR image height for the compact runnable example. -/
def inH : Nat := 2

/-- Cropped CIFAR image width for the compact runnable example. -/
def inW : Nat := 2

/-- Patch height for the image-to-token projection. -/
def patchH : Nat := 2

/-- Patch width for the image-to-token projection. -/
def patchW : Nat := 2

/-- Patch stride; equal to patch size here, so patches do not overlap. -/
def stride : Nat := 2

/-- Zero padding around the image before patch extraction. -/
def padding : Nat := 0

/-- Width of each patch token after projection into the encoder stream. -/
def dModel : Nat := 1

/-- Number of self-attention heads in the compact ViT encoder. -/
def numHeads : Nat := 1

/-- Per-head attention width; $\mathtt{numHeads}\cdot\mathtt{headDim}=\mathtt{dModel}$. -/
def headDim : Nat := 1

/-- Hidden width of the feed-forward block inside the encoder. -/
def ffnHidden : Nat := 2

/-- Number of reconstructed flattened pixels predicted by the decoder head. -/
def reconDim : Nat := 4

/--
Small ViT-MAE configuration.

The command crops CIFAR images to `2×2`, uses one image patch, and reconstructs a tiny prefix of the
flattened image. That keeps MAE in the runnable quick-check suite while still checking the patch masking,
patch embedding, transformer token, decoder, data loading, and CUDA training path.
-/
def cfg : nn.models.VitMaeConfig 2 :=
  { encoder :=
      { inChannels := inC
        spatial := tensor! [inH, inW]
        patch :=
          { outChannels := dModel
            kernel := tensor! [patchH, patchW]
            stride := tensor! [stride, stride]
            padding := tensor! [padding, padding]
            kernelNonzero := by intro i; fin_cases i <;> decide
            strideNonzero := by intro i; fin_cases i <;> simp [stride] }
        outDim := reconDim
        numHeads := numHeads
        headDim := headDim
        ffnHidden := ffnHidden }
    reconDim := reconDim }

/--
Hide one patch-index class every four patch positions.

The image remains an image tensor; the mask zeros whole patch regions before patch embedding.
-/
def maskPeriod : Nat := 4

/-- Phase of the deterministic patch mask. Changing this selects a different patch-index class. -/
def maskOffset : Nat := 0

/-- Leading sample axis used by this batched training example. -/
abbrev batchShape : List Nat := [batch]

/-- Input shape: a real batched CIFAR image tensor. -/
abbrev σ := cfg.encoder.inputShape batchShape

/-- Output shape: flattened image reconstruction. -/
abbrev τ := cfg.outputShape batchShape

/-- CIFAR-10 images are stored as `3 × 32 × 32` tensors. -/
def cifarClasses : Nat := RealData.cifarClasses

/--
Construct the trainable model.

The architecture lives in the public self-supervised model API; this example only chooses a config,
loads data, and trains it.
-/
def model : nn.Builder (nn.Sequential σ τ) :=
  nn.models.vitMaskedAutoencoder cfg batchShape
    (h_inC := by decide)
    (h_dModel := by decide)

/--
Turn a typed CIFAR image batch into the compact MAE training sample.

The input stays an image tensor with some patches zeroed out. The target is the original image
flattened to a vector because the current decoder head predicts a batched matrix.
-/
def mkMaeSample
    (b : Sample.Supervised Float
      [batch, cfg.encoder.inChannels, inH, inW]
      [batch, RealData.cifarClasses]) :
  Sample.Supervised Float σ τ := by
  let dataShape : Tensor Nat [3] :=
    tensor! [cfg.encoder.inChannels, inH, inW]
  let blocks : Tensor (Option Nat) [3] :=
    tensor! [none, some patchH, some patchW]
  have hInputShape :
      [batch, cfg.encoder.inChannels, inH, inW] = [batch] ++ dataShape.toList := by
    simp [dataShape]
  let x : Tensor Float ([batch] ++ dataShape.toList) := hInputShape ▸ Sample.x b
  simpa [σ, τ, nn.models.VitMaeConfig.outputShape,
    nn.models.VitConfig.inputShape, batchShape, cfg, dataShape] using
    ssl.BlockMAE.sample [batch] cfg.reconDim dataShape blocks
      maskPeriod maskOffset (by decide) x

/--
Public singleton dataset for masked-image reconstruction on one real CIFAR batch.

Like the compact vector generative examples, the sample itself is loaded as `Float` from the real
data boundary, then cast into the runtime-selected scalar by the public dataset constructor.
-/
def data (flags : RealData.CifarModelTrainFlags) : Trainer.Dataset σ τ :=
  Data.singletonFloatIO do
    let sampleBatch ←
      RealData.loadCifarBatch exeName batch flags.nRows flags.seed
        flags.xPath flags.yPath
    pure <| mkMaeSample <|
      RealData.cropCifarBatch batch inH inW (by decide) (by decide) sampleBatch

/-- Train the compact MAE model with the public `Trainer` surface. -/
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
      (title := "MAE CIFAR masked reconstruction")
      (notes := RealData.cifarClassifierNotes batch flags
        #[s!"maskPeriod={maskPeriod}", s!"maskOffset={maskOffset}"]))

/--
CLI entrypoint.

Useful flags:
- `--device cuda` runs the public trainer on the CUDA runtime.
- `--steps <n>` controls optimization steps.
- `--x <path> --y <path>` selects custom CIFAR-style `.npy` arrays.
- `--log <path>` writes the standard TorchLean training log JSON.
-/
def main (args : List String) : IO UInt32 :=
  TrainCommand.regressionNpy exeName args
    (fun rest => RealData.CifarModelTrainFlags.parse exeName rest defaultLogJson 10 1e-3)
    (ModelZoo.bannerWithDevice exeName "CIFAR masked reconstruction")
    train

end NN.Examples.Models.Generative.Mae
