/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Run:
  python3 scripts/datasets/download_example_data.py --cifar10
  lake exe -K cuda=true torchlean mae --cuda --steps 25
-/

module

public import NN.API.Common
public import NN.API.Data
public import NN.API.Models.SelfSupervised
public import NN.API.Models.TrainFixed
public import NN.API.SelfSupervised
public import NN.Examples.Models.Common.RealData

/-!
# Masked Autoencoder CIFAR Example

This is the compact ViT-MAE-style training path in TorchLean.

The data path is explicit:

1. load real CIFAR-10 `.npy` arrays through `NN.API.Data`;
2. take a typed image batch with shape `[batch, channels, height, width]`;
3. hide deterministic image patches with `ssl.imagePatchMaeSample`;
4. run a ViT encoder over patch tokens;
5. train a decoder head to reconstruct the original image vector.

The architecture uses one transformer encoder block and a linear pixel decoder rather than a large
asymmetric MAE decoder. The important pieces are the MAE pieces exercised by the
example: image patch masking, patch embedding, transformer tokens, and reconstruction of the
original image.
-/

@[expose] public section

open Spec Tensor
open NN.API

namespace NN.Examples.Models.Generative.Mae

/-- Command name used in error messages and CLI output. -/
def exeName : String := "torchlean mae"

/-- Default JSON loss-curve path for this command. -/
def defaultLogJson : System.FilePath := Common.modelZooTrainLog "mae"

/-- CIFAR minibatch size used by the typed MAE command. -/
def batch : Nat := 1

/-- Number of CIFAR image channels. -/
def inC : Nat := RealData.cifarChannels

/-- CIFAR image height in pixels. -/
def inH : Nat := RealData.cifarHeight

/-- CIFAR image width in pixels. -/
def inW : Nat := RealData.cifarWidth

/-- Patch height for the image-to-token projection. -/
def patchH : Nat := 16

/-- Patch width for the image-to-token projection. -/
def patchW : Nat := 16

/-- Patch stride; equal to patch size here, so patches do not overlap. -/
def stride : Nat := 16

/-- Zero padding around the image before patch extraction. -/
def padding : Nat := 0

/-- Width of each patch token after projection into the encoder stream. -/
def dModel : Nat := 8

/-- Number of self-attention heads in the compact ViT encoder. -/
def numHeads : Nat := 1

/-- Per-head attention width; `numHeads * headDim = dModel`. -/
def headDim : Nat := 8

/-- Hidden width of the feed-forward block inside the encoder. -/
def ffnHidden : Nat := 32

/-- Number of reconstructed flattened pixels predicted by the decoder head. -/
def reconDim : Nat := 256

/--
Small ViT-MAE configuration.

CIFAR-10 with `16×16` patches gives `2×2 = 4` patch tokens. The model embeds patches into
`dModel = 8`, runs one transformer encoder block, then decodes the flattened token state back to a
256-pixel prefix of the original image. The configuration keeps the command runnable while still
exercising a real patch-token transformer path. For full-image reconstruction, set
`reconDim := inC * inH * inW`.
-/
def cfg : nn.models.VitMaeConfig :=
  { batch := batch
    inC := inC
    inH := inH
    inW := inW
    patchH := patchH
    patchW := patchW
    stride := stride
    padding := padding
    dModel := dModel
    reconDim := reconDim
    numHeads := numHeads
    headDim := headDim
    ffnHidden := ffnHidden }

/--
Hide one patch-index class every four patch positions.

The image remains an image tensor; the mask zeros whole patch regions before patch embedding.
-/
def maskPeriod : Nat := 4

/-- Phase of the deterministic patch mask. Changing this selects a different patch-index class. -/
def maskOffset : Nat := 0

/-- Input shape: a real batched CIFAR image tensor. -/
abbrev σ : Shape := nn.models.vitMaeInShape cfg

/-- Output shape: flattened image reconstruction. -/
abbrev τ : Shape := nn.models.vitMaeOutShape cfg

/-- CIFAR-10 images are stored as `3 × 32 × 32` tensors. -/
def cifarClasses : Nat := RealData.cifarClasses

/--
Construct the trainable model.

The architecture lives in the public API (`NN.API.Models.SelfSupervised`); the example only chooses
a config and trains it.
-/
def mkModel : nn.M (nn.Sequential σ τ) :=
  nn.models.vitMaskedAutoencoder cfg

/--
Turn a typed CIFAR image batch into the compact MAE training sample.

The input stays an image tensor with some patches zeroed out. The target is the original image
flattened to a vector because the current decoder head predicts a batched matrix.
-/
def mkMaeSample
    (b : sample.Batch Float cfg.batch RealData.CifarImage RealData.CifarTarget) :
    sample.Supervised Float σ τ :=
  ssl.imagePatchMaeSample cfg.batch cfg.inC cfg.inH cfg.inW cfg.reconDim cfg.patchH cfg.patchW
    maskPeriod maskOffset (by decide) (sample.x b)

/--
Train and return a loss curve.

The curve is written by `main` using TorchLean's general training-log JSON format, so plotting and
dashboard tools can consume it the same way they consume the other model examples.
-/
def trainCurve (opts : TorchLean.Options) (xPath yPath : System.FilePath)
    (nRows seed steps cudaMemWatch : Nat) : IO _root_.Runtime.Training.Curve := do
  let batch ← RealData.loadCifarBatch (α := Float) exeName cfg.batch nRows seed xPath yPath
  let sample := mkMaeSample batch
  _root_.NN.API.Models.TrainFixed.curveFloat
    (mkModel := mkModel)
    (mkModuleDef := fun model => nn.mseScalarModuleDef model)
    (mkOptim := fun ps =>
      TorchLean.Optim.adam (α := Float) (paramShapes := ps)
        (lr := 1e-3) (beta1 := 0.9) (beta2 := 0.999) (epsilon := 1e-8))
    (opts := opts) (sample := sample) (steps := steps)
    (cudaMemWatch := cudaMemWatch)

/--
CLI entrypoint.

Useful flags:
- `--cuda` runs the eager training loop on the CUDA runtime.
- `--steps <n>` or `--epochs <n>` controls optimization steps.
- `--x <path> --y <path>` selects custom CIFAR-style `.npy` arrays.
- `--log <path>` writes the training curve JSON.
-/
def main (args : List String) : IO UInt32 := do
  let (rt, rest) ← Common.orThrow exeName <| TorchLean.Module.ExecConfig.parseAndStrip args
  TorchLean.Module.ExecConfig.log rt
  let opts : TorchLean.Options :=
    { backend := rt.backend
      useGpu := rt.useGpu
      fastKernels := rt.fastKernels
      fastGpuMatmulPrecision := rt.fastGpuMatmulPrecision }
  IO.println s!"{exeName}: CIFAR masked reconstruction (device={if opts.useGpu then "cuda" else "cpu"})"
  let flags ← Common.orThrow exeName <|
    RealData.parseCifarLoggedTrainFlags exeName rest defaultLogJson 10
  let curve ← trainCurve opts flags.xPath flags.yPath flags.nRows flags.seed
    flags.train.steps flags.train.cudaMemWatch
  Common.printCurveLossSummary flags.train.steps curve
  Common.writeCurveLogTo flags.train.log "MAE CIFAR masked reconstruction" curve "loss"
    (RealData.cifarTrainNotes opts flags
      #[s!"x={flags.xPath}", s!"y={flags.yPath}", s!"maskPeriod={maskPeriod}",
        s!"maskOffset={maskOffset}"])
  IO.println "OK"
  pure 0

end NN.Examples.Models.Generative.Mae
