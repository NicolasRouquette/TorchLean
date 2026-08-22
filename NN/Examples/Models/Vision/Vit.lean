/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Real-data CUDA example:
  python3 scripts/datasets/download_example_data.py --cifar10
  lake -R -K cuda=true exe torchlean vit --device cuda --n-total 1 --steps 1

This is a real-data ViT-style CIFAR-10 minibatch run:
- patch embedding via Conv2d,
- reshape + transpose to tokens,
- one Transformer encoder block,
- flatten + linear head.
-/

module


public import NN.API
public import NN.Examples.Models.Common.RealData

/-!
# ViT-Style Real-Data Example

Runnable `torchlean vit` example. It trains a compact ViT-style image classifier on a
prepared CIFAR-10 minibatch: patch embedding by convolution, token reshape, transformer block, and
linear head.

The reusable model wiring lives behind the public `TorchLean.nn.models.vit` constructor. The command
adds CIFAR loader construction and the step-limited training loop.

```bash
python3 scripts/datasets/download_example_data.py --cifar10
lake -R -K cuda=true exe torchlean vit --device cuda --n-total 1 --steps 1
```

This command is a small runtime check. Larger image-token runs belong in runtime profiling work,
not the default quick path:

```bash
lake -R -K cuda=true build
lake -R -K cuda=true exe torchlean vit --device cuda --n-total 1 --steps 1
```
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Vision.Vit

/-- CLI subcommand name used in terminal banners and parser errors. -/
def exeName : String := "torchlean vit"

/-- Default JSON loss-curve path for this command. -/
def defaultLogJson : System.FilePath := ModelZoo.trainLogPath "vit"

/--
Static minibatch size for the ViT example.

The batch axis is part of the checked model type, so changing this value changes the input and
output shapes at compile time.
-/
def batch : Nat := 1

/-- CIFAR image channels. -/
def inC : Nat := 3

/-- Height of the CIFAR crop used by this runnable ViT command. -/
def inH : Nat := 2

/-- Width of the CIFAR crop used by this runnable ViT command. -/
def inW : Nat := 2

/-- Patch height used by the convolutional patch embedding. -/
def patchH : Nat := 2

/-- Patch width used by the convolutional patch embedding. -/
def patchW : Nat := 2

/-- Patch stride; equal to patch size here, so patches do not overlap. -/
def stride : Nat := 2

/-- No zero-padding for the patch embedding. -/
def padding : Nat := 0

/--
Transformer feature width.

CIFAR rows are cropped before training. A 2×2 patch covers the whole crop here, so the command
exercises the ViT path with one image token and a small classifier head.
-/
def dModel : Nat := 1

/-- CIFAR class count, hence the output-logit width. -/
def outDim : Nat := RealData.cifarClasses

/-- Number of attention heads in the single encoder block. -/
def numHeads : Nat := 1

/-- Per-head feature width; $\mathtt{numHeads}\cdot\mathtt{headDim}=\mathtt{dModel}$. -/
def headDim : Nat := 1

/-- Feed-forward hidden width inside the encoder block. -/
def ffnHidden : Nat := 2

/-- Shared ViT configuration used by shapes and the reusable public model constructor. -/
def cfg : nn.models.VitConfig 2 :=
  { inChannels := inC
    spatial := #v[inH, inW]
    patch :=
      { outChannels := dModel
        kernel := #v[patchH, patchW]
        stride := #v[stride, stride]
        padding := #v[padding, padding]
        kernelNonzero := by intro i; fin_cases i <;> decide
        strideNonzero := by intro i; fin_cases i <;> simp [stride, Vector.get] }
    outDim := outDim
    numHeads := numHeads
    headDim := headDim
    ffnHidden := ffnHidden }

/-- Leading sample axis used by this batched training example. -/
abbrev batchShape : Shape := .dim batch .scalar

abbrev σ : Shape := cfg.inputShape batchShape

abbrev τ : Shape := cfg.outputShape batchShape

/--
Compact ViT-style classifier from the public model API.

The constructor builds patch embedding, token reshape, one encoder block, and the classifier head.
-/
def model : nn.Builder (nn.Sequential σ τ) :=
  nn.models.vit cfg batchShape
    (hInChannels := by decide)
    (hSeqLen := by
      norm_num [nn.models.VitConfig.seqLen, nn.models.VitConfig.patchSpatial, cfg,
        inH, inW, patchH, patchW, stride, padding,
        Spec.convOutSpatial, Spec.Shape.slidingWindowOutDim, Spec.Shape.ofList, Spec.Shape.size,
        Vector.get, Vector.toList, Vector.ofFn])
    (hModel := by decide)

/-- Train the CIFAR ViT with the public `Trainer` surface. -/
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
      (title := "ViT CIFAR training")
      (notes := RealData.cifarClassifierNotes batch flags))
  pure trained.report

/-- CLI entrypoint for CIFAR ViT training; CUDA is the maintained validation path. -/
def main (args : List String) : IO UInt32 :=
  TrainCommand.classificationNpy exeName args
    (fun rest => RealData.CifarModelTrainFlags.parse exeName rest defaultLogJson 1 1e-3)
    (ModelZoo.bannerWithDevice exeName "ViT CIFAR training")
    train

end NN.Examples.Models.Vision.Vit
