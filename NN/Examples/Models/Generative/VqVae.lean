/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Run:
  python3 scripts/datasets/download_example_data.py --cifar10
  lake exe -K cuda=true torchlean vqvae --cuda --steps 10
-/

module

public import NN
public import NN.API.Models.Generative
public import NN.API.Models.TrainFixed
public import NN.Examples.Models.Common.RealData
public import NN.Spec.Models.VqVae
public import NN.MLTheory.Generative.Latent.VQVAE

/-!
# VQ-VAE-Style CIFAR Example

Trains a compact vector reconstruction model with a narrow `tanh` bottleneck, paired with the
VQ-VAE spec/theory modules. The theorem-facing codebook objective lives in `NN.Spec.Models.VqVae`;
this runtime example is the executable reconstruction path.
-/

@[expose] public section

open Spec Tensor
open NN.API

namespace NN.Examples.Models.Generative.VqVae

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "torchlean vqvae"

/-- Default JSON loss-curve path for this command. -/
def defaultLogJson : System.FilePath := "data/model_zoo/vqvae_trainlog.json"

/--
Shared vector-image configuration.

The VQ-VAE runtime path uses the same compact flattened-CIFAR boundary as the autoencoder and VAE
commands, so the model comparison changes the bottleneck while keeping data handling fixed.
-/
def cfg : nn.models.VectorGenerativeConfig := nn.models.compactImageConfig

/-- Input shape: a batch of flattened CIFAR image vectors. -/
abbrev σ : Shape := nn.models.vectorDataShape cfg

/-- Target shape: reconstructed flattened CIFAR image vectors. -/
abbrev τ : Shape := nn.models.vectorDataShape cfg

/--
Trainable VQ-VAE-style vector model.

The codebook-facing objective is handled in the imported spec/theory modules; this command exercises
the executable reconstruction path with a narrow quantization-style bottleneck.
-/
def mkModel : nn.M (nn.Sequential σ τ) :=
  nn.models.vectorVqVae cfg

/--
Executable entrypoint for the compact VQ-VAE-style run.

The command loads a real CIFAR minibatch, trains the reconstruction objective, and records the same
loss-curve artifact format as the other generative commands.
-/
def main (args : List String) : IO UInt32 := do
  TorchLean.Module.run exeName args
    (.float (fun opts rest => do
      let (xPath, yPath, nRows, seed, rest) ← Common.orThrow exeName <| RealData.parseCifarFlags rest
      let (train, rest) ← Common.orThrow exeName <|
        Common.parseLoggedTrainFlags exeName rest defaultLogJson 10
      Common.orThrow exeName <| CLI.requireNoArgs rest
      let x ← RealData.loadCifarVectorBatch cfg (by decide) exeName xPath yPath nRows seed
      let sample := nn.models.reconstructionSample cfg x
      let curve ←
        _root_.NN.API.Models.TrainFixed.curveFloat
          (mkModel := mkModel)
          (mkModuleDef := fun model => nn.mseScalarModuleDef model)
          (mkOptim := fun ps =>
            TorchLean.Optim.adam (α := Float) (paramShapes := ps)
              (lr := 1e-3) (beta1 := 0.9) (beta2 := 0.999) (epsilon := 1e-8))
          (opts := opts) (sample := sample) (steps := train.steps)
          (cudaMemWatch := train.cudaMemWatch)
      let loss0 := curve.values.getD 0 0.0
      let lossN := curve.values.getD (curve.values.size - 1) loss0
      IO.println s!"  steps={train.steps} loss0={loss0} loss{train.steps}={lossN}"
      Common.writeCurveLogTo train.log "VQ-VAE-style CIFAR reconstruction" curve "loss"
        #[s!"data=cifar10", s!"latentDim={cfg.latentDim}", s!"nRows={nRows}",
          s!"device={if opts.useGpu then "cuda" else "cpu"}",
          s!"cuda_mem_watch={Common.effectiveCudaMemWatch opts train.steps train.cudaMemWatch}"]
    ))
    { banner? := some (fun opts =>
        s!"{exeName}: CIFAR VQ-VAE-style training (device={if opts.useGpu then "cuda" else "cpu"})")
      printOk := true }

end NN.Examples.Models.Generative.VqVae
