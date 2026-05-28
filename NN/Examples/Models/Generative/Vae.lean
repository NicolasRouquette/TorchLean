/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Run:
  python3 scripts/datasets/download_example_data.py --cifar10
  lake exe -K cuda=true torchlean vae --cuda --steps 10
-/

module

public import NN
public import NN.API.Models.Generative
public import NN.API.Models.TrainFixed
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

open Spec Tensor
open NN.API

namespace NN.Examples.Models.Generative.Vae

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "torchlean vae"

/-- Default JSON loss-curve path for this command. -/
def defaultLogJson : System.FilePath := "data/model_zoo/vae_trainlog.json"

/--
Shared vector-image configuration.

The runtime example uses the same flattened CIFAR data boundary as the other vector generative
commands, while the VAE-specific output shape adds latent mean/log-variance proxy channels.
-/
def cfg : nn.models.VectorGenerativeConfig := nn.models.compactImageConfig

/-- Input shape: a batch of flattened CIFAR image vectors. -/
abbrev σ : Shape := nn.models.vectorDataShape cfg

/-- Output shape: reconstruction plus latent regularization proxy channels. -/
abbrev τ : Shape := nn.models.vectorVaeOutShape cfg

/--
Trainable VAE-style vector model.

The executable target is still an MSE-style supervised sample; the imported spec/theory files state
the theorem-facing VAE objective separately.
-/
def mkModel : nn.M (nn.Sequential σ τ) :=
  nn.models.vectorVae cfg

/--
Executable entrypoint for the compact VAE-style run.

The command loads CIFAR vectors, constructs the reconstruction/latent-proxy target, trains with
Adam, and writes a standard loss curve.
-/
def main (args : List String) : IO UInt32 := do
  TorchLean.Module.run exeName args
    (.float (fun opts rest => do
      let (xPath, yPath, nRows, seed, rest) ← Common.orThrow exeName <| RealData.parseCifarFlags rest
      let (train, rest) ← Common.orThrow exeName <|
        Common.parseLoggedTrainFlags exeName rest defaultLogJson 10
      Common.orThrow exeName <| CLI.requireNoArgs rest
      let x ← RealData.loadCifarVectorBatch cfg (by decide) exeName xPath yPath nRows seed
      let sample := nn.models.vaeSample cfg x
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
      Common.writeCurveLogTo train.log "VAE-style CIFAR reconstruction" curve "loss"
        #[s!"data=cifar10", s!"latentDim={cfg.latentDim}", s!"nRows={nRows}",
          s!"device={if opts.useGpu then "cuda" else "cpu"}",
          s!"cuda_mem_watch={Common.effectiveCudaMemWatch opts train.steps train.cudaMemWatch}"]
    ))
    { banner? := some (fun opts =>
        s!"{exeName}: CIFAR beta-VAE-style training (device={if opts.useGpu then "cuda" else "cpu"})")
      printOk := true }

end NN.Examples.Models.Generative.Vae
