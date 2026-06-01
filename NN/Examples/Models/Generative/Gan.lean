/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Run:
  python3 scripts/datasets/download_example_data.py --cifar10
  lake exe -K cuda=true torchlean gan --cuda --steps 10
-/

module

public import NN
public import NN.API.Models.Generative
public import NN.Examples.Models.Common.RealData
public import NN.Spec.Models.Gan
public import NN.MLTheory.Generative.Latent.GAN

/-!
# GAN CIFAR Example

Compact LSGAN-style executable path.

This trains:
- a generator `z -> image` toward the current CIFAR minibatch as a stable warm-up objective;
- a discriminator on real CIFAR images (`1`) and deterministic noise images (`0`).

The formal LSGAN objective decomposition lives in `NN.Spec.Models.Gan` and
`NN.MLTheory.Generative.Latent.GAN`. A full alternating adversarial trainer can reuse the same
generator/discriminator constructors and data path.
-/

@[expose] public section

open Spec Tensor
open NN.API

namespace NN.Examples.Models.Generative.Gan

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "torchlean gan"

/-- Default JSON loss-curve path for this command. -/
def defaultLogJson : System.FilePath := Common.modelZooTrainLog "gan"

/--
Shared vector-image configuration.

The generator, discriminator, latent batch, score batch, and CIFAR vector batch all derive from this
record, so shape changes stay centralized.
-/
def cfg : nn.models.VectorGenerativeConfig := nn.models.compactImageConfig

/-- Latent-noise batch shape for the generator input. -/
abbrev Z : Shape := nn.models.vectorLatentShape cfg

/-- Flattened CIFAR image-vector batch shape. -/
abbrev X : Shape := nn.models.vectorDataShape cfg

/-- Discriminator score shape: one scalar score per batch row. -/
abbrev S : Shape := NN.Tensor.Shape.Mat cfg.batch 1

/-- Generator network mapping latent vectors to flattened image vectors. -/
def mkGenerator : nn.M (nn.Sequential Z X) :=
  nn.models.vectorGanGenerator cfg

/-- Discriminator network mapping flattened image vectors to scalar real/fake scores. -/
def mkDiscriminator : nn.M (nn.Sequential X S) :=
  nn.models.vectorGanDiscriminator cfg

/--
Train the compact LSGAN-style pair and return a total-loss curve.

The update rule is chosen for stable public runs: the generator first learns toward a fixed CIFAR
minibatch, while the discriminator separates that minibatch from deterministic noise.
The imported spec/theory modules carry the adversarial objective statements; this runtime path
checks that both networks, optimizers, CUDA memory reporting, and real-data loading work together.
-/
def trainCurve (opts : TorchLean.Options) (xPath yPath : System.FilePath)
    (nRows seed steps cudaMemWatch : Nat) : IO _root_.Runtime.Training.Curve := do
  nn.withModel mkGenerator fun gen => do
  nn.withModel mkDiscriminator fun disc => do
    let genDef := nn.mseScalarModuleDef gen
    let discDef := nn.mseScalarModuleDef disc
    let genM ← TorchLean.Module.instantiateWithOptions (α := Float) genDef id opts
    let discM ← TorchLean.Module.instantiateWithOptions (α := Float) discDef id opts
    let realX ← RealData.loadCifarVectorBatch cfg (by decide) exeName xPath yPath nRows seed
    let z := nn.models.latentNoise cfg seed
    let noiseX := nn.models.dataNoise cfg (seed + 17)
    let genSample : API.sample.Supervised Float Z X := API.sample.mk z realX
    let discReal : API.sample.Supervised Float X S := API.sample.mk realX (nn.models.onesScore cfg)
    let discFake : API.sample.Supervised Float X S := API.sample.mk noiseX (nn.models.zerosScore cfg)
    let genOpt :=
      TorchLean.Optim.adam (α := Float) (paramShapes := nn.paramShapes gen)
        (lr := 1e-3) (beta1 := 0.9) (beta2 := 0.999) (epsilon := 1e-8)
    let discOpt :=
      TorchLean.Optim.adam (α := Float) (paramShapes := nn.paramShapes disc)
        (lr := 1e-3) (beta1 := 0.9) (beta2 := 0.999) (epsilon := 1e-8)
    let genH ← TorchLean.Optim.handle (α := Float) genM genOpt
    let discH ← TorchLean.Optim.handle (α := Float) discM discOpt
    let mut curve : _root_.Runtime.Training.Curve := {}
    let g0 ← TorchLean.Module.forward (α := Float) genM genSample
    let d0r ← TorchLean.Module.forward (α := Float) discM discReal
    let d0f ← TorchLean.Module.forward (α := Float) discM discFake
    let mut last := Tensor.toScalar g0 + Tensor.toScalar d0r + Tensor.toScalar d0f
    curve := curve.push 0 last
    let watchEvery := Common.effectiveCudaMemWatch opts steps cudaMemWatch
    let mut memWatch? ← Common.reportCudaMemWatch opts watchEvery steps 0 none
    for step in [0:steps] do
      genH.step genSample
      discH.step discReal
      discH.step discFake
      memWatch? ← Common.reportCudaMemWatch opts watchEvery steps (step + 1) memWatch?
      let g ← TorchLean.Module.forward (α := Float) genM genSample
      let dr ← TorchLean.Module.forward (α := Float) discM discReal
      let df ← TorchLean.Module.forward (α := Float) discM discFake
      last := Tensor.toScalar g + Tensor.toScalar dr + Tensor.toScalar df
      curve := curve.push (step + 1) last
    IO.println s!"  steps={steps} totalLoss0={Tensor.toScalar g0 + Tensor.toScalar d0r + Tensor.toScalar d0f} totalLoss{steps}={last}"
    pure curve

/--
Executable entrypoint for the compact GAN-style run.

The command loads CIFAR vectors, trains generator and discriminator updates for `--steps`, and writes
the combined loss curve to the requested logging destination.
-/
def main (args : List String) : IO UInt32 := do
  TorchLean.Module.run exeName args
    (.float (fun opts rest => do
      let flags ← Common.orThrow exeName <|
        RealData.parseCifarLoggedTrainFlags exeName rest defaultLogJson 10
      let curve ← trainCurve opts flags.xPath flags.yPath flags.nRows flags.seed
        flags.train.steps flags.train.cudaMemWatch
      Common.writeCurveLogTo flags.train.log "GAN-style CIFAR training" curve "total_loss"
        (RealData.cifarTrainNotes opts flags #[s!"latentDim={cfg.latentDim}"])
    ))
    { banner? := some (fun opts =>
        s!"{exeName}: CIFAR LSGAN-style training (device={if opts.useGpu then "cuda" else "cpu"})")
      printOk := true }

end NN.Examples.Models.Generative.Gan
