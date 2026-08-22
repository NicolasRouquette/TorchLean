/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Run:
  python3 scripts/datasets/download_example_data.py --cifar10
  lake -R -K cuda=true exe torchlean gan --device cuda --steps 1 --n-total 1
-/

module

public import NN.API
public import NN.Examples.Models.Common.RealData
public import NN.Spec.Models.Gan
public import NN.Spec.Layers.Loss
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

open TorchLean

namespace NN.Examples.Models.Generative.Gan

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "torchlean gan"

/-- Default JSON loss-curve path for this command. -/
def defaultLogJson : System.FilePath := ModelZoo.trainLogPath "gan"

/--
Shared generator and discriminator dimensions.
-/
def cfg : nn.models.DenseGenerative.Config :=
  { dataDim := 16, hiddenDim := 8, latentDim := 4 }

/-- Number of vectors in each training sample. -/
def batch : Nat := 1

/-- Latent-noise batch shape for the generator input. -/
abbrev Z := cfg.latentShape (.dim batch .scalar)

/-- Flattened CIFAR image-vector batch shape. -/
abbrev X := cfg.dataShape (.dim batch .scalar)

/-- Discriminator score shape: one scalar score per batch row. -/
abbrev S : Shape := cfg.scoreShape (.dim batch .scalar)

namespace Internal

/-- Deterministic example noise in `[lo, hi)`. -/
def noise (dim seed salt : Nat) (lo hi : Float) :
    Tensor Float (.dim batch (.dim dim .scalar)) :=
  .dim fun bi => .dim fun j =>
    let offset := bi.val * dim + j.val
    let raw := (seed * 1103515245 + offset * 12345 + salt) % 997
    let unit := Float.ofNat raw / 997.0
    .scalar (lo + (hi - lo) * unit)

/-- Deterministic latent input used by the compact generator run. -/
def latentNoise (seed : Nat) : Tensor Float Z :=
  noise cfg.latentDim seed 17 (-1.0) 1.0

/-- Deterministic data-shaped input used as the discriminator's negative sample. -/
def dataNoise (seed : Nat) : Tensor Float X :=
  noise cfg.dataDim seed 91 0.0 1.0

/-- Constant discriminator target. -/
def scoreTarget (value : Float) : Tensor Float S :=
  .dim fun _ => .dim fun _ => .scalar value

end Internal

/-- Generator network mapping latent vectors to flattened image vectors. -/
def mkGenerator : nn.Builder (nn.Sequential Z X) :=
  nn.models.DenseGenerative.ganGenerator cfg (.dim batch .scalar)

/-- Discriminator network mapping flattened image vectors to scalar real/fake scores. -/
def mkDiscriminator : nn.Builder (nn.Sequential X S) :=
  nn.models.DenseGenerative.ganDiscriminator cfg (.dim batch .scalar)

/-- Mean-squared error for one supervised sample evaluated through a public prediction closure. -/
def sampleMse {σ τ : Shape}
    (predict : Tensor Float σ → IO (Tensor Float τ))
    (sample : Sample.Supervised Float σ τ) : IO Float := do
  let yhat ← predict (Sample.x sample)
  pure (_root_.Spec.mseSpec yhat (Sample.y sample))

/--
Aggregate generator and discriminator scalar losses for one LSGAN reporting step.

The metric receives public prediction functions rather than raw modules.  That is the whole point of
this example after the trainer cleanup: the GAN-specific code still defines the task objective, but
the model state, optimizer stepping, and backend details stay inside `trainer.trainPairStreams`.
-/
def totalLoss
    (predictGen : Tensor Float Z → IO (Tensor Float X))
    (predictDisc : Tensor Float X → IO (Tensor Float S))
    (genSample : Sample.Supervised Float Z X)
    (discReal discFake : Sample.Supervised Float X S) : IO Float := do
  let g ← sampleMse predictGen genSample
  let dr ← sampleMse predictDisc discReal
  let df ← sampleMse predictDisc discFake
  pure (g + dr + df)

/--
Train the compact LSGAN-style pair and return a total-loss curve.

The update rule is chosen for stable public runs: the generator first learns toward a fixed CIFAR
minibatch, while the discriminator separates that minibatch from deterministic noise.
The imported spec/theory modules carry the adversarial objective statements; this runtime path
checks that both networks, optimizers, CUDA memory reporting, and real-data loading work together.
-/
def trainCurve (opts : Options) (xPath yPath : System.FilePath)
    (nRows seed steps cudaMemWatch : Nat) : IO Training.Curve := do
  let realX ← RealData.loadCifarFeatureBatch batch cfg (by decide) exeName xPath yPath nRows seed
  let z := Internal.latentNoise seed
  let noiseX := Internal.dataNoise (seed + 17)
  let genSample : Sample.Supervised Float Z X := Sample.mk z realX
  let discReal : Sample.Supervised Float X S := Sample.mk realX (Internal.scoreTarget 1.0)
  let discFake : Sample.Supervised Float X S := Sample.mk noiseX (Internal.scoreTarget 0.0)
  let genTrainer :=
    Trainer.new mkGenerator <|
      Trainer.Config.fromRunConfig
        (Trainer.RunConfig.ofRuntimeOptions opts { optimizer := optim.adam { lr := 1e-3 } })
        .regression
        (seed := seed)
  let discTrainer :=
    Trainer.new mkDiscriminator <|
      Trainer.Config.fromRunConfig
        (Trainer.RunConfig.ofRuntimeOptions opts { optimizer := optim.adam { lr := 1e-3 } })
        .regression
        (seed := seed + 1)
  genTrainer.printInfoAs "generator"
  discTrainer.printInfoAs "discriminator"
  let trained ← genTrainer.trainPairStreams discTrainer opts
    (fun _ => genSample)
    (fun _ => [discReal, discFake])
    (fun predictGen predictDisc =>
      totalLoss predictGen predictDisc genSample discReal discFake)
    { steps := steps, log := .disabled }
    (curveEvery := 1)
    (cudaMemWatch := cudaMemWatch)
  trained.printCurveSummary "totalLoss"
  pure trained.curve

/--
Executable entrypoint for the compact GAN-style run.

The command loads CIFAR vectors, trains generator and discriminator updates for `--steps`, and writes
the combined loss curve to the requested logging destination.
-/
def main (args : List String) : IO UInt32 := do
  RealData.cifarCurve exeName args defaultLogJson 10
    (banner := ModelZoo.bannerWithDevice exeName "CIFAR LSGAN-style training")
    (seriesName := "total_loss")
    (title := "GAN-style CIFAR training")
    (extraNotes := fun _ _ => #[s!"latentDim={cfg.latentDim}"])
    (train := fun opts flags =>
      trainCurve opts flags.xPath flags.yPath flags.nRows flags.seed flags.steps flags.cudaMemWatch)

end NN.Examples.Models.Generative.Gan
