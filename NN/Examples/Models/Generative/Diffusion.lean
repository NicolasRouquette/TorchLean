/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.Examples.Models.Common.RealData

/-!
# Diffusion Training Example

Runnable `torchlean diffusion` example.

This is the maintained diffusion command. It supports two real-data modes:

- `--dataset imagenet64` (default): user-provided ImageNet/Imagenette/Tiny-ImageNet-style images
  converted to `(N,3,64,64)` `.npy` tensors.
- `--dataset cifar10`: prepared CIFAR-10 `(N,3,32,32)` arrays.

The command is one public entrypoint, but the implementation keeps separate typed branches because
Lean tracks image height and width in the tensor type.

## Why unconditional samples are still modest

The default epsilon predictor is a compact same-resolution residual CNN with a broadcast time channel.
That is enough to validate real image loading, CUDA training, logging, reconstruction diagnostics,
and DDIM replay from Lean. High-fidelity unconditional samples require more machinery: a full U-Net
with multiscale skips, richer timestep embeddings, EMA, more training, more timesteps, and runtime
support that avoids eager-autograd buffer blow-up for wider models.

## Examples

Prepare ImageNet-style data:

```bash
python3 scripts/datasets/torchlean_data_convert.py image-folder \
  --input /path/to/imagenet/train \
  --x-output data/real/imagenet64/imagenet64_train_X.npy \
  --y-output data/real/imagenet64/imagenet64_train_y.npy \
  --height 64 --width 64 --labels-from-dirs --limit 800
```

Train on ImageNet64 and save visual artifacts:

```bash
lake -R -K cuda=true build
CUDA_VISIBLE_DEVICES=0 lake -R -K cuda=true exe torchlean diffusion --device cuda \
  --dataset imagenet64 --n-total 800 --steps 1000 --hidden-c 8 --T 100 --beta-end 0.12 \
  --log data/model_zoo/diffusion_trainlog.json \
  --reference-ppm data/model_zoo/diffusion_reference.ppm \
  --noisy-ppm data/model_zoo/diffusion_noisy.ppm \
  --reconstruct-ppm data/model_zoo/diffusion_reconstruct.ppm \
  --sample-ppm data/model_zoo/diffusion_sample.ppm
```

CIFAR run:

```bash
python3 scripts/datasets/download_example_data.py --cifar10
lake -R -K cuda=true exe torchlean diffusion --device cuda --dataset cifar10 --n-total 1 --steps 1 --hidden-c 1 --T 2
```
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Generative.Diffusion

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "torchlean diffusion"

/-- Default JSON loss-curve path for this command. -/
def defaultLogJson : System.FilePath := ModelZoo.trainLogPath "diffusion"

/-- Static minibatch size used by both CIFAR-10 and ImageNet64 typed branches. -/
def batch : Nat := 1

/-- Cropped CIFAR height for the compact runnable diffusion example. -/
def cifarTinyH : Nat := 2

/-- Cropped CIFAR width for the compact runnable diffusion example. -/
def cifarTinyW : Nat := 2

local instance : NeZero cifarTinyH := ⟨by decide⟩
local instance : NeZero cifarTinyW := ⟨by decide⟩

/-- Clean image batch shape $x_0$: NCHW with the fixed command batch size. -/
abbrev cleanImageShape (c h w : Nat) : Shape :=
  .dim batch (.dim c (.dim h (.dim w .scalar)))

/-- Epsilon-model input shape: image channels plus one broadcast timestep channel. -/
abbrev noisyInputShape (c h w : Nat) : Shape :=
  .dim batch (.dim (c + 1) (.dim h (.dim w .scalar)))

/-- Shape-level configuration for the epsilon predictor. -/
def cfgFor (c h w hiddenC : Nat) [NeZero h] [NeZero w] : nn.models.EpsConvNetConfig 2 :=
  { batch := batch
    dataChannels := c
    spatial := #v[h, w]
    spatialNonzero := by
      intro i
      fin_cases i
      · change h ≠ 0
        exact NeZero.ne h
      · change w ≠ 0
        exact NeZero.ne w
    hiddenChannels := hiddenC }

/--
Build the default epsilon predictor for a specific typed image shape.

We use the plain compact epsilon CNN from the public diffusion model API. The residual denoiser stays
available in the API for larger opt-in experiments, but the runnable command should remain a quick
CUDA quick check.
-/
def mkModel (c h w hiddenC : Nat)
    [NeZero c] [NeZero h] [NeZero w] (h_hiddenC : hiddenC ≠ 0) :
    nn.M (nn.Sequential (noisyInputShape c h w) (cleanImageShape c h w)) :=
  nn.models.epsConvNet (cfgFor c h w hiddenC)
    (h_batch := by simp [cfgFor, batch])
    (h_dataC := NeZero.ne c)
    (h_inC := by exact Nat.succ_ne_zero c)
    (h_hiddenC := h_hiddenC)

/--
Convert one typed CIFAR minibatch into diffusion-space clean images.

The loader returns images in $[0,1]$; diffusion training uses $[-1,1]$, so this function performs the
range conversion after Lean has established the CIFAR NCHW shape.
-/
def cifarCleanImageBatch
    (batchSample : Sample.Batch Float batch RealData.CifarImage RealData.CifarTarget) :
    Tensor.T Float (cleanImageShape RealData.cifarChannels cifarTinyH cifarTinyW) := by
  let cropped := RealData.cropCifarBatch batch cifarTinyH cifarTinyW
    (by decide) (by decide) batchSample
  let unitImage : Tensor.T Float (cleanImageShape RealData.cifarChannels cifarTinyH cifarTinyW) := by
    simpa [cleanImageShape] using (Sample.x cropped)
  exact diffusion.toMinusOneOne unitImage

/--
Convert one typed ImageNet64 minibatch into diffusion-space clean images.

This mirrors `cifarCleanImageBatch` but keeps the ImageNet64 height/width/channel constants in the type.
-/
def imageNet64CleanImageBatch
    (batchSample : Sample.Batch Float batch RealData.ImageNet64Image RealData.ImageNet64Target) :
    Tensor.T Float (cleanImageShape RealData.imagenet64Channels RealData.imagenet64Height
      RealData.imagenet64Width) := by
  let unitImage : Tensor.T Float (cleanImageShape RealData.imagenet64Channels RealData.imagenet64Height
      RealData.imagenet64Width) := by
    simpa [cleanImageShape, RealData.ImageNet64Image] using
      (Sample.x batchSample)
  exact diffusion.toMinusOneOne unitImage

/--
Load CIFAR-10 batches as a finite list of clean diffusion images.

The function validates the `.npy` paths, builds a typed `Data.batchLoader`, drops incomplete final
batches, and returns NCHW tensors already mapped into $[-1,1]$.
-/
def loadCifarCleanImageBatches (xPath yPath : System.FilePath) (nRows seed : Nat) :
    IO (List (Tensor.T Float (cleanImageShape RealData.cifarChannels cifarTinyH cifarTinyW))) := do
  let batches ← RealData.loadCifarBatches exeName batch nRows seed xPath yPath
  pure (batches.map cifarCleanImageBatch)

/--
Load ImageNet64-style batches as a finite list of clean diffusion images.

The converter accepts ImageNet/Imagenette/Tiny-ImageNet-style folders ahead of time; this Lean path
only consumes the prepared `.npy` arrays and keeps the tensor shapes explicit.
-/
def loadImageNet64CleanImageBatches (xPath yPath : System.FilePath) (nRows seed : Nat) :
    IO (List (Tensor.T Float (cleanImageShape RealData.imagenet64Channels RealData.imagenet64Height
      RealData.imagenet64Width))) := do
  let batches ← RealData.loadImageNet64Batches exeName batch nRows seed xPath yPath
  pure (batches.map imageNet64CleanImageBatch)

/--
Run deterministic DDIM reverse steps from a starting noisy image.

This is used for unconditional sample artifacts: start from Gaussian noise, repeatedly ask the model
for $\hat{\varepsilon}$, and apply the DDIM previous-step formula.
-/
def reverseDdim {c h w : Nat}
    (predict : Tensor.T Float (noisyInputShape c h w) → IO (Tensor.T Float (cleanImageShape c h w)))
    (alphaBars : Array Float) (T : Nat) (xStart : Tensor.T Float (cleanImageShape c h w)) :
    IO (Tensor.T Float (cleanImageShape c h w)) := do
  let mut x_t := xStart
  if T > 1 then
    for tRev in [0:T] do
      let tIdx : Nat := (T - 1) - tRev
      let ab : Float := alphaBars.getD tIdx 1.0
      let abPrev : Float := if tIdx = 0 then 1.0 else alphaBars.getD (tIdx - 1) 1.0
      let tNorm : Float := Float.ofNat tIdx / Float.ofNat (T - 1)
      let epsHat ← predict <|
        diffusion.appendTimeChannel (.dim batch .scalar) #v[h, w] x_t tNorm
      x_t := diffusion.ddimPrev abPrev ab x_t epsHat
  pure x_t

/--
Reverse DDIM from a chosen timestep for reconstruction diagnostics.

This reconstruction path is separate from unconditional sampling. It corrupts a real image to a
moderate timestep, denoises from there, and checks whether reconstruction improves over the noisy
input.
-/
def reverseDdimFrom {c h w : Nat}
    (predict : Tensor.T Float (noisyInputShape c h w) → IO (Tensor.T Float (cleanImageShape c h w)))
    (alphaBars : Array Float) (T tStart : Nat) (xStart : Tensor.T Float (cleanImageShape c h w)) :
    IO (Tensor.T Float (cleanImageShape c h w)) := do
  let mut x_t := xStart
  for tRev in [0:tStart + 1] do
    let tIdx : Nat := tStart - tRev
    let ab : Float := alphaBars.getD tIdx 1.0
    let abPrev : Float := if tIdx = 0 then 1.0 else alphaBars.getD (tIdx - 1) 1.0
    let tNorm : Float := if T <= 1 then 0.0 else Float.ofNat tIdx / Float.ofNat (T - 1)
    let epsHat ← predict <|
      diffusion.appendTimeChannel (.dim batch .scalar) #v[h, w] x_t tNorm
    x_t := diffusion.ddimPrev abPrev ab x_t epsHat
  pure x_t

/--
Diffusion command-line options after parsing.

The inherited pieces make the CLI shape explicit: ordinary training flags come from `ModelZoo`,
diffusion math lives in `ModelZoo.DiffusionScheduleFlags`, visual outputs live in
`ModelZoo.ImageArtifactFlags`, and the epsilon-network width is the model-specific knob.
-/
structure DiffusionOptions extends
    CLI.Training.OptimizerOptions,
    ModelZoo.DiffusionScheduleFlags,
    ModelZoo.ImageArtifactFlags where
  /-- Hidden channel width of the epsilon predictor. -/
  hiddenC : Nat
deriving Repr

/--
Shared training loop for both CIFAR-10 and ImageNet64 branches.

The loop optimizes epsilon prediction and can emit four visual artifacts:

- `reference-ppm`: clean evaluation image,
- `noisy-ppm`: clean image after forward diffusion to `reconstruct-step`,
- `reconstruct-ppm`: DDIM denoising from that timestep,
- `sample-ppm`: unconditional DDIM sample from Gaussian noise.
-/
def trainCurveFloat {c h w : Nat} [NeZero c] [NeZero h] [NeZero w]
    (opts : Options)
    (loadBatches : IO (List (Tensor.T Float (cleanImageShape c h w))))
    (cfg : DiffusionOptions) (h_hiddenC : cfg.hiddenC ≠ 0) :
    IO Training.Curve := do
  let batches ← loadBatches
  let batchAt ← ModelZoo.orThrow exeName <|
    Data.cycleListOrError batches s!"{exeName}: no training minibatches available"
  let evalX0 := batchAt 0
  let alphaBars := diffusion.alphaBarsLinear cfg.T cfg.betaStart cfg.betaEnd
  let evalStep := if cfg.T = 0 then 0 else cfg.T / 2
  let evalSample := diffusion.noisedSample (.dim batch .scalar) #v[h, w]
    alphaBars cfg.T evalX0 (seed := opts.seed)
    (step := evalStep)
  let trainer :=
    Trainer.new (mkModel c h w cfg.hiddenC h_hiddenC) <|
      Trainer.Config.fromRunConfig
        (Trainer.runConfig opts { optimizer := optim.adam { lr := cfg.lr } })
        .regression
        (seed := opts.seed)
  trainer.printInfo
  let curveEvery : Nat := Nat.max 1 (cfg.steps / 50)
  let trained ← trainer.trainStreamFloat opts
    (fun step =>
      let x0 := batchAt step
      diffusion.noisedSample (.dim batch .scalar) #v[h, w]
        alphaBars cfg.T x0 (seed := opts.seed) (step := step + 1))
    evalSample
    { steps := cfg.steps, log := .disabled }
    (curveEvery := curveEvery)
    (cudaMemWatch := cfg.cudaMemWatch)
  let curve := trained.curve
  trained.printSummary
    match cfg.referencePpm? with
    | none => pure ()
    | some path => diffusion.writeFirstRgbPpm path evalX0
    match cfg.samplePpm? with
    | none => pure ()
    | some path => do
        let x_T := diffusion.randomEps (s := cleanImageShape c h w)
          (seed := opts.seed) (step := 999)
        let x_t ← reverseDdim trained.predict alphaBars cfg.T x_T
        diffusion.writeFirstRgbPpm path x_t
    match cfg.reconstructPpm? with
    | none => pure ()
    | some path => do
        let tIdx : Nat := if cfg.T = 0 then 0 else Nat.min (cfg.reconstructStep?.getD (cfg.T / 4)) (cfg.T - 1)
        let ab : Float := alphaBars.getD tIdx 1.0
        let sqrtAb : Float := MathFunctions.sqrt (Max.max ab 0.0)
        let sqrtOneMinusAb : Float := MathFunctions.sqrt (Max.max (1.0 - ab) 0.0)
        let eps := diffusion.randomEps (s := cleanImageShape c h w)
          (seed := opts.seed) (step := 1001)
        let noisyImage : Tensor.T Float (cleanImageShape c h w) :=
          Spec.Tensor.addSpec
            (Spec.Tensor.scaleSpec evalX0 sqrtAb)
            (Spec.Tensor.scaleSpec eps sqrtOneMinusAb)
        match cfg.noisyPpm? with
        | none => pure ()
        | some noisyPath => diffusion.writeFirstRgbPpm noisyPath noisyImage
        let x_t ← reverseDdimFrom trained.predict alphaBars cfg.T tIdx noisyImage
        diffusion.writeFirstRgbPpm path x_t
  pure curve

namespace DiffusionOptions

/--
Parse diffusion-specific training flags after runtime/device flags and dataset flags.

The shared parser handles `--steps`, `--log`, and `--cuda-mem-watch`; this parser handles diffusion
schedule parameters, model width, and optional PPM artifact paths.
-/
def parse (args : List String) :
    Except String (DiffusionOptions × List String) := do
  let (train, rest) ← CLI.Training.OptimizerOptions.parse exeName args defaultLogJson 50 1e-3
  let (hiddenC, rest) ← CLI.takeNatFlagDefault rest "hidden-c" 16
  let (schedule, rest) ← ModelZoo.DiffusionScheduleFlags.parse rest
  let (artifacts, rest) ← ModelZoo.ImageArtifactFlags.parse rest
  pure ({ toOptimizerOptions := train,
          toDiffusionScheduleFlags := schedule,
          toImageArtifactFlags := artifacts,
          hiddenC := hiddenC },
        rest)

/-- Reject unsupported diffusion hyperparameters before shape-specialized execution begins. -/
def ensureValid (cfg : DiffusionOptions) : IO Unit := do
  if cfg.hiddenC = 0 then
    throw <| IO.userError s!"{exeName}: --hidden-c must be > 0"

/-- Dataset/source note fields shared by the CIFAR-10 and ImageNet64 branches. -/
def sourceNotes
    (datasetName : String)
    (data : ModelZoo.NpyDataFlags) : Array String :=
  ModelZoo.NpyDataFlags.trainLogNotes data datasetName

/-- TrainLog note fields shared by all diffusion dataset branches. -/
def logNotes
    (cfg : DiffusionOptions)
    (dataset : String)
    (opts : Options)
    (sourceNotes : Array String := #[]) : Array String :=
  sourceNotes ++
    #[s!"dataset={dataset}",
      ModelZoo.deviceNote opts,
      s!"lr={cfg.lr}",
      s!"hiddenC={cfg.hiddenC}"] ++
    ModelZoo.DiffusionScheduleFlags.trainLogNotes cfg.toDiffusionScheduleFlags ++
    ModelZoo.ImageArtifactFlags.trainLogNotes cfg.toImageArtifactFlags

end DiffusionOptions

/-- Write the diffusion loss curve plus dataset, schedule, model, and artifact metadata. -/
def writeTrainingLog (log : Training.LogDestination) (dataset : String)
    (sourceNotes : Array String) (cfg : DiffusionOptions) (opts : Options)
    (curve : Training.Curve) : IO Unit :=
  ModelZoo.writeCurveTrainLog log "Diffusion training" curve "loss"
    (notes := cfg.logNotes dataset opts sourceNotes)

/--
Run one typed diffusion dataset branch.

The CIFAR-10 and ImageNet64 commands differ in their shape-level loader and default `.npy` paths,
but after parsing those inputs they follow the same command flow: parse training flags, reject
unused args, require $\mathtt{hiddenC}>0$, train the epsilon predictor, then write the same curve
log.
-/
def runTypedDataset {c h w : Nat} [NeZero c] [NeZero h] [NeZero w]
    (opts : Options) (args : List String)
    (datasetName : String)
    (parseData : List String → Except String (ModelZoo.NpyDataFlags × List String))
    (loadBatches : System.FilePath → System.FilePath → Nat → Nat →
      IO (List (Tensor.T Float (cleanImageShape c h w)))) : IO Unit := do
  let (data, args) ← ModelZoo.orThrow exeName <| parseData args
  let (cfg, rest) ← ModelZoo.orThrow exeName <| DiffusionOptions.parse args
  CLI.requireNoArgs exeName rest
  cfg.ensureValid
  let sourceNotes := DiffusionOptions.sourceNotes datasetName data
  let load := loadBatches data.xPath data.yPath data.nRows data.seed
  match cfg.hiddenC with
  | 0 => unreachable!
  | hc + 1 =>
      let cfg := { cfg with hiddenC := hc + 1 }
      let curve ← trainCurveFloat opts load cfg (Nat.succ_ne_zero hc)
      writeTrainingLog cfg.log datasetName sourceNotes cfg opts curve

/-- Run the ImageNet64 branch with shape-specialized model construction. -/
def runImageNet64 (opts : Options) (args : List String) : IO Unit :=
  runTypedDataset
    (c := RealData.imagenet64Channels)
    (h := RealData.imagenet64Height)
    (w := RealData.imagenet64Width)
    opts args "imagenet64"
    RealData.NpyDatasets.parseImageNet64
    loadImageNet64CleanImageBatches

/-- Run the CIFAR-10 branch with shape-specialized model construction. -/
def runCifar10 (opts : Options) (args : List String) : IO Unit :=
  runTypedDataset
    (c := RealData.cifarChannels)
    (h := cifarTinyH)
    (w := cifarTinyW)
    opts args "cifar10"
    RealData.NpyDatasets.parseCifar
    loadCifarCleanImageBatches

/--
Executable entrypoint for diffusion training.

The runtime parser selects CPU/CUDA and eager/compiled settings first; the remaining arguments select
the dataset branch and diffusion training configuration.
-/
def main (args : List String) : IO UInt32 := do
  Runtime.runFloat exeName args
    (banner := ModelZoo.bannerWithDevice exeName "diffusion trainer")
    (k := fun opts rest => do
      let (choice, rest) ← ModelZoo.orThrow exeName <| ModelZoo.ImageDatasetChoice.parse rest
      match choice with
      | .imagenet64 => runImageNet64 opts rest
      | .cifar10 => runCifar10 opts rest)

end NN.Examples.Models.Generative.Diffusion
