/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public
public import NN.Spec.Core.TensorReductionShape.ConcatSlice

/-!
# Diffusion Model Helpers (API)

Config-style diffusion model constructors plus reusable, dataset-independent DDPM/DDIM helpers.

The runnable examples decide where data comes from (CIFAR-10, ImageNet-style folders, synthetic
artifacts).  The definitions here are shape-parametric and can be reused by tests, examples, and
future proof layer specifications.
-/

@[expose] public section

namespace NN
namespace API

open Spec Tensor

namespace nn
namespace models

/-- Configuration for a minimal epsilon-predictor conv net. -/
structure EpsConvNetConfig (d : Nat) where
  batch : Nat
  dataChannels : Nat
  spatial : Vector Nat d
  spatialNonzero : ∀ i : Fin d, spatial.get i ≠ 0
  /-- Hidden channel width. -/
  hiddenChannels : Nat := 32

/-- Epsilon-predictor input shape, with one extra channel carrying the diffusion time. -/
def epsConvNetInShape {d : Nat} (cfg : EpsConvNetConfig d) : Spec.Shape :=
  .dim cfg.batch (Spec.Shape.ofList ((cfg.dataChannels + 1) :: cfg.spatial.toList))

/-- Epsilon-predictor output shape matching the denoised data channels. -/
def epsConvNetOutShape {d : Nat} (cfg : EpsConvNetConfig d) : Spec.Shape :=
  .dim cfg.batch (Spec.Shape.ofList (cfg.dataChannels :: cfg.spatial.toList))

/-- Seeded shape-preserving convolution over an arbitrary spatial rank. -/
def EpsConvNetConfig.sameConv {d : Nat} (cfg : EpsConvNetConfig d)
    (inChannels outChannels : Nat) [NeZero inChannels] :
    nn.M (nn.Sequential
      (.dim cfg.batch (Spec.Shape.ofList (inChannels :: cfg.spatial.toList)))
      (.dim cfg.batch (Spec.Shape.ofList (outChannels :: cfg.spatial.toList)))) :=
  let layer := nn.conv (leading := .dim cfg.batch .scalar)
      (inChannels := inChannels) cfg.spatial
      { outChannels := outChannels
        kernel := Vector.replicate d 1
        stride := Vector.replicate d 1
        padding := Vector.replicate d 0
        kernelNonzero := by intro i; simp [Vector.get]
        strideNonzero := by intro i; simp [Vector.get] }
  by
    simpa [Spec.Shape.concat, Spec.convOutSpatial_unit cfg.spatial cfg.spatialNonzero] using layer

/--
Build a minimal epsilon-predictor conv net:
`conv -> relu -> conv -> relu -> conv -> relu -> conv`.

This stays compact enough for the eager CUDA example while giving the CIFAR trainer more denoising
capacity than a bare two-layer network.
-/
def epsConvNet {d : Nat} (cfg : EpsConvNetConfig d)
    (h_batch : cfg.batch ≠ 0 := by decide)
    (h_dataC : cfg.dataChannels ≠ 0 := by decide)
    (h_inC : (cfg.dataChannels + 1) ≠ 0 := by decide)
    (h_hiddenC : cfg.hiddenChannels ≠ 0 := by decide) :
    nn.M (nn.Sequential (epsConvNetInShape cfg) (epsConvNetOutShape cfg)) :=
  letI : NeZero cfg.batch := ⟨h_batch⟩
  letI : NeZero cfg.dataChannels := ⟨h_dataC⟩
  letI : NeZero (cfg.dataChannels + 1) := ⟨h_inC⟩
  letI : NeZero cfg.hiddenChannels := ⟨h_hiddenC⟩
  nn.Sequential![
    cfg.sameConv (cfg.dataChannels + 1) cfg.hiddenChannels,
    relu,
    cfg.sameConv cfg.hiddenChannels cfg.hiddenChannels,
    relu,
    cfg.sameConv cfg.hiddenChannels cfg.hiddenChannels,
    relu,
    cfg.sameConv cfg.hiddenChannels cfg.dataChannels
  ]

/--
Build a stronger same-resolution residual epsilon predictor.

Architecture:

`stem conv -> relu -> residual block -> relu -> residual block -> relu -> output conv`

Each residual block has shape `hiddenC×H×W -> hiddenC×H×W` and computes
$x+\operatorname{conv}(\operatorname{relu}(\operatorname{conv}(x)))$.  This compact residual
denoiser omits U-Net downsampling, upsampling,
and multi-scale skip concatenation. It is still a useful compact architecture because
residual paths make the denoising problem much easier than a plain conv chain while staying within
the eager CUDA memory envelope used by examples.
-/
def epsResidualConvNet {d : Nat} (cfg : EpsConvNetConfig d)
    (h_batch : cfg.batch ≠ 0 := by decide)
    (h_dataC : cfg.dataChannels ≠ 0 := by decide)
    (h_inC : (cfg.dataChannels + 1) ≠ 0 := by decide)
    (h_hiddenC : cfg.hiddenChannels ≠ 0 := by decide) :
    nn.M (nn.Sequential (epsConvNetInShape cfg) (epsConvNetOutShape cfg)) :=
  letI : NeZero cfg.batch := ⟨h_batch⟩
  letI : NeZero cfg.dataChannels := ⟨h_dataC⟩
  letI : NeZero (cfg.dataChannels + 1) := ⟨h_inC⟩
  letI : NeZero cfg.hiddenChannels := ⟨h_hiddenC⟩
  nn.Sequential![
    cfg.sameConv (cfg.dataChannels + 1) cfg.hiddenChannels,
    relu,
    (do
      let block ←
        nn.Sequential![
          cfg.sameConv cfg.hiddenChannels cfg.hiddenChannels,
          relu,
          cfg.sameConv cfg.hiddenChannels cfg.hiddenChannels
        ]
      pure (nn.blocks.residual block)),
    relu,
    (do
      let block ←
        nn.Sequential![
          cfg.sameConv cfg.hiddenChannels cfg.hiddenChannels,
          relu,
          cfg.sameConv cfg.hiddenChannels cfg.hiddenChannels
        ]
      pure (nn.blocks.residual block)),
    relu,
    cfg.sameConv cfg.hiddenChannels cfg.dataChannels
  ]

end models
end nn

namespace diffusion

/-- Map a tensor from $[0,1]$ into the standard diffusion training range $[-1,1]$. -/
def toMinusOneOne {s : Spec.Shape} (x01 : Spec.Tensor Float s) : Spec.Tensor Float s :=
  Spec.Tensor.mapSpec (fun x => 2.0 * x - 1.0) x01

/--
Deterministic Gaussian epsilon tensor for an arbitrary diffusion shape.

The `(seed, step)` pair is turned into the runtime RNG key, so examples and artifact generation can
reproduce the same noising path without ambient randomness.
-/
def randomEps {s : Spec.Shape} (seed step : Nat) : Spec.Tensor Float s :=
  let key : UInt64 := _root_.Runtime.Autograd.TorchLean.Random.keyOf (seed := seed) (counter := step)
  _root_.Runtime.Autograd.TorchLean.Random.normal (α := Float) key (s := s)

/-- linear beta schedule value at timestep `t`. -/
def linearBeta (T : Nat) (betaStart betaEnd : Float) (t : Nat) : Float :=
  if T <= 1 then
    betaEnd
  else
    let u := Float.ofNat t / Float.ofNat (T - 1)
    betaStart + u * (betaEnd - betaStart)

/--
Compute the cumulative products
$\bar\alpha_t=\prod_{s\le t}(1-\beta_s)$ for a linear beta schedule.

These values connect clean data $x_0$, noised data $x_t$, and the epsilon target used by DDPM-style
training.
-/
def alphaBarsLinear (T : Nat) (betaStart betaEnd : Float) : Array Float :=
  Id.run do
    let mut a : Float := 1.0
    let mut out : Array Float := Array.mkEmpty T
    for t in [0:T] do
      let beta := linearBeta T betaStart betaEnd t
      let alpha := 1.0 - beta
      a := a * alpha
      out := out.push a
    return out

/--
Append a constant time channel after arbitrary leading axes.

The input layout is `(leading..., channels, spatial...)`.  The result preserves every leading and
spatial axis and changes only the channel count from `c` to `c + 1`.
-/
def appendTimeChannel (leading : Spec.Shape) {d c : Nat} (spatial : Vector Nat d)
    (x : Spec.Tensor Float
      (leading.concat (Spec.Shape.ofList (c :: spatial.toList)))) (tNorm : Float) :
    Spec.Tensor Float
      (leading.concat (Spec.Shape.ofList ((c + 1) :: spatial.toList))) :=
  match leading with
  | .scalar =>
      Spec.Tensor.concatLeadingAxisSpec x <|
        Tensor.dim fun _ => Spec.fill tNorm (Spec.Shape.ofList spatial.toList)
  | .dim _ rest =>
      match x with
      | .dim values =>
          Tensor.dim fun i => appendTimeChannel rest spatial (values i) tNorm

/--
Build an epsilon-prediction training sample from explicit noise.

The caller supplies `eps`, usually from the runtime RNG.  Keeping randomness outside this helper
makes the transformation reusable:

$x_t=\sqrt{\bar{\alpha}_t}\,x_0+\sqrt{1-\bar{\alpha}_t}\,\varepsilon$, with target
$\varepsilon$.
-/
def noisedSampleFromEps (leading : Spec.Shape) {d c : Nat} (spatial : Vector Nat d)
    (alphaBars : Array Float) (T : Nat)
    (x0 eps : Spec.Tensor Float
      (leading.concat (Spec.Shape.ofList (c :: spatial.toList)))) (step : Nat) :
    TorchLean.Sample.Supervised Float
      (leading.concat (Spec.Shape.ofList ((c + 1) :: spatial.toList)))
      (leading.concat (Spec.Shape.ofList (c :: spatial.toList))) :=
  let tIdx : Nat := if T = 0 then 0 else step % T
  let ab : Float := alphaBars.getD tIdx 1.0
  let sqrtAb : Float := MathFunctions.sqrt (Max.max ab 0.0)
  let sqrtOneMinusAb : Float := MathFunctions.sqrt (Max.max (1.0 - ab) 0.0)
  let x_t : Spec.Tensor Float
      (leading.concat (Spec.Shape.ofList (c :: spatial.toList))) :=
    Spec.Tensor.addSpec
      (Spec.Tensor.scaleSpec x0 sqrtAb)
      (Spec.Tensor.scaleSpec eps sqrtOneMinusAb)
  let tNorm : Float := if T <= 1 then 0.0 else Float.ofNat tIdx / Float.ofNat (T - 1)
  TorchLean.Sample.mk (appendTimeChannel leading spatial x_t tNorm) eps

/--
Build a deterministic epsilon-prediction training sample.

This is the common DDPM training step used by examples: draw reproducible Gaussian noise from
`(seed, step)`, corrupt $x_0$, and use that same noise as the target.
-/
def noisedSample (leading : Spec.Shape) {d c : Nat} (spatial : Vector Nat d)
    (alphaBars : Array Float) (T : Nat)
    (x0 : Spec.Tensor Float
      (leading.concat (Spec.Shape.ofList (c :: spatial.toList)))) (seed step : Nat) :
    TorchLean.Sample.Supervised Float
      (leading.concat (Spec.Shape.ofList ((c + 1) :: spatial.toList)))
      (leading.concat (Spec.Shape.ofList (c :: spatial.toList))) :=
  noisedSampleFromEps leading spatial alphaBars T x0
    (randomEps (s := leading.concat (Spec.Shape.ofList (c :: spatial.toList))) seed step)
    (seed + step)

/--
One deterministic DDIM reverse update ($\eta=0$).

Given $x_t$, predicted epsilon, and adjacent schedule values, this estimates $x_0$ and remixes it to
the previous timestep.

We clamp the intermediate $x_0$ estimate to the training image range $[-1,1]$.  This is the standard
"clipped denoised" stabilizer used by many DDPM/DDIM samplers: without it, a compact model can
drive one color channel far outside the data range and the final PPM exporter merely clips the
damage into saturated color blobs.
-/
def ddimPrev {s : Spec.Shape}
    (abPrev ab : Float)
    (x_t epsHat : Spec.Tensor Float s) : Spec.Tensor Float s :=
  let sqrtAb : Float := MathFunctions.sqrt (Max.max ab 0.0)
  let sqrtAbPrev : Float := MathFunctions.sqrt (Max.max abPrev 0.0)
  let sqrtOneMinusAb : Float := MathFunctions.sqrt (Max.max (1.0 - ab) 0.0)
  let sqrtOneMinusAbPrev : Float := MathFunctions.sqrt (Max.max (1.0 - abPrev) 0.0)
  let x0Hat : Spec.Tensor Float s :=
    Spec.Tensor.scaleSpec
      (Spec.Tensor.subSpec x_t (Spec.Tensor.scaleSpec epsHat sqrtOneMinusAb))
      (1.0 / (if sqrtAb > 1e-12 then sqrtAb else 1e-12))
  let x0Clipped : Spec.Tensor Float s :=
    Spec.Tensor.clampSpec x0Hat (-1.0) 1.0
  Spec.Tensor.addSpec
    (Spec.Tensor.scaleSpec x0Clipped sqrtAbPrev)
    (Spec.Tensor.scaleSpec epsHat sqrtOneMinusAbPrev)

/--
Write the first image in an RGB NCHW batch as an ASCII PPM.

This dependency-free writer emits portable image artifacts for examples and rendered diagnostics.
-/
def writeFirstRgbPpm {batch c h w : Nat}
    (path : System.FilePath) (x : Spec.Tensor Float (.dim batch (.dim c (.dim h (.dim w .scalar))))) : IO Unit := do
  if c < 3 then
    throw <| IO.userError "diffusion PPM export requires at least 3 channels"
  if let some parent := path.parent then
    IO.FS.createDirAll parent
  let clamp01 (v : Float) : Float :=
    if v < 0.0 then 0.0 else if v > 1.0 then 1.0 else v
  let toByte (v : Float) : Nat :=
    let v01 := clamp01 ((v + 1.0) / 2.0)
    Nat.min 255 ((v01 * 255.0).toUInt64.toNat)
  let hOut ← IO.FS.Handle.mk path IO.FS.Mode.write
  hOut.putStr s!"P3\n{w} {h}\n255\n"
  let getPx (ci hi wi : Nat) : Float :=
    (Spec.getSpec (α := Float) (s := .dim batch (.dim c (.dim h (.dim w .scalar)))) x [0, ci, hi, wi]).getD 0.0
  for hi in [0:h] do
    for wi in [0:w] do
      hOut.putStr s!"{toByte (getPx 0 hi wi)} {toByte (getPx 1 hi wi)} {toByte (getPx 2 hi wi)}\n"

end diffusion

end API
end NN
