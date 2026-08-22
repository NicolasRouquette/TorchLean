/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded
public import NN.Spec.Core.TensorReductionShape.ConcatSlice

/-!
# Diffusion Models

Config-style diffusion model constructors plus reusable, dataset-independent DDPM/DDIM helpers.

The runnable examples decide where data comes from (CIFAR-10, ImageNet-style folders, synthetic
artifacts).  The definitions here are shape-parametric and can be reused by tests, examples, and
future proof layer specifications.
-/

@[expose] public section

namespace TorchLean


open Spec Tensor

namespace nn
namespace models

/-- Configuration for a minimal epsilon-predictor conv net. -/
structure EpsConvNetConfig (d : Nat) where
  /-- Number of channels in the denoised sample. -/
  dataChannels : Nat
  /-- Extent of each spatial axis. -/
  spatial : Vector Nat d
  /-- Spatial axes are nonempty. -/
  spatialNonzero : ∀ i : Fin d, spatial.get i ≠ 0
  /-- Hidden channel width. -/
  hiddenChannels : Nat := 32

namespace EpsConvNetConfig

/-- Epsilon-predictor input shape, with one extra channel carrying the diffusion time. -/
def inputShape {d : Nat} (cfg : EpsConvNetConfig d)
    (leading : Spec.Shape := .scalar) : Spec.Shape :=
  leading.concat (Spec.Shape.ofList ((cfg.dataChannels + 1) :: cfg.spatial.toList))

/-- Epsilon-predictor output shape matching the denoised data channels. -/
def outputShape {d : Nat} (cfg : EpsConvNetConfig d)
    (leading : Spec.Shape := .scalar) : Spec.Shape :=
  leading.concat (Spec.Shape.ofList (cfg.dataChannels :: cfg.spatial.toList))

namespace Internal

/-- Seeded shape-preserving convolution over an arbitrary spatial rank. -/
def sameConv {d : Nat} (cfg : EpsConvNetConfig d) (leading : Spec.Shape)
    (inChannels outChannels : Nat) [NeZero inChannels] :
    nn.Builder (nn.Sequential
      (leading.concat (Spec.Shape.ofList (inChannels :: cfg.spatial.toList)))
      (leading.concat (Spec.Shape.ofList (outChannels :: cfg.spatial.toList)))) :=
  let layer := nn.conv (leading := leading)
      (inChannels := inChannels) cfg.spatial
      { outChannels := outChannels
        kernel := Vector.replicate d 1
        stride := Vector.replicate d 1
        padding := Vector.replicate d 0
        kernelNonzero := by intro i; simp [Vector.get]
        strideNonzero := by intro i; simp [Vector.get] }
  by
    simpa [Spec.Shape.concat, Spec.convOutSpatial_unit cfg.spatial cfg.spatialNonzero] using layer

end Internal
end EpsConvNetConfig

/--
Build a minimal epsilon-predictor conv net:
`conv -> relu -> conv -> relu -> conv -> relu -> conv`.

This stays compact enough for the eager CUDA example while giving the CIFAR trainer more denoising
capacity than a bare two-layer network.
-/
def epsConvNet {d : Nat} (cfg : EpsConvNetConfig d) (leading : Spec.Shape := .scalar)
    (h_dataC : cfg.dataChannels ≠ 0 := by decide)
    (h_inC : (cfg.dataChannels + 1) ≠ 0 := by decide)
    (h_hiddenC : cfg.hiddenChannels ≠ 0 := by decide) :
    nn.Builder (nn.Sequential (cfg.inputShape leading) (cfg.outputShape leading)) :=
  letI : NeZero cfg.dataChannels := ⟨h_dataC⟩
  letI : NeZero (cfg.dataChannels + 1) := ⟨h_inC⟩
  letI : NeZero cfg.hiddenChannels := ⟨h_hiddenC⟩
  nn.Sequential![
    EpsConvNetConfig.Internal.sameConv cfg leading (cfg.dataChannels + 1) cfg.hiddenChannels,
    relu,
    EpsConvNetConfig.Internal.sameConv cfg leading cfg.hiddenChannels cfg.hiddenChannels,
    relu,
    EpsConvNetConfig.Internal.sameConv cfg leading cfg.hiddenChannels cfg.hiddenChannels,
    relu,
    EpsConvNetConfig.Internal.sameConv cfg leading cfg.hiddenChannels cfg.dataChannels
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
    (leading : Spec.Shape := .scalar)
    (h_dataC : cfg.dataChannels ≠ 0 := by decide)
    (h_inC : (cfg.dataChannels + 1) ≠ 0 := by decide)
    (h_hiddenC : cfg.hiddenChannels ≠ 0 := by decide) :
    nn.Builder (nn.Sequential (cfg.inputShape leading) (cfg.outputShape leading)) :=
  letI : NeZero cfg.dataChannels := ⟨h_dataC⟩
  letI : NeZero (cfg.dataChannels + 1) := ⟨h_inC⟩
  letI : NeZero cfg.hiddenChannels := ⟨h_hiddenC⟩
  nn.Sequential![
    EpsConvNetConfig.Internal.sameConv cfg leading (cfg.dataChannels + 1) cfg.hiddenChannels,
    relu,
    (do
      let block ←
        nn.Sequential![
          EpsConvNetConfig.Internal.sameConv cfg leading cfg.hiddenChannels cfg.hiddenChannels,
          relu,
          EpsConvNetConfig.Internal.sameConv cfg leading cfg.hiddenChannels cfg.hiddenChannels
        ]
      pure (nn.blocks.residual block)),
    relu,
    (do
      let block ←
        nn.Sequential![
          EpsConvNetConfig.Internal.sameConv cfg leading cfg.hiddenChannels cfg.hiddenChannels,
          relu,
          EpsConvNetConfig.Internal.sameConv cfg leading cfg.hiddenChannels cfg.hiddenChannels
        ]
      pure (nn.blocks.residual block)),
    relu,
    EpsConvNetConfig.Internal.sameConv cfg leading cfg.hiddenChannels cfg.dataChannels
  ]

end models
end nn

namespace diffusion

/-- Map a tensor from the unit interval to the signed unit interval. -/
def unitToSignedUnit {s : Spec.Shape} (x01 : Spec.Tensor Float s) : Spec.Tensor Float s :=
  Spec.Tensor.mapSpec (fun x => 2.0 * x - 1.0) x01

/--
Deterministic Gaussian epsilon tensor for an arbitrary diffusion shape.

The `(seed, step)` pair is turned into the runtime RNG key, so examples and artifact generation can
reproduce the same noising path without ambient randomness.
-/
def normalNoise {s : Spec.Shape} (seed step : Nat) : Spec.Tensor Float s :=
  let key : UInt64 := _root_.Runtime.Autograd.TorchLean.Random.keyOf (seed := seed) (counter := step)
  _root_.Runtime.Autograd.TorchLean.Random.normal (α := Float) key (s := s)

/-- Linear beta-schedule value at timestep `t`. -/
def linearBeta (T : Nat) (betaStart betaEnd : Float) (t : Nat) : Float :=
  if T <= 1 then
    betaEnd
  else
    let u := Float.ofNat t / Float.ofNat (T - 1)
    betaStart + u * (betaEnd - betaStart)

/-- The cumulative coefficient $\bar\alpha_t=\prod_{s=0}^{t}(1-\beta_s)$. -/
def linearAlphaBar (T : Nat) (betaStart betaEnd : Float) (t : Nat) : Float :=
  (List.range (t + 1)).foldl
    (fun alpha s => alpha * (1.0 - linearBeta T betaStart betaEnd s)) 1.0

/--
The `T` cumulative coefficients of a linear beta schedule.

The length belongs to the return type, so a consumer cannot pair the coefficients with a different
timestep count.
-/
def linearAlphaBars (T : Nat) (betaStart betaEnd : Float) : Vector Float T :=
  Vector.ofFn fun t => linearAlphaBar T betaStart betaEnd t.val

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
def noisedSampleFromNoise (leading : Spec.Shape) {d c T : Nat} [NeZero T]
    (spatial : Vector Nat d) (alphaBars : Vector Float T)
    (x0 eps : Spec.Tensor Float
      (leading.concat (Spec.Shape.ofList (c :: spatial.toList)))) (step : Nat) :
    TorchLean.Sample.Supervised Float
      (leading.concat (Spec.Shape.ofList ((c + 1) :: spatial.toList)))
      (leading.concat (Spec.Shape.ofList (c :: spatial.toList))) :=
  let tIdx : Fin T :=
    ⟨step % T, Nat.mod_lt step (Nat.pos_of_ne_zero (NeZero.ne T))⟩
  let ab : Float := alphaBars.get tIdx
  let sqrtAb : Float := MathFunctions.sqrt (Max.max ab 0.0)
  let sqrtOneMinusAb : Float := MathFunctions.sqrt (Max.max (1.0 - ab) 0.0)
  let x_t : Spec.Tensor Float
      (leading.concat (Spec.Shape.ofList (c :: spatial.toList))) :=
    Spec.Tensor.addSpec
      (Spec.Tensor.scaleSpec x0 sqrtAb)
      (Spec.Tensor.scaleSpec eps sqrtOneMinusAb)
  let tNorm : Float :=
    if T <= 1 then 0.0 else Float.ofNat tIdx.val / Float.ofNat (T - 1)
  TorchLean.Sample.mk (appendTimeChannel leading spatial x_t tNorm) eps

/--
Build a deterministic epsilon-prediction training sample.

This is the common DDPM training step used by examples: draw reproducible Gaussian noise from
`(seed, step)`, corrupt $x_0$, and use that same noise as the target.
-/
def noisedSample (leading : Spec.Shape) {d c T : Nat} [NeZero T]
    (spatial : Vector Nat d) (alphaBars : Vector Float T)
    (x0 : Spec.Tensor Float
      (leading.concat (Spec.Shape.ofList (c :: spatial.toList)))) (seed step : Nat) :
    TorchLean.Sample.Supervised Float
      (leading.concat (Spec.Shape.ofList ((c + 1) :: spatial.toList)))
      (leading.concat (Spec.Shape.ofList (c :: spatial.toList))) :=
  noisedSampleFromNoise leading spatial alphaBars x0
    (normalNoise (s := leading.concat (Spec.Shape.ofList (c :: spatial.toList))) seed step)
    step

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

end diffusion

end TorchLean
