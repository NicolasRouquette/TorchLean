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
  spatial : Tensor Nat [d]
  /-- Spatial axes are nonempty. -/
  spatialNonzero : ∀ i : Fin d, spatial.getScalar i ≠ 0
  /-- Hidden channel width. -/
  hiddenChannels : Nat := 32
  /-- Geometry used by every convolution in the epsilon predictor. -/
  block : ConvGeometry d
  /-- The configured convolutions preserve the sample's spatial extent. -/
  blockPreservesSpatial :
    Spec.convOutSpatial spatial block.kernel block.stride block.padding = spatial

namespace EpsConvNetConfig

/-- Epsilon-predictor input shape, with one extra channel carrying the diffusion time. -/
def inputShape {d : Nat} (cfg : EpsConvNetConfig d)
    (leading : List Nat := []) : List Nat :=
  leading ++ (cfg.dataChannels + 1) :: cfg.spatial.toList

/-- Epsilon-predictor output shape matching the denoised data channels. -/
def outputShape {d : Nat} (cfg : EpsConvNetConfig d)
    (leading : List Nat := []) : List Nat :=
  leading ++ cfg.dataChannels :: cfg.spatial.toList

end EpsConvNetConfig

namespace Internal

/-- Implementation helper for the shape-preserving convolutions in the epsilon predictors. -/
def epsConv {d : Nat} (cfg : EpsConvNetConfig d) (leading : List Nat)
    (inChannels outChannels : Nat) [NeZero inChannels] :
    nn.Builder (nn.Sequential
      (leading ++ inChannels :: cfg.spatial.toList)
      (leading ++ outChannels :: cfg.spatial.toList)) := by
  simpa [ConvGeometry.toConv, ConvGeometry.outSpatial,
    cfg.blockPreservesSpatial] using
    (nn.conv (leading := leading) (inChannels := inChannels) cfg.spatial
      (cfg.block.toConv outChannels))

end Internal

/--
Build a minimal epsilon-predictor conv net:
`conv -> relu -> conv -> relu -> conv -> relu -> conv`.

This stays compact enough for the eager CUDA example while giving the CIFAR trainer more denoising
capacity than a bare two-layer network.
-/
def epsConvNet {d : Nat} (cfg : EpsConvNetConfig d) (leading : List Nat := [])
    (h_dataC : cfg.dataChannels ≠ 0 := by decide)
    (h_inC : (cfg.dataChannels + 1) ≠ 0 := by decide)
    (h_hiddenC : cfg.hiddenChannels ≠ 0 := by decide) :
    nn.Builder (nn.Sequential (cfg.inputShape leading) (cfg.outputShape leading)) :=
  letI : NeZero cfg.dataChannels := ⟨h_dataC⟩
  letI : NeZero (cfg.dataChannels + 1) := ⟨h_inC⟩
  letI : NeZero cfg.hiddenChannels := ⟨h_hiddenC⟩
  nn.Sequential![
    Internal.epsConv cfg leading (cfg.dataChannels + 1) cfg.hiddenChannels,
    relu,
    Internal.epsConv cfg leading cfg.hiddenChannels cfg.hiddenChannels,
    relu,
    Internal.epsConv cfg leading cfg.hiddenChannels cfg.hiddenChannels,
    relu,
    Internal.epsConv cfg leading cfg.hiddenChannels cfg.dataChannels
  ]

/--
Build a stronger same-resolution residual epsilon predictor.

Architecture:

`stem conv -> relu -> residual block -> relu -> residual block -> relu -> output conv`

Each residual block preserves `hiddenChannels :: spatial` and computes
$x+\operatorname{conv}(\operatorname{relu}(\operatorname{conv}(x)))$.  This compact residual
denoiser omits U-Net downsampling, upsampling,
and multi-scale skip concatenation. It is still a useful compact architecture because
residual paths make the denoising problem much easier than a plain conv chain while staying within
the eager CUDA memory envelope used by examples.
-/
def epsResidualConvNet {d : Nat} (cfg : EpsConvNetConfig d)
    (leading : List Nat := [])
    (h_dataC : cfg.dataChannels ≠ 0 := by decide)
    (h_inC : (cfg.dataChannels + 1) ≠ 0 := by decide)
    (h_hiddenC : cfg.hiddenChannels ≠ 0 := by decide) :
    nn.Builder (nn.Sequential (cfg.inputShape leading) (cfg.outputShape leading)) :=
  letI : NeZero cfg.dataChannels := ⟨h_dataC⟩
  letI : NeZero (cfg.dataChannels + 1) := ⟨h_inC⟩
  letI : NeZero cfg.hiddenChannels := ⟨h_hiddenC⟩
  nn.Sequential![
    Internal.epsConv cfg leading (cfg.dataChannels + 1) cfg.hiddenChannels,
    relu,
    (do
      let block ←
        nn.Sequential![
          Internal.epsConv cfg leading cfg.hiddenChannels cfg.hiddenChannels,
          relu,
          Internal.epsConv cfg leading cfg.hiddenChannels cfg.hiddenChannels
        ]
      pure (nn.blocks.residual block)),
    relu,
    (do
      let block ←
        nn.Sequential![
          Internal.epsConv cfg leading cfg.hiddenChannels cfg.hiddenChannels,
          relu,
          Internal.epsConv cfg leading cfg.hiddenChannels cfg.hiddenChannels
        ]
      pure (nn.blocks.residual block)),
    relu,
    Internal.epsConv cfg leading cfg.hiddenChannels cfg.dataChannels
  ]

end models
end nn

namespace diffusion

/-- Map a tensor from the unit interval to the signed unit interval. -/
def unitToSignedUnit {shape : List Nat} (x01 : Tensor Float shape) : Tensor Float shape :=
  Spec.Tensor.mapSpec (fun x => 2.0 * x - 1.0) x01

/--
Deterministic Gaussian epsilon tensor for an arbitrary diffusion shape.

The `(seed, step)` pair is turned into the runtime RNG key, so examples and artifact generation can
reproduce the same noising path without ambient randomness.
-/
def normalNoise {shape : List Nat} (seed step : Nat) : Tensor Float shape :=
  let key : UInt64 := _root_.Runtime.Autograd.TorchLean.Random.keyOf (seed := seed) (counter := step)
  _root_.Runtime.Autograd.TorchLean.Random.normal (α := Float) key (s := Shape.ofList shape)

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
def linearAlphaBars (T : Nat) (betaStart betaEnd : Float) : Tensor Float [T] :=
  Spec.Tensor.ofFn fun t => linearAlphaBar T betaStart betaEnd t.val

namespace Internal

/-- Recursive implementation of `diffusion.appendTimeChannel`. -/
def appendTimeChannelImpl (leading : List Nat) {d c : Nat} (spatial : Tensor Nat [d])
    (x : Tensor Float (leading ++ c :: spatial.toList)) (tNorm : Float) :
    Tensor Float (leading ++ (c + 1) :: spatial.toList) :=
  match leading with
  | [] =>
      Spec.Tensor.concatAxisSpec .scalar x <|
        Spec.Tensor.dim fun _ => Spec.fill tNorm (Shape.ofList spatial.toList)
  | _ :: rest =>
      match x with
      | .dim values =>
          Spec.Tensor.dim fun i => appendTimeChannelImpl rest spatial (values i) tNorm

end Internal

/--
Append a constant time channel after arbitrary leading dimensions.

The input layout is `(leading..., channels, spatial...)`. The result preserves every leading and
spatial axis and changes only the channel count from `c` to `c + 1`.
-/
def appendTimeChannel (leading : List Nat) {d c : Nat} (spatial : Tensor Nat [d])
    (x : Tensor Float (leading ++ c :: spatial.toList))
    (tNorm : Float) :
    Tensor Float (leading ++ (c + 1) :: spatial.toList) :=
  Internal.appendTimeChannelImpl leading spatial x tNorm

/--
Build an epsilon-prediction training sample from explicit noise.

The caller supplies `eps`, usually from the runtime RNG.  Keeping randomness outside this helper
makes the transformation reusable:

$x_t=\sqrt{\bar{\alpha}_t}\,x_0+\sqrt{1-\bar{\alpha}_t}\,\varepsilon$, with target
$\varepsilon$.
-/
def noisedSampleFromNoise (leading : List Nat) {d c T : Nat} [NeZero T]
    (spatial : Tensor Nat [d]) (alphaBars : Tensor Float [T])
    (x0 eps : Tensor Float (leading ++ c :: spatial.toList)) (step : Nat) :
    TorchLean.Sample.Supervised Float
      (leading ++ (c + 1) :: spatial.toList)
      (leading ++ c :: spatial.toList) :=
  let tIdx : Fin T :=
    ⟨step % T, Nat.mod_lt step (Nat.pos_of_ne_zero (NeZero.ne T))⟩
  let ab : Float := alphaBars.getScalar tIdx
  let sqrtAb : Float := MathFunctions.sqrt (Max.max ab 0.0)
  let sqrtOneMinusAb : Float := MathFunctions.sqrt (Max.max (1.0 - ab) 0.0)
  let x_t : Tensor Float (leading ++ c :: spatial.toList) :=
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
def noisedSample (leading : List Nat) {d c T : Nat} [NeZero T]
    (spatial : Tensor Nat [d]) (alphaBars : Tensor Float [T])
    (x0 : Tensor Float (leading ++ c :: spatial.toList)) (seed step : Nat) :
    TorchLean.Sample.Supervised Float
      (leading ++ (c + 1) :: spatial.toList)
      (leading ++ c :: spatial.toList) :=
  noisedSampleFromNoise leading spatial alphaBars x0
    (normalNoise (shape := leading ++ c :: spatial.toList) seed step)
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
def ddimPrev {shape : List Nat}
    (abPrev ab : Float)
    (x_t epsHat : Tensor Float shape) : Tensor Float shape :=
  let sqrtAb : Float := MathFunctions.sqrt (Max.max ab 0.0)
  let sqrtAbPrev : Float := MathFunctions.sqrt (Max.max abPrev 0.0)
  let sqrtOneMinusAb : Float := MathFunctions.sqrt (Max.max (1.0 - ab) 0.0)
  let sqrtOneMinusAbPrev : Float := MathFunctions.sqrt (Max.max (1.0 - abPrev) 0.0)
  let x0Hat : Tensor Float shape :=
    Spec.Tensor.scaleSpec
      (Spec.Tensor.subSpec x_t (Spec.Tensor.scaleSpec epsHat sqrtOneMinusAb))
      (1.0 / (if sqrtAb > 1e-12 then sqrtAb else 1e-12))
  let x0Clipped : Tensor Float shape :=
    Spec.Tensor.clampSpec x0Hat (-1.0) 1.0
  Spec.Tensor.addSpec
    (Spec.Tensor.scaleSpec x0Clipped sqrtAbPrev)
    (Spec.Tensor.scaleSpec epsHat sqrtOneMinusAbPrev)

end diffusion

end TorchLean
