/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded
public import NN.API.Tensor

/-!
# Generative Models

Config-style constructors for runnable generative examples.

These are vector models: examples can flatten images, train the model, and later swap in
convolutional encoders/decoders without changing the command-line/data-loading surface.
-/

@[expose] public section

namespace TorchLean


open Spec Tensor

namespace nn
namespace models

/-- Shared dimensions for vector generative examples. -/
structure VectorGenerativeConfig where
  batch : Nat
  dataDim : Nat
  hiddenDim : Nat
  latentDim : Nat
deriving Repr

/-- Convenience constructor for compact vector generative models. -/
def vectorGenerativeConfig (batch dataDim hiddenDim latentDim : Nat) : VectorGenerativeConfig :=
  { batch, dataDim, hiddenDim, latentDim }

/-- Batched data-vector shape shared by vector generative examples. -/
abbrev vectorDataShape (cfg : VectorGenerativeConfig) : Spec.Shape :=
  .dim cfg.batch (.dim cfg.dataDim .scalar)

/-- Batched latent-vector shape shared by vector generative examples. -/
abbrev vectorLatentShape (cfg : VectorGenerativeConfig) : Spec.Shape :=
  .dim cfg.batch (.dim cfg.latentDim .scalar)

/--
β-VAE-style supervised output shape.

Rows contain:
- reconstruction, length `dataDim`;
- latent mean proxy, length `latentDim`;
- latent log-variance proxy, length `latentDim`.

The runnable example trains this compact target with MSE, which is a practical path for the
runtime. The formal VAE ELBO/KL objective lives in `NN.Spec.Models.Vae` and
`NN.MLTheory.Generative.Latent.VAE`.
-/
abbrev vectorVaeOutShape (cfg : VectorGenerativeConfig) : Spec.Shape :=
  .dim cfg.batch (.dim (cfg.dataDim + 2 * cfg.latentDim) .scalar)

/-- Supervised reconstruction sample: target equals input. -/
def reconstructionSample {α : Type} (cfg : VectorGenerativeConfig)
    (x : Spec.Tensor α (vectorDataShape cfg)) :
    TorchLean.Sample.Supervised α (vectorDataShape cfg) (vectorDataShape cfg) :=
  TorchLean.Sample.mk x x

/--
Target for compact VAE-style examples.

Rows contain the reconstruction target followed by zero mean/log-variance proxy channels.
-/
def zeroLatentStatsTarget (cfg : VectorGenerativeConfig)
    (x : Spec.Tensor Float (vectorDataShape cfg)) : Spec.Tensor Float (vectorVaeOutShape cfg) :=
  Spec.Tensor.dim (fun bi =>
    let row := Spec.getAtSpec x bi
    Spec.Tensor.dim (fun j =>
      let v :=
        if h : j.val < cfg.dataDim then
          Spec.Tensor.toScalar (Spec.get row ⟨j.val, h⟩)
        else
          0.0
      Spec.Tensor.scalar v))

/-- Supervised compact VAE sample: image reconstruction plus zero latent-stat targets. -/
def vaeSample (cfg : VectorGenerativeConfig)
    (x : Spec.Tensor Float (vectorDataShape cfg)) :
    TorchLean.Sample.Supervised Float (vectorDataShape cfg) (vectorVaeOutShape cfg) :=
  TorchLean.Sample.mk x (zeroLatentStatsTarget cfg x)

/-- Deterministic matrix-valued pseudo-random tensor in `[lo, hi)`. -/
def vectorNoise (batch dim seed salt : Nat) (lo hi : Float) :
    Spec.Tensor Float (.dim batch (.dim dim .scalar)) :=
  Spec.Tensor.dim (fun bi =>
    Spec.Tensor.dim (fun j =>
      let k := bi.val * dim + j.val
      let raw := (seed * 1103515245 + k * 12345 + salt) % 997
      let u := Float.ofNat raw / 997.0
      Spec.Tensor.scalar (lo + (hi - lo) * u)))

/-- Deterministic latent noise for generator examples. -/
def latentNoise (cfg : VectorGenerativeConfig) (seed : Nat) :
    Spec.Tensor Float (vectorLatentShape cfg) :=
  vectorNoise cfg.batch cfg.latentDim seed 17 (-1.0) 1.0

/-- Deterministic data-shaped noise for discriminator examples. -/
def dataNoise (cfg : VectorGenerativeConfig) (seed : Nat) :
    Spec.Tensor Float (vectorDataShape cfg) :=
  vectorNoise cfg.batch cfg.dataDim seed 91 0.0 1.0

/-- Constant discriminator/critic target. -/
def scoreTarget (cfg : VectorGenerativeConfig) (value : Float) :
    Spec.Tensor Float (.dim cfg.batch (.dim 1 .scalar)) :=
  Spec.Tensor.dim (fun _ => Spec.Tensor.dim (fun _ => Spec.Tensor.scalar value))

/-- Target score for real samples. -/
def onesScore (cfg : VectorGenerativeConfig) : Spec.Tensor Float (.dim cfg.batch (.dim 1 .scalar)) :=
  scoreTarget cfg 1.0

/-- Target score for generated or noise samples. -/
def zerosScore (cfg : VectorGenerativeConfig) : Spec.Tensor Float (.dim cfg.batch (.dim 1 .scalar)) :=
  scoreTarget cfg 0.0

/-- Autoencoder: `x -> hidden -> latent -> hidden -> reconstruction`. -/
def vectorAutoencoder (cfg : VectorGenerativeConfig) :
    nn.Builder (nn.Sequential (vectorDataShape cfg) (vectorDataShape cfg)) :=
  nn.Sequential![
    linear cfg.dataDim cfg.hiddenDim (pfx := .dim cfg.batch .scalar),
    relu,
    linear cfg.hiddenDim cfg.latentDim (pfx := .dim cfg.batch .scalar),
    relu,
    linear cfg.latentDim cfg.hiddenDim (pfx := .dim cfg.batch .scalar),
    relu,
    linear cfg.hiddenDim cfg.dataDim (pfx := .dim cfg.batch .scalar),
    nn.sigmoid
  ]

/-- Compact β-VAE-style network producing reconstruction plus latent statistics. -/
def vectorVae (cfg : VectorGenerativeConfig) :
    nn.Builder (nn.Sequential (vectorDataShape cfg) (vectorVaeOutShape cfg)) :=
  nn.Sequential![
    linear cfg.dataDim cfg.hiddenDim (pfx := .dim cfg.batch .scalar),
    relu,
    linear cfg.hiddenDim cfg.latentDim (pfx := .dim cfg.batch .scalar),
    relu,
    linear cfg.latentDim cfg.hiddenDim (pfx := .dim cfg.batch .scalar),
    relu,
    linear cfg.hiddenDim (cfg.dataDim + 2 * cfg.latentDim)
      (pfx := .dim cfg.batch .scalar)
  ]

/-- VQ-VAE-style encoder/decoder with a narrow discrete-code proxy bottleneck. -/
def vectorVqVae (cfg : VectorGenerativeConfig) :
    nn.Builder (nn.Sequential (vectorDataShape cfg) (vectorDataShape cfg)) :=
  nn.Sequential![
    linear cfg.dataDim cfg.hiddenDim (pfx := .dim cfg.batch .scalar),
    relu,
    linear cfg.hiddenDim cfg.latentDim (pfx := .dim cfg.batch .scalar),
    nn.tanh,
    linear cfg.latentDim cfg.hiddenDim (pfx := .dim cfg.batch .scalar),
    relu,
    linear cfg.hiddenDim cfg.dataDim (pfx := .dim cfg.batch .scalar),
    nn.sigmoid
  ]

/-- Generator `z -> x`. -/
def vectorGanGenerator (cfg : VectorGenerativeConfig) :
    nn.Builder (nn.Sequential (vectorLatentShape cfg) (vectorDataShape cfg)) :=
  nn.Sequential![
    linear cfg.latentDim cfg.hiddenDim (pfx := .dim cfg.batch .scalar),
    relu,
    linear cfg.hiddenDim cfg.hiddenDim (pfx := .dim cfg.batch .scalar),
    relu,
    linear cfg.hiddenDim cfg.dataDim (pfx := .dim cfg.batch .scalar),
    nn.sigmoid
  ]

/-- Discriminator/critic `x -> score`. -/
def vectorGanDiscriminator (cfg : VectorGenerativeConfig) :
    nn.Builder (nn.Sequential (vectorDataShape cfg) (.dim cfg.batch (.dim 1 .scalar))) :=
  nn.Sequential![
    linear cfg.dataDim cfg.hiddenDim (pfx := .dim cfg.batch .scalar),
    relu,
    linear cfg.hiddenDim cfg.hiddenDim (pfx := .dim cfg.batch .scalar),
    relu,
    linear cfg.hiddenDim 1 (pfx := .dim cfg.batch .scalar),
    nn.sigmoid
  ]

end models
end nn

end TorchLean
