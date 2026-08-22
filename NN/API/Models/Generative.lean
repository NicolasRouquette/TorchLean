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

These models act on a trailing feature axis and preserve any leading dimensions. Examples may
flatten structured observations before applying them, while convolutional or operator-based
models can use their own shape-specific constructors.
-/

@[expose] public section

namespace TorchLean


open Spec Tensor

namespace nn
namespace models

namespace DenseGenerative

/-- Widths shared by dense generative models. -/
structure Config where
  /-- Width of the data feature axis. -/
  dataDim : Nat
  /-- Width of the hidden layers. -/
  hiddenDim : Nat
  /-- Width of the latent representation. -/
  latentDim : Nat
deriving Repr

/-- Data shape with arbitrary leading dimensions. -/
abbrev Config.dataShape (cfg : Config)
    (leading : Spec.Shape := .scalar) : Spec.Shape :=
  leading.appendDim cfg.dataDim

/-- Latent shape with arbitrary leading dimensions. -/
abbrev Config.latentShape (cfg : Config)
    (leading : Spec.Shape := .scalar) : Spec.Shape :=
  leading.appendDim cfg.latentDim

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
abbrev Config.vaeOutputShape (cfg : Config)
    (leading : Spec.Shape := .scalar) : Spec.Shape :=
  leading.appendDim (cfg.dataDim + 2 * cfg.latentDim)

/-- Scalar-score shape with arbitrary leading dimensions. -/
abbrev Config.scoreShape (_cfg : Config)
    (leading : Spec.Shape := .scalar) : Spec.Shape :=
  leading.appendDim 1

/-- Autoencoder: `x -> hidden -> latent -> hidden -> reconstruction`. -/
def autoencoder (cfg : Config) (leading : Spec.Shape := .scalar) :
    nn.Builder (nn.Sequential (cfg.dataShape leading) (cfg.dataShape leading)) :=
  nn.Sequential![
    linear cfg.dataDim cfg.hiddenDim (leading := leading),
    relu,
    linear cfg.hiddenDim cfg.latentDim (leading := leading),
    relu,
    linear cfg.latentDim cfg.hiddenDim (leading := leading),
    relu,
    linear cfg.hiddenDim cfg.dataDim (leading := leading),
    nn.sigmoid
  ]

/-- Compact β-VAE-style network producing reconstruction plus latent statistics. -/
def vae (cfg : Config) (leading : Spec.Shape := .scalar) :
    nn.Builder (nn.Sequential (cfg.dataShape leading) (cfg.vaeOutputShape leading)) :=
  nn.Sequential![
    linear cfg.dataDim cfg.hiddenDim (leading := leading),
    relu,
    linear cfg.hiddenDim cfg.latentDim (leading := leading),
    relu,
    linear cfg.latentDim cfg.hiddenDim (leading := leading),
    relu,
    linear cfg.hiddenDim (cfg.dataDim + 2 * cfg.latentDim)
      (leading := leading)
  ]

/-- VQ-VAE-style encoder/decoder with a narrow discrete-code proxy bottleneck. -/
def vqVae (cfg : Config) (leading : Spec.Shape := .scalar) :
    nn.Builder (nn.Sequential (cfg.dataShape leading) (cfg.dataShape leading)) :=
  nn.Sequential![
    linear cfg.dataDim cfg.hiddenDim (leading := leading),
    relu,
    linear cfg.hiddenDim cfg.latentDim (leading := leading),
    nn.tanh,
    linear cfg.latentDim cfg.hiddenDim (leading := leading),
    relu,
    linear cfg.hiddenDim cfg.dataDim (leading := leading),
    nn.sigmoid
  ]

/-- Generator `z -> x`. -/
def ganGenerator (cfg : Config) (leading : Spec.Shape := .scalar) :
    nn.Builder (nn.Sequential (cfg.latentShape leading) (cfg.dataShape leading)) :=
  nn.Sequential![
    linear cfg.latentDim cfg.hiddenDim (leading := leading),
    relu,
    linear cfg.hiddenDim cfg.hiddenDim (leading := leading),
    relu,
    linear cfg.hiddenDim cfg.dataDim (leading := leading),
    nn.sigmoid
  ]

/-- Discriminator/critic `x -> score`. -/
def ganDiscriminator (cfg : Config)
    (leading : Spec.Shape := .scalar) :
    nn.Builder (nn.Sequential (cfg.dataShape leading) (cfg.scoreShape leading)) :=
  nn.Sequential![
    linear cfg.dataDim cfg.hiddenDim (leading := leading),
    relu,
    linear cfg.hiddenDim cfg.hiddenDim (leading := leading),
    relu,
    linear cfg.hiddenDim 1 (leading := leading),
    nn.sigmoid
  ]

end DenseGenerative
end models
end nn

end TorchLean
