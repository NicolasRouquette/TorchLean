/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded
public import NN.Tensor

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
    (leading : List Nat := []) : List Nat :=
  leading ++ [cfg.dataDim]

/-- Latent shape with arbitrary leading dimensions. -/
abbrev Config.latentShape (cfg : Config)
    (leading : List Nat := []) : List Nat :=
  leading ++ [cfg.latentDim]

/--
Output shape for supervised reconstruction with two auxiliary latent-statistic vectors.

Rows contain a reconstruction of length `dataDim`, followed by two vectors of length `latentDim`.
The constructor does not prescribe probabilistic semantics for those auxiliary values.
-/
abbrev Config.reconstructionStatisticsShape (cfg : Config)
    (leading : List Nat := []) : List Nat :=
  leading ++ [cfg.dataDim + 2 * cfg.latentDim]

/-- Scalar-score shape with arbitrary leading dimensions. -/
abbrev Config.scoreShape (_cfg : Config)
    (leading : List Nat := []) : List Nat :=
  leading ++ [1]

/-- Autoencoder: `x -> hidden -> latent -> hidden -> reconstruction`. -/
def autoencoder (cfg : Config) (leading : List Nat := []) :
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

/-- Supervised bottleneck network producing a reconstruction and two auxiliary latent vectors. -/
def reconstructionWithLatentStatistics (cfg : Config) (leading : List Nat := []) :
    nn.Builder
      (nn.Sequential (cfg.dataShape leading) (cfg.reconstructionStatisticsShape leading)) :=
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

/-- Autoencoder with a narrow continuous bottleneck bounded by `tanh`. -/
def tanhBottleneckAutoencoder (cfg : Config) (leading : List Nat := []) :
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
def ganGenerator (cfg : Config) (leading : List Nat := []) :
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
    (leading : List Nat := []) :
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
