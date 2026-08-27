/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded

/-!
# Convolutional Classifier

The classifier is polymorphic in both its leading dimensions and the number of spatial axes.
Convolution and pooling use the same vector-valued configuration for signals, images, volumes,
and higher-dimensional data.
-/

@[expose] public section

namespace TorchLean

namespace nn
namespace models

/-- Configuration for a compact convolutional classifier. -/
structure CnnConfig (d : Nat) where
  /-- Number of channels in each input sample. -/
  inChannels : Nat
  /-- Extent of each spatial axis. -/
  spatial : Tensor Nat [d]
  /-- Number of classifier outputs per sample. -/
  outDim : Nat
  /-- Convolution stage. -/
  conv : Conv d
  /-- Max-pooling stage. -/
  pool : Pool d

/-- Spatial extent after convolution. -/
def CnnConfig.afterConv {d : Nat} (cfg : CnnConfig d) : Tensor Nat [d] :=
  Spec.convOutSpatial cfg.spatial cfg.conv.kernel cfg.conv.stride cfg.conv.padding

/-- Spatial extent after pooling. -/
def CnnConfig.afterPool {d : Nat} (cfg : CnnConfig d) : Tensor Nat [d] :=
  Spec.poolOutSpatialPad cfg.afterConv cfg.pool.kernel cfg.pool.stride cfg.pool.padding

/-- Number of features presented to the classifier head. -/
def CnnConfig.featureCount {d : Nat} (cfg : CnnConfig d) : Nat :=
  (cfg.conv.outChannels :: cfg.afterPool.toList).prod

namespace CnnConfig

/-- Input shape with arbitrary leading dimensions. -/
def inputShape {d : Nat} (cfg : CnnConfig d) (leading : List Nat := []) : List Nat :=
  leading ++ cfg.inChannels :: cfg.spatial.toList

/-- Classifier output shape with the same leading dimensions as the input. -/
def outputShape {d : Nat} (cfg : CnnConfig d) (leading : List Nat := []) : List Nat :=
  leading ++ [cfg.outDim]

end CnnConfig

/-- Build `convolution -> activation -> max pool -> flatten -> linear`. -/
def cnn {d : Nat} (cfg : CnnConfig d) (leading : List Nat := [])
    (hInChannels : cfg.inChannels ≠ 0 := by decide) :
    Builder (Sequential (cfg.inputShape leading) (cfg.outputShape leading)) :=
  letI : NeZero cfg.inChannels := ⟨hInChannels⟩
  let convolution := conv (leading := leading) cfg.spatial cfg.conv
  let pooling := maxPool (leading := leading) cfg.afterConv cfg.pool
  nn.Sequential![
    convolution,
    relu,
    pooling,
    flattenAfter leading,
    linear cfg.featureCount cfg.outDim (leading := leading)
  ]

end models
end nn
end TorchLean
