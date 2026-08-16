/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded

/-!
# Convolutional Classifier

The classifier is polymorphic in the number of spatial axes. Its input has shape
`(batch, channels, spatial...)`; convolution and pooling use the same vector-valued configuration
for signals, images, volumes, and higher-dimensional data.
-/

@[expose] public section

namespace TorchLean

namespace nn
namespace models

/-- Configuration for a compact convolutional classifier. -/
structure CNNConfig (d : Nat) where
  /-- Number of independent samples processed together. -/
  batch : Nat
  /-- Number of channels in each input sample. -/
  inChannels : Nat
  /-- Extent of each spatial axis. -/
  spatial : Vector Nat d
  /-- Number of classifier outputs per sample. -/
  outDim : Nat
  /-- Convolution stage. -/
  conv : Conv d
  /-- Max-pooling stage. -/
  pool : Pool d

/-- Spatial extent after convolution. -/
def CNNConfig.afterConv {d : Nat} (cfg : CNNConfig d) : Vector Nat d :=
  Spec.convOutSpatial cfg.spatial cfg.conv.kernel cfg.conv.stride cfg.conv.padding

/-- Spatial extent after pooling. -/
def CNNConfig.afterPool {d : Nat} (cfg : CNNConfig d) : Vector Nat d :=
  Spec.poolOutSpatialPad cfg.afterConv cfg.pool.kernel cfg.pool.stride cfg.pool.padding

/-- Number of features presented to the classifier head. -/
def CNNConfig.featureCount {d : Nat} (cfg : CNNConfig d) : Nat :=
  Spec.Shape.size (Spec.Shape.ofList (cfg.conv.outChannels :: cfg.afterPool.toList))

/-- Input tensor shape `(batch, inChannels, spatial...)`. -/
def cnnInShape {d : Nat} (cfg : CNNConfig d) : Spec.Shape :=
  .dim cfg.batch (Spec.Shape.ofList (cfg.inChannels :: cfg.spatial.toList))

/-- Classifier output shape `(batch, outDim)`. -/
def cnnOutShape {d : Nat} (cfg : CNNConfig d) : Spec.Shape :=
  .dim cfg.batch (.dim cfg.outDim .scalar)

/-- Build `convolution -> activation -> max pool -> flatten -> linear`. -/
def cnn {d : Nat} (cfg : CNNConfig d) (hInChannels : cfg.inChannels ≠ 0 := by decide) :
    Builder (Sequential (cnnInShape cfg) (cnnOutShape cfg)) :=
  letI : NeZero cfg.inChannels := ⟨hInChannels⟩
  let convolution := conv (leading := .dim cfg.batch .scalar) cfg.spatial cfg.conv
  let pooling := maxPool (leading := .dim cfg.batch .scalar) cfg.afterConv cfg.pool
  nn.Sequential![
    convolution,
    relu,
    pooling,
    flattenLeading (.dim cfg.batch .scalar),
    linear cfg.featureCount cfg.outDim (pfx := .dim cfg.batch .scalar)
  ]

end models
end nn
end TorchLean
