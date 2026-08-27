/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded

/-!
# Residual Convolutional Classifier

The model is polymorphic in its leading dimensions and spatial rank. Residual branches operate on
a common typed shape, and global average pooling reduces every spatial axis before the classifier
head.
-/

@[expose] public section

namespace TorchLean

namespace nn
namespace models

/-- Configuration for a residual classifier over `d` spatial axes. -/
structure ResNetConfig (d : Nat) where
  /-- Number of channels in each input sample. -/
  inChannels : Nat
  /-- Extent of each spatial axis. -/
  spatial : Tensor Nat [d]
  /-- Spatial axes are nonempty, as required by global average pooling. -/
  spatialNonzero : ∀ i : Fin d, spatial.getScalar i ≠ 0
  /-- Channel width used by the residual trunk. -/
  hiddenChannels : Nat
  /-- Geometry used by the stem and residual convolutions. -/
  block : ConvGeometry d
  /-- The configured convolutions preserve the residual trunk's spatial extent. -/
  blockPreservesSpatial :
    Spec.convOutSpatial spatial block.kernel block.stride block.padding = spatial
  /-- Number of classifier logits per sample. -/
  numClasses : Nat

namespace ResNetConfig

/-- Input shape with arbitrary leading dimensions. -/
def inputShape {d : Nat} (cfg : ResNetConfig d) (leading : List Nat := []) : List Nat :=
  leading ++ cfg.inChannels :: cfg.spatial.toList

/-- Activation shape shared by the residual branches. -/
def hiddenShape {d : Nat} (cfg : ResNetConfig d) (leading : List Nat := []) : List Nat :=
  leading ++ cfg.hiddenChannels :: cfg.spatial.toList

/-- Classifier output shape with the same leading dimensions as the input. -/
abbrev outputShape {d : Nat} (cfg : ResNetConfig d) (leading : List Nat := []) : List Nat :=
  leading ++ [cfg.numClasses]

end ResNetConfig

/-- Build a convolutional stem, two residual blocks, global pooling, and a linear classifier. -/
def resnet {d : Nat} (cfg : ResNetConfig d) (leading : List Nat := [])
    (hInChannels : cfg.inChannels ≠ 0 := by decide)
    (hHiddenChannels : cfg.hiddenChannels ≠ 0 := by decide) :
    Builder (Sequential (cfg.inputShape leading) (cfg.outputShape leading)) :=
  letI : NeZero cfg.inChannels := ⟨hInChannels⟩
  letI : NeZero cfg.hiddenChannels := ⟨hHiddenChannels⟩
  let stemRaw := conv leading (inChannels := cfg.inChannels) cfg.spatial
    (cfg.block.toConv cfg.hiddenChannels)
  let stem : Builder (Sequential (cfg.inputShape leading) (cfg.hiddenShape leading)) := by
    simpa [ResNetConfig.inputShape, ResNetConfig.hiddenShape, ConvGeometry.toConv,
      ConvGeometry.outSpatial, cfg.blockPreservesSpatial] using stemRaw
  let hiddenConvRaw := conv leading (inChannels := cfg.hiddenChannels) cfg.spatial
    (cfg.block.toConv cfg.hiddenChannels)
  let hiddenConv : Builder (Sequential (cfg.hiddenShape leading) (cfg.hiddenShape leading)) := by
    simpa [ResNetConfig.hiddenShape, ConvGeometry.toConv,
      ConvGeometry.outSpatial, cfg.blockPreservesSpatial] using hiddenConvRaw
  let residualBranch := do
    let branch ← nn.Sequential![hiddenConv, relu, hiddenConv]
    return blocks.residual branch
  let pooling := globalAvgPool leading
    (channels := cfg.hiddenChannels) cfg.spatial cfg.spatialNonzero
  nn.Sequential![
    stem,
    relu,
    residualBranch,
    relu,
    residualBranch,
    relu,
    pooling,
    linear cfg.hiddenChannels cfg.numClasses (leading := leading)
  ]

end models
end nn
end TorchLean
