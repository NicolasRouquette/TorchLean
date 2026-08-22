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
  spatial : Vector Nat d
  /-- Spatial axes are nonempty, as required by global average pooling. -/
  spatialNonzero : ∀ i : Fin d, spatial.get i ≠ 0
  /-- Channel width used by the residual trunk. -/
  hiddenChannels : Nat
  /-- Number of classifier logits per sample. -/
  numClasses : Nat

namespace ResNetConfig

/-- Input shape with arbitrary leading dimensions. -/
def inputShape {d : Nat} (cfg : ResNetConfig d) (leading : Spec.Shape := .scalar) : Spec.Shape :=
  leading.concat (Spec.Shape.ofList (cfg.inChannels :: cfg.spatial.toList))

/-- Activation shape shared by the residual branches. -/
def hiddenShape {d : Nat} (cfg : ResNetConfig d) (leading : Spec.Shape := .scalar) : Spec.Shape :=
  leading.concat (Spec.Shape.ofList (cfg.hiddenChannels :: cfg.spatial.toList))

/-- Classifier output shape with the same leading dimensions as the input. -/
abbrev outputShape {d : Nat} (cfg : ResNetConfig d) (leading : Spec.Shape := .scalar) : Spec.Shape :=
  leading.appendDim cfg.numClasses

end ResNetConfig

namespace Internal

/-- Shape-preserving convolution used by the residual trunk. -/
def sameSpatialConv {d : Nat} (leading : Spec.Shape) (spatial : Vector Nat d)
    (spatialNonzero : ∀ i : Fin d, spatial.get i ≠ 0)
    (inChannels outChannels : Nat) [NeZero inChannels] :
    Builder (Sequential
      (leading.concat (Spec.Shape.ofList (inChannels :: spatial.toList)))
      (leading.concat (Spec.Shape.ofList (outChannels :: spatial.toList)))) :=
  let layer := conv (leading := leading)
      (inChannels := inChannels) spatial
      { outChannels := outChannels
        kernel := Vector.replicate d 1
        stride := Vector.replicate d 1
        padding := Vector.replicate d 0
        kernelNonzero := by intro i; simp [Vector.get]
        strideNonzero := by intro i; simp [Vector.get] }
  by
      simpa [Spec.Shape.concat, Spec.convOutSpatial_unit spatial spatialNonzero] using layer

end Internal

/-- Build a convolutional stem, two residual blocks, global pooling, and a linear classifier. -/
def resnet {d : Nat} (cfg : ResNetConfig d) (leading : Spec.Shape := .scalar)
    (hInChannels : cfg.inChannels ≠ 0 := by decide)
    (hHiddenChannels : cfg.hiddenChannels ≠ 0 := by decide) :
    Builder (Sequential (cfg.inputShape leading) (cfg.outputShape leading)) :=
  letI : NeZero cfg.inChannels := ⟨hInChannels⟩
  letI : NeZero cfg.hiddenChannels := ⟨hHiddenChannels⟩
  let stem :=
    Internal.sameSpatialConv leading cfg.spatial cfg.spatialNonzero
      cfg.inChannels cfg.hiddenChannels
  let hiddenConv :=
    Internal.sameSpatialConv leading cfg.spatial cfg.spatialNonzero
      cfg.hiddenChannels cfg.hiddenChannels
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
