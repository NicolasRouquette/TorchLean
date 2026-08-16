/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded

/-!
# Vision Transformer

Patch embedding is an arbitrary-dimensional convolution. The spatial output is flattened into a
token axis before the Transformer block, so the construction applies equally to one-dimensional
signals, images, volumes, and higher-dimensional grids.
-/

@[expose] public section

namespace TorchLean

namespace nn
namespace models

/-- Configuration for a Transformer over patches from a `d`-dimensional spatial domain. -/
structure ViTConfig (d : Nat) where
  /-- Number of independent samples processed together. -/
  batch : Nat
  /-- Number of channels in each input sample. -/
  inChannels : Nat
  /-- Extent of each spatial axis. -/
  spatial : Vector Nat d
  /-- Convolution that extracts and embeds patches. -/
  patch : Conv d
  /-- Number of classifier outputs per sample. -/
  outDim : Nat
  /-- Number of attention heads. -/
  numHeads : Nat
  /-- Width of each attention head. -/
  headDim : Nat
  /-- Width of the feed-forward sublayer. -/
  ffnHidden : Nat

/-- Spatial extent of the patch embedding. -/
def ViTConfig.patchSpatial {d : Nat} (cfg : ViTConfig d) : Vector Nat d :=
  Spec.convOutSpatial cfg.spatial cfg.patch.kernel cfg.patch.stride cfg.patch.padding

/-- Number of patch tokens. -/
def ViTConfig.seqLen {d : Nat} (cfg : ViTConfig d) : Nat :=
  Spec.Shape.size (Spec.Shape.ofList cfg.patchSpatial.toList)

/-- Number of flattened features passed to the classifier. -/
def ViTConfig.flatDim {d : Nat} (cfg : ViTConfig d) : Nat :=
  Spec.Shape.size (.dim cfg.seqLen (.dim cfg.patch.outChannels .scalar))

/-- Input shape `(batch, inChannels, spatial...)`. -/
def vitInShape {d : Nat} (cfg : ViTConfig d) : Spec.Shape :=
  .dim cfg.batch (Spec.Shape.ofList (cfg.inChannels :: cfg.spatial.toList))

/-- Shape produced by patch embedding. -/
def vitConvOutShape {d : Nat} (cfg : ViTConfig d) : Spec.Shape :=
  .dim cfg.batch (Spec.Shape.ofList (cfg.patch.outChannels :: cfg.patchSpatial.toList))

/-- Token shape `(batch, sequence, embedding)`. -/
def vitTokensShape {d : Nat} (cfg : ViTConfig d) : Spec.Shape :=
  .dim cfg.batch (.dim cfg.seqLen (.dim cfg.patch.outChannels .scalar))

/-- Classifier output shape `(batch, outDim)`. -/
def vitOutShape {d : Nat} (cfg : ViTConfig d) : Spec.Shape :=
  .dim cfg.batch (.dim cfg.outDim .scalar)

/-- flatten the patch grid into a sequence and move channels to the final axis. -/
def spatialToTokens {d : Nat} (cfg : ViTConfig d) :
    Layer (vitConvOutShape cfg) (vitTokensShape cfg) :=
  { kind := "SpatialToTokens"
    stateShapes := []
    initState := .nil
    requiresGrad := []
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => (show m (_root_.Runtime.Autograd.TorchLean.RefTy
            (m := m) (α := α) (vitTokensShape cfg)) from do
          let middle : Spec.Shape :=
            .dim cfg.batch (.dim cfg.patch.outChannels (.dim cfg.seqLen .scalar))
          let flattened ←
            _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
              (s₁ := vitConvOutShape cfg) (s₂ := middle) x (by
                have hInner :
                    Spec.Shape.size
                        (Spec.Shape.ofList (cfg.patch.outChannels :: cfg.patchSpatial.toList)) =
                      cfg.patch.outChannels * cfg.seqLen := by
                  simp [ViTConfig.seqLen, Spec.Shape.ofList, Spec.Shape.size]
                simp [vitConvOutShape, middle, Spec.Shape.size, hInner])
          _root_.Runtime.Autograd.Torch.swapAdjacentAtDepth
            (m := m) (α := α) (s := middle) 1 flattened) }

/-- Build patch embedding, one Transformer encoder block, and a linear classifier. -/
def vit {d : Nat} (cfg : ViTConfig d)
    (hInChannels : cfg.inChannels ≠ 0 := by decide)
    (hSeqLen : cfg.seqLen ≠ 0 := by decide)
    (hModel : cfg.patch.outChannels ≠ 0 := by decide) :
    Builder (Sequential (vitInShape cfg) (vitOutShape cfg)) :=
  letI : NeZero cfg.inChannels := ⟨hInChannels⟩
  letI : NeZero cfg.seqLen := ⟨hSeqLen⟩
  letI : NeZero cfg.patch.outChannels := ⟨hModel⟩
  let patchEmbedding :=
    conv (leading := .dim cfg.batch .scalar) cfg.spatial cfg.patch
  nn.Sequential![
    patchEmbedding,
    lift (of (spatialToTokens cfg)),
    transformerEncoderBlock
      { numHeads := cfg.numHeads
        headDim := cfg.headDim
        ffnHidden := cfg.ffnHidden
        activation := .gelu
        dropout? := none },
    flattenLeading (.dim cfg.batch .scalar),
    linear cfg.flatDim cfg.outDim (pfx := .dim cfg.batch .scalar)
  ]

end models
end nn
end TorchLean
