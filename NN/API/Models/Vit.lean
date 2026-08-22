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
structure VitConfig (d : Nat) where
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
def VitConfig.patchSpatial {d : Nat} (cfg : VitConfig d) : Vector Nat d :=
  Spec.convOutSpatial cfg.spatial cfg.patch.kernel cfg.patch.stride cfg.patch.padding

/-- Number of patch tokens. -/
def VitConfig.seqLen {d : Nat} (cfg : VitConfig d) : Nat :=
  Spec.Shape.size (Spec.Shape.ofList cfg.patchSpatial.toList)

/-- Number of flattened features passed to the classifier. -/
def VitConfig.flatDim {d : Nat} (cfg : VitConfig d) : Nat :=
  Spec.Shape.size (.dim cfg.seqLen (.dim cfg.patch.outChannels .scalar))

/-- Input shape after prepending the axes mapped independently by the model. -/
abbrev VitConfig.inputShape {d : Nat} (cfg : VitConfig d)
    (leading : Spec.Shape := .scalar) : Spec.Shape :=
  leading.concat (Spec.Shape.ofList (cfg.inChannels :: cfg.spatial.toList))

/-- Shape produced by patch embedding. -/
abbrev VitConfig.patchShape {d : Nat} (cfg : VitConfig d)
    (leading : Spec.Shape := .scalar) : Spec.Shape :=
  leading.concat (Spec.Shape.ofList (cfg.patch.outChannels :: cfg.patchSpatial.toList))

/-- Token shape obtained by flattening the patch grid and moving channels to the final axis. -/
abbrev VitConfig.tokenShape {d : Nat} (cfg : VitConfig d)
    (leading : Spec.Shape := .scalar) : Spec.Shape :=
  leading.concat (.dim cfg.seqLen (.dim cfg.patch.outChannels .scalar))

/-- Classifier output shape after preserving every leading axis. -/
abbrev VitConfig.outputShape {d : Nat} (cfg : VitConfig d)
    (leading : Spec.Shape := .scalar) : Spec.Shape :=
  leading.appendDim cfg.outDim

/-- Flatten the patch grid into a sequence and move channels to the final axis. -/
def spatialToTokens {d : Nat} (cfg : VitConfig d) (leading : Spec.Shape := .scalar) :
    Layer (cfg.patchShape leading) (cfg.tokenShape leading) :=
  { kind := "SpatialToTokens"
    stateShapes := []
    initState := .nil
    requiresGrad := []
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => (show m (_root_.Runtime.Autograd.TorchLean.RefTy
            (m := m) (α := α) (cfg.tokenShape leading)) from do
          let middle : Spec.Shape :=
            leading.concat (.dim cfg.patch.outChannels (.dim cfg.seqLen .scalar))
          let flattened ←
            _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
              (s₁ := cfg.patchShape leading) (s₂ := middle) x (by
                have hInner :
                    Spec.Shape.size
                        (Spec.Shape.ofList (cfg.patch.outChannels :: cfg.patchSpatial.toList)) =
                      cfg.patch.outChannels * cfg.seqLen := by
                  simp [VitConfig.seqLen, Spec.Shape.ofList, Spec.Shape.size]
                simp [VitConfig.patchShape, middle, Spec.Shape.size_concat,
                  Spec.Shape.size, hInner])
          let tokens ← _root_.Runtime.Autograd.Torch.swapAdjacentAtDepth
            (m := m) (α := α) (s := middle) leading.rank flattened
          return (by simpa [middle, VitConfig.tokenShape] using tokens)) }

/--
Build the patch and Transformer portion of a vision transformer.

The result retains one token per patch. Classification, reconstruction, and other tasks can attach
their own heads without rebuilding the patch pipeline.
-/
def vitEncoder {d : Nat} (cfg : VitConfig d) (leading : Spec.Shape := .scalar)
    (hInChannels : cfg.inChannels ≠ 0 := by decide)
    (hSeqLen : cfg.seqLen ≠ 0 := by decide)
    (hModel : cfg.patch.outChannels ≠ 0 := by decide) :
    Builder (Sequential (cfg.inputShape leading) (cfg.tokenShape leading)) :=
  letI : NeZero cfg.inChannels := ⟨hInChannels⟩
  letI : NeZero cfg.seqLen := ⟨hSeqLen⟩
  letI : NeZero cfg.patch.outChannels := ⟨hModel⟩
  do
    let patchEmbedding ←
      conv (leading := leading) (inChannels := cfg.inChannels) cfg.spatial cfg.patch
    let encoderRaw ← transformerEncoderBlock leading
      { numHeads := cfg.numHeads
        headDim := cfg.headDim
        ffnHidden := cfg.ffnHidden
        activation := .gelu
        dropout? := none }
    let encoder : Sequential (cfg.tokenShape leading) (cfg.tokenShape leading) := by
      simpa only [VitConfig.tokenShape, Spec.Shape.concat_appendDim, Spec.Shape.appendDim] using
        encoderRaw
    pure <| patchEmbedding >>> of (spatialToTokens cfg leading) >>> encoder

/-- Build a vision Transformer encoder followed by a linear classifier. -/
def vit {d : Nat} (cfg : VitConfig d) (leading : Spec.Shape := .scalar)
    (hInChannels : cfg.inChannels ≠ 0 := by decide)
    (hSeqLen : cfg.seqLen ≠ 0 := by decide)
    (hModel : cfg.patch.outChannels ≠ 0 := by decide) :
    Builder (Sequential (cfg.inputShape leading) (cfg.outputShape leading)) := do
  let encoder ← vitEncoder cfg leading hInChannels hSeqLen hModel
  let flatten ← flattenLeading leading
    (s := .dim cfg.seqLen (.dim cfg.patch.outChannels .scalar))
  let classifier ← linear cfg.flatDim cfg.outDim (leading := leading)
  pure <| encoder >>> flatten >>> classifier

end models
end nn
end TorchLean
