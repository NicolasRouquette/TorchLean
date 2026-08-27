/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Shape
public import NN.API.Runtime
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

/-- How a vision Transformer turns encoded tokens into one vector per sample. -/
inductive VitPooling where
  /-- Average every encoded patch token. -/
  | mean
  /-- Prepend a learned class slot and select it after the encoder. -/
  | cls
deriving DecidableEq, Repr

/-- Configuration for a Transformer over patches from a `d`-dimensional spatial domain. -/
structure VitConfig (d : Nat) where
  /-- Number of channels in each input sample. -/
  inChannels : Nat
  /-- Extent of each spatial axis. -/
  spatial : Tensor Nat [d]
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
  /-- Number of Transformer encoder blocks. -/
  numLayers : Nat := 1
  /-- Reduction used by the classification head. -/
  pooling : VitPooling := .mean

/-- Spatial extent of the patch embedding. -/
def VitConfig.patchSpatial {d : Nat} (cfg : VitConfig d) : Tensor Nat [d] :=
  Spec.convOutSpatial cfg.spatial cfg.patch.kernel cfg.patch.stride cfg.patch.padding

/-- Number of patch tokens. -/
def VitConfig.seqLen {d : Nat} (cfg : VitConfig d) : Nat :=
  cfg.patchSpatial.toList.prod

/-- Number of slots inserted before the patch sequence. -/
def VitConfig.prefixLen {d : Nat} (cfg : VitConfig d) : Nat :=
  match cfg.pooling with
  | .mean => 0
  | .cls => 1

/-- Number of tokens passed through the Transformer encoder. -/
def VitConfig.encodedSeqLen {d : Nat} (cfg : VitConfig d) : Nat :=
  cfg.prefixLen + cfg.seqLen

/-- Number of scalar features in the complete encoded token sequence. -/
def VitConfig.flatDim {d : Nat} (cfg : VitConfig d) : Nat :=
  cfg.encodedSeqLen * cfg.patch.outChannels

/-- Input shape after prepending the axes mapped independently by the model. -/
abbrev VitConfig.inputShape {d : Nat} (cfg : VitConfig d)
    (leading : List Nat := []) : List Nat :=
  leading ++ cfg.inChannels :: cfg.spatial.toList

/-- Shape produced by patch embedding. -/
abbrev VitConfig.patchShape {d : Nat} (cfg : VitConfig d)
    (leading : List Nat := []) : List Nat :=
  leading ++ cfg.patch.outChannels :: cfg.patchSpatial.toList

/-- Token shape obtained by flattening the patch grid and moving channels to the final axis. -/
abbrev VitConfig.tokenShape {d : Nat} (cfg : VitConfig d)
    (leading : List Nat := []) : List Nat :=
  leading ++ [cfg.seqLen, cfg.patch.outChannels]

/-- Token shape after adding the optional class slot. -/
abbrev VitConfig.encodedTokenShape {d : Nat} (cfg : VitConfig d)
    (leading : List Nat := []) : List Nat :=
  leading ++ [cfg.encodedSeqLen, cfg.patch.outChannels]

/-- Classifier output shape after preserving every leading axis. -/
abbrev VitConfig.outputShape {d : Nat} (cfg : VitConfig d)
    (leading : List Nat := []) : List Nat :=
  leading ++ [cfg.outDim]

namespace Internal

/-- Implementation layer that turns a patch grid into a token sequence. -/
def spatialToTokens {d : Nat} (cfg : VitConfig d) (leading : List Nat := []) :
    Layer (cfg.patchShape leading) (cfg.tokenShape leading) :=
  { kind := "SpatialToTokens"
    stateShapes := []
    initState := .nil
    requiresGrad := #[]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x => (show m (_root_.TorchLean.Runtime.ValueRef
            (m := m) (α := α) (cfg.tokenShape leading)) from do
          let middle : Shape :=
            Shape.ofList (leading ++ [cfg.patch.outChannels, cfg.seqLen])
          let flattened ←
            _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
              (s₁ := cfg.patchShape leading) (s₂ := middle) x (by
                change Shape.size (Shape.ofList
                    (leading ++ cfg.patch.outChannels :: cfg.patchSpatial.toList)) =
                  Shape.size (Shape.ofList
                    (leading ++ [cfg.patch.outChannels, cfg.seqLen]))
                simp only [_root_.Spec.Shape.size_ofList, List.prod_append,
                  List.prod_cons, List.prod_nil, Nat.mul_one, VitConfig.seqLen])
          let tokens ← _root_.Runtime.Autograd.Torch.swapAdjacentAtDepth
            (m := m) (α := α) (s := middle) (Shape.ofList leading).rank flattened
          return (by
            simpa [middle, VitConfig.tokenShape, Shape.ofList_append] using tokens)) }

/-- Implementation layer that prepends the optional class-token slot. -/
def prependVitPrefix {d : Nat} (cfg : VitConfig d) (leading : List Nat := []) :
    Sequential (cfg.tokenShape leading) (cfg.encodedTokenShape leading) :=
  let prefixShape : Shape :=
    [cfg.prefixLen, cfg.patch.outChannels]
  let core : Sequential
      [cfg.seqLen, cfg.patch.outChannels]
      [cfg.encodedSeqLen, cfg.patch.outChannels] :=
    nn.of
      { kind := "PrependVitPrefix"
        stateShapes := []
        initState := .nil
        requiresGrad := #[]
        forward := fun _ {α} _ _ =>
          fun {m} _ _ =>
            fun x =>
              (show m (_root_.TorchLean.Runtime.ValueRef
                  (m := m) (α := α)
                  [cfg.encodedSeqLen, cfg.patch.outChannels]) from do
              let zeroPrefix ← _root_.Runtime.Autograd.Torch.const
                (m := m) (α := α) (Spec.zeros α prefixShape)
              let result ← _root_.Runtime.Autograd.Torch.concatLeadingAxis
                (m := m) (α := α) zeroPrefix x
              return (by
                simpa [VitConfig.encodedSeqLen, prefixShape] using result)) }
  by
    simpa only [VitConfig.tokenShape, VitConfig.encodedTokenShape,
      Shape.ofList_append] using
      nn.Internal.mapEach (Shape.ofList leading) core

/-- Implementation layer that moves the embedding width before the token axis. -/
def tokensToChannels {d : Nat} (cfg : VitConfig d) (leading : List Nat := []) :
    Sequential
      (cfg.encodedTokenShape leading)
      (leading ++ [cfg.patch.outChannels, cfg.encodedSeqLen]) :=
  nn.of
    { kind := "TokensToChannels"
      stateShapes := []
      initState := .nil
      requiresGrad := #[]
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun x =>
            (show m (_root_.TorchLean.Runtime.ValueRef
                (m := m) (α := α)
                (leading ++ [cfg.patch.outChannels, cfg.encodedSeqLen])) from by
              simpa [VitConfig.encodedTokenShape, Shape.ofList_append] using
                (_root_.Runtime.Autograd.Torch.swapAdjacentAtDepth
                  (m := m) (α := α) (s := cfg.encodedTokenShape leading)
                    (Shape.ofList leading).rank x)) }

/-- Implementation layer for mean-pool classification. -/
def meanVitTokens {d : Nat} (cfg : VitConfig d) (leading : List Nat := [])
    (hEncodedSeqLen : cfg.encodedSeqLen ≠ 0) :
    Sequential (cfg.encodedTokenShape leading)
      (leading ++ [cfg.patch.outChannels]) :=
  let spatial : Tensor Nat [1] :=
    Spec.Tensor.ofArrayExact #[cfg.encodedSeqLen] (by simp)
  let hSpatial : ∀ i : Fin 1, spatial.getScalar i ≠ 0 := by
    intro i
    fin_cases i
    simpa [spatial] using hEncodedSeqLen
  let pool : Sequential
      (leading ++ [cfg.patch.outChannels, cfg.encodedSeqLen])
      (leading ++ [cfg.patch.outChannels]) := by
    simpa [spatial, Shape.ofList_append, Shape.appendDim_eq_concat] using
      nn.Internal.globalAvgPool leading
        (channels := cfg.patch.outChannels) spatial hSpatial
  seq! tokensToChannels cfg leading, pool

/-- Implementation layer for class-token classification. -/
def firstVitToken {d : Nat} (cfg : VitConfig d) (leading : List Nat := [])
    (hEncodedSeqLen : cfg.encodedSeqLen ≠ 0) :
    Sequential (cfg.encodedTokenShape leading)
      (leading ++ [cfg.patch.outChannels]) :=
  let core : Sequential
      [cfg.encodedSeqLen, cfg.patch.outChannels]
      [cfg.patch.outChannels] :=
    nn.of
      { kind := "FirstVitToken"
        stateShapes := []
        initState := .nil
        requiresGrad := #[]
        forward := fun _ {α} _ _ =>
          fun {m} _ _ =>
          fun x =>
            (show m (_root_.TorchLean.Runtime.ValueRef
                (m := m) (α := α) [cfg.patch.outChannels]) from do
              have hOne : 0 + 1 ≤ cfg.encodedSeqLen := by
                simpa using Nat.one_le_iff_ne_zero.mpr hEncodedSeqLen
              let row ← _root_.Runtime.Autograd.Torch.sliceLeadingAxisRange
                (m := m) (α := α) 0 1 hOne x
              _root_.Runtime.Autograd.Torch.reshape
                (m := m) (α := α)
                (s₁ := [1, cfg.patch.outChannels])
                (s₂ := [cfg.patch.outChannels])
                row (by simp [Shape.size])) }
  by
    simpa only [VitConfig.encodedTokenShape, Shape.ofList_append,
      Shape.appendDim_eq_concat] using
      nn.Internal.mapEach (Shape.ofList leading) core

end Internal

/--
Build the patch and Transformer portion of a vision transformer.

The result retains one token per patch. Classification, reconstruction, and other tasks can attach
their own heads without rebuilding the patch pipeline.
-/
def vitEncoder {d : Nat} (cfg : VitConfig d) (leading : List Nat := [])
    (hInChannels : cfg.inChannels ≠ 0 := by decide)
    (hSeqLen : cfg.seqLen ≠ 0 := by decide)
    (hModel : cfg.patch.outChannels ≠ 0 := by decide) :
    Builder (Sequential (cfg.inputShape leading) (cfg.encodedTokenShape leading)) :=
  letI : NeZero cfg.inChannels := ⟨hInChannels⟩
  letI : NeZero cfg.seqLen := ⟨hSeqLen⟩
  letI : NeZero cfg.patch.outChannels := ⟨hModel⟩
  let hEncodedSeqLen : cfg.encodedSeqLen ≠ 0 := by
    cases hPooling : cfg.pooling <;>
      simp [VitConfig.encodedSeqLen, VitConfig.prefixLen, hPooling, hSeqLen]
  letI : NeZero cfg.encodedSeqLen := ⟨hEncodedSeqLen⟩
  do
    let patchEmbedding ←
      conv (leading := leading) (inChannels := cfg.inChannels) cfg.spatial cfg.patch
    let positions ← learnedPositionalEmbedding leading
      (seqLen := cfg.encodedSeqLen) (embedDim := cfg.patch.outChannels)
    let encoderRaw ← transformerEncoderStack leading
      { layers := cfg.numLayers
        block :=
          { numHeads := cfg.numHeads
            headDim := cfg.headDim
            ffnHidden := cfg.ffnHidden
            activation := .gelu
            dropout? := none } }
    let encoder : Sequential
        (cfg.encodedTokenShape leading) (cfg.encodedTokenShape leading) := by
      simpa only [VitConfig.encodedTokenShape, Shape.ofList_append] using encoderRaw
    pure <| patchEmbedding >>> of (Internal.spatialToTokens cfg leading) >>>
      Internal.prependVitPrefix cfg leading >>> positions >>> encoder

/-- Build a vision Transformer encoder followed by a linear classifier. -/
def vit {d : Nat} (cfg : VitConfig d) (leading : List Nat := [])
    (hInChannels : cfg.inChannels ≠ 0 := by decide)
    (hSeqLen : cfg.seqLen ≠ 0 := by decide)
    (hModel : cfg.patch.outChannels ≠ 0 := by decide) :
    Builder (Sequential (cfg.inputShape leading) (cfg.outputShape leading)) := do
  let encoder ← vitEncoder cfg leading hInChannels hSeqLen hModel
  let hEncodedSeqLen : cfg.encodedSeqLen ≠ 0 := by
    cases hPooling : cfg.pooling <;>
      simp [VitConfig.encodedSeqLen, VitConfig.prefixLen, hPooling, hSeqLen]
  let pool : Sequential
      (cfg.encodedTokenShape leading)
      (leading ++ [cfg.patch.outChannels]) :=
    match hPooling : cfg.pooling with
    | .mean => Internal.meanVitTokens cfg leading hEncodedSeqLen
    | .cls => Internal.firstVitToken cfg leading hEncodedSeqLen
  let classifier ← linear cfg.patch.outChannels cfg.outDim (leading := leading)
  pure <| encoder >>> pool >>> classifier

end models
end nn
end TorchLean
