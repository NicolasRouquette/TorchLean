/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded

/-!
# Transformer Models

Small config-style Transformer constructors for runnable examples.
-/

@[expose] public section

namespace TorchLean


open Spec Tensor

namespace nn
namespace models

/-- Configuration for a single Transformer encoder block over token embeddings. -/
structure TransformerEncoderConfig where
  /-- Number of tokens in each sequence. -/
  seqLen : Nat
  /-- Width of each token embedding. -/
  dModel : Nat
  /-- Number of attention heads. -/
  numHeads : Nat
  /-- Width of each attention head. -/
  headDim : Nat
  /-- Hidden width of the feed-forward sublayer. -/
  ffnHidden : Nat
  /-- Pointwise activation in the feed-forward sublayer. -/
  activation : _root_.Activation.Kind := .gelu
deriving Repr

namespace TransformerEncoderConfig

/-- Token-embedding shape with arbitrary leading dimensions. -/
abbrev shape (cfg : TransformerEncoderConfig)
    (leading : Spec.Shape := .scalar) : Spec.Shape :=
  (leading.concat (.dim cfg.seqLen .scalar)).appendDim cfg.dModel

end TransformerEncoderConfig

/-- Build one Transformer encoder block over arbitrary leading dimensions. -/
def transformerEncoder (cfg : TransformerEncoderConfig) (leading : Spec.Shape := .scalar)
    (h_seqLen : cfg.seqLen ≠ 0 := by decide)
    (h_dModel : cfg.dModel ≠ 0 := by decide) :
    nn.Builder (nn.Sequential (cfg.shape leading) (cfg.shape leading)) :=
  letI : NeZero cfg.seqLen := ⟨h_seqLen⟩
  letI : NeZero cfg.dModel := ⟨h_dModel⟩
  nn.transformerEncoderBlock leading
    { numHeads := cfg.numHeads
      headDim := cfg.headDim
      ffnHidden := cfg.ffnHidden
      activation := cfg.activation
      dropout? := none }

end models
end nn

end TorchLean
