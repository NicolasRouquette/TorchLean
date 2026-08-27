/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded

/-!
# Causal Transformer Architecture

Configuration, shape families, and graph constructors for GPT-style causal Transformers.
-/

@[expose] public section

namespace TorchLean


open Spec Tensor

namespace nn
namespace models
namespace CausalTransformer

/--
Configuration shared by TorchLean's GPT-style causal language models.

The model has the common GPT-2 “shape”:

`embedding → learned positional embedding → (masked self-attention + FFN)×layers → LayerNorm`
`→ linear`

The configuration is independent of how token ids enter the model. One-hot, bounded-token, and
pre-embedded constructors reuse the same Transformer width, depth, and output vocabulary.
-/
structure Config where
  seqLen : Nat
  vocab : Nat
  numHeads : Nat
  headDim : Nat
  ffnHidden : Nat
  layers : Nat
  /-- Feed-forward activation used in every Transformer block. -/
  activation : _root_.Activation.Kind := .gelu
  /-- Dropout probability for attention and feed-forward outputs. -/
  dropout? : Option Float := none
  /-- Use pre-normalized Transformer blocks. -/
  normFirst : Bool := false
  /-- Add a trainable bias after each attention output projection. -/
  attentionOutputBias : Bool := false
  /-- Shared initialization for embedding and projection weights. `none` keeps layer defaults. -/
  parameterInit? : Option _root_.Runtime.Autograd.Torch.Init.Scheme := none
  /--
  Initialization for projections whose outputs are added to residual streams.

  GPT-2 scales these weights by network depth. Keeping the setting explicit lets other causal
  Transformers use their own residual initialization without changing the block implementation.
  -/
  residualProjectionInit? : Option _root_.Runtime.Autograd.Torch.Init.Scheme := none
  /-- Seed stride used when initializing repeated blocks. -/
  seedStride : Nat := 100
deriving Repr

/-- Transformer width implied by `numHeads * headDim`. -/
def Config.dModel (cfg : Config) : Nat :=
  cfg.numHeads * cfg.headDim

/-- Token-id shape `leading ++ seqLen`. -/
abbrev tokenShape (cfg : Config) (leading : List Nat := []) : List Nat :=
  leading ++ [cfg.seqLen]

/-- Vocabulary-grid shape `leading ++ (seqLen × vocab)` used by inputs and output logits. -/
abbrev vocabularyShape (cfg : Config) (leading : List Nat := []) : List Nat :=
  leading ++ [cfg.seqLen, cfg.vocab]

/-- Embedded-token shape `leading ++ (seqLen × dModel)`. -/
abbrev embeddingShape (cfg : Config) (leading : List Nat := []) : List Nat :=
  leading ++ [cfg.seqLen, cfg.dModel]

/--
Causal Transformer hidden-state stack after token embeddings have been computed.

The stack adds learned positions, applies causally masked Transformer blocks, and finishes with
LayerNorm. It deliberately has no vocabulary projection. Language models can therefore choose an
independent output head or reuse their token-embedding matrix.
-/
def hidden (cfg : Config) (leading : List Nat := [])
    (h_seqLen : cfg.seqLen ≠ 0 := by decide)
    (h_dModel : cfg.dModel ≠ 0 := by decide) :
    nn.Builder (nn.Sequential (embeddingShape cfg leading) (embeddingShape cfg leading)) :=
  letI : NeZero cfg.seqLen := ⟨h_seqLen⟩
  letI : NeZero cfg.dModel := ⟨h_dModel⟩
  let dModel := cfg.dModel
  let encCfg : nn.blocks.TransformerEncoderStack :=
    { layers := cfg.layers
      block :=
        { numHeads := cfg.numHeads
          headDim := cfg.headDim
          ffnHidden := cfg.ffnHidden
          activation := cfg.activation
          dropout? := cfg.dropout?
          normFirst := cfg.normFirst
          attentionOutputBias := cfg.attentionOutputBias
          weightInit? := cfg.parameterInit?
          residualOutputInit? := cfg.residualProjectionInit? }
      seedStride := cfg.seedStride }
  let posInit := cfg.parameterInit?.getD (.uniform (-0.02) 0.02)
  do
    let positionalRaw ← nn.learnedPositionalEmbedding leading
      (seqLen := cfg.seqLen) (embedDim := dModel) { posInit := posInit }
    let positional : nn.Sequential (embeddingShape cfg leading) (embeddingShape cfg leading) := by
      simpa only [dModel, embeddingShape, Config.dModel, Shape.concat_appendDim,
        Shape.appendDim] using positionalRaw
    let blocksRaw ← nn.transformerEncoderStack leading
      (n := cfg.seqLen) (dModel := dModel) encCfg
      (mask := some (Spec.causalMask cfg.seqLen))
    let blocks : nn.Sequential (embeddingShape cfg leading) (embeddingShape cfg leading) := by
      simpa only [dModel, embeddingShape, Config.dModel, Shape.concat_appendDim,
        Shape.appendDim] using blocksRaw
    let normalizationRaw ←
      nn.layerNorm (leading ++ [cfg.seqLen]) (width := dModel)
    let normalization :
        nn.Sequential (embeddingShape cfg leading) (embeddingShape cfg leading) := by
      simpa [dModel, embeddingShape, Config.dModel, Shape.ofList_append,
        Shape.appendDim_eq_concat, Shape.concat_assoc] using normalizationRaw
    pure (positional >>> blocks >>> normalization)

/--
GPT-style causal Transformer body with an independent affine vocabulary head.

Use `hidden` when the caller needs hidden states or a tied
token-embedding/output matrix.
-/
def fromEmbeddings (cfg : Config) (leading : List Nat := [])
    (h_seqLen : cfg.seqLen ≠ 0 := by decide)
    (h_dModel : cfg.dModel ≠ 0 := by decide) :
    nn.Builder (nn.Sequential (embeddingShape cfg leading) (vocabularyShape cfg leading)) := do
  let hidden ← hidden cfg leading h_seqLen h_dModel
  let headRaw ← nn.linearWith cfg.dModel cfg.vocab { weightInit? := cfg.parameterInit? }
    (leading := leading ++ [cfg.seqLen])
  let head : nn.Sequential (embeddingShape cfg leading) (vocabularyShape cfg leading) := by
    simpa [embeddingShape, vocabularyShape, Shape.ofList_append,
      Shape.appendDim_eq_concat, Shape.concat_assoc] using headRaw
  pure (hidden >>> head)

/--
Build a GPT-2-style causal language model over one-hot tokens.

This is the shared constructor used by the runnable GPT-2 examples. It stays in `nn.Builder` so it
composes with the rest of the API-layer model-building interface.
-/
def oneHot (cfg : Config) (leading : List Nat := [])
    (h_seqLen : cfg.seqLen ≠ 0 := by decide)
    (h_dModel : cfg.dModel ≠ 0 := by decide) :
    nn.Builder (nn.Sequential (vocabularyShape cfg leading) (vocabularyShape cfg leading)) :=
  letI : NeZero cfg.seqLen := ⟨h_seqLen⟩
  letI : NeZero cfg.dModel := ⟨h_dModel⟩
  let dModel := cfg.dModel
  let embeddingInit := cfg.parameterInit?.getD (.uniform (-0.02) 0.02)
  do
    let embedRaw ← nn.oneHotEmbedding cfg.vocab dModel { weightInit := embeddingInit }
      (leading := leading ++ [cfg.seqLen])
    let embed : nn.Sequential (vocabularyShape cfg leading) (embeddingShape cfg leading) := by
      simpa [dModel, vocabularyShape, embeddingShape, Config.dModel,
        Shape.ofList_append, Shape.appendDim_eq_concat,
        Shape.concat_assoc] using embedRaw
    let body ← fromEmbeddings cfg leading (h_seqLen := h_seqLen) (h_dModel := h_dModel)
    pure (embed >>> body)

end CausalTransformer
end models
end nn

end TorchLean
