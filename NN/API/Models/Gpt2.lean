/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded
public import NN.API.Text.Generation
public import NN.Runtime.Autograd.TorchLean.NN

/-!
# Causal Transformer Models

This module defines the model structure shared by TorchLean's causal language-model examples.
It supports one-hot inputs and integer token tensors, with either an independent vocabulary head
or a projection tied to the token-embedding table. Tokenization and checkpoint formats live in
their respective API modules.
-/

@[expose] public section

namespace TorchLean


open Spec Tensor

namespace nn
namespace models

/--
Configuration shared by TorchLean's GPT-style causal language models.

The model has the common GPT-2 “shape”:

`embedding → learned positional embedding → (masked self-attention + FFN)×layers → LayerNorm → linear`

The configuration is independent of how token ids enter the model. One-hot, integer-token, and
pre-embedded constructors reuse the same Transformer width, depth, and output vocabulary.
-/
structure CausalTransformerConfig where
  batch : Nat
  seqLen : Nat
  vocab : Nat
  numHeads : Nat
  headDim : Nat
  ffnHidden : Nat
  layers : Nat
  /-- Feed-forward activation used in every Transformer block. -/
  activation : nn.blocks.Activation := .gelu
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
def CausalTransformerConfig.dModel (cfg : CausalTransformerConfig) : Nat :=
  cfg.numHeads * cfg.headDim

/-- Vocabulary-grid shape `(batch × seqLen × vocab)` used by one-hot inputs and output logits. -/
abbrev causalVocabularyShape (cfg : CausalTransformerConfig) : Spec.Shape :=
  shape![cfg.batch, cfg.seqLen, cfg.vocab]

/-- Embedded-token tensor shape `(batch × seqLen × dModel)`. -/
abbrev causalEmbeddingShape (cfg : CausalTransformerConfig) : Spec.Shape :=
  shape![cfg.batch, cfg.seqLen, cfg.dModel]

/--
Causal Transformer hidden-state stack after token embeddings have been computed.

The stack adds learned positions, applies causally masked Transformer blocks, and finishes with
LayerNorm. It deliberately has no vocabulary projection. Language models can therefore choose an
independent output head or reuse their token-embedding matrix.
-/
def causalTransformerHiddenFromEmbeddings (cfg : CausalTransformerConfig)
    (h_seqLen : cfg.seqLen ≠ 0 := by decide)
    (h_dModel : cfg.dModel ≠ 0 := by decide) :
    nn.M (nn.Sequential (causalEmbeddingShape cfg) (causalEmbeddingShape cfg)) :=
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
  nn.Sequential![
    nn.learnedPositionalEmbedding (batch := cfg.batch) (seqLen := cfg.seqLen) (embedDim := dModel)
      { posInit := posInit },
    nn.transformerEncoderStack (batch := cfg.batch) (n := cfg.seqLen) (dModel := dModel) encCfg
      (mask := some (text.causalMask cfg.seqLen)),
    nn.layerNorm (batch := cfg.batch) (seqLen := cfg.seqLen) (embedDim := dModel)
  ]

/--
GPT-style causal Transformer body with an independent affine vocabulary head.

Use `causalTransformerHiddenFromEmbeddings` when the caller needs hidden states or a tied
token-embedding/output matrix.
-/
def causalTransformerFromEmbeddings (cfg : CausalTransformerConfig)
    (h_seqLen : cfg.seqLen ≠ 0 := by decide)
    (h_dModel : cfg.dModel ≠ 0 := by decide) :
    nn.M (nn.Sequential (causalEmbeddingShape cfg) (causalVocabularyShape cfg)) := do
  let hidden ← causalTransformerHiddenFromEmbeddings cfg h_seqLen h_dModel
  let head ← nn.linearWith cfg.dModel cfg.vocab { weightInit? := cfg.parameterInit? }
    (pfx := .dim cfg.batch (.dim cfg.seqLen .scalar))
  pure (hidden >>> head)

/--
Build a GPT-2-style causal language model over one-hot tokens.

This is the shared constructor used by the runnable GPT-2 examples. It stays in `nn.M` so it
composes with the rest of the API-layer model-building interface.
-/
def causalTransformerOneHot (cfg : CausalTransformerConfig)
    (h_seqLen : cfg.seqLen ≠ 0 := by decide)
    (h_dModel : cfg.dModel ≠ 0 := by decide) :
    nn.M (nn.Sequential (causalVocabularyShape cfg) (causalVocabularyShape cfg)) :=
  letI : NeZero cfg.seqLen := ⟨h_seqLen⟩
  letI : NeZero cfg.dModel := ⟨h_dModel⟩
  let dModel := cfg.dModel
  let embeddingInit := cfg.parameterInit?.getD (.uniform (-0.02) 0.02)
  nn.embedding cfg.vocab dModel { wInit := embeddingInit }
    (pfx := .dim cfg.batch (.dim cfg.seqLen .scalar)) >>= fun embed =>
  causalTransformerFromEmbeddings cfg (h_seqLen := h_seqLen) (h_dModel := h_dModel) >>= fun body =>
  pure (embed >>> body)

/-- Flattened shape of one batch of token ids or row weights. -/
abbrev causalTokenShape (cfg : CausalTransformerConfig) : Spec.Shape :=
  .dim (cfg.batch * cfg.seqLen) .scalar

/-- Parameter shapes for an indexed-token model with an independent vocabulary head. -/
abbrev causalTransformerTokenParamShapes (cfg : CausalTransformerConfig)
    (body : nn.Sequential (causalEmbeddingShape cfg) (causalVocabularyShape cfg)) :
    List Spec.Shape :=
  (.dim cfg.vocab (.dim cfg.dModel .scalar)) :: paramShapes body

/--
Run indexed-token embedding and a causal Transformer body in any TorchLean backend.

Tokens remain `Nat` tensors throughout the call. Only the embedding table and Transformer
parameters use the model scalar type, so the program cannot differentiate with respect to token ids
or reinterpret them as floating-point data.
-/
def causalTransformerTokenForward
    {α : Type} [Context α] [DecidableEq Spec.Shape]
    {m : Type → Type} [Monad m]
    [_root_.Runtime.Autograd.Torch.Ops (m := m) (α := α)]
    (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
    (cfg : CausalTransformerConfig)
    (body : nn.Sequential (causalEmbeddingShape cfg) (causalVocabularyShape cfg))
    (params : _root_.Runtime.Autograd.Torch.RefList
      (_root_.Runtime.Autograd.TorchLean.NN.Seq.RefT (m := m) (α := α))
      (causalTransformerTokenParamShapes cfg body))
    (tokens : _root_.Runtime.Autograd.Torch.NatTensorRef
      (m := m) (α := α) (causalTokenShape cfg)) :
    m (_root_.Runtime.Autograd.TorchLean.NN.Seq.RefT
      (m := m) (α := α) (causalVocabularyShape cfg)) := do
  let .cons tokenEmbedding bodyParams := params
  let embeddings ← _root_.Runtime.Autograd.TorchLean.F.embeddingBatchSeqNat
    (m := m) (α := α) (vocab := cfg.vocab) (dim := cfg.dModel)
    (batch := cfg.batch) (seqLen := cfg.seqLen) tokenEmbedding tokens
  _root_.Runtime.Autograd.TorchLean.NN.Seq.forwardParams
    (model := body) (α := α) (m := m) mode bodyParams embeddings

/-- Backend-polymorphic indexed-token forward program. -/
def causalTransformerTokenProgramWithMode
    (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
    (cfg : CausalTransformerConfig)
    (body : nn.Sequential (causalEmbeddingShape cfg) (causalVocabularyShape cfg))
    {α : Type} [Context α] [DecidableEq Spec.Shape] :
    _root_.Runtime.Autograd.TorchLean.ProgramWithNatInputs α
      (causalTransformerTokenParamShapes cfg body) [causalTokenShape cfg]
      (causalVocabularyShape cfg) :=
  fun {m} _ _ =>
    _root_.Runtime.Autograd.Torch.CurriedRef.curry
      (Ref := _root_.Runtime.Autograd.TorchLean.NN.Seq.RefT (m := m) (α := α))
      (ss := causalTransformerTokenParamShapes cfg body)
      (β := _root_.Runtime.Autograd.Torch.CurriedRef
        (fun s => _root_.Runtime.Autograd.Torch.NatTensorRef (m := m) (α := α) s)
        [causalTokenShape cfg]
        (m (_root_.Runtime.Autograd.TorchLean.NN.Seq.RefT
          (m := m) (α := α) (causalVocabularyShape cfg))))
      (fun params => fun tokens =>
        causalTransformerTokenForward (m := m) (α := α) mode cfg body params tokens)

/-- Evaluation-mode indexed-token forward program. -/
def causalTransformerTokenProgram
    (cfg : CausalTransformerConfig)
    (body : nn.Sequential (causalEmbeddingShape cfg) (causalVocabularyShape cfg))
    {α : Type} [Context α] [DecidableEq Spec.Shape] :
    _root_.Runtime.Autograd.TorchLean.ProgramWithNatInputs α
      (causalTransformerTokenParamShapes cfg body) [causalTokenShape cfg]
      (causalVocabularyShape cfg) :=
  causalTransformerTokenProgramWithMode .eval cfg body

/-- Parameter shapes for an indexed-token model whose embedding and output projection are tied. -/
abbrev causalTransformerTiedTokenParamShapes (cfg : CausalTransformerConfig)
    (body : nn.Sequential (causalEmbeddingShape cfg) (causalEmbeddingShape cfg)) :
    List Spec.Shape :=
  (.dim cfg.vocab (.dim cfg.dModel .scalar)) :: paramShapes body

/--
Run a causal Transformer whose token lookup and vocabulary projection share one matrix.

The matrix has shape `(vocab, dModel)`. The forward pass gathers its rows to construct the input
embeddings, runs the hidden-state Transformer, and multiplies the final hidden states by its
transpose. Consequently, gradients from both lookup and prediction accumulate into the same
parameter.
-/
def causalTransformerTiedTokenForward
    {α : Type} [Context α] [DecidableEq Spec.Shape]
    {m : Type → Type} [Monad m]
    [_root_.Runtime.Autograd.Torch.Ops (m := m) (α := α)]
    (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
    (cfg : CausalTransformerConfig)
    (body : nn.Sequential (causalEmbeddingShape cfg) (causalEmbeddingShape cfg))
    (params : _root_.Runtime.Autograd.Torch.RefList
      (_root_.Runtime.Autograd.TorchLean.NN.Seq.RefT (m := m) (α := α))
      (causalTransformerTiedTokenParamShapes cfg body))
    (tokens : _root_.Runtime.Autograd.Torch.NatTensorRef
      (m := m) (α := α) (causalTokenShape cfg)) :
    m (_root_.Runtime.Autograd.TorchLean.NN.Seq.RefT
      (m := m) (α := α) (causalVocabularyShape cfg)) := do
  let .cons tokenEmbedding bodyParams := params
  let embeddings ← _root_.Runtime.Autograd.TorchLean.F.embeddingBatchSeqNat
    (m := m) (α := α) (vocab := cfg.vocab) (dim := cfg.dModel)
    (batch := cfg.batch) (seqLen := cfg.seqLen) tokenEmbedding tokens
  let hidden ← _root_.Runtime.Autograd.TorchLean.NN.Seq.forwardParams
    (model := body) (α := α) (m := m) mode bodyParams embeddings
  let hiddenRows ← _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
    (s₁ := causalEmbeddingShape cfg)
    (s₂ := .dim (cfg.batch * cfg.seqLen) (.dim cfg.dModel .scalar))
    hidden (by simp [_root_.Spec.Shape.size, Nat.mul_assoc])
  let projection ← _root_.Runtime.Autograd.Torch.transpose2d (m := m) (α := α)
    (mDim := cfg.vocab) (nDim := cfg.dModel) tokenEmbedding
  let logitsRows ← _root_.Runtime.Autograd.Torch.matmul (m := m) (α := α)
    (mDim := cfg.batch * cfg.seqLen) (nDim := cfg.dModel) (pDim := cfg.vocab)
    hiddenRows projection
  _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
    (s₁ := .dim (cfg.batch * cfg.seqLen) (.dim cfg.vocab .scalar))
    (s₂ := causalVocabularyShape cfg) logitsRows
    (by simp [_root_.Spec.Shape.size, Nat.mul_assoc])

/-- Backend-polymorphic tied-token forward program. -/
def causalTransformerTiedTokenProgramWithMode
    (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
    (cfg : CausalTransformerConfig)
    (body : nn.Sequential (causalEmbeddingShape cfg) (causalEmbeddingShape cfg))
    {α : Type} [Context α] [DecidableEq Spec.Shape] :
    _root_.Runtime.Autograd.TorchLean.ProgramWithNatInputs α
      (causalTransformerTiedTokenParamShapes cfg body) [causalTokenShape cfg]
      (causalVocabularyShape cfg) :=
  fun {m} _ _ =>
    _root_.Runtime.Autograd.Torch.CurriedRef.curry
      (Ref := _root_.Runtime.Autograd.TorchLean.NN.Seq.RefT (m := m) (α := α))
      (ss := causalTransformerTiedTokenParamShapes cfg body)
      (β := _root_.Runtime.Autograd.Torch.CurriedRef
        (fun s => _root_.Runtime.Autograd.Torch.NatTensorRef (m := m) (α := α) s)
        [causalTokenShape cfg]
        (m (_root_.Runtime.Autograd.TorchLean.NN.Seq.RefT
          (m := m) (α := α) (causalVocabularyShape cfg))))
      (fun params => fun tokens =>
        causalTransformerTiedTokenForward (m := m) (α := α) mode cfg body params tokens)

/-- Evaluation-mode tied-token forward program. -/
def causalTransformerTiedTokenProgram
    (cfg : CausalTransformerConfig)
    (body : nn.Sequential (causalEmbeddingShape cfg) (causalEmbeddingShape cfg))
    {α : Type} [Context α] [DecidableEq Spec.Shape] :
    _root_.Runtime.Autograd.TorchLean.ProgramWithNatInputs α
      (causalTransformerTiedTokenParamShapes cfg body) [causalTokenShape cfg]
      (causalVocabularyShape cfg) :=
  causalTransformerTiedTokenProgramWithMode .eval cfg body

/--
Scalar loss for causal language modeling with integer token ids.

The public one-hot constructor above is useful for small teaching examples because the input is an
ordinary Float tensor.  File-backed tokenized datasets use the representation found in
language-model training systems: token ids are `Nat`s, the embedding table is a trainable Float
parameter, and the loss gathers the target classes directly instead of building one-hot targets.

`tokens` and `targets` are flattened `(batch * seqLen)` vectors.  This matches the backend gather
ops and keeps dataset storage simple; the embedding helper reshapes gathered rows back to
`(batch, seqLen, dModel)` before running the Transformer body.
-/
def causalTransformerTokenScalarModuleDefWithMode
    (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
    (cfg : CausalTransformerConfig)
    (body : nn.Sequential (causalEmbeddingShape cfg) (causalVocabularyShape cfg))
    (reduction : _root_.TorchLean.Loss.Reduction := .mean) :
    TorchLean.Module.ScalarModuleDef
      (causalTransformerTokenParamShapes cfg body) []
      [causalTokenShape cfg, causalTokenShape cfg] :=
  let embeddingInit := cfg.parameterInit?.getD (.uniform (-0.02) 0.02)
  { initParams := .cons
      (_root_.Runtime.Autograd.Torch.Init.tensor embeddingInit (seed := 0)) (initParams body)
    runtimeInit :=
      match _root_.Runtime.Autograd.TorchLean.NN.Seq.runtimeInit? body with
      | some bodyPlan => some (.cons
          (_root_.Runtime.Autograd.TorchLean.Module.RuntimeInit.FloatInit.ofScheme
            embeddingInit 0) bodyPlan)
      | none => none
    initRequiresGrad := true :: paramRequiresGrad body
    loss := fun {α} => by
      intro _ _
      exact fun {m} _ _ =>
        _root_.Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := _root_.Runtime.Autograd.TorchLean.NN.Seq.RefT (m := m) (α := α))
          (ss := causalTransformerTokenParamShapes cfg body ++ ([] : List Spec.Shape))
          (β := _root_.Runtime.Autograd.Torch.CurriedRef
            (fun s => _root_.Runtime.Autograd.Torch.NatTensorRef (m := m) (α := α) s)
            [causalTokenShape cfg, causalTokenShape cfg]
            (m (_root_.Runtime.Autograd.TorchLean.NN.Seq.RefT (m := m) (α := α)
              Spec.Shape.scalar)))
          (fun args => fun tokens => fun targets =>
            (do
            let (params, empty) :=
              _root_.Runtime.Autograd.Torch.RefList.split
                (Ref := _root_.Runtime.Autograd.TorchLean.NN.Seq.RefT (m := m) (α := α))
                (ss₁ := causalTransformerTokenParamShapes cfg body)
                (ss₂ := ([] : List Spec.Shape)) args
            let .nil := empty
            let logits ← causalTransformerTokenForward
              (m := m) (α := α) mode cfg body params tokens
            let logitsRows ← _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
              (s₁ := .dim cfg.batch (.dim cfg.seqLen (.dim cfg.vocab .scalar)))
              (s₂ := .dim (cfg.batch * cfg.seqLen) (.dim cfg.vocab .scalar))
              logits (by
                simp [_root_.Spec.Shape.size, Nat.mul_assoc])
            _root_.TorchLean.Loss.crossEntropyRowsNat (m := m) (α := α)
              (rows := cfg.batch * cfg.seqLen) (classes := cfg.vocab)
              logitsRows targets (reduction := reduction) :
              m (_root_.Runtime.Autograd.TorchLean.NN.Seq.RefT
                (m := m) (α := α) Spec.Shape.scalar))) }

/-- Training-mode wrapper for integer-token causal language modeling. -/
def causalTransformerTokenScalarModuleDef (cfg : CausalTransformerConfig)
    (body : nn.Sequential (causalEmbeddingShape cfg) (causalVocabularyShape cfg))
    (reduction : _root_.TorchLean.Loss.Reduction := .mean) :
    TorchLean.Module.ScalarModuleDef
      (causalTransformerTokenParamShapes cfg body) []
      [causalTokenShape cfg, causalTokenShape cfg] :=
  causalTransformerTokenScalarModuleDefWithMode .train cfg body (reduction := reduction)

/-- Scalar causal-language-model loss for a tied token embedding and output projection. -/
def causalTransformerTiedTokenScalarModuleDefWithMode
    (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
    (cfg : CausalTransformerConfig)
    (body : nn.Sequential (causalEmbeddingShape cfg) (causalEmbeddingShape cfg))
    (reduction : _root_.TorchLean.Loss.Reduction := .mean) :
    TorchLean.Module.ScalarModuleDef
      (causalTransformerTiedTokenParamShapes cfg body) []
      [causalTokenShape cfg, causalTokenShape cfg] :=
  let embeddingInit := cfg.parameterInit?.getD (.uniform (-0.02) 0.02)
  { initParams := .cons
      (_root_.Runtime.Autograd.Torch.Init.tensor embeddingInit (seed := 0)) (initParams body)
    runtimeInit :=
      match _root_.Runtime.Autograd.TorchLean.NN.Seq.runtimeInit? body with
      | some bodyPlan => some (.cons
          (_root_.Runtime.Autograd.TorchLean.Module.RuntimeInit.FloatInit.ofScheme
            embeddingInit 0) bodyPlan)
      | none => none
    initRequiresGrad := true :: paramRequiresGrad body
    loss := fun {α} => by
      intro _ _
      exact fun {m} _ _ =>
        _root_.Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := _root_.Runtime.Autograd.TorchLean.NN.Seq.RefT (m := m) (α := α))
          (ss := causalTransformerTiedTokenParamShapes cfg body ++ ([] : List Spec.Shape))
          (β := _root_.Runtime.Autograd.Torch.CurriedRef
            (fun s => _root_.Runtime.Autograd.Torch.NatTensorRef (m := m) (α := α) s)
            [causalTokenShape cfg, causalTokenShape cfg]
            (m (_root_.Runtime.Autograd.TorchLean.NN.Seq.RefT (m := m) (α := α)
              Spec.Shape.scalar)))
          (fun args => fun tokens => fun targets => (do
            let (params, empty) :=
              _root_.Runtime.Autograd.Torch.RefList.split
                (Ref := _root_.Runtime.Autograd.TorchLean.NN.Seq.RefT (m := m) (α := α))
                (ss₁ := causalTransformerTiedTokenParamShapes cfg body)
                (ss₂ := ([] : List Spec.Shape)) args
            let .nil := empty
            let logits ← causalTransformerTiedTokenForward
              (m := m) (α := α) mode cfg body params tokens
            let logitsRows ← _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
              (s₁ := causalVocabularyShape cfg)
              (s₂ := .dim (cfg.batch * cfg.seqLen) (.dim cfg.vocab .scalar))
              logits (by simp [_root_.Spec.Shape.size, Nat.mul_assoc])
            _root_.TorchLean.Loss.crossEntropyRowsNat (m := m) (α := α)
              (rows := cfg.batch * cfg.seqLen) (classes := cfg.vocab)
              logitsRows targets (reduction := reduction) :
              m (_root_.Runtime.Autograd.TorchLean.NN.Seq.RefT
                (m := m) (α := α) Spec.Shape.scalar))) }

/-- Training-mode wrapper for tied-token causal language modeling. -/
def causalTransformerTiedTokenScalarModuleDef
    (cfg : CausalTransformerConfig)
    (body : nn.Sequential (causalEmbeddingShape cfg) (causalEmbeddingShape cfg))
    (reduction : _root_.TorchLean.Loss.Reduction := .mean) :
    TorchLean.Module.ScalarModuleDef
      (causalTransformerTiedTokenParamShapes cfg body) []
      [causalTokenShape cfg, causalTokenShape cfg] :=
  causalTransformerTiedTokenScalarModuleDefWithMode .train cfg body (reduction := reduction)

end models
end nn

end TorchLean
