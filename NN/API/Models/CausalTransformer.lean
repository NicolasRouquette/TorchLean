/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded
public import NN.API.Module.Execution
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
namespace CausalTransformer

/--
Configuration shared by TorchLean's GPT-style causal language models.

The model has the common GPT-2 “shape”:

`embedding → learned positional embedding → (masked self-attention + FFN)×layers → LayerNorm → linear`

The configuration is independent of how token ids enter the model. One-hot, integer-token, and
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
abbrev tokenShape (cfg : Config) (leading : Spec.Shape := .scalar) : Spec.Shape :=
  leading.appendDim cfg.seqLen

/-- Vocabulary-grid shape `leading ++ (seqLen × vocab)` used by inputs and output logits. -/
abbrev vocabularyShape (cfg : Config) (leading : Spec.Shape := .scalar) : Spec.Shape :=
  leading.concat (.dim cfg.seqLen (.dim cfg.vocab .scalar))

/-- Embedded-token shape `leading ++ (seqLen × dModel)`. -/
abbrev embeddingShape (cfg : Config) (leading : Spec.Shape := .scalar) : Spec.Shape :=
  leading.concat (.dim cfg.seqLen (.dim cfg.dModel .scalar))

/--
Causal Transformer hidden-state stack after token embeddings have been computed.

The stack adds learned positions, applies causally masked Transformer blocks, and finishes with
LayerNorm. It deliberately has no vocabulary projection. Language models can therefore choose an
independent output head or reuse their token-embedding matrix.
-/
def hidden (cfg : Config) (leading : Spec.Shape := .scalar)
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
      simpa only [dModel, embeddingShape, Config.dModel, Spec.Shape.concat_appendDim,
        Spec.Shape.appendDim] using positionalRaw
    let blocksRaw ← nn.transformerEncoderStack leading
      (n := cfg.seqLen) (dModel := dModel) encCfg
      (mask := some (text.causalMask cfg.seqLen))
    let blocks : nn.Sequential (embeddingShape cfg leading) (embeddingShape cfg leading) := by
      simpa only [dModel, embeddingShape, Config.dModel, Spec.Shape.concat_appendDim,
        Spec.Shape.appendDim] using blocksRaw
    let normalizationRaw ←
      nn.layerNorm (leading.concat (.dim cfg.seqLen .scalar)) (width := dModel)
    let normalization :
        nn.Sequential (embeddingShape cfg leading) (embeddingShape cfg leading) := by
      simpa only [dModel, embeddingShape, Config.dModel, Spec.Shape.concat_appendDim,
        Spec.Shape.appendDim] using normalizationRaw
    pure (positional >>> blocks >>> normalization)

/--
GPT-style causal Transformer body with an independent affine vocabulary head.

Use `hidden` when the caller needs hidden states or a tied
token-embedding/output matrix.
-/
def fromEmbeddings (cfg : Config) (leading : Spec.Shape := .scalar)
    (h_seqLen : cfg.seqLen ≠ 0 := by decide)
    (h_dModel : cfg.dModel ≠ 0 := by decide) :
    nn.Builder (nn.Sequential (embeddingShape cfg leading) (vocabularyShape cfg leading)) := do
  let hidden ← hidden cfg leading h_seqLen h_dModel
  let headRaw ← nn.linearWith cfg.dModel cfg.vocab { weightInit? := cfg.parameterInit? }
    (leading := leading.concat (.dim cfg.seqLen .scalar))
  let head : nn.Sequential (embeddingShape cfg leading) (vocabularyShape cfg leading) := by
    simpa only [embeddingShape, vocabularyShape, Spec.Shape.concat_appendDim,
      Spec.Shape.appendDim] using headRaw
  pure (hidden >>> head)

/--
Build a GPT-2-style causal language model over one-hot tokens.

This is the shared constructor used by the runnable GPT-2 examples. It stays in `nn.Builder` so it
composes with the rest of the API-layer model-building interface.
-/
def oneHot (cfg : Config) (leading : Spec.Shape := .scalar)
    (h_seqLen : cfg.seqLen ≠ 0 := by decide)
    (h_dModel : cfg.dModel ≠ 0 := by decide) :
    nn.Builder (nn.Sequential (vocabularyShape cfg leading) (vocabularyShape cfg leading)) :=
  letI : NeZero cfg.seqLen := ⟨h_seqLen⟩
  letI : NeZero cfg.dModel := ⟨h_dModel⟩
  let dModel := cfg.dModel
  let embeddingInit := cfg.parameterInit?.getD (.uniform (-0.02) 0.02)
  do
    let embedRaw ← nn.oneHotEmbedding cfg.vocab dModel { weightInit := embeddingInit }
      (leading := leading.concat (.dim cfg.seqLen .scalar))
    let embed : nn.Sequential (vocabularyShape cfg leading) (embeddingShape cfg leading) := by
      simpa only [dModel, vocabularyShape, embeddingShape, Config.dModel,
        Spec.Shape.concat_appendDim, Spec.Shape.appendDim] using embedRaw
    let body ← fromEmbeddings cfg leading (h_seqLen := h_seqLen) (h_dModel := h_dModel)
    pure (embed >>> body)

/-- Reject a token tensor containing an id outside a model's vocabulary. -/
def validateTokenIds (cfg : Config) (role : String) {s : Spec.Shape}
    (ids : Spec.Tensor Nat s) : Except String Unit :=
  if Spec.Tensor.allSpec (fun tokenId => decide (tokenId < cfg.vocab)) ids then
    pure ()
  else
    throw s!"causal Transformer: {role} token id outside vocabulary of size {cfg.vocab}"

/-- Indexed-token causal Transformer with an independent vocabulary head. -/
def Indexed.model (cfg : Config) {leading : Spec.Shape}
    (body : nn.Sequential
      (embeddingShape cfg leading)
      (vocabularyShape cfg leading)) :
    nn.IndexedModel (tokenShape cfg leading) (vocabularyShape cfg leading) :=
  let embeddingInit := cfg.parameterInit?.getD (.uniform (-0.02) 0.02)
  let embedding :=
    (nn.Internal.embedding cfg.vocab cfg.dModel { weightInit := embeddingInit }).model
      (tokenShape cfg leading)
  embedding.andThen (by
    simpa [tokenShape, embeddingShape, Spec.Shape.appendDim_appendDim_eq_concat] using body)

/-- Model-state shapes for an indexed-token model with an independent vocabulary head. -/
abbrev Indexed.stateShapes (cfg : Config) {leading : Spec.Shape}
    (body : nn.Sequential
      (embeddingShape cfg leading)
      (vocabularyShape cfg leading)) :
    List Spec.Shape :=
  (Indexed.model cfg body).stateShapes

/--
Run indexed-token embedding and a causal Transformer body in any TorchLean backend.

Tokens remain `Nat` tensors throughout the call. Only the embedding table and Transformer
parameters use the model scalar type, so the program cannot differentiate with respect to token ids
or reinterpret them as floating-point data.
-/
def Indexed.forward
    {α : Type} [Context α] [DecidableEq Spec.Shape]
    {m : Type → Type} [Monad m]
    [_root_.Runtime.Autograd.Torch.Ops (m := m) (α := α)]
    (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
    (cfg : Config)
    {leading : Spec.Shape}
    (body : nn.Sequential
      (embeddingShape cfg leading)
      (vocabularyShape cfg leading))
    (state : _root_.Runtime.Autograd.Torch.RefList
      (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α))
      (Indexed.stateShapes cfg body))
    (tokens : _root_.Runtime.Autograd.Torch.NatTensorRef
      (m := m) (α := α) (tokenShape cfg leading)) :
    m (_root_.Runtime.Autograd.TorchLean.RefTy
      (m := m) (α := α) (vocabularyShape cfg leading)) := do
  let model := Indexed.model cfg body
  let withInputs := _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
    (Ref := _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α))
    (ss := model.stateShapes)
    (β := _root_.Runtime.Autograd.Torch.CurriedRef
      (fun s => _root_.Runtime.Autograd.Torch.NatTensorRef (m := m) (α := α) s)
      [tokenShape cfg leading]
      (m (_root_.Runtime.Autograd.TorchLean.RefTy
        (m := m) (α := α) (vocabularyShape cfg leading))))
    (model.forward mode (α := α)) state
  withInputs tokens

/-- Execution-polymorphic indexed-token forward program. -/
def Indexed.programWithMode
    (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
    (cfg : Config)
    {leading : Spec.Shape}
    (body : nn.Sequential
      (embeddingShape cfg leading)
      (vocabularyShape cfg leading))
    {α : Type} [Context α] [DecidableEq Spec.Shape] :
    _root_.Runtime.Autograd.TorchLean.ProgramWithNatInputs α
      (Indexed.stateShapes cfg body) [tokenShape cfg leading]
      (vocabularyShape cfg leading) :=
  (Indexed.model cfg body).forward mode (α := α)

/-- Evaluation-mode indexed-token forward program. -/
def Indexed.program
    (cfg : Config)
    {leading : Spec.Shape}
    (body : nn.Sequential
      (embeddingShape cfg leading)
      (vocabularyShape cfg leading))
    {α : Type} [Context α] [DecidableEq Spec.Shape] :
    _root_.Runtime.Autograd.TorchLean.ProgramWithNatInputs α
      (Indexed.stateShapes cfg body) [tokenShape cfg leading]
      (vocabularyShape cfg leading) :=
  Indexed.programWithMode .eval cfg body

/-- Model-state shapes for a model whose embedding and output projection are tied. -/
abbrev Tied.stateShapes (cfg : Config)
    {leading : Spec.Shape}
    (body : nn.Sequential (embeddingShape cfg leading) (embeddingShape cfg leading)) :
    List Spec.Shape :=
  (.dim cfg.vocab (.dim cfg.dModel .scalar)) :: nn.stateShapes body

/--
Run a causal Transformer whose token lookup and vocabulary projection share one matrix.

The matrix has shape `(vocab, dModel)`. The forward pass gathers its rows to construct the input
embeddings, runs the hidden-state Transformer, and multiplies the final hidden states by its
transpose. Consequently, gradients from both lookup and prediction accumulate into the same
parameter.
-/
def Tied.forward
    {α : Type} [Context α] [DecidableEq Spec.Shape]
    {m : Type → Type} [Monad m]
    [_root_.Runtime.Autograd.Torch.Ops (m := m) (α := α)]
    (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
    (cfg : Config)
    {leading : Spec.Shape}
    (body : nn.Sequential
      (embeddingShape cfg leading)
      (embeddingShape cfg leading))
    (params : _root_.Runtime.Autograd.Torch.RefList
      (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α))
      (Tied.stateShapes cfg body))
    (tokens : _root_.Runtime.Autograd.Torch.NatTensorRef
      (m := m) (α := α) (tokenShape cfg leading)) :
    m (_root_.Runtime.Autograd.TorchLean.RefTy
      (m := m) (α := α) (vocabularyShape cfg leading)) := do
  let .cons tokenEmbedding bodyParams := params
  let embeddingsRaw ← _root_.Runtime.Autograd.TorchLean.F.embeddingNatOrZero
    (m := m) (α := α) (vocab := cfg.vocab) (dim := cfg.dModel) tokenEmbedding tokens
  let embeddings : _root_.Runtime.Autograd.TorchLean.RefTy
      (m := m) (α := α) (embeddingShape cfg leading) := by
    simpa [tokenShape, embeddingShape, Spec.Shape.appendDim_appendDim_eq_concat] using embeddingsRaw
  let hidden ← _root_.Runtime.Autograd.TorchLean.NN.Seq.forwardState
    (model := body) (α := α) (m := m) mode bodyParams embeddings
  let hiddenRows ← _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
    (s₁ := embeddingShape cfg leading)
    (s₂ := .dim (tokenShape cfg leading).size (.dim cfg.dModel .scalar))
    hidden (by simp [tokenShape, embeddingShape, _root_.Spec.Shape.size_appendDim,
      _root_.Spec.Shape.size_concat, _root_.Spec.Shape.size, Nat.mul_assoc])
  let projection ← _root_.Runtime.Autograd.Torch.transpose2d (m := m) (α := α)
    (mDim := cfg.vocab) (nDim := cfg.dModel) tokenEmbedding
  let logitsRows ← _root_.Runtime.Autograd.Torch.mm (m := m) (α := α)
    (mDim := (tokenShape cfg leading).size) (nDim := cfg.dModel) (pDim := cfg.vocab)
    hiddenRows projection
  _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
    (s₁ := .dim (tokenShape cfg leading).size (.dim cfg.vocab .scalar))
    (s₂ := vocabularyShape cfg leading) logitsRows
    (by simp [tokenShape, vocabularyShape, _root_.Spec.Shape.size_appendDim,
      _root_.Spec.Shape.size_concat, _root_.Spec.Shape.size, Nat.mul_assoc])

/-- Execution-polymorphic tied-token forward program. -/
def Tied.programWithMode
    (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
    (cfg : Config)
    {leading : Spec.Shape}
    (body : nn.Sequential
      (embeddingShape cfg leading)
      (embeddingShape cfg leading))
    {α : Type} [Context α] [DecidableEq Spec.Shape] :
    _root_.Runtime.Autograd.TorchLean.ProgramWithNatInputs α
      (Tied.stateShapes cfg body) [tokenShape cfg leading]
      (vocabularyShape cfg leading) :=
  fun {m} _ _ =>
    _root_.Runtime.Autograd.Torch.CurriedRef.curry
      (Ref := _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α))
      (ss := Tied.stateShapes cfg body)
      (β := _root_.Runtime.Autograd.Torch.CurriedRef
        (fun s => _root_.Runtime.Autograd.Torch.NatTensorRef (m := m) (α := α) s)
        [tokenShape cfg leading]
        (m (_root_.Runtime.Autograd.TorchLean.RefTy
          (m := m) (α := α) (vocabularyShape cfg leading))))
      (fun params => fun tokens =>
        Tied.forward (m := m) (α := α) mode cfg body params tokens)

/-- Evaluation-mode tied-token forward program. -/
def Tied.program
    (cfg : Config)
    {leading : Spec.Shape}
    (body : nn.Sequential
      (embeddingShape cfg leading)
      (embeddingShape cfg leading))
    {α : Type} [Context α] [DecidableEq Spec.Shape] :
    _root_.Runtime.Autograd.TorchLean.ProgramWithNatInputs α
      (Tied.stateShapes cfg body) [tokenShape cfg leading]
      (vocabularyShape cfg leading) :=
  Tied.programWithMode .eval cfg body

/--
Scalar loss for causal language modeling with integer token ids.

The public one-hot constructor above is useful for small teaching examples because the input is an
ordinary scalar tensor. File-backed tokenized datasets use the representation found in
language-model training systems: token ids are `Nat`s, the embedding table uses the selected model
scalar, and the loss gathers target classes directly instead of building one-hot targets.

`tokens` and `targets` have shape `leading ++ seqLen`. The embedding implementation may flatten
that shape for a gather kernel, but the public model preserves every leading axis.
-/
def Indexed.objectiveWithMode
    (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
    (cfg : Config)
    {leading : Spec.Shape}
    (body : nn.Sequential
      (embeddingShape cfg leading)
      (vocabularyShape cfg leading))
    (reduction : _root_.TorchLean.Loss.Reduction := .mean) :
    TorchLean.Module.ObjectiveDef
      (Indexed.stateShapes cfg body) []
      [tokenShape cfg leading, tokenShape cfg leading] :=
  let model := Indexed.model cfg body
  { initState := model.initState
    runtimeInit := model.runtimeInit
    requiresGrad := model.requiresGrad
    validateNatInputs := fun
      | .cons tokens (.cons targets .nil) => do
          validateTokenIds cfg "input" tokens
          validateTokenIds cfg "target" targets
    loss := fun {α} => by
      intro _ _
      exact fun {m} _ _ =>
        _root_.Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α))
          (ss := Indexed.stateShapes cfg body ++ ([] : List Spec.Shape))
          (β := _root_.Runtime.Autograd.Torch.CurriedRef
            (fun s => _root_.Runtime.Autograd.Torch.NatTensorRef (m := m) (α := α) s)
            [tokenShape cfg leading, tokenShape cfg leading]
            (m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α)
              Spec.Shape.scalar)))
          (fun args => fun tokens => fun targets =>
            (do
            let (params, empty) :=
              _root_.Runtime.Autograd.Torch.RefList.split
                (Ref := _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α))
                (ss₁ := Indexed.stateShapes cfg body)
                (ss₂ := ([] : List Spec.Shape)) args
            let .nil := empty
            let logits ← Indexed.forward
              (m := m) (α := α) mode cfg body params tokens
            let logitsRows ← _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
              (s₁ := vocabularyShape cfg leading)
              (s₂ := .dim (tokenShape cfg leading).size (.dim cfg.vocab .scalar))
              logits (by
                simp [tokenShape, vocabularyShape, _root_.Spec.Shape.size_appendDim,
                  _root_.Spec.Shape.size_concat, _root_.Spec.Shape.size, Nat.mul_assoc])
            let flatTargets := _root_.Runtime.Autograd.Torch.mapNatTensor (m := m) (α := α)
              (fun x => Spec.Tensor.reshapeSpec
                (s₁ := tokenShape cfg leading)
                (s₂ := .dim (tokenShape cfg leading).size .scalar) x (by
                  simp [_root_.Spec.Shape.size])) targets
            _root_.TorchLean.Loss.crossEntropyRowsNat (m := m) (α := α)
              (rows := (tokenShape cfg leading).size) (classes := cfg.vocab)
              logitsRows flatTargets (reduction := reduction) :
              m (_root_.Runtime.Autograd.TorchLean.RefTy
                (m := m) (α := α) Spec.Shape.scalar))) }

/-- Training-mode wrapper for integer-token causal language modeling. -/
def Indexed.objective (cfg : Config) {leading : Spec.Shape}
    (body : nn.Sequential
      (embeddingShape cfg leading)
      (vocabularyShape cfg leading))
    (reduction : _root_.TorchLean.Loss.Reduction := .mean) :
    TorchLean.Module.ObjectiveDef
      (Indexed.stateShapes cfg body) []
      [tokenShape cfg leading, tokenShape cfg leading] :=
  Indexed.objectiveWithMode .train cfg body (reduction := reduction)

/-- Scalar causal-language-model loss for a tied token embedding and output projection. -/
def Tied.objectiveWithMode
    (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
    (cfg : Config)
    {leading : Spec.Shape}
    (body : nn.Sequential
      (embeddingShape cfg leading)
      (embeddingShape cfg leading))
    (reduction : _root_.TorchLean.Loss.Reduction := .mean) :
    TorchLean.Module.ObjectiveDef
      (Tied.stateShapes cfg body) []
      [tokenShape cfg leading, tokenShape cfg leading] :=
  let embeddingInit := cfg.parameterInit?.getD (.uniform (-0.02) 0.02)
  { initState := .cons
      (_root_.Runtime.Autograd.Torch.Init.tensor embeddingInit (seed := 0)) (initState body)
    runtimeInit :=
      match _root_.Runtime.Autograd.TorchLean.NN.Seq.runtimeInit? body with
      | some bodyPlan => some (.cons
          (_root_.Runtime.Autograd.TorchLean.Module.RuntimeInit.FloatInit.ofScheme
            embeddingInit 0) bodyPlan)
      | none => none
    requiresGrad := true :: requiresGrad body
    validateNatInputs := fun
      | .cons tokens (.cons targets .nil) => do
          validateTokenIds cfg "input" tokens
          validateTokenIds cfg "target" targets
    loss := fun {α} => by
      intro _ _
      exact fun {m} _ _ =>
        _root_.Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α))
          (ss := Tied.stateShapes cfg body ++ ([] : List Spec.Shape))
          (β := _root_.Runtime.Autograd.Torch.CurriedRef
            (fun s => _root_.Runtime.Autograd.Torch.NatTensorRef (m := m) (α := α) s)
            [tokenShape cfg leading, tokenShape cfg leading]
            (m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α)
              Spec.Shape.scalar)))
          (fun args => fun tokens => fun targets => (do
            let (params, empty) :=
              _root_.Runtime.Autograd.Torch.RefList.split
                (Ref := _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α))
                (ss₁ := Tied.stateShapes cfg body)
                (ss₂ := ([] : List Spec.Shape)) args
            let .nil := empty
            let logits ← Tied.forward
              (m := m) (α := α) mode cfg body params tokens
            let logitsRows ← _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
              (s₁ := vocabularyShape cfg leading)
              (s₂ := .dim (tokenShape cfg leading).size (.dim cfg.vocab .scalar))
              logits (by
                simp [tokenShape, vocabularyShape, _root_.Spec.Shape.size_appendDim,
                  _root_.Spec.Shape.size_concat, _root_.Spec.Shape.size, Nat.mul_assoc])
            let flatTargets := _root_.Runtime.Autograd.Torch.mapNatTensor (m := m) (α := α)
              (fun x => Spec.Tensor.reshapeSpec
                (s₁ := tokenShape cfg leading)
                (s₂ := .dim (tokenShape cfg leading).size .scalar) x (by
                  simp [_root_.Spec.Shape.size])) targets
            _root_.TorchLean.Loss.crossEntropyRowsNat (m := m) (α := α)
              (rows := (tokenShape cfg leading).size) (classes := cfg.vocab)
              logitsRows flatTargets (reduction := reduction) :
              m (_root_.Runtime.Autograd.TorchLean.RefTy
                (m := m) (α := α) Spec.Shape.scalar))) }

/-- Training-mode wrapper for tied-token causal language modeling. -/
def Tied.objective
    (cfg : Config)
    {leading : Spec.Shape}
    (body : nn.Sequential
      (embeddingShape cfg leading)
      (embeddingShape cfg leading))
    (reduction : _root_.TorchLean.Loss.Reduction := .mean) :
    TorchLean.Module.ObjectiveDef
      (Tied.stateShapes cfg body) []
      [tokenShape cfg leading, tokenShape cfg leading] :=
  Tied.objectiveWithMode .train cfg body (reduction := reduction)

end CausalTransformer
end models
end nn

end TorchLean
