/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Models.CausalTransformer.Architecture
public import NN.API.Module.Execution
public import NN.API.Runtime
public import NN.Runtime.Autograd.TorchLean.NN

/-!
# Causal Transformer Runtime Support

Indexed-token and tied-weight model execution, programs, and causal-language-model objectives.
-/

@[expose] public section

namespace TorchLean


open Spec Tensor

namespace nn
namespace models
namespace CausalTransformer

/-- Indexed-token causal Transformer with an independent vocabulary head. -/
def Indexed.model (cfg : Config) [NeZero cfg.vocab] {leading : List Nat}
    (body : nn.Sequential
      (embeddingShape cfg leading)
      (vocabularyShape cfg leading)) :
    nn.IndexedModel (tokenShape cfg leading) (vocabularyShape cfg leading) (Fin cfg.vocab) :=
  let embeddingInit := cfg.parameterInit?.getD (.uniform (-0.02) 0.02)
  let embedding : nn.IndexedModel
      (tokenShape cfg leading) (embeddingShape cfg leading) (Fin cfg.vocab) := by
    simpa [tokenShape, embeddingShape, Shape.ofList_append,
      Shape.appendDim_eq_concat, Shape.concat_assoc] using
      (nn.Internal.embedding cfg.vocab cfg.dModel { weightInit := embeddingInit }).model
        (leading ++ [cfg.seqLen])
  embedding.andThen body

/--
Run indexed-token embedding and a causal Transformer body in any TorchLean backend.

Tokens remain bounded `Fin cfg.vocab` tensors throughout the call. Only the embedding table and
Transformer parameters use the model scalar type, so token ids cannot receive gradients or be
reinterpreted as floating-point data.
-/
def Indexed.forward
    {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m]
    [_root_.Runtime.Autograd.Torch.Ops (m := m) (α := α)]
    (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
    (cfg : Config) [NeZero cfg.vocab]
    {leading : List Nat}
    (body : nn.Sequential
      (embeddingShape cfg leading)
      (vocabularyShape cfg leading))
    (state : _root_.Runtime.Autograd.Torch.RefList
      (_root_.TorchLean.Runtime.ValueRef (m := m) (α := α))
      (Indexed.model cfg body).stateShapes)
    (tokens : _root_.Runtime.Autograd.Torch.DataRef
      (m := m) (α := α) (Fin cfg.vocab) (tokenShape cfg leading)) :
    m (_root_.TorchLean.Runtime.ValueRef
      (m := m) (α := α) (vocabularyShape cfg leading)) := do
  let model := Indexed.model cfg body
  let withInputs := _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
    (Ref := _root_.TorchLean.Runtime.ValueRef (m := m) (α := α))
    (ss := model.stateShapes)
    (β := _root_.Runtime.Autograd.Torch.CurriedRef
      (fun s => _root_.Runtime.Autograd.Torch.DataRef
        (m := m) (α := α) (Fin cfg.vocab) s)
      [tokenShape cfg leading]
      (m (_root_.TorchLean.Runtime.ValueRef
        (m := m) (α := α) (vocabularyShape cfg leading))))
    (model.forward mode (α := α)) state
  withInputs tokens

/-- Execution-polymorphic indexed-token forward program. -/
def Indexed.programWithMode
    (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
    (cfg : Config) [NeZero cfg.vocab]
    {leading : List Nat}
    (body : nn.Sequential
      (embeddingShape cfg leading)
      (vocabularyShape cfg leading))
    {α : Type} [Context α] [DecidableEq Shape] :
    _root_.Runtime.Autograd.TorchLean.ProgramWithDataInputs α (Fin cfg.vocab)
      (Indexed.model cfg body).stateShapes [tokenShape cfg leading]
      (vocabularyShape cfg leading) :=
  (Indexed.model cfg body).forward mode (α := α)

/-- Evaluation-mode indexed-token forward program. -/
def Indexed.program
    (cfg : Config) [NeZero cfg.vocab]
    {leading : List Nat}
    (body : nn.Sequential
      (embeddingShape cfg leading)
      (vocabularyShape cfg leading))
    {α : Type} [Context α] [DecidableEq Shape] :
    _root_.Runtime.Autograd.TorchLean.ProgramWithDataInputs α (Fin cfg.vocab)
      (Indexed.model cfg body).stateShapes [tokenShape cfg leading]
      (vocabularyShape cfg leading) :=
  Indexed.programWithMode .eval cfg body

/-- Model-state shapes for a model whose embedding and output projection are tied. -/
abbrev Tied.stateShapes (cfg : Config)
    {leading : List Nat}
    (body : nn.Sequential (embeddingShape cfg leading) (embeddingShape cfg leading)) :
    List Shape :=
  [cfg.vocab, cfg.dModel] :: nn.stateShapes body

/--
Run a causal Transformer whose token lookup and vocabulary projection share one matrix.

The matrix has shape `(vocab, dModel)`. The forward pass gathers its rows to construct the input
embeddings, runs the hidden-state Transformer, and multiplies the final hidden states by its
transpose. Consequently, gradients from both lookup and prediction accumulate into the same
parameter.
-/
def Tied.forward
    {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m]
    [_root_.Runtime.Autograd.Torch.Ops (m := m) (α := α)]
    (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
    (cfg : Config) [NeZero cfg.vocab]
    {leading : List Nat}
    (body : nn.Sequential
      (embeddingShape cfg leading)
      (embeddingShape cfg leading))
    (params : _root_.Runtime.Autograd.Torch.RefList
      (_root_.TorchLean.Runtime.ValueRef (m := m) (α := α))
      (Tied.stateShapes cfg body))
    (tokens : _root_.Runtime.Autograd.Torch.DataRef
      (m := m) (α := α) (Fin cfg.vocab) (tokenShape cfg leading)) :
    m (_root_.TorchLean.Runtime.ValueRef
      (m := m) (α := α) (vocabularyShape cfg leading)) := do
  let .cons tokenEmbedding bodyParams := params
  let embeddingsRaw ← _root_.Runtime.Autograd.TorchLean.F.embedding
    (m := m) (α := α) (vocab := cfg.vocab) (dim := cfg.dModel) tokenEmbedding tokens
  let embeddings : _root_.TorchLean.Runtime.ValueRef
      (m := m) (α := α) (embeddingShape cfg leading) := by
    simpa [tokenShape, embeddingShape, Shape.appendDim_appendDim_eq_concat] using embeddingsRaw
  let hidden ← _root_.Runtime.Autograd.TorchLean.NN.Seq.forwardState
    (model := body) (α := α) (m := m) mode bodyParams embeddings
  let hiddenRows ← _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
    (s₁ := embeddingShape cfg leading)
    (s₂ := [(tokenShape cfg leading).prod, cfg.dModel])
    hidden (by
      simp [embeddingShape, tokenShape, Shape.size_ofList, Shape.size_concat, Shape.size,
        List.prod_append, Nat.mul_assoc])
  let projection ← _root_.Runtime.Autograd.Torch.swapAdjacentAtDepth (m := m) (α := α)
    (s := [cfg.vocab, cfg.dModel]) 0 tokenEmbedding
  let logitsRows ← _root_.Runtime.Autograd.Torch.matmul (m := m) (α := α)
    (batchA := .scalar) (batchB := .scalar) (batch := .scalar)
    (mDim := (tokenShape cfg leading).prod) (nDim := cfg.dModel) (pDim := cfg.vocab)
    hiddenRows projection
  _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
    (s₁ := [(tokenShape cfg leading).prod, cfg.vocab])
    (s₂ := vocabularyShape cfg leading) logitsRows
    (by
      simp [vocabularyShape, tokenShape, Shape.size_ofList, Shape.size_concat, Shape.size,
        List.prod_append, Nat.mul_assoc])

/-- Execution-polymorphic tied-token forward program. -/
def Tied.programWithMode
    (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
    (cfg : Config) [NeZero cfg.vocab]
    {leading : List Nat}
    (body : nn.Sequential
      (embeddingShape cfg leading)
      (embeddingShape cfg leading))
    {α : Type} [Context α] [DecidableEq Shape] :
    _root_.Runtime.Autograd.TorchLean.ProgramWithDataInputs α (Fin cfg.vocab)
      (Tied.stateShapes cfg body) [tokenShape cfg leading]
      (vocabularyShape cfg leading) :=
  fun {m} _ _ =>
    _root_.Runtime.Autograd.Torch.CurriedRef.curry
      (Ref := _root_.TorchLean.Runtime.ValueRef (m := m) (α := α))
      (ss := Tied.stateShapes cfg body)
      (β := _root_.Runtime.Autograd.Torch.CurriedRef
        (fun s => _root_.Runtime.Autograd.Torch.DataRef
          (m := m) (α := α) (Fin cfg.vocab) s)
        [tokenShape cfg leading]
        (m (_root_.TorchLean.Runtime.ValueRef
          (m := m) (α := α) (vocabularyShape cfg leading))))
      (fun params => fun tokens =>
        Tied.forward (m := m) (α := α) mode cfg body params tokens)

/-- Evaluation-mode tied-token forward program. -/
def Tied.program
    (cfg : Config) [NeZero cfg.vocab]
    {leading : List Nat}
    (body : nn.Sequential
      (embeddingShape cfg leading)
      (embeddingShape cfg leading))
    {α : Type} [Context α] [DecidableEq Shape] :
    _root_.Runtime.Autograd.TorchLean.ProgramWithDataInputs α (Fin cfg.vocab)
      (Tied.stateShapes cfg body) [tokenShape cfg leading]
      (vocabularyShape cfg leading) :=
  Tied.programWithMode .eval cfg body

/--
Scalar loss for causal language modeling with bounded token ids.

The public one-hot constructor above is useful for small teaching examples because the input is an
ordinary scalar tensor. File-backed tokenized datasets use bounded token ids, while the embedding
table uses the selected model scalar and the loss gathers target classes directly instead of
building one-hot targets.

`tokens` and `targets` have shape `leading ++ seqLen`. The embedding implementation may flatten
that shape for a gather kernel, but the public model preserves every leading axis.
-/
def Indexed.objectiveWithMode
    (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
    (cfg : Config) [NeZero cfg.vocab]
    {leading : List Nat}
    (body : nn.Sequential
      (embeddingShape cfg leading)
      (vocabularyShape cfg leading))
    (reduction : _root_.TorchLean.Loss.Reduction := .mean) :
    TorchLean.Module.ObjectiveDef (Fin cfg.vocab)
      (Indexed.model cfg body).stateShapes []
      [tokenShape cfg leading, tokenShape cfg leading] :=
  let model := Indexed.model cfg body
  { initState := model.initState
    runtimeInit := model.runtimeInit
    requiresGrad := model.requiresGrad
    loss := fun {α} => by
      intro _ _
      exact fun {m} _ _ =>
        _root_.Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := _root_.TorchLean.Runtime.ValueRef (m := m) (α := α))
          (ss := (Indexed.model cfg body).stateShapes ++ ([] : List Shape))
          (β := _root_.Runtime.Autograd.Torch.CurriedRef
            (fun s => _root_.Runtime.Autograd.Torch.DataRef
              (m := m) (α := α) (Fin cfg.vocab) s)
            [tokenShape cfg leading, tokenShape cfg leading]
            (m (_root_.TorchLean.Runtime.ValueRef (m := m) (α := α)
              Shape.scalar)))
          (fun args => fun tokens => fun targets =>
            (do
            let (params, empty) :=
              _root_.Runtime.Autograd.Torch.RefList.split
                (Ref := _root_.TorchLean.Runtime.ValueRef (m := m) (α := α))
                (ss₁ := (Indexed.model cfg body).stateShapes)
                (ss₂ := ([] : List Shape)) args
            let .nil := empty
            let logits ← Indexed.forward
              (m := m) (α := α) mode cfg body params tokens
            let logitsIndexed : _root_.TorchLean.Runtime.ValueRef (m := m) (α := α)
                ((Shape.ofList (tokenShape cfg leading)).concat
                  (Shape.ofList [cfg.vocab])) := by
              simpa [vocabularyShape, tokenShape, Shape.ofList_append] using logits
            let targetsIndexed : _root_.Runtime.Autograd.Torch.DataRef
                (m := m) (α := α) (Fin cfg.vocab)
                ((Shape.ofList (tokenShape cfg leading)).concat Shape.scalar) := by
              simpa using targets
            _root_.TorchLean.Loss.crossEntropy (m := m) (α := α)
              (leading := tokenShape cfg leading) (trailing := Shape.scalar)
              (classes := cfg.vocab) (Shape.ofList (tokenShape cfg leading)).rank rfl
              logitsIndexed targetsIndexed (reduction := reduction) :
              m (_root_.TorchLean.Runtime.ValueRef
                (m := m) (α := α) Shape.scalar))) }

/-- Training-mode wrapper for bounded-token causal language modeling. -/
def Indexed.objective (cfg : Config) [NeZero cfg.vocab] {leading : List Nat}
    (body : nn.Sequential
      (embeddingShape cfg leading)
      (vocabularyShape cfg leading))
    (reduction : _root_.TorchLean.Loss.Reduction := .mean) :
    TorchLean.Module.ObjectiveDef (Fin cfg.vocab)
      (Indexed.model cfg body).stateShapes []
      [tokenShape cfg leading, tokenShape cfg leading] :=
  Indexed.objectiveWithMode .train cfg body (reduction := reduction)

/-- Scalar causal-language-model loss for a tied token embedding and output projection. -/
def Tied.objectiveWithMode
    (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
    (cfg : Config) [NeZero cfg.vocab]
    {leading : List Nat}
    (body : nn.Sequential
      (embeddingShape cfg leading)
      (embeddingShape cfg leading))
    (reduction : _root_.TorchLean.Loss.Reduction := .mean) :
    TorchLean.Module.ObjectiveDef (Fin cfg.vocab)
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
    requiresGrad := #[true] ++ requiresGrad body
    loss := fun {α} => by
      intro _ _
      exact fun {m} _ _ =>
        _root_.Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := _root_.TorchLean.Runtime.ValueRef (m := m) (α := α))
          (ss := Tied.stateShapes cfg body ++ ([] : List Shape))
          (β := _root_.Runtime.Autograd.Torch.CurriedRef
            (fun s => _root_.Runtime.Autograd.Torch.DataRef
              (m := m) (α := α) (Fin cfg.vocab) s)
            [tokenShape cfg leading, tokenShape cfg leading]
            (m (_root_.TorchLean.Runtime.ValueRef (m := m) (α := α)
              Shape.scalar)))
          (fun args => fun tokens => fun targets => (do
            let (params, empty) :=
              _root_.Runtime.Autograd.Torch.RefList.split
                (Ref := _root_.TorchLean.Runtime.ValueRef (m := m) (α := α))
                (ss₁ := Tied.stateShapes cfg body)
                (ss₂ := ([] : List Shape)) args
            let .nil := empty
            let logits ← Tied.forward
              (m := m) (α := α) mode cfg body params tokens
            let logitsIndexed : _root_.TorchLean.Runtime.ValueRef (m := m) (α := α)
                ((Shape.ofList (tokenShape cfg leading)).concat
                  (Shape.ofList [cfg.vocab])) := by
              simpa [vocabularyShape, tokenShape, Shape.ofList_append] using logits
            let targetsIndexed : _root_.Runtime.Autograd.Torch.DataRef
                (m := m) (α := α) (Fin cfg.vocab)
                ((Shape.ofList (tokenShape cfg leading)).concat Shape.scalar) := by
              simpa using targets
            _root_.TorchLean.Loss.crossEntropy (m := m) (α := α)
              (leading := tokenShape cfg leading) (trailing := Shape.scalar)
              (classes := cfg.vocab) (Shape.ofList (tokenShape cfg leading)).rank rfl
              logitsIndexed targetsIndexed (reduction := reduction) :
              m (_root_.TorchLean.Runtime.ValueRef
                (m := m) (α := α) Shape.scalar))) }

/-- Training-mode wrapper for tied-token causal language modeling. -/
def Tied.objective
    (cfg : Config) [NeZero cfg.vocab]
    {leading : List Nat}
    (body : nn.Sequential
      (embeddingShape cfg leading)
      (embeddingShape cfg leading))
    (reduction : _root_.TorchLean.Loss.Reduction := .mean) :
    TorchLean.Module.ObjectiveDef (Fin cfg.vocab)
      (Tied.stateShapes cfg body) []
      [tokenShape cfg leading, tokenShape cfg leading] :=
  Tied.objectiveWithMode .train cfg body (reduction := reduction)

end CausalTransformer
end models
end nn

end TorchLean
