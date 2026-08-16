/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Embedding
public import NN.Spec.Module.SpecModule

/-!
# Embedding

Module wrapper for the differentiable one-hot presentation of an embedding table. The indexed
semantics is `Embedding.lookup`; this wrapper exists for proofs that treat the same table as the
linear map `oneHot @ weight`.
-/

@[expose] public section


namespace Spec

open Tensor
open ModSpec

variable {α : Type} [Context α]

/-- One-hot embedding module: `(seqLen, vocab)` to `(seqLen, embedDim)`. -/
def oneHotEmbeddingModule {vocab embedDim seqLen : Nat}
  (embedding : Embedding vocab embedDim α) :
  NNModuleSpec α (.dim seqLen (.dim vocab .scalar)) (.dim seqLen (.dim embedDim .scalar)) :=
{ forward := fun oneHot => embedding.oneHot oneHot
  kind := "OneHotEmbedding"
  export_func := {
    toPyTorch := s!"OneHotEmbedding(vocab={vocab}, embedDim={embedDim})"
    dimensions := (vocab, embedDim)
  } }

end Spec
