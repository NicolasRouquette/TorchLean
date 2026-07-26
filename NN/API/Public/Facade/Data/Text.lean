/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.Data.Datasets

/-!
# TorchLean Public Text Data

Causal language-model sample and dataset constructors.
-/

@[expose] public section

namespace TorchLean

namespace Data

def regressionGrid (lo hi : Float) (count : Nat) (target : Float → Float → Float) :
    Trainer.Dataset (.dim 2 .scalar) (.dim 1 .scalar) :=
  { build := fun {α} _ => pure <|
      let X : Tensor.T Float (.dim (count * count) (.dim 2 .scalar)) :=
        NN.API.Samples.squareGrid lo hi count
      let Y : Tensor.T Float (.dim (count * count) (.dim 1 .scalar)) :=
        NN.API.Samples.regressionTargetsFloat X target
      supervisedFromLeadingAxisFloat (α := α) X Y }

/--
Build a batched one-hot causal-language-model sample by repeating one token window across every
batch row.

The token list represents a `seqLen + 1` window. Shorter lists are padded and longer lists are
truncated by the causal-LM construction.
-/
def causalLmOneHotSample
    {α : Type} [Runtime.SemanticScalar α] [Runtime.Scalar α]
    (batch seqLen vocab : Nat) (tokens : List Nat) (padId : Nat := 0) :
    SupervisedSample α (shape![batch, seqLen, vocab]) (shape![batch, seqLen, vocab]) :=
  NN.API.text.causalLmSampleOneHotBatch (α := α) batch seqLen vocab tokens (padId := padId)

/--
Build a batched one-hot causal-language-model sample from one token window per batch row.

Use this for GPT-style examples that already know the per-row `(seqLen + 1)` token window they want
each batch row to see.
-/
def causalLmOneHotSampleRows
    {α : Type} [Runtime.SemanticScalar α] [Runtime.Scalar α]
    (batch seqLen vocab : Nat) (tokensAt : Fin batch → List Nat) (padId : Nat := 0) :
    SupervisedSample α (shape![batch, seqLen, vocab]) (shape![batch, seqLen, vocab]) :=
  NN.API.text.causalLmSampleOneHotBatchRows
    (α := α) batch seqLen vocab tokensAt (padId := padId)

/--
Build a batched one-hot causal-language-model sample from an array of per-row token windows.

Rows past the end of the array use the explicit `fallback` window, so partial-batch behavior stays
visible at the call site.
-/
def causalLmOneHotSampleRowsFromArray
    {α : Type} [Runtime.SemanticScalar α] [Runtime.Scalar α]
    (batch seqLen vocab : Nat) (windows : Array (List Nat)) (fallback : List Nat)
    (padId : Nat := 0) :
    SupervisedSample α (shape![batch, seqLen, vocab]) (shape![batch, seqLen, vocab]) :=
  let tokensAt (i : Fin batch) : List Nat :=
    windows.getD i.val fallback
  causalLmOneHotSampleRows (α := α) batch seqLen vocab tokensAt (padId := padId)

/--
Build a batched one-hot causal-language-model sample from a token array by choosing one
deterministic `(seqLen + 1)` window per batch row.

Use this for GPT-style trainers that keep a tokenized corpus in memory and derive each batch from
the same `(tokens, seed, step)` rule.
-/
def causalLmOneHotSampleRowsFromTokenArray
    {α : Type} [Runtime.SemanticScalar α] [Runtime.Scalar α]
    (batch seqLen vocab : Nat) (tokens : Array Nat) (seed step : Nat) (padId : Nat := 0) :
    SupervisedSample α (shape![batch, seqLen, vocab]) (shape![batch, seqLen, vocab]) :=
  let idsAt :=
    text.Corpus.randomBatchTokenWindows tokens batch seqLen seed step (padId := padId)
  causalLmOneHotSampleRows (α := α) batch seqLen vocab idsAt (padId := padId)

/--
Flatten one `(seqLen + 1)` token window into causal-LM `(x, y)` id lists.

For a window $[t_0,t_1,\ldots,t_{\mathrm{seqLen}}]$, the model input is
$[t_0,\ldots,t_{\mathrm{seqLen}-1}]$ and the target is
$[t_1,\ldots,t_{\mathrm{seqLen}}]$. Short windows are padded rather than rejected so tiny toy corpora can still
exercise the training loop.
-/
def causalLmTokenIdRows (seqLen : Nat) (window : List Nat) (padId : Nat := 0) :
    List Nat × List Nat :=
  let x := (List.range seqLen).map (fun i => window.getD i padId)
  let y := (List.range seqLen).map (fun i => window.getD (i + 1) padId)
  (x, y)

/--
Build a float tensor of integer token ids from a flat `List Nat` of length `batch * seqLen`.

The float encoding is deliberately narrow: it is only a transport format for the scalar-module
runtime API. The receiving runtime checks that these floats are still exact, nonnegative integer
token ids before using them as embedding indices.
-/
def causalLmTokenIdFloatVec {α : Type} [Runtime.SemanticScalar α] [Runtime.Scalar α]
    (batch seqLen : Nat) (tokens : List Nat) :
    Tensor.T α (.dim (batch * seqLen) .scalar) :=
  let xF : _root_.Spec.Tensor Float (.dim (batch * seqLen) .scalar) :=
    _root_.Spec.Tensor.dim (fun i : Fin (batch * seqLen) =>
      _root_.Spec.Tensor.scalar (Float.ofNat (tokens.getD i.val 0)))
  Tensor.castFloat Runtime.ofFloat xF

/--
Build a batched token-id causal-language-model sample from an array-backed corpus.

Token ids are passed as float inputs so the training loop can swap windows each step without
re-instantiating the scalar module.

This is the token-id analogue of `causalLmOneHotSampleRowsFromTokenArray`: it keeps the same
sampling policy, but avoids materializing `(batch, seqLen, vocab)` one-hot tensors.
-/
def causalLmTokenIdSampleRowsFromTokenArray
    {α : Type} [Runtime.SemanticScalar α] [Runtime.Scalar α]
    (batch seqLen : Nat) (tokens : Array Nat) (seed step : Nat) (padId : Nat := 0) :
    SupervisedSample α (.dim (batch * seqLen) .scalar) (.dim (batch * seqLen) .scalar) :=
  let idsAt :=
    text.Corpus.randomBatchTokenWindows tokens batch seqLen seed step (padId := padId)
  -- Keep the flattening order explicit: batch rows are concatenated, and each row contributes
  -- `seqLen` input ids plus `seqLen` next-token targets.
  let (xList, yList) :=
    (List.finRange batch).foldl (fun (acc : List Nat × List Nat) bi =>
      let (xs, ys) := acc
      let (xRow, yRow) := causalLmTokenIdRows seqLen (idsAt bi) (padId := padId)
      (xs ++ xRow, ys ++ yRow)) ([], [])
  Sample.mk
    (causalLmTokenIdFloatVec (α := α) batch seqLen xList)
    (causalLmTokenIdFloatVec (α := α) batch seqLen yList)

/--
Build one unbatched one-hot causal-language-model sample directly from a token list.

The token list represents a `seqLen + 1` window. Shorter lists are padded and longer lists are
truncated by the causal-LM construction.
-/
def causalLmOneHotMatSample
    {α : Type} [Runtime.SemanticScalar α] [Runtime.Scalar α]
    (seqLen vocab : Nat) (tokens : List Nat) :
    SupervisedSample α (.dim seqLen (.dim vocab .scalar)) (.dim seqLen (.dim vocab .scalar)) :=
  let (xF, yF) := text.causalLmXYOneHotMatFloat seqLen vocab tokens
  let x : Tensor.T α (.dim seqLen (.dim vocab .scalar)) :=
    Tensor.castFloat Runtime.ofFloat xF
  let y : Tensor.T α (.dim seqLen (.dim vocab .scalar)) :=
    Tensor.castFloat Runtime.ofFloat yF
  Sample.mk x y

/--
Build one unbatched one-hot causal-language-model sample from a text corpus string.

This takes one `(seqLen + 1)` byte window from the UTF-8 bytes of `input`, converts it to one-hot
`x/y` matrices, and casts the result into the runtime-selected scalar.
-/
def textCausalSample
    {α : Type} [Runtime.SemanticScalar α] [Runtime.Scalar α]
    (seqLen vocab : Nat) (input : String) :
    SupervisedSample α (.dim seqLen (.dim vocab .scalar)) (.dim seqLen (.dim vocab .scalar)) :=
  let bytes := input.toUTF8
  let toks := (text.byteTokenWindow bytes (seqLen + 1)).map (fun b => b % vocab)
  causalLmOneHotMatSample (α := α) seqLen vocab toks

/--
Build one fixed-batch one-hot causal-language-model sample from a text corpus string by repeating
the same text window across every batch row.
-/
def textCausalBatchSample
    {α : Type} [Runtime.SemanticScalar α] [Runtime.Scalar α]
    (batch seqLen vocab : Nat) (input : String) :
    SupervisedSample α (_root_.Spec.Shape.dim batch (.dim seqLen (.dim vocab .scalar)))
      (_root_.Spec.Shape.dim batch (.dim seqLen (.dim vocab .scalar))) :=
  let s := textCausalSample (α := α) seqLen vocab input
  Sample.mk
    (_root_.Spec.Tensor.dim (fun _ => Sample.x s))
    (_root_.Spec.Tensor.dim (fun _ => Sample.y s))

/--
Build a runtime-polymorphic dataset containing one unbatched causal-language-model sample from a
text corpus string.
-/
def textCausalDataset
    (seqLen vocab : Nat) (input : String) :
    Trainer.Dataset (.dim seqLen (.dim vocab .scalar)) (.dim seqLen (.dim vocab .scalar)) :=
  Data.singletonFrom input (fun {α} _ _ text =>
    textCausalSample (α := α) seqLen vocab text)

/--
Build a runtime-polymorphic dataset containing one causal-language-model sample repeated across a
fixed batch axis.

Use this when the model itself owns the batch dimension but the example naturally starts from one
text window.
-/
def textCausalBatchDataset
    (batch seqLen vocab : Nat) (input : String) :
    Trainer.Dataset (_root_.Spec.Shape.dim batch (.dim seqLen (.dim vocab .scalar)))
      (_root_.Spec.Shape.dim batch (.dim seqLen (.dim vocab .scalar))) :=
  Data.singletonFrom input (fun {α} _ _ text =>
    textCausalBatchSample (α := α) batch seqLen vocab text)

end Data

end TorchLean
