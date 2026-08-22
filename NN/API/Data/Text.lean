/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Data.Training
public import NN.API.Text.Tokenizer

/-!
# Text Datasets

Causal language-model sample and dataset constructors.
-/

@[expose] public section

namespace TorchLean

namespace Data

namespace CausalLM

/--
Build one one-hot input/target pair for next-token prediction.

The target is shifted by one token. Short windows are padded with `padId`.
Any token id outside the vocabulary becomes a zero row. Indexed-token models validate ids before
gathering and should be preferred for nontrivial language models.
-/
def oneHotPair
    {α : Type} [Zero α] [One α]
    (seqLen vocab : Nat) (tokens : List Nat) (padId : Nat := 0) :
    Tensor α (shape![seqLen, vocab]) × Tensor α (shape![seqLen, vocab]) :=
  let input : Tensor α (shape![seqLen, vocab]) :=
    NN.Tensor.oneHotIndicesOrZero (α := α) vocab <|
      _root_.Spec.Tensor.dim fun t =>
        _root_.Spec.Tensor.scalar (tokens.getD t.val padId)
  let target : Tensor α (shape![seqLen, vocab]) :=
    NN.Tensor.oneHotIndicesOrZero (α := α) vocab <|
      _root_.Spec.Tensor.dim fun t =>
        _root_.Spec.Tensor.scalar (tokens.getD (t.val + 1) padId)
  (input, target)

/-- One-hot encode one causal-LM input window and repeat it across the batch axis. -/
def oneHotInputs
    {α : Type} [Zero α] [One α]
    (batch seqLen vocab : Nat) (tokens : List Nat) (padId : Nat := 0) :
    Tensor α (shape![batch, seqLen, vocab]) :=
  let (input, _) := oneHotPair (α := α) seqLen vocab tokens padId
  _root_.Spec.Tensor.dim fun _ => input

/-- One-hot encode a distinct causal-LM input window for each batch row. -/
def oneHotInputsRows
    {α : Type} [Zero α] [One α]
    (batch seqLen vocab : Nat) (tokensAt : Fin batch → List Nat) (padId : Nat := 0) :
    Tensor α (shape![batch, seqLen, vocab]) :=
  _root_.Spec.Tensor.dim fun i =>
    (oneHotPair (α := α) seqLen vocab (tokensAt i) padId).1

/--
Build a batched one-hot causal-language-model sample by repeating one token window across every
batch row.

The token list represents a `seqLen + 1` window. Shorter lists are padded and longer lists are
truncated by the causal-LM construction. A token id outside the vocabulary becomes a zero row.
-/
def oneHotBatch
    {α : Type} [Zero α] [One α]
    (batch seqLen vocab : Nat) (tokens : List Nat) (padId : Nat := 0) :
    Sample.Supervised α (shape![batch, seqLen, vocab]) (shape![batch, seqLen, vocab]) :=
  let (input, target) := oneHotPair (α := α) seqLen vocab tokens padId
  Sample.mk
    (_root_.Spec.Tensor.dim fun _ => input)
    (_root_.Spec.Tensor.dim fun _ => target)

/--
Build a batched one-hot causal-language-model sample from one token window per batch row.

Use this for GPT-style examples that already know the per-row `(seqLen + 1)` token window they want
each batch row to see. A token id outside the vocabulary becomes a zero row.
-/
def oneHotBatchRows
    {α : Type} [Zero α] [One α]
    (batch seqLen vocab : Nat) (tokensAt : Fin batch → List Nat) (padId : Nat := 0) :
    Sample.Supervised α (shape![batch, seqLen, vocab]) (shape![batch, seqLen, vocab]) :=
  Sample.mk
    (_root_.Spec.Tensor.dim fun i =>
      (oneHotPair (α := α) seqLen vocab (tokensAt i) padId).1)
    (_root_.Spec.Tensor.dim fun i =>
      (oneHotPair (α := α) seqLen vocab (tokensAt i) padId).2)

/--
Build a batched one-hot causal-language-model sample from a token array by choosing one
deterministic `(seqLen + 1)` window per batch row.

Use this for GPT-style trainers that keep a tokenized corpus in memory and derive each batch from
the same `(tokens, seed, step)` rule.
-/
def oneHotBatchFromTokenArray
    {α : Type} [Zero α] [One α]
    (batch seqLen vocab : Nat) (tokens : Array Nat) (seed step : Nat) (padId : Nat := 0) :
    Sample.Supervised α (shape![batch, seqLen, vocab]) (shape![batch, seqLen, vocab]) :=
  let idsAt :=
    TorchLean.text.Corpus.randomBatchTokenWindows tokens batch seqLen seed step (padId := padId)
  oneHotBatchRows (α := α) batch seqLen vocab (fun i => (idsAt i).toList) (padId := padId)

/--
Split one exact `(seqLen + 1)` token window into causal-LM input and target vectors.

For a window $[t_0,t_1,\ldots,t_{\mathrm{seqLen}}]$, the model input is
$[t_0,\ldots,t_{\mathrm{seqLen}-1}]$ and the target is
$[t_1,\ldots,t_{\mathrm{seqLen}}]$. Corpus readers perform any requested padding before constructing
this vector, so the split itself cannot invent or discard tokens.
-/
def splitWindow {seqLen : Nat} (window : Vector Nat (seqLen + 1)) :
    Vector Nat seqLen × Vector Nat seqLen :=
  (Vector.ofFn (fun i => window.get i.castSucc),
    Vector.ofFn (fun i => window.get i.succ))

/--
Build an indexed-token causal-language-model batch from an array-backed corpus.

The result contains the input and next-token target as separate `Nat` tensors. Keeping their scalar
type discrete avoids one-hot expansion and prevents token ids from entering floating-point
autograd.
-/
def tokenBatch
    (batch seqLen : Nat) (tokens : Array Nat) (seed step : Nat) (padId : Nat := 0) :
    Tensor Nat (.dim batch (.dim seqLen .scalar)) ×
      Tensor Nat (.dim batch (.dim seqLen .scalar)) :=
  let idsAt :=
    TorchLean.text.Corpus.randomBatchTokenWindows tokens batch seqLen seed step (padId := padId)
  let input : Tensor Nat (.dim batch (.dim seqLen .scalar)) :=
    _root_.Spec.Tensor.dim fun bi =>
      _root_.Spec.Tensor.vector fun si => (splitWindow (idsAt bi)).1.get si
  let target : Tensor Nat (.dim batch (.dim seqLen .scalar)) :=
    _root_.Spec.Tensor.dim fun bi =>
      _root_.Spec.Tensor.vector fun si => (splitWindow (idsAt bi)).2.get si
  (input, target)

/--
Read the target-mask row paired with a causal-language-model window.

The input window begins at `offset`, while its first prediction target is the following token.
Consequently the returned row starts at `targetMask[offset + 1]`.
-/
def targetMaskRow
    (seqLen : Nat) (targetMask : Array Bool) (offset : Nat) : Vector Bool seqLen :=
  Vector.ofFn fun i => targetMask.getD (offset + i.val + 1) false

/--
Build a token-id causal-language-model batch with an explicit target mask.

`targetMask[i]` states whether token `tokens[i]` should contribute when it is used as a next-token
target. The mask therefore shifts with the labels: a row containing
`tokens[offset], ..., tokens[offset + seqLen]` receives target weights from
`targetMask[offset + 1], ..., targetMask[offset + seqLen]`.

Active rows receive weight `1 / activeCount`, so the resulting weighted cross entropy is a mean over
exactly the selected targets. If a sampled batch has no active target, all weights are zero and the
loss is zero. A shorter mask is treated as false beyond its end.
-/
def maskedTokenBatch
    {α : Type} [_root_.Context α] [Runtime.FromFloat α]
    (batch seqLen : Nat) (tokens : Array Nat) (targetMask : Array Bool)
    (seed step : Nat) (padId : Nat := 0) :
    Tensor Nat (.dim batch (.dim seqLen .scalar)) ×
      Tensor Nat (.dim batch (.dim seqLen .scalar)) ×
      Tensor α (.dim batch (.dim seqLen .scalar)) :=
  let offsetAt := TorchLean.text.Corpus.randomBatchOffsets tokens.size seqLen batch seed step
  let windowAt (bi : Fin batch) : Vector Nat (seqLen + 1) :=
    TorchLean.text.Corpus.tokenArrayWindow tokens (seqLen + 1) (offsetAt bi) padId
  let enabledAt (bi : Fin batch) : Vector Bool seqLen :=
    targetMaskRow seqLen targetMask (offsetAt bi)
  let input : Tensor Nat (.dim batch (.dim seqLen .scalar)) :=
    _root_.Spec.Tensor.dim fun bi =>
      _root_.Spec.Tensor.vector fun si => (splitWindow (windowAt bi)).1.get si
  let target : Tensor Nat (.dim batch (.dim seqLen .scalar)) :=
    _root_.Spec.Tensor.dim fun bi =>
      _root_.Spec.Tensor.vector fun si => (splitWindow (windowAt bi)).2.get si
  let enabledTargets : Vector Bool (batch * seqLen) :=
    Vector.ofFn fun i =>
      let p := (finProdFinEquiv : Fin batch × Fin seqLen ≃ Fin (batch * seqLen)).symm i
      (enabledAt p.1).get p.2
  let activeCount := enabledTargets.toArray.foldl (fun count active =>
    if active then count + 1 else count) 0
  let activeWeight :=
    if activeCount = 0 then 0 else 1.0 / Float.ofNat activeCount
  let weights : Tensor Float (.dim batch (.dim seqLen .scalar)) :=
    _root_.Spec.Tensor.dim fun bi =>
      _root_.Spec.Tensor.vector fun si =>
        if (enabledAt bi).get si then activeWeight else 0
  (input, target, Tensor.map Runtime.ofFloat weights)

/--
Build one unbatched one-hot causal-language-model sample directly from a token list.

The token list represents a `seqLen + 1` window. Shorter lists are padded and longer lists are
truncated by the causal-LM construction.
-/
def oneHotSample
    {α : Type} [Zero α] [One α]
    (seqLen vocab : Nat) (tokens : List Nat) :
    Sample.Supervised α (.dim seqLen (.dim vocab .scalar)) (.dim seqLen (.dim vocab .scalar)) :=
  let (input, target) := oneHotPair (α := α) seqLen vocab tokens
  Sample.mk input target

/--
Build one unbatched one-hot causal-language-model sample from a text corpus string.

This takes one `(seqLen + 1)` byte window from the UTF-8 bytes of `input`, converts it to one-hot
`x/y` matrices, and casts the result into the runtime-selected scalar.
-/
def byteSample
    {α : Type} [Zero α] [One α]
    (seqLen vocab : Nat) (input : String) :
    Sample.Supervised α (.dim seqLen (.dim vocab .scalar)) (.dim seqLen (.dim vocab .scalar)) :=
  let bytes := input.toUTF8
  let toks := (TorchLean.text.byteTokenWindow bytes (seqLen + 1)).map (fun b => b % vocab)
  oneHotSample (α := α) seqLen vocab toks.toList

/--
Build one fixed-batch one-hot causal-language-model sample from a text corpus string by repeating
the same text window across every batch row.
-/
def byteBatch
    {α : Type} [Zero α] [One α]
    (batch seqLen vocab : Nat) (input : String) :
    Sample.Supervised α (_root_.Spec.Shape.dim batch (.dim seqLen (.dim vocab .scalar)))
      (_root_.Spec.Shape.dim batch (.dim seqLen (.dim vocab .scalar))) :=
  let s := byteSample (α := α) seqLen vocab input
  Sample.mk
    (_root_.Spec.Tensor.dim (fun _ => Sample.x s))
    (_root_.Spec.Tensor.dim (fun _ => Sample.y s))

/--
Build a runtime-polymorphic dataset containing one unbatched causal-language-model sample from a
text corpus string.
-/
def byteDataset
    (seqLen vocab : Nat) (input : String) :
    Trainer.DataSource (.dim seqLen (.dim vocab .scalar)) (.dim seqLen (.dim vocab .scalar)) :=
  Data.singletonFrom input (fun {α} _ _ text =>
    byteSample (α := α) seqLen vocab text)

/--
Build a runtime-polymorphic dataset containing one causal-language-model sample repeated across a
fixed batch axis.

Use this when the model itself owns the batch dimension but the example naturally starts from one
text window.
-/
def byteBatchDataset
    (batch seqLen vocab : Nat) (input : String) :
    Trainer.DataSource (_root_.Spec.Shape.dim batch (.dim seqLen (.dim vocab .scalar)))
      (_root_.Spec.Shape.dim batch (.dim seqLen (.dim vocab .scalar))) :=
  Data.singletonFrom input (fun {α} _ _ text =>
    byteBatch (α := α) batch seqLen vocab text)

end CausalLM

end Data

end TorchLean
