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

def regressionGrid (lo hi : Float) (count : Nat) (target : Float → Float → Float) :
    Trainer.Dataset (.dim 2 .scalar) (.dim 1 .scalar) :=
  { build := fun {α} _ => pure <|
      let X : Tensor.T Float (.dim (count * count) (.dim 2 .scalar)) :=
        TorchLean.Data.Synthetic.squareGrid lo hi count
      let Y : Tensor.T Float (.dim (count * count) (.dim 1 .scalar)) :=
        TorchLean.Data.Synthetic.regressionTargetsFloat X target
      supervisedFromLeadingAxisFloat (α := α) X Y }

/--
Build a batched one-hot causal-language-model sample by repeating one token window across every
batch row.

The token list represents a `seqLen + 1` window. Shorter lists are padded and longer lists are
truncated by the causal-LM construction.
-/
def causalLmOneHotSample
    {α : Type} [_root_.Context α] [Runtime.FromFloat α]
    (batch seqLen vocab : Nat) (tokens : List Nat) (padId : Nat := 0) :
    SupervisedSample α (shape![batch, seqLen, vocab]) (shape![batch, seqLen, vocab]) :=
  TorchLean.text.causalLmSampleOneHotBatch (α := α) batch seqLen vocab tokens (padId := padId)

/--
Build a batched one-hot causal-language-model sample from one token window per batch row.

Use this for GPT-style examples that already know the per-row `(seqLen + 1)` token window they want
each batch row to see.
-/
def causalLmOneHotSampleRows
    {α : Type} [_root_.Context α] [Runtime.FromFloat α]
    (batch seqLen vocab : Nat) (tokensAt : Fin batch → List Nat) (padId : Nat := 0) :
    SupervisedSample α (shape![batch, seqLen, vocab]) (shape![batch, seqLen, vocab]) :=
  TorchLean.text.causalLmSampleOneHotBatchRows
    (α := α) batch seqLen vocab tokensAt (padId := padId)

/--
Build a batched one-hot causal-language-model sample from an array of per-row token windows.

Rows past the end of the array use the explicit `fallback` window, so partial-batch behavior stays
visible at the call site.
-/
def causalLmOneHotSampleRowsFromArray
    {α : Type} [_root_.Context α] [Runtime.FromFloat α]
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
    {α : Type} [_root_.Context α] [Runtime.FromFloat α]
    (batch seqLen vocab : Nat) (tokens : Array Nat) (seed step : Nat) (padId : Nat := 0) :
    SupervisedSample α (shape![batch, seqLen, vocab]) (shape![batch, seqLen, vocab]) :=
  let idsAt :=
    TorchLean.text.Corpus.randomBatchTokenWindows tokens batch seqLen seed step (padId := padId)
  causalLmOneHotSampleRows (α := α) batch seqLen vocab idsAt (padId := padId)

/--
Flatten one `(seqLen + 1)` token window into causal-LM `(x, y)` id lists.

For a window $[t_0,t_1,\ldots,t_{\mathrm{seqLen}}]$, the model input is
$[t_0,\ldots,t_{\mathrm{seqLen}-1}]$ and the target is
$[t_1,\ldots,t_{\mathrm{seqLen}}]$. Short windows are padded rather than rejected so small corpora
can still exercise the training loop.
-/
def causalLmTokenIdRows (seqLen : Nat) (window : List Nat) (padId : Nat := 0) :
    List Nat × List Nat :=
  let tokenArray := window.toArray
  let x := (List.range seqLen).map (fun i => tokenArray.getD i padId)
  let y := (List.range seqLen).map (fun i => tokenArray.getD (i + 1) padId)
  (x, y)

/--
Materialize row weights for a flattened `(batch × seqLen)` objective.

Extra values are ignored and missing values receive weight zero. The masked sample builder below
constructs exactly `batch * seqLen` entries.
-/
def causalLmRowWeightVec {α : Type} [_root_.Context α] [Runtime.FromFloat α]
    (batch seqLen : Nat) (weights : Array Float) :
    Tensor.T α (.dim (batch * seqLen) .scalar) :=
  let weightsF : _root_.Spec.Tensor Float (.dim (batch * seqLen) .scalar) :=
    _root_.Spec.Tensor.dim (fun i : Fin (batch * seqLen) =>
      _root_.Spec.Tensor.scalar (weights.getD i.val 0))
  Tensor.castFloat Runtime.ofFloat weightsF

/--
Build an indexed-token causal-language-model batch from an array-backed corpus.

The result contains the input and next-token target as separate `Nat` tensors. Keeping their scalar
type discrete avoids one-hot expansion and prevents token ids from entering floating-point
autograd.
-/
def causalLmTokenBatchFromTokenArray
    (batch seqLen : Nat) (tokens : Array Nat) (seed step : Nat) (padId : Nat := 0) :
    Tensor.T Nat (.dim (batch * seqLen) .scalar) ×
      Tensor.T Nat (.dim (batch * seqLen) .scalar) :=
  let idsAt :=
    TorchLean.text.Corpus.randomBatchTokenWindows tokens batch seqLen seed step (padId := padId)
  let (xTokens, yTokens) := Id.run do
    let mut xs := Array.mkEmpty (batch * seqLen)
    let mut ys := Array.mkEmpty (batch * seqLen)
    for bi in List.finRange batch do
      let (xRow, yRow) := causalLmTokenIdRows seqLen (idsAt bi) (padId := padId)
      for token in xRow do
        xs := xs.push token
      for token in yRow do
        ys := ys.push token
    return (xs, ys)
  (Tensor.vectorFromArray (batch * seqLen) xTokens 0,
    Tensor.vectorFromArray (batch * seqLen) yTokens 0)

/--
Read the target-mask row paired with a causal-language-model window.

The input window begins at `offset`, while its first prediction target is the following token.
Consequently the returned row starts at `targetMask[offset + 1]`.
-/
def causalLmTargetMaskRow
    (seqLen : Nat) (targetMask : Array Bool) (offset : Nat) : List Bool :=
  (List.range seqLen).map fun i =>
    targetMask.getD (offset + i + 1) false

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
def causalLmMaskedTokenBatchFromArrays
    {α : Type} [_root_.Context α] [Runtime.FromFloat α]
    (batch seqLen : Nat) (tokens : Array Nat) (targetMask : Array Bool)
    (seed step : Nat) (padId : Nat := 0) :
    Tensor.T Nat (.dim (batch * seqLen) .scalar) ×
      Tensor.T Nat (.dim (batch * seqLen) .scalar) ×
      Tensor.T α (.dim (batch * seqLen) .scalar) :=
  let offsetAt := TorchLean.text.Corpus.randomBatchOffsets tokens.size seqLen batch seed step
  let (xTokens, yTokens, enabledTargets) := Id.run do
    let mut xs := Array.mkEmpty (batch * seqLen)
    let mut ys := Array.mkEmpty (batch * seqLen)
    let mut enabled := Array.mkEmpty (batch * seqLen)
    for bi in List.finRange batch do
      let offset := offsetAt bi
      for i in [0:seqLen] do
        xs := xs.push (tokens.getD (offset + i) padId)
        ys := ys.push (tokens.getD (offset + i + 1) padId)
      for active in causalLmTargetMaskRow seqLen targetMask offset do
        enabled := enabled.push active
    return (xs, ys, enabled)
  let activeCount := enabledTargets.foldl (fun count active =>
    if active then count + 1 else count) 0
  let activeWeight :=
    if activeCount = 0 then 0 else 1.0 / Float.ofNat activeCount
  let rowWeights := enabledTargets.map (fun active => if active then activeWeight else 0)
  (Tensor.vectorFromArray (batch * seqLen) xTokens 0,
    Tensor.vectorFromArray (batch * seqLen) yTokens 0,
    causalLmRowWeightVec (α := α) batch seqLen rowWeights)

/--
Build one unbatched one-hot causal-language-model sample directly from a token list.

The token list represents a `seqLen + 1` window. Shorter lists are padded and longer lists are
truncated by the causal-LM construction.
-/
def causalLmOneHotMatSample
    {α : Type} [_root_.Context α] [Runtime.FromFloat α]
    (seqLen vocab : Nat) (tokens : List Nat) :
    SupervisedSample α (.dim seqLen (.dim vocab .scalar)) (.dim seqLen (.dim vocab .scalar)) :=
  let (xF, yF) := TorchLean.text.causalLmXYOneHotMatFloat seqLen vocab tokens
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
    {α : Type} [_root_.Context α] [Runtime.FromFloat α]
    (seqLen vocab : Nat) (input : String) :
    SupervisedSample α (.dim seqLen (.dim vocab .scalar)) (.dim seqLen (.dim vocab .scalar)) :=
  let bytes := input.toUTF8
  let toks := (TorchLean.text.byteTokenWindow bytes (seqLen + 1)).map (fun b => b % vocab)
  causalLmOneHotMatSample (α := α) seqLen vocab toks

/--
Build one fixed-batch one-hot causal-language-model sample from a text corpus string by repeating
the same text window across every batch row.
-/
def textCausalBatchSample
    {α : Type} [_root_.Context α] [Runtime.FromFloat α]
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
