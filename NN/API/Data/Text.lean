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

namespace Internal

/-- Recursive implementation behind the list-shaped `oneHotPair` interface. -/
def oneHotPairShape {α : Type} [Zero α] [One α] :
    (leading : Shape) → (seqLen vocab : Nat) →
      Tensor (Fin vocab) (leading.appendDim (seqLen + 1)) →
      Tensor α ((leading.appendDim seqLen).appendDim vocab) ×
        Tensor α ((leading.appendDim seqLen).appendDim vocab)
  | .scalar, seqLen, vocab, tokens =>
      let input : Tensor α [seqLen, vocab] :=
        TorchLean.Tensor.oneHotIndices (α := α) vocab <|
          Tensor.stack 0 fun t => Tensor.full [] (tokens.getScalar t.castSucc)
      let target : Tensor α [seqLen, vocab] :=
        TorchLean.Tensor.oneHotIndices (α := α) vocab <|
          Tensor.stack 0 fun t => Tensor.full [] (tokens.getScalar t.succ)
      (input, target)
  | .dim _ rest, seqLen, vocab, .dim windows =>
      let pairAt := fun i => oneHotPairShape (α := α) rest seqLen vocab (windows i)
      (.dim fun i => (pairAt i).1, .dim fun i => (pairAt i).2)

end Internal

/--
Build one one-hot input/target pair for next-token prediction.

The target is shifted by one token along the final input axis. Any leading dimensions are preserved,
and `Fin vocab` makes every id valid for the chosen vocabulary.
-/
def oneHotPair {α : Type} [Zero α] [One α]
    (leading : List Nat) (seqLen vocab : Nat)
    (tokens : Tensor (Fin vocab) (leading ++ [seqLen + 1])) :
    Tensor α (leading ++ [seqLen, vocab]) × Tensor α (leading ++ [seqLen, vocab]) := by
  let tokens' : Tensor (Fin vocab) ((Shape.ofList leading).appendDim (seqLen + 1)) := by
    simpa only [Shape.ofList_append, Shape.appendDim_eq_concat] using tokens
  simpa only [Shape.ofList_append, Shape.appendDim_eq_concat, Shape.concat_assoc] using
    Internal.oneHotPairShape (α := α) (Shape.ofList leading) seqLen vocab tokens'

/-- One-hot encode every bounded token id along a new final vocabulary dimension. -/
def oneHotInputs
    {α : Type} [Zero α] [One α]
    {shape : List Nat} (vocab : Nat) (tokens : Tensor (Fin vocab) shape) :
    Tensor α (shape ++ [vocab]) := by
  simpa only [Shape.ofList_append, Shape.appendDim_eq_concat] using
    TorchLean.Tensor.oneHotIndices (α := α) vocab tokens

/--
Build a one-hot causal-language-model sample with the listed leading dimensions.

The input and target both have shape `leading ++ [seqLen, vocab]`.
-/
def oneHotSample
    {α : Type} [Zero α] [One α]
    (leading : List Nat) (seqLen vocab : Nat)
    (tokens : Tensor (Fin vocab) (leading ++ [seqLen + 1])) :
    Sample.Supervised α (leading ++ [seqLen, vocab]) (leading ++ [seqLen, vocab]) :=
  let (input, target) := oneHotPair (α := α) leading seqLen vocab tokens
  Sample.mk input target

/--
Build a batched one-hot causal-language-model sample from a token array by choosing one
deterministic `(seqLen + 1)` window per batch row.

Use this for GPT-style trainers that keep a tokenized corpus in memory and derive each batch from
the same `(tokens, seed, step)` rule.
-/
def oneHotBatchFromTokenArray
    {α : Type} [Zero α] [One α]
    (batch seqLen vocab : Nat) (tokens : Array Nat) (seed step : Nat) (padId : Nat := 0) :
    Except String (Sample.Supervised α [batch, seqLen, vocab] [batch, seqLen, vocab]) := do
  let idsAt :=
    TorchLean.text.Corpus.randomBatchTokenWindows tokens batch seqLen seed step (padId := padId)
  let bounded ← TorchLean.Tensor.checkIndices vocab idsAt
  pure (oneHotSample (α := α) [batch] seqLen vocab bounded)

/--
Split one exact `(seqLen + 1)` token window into causal-LM input and target tensors.

For a window $[t_0,t_1,\ldots,t_{\mathrm{seqLen}}]$, the model input is
$[t_0,\ldots,t_{\mathrm{seqLen}-1}]$ and the target is
$[t_1,\ldots,t_{\mathrm{seqLen}}]$. Corpus readers perform any requested padding before constructing
this tensor, so the split itself cannot invent or discard tokens.
-/
def splitWindow {β : Type} {seqLen : Nat} (window : Tensor β [seqLen + 1]) :
    Tensor β [seqLen] × Tensor β [seqLen] :=
  (Spec.Tensor.ofFn (fun i => window.getScalar i.castSucc),
    Spec.Tensor.ofFn (fun i => window.getScalar i.succ))

/--
Build an indexed-token causal-language-model batch from an array-backed corpus.

The result contains the input and next-token target as separate bounded-index tensors. Validation
happens once at this corpus boundary; model code therefore has no out-of-range token case.
-/
def tokenBatch
    (vocab batch seqLen : Nat) (tokens : Array Nat) (seed step : Nat) (padId : Nat := 0) :
    Except String (Tensor (Fin vocab) [batch, seqLen] ×
      Tensor (Fin vocab) [batch, seqLen]) := do
  let idsAt :=
    TorchLean.text.Corpus.randomBatchTokenWindows tokens batch seqLen seed step (padId := padId)
  let bounded ← TorchLean.Tensor.checkIndices vocab idsAt
  let input : Tensor (Fin vocab) [batch, seqLen] :=
    Tensor.stack 0 fun bi =>
      Tensor.stack 0 fun si => Tensor.full [] ((splitWindow (bounded.get bi)).1.getScalar si)
  let target : Tensor (Fin vocab) [batch, seqLen] :=
    Tensor.stack 0 fun bi =>
      Tensor.stack 0 fun si => Tensor.full [] ((splitWindow (bounded.get bi)).2.getScalar si)
  pure (input, target)

namespace Internal

/--
Read the target-mask row paired with a causal-language-model window.

The input window begins at `offset`, while its first prediction target is the following token.
Consequently the returned row starts at `targetMask[offset + 1]`.
-/
def targetMaskRow
    (seqLen : Nat) (targetMask : Array Bool) (offset : Nat) : Tensor Bool [seqLen] :=
  Spec.Tensor.ofFn fun i => targetMask.getD (offset + i.val + 1) false

end Internal

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
    (vocab batch seqLen : Nat) (tokens : Array Nat) (targetMask : Array Bool)
    (seed step : Nat) (padId : Nat := 0) :
    Except String (Tensor (Fin vocab) [batch, seqLen] ×
      Tensor (Fin vocab) [batch, seqLen] ×
      Tensor α [batch, seqLen]) := do
  let offsetAt := TorchLean.text.Corpus.randomBatchOffsets tokens.size seqLen batch seed step
  let windowAt (bi : Fin batch) : Tensor Nat [seqLen + 1] :=
    TorchLean.text.Corpus.tokenArrayWindow tokens (seqLen + 1) (offsetAt bi) padId
  let enabledAt (bi : Fin batch) : Tensor Bool [seqLen] :=
    Internal.targetMaskRow seqLen targetMask (offsetAt bi)
  let bounded ← TorchLean.Tensor.checkIndices vocab <| Tensor.stack 0 windowAt
  let input : Tensor (Fin vocab) [batch, seqLen] :=
    Tensor.stack 0 fun bi =>
      Tensor.stack 0 fun si => Tensor.full [] ((splitWindow (bounded.get bi)).1.getScalar si)
  let target : Tensor (Fin vocab) [batch, seqLen] :=
    Tensor.stack 0 fun bi =>
      Tensor.stack 0 fun si => Tensor.full [] ((splitWindow (bounded.get bi)).2.getScalar si)
  let enabledTargets : Tensor Bool [batch * seqLen] :=
    Spec.Tensor.ofFn fun i =>
      let p := (finProdFinEquiv : Fin batch × Fin seqLen ≃ Fin (batch * seqLen)).symm i
      (enabledAt p.1).getScalar p.2
  let activeCount := enabledTargets.toArray.foldl (fun count active =>
    if active then count + 1 else count) 0
  let activeWeight :=
    if activeCount = 0 then 0 else 1.0 / Float.ofNat activeCount
  let weights : Tensor Float [batch, seqLen] :=
    Tensor.stack 0 fun bi =>
      Tensor.stack 0 fun si =>
        Tensor.full [] (if (enabledAt bi).getScalar si then activeWeight else 0)
  pure (input, target, Tensor.map Runtime.ofFloat weights)

/--
Build one unbatched one-hot causal-language-model sample from a text corpus string.

This takes one `(seqLen + 1)` byte window from the UTF-8 bytes of `input`, converts it to one-hot
`x/y` matrices, and casts the result into the runtime-selected scalar.
-/
def byteSample
    {α : Type} [Zero α] [One α]
    (seqLen vocab : Nat) (encode : Nat → Fin vocab) (input : String) :
    Sample.Supervised α [seqLen, vocab] [seqLen, vocab] :=
  let bytes := input.toUTF8
  let toks := (TorchLean.text.byteTokenWindow bytes (seqLen + 1)).map encode
  oneHotSample (α := α) [] seqLen vocab toks

/--
Build one fixed-batch one-hot causal-language-model sample from a text corpus string by repeating
the same text window across every batch row.
-/
def byteBatch
    {α : Type} [Zero α] [One α]
    (batch seqLen vocab : Nat) (encode : Nat → Fin vocab) (input : String) :
    Sample.Supervised α [batch, seqLen, vocab] [batch, seqLen, vocab] :=
  let s := byteSample (α := α) seqLen vocab encode input
  Sample.mk
    (Tensor.repeatAxis 0 batch (Sample.x s))
    (Tensor.repeatAxis 0 batch (Sample.y s))

end CausalLM

end Data

end TorchLean
