/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Text.Options

/-!
# Text Generation

Score filtering, top-k sampling, logit extraction, decoding, and causal masks used by language-model examples.
-/

@[expose] public section

namespace TorchLean
namespace text

open Spec Spec.Tensor
open TorchLean.Tensor

/--
Return the indices of the top `k` allowed, non-NaN scores, largest first.

The optional predicate filters token ids without replacing their scores by a finite sentinel. This
matters when model logits are unbounded: a disallowed token must never become selectable merely
because every allowed score is smaller than an arbitrary masking constant.
-/
def topKIndices {vocab : Nat} (scores : Tensor Float [vocab]) (k : Nat)
    (allowId : Fin vocab → Bool := fun _ => true) : Array (Fin vocab) := Id.run do
  let k := Nat.min k vocab
  let mut selected : Array (Fin vocab) := #[]
  for _ in [0:k] do
    let mut best? : Option (Fin vocab) := none
    for i in List.finRange vocab do
      let score := scores.getScalar i
      if allowId i && !selected.contains i && !score.isNaN then
        match best? with
        | none => best? := some i
        | some best =>
            if score > scores.getScalar best then
              best? := some i
    match best? with
    | none => pure ()
    | some best =>
        selected := selected.push best
  return selected

/-- Greedy `argmax`, or `none` when no allowed non-NaN score exists. -/
def greedyIndex {vocab : Nat} (scores : Tensor Float [vocab])
    (allowId : Fin vocab → Bool := fun _ => true) : Option (Fin vocab) :=
  (topKIndices scores 1 allowId)[0]?

/--
Apply a repetition penalty by subtracting
$\mathrm{repeatPenalty}\,\mathrm{count}(\mathrm{token})$ for tokens
appearing in `recent`.

This is a local sampling heuristic; it is not the same as the presence or frequency penalties used by
hosted APIs, but it gives examples a deterministic way to discourage immediate repetition.
-/
def penalizeRepeats {vocab : Nat} (scores : Tensor Float [vocab])
    (recent : Array Nat) (repeatPenalty : Float) : Tensor Float [vocab] :=
  if repeatPenalty <= 0.0 then
    scores
  else
    Spec.Tensor.ofFn fun i =>
      let c := recent.foldl (fun acc t => acc + (if t = i.val then 1 else 0)) 0
      scores.getScalar i - repeatPenalty * Float.ofNat c

/-- Printable ASCII bytes plus newline. -/
def printableAsciiByte (i : Nat) : Bool :=
  i = 10 || (32 ≤ i && i ≤ 126)

/-- Escape one byte token for display inside a quoted string. -/
def escapeByteId (b : Nat) : String :=
  if b = 10 then "\\n"
  else if b = 9 then "\\t"
  else if b = 34 then "\\\""
  else if b = 92 then "\\\\"
  else if 32 ≤ b && b ≤ 126 then
    String.singleton (Char.ofNat b)
  else
    let hex : Array Char := #['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
    let hi := (b / 16) % 16
    let lo := b % 16
    "\\x" ++ String.singleton (hex.getD hi '0') ++ String.singleton (hex.getD lo '0')

/-- Escape byte ids as a one-line quoted display string. -/
def escapeByteIdsForDisplay (ids : Array Nat) : String :=
  "\"" ++ String.join (ids.toList.map escapeByteId) ++ "\""

/--
Sample an allowed token id using temperature and top-k sampling.

NaN scores are excluded. An infinite maximum is selected directly because subtracting it during
softmax normalization would produce NaN. The randomness is deterministic given `(seed, counter)`.
-/
def sampleTopKIndex {vocab : Nat} (scores : Tensor Float [vocab])
    (temperature : Float) (topK seed counter : Nat)
    (allowId : Fin vocab → Bool := fun _ => true) : Option (Fin vocab) :=
  Id.run do
    if temperature.isNaN || temperature <= 0.0 then
      return none
    let candidateCount := if topK = 0 then vocab else topK
    let candidates := topKIndices scores candidateCount allowId
    let some first := candidates[0]? | return none
    let maxScore := scores.getScalar first / temperature
    if maxScore.isInf then
      return some first
    let scaled := candidates.map (fun i => (i, scores.getScalar i / temperature))
    let weights := scaled.map (fun p => (p.1, _root_.MathFunctions.exp (p.2 - maxScore)))
    let total := weights.foldl (fun acc p => acc + p.2) 0.0
    if total.isNaN || total <= 0.0 then
      return some first
    let key := _root_.Runtime.Autograd.TorchLean.Random.keyOf seed counter
    let denom : Nat := (2 : Nat) ^ 32
    let uNat := _root_.Runtime.Autograd.TorchLean.Random.sampleNat key 0 denom
    let u := Float.ofNat uNat / Float.ofNat denom
    let target := u * total
    let mut cumulative := 0.0
    for (token, weight) in weights do
      cumulative := cumulative + weight
      if target < cumulative then
        return some token
    return some first

/-- Select the next token, rejecting an empty allow-list or invalid sampling temperature. -/
def chooseNextToken {vocab : Nat} (scores : Tensor Float [vocab])
    (opts : GenerationOptions) (counter : Nat) (recent : Array Nat := #[])
    (allowId : Fin vocab → Bool := fun _ => true) : Except String (Fin vocab) := do
  let scores := penalizeRepeats scores recent opts.repeatPenalty
  let selected? :=
    if opts.topK = 1 then
      greedyIndex scores allowId
    else
      sampleTopKIndex scores opts.temperature opts.topK opts.seed counter allowId
  match selected? with
  | some token => pure token
  | none => throw "no allowed non-NaN token score is available for generation"

/--
Autoregressively extend token ids with a model-provided score callback.

The callback receives an exact-length context window and the sequence position whose logits should
be used for the next token. The shared policy crops to the last `seqLen` tokens, pads, applies repeat
penalties, samples by top-k/temperature, and appends one token per step.
-/
def autoregressiveTokenIds {vocab : Nat}
    (seqLen padId : Nat)
    (promptIds : Array Nat)
    (opts : GenerationOptions)
    (scoreWindow : Tensor Nat [seqLen] → Fin seqLen → IO (Tensor Float [vocab]))
    (allowId : Fin vocab → Bool := fun _ => true) :
    IO (Array Nat) := do
  if hSeqLen : seqLen = 0 then
    pure promptIds
  else
    let rec loop (ids : Array Nat) : Nat → IO (Array Nat)
      | 0 => pure ids
      | n + 1 => do
          let generatedSoFar := opts.generate - (n + 1)
          let start := if ids.size > seqLen then ids.size - seqLen else 0
          let window := (ids.drop start).take seqLen
          let predPos : Fin seqLen :=
            ⟨Nat.min (if window.isEmpty then 0 else window.size - 1) (seqLen - 1),
              Nat.lt_of_le_of_lt
                (Nat.min_le_right _ (seqLen - 1))
                (Nat.sub_lt (Nat.pos_of_ne_zero hSeqLen) (by decide))⟩
          let padded : Tensor Nat [seqLen] :=
            Spec.Tensor.ofFn (fun i => window.getD i.val padId)
          let scores ← scoreWindow padded predPos
          let recent :=
            if opts.repeatWindow = 0 then
              #[]
            else
              ids.drop (ids.size - Nat.min ids.size opts.repeatWindow)
          let nextTok ←
            match chooseNextToken scores opts generatedSoFar recent allowId with
            | .ok token => pure token
            | .error message => throw (IO.userError message)
          loop (ids.push nextTok.val) n
    loop promptIds opts.generate

/-- Extract the vocabulary-score row at one statically valid sequence position. -/
def logitScoresAt {α : Type} {seqLen vocab : Nat}
    (logits : Tensor α [seqLen, vocab])
    (pos : Fin seqLen) : Tensor α [vocab] :=
  Tensor.get logits pos

/-- Extract a vocabulary-score row from batched logits. -/
def batchLogitScoresAt {α : Type} {batch seqLen vocab : Nat}
    (logits : Tensor α [batch, seqLen, vocab])
    (batchIdx : Fin batch) (pos : Fin seqLen) : Tensor α [vocab] :=
  logitScoresAt (Tensor.get logits batchIdx) pos

/--
Decode a matrix of token logits by taking `argmax` independently at each sequence position.

The shape is `(seqLen × vocab)`, i.e. one logits vector per token position. This helper is for
inspection/debugging and is not differentiable.
-/
def argmaxTokens {α : Type} [LT α]
    [DecidableRel ((· > ·) : α → α → Prop)]
    {seqLen vocab : Nat} (logits : Tensor α [seqLen, vocab]) : Array Nat :=
  Array.ofFn fun t : Fin seqLen =>
    match _root_.TorchLean.Metrics.argmax? (α := α) (Tensor.get logits t) with
    | some i => i.val
    | none => 0

/-- Decode `(seqLen × vocab)` logits as text using a tokenizer. -/
def decodeArgmaxLogits {α : Type} [LT α]
    [DecidableRel ((· > ·) : α → α → Prop)]
    (t : Tokenizer) {seqLen vocab : Nat} (logits : Tensor α [seqLen, vocab]) :
    String :=
  t.decode (argmaxTokens (α := α) logits)

/-- Extract `batchIdx` from batched logits and return the per-position argmax token ids. -/
def argmaxBatchTokens {α : Type} [LT α]
    [DecidableRel ((· > ·) : α → α → Prop)]
    {batch seqLen vocab : Nat}
    (logits : Tensor α [batch, seqLen, vocab]) (batchIdx : Fin batch) :
    Array Nat :=
  argmaxTokens (α := α) (Tensor.get logits batchIdx)

/-- Decode one batch row of `(batch × seqLen × vocab)` logits as text. -/
def decodeArgmaxBatchLogits {α : Type} [LT α]
    [DecidableRel ((· > ·) : α → α → Prop)]
    (t : Tokenizer) {batch seqLen vocab : Nat}
    (logits : Tensor α [batch, seqLen, vocab]) (batchIdx : Fin batch) :
    String :=
  t.decode (argmaxBatchTokens (α := α) logits batchIdx)

end text
end TorchLean
