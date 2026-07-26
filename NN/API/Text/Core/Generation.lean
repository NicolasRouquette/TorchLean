/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Text.Core.Options

/-!
# Text Generation

Score filtering, top-k sampling, logit extraction, decoding, and causal masks used by language-model examples.
-/

@[expose] public section

namespace NN
namespace API
namespace text

open Spec Spec.Tensor
open NN.Tensor

/--
Return the indices of the top `k` scores (largest first).

This deterministic utility is used by the GPT-style examples. The direct `O(k*vocab)` implementation
is adequate for the vocabulary sizes and top-k values used by these executable examples.
-/
def topKIndices (scores : Array Float) (k : Nat) : List Nat := Id.run do
  let k := Nat.min k scores.size
  let mut used : Array Bool := Array.replicate scores.size false
  let mut out : List Nat := []
  for _ in [0:k] do
    let mut bestIdx := 0
    let mut bestVal := -1.0e30
    for i in [0:scores.size] do
      if !used.getD i false then
        let v := scores.getD i (-1.0e30)
        if v > bestVal then
          bestIdx := i
          bestVal := v
    used := used.set! bestIdx true
    out := out ++ [bestIdx]
  return out

/-- Greedy `argmax` index. -/
def greedyIndex (scores : Array Float) : Nat :=
  (topKIndices scores 1).head?.getD 0

/--
Apply a repetition penalty by subtracting
$\mathrm{repeatPenalty}\,\mathrm{count}(\mathrm{token})$ for tokens
appearing in `recent`.

This is a local sampling heuristic; it is not the same as the presence or frequency penalties used by
hosted APIs, but it gives examples a deterministic way to discourage immediate repetition.
-/
def penalizeRepeats (scores : Array Float) (recent : List Nat) (repeatPenalty : Float) : Array Float :=
  if repeatPenalty <= 0.0 then
    scores
  else
    scores.mapIdx (fun i score =>
      let c := recent.foldl (fun acc t => acc + (if t = i then 1 else 0)) 0
      score - repeatPenalty * Float.ofNat c)

/--
Mask scores by an allow-list predicate (disallowed ids get a very negative score).

This is mainly used by byte-level examples to optionally restrict output to printable ASCII.
-/
def restrictScores (scores : Array Float) (allowId : Nat → Bool) : Array Float :=
  scores.mapIdx (fun i s => if allowId i then s else (-1.0e30))

/-- Apply repeat penalty and an allow-list mask before sampling. -/
def prepareScoresForGeneration (scores : Array Float) (recent : List Nat)
    (repeatPenalty : Float) (allowId : Nat → Bool := fun _ => true) : Array Float :=
  restrictScores (penalizeRepeats scores recent repeatPenalty) allowId

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
def escapeByteIdsForDisplay (ids : List Nat) : String :=
  "\"" ++ String.join (ids.map escapeByteId) ++ "\""

/--
Sample one token id from `scores` using temperature + top-k sampling.

The randomness is deterministic given `(seed, counter)`, so a run with the same flags produces the
same sampled text.
-/
def sampleTopKIndex (scores : Array Float) (temperature : Float) (topK seed counter : Nat) : Nat :=
  Id.run do
    let candidates :=
      if topK = 0 || topK >= scores.size then
        List.range scores.size
      else
        topKIndices scores topK
    let scaled := candidates.map (fun i => (i, scores.getD i (-1.0e30) / temperature))
    let maxScore := scaled.foldl (fun m p => if p.2 > m then p.2 else m) (-1.0e30)
    let weights := scaled.map (fun p => (p.1, _root_.MathFunctions.exp (p.2 - maxScore)))
    let total := weights.foldl (fun acc p => acc + p.2) 0.0
    if total <= 0.0 then
      return candidates.head?.getD 0
    let key := _root_.Runtime.Autograd.TorchLean.Random.keyOf seed counter
    let denom : Nat := (2 : Nat) ^ 32
    let uNat := _root_.Runtime.Autograd.TorchLean.Random.sampleNat key 0 denom
    let u := Float.ofNat uNat / Float.ofNat denom
    let target := u * total
    let init : Nat := candidates.head?.getD 0
    return (weights.foldl
      (fun (acc : Nat × Float) p =>
        let chosen := acc.1
        let cum := acc.2
        let chosen' := if cum <= target then p.1 else chosen
        (chosen', cum + p.2))
      (init, 0.0)).1

/-- Select the next token from prepared logits using greedy or temperature/top-k sampling. -/
def chooseNextToken (scores : Array Float) (opts : GenerationOptions) (counter : Nat)
    (recent : List Nat := []) (allowId : Nat → Bool := fun _ => true) : Nat :=
  let scores := prepareScoresForGeneration scores recent opts.repeatPenalty allowId
  if opts.topK = 1 then
    greedyIndex scores
  else
    sampleTopKIndex scores opts.temperature opts.topK opts.seed counter

/--
Autoregressively extend token ids with a model-provided score callback.

The callback receives the padded context window and the sequence position whose logits should be
used for the next token. The shared policy crops to the last `seqLen` tokens, pads, applies repeat
penalties, samples by top-k/temperature, and appends one token per step.
-/
partial def autoregressiveTokenIds
    (seqLen padId : Nat)
    (promptIds : List Nat)
    (opts : GenerationOptions)
    (scoreWindow : List Nat → Nat → IO (Array Float))
    (allowId : Nat → Bool := fun _ => true)
    (sanitize : Nat → Nat := fun tok => tok) :
    IO (List Nat) := do
  if seqLen = 0 then
    pure promptIds
  else
    let rec loop (ids : List Nat) : Nat → IO (List Nat)
      | 0 => pure ids
      | n + 1 => do
          let generatedSoFar := opts.generate - (n + 1)
          let start := if ids.length > seqLen then ids.length - seqLen else 0
          let window := (ids.drop start).take seqLen
          let predPos := if window.isEmpty then 0 else window.length - 1
          let padded := window ++ List.replicate (seqLen - window.length) padId
          let scores ← scoreWindow padded predPos
          let recent :=
            if opts.repeatWindow = 0 then
              []
            else
              ids.drop (ids.length - Nat.min ids.length opts.repeatWindow)
          let nextTok := sanitize (chooseNextToken scores opts generatedSoFar recent allowId)
          loop (ids ++ [nextTok]) n
    loop promptIds opts.generate

/-- Extract the vocabulary-score row at one sequence position. -/
def logitScoresAt {seqLen vocab : Nat}
    (logits : Spec.Tensor Float (.dim seqLen (.dim vocab .scalar))) (pos : Nat) : Array Float :=
  if h : seqLen = 0 then
    #[]
  else
    let pos : Fin seqLen :=
      ⟨Nat.min pos (seqLen - 1),
        Nat.lt_of_le_of_lt
          (Nat.min_le_right pos (seqLen - 1))
          (Nat.sub_lt (Nat.pos_of_ne_zero h) (by decide))⟩
    match logits with
    | Spec.Tensor.dim rows =>
        match rows pos with
        | Spec.Tensor.dim cols =>
            Array.ofFn (fun j : Fin vocab =>
              match cols j with
              | Spec.Tensor.scalar x => x)

/-- Extract a vocabulary-score row from batched logits. -/
def batchLogitScoresAt {batch seqLen vocab : Nat}
    (logits : Spec.Tensor Float (.dim batch (.dim seqLen (.dim vocab .scalar))))
    (batchIdx : Fin batch) (pos : Nat) : Array Float :=
  match logits with
  | Spec.Tensor.dim batches =>
      logitScoresAt (batches batchIdx) pos

/--
Decode a matrix of token logits by taking `argmax` independently at each sequence position.

The shape is `(seqLen × vocab)`, i.e. one logits vector per token position. This helper is for
inspection/debugging and is not differentiable.
-/
def argmaxTokenIdsFromLogits {α : Type} [LT α]
    [DecidableRel ((· > ·) : α → α → Prop)]
    {seqLen vocab : Nat} (logits : Spec.Tensor α (.dim seqLen (.dim vocab .scalar))) : List Nat :=
  match logits with
  | Spec.Tensor.dim rows =>
      (List.finRange seqLen).map (fun t =>
        match _root_.Runtime.Autograd.TorchLean.Metrics.argmax? (α := α) (n := vocab) (rows t) with
        | some i => i.val
        | none => 0)

/-- Decode `(seqLen × vocab)` logits as text using a tokenizer. -/
def decodeArgmaxLogits {α : Type} [LT α]
    [DecidableRel ((· > ·) : α → α → Prop)]
    (t : Tokenizer) {seqLen vocab : Nat} (logits : Spec.Tensor α (.dim seqLen (.dim vocab .scalar))) :
    String :=
  t.decode (argmaxTokenIdsFromLogits (α := α) logits)

/-- Extract `batchIdx` from batched logits and return the per-position argmax token ids. -/
def argmaxTokenIdsFromBatchLogits {α : Type} [LT α]
    [DecidableRel ((· > ·) : α → α → Prop)]
    {batch seqLen vocab : Nat}
    (logits : Spec.Tensor α (.dim batch (.dim seqLen (.dim vocab .scalar)))) (batchIdx : Fin batch) :
    List Nat :=
  match logits with
  | Spec.Tensor.dim batches =>
      argmaxTokenIdsFromLogits (α := α) (batches batchIdx)

/-- Decode one batch row of `(batch × seqLen × vocab)` logits as text. -/
def decodeArgmaxBatchLogits {α : Type} [LT α]
    [DecidableRel ((· > ·) : α → α → Prop)]
    (t : Tokenizer) {batch seqLen vocab : Nat}
    (logits : Spec.Tensor α (.dim batch (.dim seqLen (.dim vocab .scalar)))) (batchIdx : Fin batch) :
    String :=
  t.decode (argmaxTokenIdsFromBatchLogits (α := α) logits batchIdx)

/--
Causal (autoregressive) attention mask of shape `(seqLen × seqLen)`.

Entry $(i,j)$ is `true` iff $j \leq i$, meaning position $i$ may attend to itself and earlier
positions but not to future positions.
-/
def causalMask (seqLen : Nat) : Spec.Tensor Bool (.dim seqLen (.dim seqLen .scalar)) :=
  Spec.Tensor.dim (fun i =>
    Spec.Tensor.dim (fun j =>
      Spec.Tensor.scalar (j ≤ i)))

end text
end API
end NN
