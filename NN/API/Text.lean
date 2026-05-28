/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

-- shake: keep-all

public import NN.API.Common
public import NN.API.TList
public import NN.Runtime.Autograd.TorchLean.Metrics
public import NN.Runtime.Autograd.TorchLean.Random

import Mathlib.Algebra.Order.Algebra

/-!
# API Text

Text and NLP helpers for TorchLean examples.

TorchLean’s executable runtime expects inputs as floating tensors, so runtime and autograd
code can handle them with the same typed tensor APIs as parameters. For language models this means
we commonly represent
token ids as **one-hot / token-distribution** tensors of shape:

`(batch × seqLen × vocab)`

and implement “token embeddings” as a matrix multiply against an embedding table.

This module provides:
- a tokenizer interface (with a byte-level tokenizer),
- helpers to turn token streams into one-hot tensors,
- “next-token prediction” sample builders used by GPT-style examples,
- display helpers for turning model logits back into readable token predictions.
-/

@[expose] public section

namespace NN
namespace API

namespace text

open Spec Tensor
open NN.Tensor

/-! ## Tokenizers -/

/-- Tokenizer interface (encode/decode). -/
structure Tokenizer where
  /-- Vocabulary size (token ids are expected to be in `[0, vocabSize)`). -/
  vocabSize : Nat
  /-- Encode a string into token ids. -/
  encode : String → List Nat
  /-- Decode token ids back into a string. -/
  decode : List Nat → String

namespace Tokenizer

/-- Convert token ids to bytes, truncating each id modulo 256. -/
def byteArrayOfIds (ids : List Nat) : ByteArray :=
  ids.foldl (fun acc n => acc.push (UInt8.ofNat (n % 256))) ByteArray.empty

/--
Decode byte tokens as UTF-8 when possible, falling back to a byte-wise display mode for generated
byte streams that are not valid UTF-8. This keeps `decode (encode s) = s` for arbitrary UTF-8
strings while remaining total on model output.
-/
def decodeByteIds (ids : List Nat) : String :=
  let bytes := byteArrayOfIds ids
  match String.fromUTF8? bytes with
  | some s => s
  | none => String.ofList (ids.map (fun n => Char.ofNat (n % 256)))

/-- Byte-level UTF-8 tokenizer: each byte is one token in `[0,256)`. -/
def byte : Tokenizer where
  vocabSize := 256
  encode := fun s => (s.toByteArray.toList.map (fun b => b.toNat))
  decode := decodeByteIds

/--
Build a character-level tokenizer from an explicit alphabet.

This is the TorchLean analogue of the `stoi/itos` tables used in character-level GPT examples
(including Karpathy's "char-gpt" / minGPT walkthroughs): `encode` maps characters to ids
`0..alphabet.size-1`, and `decode` maps ids back to characters.

Notes:
- This tokenizer is deterministic given `alphabet`; callers are responsible for choosing how to
  construct the alphabet (e.g. `sorted(set(data))`).
- Characters not present in the alphabet map to `unkId` (default 0). This keeps `encode` total.
- Ids outside `[0, vocabSize)` decode to the `unkChar` (default `?`).
-/
def ofAlphabet (alphabet : Array Char) (unkId : Nat := 0) (unkChar : Char := '?') : Tokenizer :=
  let vocabSize := alphabet.size
  { vocabSize := vocabSize
    encode := fun s =>
      s.toList.map (fun c =>
        match alphabet.findIdx? (fun a => a = c) with
        | some i => i
        | none => Nat.min unkId (Nat.max 0 (vocabSize - 1)))
    decode := fun ids =>
      String.ofList <|
        ids.map (fun n =>
          alphabet.getD n unkChar) }

/-- Encode and pad/truncate to a fixed length, returning a length-indexed `Vector`. -/
def encodeVec (t : Tokenizer) (n : Nat) (s : String) (padId : Nat := 0) : Vector Nat n :=
  let toks := t.encode s
  Vector.ofFn (fun i => toks.getD i.val padId)

/-- Encode a batch of strings, padding/truncating each to length `seqLen`. -/
def encodeBatchVec (t : Tokenizer) (batch seqLen : Nat) (ss : List String) (padId : Nat := 0) :
    Vector (Vector Nat seqLen) batch :=
  Vector.ofFn (fun bi =>
    let s := ss.getD bi.val ""
    encodeVec t seqLen s padId)

end Tokenizer

/-! ## One-Hot Token Tensors -/

/-- One-hot vector for a single token id (`Vec vocab`). Out-of-range ids map to all-zeros. -/
def oneHotTokenFloat (vocab tokenId : Nat) : Tensor Float (Shape.Vec vocab) :=
  NN.Tensor.oneHotNat (α := Float) vocab tokenId

/-- One-hot encode a fixed-length token sequence as a matrix `(seqLen × vocab)`. -/
def tokensToOneHotMatFloat {seqLen vocab : Nat} (tokens : Vector Nat seqLen) :
    Tensor Float (Shape.Mat seqLen vocab) :=
  Tensor.dim (fun t => oneHotTokenFloat vocab (tokens.get t))

/-- One-hot encode a fixed-size batch of token sequences as `(batch × seqLen × vocab)`. -/
def tokensToOneHotBatchFloat {batch seqLen vocab : Nat} (tokens : Vector (Vector Nat seqLen) batch) :
    Tensor Float (.dim batch (Shape.Mat seqLen vocab)) :=
  Tensor.dim (fun bi => tokensToOneHotMatFloat (tokens := tokens.get bi))

/-! ## Causal LM Samples -/

/--
Build a `(x, y)` pair for next-token prediction from a token stream.

`x[t] = oneHot(tokens[t])`
`y[t] = oneHot(tokens[t+1])`

If the stream is too short, we pad with `padId`.
-/
def causalLmXYOneHotMatFloat (seqLen vocab : Nat) (tokens : List Nat) (padId : Nat := 0) :
    Tensor Float (Shape.Mat seqLen vocab) × Tensor Float (Shape.Mat seqLen vocab) :=
  let x : Tensor Float (Shape.Mat seqLen vocab) :=
    Tensor.dim (fun t => oneHotTokenFloat vocab (tokens.getD t.val padId))
  let y : Tensor Float (Shape.Mat seqLen vocab) :=
    Tensor.dim (fun t => oneHotTokenFloat vocab (tokens.getD (t.val + 1) padId))
  (x, y)

/--
Build a batched causal-LM `(x, y)` pair from one token window per batch row.

This is the text analogue of image/tabular minibatching:

- row `i` receives its own token window `tokensAt i`;
- `x[i,t]` is `tokensAt i[t]`;
- `y[i,t]` is `tokensAt i[t+1]`;
- short rows are padded with `padId`.

GPT-style examples share this batching logic. The contract is explicit: a text batch is a typed
tensor of shape `(batch, seqLen, vocab)`, just like the vision loader collates rows into
`(batch, C, H, W)`.
-/
def causalLmXYOneHotBatchRowsFloat (batch seqLen vocab : Nat)
    (tokensAt : Fin batch → List Nat) (padId : Nat := 0) :
    Tensor Float (.dim batch (Shape.Mat seqLen vocab)) ×
      Tensor Float (.dim batch (Shape.Mat seqLen vocab)) :=
  let x : Tensor Float (.dim batch (Shape.Mat seqLen vocab)) :=
    Tensor.dim (fun bi => (causalLmXYOneHotMatFloat seqLen vocab (tokensAt bi) padId).1)
  let y : Tensor Float (.dim batch (Shape.Mat seqLen vocab)) :=
    Tensor.dim (fun bi => (causalLmXYOneHotMatFloat seqLen vocab (tokensAt bi) padId).2)
  (x, y)

/--
One-hot encode a causal-LM input window as a batched tensor.

Token ids are read from `tokens`, missing positions use `padId`, and every batch row receives the
same window. Use `causalLmSampleOneHotBatchRows` when rows should come from different corpus
offsets.
-/
def causalLmXOneHotBatch {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    (batch seqLen vocab : Nat) (tokens : List Nat) (padId : Nat := 0) :
    Tensor α (.dim batch (Shape.Mat seqLen vocab)) :=
  let x2DF : Tensor Float (Shape.Mat seqLen vocab) :=
    Tensor.dim (fun t => oneHotTokenFloat vocab (tokens.getD t.val padId))
  let x2D : Tensor α (Shape.Mat seqLen vocab) :=
    Common.castTensor Runtime.ofFloat x2DF
  Tensor.dim (fun _bi => x2D)

/--
Build a batched supervised next-token sample from a token stream.

The target is shifted by one position: `x[t] = tokens[t]`, `y[t] = tokens[t+1]`. Every batch row
receives the same window, which is useful for prompt evaluation, deterministic checks, and synthetic
sequence tasks.
-/
def causalLmSampleOneHotBatch {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    (batch seqLen vocab : Nat) (tokens : List Nat) (padId : Nat := 0) :
    sample.Supervised α (.dim batch (Shape.Mat seqLen vocab))
      (.dim batch (Shape.Mat seqLen vocab)) :=
  let (x2DF, y2DF) := causalLmXYOneHotMatFloat (seqLen := seqLen) (vocab := vocab)
    tokens (padId := padId)
  let x2D : Tensor α (Shape.Mat seqLen vocab) :=
    Common.castTensor Runtime.ofFloat x2DF
  let y2D : Tensor α (Shape.Mat seqLen vocab) :=
    Common.castTensor Runtime.ofFloat y2DF
  let x : Tensor α (.dim batch (Shape.Mat seqLen vocab)) := Tensor.dim (fun _bi => x2D)
  let y : Tensor α (.dim batch (Shape.Mat seqLen vocab)) := Tensor.dim (fun _bi => y2D)
  sample.mk x y

/--
Build a batched supervised causal-LM sample from one token window per batch row.

Use this for GPT-style minibatches with distinct corpus windows. `causalLmSampleOneHotBatch` remains
useful when every batch row should repeat a fixed prompt or synthetic sequence.
-/
def causalLmSampleOneHotBatchRows {α : Type} [Semantics.Scalar α] [Runtime.Scalar α]
    (batch seqLen vocab : Nat) (tokensAt : Fin batch → List Nat) (padId : Nat := 0) :
    sample.Supervised α (.dim batch (Shape.Mat seqLen vocab))
      (.dim batch (Shape.Mat seqLen vocab)) :=
  let (xF, yF) := causalLmXYOneHotBatchRowsFloat batch seqLen vocab tokensAt padId
  let x : Tensor α (.dim batch (Shape.Mat seqLen vocab)) :=
    Common.castTensor Runtime.ofFloat xF
  let y : Tensor α (.dim batch (Shape.Mat seqLen vocab)) :=
    Common.castTensor Runtime.ofFloat yF
  sample.mk x y

/-! ## Byte-Corpus Windows -/

/--
Read one byte token from a raw corpus, returning `padId` past the end.

This is byte-level rather than BPE-level: examples can train causal language models directly from a
text file without depending on an external tokenizer artifact. GPT-2 BPE support lives in
`NN.API.Text.Bpe`.
-/
def byteAtD (bytes : ByteArray) (i : Nat) (padId : Nat := 0) : Nat :=
  if i < bytes.size then
    (ByteArray.get! bytes i).toNat
  else
    padId

/--
Extract a fixed-length byte-token window from a raw corpus.

`offset` is measured in bytes, not Unicode characters. That is the right behavior for byte-level
causal language modeling and avoids hidden UTF-8 slicing assumptions.
-/
def byteTokenWindow (bytes : ByteArray) (n : Nat) (offset : Nat := 0)
    (padId : Nat := 0) : List Nat :=
  (List.range n).map (fun i => byteAtD bytes (offset + i) padId)

/-! ## Corpus Helpers -/

namespace Corpus

/--
Read a UTF-8 text file with a caller-supplied preparation hint.

The examples pass their executable name and a concrete hint so failures point users to the exact
download or conversion command for that dataset.
-/
def readUtf8File (exeName : String) (path : System.FilePath) (missingHint : String) :
    IO String := do
  if !(← path.pathExists) then
    throw <| IO.userError s!"{exeName}: dataset file not found: {path}\n{missingHint}"
  let txt ← IO.FS.readFile path
  if txt.isEmpty then
    throw <| IO.userError s!"{exeName}: dataset file is empty: {path}"
  pure txt

/--
Read a raw byte corpus and optionally enforce a minimum size.

`allowSmallData` is an explicit override for bounded local runs. Corpus-training commands can set
`minBytes` to the scale they expect and require users to acknowledge smaller local files.
-/
def readByteFile
    (exeName : String) (path : System.FilePath) (allowSmallData : Bool)
    (minBytes seqLen : Nat) : IO ByteArray := do
  if !(← path.pathExists) then
    throw <| IO.userError s!"{exeName}: dataset file not found: {path}"
  let bytes ← IO.FS.readBinFile path
  if bytes.size <= seqLen then
    throw <| IO.userError s!"{exeName}: dataset is too small for a {seqLen}-token window"
  if !allowSmallData && bytes.size < minBytes then
    throw <| IO.userError (
      s!"{exeName}: corpus is {bytes.size} bytes; real GPU training expects at least " ++
      s!"{minBytes} bytes. For bounded local runs, pass --allow-small-data.")
  pure bytes

/--
Parse a text-corpus flag set and return `(text, remainingArgs)`.

Supported forms:
- `--data-file PATH`
- any named alias in `aliases`, such as `("--tiny-shakespeare", path)`
- no data flag, which uses `defaultPath`
-/
partial def takeUtf8Input
    (exeName : String) (defaultPath : System.FilePath)
    (aliases : List (String × System.FilePath)) (missingHint : String) :
    List String → IO (String × List String)
  | [] => do
      let txt ← readUtf8File exeName defaultPath missingHint
      pure (txt, [])
  | "--data-file" :: path :: rest => do
      let txt ← readUtf8File exeName path missingHint
      pure (txt, rest)
  | "--data-file" :: [] =>
      throw <| IO.userError s!"{exeName}: --data-file expects a path"
  | a :: rest => do
      match aliases.find? (fun p => p.1 = a) with
      | some (_, path) => do
          let txt ← readUtf8File exeName path missingHint
          pure (txt, rest)
      | none => do
          let (txt, rest') ← takeUtf8Input exeName defaultPath aliases missingHint rest
          pure (txt, a :: rest')

/-- Deterministic sliding-window offset for a byte corpus. -/
def byteOffset (bytes : ByteArray) (i seqLen : Nat) : Nat :=
  let window := seqLen + 1
  let maxStart := bytes.size - window
  if maxStart == 0 then 0 else (i * seqLen) % maxStart

/-- Deterministic sliding-window offset for an already-tokenized corpus. -/
def tokenOffset (tokens : Array Nat) (i seqLen : Nat) : Nat :=
  let window := seqLen + 1
  let maxStart := tokens.size - window
  if maxStart == 0 then 0 else (i * seqLen) % maxStart

/--
Number of legal start positions for a `(seqLen + 1)` next-token window.

We return at least one start position so bounded corpora stay total; callers can still enforce a
minimum corpus size before training.
-/
def usableTokenStarts (tokenCount seqLen : Nat) : Nat :=
  if tokenCount > seqLen + 1 then tokenCount - seqLen - 1 else 1

/-- Extract a fixed token window from an array-backed token corpus. -/
def tokenArrayWindow (tokens : Array Nat) (n offset : Nat) (padId : Nat := 0) : List Nat :=
  (List.range n).map (fun i => tokens.getD (offset + i) padId)

/--
Deterministic minGPT-style random offsets for one training batch.

The result is a function `Fin batch → Nat`: one corpus start offset per row.  We derive the random
key from `(seed, step)` and then draw row offsets by the row index, so the run is reproducible
without using ambient IO randomness.  This is the text equivalent of a shuffled `DataLoader` epoch.
-/
def randomBatchOffsets (tokenCount seqLen batch seed step : Nat) : Fin batch → Nat :=
  let usable := usableTokenStarts tokenCount seqLen
  let key : UInt64 := _root_.Runtime.Autograd.TorchLean.Random.keyOf seed step
  fun bi => _root_.Runtime.Autograd.TorchLean.Random.sampleNat key bi.val usable

/--
Build token windows for one deterministic random text batch.

Each row gets `seqLen + 1` ids so downstream causal-LM helpers can form both `x` and shifted `y`.
The helper is token-array based, so byte, character, BPE, and synthetic tokenizers can all produce an
`Array Nat` and reuse the same batching semantics.
-/
def randomBatchTokenWindows (tokens : Array Nat) (batch seqLen seed step : Nat)
    (padId : Nat := 0) : Fin batch → List Nat :=
  let offsetAt := randomBatchOffsets tokens.size seqLen batch seed step
  fun bi => tokenArrayWindow tokens (seqLen + 1) (offsetAt bi) padId

/-- Check whether `pat` occurs in `xs` at offset `off`. -/
def startsWithAt (xs pat : Array Nat) (off : Nat) : Bool := Id.run do
  if off + pat.size > xs.size then
    return false
  for j in [0:pat.size] do
    if xs.getD (off + j) 0 ≠ pat.getD j 0 then
      return false
  return true

/-- Find the first offset where `pat` appears in `xs`. -/
def findWindow? (xs pat : Array Nat) : Option Nat := Id.run do
  if pat.isEmpty then
    return some 0
  for i in [0:xs.size] do
    if startsWithAt xs pat i then
      return some i
  return none

/--
Choose training-window offsets, biased toward a prompt occurrence when the corpus contains it.

If the prompt is present in the corpus, a portion of the sampled windows covers nearby text. That
keeps generation reports tied to text the model actually saw during training.
-/
def promptAwareOffsets (tokenCount seqLen windows : Nat) (promptOffset? : Option Nat) : List Nat :=
  match promptOffset? with
  | none =>
      let usable := if tokenCount > seqLen + 1 then tokenCount - seqLen - 1 else 1
      (List.range windows).map (fun i => (i * seqLen) % usable)
  | some off =>
      let usable := if tokenCount > seqLen + 1 then tokenCount - seqLen - 1 else 1
      let start := if off > windows / 4 then off - windows / 4 else 0
      (List.range windows).map (fun i => (start + i) % usable)

end Corpus

/-! ## Causal LM Display Helpers -/

/--
Return a fixed-length token window from a text string.

`offset = 0` is the model prompt window; `offset = 1` is the usual next-token target window for
causal language modeling. Missing tokens are padded with `padId`, matching
`causalLmXYOneHotMatFloat`.
-/
def tokenWindow (t : Tokenizer) (n : Nat) (input : String) (offset : Nat := 0)
    (padId : Nat := 0) : List Nat :=
  let toks := t.encode input
  (List.range n).map (fun i => toks.getD (offset + i) padId)

/-- Decode a fixed token window extracted by `tokenWindow`. -/
def decodeWindow (t : Tokenizer) (n : Nat) (input : String) (offset : Nat := 0)
    (padId : Nat := 0) : String :=
  t.decode (tokenWindow t n input (offset := offset) (padId := padId))

/--
Escape a short text fragment for one-line terminal output.

Display-only: this does not change tokenizer semantics. It keeps examples readable when a predicted
byte sequence contains quotes, backslashes, tabs, or newlines.
-/
def escapeForDisplay (s : String) : String :=
  "\"" ++ (s.replace "\\" "\\\\" |>.replace "\"" "\\\"" |>.replace "\n" "\\n"
    |>.replace "\t" "\\t") ++ "\""

/-! ## Sampling Helpers (Top-k) -/

/-- Shared text-generation flags for GPT-style examples. -/
structure GenerationOptions where
  /-- Prompt used to seed autoregressive generation. -/
  prompt : String
  /-- Number of new tokens to append. -/
  generate : Nat
  /-- Softmax temperature. Must be positive. -/
  temperature : Float
  /-- Top-k cutoff. `1` gives greedy decoding. -/
  topK : Nat
  /-- Penalty subtracted for repeated recent tokens. `0` disables it. -/
  repeatPenalty : Float
  /-- Number of recent tokens considered by the repeat penalty. `0` disables the window. -/
  repeatWindow : Nat
  /-- Deterministic RNG seed for sampling. -/
  seed : Nat
  /-- Restrict generated ids to a model-specific ASCII allow-list. -/
  asciiOnly : Bool
deriving Repr

/-- Defaults for `parseGenerationOptions`. -/
structure GenerationDefaults where
  prompt : String := "First Citizen:"
  generate : Nat := 64
  temperature : Float := 0.85
  topK : Nat := 12
  repeatPenalty : Float := 1.25
  repeatWindow : Nat := 24
  seed : Nat := 0
  asciiOnly : Bool := false
deriving Repr

/-- Parse `--ascii-only`, accepting either a bare flag or `true`/`false` value. -/
def parseAsciiOnlyFlag (exeName : String) (args : List String) :
    Except String (Bool × List String) := do
  let (raw?, args) ← CLI.takeFlagValueOnce args "ascii-only"
  let (bare, args) ← CLI.takeBoolFlagOnce args "ascii-only"
  match raw? with
  | none => pure (bare, args)
  | some v =>
      if v = "true" || v = "1" then
        pure (true, args)
      else if v = "false" || v = "0" then
        pure (false, args)
      else
        throw s!"{exeName}: --ascii-only expects true/false (or 1/0), got {v}"

/--
Parse the generation flags shared by GPT-style examples.

The model file still owns its training/data flags. This helper only handles prompt, sampling, repeat
penalty, deterministic seed, and ASCII restriction.
-/
def parseGenerationOptions (exeName : String) (args : List String)
    (defaults : GenerationDefaults := {}) :
    Except String (GenerationOptions × List String) := do
  let (prompt?, args) ← CLI.takeFlagValueOnce args "prompt"
  let (generate?, args) ← CLI.takeNatFlagOnce args "generate"
  let (temperature?, args) ← CLI.takeFloatFlagOnce args "temperature"
  let (topK?, args) ← CLI.takeNatFlagOnce args "top-k"
  let (repeatPenalty?, args) ← CLI.takeFloatFlagOnce args "repeat-penalty"
  let (repeatWindow?, args) ← CLI.takeNatFlagOnce args "repeat-window"
  let (seed?, args) ← CLI.takeNatFlagOnce args "sample-seed"
  let (asciiOnly, args) ← parseAsciiOnlyFlag exeName args
  let temperature := temperature?.getD defaults.temperature
  if temperature <= 0.0 then
    throw s!"{exeName}: --temperature must be > 0"
  let repeatPenalty := repeatPenalty?.getD defaults.repeatPenalty
  if repeatPenalty < 0.0 then
    throw s!"{exeName}: --repeat-penalty must be >= 0"
  pure ({ prompt := prompt?.getD defaults.prompt
          generate := generate?.getD defaults.generate
          temperature := temperature
          topK := topK?.getD defaults.topK
          repeatPenalty := repeatPenalty
          repeatWindow := repeatWindow?.getD defaults.repeatWindow
          seed := seed?.getD defaults.seed
          asciiOnly := asciiOnly || defaults.asciiOnly }, args)

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
Apply a lightweight repetition penalty by subtracting `repeatPenalty * count(token)` for tokens
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
Decode a matrix of token logits by taking `argmax` independently at each sequence position.

The shape is `(seqLen × vocab)`, i.e. one logits vector per token position. This helper is for
inspection/debugging and is not differentiable.
-/
def argmaxTokenIdsFromLogits {α : Type} [LT α]
    [DecidableRel ((· > ·) : α → α → Prop)]
    {seqLen vocab : Nat} (logits : Tensor α (Shape.Mat seqLen vocab)) : List Nat :=
  match logits with
  | Tensor.dim rows =>
      (List.finRange seqLen).map (fun t =>
        match _root_.Runtime.Autograd.TorchLean.Metrics.argmax? (α := α) (n := vocab) (rows t) with
        | some i => i.val
        | none => 0)

/-- Decode `(seqLen × vocab)` logits as text using a tokenizer. -/
def decodeArgmaxLogits {α : Type} [LT α]
    [DecidableRel ((· > ·) : α → α → Prop)]
    (t : Tokenizer) {seqLen vocab : Nat} (logits : Tensor α (Shape.Mat seqLen vocab)) :
    String :=
  t.decode (argmaxTokenIdsFromLogits (α := α) logits)

/-- Extract `batchIdx` from batched logits and return the per-position argmax token ids. -/
def argmaxTokenIdsFromBatchLogits {α : Type} [LT α]
    [DecidableRel ((· > ·) : α → α → Prop)]
    {batch seqLen vocab : Nat}
    (logits : Tensor α (.dim batch (Shape.Mat seqLen vocab))) (batchIdx : Fin batch) :
    List Nat :=
  match logits with
  | Tensor.dim batches =>
      argmaxTokenIdsFromLogits (α := α) (batches batchIdx)

/-- Decode one batch row of `(batch × seqLen × vocab)` logits as text. -/
def decodeArgmaxBatchLogits {α : Type} [LT α]
    [DecidableRel ((· > ·) : α → α → Prop)]
    (t : Tokenizer) {batch seqLen vocab : Nat}
    (logits : Tensor α (.dim batch (Shape.Mat seqLen vocab))) (batchIdx : Fin batch) :
    String :=
  t.decode (argmaxTokenIdsFromBatchLogits (α := α) logits batchIdx)

/--
Causal (autoregressive) attention mask of shape `(seqLen × seqLen)`.

Entry `(i, j)` is `true` iff `j ≤ i`, meaning position `i` may attend to itself and earlier
positions but not to future positions.
-/
def causalMask (seqLen : Nat) : Tensor Bool (.dim seqLen (.dim seqLen .scalar)) :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      Tensor.scalar (j ≤ i)))

end text

end API
end NN
