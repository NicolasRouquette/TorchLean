/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

-- shake: keep-all

public import NN.API.Tensor
public import NN.API.TensorPack
public import NN.Runtime.Autograd.TorchLean.Metrics
public import NN.Runtime.Autograd.TorchLean.Random

import Mathlib.Algebra.Order.Algebra

/-!
# Tokenizers and Text Tensors

Text and NLP helpers for TorchLean examples.

Language models may keep token ids as `Nat` tensors and gather embedding rows directly. Small
examples can instead use one-hot tensors of shape `(batch × seqLen × vocab)`. Both representations
remain separate from floating-point model parameters at the API boundary.

This module provides:
- a tokenizer interface (with a byte-level tokenizer),
- helpers to turn token streams into one-hot tensors,
- “next-token prediction” sample builders used by GPT-style examples,
- display helpers for turning model logits back into readable token predictions.
-/

@[expose] public section

namespace TorchLean
namespace text

open Spec Spec.Tensor
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
byte streams that are not valid UTF-8. For valid UTF-8 strings,
$\operatorname{decode}(\operatorname{encode}(s))=s$; model
output remains printable even when the byte stream is invalid UTF-8.
-/
def decodeByteIds (ids : List Nat) : String :=
  let bytes := byteArrayOfIds ids
  match String.fromUTF8? bytes with
  | some s => s
  | none => String.ofList (ids.map (fun n => Char.ofNat (n % 256)))

/-- Byte-level UTF-8 tokenizer: each byte is one token in $[0,256)$. -/
def byte : Tokenizer where
  vocabSize := 256
  encode := fun s => (s.toByteArray.toList.map (fun b => b.toNat))
  decode := decodeByteIds

/--
Build a character-level tokenizer from an explicit alphabet.

The resulting `encode`/`decode` pair has the same role as the `stoi`/`itos` tables used in
character-level GPT examples: `encode` maps characters to ids
`0..alphabet.size-1`, and `decode` maps ids back to characters.

The `unkId` argument proves that the alphabet is nonempty and identifies the token used for a
character outside the alphabet. Ids outside `[0, alphabet.size)` decode to `unkChar`.
-/
def ofAlphabet (alphabet : Array Char) (unkId : Fin alphabet.size) (unkChar : Char := '?') :
    Tokenizer :=
  let vocabSize := alphabet.size
  { vocabSize := vocabSize
    encode := fun s =>
      s.toList.map (fun c =>
        match alphabet.findIdx? (fun a => a = c) with
        | some i => i
        | none => unkId.val)
    decode := fun ids =>
      String.ofList <|
        ids.map (fun n =>
          alphabet.getD n unkChar) }

/-- Encode a string and pad or truncate it to exactly `n` token ids. -/
def encodeFixed (t : Tokenizer) (n : Nat) (s : String) (padId : Nat := 0) : Vector Nat n :=
  let toks := t.encode s
  Vector.ofFn (fun i => toks.getD i.val padId)

/-- Encode exactly `batch` strings, padding or truncating each row to `seqLen` token ids. -/
def encodeFixedBatch {batch : Nat} (t : Tokenizer) (seqLen : Nat) (ss : Vector String batch)
    (padId : Nat := 0) :
    Vector (Vector Nat seqLen) batch :=
  Vector.ofFn (fun bi =>
    encodeFixed t seqLen (ss.get bi) padId)

end Tokenizer

/-! ## Byte-Corpus Windows -/

/--
Read one byte token from a raw corpus, returning `padId` past the end.

This is byte-level rather than BPE-level: examples can train causal language models directly from a
text file without depending on an external tokenizer artifact. GPT-2 BPE support lives in
`NN.API.Text.Bpe`.
-/
def byteAtOrPad (bytes : ByteArray) (i : Nat) (padId : Nat := 0) : Nat :=
  match bytes[i]? with
  | some b => b.toNat
  | none => padId

/--
Extract a fixed-length byte-token window from a raw corpus.

`offset` is measured in bytes, as required for byte-level causal language modeling. This avoids
hidden UTF-8 slicing assumptions.
-/
def byteTokenWindow (bytes : ByteArray) (n : Nat) (offset : Nat := 0)
    (padId : Nat := 0) : Vector Nat n :=
  Vector.ofFn (fun i => byteAtOrPad bytes (offset + i.val) padId)

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
def takeUtf8Input
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

/-- Choose deterministic, approximately evenly spaced starts for fixed-width token windows. -/
def evenlySpacedOffsets (tokenCount seqLen windows : Nat) : List Nat :=
  let usable := usableTokenStarts tokenCount seqLen
  let stride := Nat.max 1 (usable / Nat.max 1 windows)
  (List.range windows).map fun i => (i * stride) % usable

/-- Extract a fixed token window from an array-backed token corpus. -/
def tokenArrayWindow (tokens : Array Nat) (n offset : Nat) (padId : Nat := 0) : Vector Nat n :=
  Vector.ofFn (fun i => tokens.getD (offset + i.val) padId)

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
    (padId : Nat := 0) : Fin batch → Vector Nat (seqLen + 1) :=
  let offsetAt := randomBatchOffsets tokens.size seqLen batch seed step
  fun bi => tokenArrayWindow tokens (seqLen + 1) (offsetAt bi) padId

/-- Check whether `pat` occurs in `xs` at offset `off`. -/
def startsWithAt (xs pat : Array Nat) (off : Nat) : Bool := Id.run do
  if off + pat.size > xs.size then
    return false
  for j in [0:pat.size] do
    match xs[off + j]?, pat[j]? with
    | some x, some p =>
        if x ≠ p then
          return false
    | _, _ => return false
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
  let usable := usableTokenStarts tokenCount seqLen
  match promptOffset? with
  | none =>
      (List.range windows).map (fun i => (i * seqLen) % usable)
  | some off =>
      let start := if off > windows / 4 then off - windows / 4 else 0
      (List.range windows).map (fun i => (start + i) % usable)

end Corpus

end text
end TorchLean
