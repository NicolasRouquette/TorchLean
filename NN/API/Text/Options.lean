/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Text.Tokenizer
public import NN.API.CLI.Training

/-!
# Text Workflow Configuration

Display helpers, generation and corpus option records, training-log metadata, and CLI parsers for text workflows.
-/

@[expose] public section

namespace TorchLean
namespace text

/-! ## Causal LM Display Helpers -/

/--
Return a fixed-length token window from a text string.

`offset = 0` is the model prompt window; `offset = 1` is the usual next-token target window for
causal language modeling. Missing tokens are padded with `padId`, matching
`Data.CausalLM.oneHotPair`.
-/
def tokenWindow (t : Tokenizer) (n : Nat) (input : String) (offset : Nat := 0)
    (padId : Nat := 0) : Tensor Nat [n] :=
  let toks := t.encode input
  Spec.Tensor.ofFn (fun i => toks.getD (offset + i.val) padId)

/-- Decode a fixed token window extracted by `tokenWindow`. -/
def decodeWindow (t : Tokenizer) (n : Nat) (input : String) (offset : Nat := 0)
    (padId : Nat := 0) : String :=
  t.decode (tokenWindow t n input (offset := offset) (padId := padId)).toArray

/--
Escape a short text fragment for one-line terminal output.

Display-only: this does not change tokenizer semantics. Quotes and backslashes use their usual
escapes, common whitespace controls use `\\n`, `\\r`, and `\\t`, and every other ASCII control
character is written as `\\xNN`. Thus byte-token predictions cannot turn a log into a binary file.
-/
def escapeForDisplay (s : String) : String :=
  let hexDigit := fun n =>
    (#['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']).getD
      (n % 16) '0'
  let escapeChar := fun c =>
    match c with
    | '\\' => "\\\\"
    | '"' => "\\\""
    | '\n' => "\\n"
    | '\r' => "\\r"
    | '\t' => "\\t"
    | _ =>
        let n := c.toNat
        if n < 32 || n = 127 then
          "\\x" ++ String.singleton (hexDigit (n / 16)) ++ String.singleton (hexDigit n)
        else
          String.singleton c
  "\"" ++ String.join (s.toList.map escapeChar) ++ "\""

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
  match TorchLean.CLI.takeSwitchDefault args "ascii-only" false with
  | .ok result => pure result
  | .error e => throw s!"{exeName}: {e}"

/--
Parse the generation flags shared by GPT-style examples.

The model file still owns its training/data flags. This helper only handles prompt, sampling, repeat
penalty, deterministic seed, and ASCII restriction.
-/
def parseGenerationOptions (exeName : String) (args : List String)
    (defaults : GenerationDefaults := {}) :
    Except String (GenerationOptions × List String) := do
  let (prompt, args) ← TorchLean.CLI.takeFlagValueDefault args "prompt" defaults.prompt
  let (generate, args) ← TorchLean.CLI.takeNatFlagDefault args "generate" defaults.generate
  let (temperature, args) ←
    TorchLean.CLI.takePositiveFloatFlag args exeName "temperature" defaults.temperature
  let (topK, args) ← TorchLean.CLI.takeNatFlagDefault args "top-k" defaults.topK
  let (repeatPenalty, args) ←
    TorchLean.CLI.takeNonnegativeFloatFlag args exeName "repeat-penalty" defaults.repeatPenalty
  let (repeatWindow, args) ← TorchLean.CLI.takeNatFlagDefault args "repeat-window" defaults.repeatWindow
  let (seed, args) ← TorchLean.CLI.takeNatFlagDefault args "sample-seed" defaults.seed
  let (asciiOnly, args) ← parseAsciiOnlyFlag exeName args
  pure ({ prompt := prompt
          generate := generate
          temperature := temperature
          topK := topK
          repeatPenalty := repeatPenalty
          repeatWindow := repeatWindow
          seed := seed
          asciiOnly := asciiOnly || defaults.asciiOnly }, args)

namespace GenerationOptions

/- Convert a concrete generation option record back to parser defaults. -/
def toDefaults (opts : GenerationOptions) : GenerationDefaults :=
  { prompt := opts.prompt
    generate := opts.generate
    temperature := opts.temperature
    topK := opts.topK
    repeatPenalty := opts.repeatPenalty
    repeatWindow := opts.repeatWindow
    seed := opts.seed
    asciiOnly := opts.asciiOnly }

/--
Parse generation flags using a full `GenerationOptions` value as defaults.

This is the public API shape used by model commands: they provide a concrete default prompt and
sampling policy, and the shared parser handles the stable CLI surface.
-/
def parse
    (exeName : String)
    (args : List String)
    (defaults : GenerationOptions) :
    Except String (GenerationOptions × List String) :=
  parseGenerationOptions exeName args defaults.toDefaults

end GenerationOptions

/-! ## Text Workflow Option Records -/

/-- Required text-corpus path plus the explicit small-data option used by local corpus trainers. -/
structure TextCorpusOptions where
  /-- UTF-8 or raw-byte corpus path selected by `--data-file`. -/
  dataFile : System.FilePath
  /-- Allow local runs below the normal corpus-size floor. -/
  allowSmallData : Bool
deriving Repr

namespace TextCorpusOptions

/-- Parse the required `--data-file` corpus flag and optional `--allow-small-data` switch. -/
def parse
    (exeName : String)
    (args : List String) :
    Except String (TextCorpusOptions × List String) := do
  let (dataFile, args) ← TorchLean.CLI.takeRequiredPathFlag args "data-file" (exeName := exeName)
  let (allowSmallData, args) ← TorchLean.CLI.takeBoolFlagOnce args "allow-small-data"
  pure ({ dataFile := dataFile, allowSmallData := allowSmallData }, args)

end TextCorpusOptions

/-- Optional text-corpus path selected by `--data-file`, with caller-supplied default. -/
structure TextCorpusPathOptions where
  /-- Local text corpus path. -/
  path : System.FilePath
deriving Repr

namespace TextCorpusPathOptions

/-- Parse an optional `--data-file` flag using the supplied default path. -/
def parse
    (args : List String)
    (defaultPath : System.FilePath) :
    Except String (TextCorpusPathOptions × List String) := do
  let (path, args) ← TorchLean.CLI.takePathFlagDefault args "data-file" defaultPath
  pure ({ path := path }, args)

end TextCorpusPathOptions

/-- Optional second corpus pass after the main training run. -/
structure FinetuneOptions where
  /-- Optional corpus used for a second fine-tuning pass. -/
  finetuneFile? : Option System.FilePath
  /-- Number of optimizer steps used on that second corpus when present. -/
  finetuneSteps : Nat
deriving Repr

namespace FinetuneOptions

/--
Parse the optional `--finetune-file` / `--finetune-steps` pair.

The caller supplies the default step count so commands can reuse their main training-step default.
-/
def parse
    (args : List String)
    (defaultSteps : Nat) :
    Except String (FinetuneOptions × List String) := do
  let (finetuneFile?, args) ← TorchLean.CLI.takePathFlagOnce args "finetune-file"
  let (finetuneSteps, args) ← TorchLean.CLI.takeNatFlagDefault args "finetune-steps" defaultSteps
  pure ({ finetuneFile? := finetuneFile?
          finetuneSteps := finetuneSteps }, args)

end FinetuneOptions

/-- Optional GPT-2 BPE tokenizer bundle plus an optional bounded-text cap. -/
structure BpeCorpusOptions where
  /-- Optional GPT-2 `vocab.json` path. Must be paired with `bpeMerges?`. -/
  bpeVocab? : Option System.FilePath
  /-- Optional GPT-2 `merges.txt` path. Must be paired with `bpeVocab?`. -/
  bpeMerges? : Option System.FilePath
  /-- Optional text-character cap for bounded local BPE runs. -/
  maxChars? : Option Nat
deriving Repr

namespace BpeCorpusOptions

/--
Parse the optional GPT-2 BPE tokenizer bundle.

`--bpe-vocab` and `--bpe-merges` must appear together; `--max-chars` is independent.
-/
def parse
    (args : List String) :
    Except String (BpeCorpusOptions × List String) := do
  let ((bpeVocab?, bpeMerges?), args) ←
    TorchLean.CLI.takePairedPathFlags args "bpe-vocab" "bpe-merges"
  let (maxCharsRaw?, args) ← TorchLean.CLI.takeNatFlagOnce args "max-chars"
  pure ({ bpeVocab? := bpeVocab?
          bpeMerges? := bpeMerges?
          maxChars? := maxCharsRaw? }, args)

end BpeCorpusOptions

/-- Shared terminal-REPL toggle used by interactive text examples. -/
structure InteractiveOptions where
  /-- Keep the trained model alive and read prompts from stdin. -/
  interactive : Bool
deriving Repr

namespace InteractiveOptions

/-- Parse the shared `--interactive` flag used by text examples with a terminal prompt loop. -/
def parse
    (args : List String) :
    Except String (InteractiveOptions × List String) := do
  let (interactive, args) ← TorchLean.CLI.takeBoolFlagOnce args "interactive"
  pure ({ interactive := interactive }, args)

end InteractiveOptions

/-- Shared prompt plus continuation-length options for simple text-generation commands. -/
structure PromptGenerationOptions where
  /-- Prompt used for before/after reports and generation. -/
  prompt : String
  /-- Number of generated tokens or characters after training. -/
  generate : Nat
deriving Repr

namespace PromptGenerationOptions

/-- Parse the shared `--prompt` / `--generate` flags. -/
def parse
    (args : List String)
    (defaults : PromptGenerationOptions) :
    Except String (PromptGenerationOptions × List String) := do
  let (prompt, args) ← TorchLean.CLI.takeFlagValueDefault args "prompt" defaults.prompt
  let (generate, args) ← TorchLean.CLI.takeNatFlagDefault args "generate" defaults.generate
  pure ({ prompt := prompt
          generate := generate }, args)

end PromptGenerationOptions

/-! ## Text TrainLog Notes -/

/--
TrainLog note fields for generation-capable text commands.

The stable generation surface is prompt, continuation length, temperature/top-k, repetition
control, RNG seed, and ASCII-only filtering. Model commands can prepend dataset or architecture
notes through `extra`.
-/
def generationNotes
    (gen : GenerationOptions)
    (generated? : Option String := none)
    (extra : Array String := #[]) : Array String :=
  extra ++
    #[s!"prompt={escapeForDisplay gen.prompt}",
      s!"generate={gen.generate}",
      s!"temperature={gen.temperature}",
      s!"top_k={gen.topK}",
      s!"sample_seed={gen.seed}",
      s!"repeat_penalty={gen.repeatPenalty}",
      s!"repeat_window={gen.repeatWindow}",
      s!"ascii_only={gen.asciiOnly}"] ++
    match generated? with
    | some generated => #[s!"generated={generated}"]
    | none => #[]

/--
TrainLog note fields for prompt-based text commands that do not expose the full sampling surface.
-/
def promptGenerationNotes
    (gen : PromptGenerationOptions)
    (generated? : Option String := none)
    (extra : Array String := #[]) : Array String :=
  extra ++
    #[s!"prompt={escapeForDisplay gen.prompt}",
      s!"generate={gen.generate}"] ++
    match generated? with
    | some generated => #[s!"generated={generated}"]
    | none => #[]

/-- Write a before/after loss log for a generation-capable text training command. -/
def writeGenerationTrainLog
    (log : _root_.Runtime.Training.LogDestination)
    (title : String)
    (steps : Nat)
    (beforeLoss afterLoss : Float)
    (gen : GenerationOptions)
    (generated? : Option String := none)
    (extra : Array String := #[]) : IO Unit :=
  TorchLean.Training.writeLossComparisonTo log title steps beforeLoss afterLoss
    (generationNotes gen generated? extra)

/-- Write a before/after loss log for a prompt-based text training command. -/
def writePromptTrainLog
    (log : _root_.Runtime.Training.LogDestination)
    (title : String)
    (steps : Nat)
    (beforeLoss afterLoss : Float)
    (gen : PromptGenerationOptions)
    (generated? : Option String := none)
    (extra : Array String := #[]) : IO Unit :=
  TorchLean.Training.writeLossComparisonTo log title steps beforeLoss afterLoss
    (promptGenerationNotes gen generated? extra)

/-! ## Text Training Option Combinators -/

/-- Number of corpus windows used by a finite or cyclic text-training command. -/
structure WindowOptions where
  /-- Number of windows available to the training sampler. -/
  windows : Nat
deriving Repr

namespace WindowOptions

/-- Parse a positive `--windows` value. -/
def parse
    (exeName : String)
    (args : List String)
    (defaultWindows : Nat) :
    Except String (WindowOptions × List String) := do
  let (windows, args) ←
    TorchLean.CLI.takePositiveNatFlag args exeName "windows" defaultWindows
  pure ({ windows }, args)

end WindowOptions

/-- Optional model-checkpoint paths for text training and generation. -/
structure CheckpointOptions where
  /-- Checkpoint loaded before training or generation. -/
  loadCheckpoint? : Option System.FilePath
  /-- Checkpoint written after training. -/
  saveCheckpoint? : Option System.FilePath
deriving Repr

namespace CheckpointOptions

/-- Parse `--load-checkpoint` and `--save-checkpoint`. -/
def parse (args : List String) : Except String (CheckpointOptions × List String) := do
  let (loadCheckpoint?, args) ← TorchLean.CLI.takePathFlagOnce args "load-checkpoint"
  let (saveCheckpoint?, args) ← TorchLean.CLI.takePathFlagOnce args "save-checkpoint"
  pure ({ loadCheckpoint?, saveCheckpoint? }, args)

end CheckpointOptions

end text
end TorchLean
