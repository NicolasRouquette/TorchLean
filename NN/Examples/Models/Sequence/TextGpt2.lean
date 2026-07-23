/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

GPU-only corpus-training example:
  lake -R -K cuda=true build
  lake -R -K cuda=true exe torchlean text_gpt2 --device cuda \
    --data-file data/real/text/tinystories_valid.txt \
    --allow-small-data --steps 1 --generate 0

Prepare that file with:
  python3 scripts/datasets/download_example_data.py --tinystories-valid

GPT-2 BPE tokenizer run:
  lake -R -K cuda=true exe torchlean text_gpt2 --device cuda \
    --data-file data/real/text/tiny_shakespeare.txt \
    --bpe-vocab data/real/gpt2/vocab.json \
    --bpe-merges data/real/gpt2/merges.txt \
    --allow-small-data --max-chars 20000 --steps 10 \
    --prompt "First Citizen:" --generate 8

Local file run:
  lake -R -K cuda=true exe torchlean text_gpt2 --device cuda \
    --data-file /tmp/tiny.txt --allow-small-data --steps 1 --generate 0
-/

module

public import NN.API
public import NN.Examples.ModelZoo

/-!
# GPU GPT-2 Corpus Trainer

This command trains GPT-2-style models from text in TorchLean.

The model is initialized inside TorchLean and trained by the TorchLean runtime. It does not load a
pretrained PyTorch/Hugging Face checkpoint:

* reusable tokenization lives under `TorchLean.text`,
* the compact GPT-2-style architecture lives under `TorchLean.nn.models`,
* the runnable corpus trainer enforces CUDA by default.

The default path keeps the byte-level model compact so the corpus trainer is quick to run. Passing
`--bpe-vocab` and `--bpe-merges` switches to the Lean-native GPT-2 BPE tokenizer, using the standard
50,257-way GPT-2 token vocabulary. That BPE path still trains a randomly initialized model in
TorchLean; it does not load a pretrained checkpoint.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Sequence.TextGpt2

/-- Runner subcommand name. This subcommand trains a randomly initialized GPT-2-style model. -/
def exeName : String := "torchlean text_gpt2"

/-- Default JSON loss-curve path for this command. -/
def defaultLogJson : System.FilePath := ModelZoo.trainLogPath "text_gpt2"

/-- Minimum corpus size for the default public training path: 100 MiB. -/
def minTrainingBytes : Nat :=
  100 * 1024 * 1024

/--
Default byte-level context window for the CUDA corpus trainer.

Keeping this near the file top lets corpus validation and the model architecture agree without
depending on declaration order.
-/
def byteSeqLen : Nat := 1

/-- Read the primary raw text corpus. -/
def readCorpusBytes (opts : text.CorpusLoggedPromptInteractiveOptions) : IO ByteArray :=
  text.Corpus.readByteFile exeName opts.corpus.dataFile opts.corpus.allowSmallData minTrainingBytes byteSeqLen

namespace ByteGpt2

/-- Compact byte-level vocabulary for the default corpus path. -/
def vocab : Nat := 8

/-- Single-sequence batch for the byte-level corpus path. -/
def batch : Nat := 1

/--
Interactive context window.

This shares the folder-level byte context constant so corpus validation, byte training, and BPE
training use the same tensor layout. Larger windows require more allocator headroom, not
something we should quietly make the default before allocator pressure is solved.
-/
def seqLen : Nat := byteSeqLen

/-- Number of attention heads in the compact byte-level Transformer. -/
def numHeads : Nat := 1

/-- Per-head width. -/
def headDim : Nat := 1

/-- Transformer embedding width. -/
def dModel : Nat := numHeads * headDim

/-- Feed-forward hidden width. -/
def ffnHidden : Nat := 2

/-- Number of Transformer blocks. -/
def layers : Nat := 1

local instance : NeZero seqLen := ⟨by decide⟩
local instance : NeZero dModel := ⟨by decide⟩

/-- Byte-level GPT configuration shared by shapes and the model constructor. -/
def cfg : nn.models.CausalTransformerConfig :=
  { batch := batch
    seqLen := seqLen
    vocab := vocab
    numHeads := numHeads
    headDim := headDim
    ffnHidden := ffnHidden
    layers := layers }

/-- Input shape: byte-level one-hot token sequence. -/
abbrev σ : Shape :=
  nn.models.causalVocabularyShape cfg

/-- Output shape: one byte-logit row per input position. -/
abbrev τ : Shape :=
  σ

/--
Runnable byte-level GPT-style model for corpus pretraining/fine-tuning.

The model is compact enough for the eager CUDA path while still exercising attention, feed-forward
layers, byte tokenization, and the interactive prompt loop.
-/
def model : nn.M (nn.Sequential σ τ) :=
  nn.models.causalTransformerOneHot cfg

end ByteGpt2

/-- Build one byte-level training sample from a corpus byte offset. -/
def mkByteCorpusSample (bytes : ByteArray) (i : Nat) :
    SupervisedSample Float ByteGpt2.σ ByteGpt2.τ :=
  let toks := (text.byteTokenWindow bytes (ByteGpt2.seqLen + 1)
    (offset := text.Corpus.byteOffset bytes i ByteGpt2.seqLen)
    ).map (· % ByteGpt2.vocab)
  Data.causalLmOneHotSample (α := Float) ByteGpt2.batch ByteGpt2.seqLen ByteGpt2.vocab toks

/-- Build one byte-level prompt sample for before/after generation reports. -/
def mkBytePromptSample (prompt : String) : SupervisedSample Float ByteGpt2.σ ByteGpt2.τ :=
  let ids := text.Tokenizer.byte.encode prompt
  let start := if ids.length > ByteGpt2.seqLen then ids.length - ByteGpt2.seqLen else 0
  let window := ((ids.drop start).take ByteGpt2.seqLen).map (· % ByteGpt2.vocab)
  Data.causalLmOneHotSample (α := Float) ByteGpt2.batch ByteGpt2.seqLen ByteGpt2.vocab window

/-- Greedy byte-level generation from the trained model. -/
def generateByteGreedy
    (predict : Tensor.T Float ByteGpt2.σ → IO (Tensor.T Float ByteGpt2.τ))
    (prompt : String) (steps : Nat) : IO String := do
  let gen : text.GenerationOptions :=
    { prompt := prompt
      generate := steps
      temperature := 1.0
      topK := 1
      repeatPenalty := 0.0
      repeatWindow := 0
      seed := 0
      asciiOnly := false }
  let ids ←
    text.autoregressiveTokenIds ByteGpt2.seqLen 0 (text.Tokenizer.byte.encode prompt) gen
      (fun padded predPos => do
        let x := text.causalLmXOneHotBatch (α := Float)
          ByteGpt2.batch ByteGpt2.seqLen ByteGpt2.vocab (padded.map (· % ByteGpt2.vocab))
        let logits ← predict x
        pure (text.batchLogitScoresAt logits ⟨0, by decide⟩ predPos))
      (sanitize := fun tok => if tok < ByteGpt2.vocab then tok else 0)
  pure (text.Tokenizer.byte.decode ids)

/-- Terminal prompt loop for the trained byte-level model. -/
partial def interactiveByteLoop
    (predict : Tensor.T Float ByteGpt2.σ → IO (Tensor.T Float ByteGpt2.τ))
    (generate : Nat) : IO Unit := do
  IO.println s!"  interactive: enter a prompt; empty line or :q exits (window={ByteGpt2.seqLen} bytes, generate={generate})"
  let stdin ← IO.getStdin
  let rec loop : IO Unit := do
    IO.print "  prompt> "
    let line ← stdin.getLine
    let prompt := line.trimAscii.toString
    if prompt = "" || prompt = ":q" || prompt = ":quit" then
      IO.println "  interactive: done"
    else
      let out ← generateByteGreedy predict prompt generate
      IO.println s!"  response={text.escapeForDisplay out}"
      loop
  loop

namespace BpeGpt2

/--
Compact vocabulary used by the runnable BPE training path.

The tokenizer still uses GPT-2's real 50,257-token BPE files. For this Lean/CUDA model
we project the corpus tokens into a local vocabulary of the first observed BPE ids. A full 50k-way
output head is a much larger training run; this example focuses on the tokenizer/data path.
-/
def vocab : Nat := 512

/-- Batch size for the BPE corpus path. -/
def batch : Nat := 2

/-- Short context window used by the trainer. -/
def seqLen : Nat := byteSeqLen

/-- Number of attention heads in the miniature BPE Transformer. -/
def numHeads : Nat := 1

/-- Per-head width for the BPE Transformer. -/
def headDim : Nat := 8

/-- Transformer embedding width. -/
def dModel : Nat := numHeads * headDim

/-- Feed-forward hidden width. -/
def ffnHidden : Nat := 32

/-- Number of Transformer blocks. -/
def layers : Nat := 1

local instance : NeZero seqLen := ⟨by decide⟩
local instance : NeZero dModel := ⟨by decide⟩

/-- BPE GPT configuration shared by shapes and the model constructor. -/
def cfg : nn.models.CausalTransformerConfig :=
  { batch := batch
    seqLen := seqLen
    vocab := vocab
    numHeads := numHeads
    headDim := headDim
    ffnHidden := ffnHidden
    layers := layers }

/-- Input shape: local-BPE one-hot token batch. -/
abbrev σ : Shape :=
  nn.models.causalVocabularyShape cfg

/-- Output shape: one local-BPE logit row per input position. -/
abbrev τ : Shape :=
  σ

/--
Compact GPT-2-style model with the real GPT-2 BPE tokenizer path.

This TorchLean-native Transformer reads GPT-2 BPE tokenizer files and uses a local output
projection over the corpus ids it observes. Its architecture and scale differ from OpenAI
GPT-2-small.
-/
def model : nn.M (nn.Sequential σ τ) :=
  nn.models.causalTransformerOneHot cfg

end BpeGpt2

/-- Build one BPE training sample from a tokenized corpus. -/
def mkBpeCorpusSample (tokens : Array Nat) (i : Nat) :
    SupervisedSample Float BpeGpt2.σ BpeGpt2.τ :=
  -- The BPE model uses a real batch as well: each batch row gets a different deterministic
  -- corpus window, while the vocabulary stays small enough for a runnable example.
  Data.causalLmOneHotSampleRowsFromTokenArray
    (α := Float) BpeGpt2.batch BpeGpt2.seqLen BpeGpt2.vocab tokens 0 i

/-- Turn a BPE prompt into one model input window. -/
def mkBpePromptSample
    (tok : text.Gpt2Bpe.Tokenizer) (lv : text.LocalBpeVocab) (prompt : String) :
    Except String (SupervisedSample Float BpeGpt2.σ BpeGpt2.τ) := do
  let ids ← (text.Gpt2Bpe.encode tok prompt).map (fun ids => ids.map lv.toLocal)
  let start := if ids.length > BpeGpt2.seqLen then ids.length - BpeGpt2.seqLen else 0
  let window := (ids.drop start).take BpeGpt2.seqLen
  pure <| Data.causalLmOneHotSample (α := Float)
    BpeGpt2.batch BpeGpt2.seqLen BpeGpt2.vocab window

/-- Decode original GPT-2 BPE ids with the loaded tokenizer. -/
def decodeBpeD (tok : text.Gpt2Bpe.Tokenizer) (ids : List Nat) : String :=
  text.Gpt2Bpe.decodeD tok ids

/-- Decode local BPE ids by mapping them back to original GPT-2 ids first. -/
def decodeLocalBpeD (tok : text.Gpt2Bpe.Tokenizer) (lv : text.LocalBpeVocab) (ids : List Nat) :
    String :=
  decodeBpeD tok (ids.map lv.toOriginal)

/-- Print an argmax prediction report for a prompt under the BPE model. -/
def printBpePredictionProbe
    (tok : text.Gpt2Bpe.Tokenizer)
    (lv : text.LocalBpeVocab)
    (predict : Tensor.T Float BpeGpt2.σ → IO (Tensor.T Float BpeGpt2.τ))
    (label prompt : String) : IO Unit := do
  let sample ← ModelZoo.orThrow exeName <| mkBpePromptSample tok lv prompt
  let logits ← predict (Sample.x sample)
  let ids := text.argmaxTokenIdsFromBatchLogits (α := Float)
    (batch := BpeGpt2.batch) (seqLen := BpeGpt2.seqLen) (vocab := BpeGpt2.vocab)
    (batchIdx := ⟨0, by decide⟩) logits
  IO.println s!"  {label} pred={text.escapeForDisplay (decodeLocalBpeD tok lv ids)}"
  IO.println s!"  prompt={text.escapeForDisplay prompt}"

/--
Greedy BPE generation by repeatedly feeding the last `seqLen` tokens and appending the final-position
argmax. This is a deterministic sampling path for inspecting the trained next-token model.
-/
def generateBpeGreedy
    (tok : text.Gpt2Bpe.Tokenizer)
    (lv : text.LocalBpeVocab)
    (predict : Tensor.T Float BpeGpt2.σ → IO (Tensor.T Float BpeGpt2.τ))
    (prompt : String) (steps : Nat) : IO String := do
  let initOrigIds ← ModelZoo.orThrow exeName <| text.Gpt2Bpe.encode tok prompt
  let initIds := initOrigIds.map lv.toLocal
  let gen : text.GenerationOptions :=
    { prompt := prompt
      generate := steps
      temperature := 1.0
      topK := 1
      repeatPenalty := 0.0
      repeatWindow := 0
      seed := 0
      asciiOnly := false }
  let ids ←
    text.autoregressiveTokenIds BpeGpt2.seqLen 0 initIds gen
      (fun padded predPos => do
        let x := text.causalLmXOneHotBatch (α := Float)
          BpeGpt2.batch BpeGpt2.seqLen BpeGpt2.vocab padded
        let logits ← predict x
        pure (text.batchLogitScoresAt logits ⟨0, by decide⟩ predPos))
      (sanitize := fun tok => if tok < BpeGpt2.vocab then tok else 0)
  pure (decodeLocalBpeD tok lv ids)

/--
Train the GPT-2-style model over a text corpus using CUDA.

This performs one optimizer step per corpus window, rather than materializing the entire dataset in
memory. The example is compact by GPT-2 standards, but the data path is real:
file bytes → token windows → one-hot tensors → TorchLean CUDA training.
-/
def trainCorpusFloat (opts : Options)
    (trainOpts : text.CorpusLoggedPromptInteractiveOptions)
    (bytes : ByteArray) : IO Unit := do
  let sample0 := mkByteCorpusSample bytes 0
  let first := text.byteTokenWindow bytes (ByteGpt2.seqLen + 1)
  IO.println s!"  mode=byte bytes={bytes.size} steps={trainOpts.steps} window={ByteGpt2.seqLen}"
  IO.println s!"  first prompt={text.escapeForDisplay (text.Tokenizer.byte.decode (first.take ByteGpt2.seqLen))}"
  IO.println s!"  first target={text.escapeForDisplay (text.Tokenizer.byte.decode (first.drop 1))}"
  let ftBytes? ←
    match trainOpts.finetune.finetuneFile? with
    | none => pure none
    | some path => do
        let ftBytes ←
          text.Corpus.readByteFile exeName path trainOpts.corpus.allowSmallData minTrainingBytes
            ByteGpt2.seqLen
        pure (some ftBytes)
  let pretrainSamples :=
    (List.range trainOpts.steps).map (fun step => mkByteCorpusSample bytes step)
  let finetuneSamples :=
    match ftBytes? with
    | none => []
    | some ftBytes =>
        (List.range trainOpts.finetune.finetuneSteps).map (fun step => mkByteCorpusSample ftBytes step)
  let allSamples := pretrainSamples ++ finetuneSamples
  let totalSteps := trainOpts.steps +
    match ftBytes? with
    | none => 0
    | some _ => trainOpts.finetune.finetuneSteps
  let trainSamples :=
    match allSamples with
    | [] => [sample0]
    | xs => xs
  let run := Trainer.runConfig opts { optimizer := optim.adam { lr := 1e-3 } }
  let trainer := Trainer.new ByteGpt2.model <|
    Trainer.Config.fromRunConfig run .crossEntropy
  trainer.printInfo
  /-
  Each optimizer step sees one deterministic corpus window.  Materializing those windows as a
  finite dataset makes the training schedule inspectable: pretraining windows come first, optional
  finetune windows come after them, and the public trainer owns the optimizer/checkpoint mechanics.
  -/
  let trained ← trainer.train
    (Data.floatSamples trainSamples)
    { steps := totalSteps
      log := .disabled
      logEvery := Nat.max 1 (totalSteps / 10)
      cudaMemWatch := trainOpts.cudaMemWatch }
  let (beforeLoss, afterLoss) ←
    Trainer.TrainSummary.requireAndPrintFloatLosses exeName trained.report
      (steps? := some totalSteps)
  let generated ← generateByteGreedy trained.predict trainOpts.prompt trainOpts.generate
  IO.println s!"  greedy generated={text.escapeForDisplay generated}"
  text.writePromptTrainLog
    trainOpts.log "GPT-2 byte corpus training" totalSteps beforeLoss afterLoss
    trainOpts.toPromptGenerationOptions (some generated)
    #[s!"data={trainOpts.corpus.dataFile}", ModelZoo.deviceNote opts,
      s!"bytes={bytes.size}"]
  if trainOpts.interactive then
    interactiveByteLoop trained.predict trainOpts.generate

/-- Load and tokenize the text corpus with GPT-2 BPE. -/
def loadBpeCorpusTokens
    (trainOpts : text.CorpusLoggedPromptInteractiveOptions)
    (tok : text.Gpt2Bpe.Tokenizer) :
    IO (Array Nat) := do
  let fullCorpusText ← IO.FS.readFile trainOpts.corpus.dataFile
  let corpusText :=
    match trainOpts.bpe.maxChars? with
    | some n => (fullCorpusText.take n).toString
    | none => fullCorpusText
  let ids ← ModelZoo.orThrow exeName <| text.Gpt2Bpe.encode tok corpusText
  let arr := ids.toArray
  if arr.size <= BpeGpt2.seqLen then
    throw <| IO.userError s!"{exeName}: BPE corpus is too small for a {BpeGpt2.seqLen}-token window"
  pure arr

/-- Print the first BPE training window for inspecting tokenization and windowing. -/
def printBpeCorpusPreview (tok : text.Gpt2Bpe.Tokenizer) (lv : text.LocalBpeVocab)
    (tokens : Array Nat) : IO Unit := do
  let first := text.Corpus.tokenArrayWindow tokens (BpeGpt2.seqLen + 1) 0
  IO.println s!"  first local BPE ids={first}"
  IO.println s!"  first prompt={text.escapeForDisplay (decodeLocalBpeD tok lv (first.take BpeGpt2.seqLen))}"
  IO.println s!"  first target={text.escapeForDisplay (decodeLocalBpeD tok lv (first.drop 1))}"

/--
Train the compact GPT-2-style model with the real GPT-2 BPE tokenizer.

This exercises the GPT-2 tokenizer/vocabulary path and can overfit local windows. It is not a
pretrained GPT-2 checkpoint; it is a randomly initialized TorchLean model trained by this command.
-/
def trainBpeCorpusFloat (opts : Options)
    (trainOpts : text.CorpusLoggedPromptInteractiveOptions)
    (tok : text.Gpt2Bpe.Tokenizer) (lv : text.LocalBpeVocab) (tokens : Array Nat) : IO Unit := do
  IO.println s!"  mode=bpe local-vocab={lv.size}/{BpeGpt2.vocab} tokens={tokens.size} steps={trainOpts.steps}"
  printBpeCorpusPreview tok lv tokens
  let sample0 := mkBpeCorpusSample tokens 0
  let samples :=
    match (List.range trainOpts.steps).map (fun step => mkBpeCorpusSample tokens step) with
    | [] => [sample0]
    | xs => xs
  let run := Trainer.runConfig opts { optimizer := optim.adam { lr := 1e-3 } }
  let trainer := Trainer.new BpeGpt2.model <|
    Trainer.Config.fromRunConfig run .crossEntropy
  trainer.printInfo
  /-
  BPE mode uses the real GPT-2 tokenizer but a compact local output vocabulary.  The public trainer
  sees the already-projected one-hot windows; decoding remains here because it is presentation logic,
  not training machinery.
  -/
  let trained ← trainer.train
    (Data.floatSamples samples)
    { steps := trainOpts.steps
      log := .disabled
      logEvery := Nat.max 1 (trainOpts.steps / 10)
      cudaMemWatch := trainOpts.cudaMemWatch }
  let (beforeLoss, afterLoss) ←
    Trainer.TrainSummary.requireAndPrintFloatLosses exeName trained.report
      (steps? := some trainOpts.steps)
  printBpePredictionProbe tok lv trained.predict "after " trainOpts.prompt
  let generated ← generateBpeGreedy tok lv trained.predict trainOpts.prompt trainOpts.generate
  IO.println s!"  greedy generated={text.escapeForDisplay generated}"
  text.writePromptTrainLog
    trainOpts.log "GPT-2 BPE corpus training" trainOpts.steps beforeLoss afterLoss
    trainOpts.toPromptGenerationOptions (some generated)
    #[s!"data={trainOpts.corpus.dataFile}", ModelZoo.deviceNote opts,
      s!"localVocab={lv.size}/{BpeGpt2.vocab}", s!"tokens={tokens.size}"]

/-- CLI entrypoint for CUDA byte/BPE corpus training. -/
def main (args : List String) : IO UInt32 := do
  Runtime.runCudaFloat exeName args
    (banner := ModelZoo.bannerWithDevice exeName "GPU corpus trainer")
    (k := fun opts rest => do
      let (trainOpts, rest) ← ModelZoo.orThrow exeName <|
        text.CorpusLoggedPromptInteractiveOptions.parse
          exeName rest defaultLogJson 1
            { prompt := "First Citizen:"
              generate := 0 }
      CLI.requireNoArgs exeName rest
      let bytes ← readCorpusBytes trainOpts
      match trainOpts.bpe.bpeVocab?, trainOpts.bpe.bpeMerges? with
      | some vocabPath, some mergesPath =>
          let tok ← text.Gpt2Bpe.loadWithProgress exeName vocabPath mergesPath
          let capMsg :=
            match trainOpts.bpe.maxChars? with
            | some n => s!" (max chars={n})"
            | none => ""
          IO.eprintln s!"{exeName}: encoding BPE corpus{capMsg}"
          let tokens ← loadBpeCorpusTokens trainOpts tok
          IO.eprintln s!"{exeName}: encoded BPE corpus original-tokens={tokens.size}"
          let promptIds ← ModelZoo.orThrow exeName <| text.Gpt2Bpe.encode tok trainOpts.prompt
          let lv := text.buildLocalBpeVocab BpeGpt2.vocab tokens promptIds.toArray
          let localTokens := text.localizeBpeTokens lv tokens
          IO.eprintln s!"{exeName}: projected BPE ids to local vocabulary {lv.size}/{BpeGpt2.vocab}"
          trainBpeCorpusFloat opts trainOpts tok lv localTokens
      | _, _ =>
          trainCorpusFloat opts trainOpts bytes)

end NN.Examples.Models.Sequence.TextGpt2
