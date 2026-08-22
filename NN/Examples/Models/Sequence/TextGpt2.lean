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

namespace NN.Examples.Models.Sequence.TextGPT2

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

/-- Command-local corpus, training, tokenization, generation, and prompt-loop controls. -/
structure CorpusOptions extends
    CLI.Training.RunOptions, text.PromptGenerationOptions, text.InteractiveOptions where
  /-- Primary corpus and explicit small-data override. -/
  corpus : text.TextCorpusOptions
  /-- Optional second corpus pass. -/
  finetune : text.FinetuneOptions
  /-- Optional GPT-2 BPE tokenizer bundle. -/
  bpe : text.BpeCorpusOptions
deriving Repr

namespace CorpusOptions

/-- Parse the complete option surface owned by the `text_gpt2` executable. -/
def parse (args : List String) : Except String (CorpusOptions × List String) := do
  let (corpus, args) ← text.TextCorpusOptions.parse exeName args
  let (training, args) ← CLI.Training.RunOptions.parse exeName args defaultLogJson 1
  let (prompt, args) ← text.PromptGenerationOptions.parse args
    { prompt := "First Citizen:", generate := 0 }
  let (interactive, args) ← text.InteractiveOptions.parse args
  let (finetune, args) ← text.FinetuneOptions.parse args training.steps
  let (bpe, args) ← text.BpeCorpusOptions.parse args
  pure ({ toRunOptions := training
          toPromptGenerationOptions := prompt
          toInteractiveOptions := interactive
          corpus
          finetune
          bpe }, args)

end CorpusOptions

/-- Read the primary raw text corpus. -/
def readCorpusBytes (opts : CorpusOptions) : IO ByteArray :=
  text.Corpus.readByteFile exeName opts.corpus.dataFile opts.corpus.allowSmallData minTrainingBytes byteSeqLen

namespace ByteModel

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
def cfg : nn.models.CausalTransformer.Config :=
  { seqLen := seqLen
    vocab := vocab
    numHeads := numHeads
    headDim := headDim
    ffnHidden := ffnHidden
    layers := layers }

/-- Input shape: byte-level one-hot token sequence. -/
abbrev σ : Shape :=
  nn.models.CausalTransformer.vocabularyShape cfg (.dim batch .scalar)

/-- Output shape: one byte-logit row per input position. -/
abbrev τ : Shape :=
  σ

/--
Runnable byte-level GPT-style model for corpus pretraining/fine-tuning.

The model is compact enough for the eager CUDA path while still exercising attention, feed-forward
layers, byte tokenization, and the interactive prompt loop.
-/
def model : nn.Builder (nn.Sequential σ τ) :=
  nn.models.CausalTransformer.oneHot cfg (.dim batch .scalar)

end ByteModel

/-- Build one byte-level training sample from a corpus byte offset. -/
def mkByteCorpusSample (bytes : ByteArray) (i : Nat) :
    Sample.Supervised Float ByteModel.σ ByteModel.τ :=
  let toks := (text.byteTokenWindow bytes (ByteModel.seqLen + 1)
    (offset := text.Corpus.byteOffset bytes i ByteModel.seqLen)
    ).map (· % ByteModel.vocab)
  Data.CausalLM.oneHotBatch
    (α := Float) ByteModel.batch ByteModel.seqLen ByteModel.vocab toks.toList

/-- Build one byte-level prompt sample for before/after generation reports. -/
def mkBytePromptSample (prompt : String) : Sample.Supervised Float ByteModel.σ ByteModel.τ :=
  let ids := text.Tokenizer.byte.encode prompt
  let start := if ids.length > ByteModel.seqLen then ids.length - ByteModel.seqLen else 0
  let window := ((ids.drop start).take ByteModel.seqLen).map (· % ByteModel.vocab)
  Data.CausalLM.oneHotBatch
    (α := Float) ByteModel.batch ByteModel.seqLen ByteModel.vocab window

/-- Greedy byte-level generation from the trained model. -/
def generateByteGreedy
    (predict : Tensor Float ByteModel.σ → IO (Tensor Float ByteModel.τ))
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
    text.autoregressiveTokenIds ByteModel.seqLen 0 (text.Tokenizer.byte.encode prompt) gen
      (fun padded predPos => do
        let x := Data.CausalLM.oneHotInputs (α := Float)
          ByteModel.batch ByteModel.seqLen ByteModel.vocab
            (padded.map (· % ByteModel.vocab)).toList
        let logits ← predict x
        pure (text.batchLogitScoresAt logits ⟨0, by decide⟩ predPos))
      (sanitize := fun tok => if tok < ByteModel.vocab then tok else 0)
  pure (text.Tokenizer.byte.decode ids)

/-- Terminal prompt loop for the trained byte-level model. -/
partial def interactiveByteLoop
    (predict : Tensor Float ByteModel.σ → IO (Tensor Float ByteModel.τ))
    (generate : Nat) : IO Unit := do
  IO.println s!"  interactive: enter a prompt; empty line or :q exits (window={ByteModel.seqLen} bytes, generate={generate})"
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

namespace BPEModel

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
def cfg : nn.models.CausalTransformer.Config :=
  { seqLen := seqLen
    vocab := vocab
    numHeads := numHeads
    headDim := headDim
    ffnHidden := ffnHidden
    layers := layers }

/-- Input shape: local-BPE one-hot token batch. -/
abbrev σ : Shape :=
  nn.models.CausalTransformer.vocabularyShape cfg (.dim batch .scalar)

/-- Output shape: one local-BPE logit row per input position. -/
abbrev τ : Shape :=
  σ

/--
Compact GPT-2-style model with the real GPT-2 BPE tokenizer path.

This TorchLean-native Transformer reads GPT-2 BPE tokenizer files and uses a local output
projection over the corpus ids it observes. Its architecture and scale differ from OpenAI
GPT-2-small.
-/
def model : nn.Builder (nn.Sequential σ τ) :=
  nn.models.CausalTransformer.oneHot cfg (.dim batch .scalar)

end BPEModel

/-! ## Example-local compact vocabulary -/

/--
Projection from GPT-2 tokenizer ids to the smaller vocabulary used by this bounded example.

This is an example implementation detail, not part of the reusable tokenizer API.
-/
structure LocalBPEVocab where
  /-- Original tokenizer id for each local id. -/
  originals : Array Nat
  /-- Reverse lookup from original tokenizer id to local id. -/
  toLocalMap : Std.HashMap Nat Nat

namespace LocalBPEVocab

/-- Number of entries in the compact vocabulary. -/
def size (vocab : LocalBPEVocab) : Nat :=
  vocab.originals.size

/-- Map a GPT-2 token id into the compact vocabulary, using local id `0` for unknown ids. -/
def toLocal (vocab : LocalBPEVocab) (id : Nat) : Nat :=
  (vocab.toLocalMap[id]?).getD 0

/-- Map a compact token id back to its GPT-2 token id. -/
def toOriginal (vocab : LocalBPEVocab) (localId : Nat) : Nat :=
  vocab.originals.getD localId (vocab.originals.getD 0 0)

end LocalBPEVocab

/-- Build the compact vocabulary from the corpus and prompt ids used by this example. -/
def buildLocalBPEVocab
    (maxVocab : Nat) (corpusIds promptIds : Array Nat) : LocalBPEVocab :=
  Id.run do
    let mut originals : Array Nat := #[0]
    let mut map : Std.HashMap Nat Nat := (Std.HashMap.emptyWithCapacity).insert 0 0
    let addId (originals : Array Nat) (map : Std.HashMap Nat Nat) (id : Nat) :
        Array Nat × Std.HashMap Nat Nat :=
      if map.contains id || originals.size ≥ maxVocab then
        (originals, map)
      else
        let localId := originals.size
        (originals.push id, map.insert id localId)
    for id in corpusIds do
      let p := addId originals map id
      originals := p.1
      map := p.2
    for id in promptIds do
      let p := addId originals map id
      originals := p.1
      map := p.2
    return { originals := originals, toLocalMap := map }

/-- Apply the compact vocabulary to an array of GPT-2 token ids. -/
def localizeBPETokens (vocab : LocalBPEVocab) (tokens : Array Nat) : Array Nat :=
  tokens.map vocab.toLocal

/-- Build one BPE training sample from a tokenized corpus. -/
def mkBpeCorpusSample (tokens : Array Nat) (i : Nat) :
    Sample.Supervised Float BPEModel.σ BPEModel.τ :=
  -- The BPE model uses a real batch as well: each batch row gets a different deterministic
  -- corpus window, while the vocabulary stays small enough for a runnable example.
  Data.CausalLM.oneHotBatchFromTokenArray
    (α := Float) BPEModel.batch BPEModel.seqLen BPEModel.vocab tokens 0 i

/-- Turn a BPE prompt into one model input window. -/
def mkBpePromptSample
    (tok : text.GPT2BPE.Tokenizer) (lv : LocalBPEVocab) (prompt : String) :
    Except String (Sample.Supervised Float BPEModel.σ BPEModel.τ) := do
  let ids ← (text.GPT2BPE.encode tok prompt).map (fun ids => ids.map lv.toLocal)
  let start := if ids.length > BPEModel.seqLen then ids.length - BPEModel.seqLen else 0
  let window := (ids.drop start).take BPEModel.seqLen
  pure <| Data.CausalLM.oneHotBatch (α := Float)
    BPEModel.batch BPEModel.seqLen BPEModel.vocab window

/-- Decode original GPT-2 BPE ids with the loaded tokenizer. -/
def decodeBPEOrEmpty (tok : text.GPT2BPE.Tokenizer) (ids : List Nat) : String :=
  text.GPT2BPE.decodeOrEmpty tok ids

/-- Decode local BPE ids by mapping them back to original GPT-2 ids first. -/
def decodeLocalBPEOrEmpty (tok : text.GPT2BPE.Tokenizer) (lv : LocalBPEVocab) (ids : List Nat) :
    String :=
  decodeBPEOrEmpty tok (ids.map lv.toOriginal)

/-- Print an argmax prediction report for a prompt under the BPE model. -/
def printBpePredictionProbe
    (tok : text.GPT2BPE.Tokenizer)
    (lv : LocalBPEVocab)
    (predict : Tensor Float BPEModel.σ → IO (Tensor Float BPEModel.τ))
    (label prompt : String) : IO Unit := do
  let sample ← ModelZoo.orThrow exeName <| mkBpePromptSample tok lv prompt
  let logits ← predict (Sample.x sample)
  let ids := text.argmaxBatchTokens (α := Float)
    (batch := BPEModel.batch) (seqLen := BPEModel.seqLen) (vocab := BPEModel.vocab)
    (batchIdx := ⟨0, by decide⟩) logits
  IO.println s!"  {label} pred={text.escapeForDisplay (decodeLocalBPEOrEmpty tok lv ids)}"
  IO.println s!"  prompt={text.escapeForDisplay prompt}"

/--
Greedy BPE generation by repeatedly feeding the last `seqLen` tokens and appending the final-position
argmax. This is a deterministic sampling path for inspecting the trained next-token model.
-/
def generateBpeGreedy
    (tok : text.GPT2BPE.Tokenizer)
    (lv : LocalBPEVocab)
    (predict : Tensor Float BPEModel.σ → IO (Tensor Float BPEModel.τ))
    (prompt : String) (steps : Nat) : IO String := do
  let initOrigIds ← ModelZoo.orThrow exeName <| text.GPT2BPE.encode tok prompt
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
    text.autoregressiveTokenIds BPEModel.seqLen 0 initIds gen
      (fun padded predPos => do
        let x := Data.CausalLM.oneHotInputs (α := Float)
          BPEModel.batch BPEModel.seqLen BPEModel.vocab padded.toList
        let logits ← predict x
        pure (text.batchLogitScoresAt logits ⟨0, by decide⟩ predPos))
      (sanitize := fun tok => if tok < BPEModel.vocab then tok else 0)
  pure (decodeLocalBPEOrEmpty tok lv ids)

/--
Train the GPT-2-style model over a text corpus using CUDA.

This performs one optimizer step per corpus window, rather than materializing the entire dataset in
memory. The example is compact by GPT-2 standards, but the data path is real:
file bytes → token windows → one-hot tensors → TorchLean CUDA training.
-/
def trainCorpusFloat (opts : Options)
    (trainOpts : CorpusOptions)
    (bytes : ByteArray) : IO Unit := do
  let sample0 := mkByteCorpusSample bytes 0
  let first := text.byteTokenWindow bytes (ByteModel.seqLen + 1)
  IO.println s!"  mode=byte bytes={bytes.size} steps={trainOpts.steps} window={ByteModel.seqLen}"
  IO.println s!"  first prompt={text.escapeForDisplay (text.Tokenizer.byte.decode (first.toList.take ByteModel.seqLen))}"
  IO.println s!"  first target={text.escapeForDisplay (text.Tokenizer.byte.decode (first.toList.drop 1))}"
  let ftBytes? ←
    match trainOpts.finetune.finetuneFile? with
    | none => pure none
    | some path => do
        let ftBytes ←
          text.Corpus.readByteFile exeName path trainOpts.corpus.allowSmallData minTrainingBytes
            ByteModel.seqLen
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
  let run := Trainer.RunConfig.ofRuntimeOptions opts { optimizer := optim.adam { lr := 1e-3 } }
  let trainer := Trainer.new ByteModel.model <|
    Trainer.Config.fromRunConfig run (.oneHotCrossEntropy 2)
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
    Trainer.TrainSummary.printFloatLosses exeName trained.report
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
    (trainOpts : CorpusOptions)
    (tok : text.GPT2BPE.Tokenizer) :
    IO (Array Nat) := do
  let fullCorpusText ← IO.FS.readFile trainOpts.corpus.dataFile
  let corpusText :=
    match trainOpts.bpe.maxChars? with
    | some n => (fullCorpusText.take n).toString
    | none => fullCorpusText
  let ids ← ModelZoo.orThrow exeName <| text.GPT2BPE.encode tok corpusText
  let arr := ids.toArray
  if arr.size <= BPEModel.seqLen then
    throw <| IO.userError s!"{exeName}: BPE corpus is too small for a {BPEModel.seqLen}-token window"
  pure arr

/-- Print the first BPE training window for inspecting tokenization and windowing. -/
def printBpeCorpusPreview (tok : text.GPT2BPE.Tokenizer) (lv : LocalBPEVocab)
    (tokens : Array Nat) : IO Unit := do
  let first := text.Corpus.tokenArrayWindow tokens (BPEModel.seqLen + 1) 0
  IO.println s!"  first local BPE ids={first.toList}"
  IO.println s!"  first prompt={text.escapeForDisplay (decodeLocalBPEOrEmpty tok lv (first.toList.take BPEModel.seqLen))}"
  IO.println s!"  first target={text.escapeForDisplay (decodeLocalBPEOrEmpty tok lv (first.toList.drop 1))}"

/--
Train the compact GPT-2-style model with the real GPT-2 BPE tokenizer.

This exercises the GPT-2 tokenizer/vocabulary path and can overfit local windows. It is not a
pretrained GPT-2 checkpoint; it is a randomly initialized TorchLean model trained by this command.
-/
def trainBpeCorpusFloat (opts : Options)
    (trainOpts : CorpusOptions)
    (tok : text.GPT2BPE.Tokenizer) (lv : LocalBPEVocab) (tokens : Array Nat) : IO Unit := do
  IO.println s!"  mode=bpe local-vocab={lv.size}/{BPEModel.vocab} tokens={tokens.size} steps={trainOpts.steps}"
  printBpeCorpusPreview tok lv tokens
  let sample0 := mkBpeCorpusSample tokens 0
  let samples :=
    match (List.range trainOpts.steps).map (fun step => mkBpeCorpusSample tokens step) with
    | [] => [sample0]
    | xs => xs
  let run := Trainer.RunConfig.ofRuntimeOptions opts { optimizer := optim.adam { lr := 1e-3 } }
  let trainer := Trainer.new BPEModel.model <|
    Trainer.Config.fromRunConfig run (.oneHotCrossEntropy 2)
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
    Trainer.TrainSummary.printFloatLosses exeName trained.report
      (steps? := some trainOpts.steps)
  printBpePredictionProbe tok lv trained.predict "after " trainOpts.prompt
  let generated ← generateBpeGreedy tok lv trained.predict trainOpts.prompt trainOpts.generate
  IO.println s!"  greedy generated={text.escapeForDisplay generated}"
  text.writePromptTrainLog
    trainOpts.log "GPT-2 BPE corpus training" trainOpts.steps beforeLoss afterLoss
    trainOpts.toPromptGenerationOptions (some generated)
    #[s!"data={trainOpts.corpus.dataFile}", ModelZoo.deviceNote opts,
      s!"localVocab={lv.size}/{BPEModel.vocab}", s!"tokens={tokens.size}"]

/-- CLI entrypoint for CUDA byte/BPE corpus training. -/
def main (args : List String) : IO UInt32 := do
  Module.Command.runCudaFloat32 exeName args
    (banner := ModelZoo.bannerWithDevice exeName "GPU corpus trainer")
    (k := fun opts rest => do
      let (trainOpts, rest) ← ModelZoo.orThrow exeName <|
        CorpusOptions.parse rest
      CLI.requireNoArgs exeName rest
      let bytes ← readCorpusBytes trainOpts
      match trainOpts.bpe.bpeVocab?, trainOpts.bpe.bpeMerges? with
      | some vocabPath, some mergesPath =>
          let tok ← text.GPT2BPE.loadWithProgress exeName vocabPath mergesPath
          let capMsg :=
            match trainOpts.bpe.maxChars? with
            | some n => s!" (max chars={n})"
            | none => ""
          IO.eprintln s!"{exeName}: encoding BPE corpus{capMsg}"
          let tokens ← loadBpeCorpusTokens trainOpts tok
          IO.eprintln s!"{exeName}: encoded BPE corpus original-tokens={tokens.size}"
          let promptIds ← ModelZoo.orThrow exeName <| text.GPT2BPE.encode tok trainOpts.prompt
          let lv := buildLocalBPEVocab BPEModel.vocab tokens promptIds.toArray
          let localTokens := localizeBPETokens lv tokens
          IO.eprintln s!"{exeName}: projected BPE ids to local vocabulary {lv.size}/{BPEModel.vocab}"
          trainBpeCorpusFloat opts trainOpts tok lv localTokens
      | _, _ =>
          trainCorpusFloat opts trainOpts bytes)

end NN.Examples.Models.Sequence.TextGPT2
