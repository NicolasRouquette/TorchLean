/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

CUDA text example:
  lake -R -K cuda=true exe torchlean gpt2 --device cuda --steps 1 --windows 1 --generate 0
  lake -R -K cuda=true exe torchlean gpt2 --device cuda --tiny-shakespeare --prompt "First Citizen:" --steps 1 \
    --windows 1 --generate 0 --temperature 0.85 --top-k 12 --sample-seed 7
  lake -R -K cuda=true exe torchlean gpt2 --device cuda --tiny-shakespeare --steps 1 --windows 1 \
    --save-checkpoint data/model_zoo/gpt2_shakespeare.state.json
  lake -R -K cuda=true exe torchlean gpt2_saved --device cuda --checkpoint data/model_zoo/gpt2_shakespeare.state.json \
    --prompt "First Citizen:" --generate 0

Dataset example:
  python3 scripts/datasets/download_example_data.py --tiny-shakespeare
  lake -R -K cuda=true exe torchlean gpt2 --device cuda --tiny-shakespeare --steps 1 --windows 1 --generate 0

This is a GPT-2-style *causal* language-model command (byte-level tokens).

Performance note: use CUDA for this example. The pure Lean CPU path exists for debugging tiny model
states, but Transformer workloads are too slow there for a useful run. The default command uses one
training window so it finishes quickly; pass larger `--steps`, `--windows`, and `--generate` values
when you want a real text experiment.
The command exercises masked self-attention, LayerNorm, and feed-forward blocks through the public
`TorchLean.nn` model constructors and `TorchLean.text` token tools.

After a run that writes `--log <path>`, you can view the prompt and sampled continuation in the
infoview via:

`#gpt2_train_log_file_view "<path>"`
-/

module


public import NN.API
public import NN.Examples.Models.Common.RealData

/-!
GPT-2 style sequence model example.

The runnable causal language-model path includes training, generation, and infoview support. It uses
the same public TorchLean model API that the command-line example uses.
-/

/-!
# GPT-2-Style Causal Language Model Example

Runnable `torchlean gpt2` example. It builds a GPT-2-style causal transformer over
byte-level tokens, with optional real text input from tiny-shakespeare or `--data-file PATH`.

For the simplest "Karpathy-style single text file" path, use `torchlean chargpt`
(character-level tokenizer). This `gpt2` command is byte-level and shows the Transformer block
wiring and save/reload loop.

```bash
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
lake -R -K cuda=true exe torchlean gpt2 --device cuda --tiny-shakespeare --steps 1 --windows 1 --generate 0
```
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Sequence.Gpt2

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "torchlean gpt2"

/-- Default JSON loss-curve path for this command. -/
def defaultLogJson : System.FilePath := ModelZoo.trainLogPath "gpt2"

/-- Complete command help, including the text and training flags parsed after runtime selection. -/
def usage : String :=
  Module.Command.usage exeName ++ "\n" ++ String.intercalate "\n"
    [ "Text data:"
    , "  --data-file PATH | --tiny-shakespeare | --tinystories-valid"
    , ""
    , "Training:"
    , "  --steps N          optimizer updates"
    , "  --batch-size N     corpus windows accumulated per update"
    , "  --windows N        corpus windows available to training"
    , "  --lr X             Adam learning rate"
    , "  --log PATH|false   write a TrainLog JSON, or disable logging"
    , "  --cuda-mem-watch N sample CUDA allocator state every N updates"
    , "  --load-checkpoint PATH | --save-checkpoint PATH"
    , ""
    , "Generation:"
    , "  --prompt TEXT --generate N --temperature X --top-k N"
    , "  --repeat-penalty X --repeat-window N --sample-seed N"
    , "  --ascii-only [true|false] --interactive"
    ]

/-- Batch size for the byte-level causal Transformer. -/
def batch : Nat := 1

/-- Prompt/target window length for the runnable GPT example. -/
def seqLen : Nat := 1

/-- Byte vocabulary width used by the one-hot tokenizer. -/
def vocab : Nat := 8

/-- Toy byte bucketing: encode byte id `b` as `b % 8`; collisions are intentional. -/
def byteBucket (id : Nat) : Fin vocab :=
  ⟨id % vocab, Nat.mod_lt _ (by decide)⟩

/-- Number of attention heads in the miniature Transformer block. -/
def numHeads : Nat := 1

/-- Per-head embedding width. The model dimension is
$\mathtt{numHeads}\cdot\mathtt{headDim}$. -/
def headDim : Nat := 1

/-- Transformer embedding width. -/
def dModel : Nat := numHeads * headDim

/-- Hidden width of the feed-forward sublayer. -/
def ffnHidden : Nat := 2

/-- Number of Transformer encoder blocks. -/
def layers : Nat := 1

local instance : NeZero seqLen := ⟨by decide⟩
local instance : NeZero dModel := ⟨by decide⟩

/-- Input shape: batched byte-level one-hot token windows. -/
abbrev σ : List Nat :=
  [batch, seqLen, vocab]

/-- Output shape: one vocabulary-logit row for every input token position. -/
abbrev τ : List Nat :=
  σ

/-- Public GPT-style causal Transformer constructor specialized to the byte-level config. -/
def model : nn.Builder (nn.Sequential σ τ) :=
  nn.models.CausalTransformer.oneHot
    { seqLen := seqLen
      vocab := vocab
      numHeads := numHeads
      headDim := headDim
      ffnHidden := ffnHidden
      layers := layers }
    [batch]

/-- Command-local controls for GPT training, checkpointing, generation, and the prompt loop. -/
structure TrainOptions extends
    CLI.Training.OptimizerOptions, text.GenerationOptions, text.WindowOptions,
    text.CheckpointOptions, text.InteractiveOptions where
deriving Repr

namespace TrainOptions

/-- Parse the byte-level GPT command's training and generation flags. -/
def parse (args : List String) (defaultSteps : Nat) :
    Except String (TrainOptions × List String) := do
  let (training, args) ←
    CLI.Training.OptimizerOptions.parse exeName args defaultLogJson defaultSteps 0.001
      (allowZeroSteps := true)
  let (window, args) ← text.WindowOptions.parse exeName args 1
  let (generation, args) ← text.GenerationOptions.parse exeName args
    { prompt := "First Citizen:"
      generate := 0
      temperature := 0.85
      topK := 12
      repeatPenalty := 1.25
      repeatWindow := 24
      seed := 0
      asciiOnly := false }
  let (checkpoint, args) ← text.CheckpointOptions.parse args
  let (interactive, args) ← text.InteractiveOptions.parse args
  pure ({ toOptimizerOptions := training
          toGenerationOptions := generation
          toWindowOptions := window
          toCheckpointOptions := checkpoint
          toInteractiveOptions := interactive }, args)

end TrainOptions

/-- Build a batched causal-LM sample by repeating one token window across all rows. -/
def mkSampleFromTokenIds (toks : Tensor Nat [seqLen + 1]) : Sample.Supervised Float σ τ :=
  Data.CausalLM.oneHotSample (α := Float) [batch] seqLen vocab
    (Tensor.repeatAxis 0 batch (toks.map byteBucket))

/-- Build a batch sample from exactly one token window per batch row. -/
def mkSampleBatchFromTokenIds (idsByBatch : Tensor Nat [batch, seqLen + 1]) :
    Sample.Supervised Float σ τ :=
  Data.CausalLM.oneHotSample (α := Float) [batch] seqLen vocab
    (idsByBatch.map byteBucket)

/--
Parse GPT-2-specific data flags and return the training corpus plus remaining runtime flags.
-/
def takeInputText (args : List String) : IO (String × List String) :=
  text.Corpus.takeUtf8Input exeName RealData.tinyShakespearePath
    [("--tiny-shakespeare", RealData.tinyShakespearePath),
      ("--tinystories-valid", RealData.tinyStoriesValidPath)]
    RealData.missingTinyShakespeareOrTinyStoriesHint args

/-- Byte-token window used for reporting prompt/target text. -/
def tokenWindowIds (input : String) (offset : Nat) : Array Nat :=
  (text.tokenWindow text.Tokenizer.byte seqLen input (offset := offset) (padId := 32)).toArray

/-- Print a compact before/after language-model report for the first batch row. -/
def printPredictionReport (label : String) (input : String) (logits : Tensor Float σ) :
    IO Unit := do
  let predIds := text.argmaxBatchTokens (α := Float) logits ⟨0, by decide⟩
  IO.println s!"  {label} pred={text.escapeByteIdsForDisplay predIds}"
  IO.println s!"  prompt={text.escapeByteIdsForDisplay (tokenWindowIds input 0)}"
  IO.println s!"  target={text.escapeByteIdsForDisplay (tokenWindowIds input 1)}"

/-- Convert byte ids into the typed batched one-hot input tensor used for generation. -/
def inputTensorFromIds (ids : Tensor Nat [seqLen]) : Tensor Float σ :=
  Tensor.repeatAxis 0 batch <|
    Data.CausalLM.oneHotInputs (α := Float) vocab (ids.map byteBucket)

/--
Fitted byte-level GPT predictor.

Training, saved-checkpoint inference, and future optimized runners all provide this one closure.
Generation only needs a logit-producing function; it does not depend on where the logits came from.
-/
abbrev Predictor :=
  Tensor Float σ → IO (Tensor Float τ)

mutual

/-- Autoregressively extend byte token ids using a trained byte-level GPT model. -/
partial def generateSampledFromIds
    (predict : Predictor)
    (promptIds : Array Nat) (steps : Nat) (temperature : Float) (topK seed repeatWindow : Nat)
    (repeatPenalty : Float) (asciiOnly : Bool) : IO (Array Nat) := do
  let gen : text.GenerationOptions :=
    { prompt := ""
      generate := steps
      temperature := temperature
      topK := topK
      repeatPenalty := repeatPenalty
      repeatWindow := repeatWindow
      seed := seed
      asciiOnly := asciiOnly }
  let allowId := if asciiOnly then text.printableAsciiByte else fun _ => true
  let ids ←
    text.autoregressiveTokenIds seqLen 32 promptIds gen
      (fun padded predPos => do
        let logits ← predict (inputTensorFromIds padded)
        pure (text.batchLogitScoresAt logits ⟨0, by decide⟩ predPos))
      (allowId := fun i => allowId i.val)
  pure ids

/-- Encode a string prompt and autoregressively extend it. -/
partial def generateSampled
    (predict : Predictor)
    (prompt : String) (steps : Nat) (temperature : Float) (topK seed repeatWindow : Nat)
    (repeatPenalty : Float) (asciiOnly : Bool) : IO (Array Nat) := do
  let init := text.Tokenizer.byte.encode prompt
  generateSampledFromIds predict init steps temperature topK seed repeatWindow repeatPenalty
    asciiOnly

end

/-- Build a finite cyclic training set from corpus text, biased toward the prompt when present. -/
def samplesFromCorpus (input _prompt : String) (windows : Nat) :
    Array (Sample.Supervised Float σ τ) :=
  let toks := text.Tokenizer.byte.encode input
  let offs := text.Corpus.promptAwareOffsets toks.size seqLen windows none
  offs.map (fun off =>
    let idsByBatch : Tensor Nat [batch, seqLen + 1] :=
      Tensor.stack 0 fun i =>
        let off' := (off + i.val * (seqLen / 2 + 1)) % Nat.max 1 (toks.size - (seqLen + 1))
        text.tokenWindow text.Tokenizer.byte (seqLen + 1) input
          (offset := off') (padId := 32)
    mkSampleBatchFromTokenIds idsByBatch)

/--
Interactive prompt loop for the in-memory Float model.

Each line is appended to the current byte context, decoded through the trained local model, and then
kept as context for the next prompt unless the user clears it.
-/
partial def interactiveLoopFloat
    (predict : Predictor)
    (train : TrainOptions) :
    IO Unit := do
  IO.println s!"  interactive: enter text; :q exits, :clear resets, :show prints context (window={seqLen} bytes)"
  let stdin ← IO.getStdin
  let rec loop (ctx : Array Nat) : IO Unit := do
    IO.print "  prompt> "
    let line ← stdin.getLine
    let prompt := line.trimAscii.toString
    if prompt = "" || prompt = ":q" || prompt = ":quit" then
      IO.println "  interactive: done"
    else if prompt = ":clear" then
      IO.println "  interactive: cleared context"
      loop #[]
    else if prompt = ":show" then
      IO.println s!"  context={text.escapeByteIdsForDisplay ctx}"
      loop ctx
    else
      let inputIds := ctx ++ text.Tokenizer.byte.encode prompt ++ #[10]
      let outIds ←
        generateSampledFromIds predict inputIds train.generate
          train.temperature train.topK train.seed train.repeatWindow train.repeatPenalty train.asciiOnly
      let genOnly := outIds.drop inputIds.size
      IO.println s!"  generated={text.escapeByteIdsForDisplay genOnly}"
      loop outIds
  loop #[]

/--
Train the byte-level model and decode a prediction report.

Inputs and reports use host `Float`, while `RunConfig.scalar` selects the arithmetic used by the
trainer. The command fixes that runtime choice to native `Float32`.
-/
def trainAndDecode (opts : Options) (input : String)
    (train : TrainOptions) :
    IO (Float × Float × String) := do
  let samples := samplesFromCorpus input train.prompt train.windows
  let reportSample :=
    Data.CausalLM.byteBatch (α := Float) batch seqLen vocab byteBucket train.prompt
  let run := Trainer.RunConfig.ofRuntimeOptions opts { optimizer := optim.adam { lr := train.lr } }
  let task : Trainer.Task τ := .oneHotCrossEntropy 2
  let trainer := Trainer.new model (Trainer.Config.fromRunConfig run task)
  trainer.printInfo

  /-
  The GPT-2 command trains on a bounded, prompt-aware window table.  That makes the training
  schedule explicit and reproducible, and it lets the public trainer own checkpointing and optimizer
  state.  The example stays focused on text windows, decoding, and generation instead of runtime
  module bookkeeping.
  -/
  let trained ← trainer.train
    (Data.floatSamples samples)
    { steps := train.steps
      batchSize := train.batchSize
      cudaMemWatch := train.cudaMemWatch
      log := .disabled
      loadCheckpoint? := train.loadCheckpoint?
      saveCheckpoint? := train.saveCheckpoint? }
  let (beforeLoss, afterLoss) ←
    Trainer.TrainSummary.printFloatLosses exeName trained.report
      (steps? := some train.steps) (lr? := some train.lr)

  let afterLogits ← trained.predict (Sample.x reportSample)
  printPredictionReport "after " train.prompt afterLogits
  let generatedIds ← generateSampled trained.predict train.prompt train.generate
    train.temperature train.topK train.seed train.repeatWindow train.repeatPenalty train.asciiOnly
  let generated := text.escapeByteIdsForDisplay generatedIds
  IO.println s!"  generated={generated}"
  IO.println s!"  corpus_bytes={input.toByteArray.size} windows={samples.size}"
  IO.println s!"  sampling=top_k({train.topK}), temperature={train.temperature}, seed={train.seed}"
  IO.println s!"  repetition_penalty={train.repeatPenalty} repeat_window={train.repeatWindow}"
  if train.interactive then
    interactiveLoopFloat trained.predict train
  let cudaMemWatch := Trainer.Manual.CUDAMemory.cadence opts train.steps train.cudaMemWatch
  text.writeGenerationTrainLog
    train.log "GPT-2 byte prompt training" train.steps beforeLoss afterLoss
    train.toGenerationOptions generated
    #[ModelZoo.deviceNote opts,
      s!"windows={train.windows}",
      s!"cuda_mem_watch={cudaMemWatch}"]
  pure (beforeLoss, afterLoss, generated)

/-- CLI entrypoint for byte-level GPT training, sampling, logging, and checkpointing. -/
def main (args : List String) : IO UInt32 := do
  Module.Command.runFloat32 exeName args
    (banner := ModelZoo.bannerWithDevice exeName "causal LM training")
    (usage? := some usage)
    (k := fun opts rest => do
      let (input, rest) ← takeInputText rest
      let defaultSteps : Nat := if opts.usesCuda then 1 else 0
      let (train, rest) ← ModelZoo.orThrow exeName <|
        TrainOptions.parse rest defaultSteps
      CLI.requireNoArgs exeName rest
      let (_L0, _L1, _generated) ← trainAndDecode opts input train)

end NN.Examples.Models.Sequence.Gpt2
