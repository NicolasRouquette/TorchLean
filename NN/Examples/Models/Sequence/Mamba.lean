/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.Examples.Models.Common.RealData

/-!
# Mamba Text Training

Runnable byte-level language-model training with the public Mamba API constructor.

The model is trainable end-to-end:

`mamba(seqLen, vocab, stateDim) → linear(stateDim → vocab)`

and the same code runs on CPU or CUDA through TorchLean autograd.

```bash
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
lake -R -K cuda=true exe torchlean mamba --device cuda --tiny-shakespeare --steps 1 --windows 1 --generate 0
```
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Sequence.Mamba

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "torchlean mamba"

/-- Default JSON loss-curve path for this command. -/
def defaultLogJson : System.FilePath := ModelZoo.trainLogPath "mamba"

/-- Complete command help, including the text and training flags parsed after runtime selection. -/
def usage : String :=
  Module.Command.usage exeName ++ "\n" ++ String.intercalate "\n"
    [ "Text data:"
    , "  --data-file PATH | --tiny-shakespeare | --tinystories-valid"
    , ""
    , "Training:"
    , "  --steps N          optimizer updates (default: 1)"
    , "  --batch-size N     corpus windows accumulated per update (default: 1)"
    , "  --windows N        corpus windows available to training (default: 1)"
    , "  --lr X             Adam learning rate (default: 0.002)"
    , "  --log PATH|false   write a TrainLog JSON, or disable logging"
    , "  --cuda-mem-watch N sample CUDA allocator state every N updates"
    , ""
    , "Generation:"
    , "  --prompt TEXT --generate N --temperature X --top-k N --sample-seed N"
    ]

/-- Training and generation context length for the Mamba text example. -/
def seqLen : Nat := 2

/-- Byte tokenizer used by this sequence model. -/
def tokenizer : text.Tokenizer := text.Tokenizer.byte

/-- Mamba text-model configuration shared by shapes and the constructor. -/
def cfg : nn.models.Mamba.Config :=
  { vocab := 32
    stateDim := 4 }

/-- Toy byte bucketing: encode byte id `b` as `b % 32`; collisions are intentional. -/
def byteBucket (id : Nat) : Fin cfg.vocab :=
  ⟨id % cfg.vocab, Nat.mod_lt _ (by decide)⟩

/-- Input shape: one sequence of one-hot byte tokens. -/
abbrev σ : List Nat := [seqLen, cfg.vocab]

/-- Output shape: one vocabulary-logit row per input position. -/
abbrev τ : List Nat := [seqLen, cfg.vocab]

/-- Public Mamba language-model constructor specialized to the example config. -/
def model : nn.Builder (nn.Sequential σ τ) :=
  nn.models.Mamba.textLM cfg seqLen

/-- Command-local training, sampling, and corpus-window controls. -/
structure TrainOptions extends
    CLI.Training.OptimizerOptions, text.GenerationOptions, text.WindowOptions where
deriving Repr

namespace TrainOptions

/-- Parse the Mamba command's training and sampling flags. -/
def parse (args : List String) : Except String (TrainOptions × List String) := do
  let (training, args) ←
    CLI.Training.OptimizerOptions.parse exeName args defaultLogJson 1 0.002
  let (window, args) ← text.WindowOptions.parse exeName args 1
  let (generation, args) ← text.GenerationOptions.parse exeName args
    { prompt := "First Citizen:"
      generate := 0
      temperature := 0.9
      topK := 16
      repeatPenalty := 1.0
      repeatWindow := 0
      seed := 0
      asciiOnly := false }
  pure ({ toOptimizerOptions := training
          toGenerationOptions := generation
          toWindowOptions := window }, args)

end TrainOptions

/-- Convert a token window into the one-hot next-token sample consumed by the Mamba model. -/
def sampleFromTokenIds (ids : Tensor Nat [seqLen + 1]) : Sample.Supervised Float σ τ :=
  let (xF, yF) := Data.CausalLM.oneHotPair (α := Float) []
    (seqLen := seqLen) (vocab := cfg.vocab) (ids.map byteBucket)
  Sample.mk xF yF

/-- Build a finite cyclic training set from corpus text, biased toward the prompt when present. -/
def samplesFromCorpus (input _prompt : String) (windows : Nat) :
    Array (Sample.Supervised Float σ τ) :=
  let toks := tokenizer.encode input
  let offsets :=
    text.Corpus.evenlySpacedOffsets toks.size seqLen windows
  offsets.map (fun off =>
    -- Slice real corpus text into a tiny next-token window. Larger `--windows` values give a more
    -- interesting training run, but the default stays small so the command is a reliable quick check.
    let ids := text.tokenWindow tokenizer (seqLen + 1) input (offset := off) (padId := 32)
    sampleFromTokenIds ids)

/-- Print the current argmax prediction beside the prompt and shifted target text. -/
def printPredictionReport (label prompt : String) (logits : Tensor Float τ) : IO Unit := do
  IO.println s!"  {label} pred={text.escapeForDisplay (text.decodeArgmaxLogits tokenizer logits)}"
  IO.println s!"  prompt={text.escapeForDisplay (text.decodeWindow tokenizer seqLen prompt (padId := 32))}"
  IO.println s!"  target={text.escapeForDisplay (text.decodeWindow tokenizer seqLen prompt (offset := 1) (padId := 32))}"

/-- Convert a prompt window into the typed one-hot input tensor used during generation. -/
def inputTensorFromIds (ids : Tensor Nat [seqLen]) : Tensor Float σ :=
  TorchLean.Tensor.oneHotIndices (α := Float) cfg.vocab (ids.map byteBucket)

/-- Autoregressively extend a prompt using the trained Mamba parameters. -/
partial def generateSampled
    (predict : Tensor Float σ → IO (Tensor Float τ))
    (prompt : String) (steps : Nat) (temperature : Float) (topK seed : Nat) : IO String := do
  let gen : text.GenerationOptions :=
    { prompt := prompt
      generate := steps
      temperature := temperature
      topK := topK
      repeatPenalty := 0.0
      repeatWindow := 0
      seed := seed
      asciiOnly := false }
  let ids ←
    text.autoregressiveTokenIds seqLen 32 (tokenizer.encode prompt) gen
      (fun padded predPos => do
        let logits ← predict (inputTensorFromIds padded)
        pure (text.logitScoresAt logits predPos))
  pure (tokenizer.decode ids)

/-- Train the Mamba language model and print before/after prediction and generation reports. -/
def trainOnText (opts : Options) (input : String)
    (train : TrainOptions) :
    IO (Float × Float) := do
  let samples := samplesFromCorpus input train.prompt train.windows
  let reportSample := sampleFromTokenIds (text.tokenWindow tokenizer (seqLen + 1) train.prompt
    (padId := 32))
  let run := Trainer.RunConfig.ofRuntimeOptions opts { optimizer := optim.adam { lr := train.lr } }
  let trainer := Trainer.new model <|
    Trainer.Config.fromRunConfig run (.oneHotCrossEntropy 1)
  let cudaMemWatch := Trainer.Manual.CUDAMemory.cadence opts train.steps train.cudaMemWatch
  let trained ← trainer.train
    (Data.floatSamples samples)
    (CLI.Training.OptimizerOptions.toTrainerOptions train.toOptimizerOptions
      (title := "Mamba text training")
      (notes := #[ModelZoo.deviceNote opts, s!"windows={train.windows}",
        s!"cuda_mem_watch={cudaMemWatch}"])).disableLog
  let afterLogits ← trained.predict (Sample.x reportSample)
  printPredictionReport "after " train.prompt afterLogits
  let (beforeLoss, afterLoss) ←
    Trainer.TrainSummary.printFloatLosses exeName trained.report
      (steps? := some train.steps) (lr? := some train.lr)
  let generated ← generateSampled trained.predict train.prompt train.generate
    train.temperature train.topK train.seed
  IO.println s!"  generated={text.escapeForDisplay generated}"
  IO.println s!"  corpus_bytes={input.toByteArray.size} windows={samples.size}"
  IO.println s!"  sampling=top_k({train.topK}), temperature={train.temperature}, seed={train.seed}"
  pure (beforeLoss, afterLoss)

/-- CLI entrypoint for the Mamba text command. -/
def main (args : List String) : IO UInt32 := do
  Module.Command.runFloat32 exeName args
    (banner := ModelZoo.bannerWithDevice exeName "Mamba text training")
    (usage? := some usage)
    (k := fun opts rest => do
      let (corpus, rest) ← ModelZoo.orThrow exeName <| RealData.TextCorpusFlags.parse rest
      let (train, rest) ← ModelZoo.orThrow exeName <|
        TrainOptions.parse rest
      CLI.requireNoArgs exeName rest
      let input ← RealData.TextCorpusFlags.read exeName corpus
      let (beforeLoss, afterLoss) ← trainOnText opts input train
      let extraNotes :=
        #[s!"data={corpus.path}", ModelZoo.deviceNote opts,
          s!"windows={train.windows}", s!"lr={train.lr}",
          ModelZoo.cudaMemoryNote opts train.steps train.cudaMemWatch]
      text.writeGenerationTrainLog
        train.log "Mamba text training" train.steps beforeLoss afterLoss
        train.toGenerationOptions none extraNotes
    )

end NN.Examples.Models.Sequence.Mamba
