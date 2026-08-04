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

/-- Training and generation context length for the Mamba text example. -/
def seqLen : Nat := 2

/-- Byte tokenizer used by this sequence model. -/
def tokenizer : text.Tokenizer := text.Tokenizer.byte

/-- Mamba text-model configuration shared by shapes and the constructor. -/
def cfg : nn.models.MambaTextConfig :=
  { vocab := 32
    stateDim := 4
    ssmStateDim := 2
    convWidth := 3 }

/-- Input shape: one sequence of one-hot byte tokens. -/
abbrev σ := nn.models.mambaTokenMat cfg seqLen

/-- Output shape: one vocabulary-logit row per input position. -/
abbrev τ := nn.models.mambaLogitMat cfg seqLen

/-- Public Mamba language-model constructor specialized to the example config. -/
def model : nn.M (nn.Sequential σ τ) :=
  nn.models.mambaTextLm cfg seqLen

/-- Convert a token window into the one-hot next-token sample consumed by the Mamba model. -/
def sampleFromTokenIds (ids : List Nat) : SupervisedSample Float σ τ :=
  let (xF, yF) := text.causalLmXYOneHotMatFloat
    (seqLen := seqLen) (vocab := cfg.vocab) (ids.map (· % cfg.vocab))
  Sample.mk xF yF

/-- Build a finite cyclic training set from corpus text, biased toward the prompt when present. -/
def samplesFromCorpus (input _prompt : String) (windows : Nat) :
    Array (SupervisedSample Float σ τ) :=
  let toks := tokenizer.encode input
  let offsets :=
    nn.models.mambaTrainingOffsets toks.length seqLen windows
  offsets.toArray.map (fun off =>
    -- Slice real corpus text into a tiny next-token window. Larger `--windows` values give a more
    -- interesting training run, but the default stays small so the command is a reliable quick check.
    let ids := text.tokenWindow tokenizer (seqLen + 1) input (offset := off) (padId := 32)
    sampleFromTokenIds ids)

/-- Print the current argmax prediction beside the prompt and shifted target text. -/
def printPredictionReport (label prompt : String) (logits : Tensor.T Float τ) : IO Unit := do
  IO.println s!"  {label} pred={text.escapeForDisplay (text.decodeArgmaxLogits tokenizer logits)}"
  IO.println s!"  prompt={text.escapeForDisplay (text.decodeWindow tokenizer seqLen prompt (padId := 32))}"
  IO.println s!"  target={text.escapeForDisplay (text.decodeWindow tokenizer seqLen prompt (offset := 1) (padId := 32))}"

/-- Convert a prompt window into the typed one-hot input tensor used during generation. -/
def inputTensorFromIds (ids : List Nat) : Tensor.T Float σ :=
  let (xF, _) := text.causalLmXYOneHotMatFloat
    (seqLen := seqLen) (vocab := cfg.vocab) (ids.map (· % cfg.vocab))
  xF

/-- Autoregressively extend a prompt using the trained Mamba parameters. -/
partial def generateSampled
    (predict : Tensor.T Float σ → IO (Tensor.T Float τ))
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
    (train : text.WindowedTrainGenerationOptions) :
    IO (Float × Float) := do
  let samples := samplesFromCorpus input train.prompt train.windows
  let reportSample := sampleFromTokenIds (text.tokenWindow tokenizer (seqLen + 1) train.prompt
    (padId := 32))
  let run := Trainer.runConfig opts { optimizer := optim.adam { lr := train.lr } }
  let trainer := Trainer.new model <|
    Trainer.Config.fromRunConfig run .crossEntropy
  let cudaMemWatch := ModelZoo.effectiveCudaMemWatch opts train.steps train.cudaMemWatch
  let trained ← trainer.train
    (Data.floatSampleArray samples)
    (CLI.Training.OptimizerOptions.toTrainerOptions train.toOptimizerOptions
      (title := "Mamba text training")
      (notes := #[ModelZoo.deviceNote opts, s!"windows={train.windows}",
        s!"cuda_mem_watch={cudaMemWatch}"])).disableLog
  let afterLogits ← trained.predict (Sample.x reportSample)
  printPredictionReport "after " train.prompt afterLogits
  let (beforeLoss, afterLoss) ←
    Trainer.TrainSummary.requireAndPrintFloatLosses exeName trained.report
      (steps? := some train.steps) (lr? := some train.lr)
  let generated ← generateSampled trained.predict train.prompt train.generate
    train.temperature train.topK train.seed
  IO.println s!"  generated={text.escapeForDisplay generated}"
  IO.println s!"  corpus_bytes={input.toByteArray.size} windows={samples.size}"
  IO.println s!"  sampling=top_k({train.topK}), temperature={train.temperature}, seed={train.seed}"
  pure (beforeLoss, afterLoss)

/-- CLI entrypoint for the Mamba text command. -/
def main (args : List String) : IO UInt32 := do
  Runtime.runFloat exeName args
    (banner := ModelZoo.bannerWithDevice exeName "Mamba text training")
    (k := fun opts rest => do
      let (corpus, rest) ← ModelZoo.orThrow exeName <| RealData.TextCorpusFlags.parse rest
      let (train, rest) ← ModelZoo.orThrow exeName <|
        text.WindowedTrainGenerationOptions.parse
          exeName rest defaultLogJson 1 0.002 1
            { prompt := "First Citizen:"
              generate := 0
              temperature := 0.9
              topK := 16
              repeatPenalty := 1.0
              repeatWindow := 0
              seed := 0
              asciiOnly := false }
      CLI.requireNoArgs exeName rest
      let input ← RealData.TextCorpusFlags.read exeName corpus
      let (beforeLoss, afterLoss) ← trainOnText opts input train
      let extraNotes :=
        #[s!"data={corpus.path}", ModelZoo.deviceNote opts,
          s!"windows={train.windows}", s!"lr={train.lr}",
          ModelZoo.cudaMemWatchNote opts train.steps train.cudaMemWatch]
      text.writeGenerationTrainLog
        train.log "Mamba text training" train.steps beforeLoss afterLoss
        train.toGenerationOptions none extraNotes
    )

end NN.Examples.Models.Sequence.Mamba
