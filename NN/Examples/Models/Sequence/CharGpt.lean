/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.Examples.Models.Common.RealData

/-!
# Char-GPT (minGPT-style) Example

This example follows the character-level Transformer from Andrej Karpathy's
"Let's build GPT: from scratch, in code, spelled out" lecture:

- build an alphabet (`itos`) from the training text,
- build a `stoi` tokenizer from that alphabet,
- train a compact causal Transformer to predict the next character,
- sample text continuations from a prompt.

The `karpathy` preset follows the lecture configuration: batch size 64, context length 256, width 384,
six attention heads, six pre-normalized Transformer blocks, ReLU feed-forward layers, dropout 0.2,
AdamW, and 5,000 updates. The CUDA command executes the numerical path in float32. TorchLean applies
dropout to the attention and feed-forward
sublayer outputs; unlike the lecture code, it does not yet apply a second dropout to the attention
weights themselves. All dimensions remain command-line choices.

Training draws a fresh deterministic batch of corpus windows at every step.  The windows are built
on demand, so a long run does not retain thousands of large one-hot tensors in host memory.

Quick check:

```bash
lake -R -K cuda=true build torchlean:exe
lake -R -K cuda=true exe torchlean chargpt --device cuda --tiny-shakespeare --preset smoke
```

Full lecture experiment:

```bash
lake -R -K cuda=true exe torchlean chargpt --device cuda --tiny-shakespeare --preset karpathy
```

Reference: <https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py>.
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Sequence.CharGpt

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "torchlean chargpt"

/-- Parse corpus flags and return the UTF-8 training text plus remaining CLI arguments. -/
def takeInputText (args : List String) : IO (String × List String) :=
  text.Corpus.takeUtf8Input exeName RealData.tinyShakespearePath
    [("--tiny-shakespeare", RealData.tinyShakespearePath)] RealData.missingTinyShakespeareHint args

/-- Build a deterministic character alphabet from the corpus. -/
def buildAlphabet (s : String) : Array Char :=
  let chars : List Char := s.toList.eraseDups
  -- Deterministic order: sort by codepoint.
  let sorted := List.mergeSort chars (fun a b => decide (a.toNat ≤ b.toNat))
  sorted.toArray

/-- Default JSON loss-curve path for this command. -/
def defaultLogJson : System.FilePath := ModelZoo.trainLogPath "chargpt"

/-- Architecture and evaluation controls independent of the corpus and runtime device. -/
structure ExperimentConfig where
  width : Nat
  heads : Nat
  layers : Nat
  dropout : Float
  steps : Nat
  batch : Nat
  seqLen : Nat
  learningRate : Float
  evalEvery : Nat
  evalIters : Nat
  generate : Nat
  deriving Repr

namespace ExperimentConfig

/-- Fast configuration used to validate the complete training and generation path. -/
def smoke : ExperimentConfig :=
  { width := 32
    heads := 4
    layers := 2
    dropout := 0.0
    steps := 2
    batch := 2
    seqLen := 16
    learningRate := 3e-4
    evalEvery := 1
    evalIters := 1
    generate := 32 }

/-- Hyperparameters from Karpathy's final Tiny Shakespeare lecture model. -/
def karpathy : ExperimentConfig :=
  { width := 384
    heads := 6
    layers := 6
    dropout := 0.2
    steps := 5000
    batch := 64
    seqLen := 256
    learningRate := 3e-4
    evalEvery := 500
    evalIters := 200
    generate := 500 }

def ofName (name : String) : Except String ExperimentConfig :=
  match name.trimAscii.toString.toLower with
  | "smoke" => .ok smoke
  | "karpathy" => .ok karpathy
  | other => .error s!"unknown --preset '{other}'; expected smoke or karpathy"

end ExperimentConfig

/-- Help text for character-level GPT training. -/
def usage : String :=
  String.intercalate "\n"
    [ "torchlean chargpt: character-level GPT training"
    , ""
    , "Usage:"
    , "  lake -R -K cuda=true exe torchlean chargpt --device cuda --tiny-shakespeare --preset PRESET [flags]"
    , ""
    , "Presets:"
    , "  smoke       two-update end-to-end CUDA check"
    , "  karpathy    full Tiny Shakespeare lecture experiment"
    , ""
    , "Architecture:"
    , "  --width N       embedding width"
    , "  --heads N       attention heads; must divide width"
    , "  --layers N      Transformer blocks"
    , "  --dropout P     dropout probability in [0, 1)"
    , "  --batch N       training windows per update"
    , "  --seq-len N     context length"
    , "  --steps N       optimizer updates"
    , "  --lr FLOAT      AdamW learning rate"
    , "  --eval-every N  validation cadence; 0 disables intermediate evaluation"
    , "  --eval-iters N  batches averaged at each validation point"
    , "  --load-params P load an exact Float parameter checkpoint"
    , "  --save-params P save final parameters"
    , ""
    , "Notes:"
    , "  - Training and validation use disjoint 90/10 corpus splits."
    , "  - CUDA runs use TorchLean's float32 runtime; CPU runs use Lean's host Float."
    ]

/-- Decode token ids for terminal output with control characters escaped. -/
def escapeCharIdsForDisplay (t : text.Tokenizer) (ids : List Nat) : String :=
  text.escapeForDisplay (t.decode ids)

/-- Printable-ASCII generation filter used by `--ascii-only`. -/
def asciiAllowed (c : Char) : Bool :=
  c = '\n' || (32 ≤ c.toNat && c.toNat ≤ 126)

/-- Fitted predictor for a runtime-sized character GPT model. -/
abbrev Predictor (batch seqLen vocab : Nat) :=
  Tensor.T Nat (.dim (batch * seqLen) .scalar) →
    IO (Tensor.T Float (shape![batch, seqLen, vocab]))

/-- Autoregressively extend character token ids using a trained CharGPT model. -/
partial def generateSampledFromIds
    (batch seqLen vocab : Nat)
    (predict : Predictor batch seqLen vocab)
    (promptIds : List Nat)
    (steps : Nat) (temperature : Float) (topK seed repeatWindow : Nat)
    (repeatPenalty : Float) (allowId : Nat → Bool := fun _ => true)
    (padId : Nat := 0) : IO (List Nat) := do
  let gen : text.GenerationOptions :=
    { prompt := ""
      generate := steps
      temperature := temperature
      topK := topK
      repeatPenalty := repeatPenalty
      repeatWindow := repeatWindow
      seed := seed
      asciiOnly := false }
  if seqLen = 0 then
    -- The CLI rejects this case, but keeping the definition total makes the stream reusable.
    pure promptIds
  else if hBatch : batch = 0 then
    -- Likewise, `--batch 0` is rejected by the CLI parser.
    pure promptIds
  else
  let b0 : Fin batch := ⟨0, Nat.pos_of_ne_zero hBatch⟩
  text.autoregressiveTokenIds seqLen padId promptIds gen
    (fun padded predPos => do
        let x : Tensor.T Nat (.dim (batch * seqLen) .scalar) :=
          Tensor.vectorFromArray (batch * seqLen) padded.toArray padId
        let logits ← predict x
        pure (text.batchLogitScoresAt logits b0 predPos))
    (allowId := allowId)

/-- CLI entrypoint for character-level GPT training and sampling. -/
def main (args : List String) : IO UInt32 := do
  if args.contains "--help" || args.contains "-h" then
    IO.println usage
    return 0
  Runtime.runFloat exeName args
    (banner := fun _ => s!"{exeName}: char-level GPT training")
    (k := fun opts rest => do
      let (corpus, rest) ← takeInputText rest
      let (presetName?, rest) ← ModelZoo.orThrow exeName <|
        TorchLean.CLI.takeFlagValueOnce rest "preset"
      let defaults ← ModelZoo.orThrow exeName <|
        ExperimentConfig.ofName (presetName?.getD "smoke")
      let presetName := presetName?.getD "smoke" |>.trimAscii.toString.toLower
      let defaultPrompt :=
        if presetName == "karpathy" then "\n" else "First Citizen:"
      let generationDefaults : text.GenerationOptions :=
        if presetName == "karpathy" then
          { prompt := defaultPrompt
            generate := defaults.generate
            temperature := 1.0
            topK := 0
            repeatPenalty := 1.0
            repeatWindow := 0
            seed := 1337
            asciiOnly := false }
        else
          { prompt := defaultPrompt
            generate := defaults.generate
            temperature := 0.9
            topK := 12
            repeatPenalty := 1.15
            repeatWindow := 64
            seed := 7
            asciiOnly := false }
      let (train, rest) ← ModelZoo.orThrow exeName <|
        text.BatchedCheckpointedWindowedTrainGenerationOptions.parse
          exeName rest defaultLogJson defaults.steps defaults.learningRate 1
          defaults.batch defaults.seqLen generationDefaults
          (allowZeroSteps := true)
      let (width, rest) ← ModelZoo.orThrow exeName <|
        TorchLean.CLI.takePositiveNatFlagDefault rest exeName "width" defaults.width
      let (heads, rest) ← ModelZoo.orThrow exeName <|
        TorchLean.CLI.takePositiveNatFlagDefault rest exeName "heads" defaults.heads
      let (layers, rest) ← ModelZoo.orThrow exeName <|
        TorchLean.CLI.takePositiveNatFlagDefault rest exeName "layers" defaults.layers
      let (dropout, rest) ← ModelZoo.orThrow exeName <|
        TorchLean.CLI.takeNonnegativeFloatFlagDefault rest exeName "dropout" defaults.dropout
      let (evalEvery, rest) ← ModelZoo.orThrow exeName <|
        TorchLean.CLI.takeNatFlagDefault rest "eval-every" defaults.evalEvery
      let (evalIters, rest) ← ModelZoo.orThrow exeName <|
        TorchLean.CLI.takePositiveNatFlagDefault rest exeName "eval-iters" defaults.evalIters
      CLI.requireNoArgs exeName rest
      if dropout >= 1.0 then
        throw <| IO.userError s!"{exeName}: --dropout must be smaller than 1"
      if width % heads != 0 then
        throw <| IO.userError s!"{exeName}: --heads must divide --width"
      let alphabetFull := buildAlphabet corpus
      let tok := text.Tokenizer.ofAlphabet alphabetFull (unkId := 0) (unkChar := '?')
      let vocab := tok.vocabSize
      let batch := train.batch
      let seqLen := train.seqLen

      let cfg : nn.models.CausalTransformerConfig :=
        { batch := batch
          seqLen := seqLen
          vocab := vocab
          numHeads := heads
          headDim := width / heads
          ffnHidden := 4 * width
          layers := layers
          activation := .relu
          dropout? := if dropout == 0.0 then none else some dropout
          normFirst := true
          attentionOutputBias := true
          parameterInit? := some (.normal 0.0 0.02) }
      let seqWitness : Fin seqLen ←
        if h : 0 < seqLen then
          pure ⟨0, h⟩
        else
          throw <| IO.userError s!"{exeName}: --seq-len must be positive"
      let modelWitness : Fin cfg.dModel ←
        if h : 0 < cfg.dModel then
          pure ⟨0, h⟩
        else
          throw <| IO.userError s!"{exeName}: model width must be positive"
      have hSeq : seqLen ≠ 0 := by
        intro h
        rw [h] at seqWitness
        exact Fin.elim0 seqWitness
      have hModel : cfg.dModel ≠ 0 := by
        intro h
        rw [h] at modelWitness
        exact Fin.elim0 modelWitness
      let body : nn.Sequential (nn.models.causalEmbeddingShape cfg)
          (nn.models.causalVocabularyShape cfg) :=
        nn.run train.seed <|
          nn.models.causalTransformerFromEmbeddings cfg (h_seqLen := hSeq) (h_dModel := hModel)

      let allTokens := (tok.encode corpus).toArray
      let split := allTokens.size * 9 / 10
      let trainTokens := allTokens.extract 0 split
      let valTokens := allTokens.extract split allTokens.size
      if trainTokens.size <= seqLen || valTokens.size <= seqLen then
        throw <| IO.userError s!"{exeName}: corpus split is too short for context length {seqLen}"

      let mkBatchSample (step : Nat) :=
        Data.causalLmTokenBatchFromTokenArray
          batch seqLen trainTokens train.seed step (padId := 0)
      let mkValSample (step : Nat) :=
        Data.causalLmTokenBatchFromTokenArray
          batch seqLen valTokens (train.seed + 1000003) step (padId := 0)
      let trainDef := nn.models.causalTransformerTokenScalarModuleDef cfg body
      let evalDef := nn.models.causalTransformerTokenScalarModuleDefWithMode .eval cfg body
      let module ← TorchLean.Module.instantiateFloat trainDef opts
      match train.loadParams? with
      | none => pure ()
      | some path =>
          Checkpoint.loadModuleParams module path
          IO.println s!"  loaded params: {path}"
      let optimizer := _root_.Runtime.Autograd.TorchLean.Optim.adamw
        (α := Float) (paramShapes := nn.models.causalTransformerTokenParamShapes cfg body)
        train.lr 0.01 0.9 0.999 1e-8
      let optimizerState ← TorchLean.Module.initOptim module optimizer
      let optimizerStateRef ← IO.mkRef optimizerState
      let storedScalarCount :=
        (nn.models.causalTransformerTokenParamShapes cfg body).foldl
          (fun total shape => total + Shape.size shape) 0
      let trainableParameterCount :=
        (List.zip (nn.models.causalTransformerTokenParamShapes cfg body)
          (true :: nn.paramRequiresGrad body)).foldl
          (fun total entry => if entry.2 then total + Shape.size entry.1 else total) 0
      IO.println s!"  trainable_parameters={trainableParameterCount}"
      if storedScalarCount != trainableParameterCount then
        IO.println s!"  non_trainable_state_scalars={storedScalarCount - trainableParameterCount}"
      let evalLoss : IO Float := do
        let losses ← (List.range evalIters).mapM fun i => do
          let (tokens, targets) := mkValSample i
          let loss ← TorchLean.Module.forwardWithParams
            evalDef opts module.trainer.params .nil (.cons tokens (.cons targets .nil))
          pure (Spec.Tensor.toScalar loss)
        pure (losses.foldl (· + ·) 0.0 / Float.ofNat losses.length)
      let beforeLoss ← evalLoss
      IO.println s!"  step 0: val loss={beforeLoss}"
      for step in [0:train.steps] do
        let state ← optimizerStateRef.get
        let (tokens, targets) := mkBatchSample step
        let state' ← TorchLean.Module.stepWith module optimizer state .nil
          (.cons tokens (.cons targets .nil))
        optimizerStateRef.set state'
        let done := step + 1
        if evalEvery != 0 && (done % evalEvery == 0 || done == train.steps) then
          let loss ← evalLoss
          IO.println s!"  step {done}: val loss={loss}"
      let afterLoss ← evalLoss
      let forwardProgram : _root_.Runtime.Autograd.TorchLean.ProgramWithNatInputs Float
          (nn.models.causalTransformerTokenParamShapes cfg body ++ [])
          [nn.models.causalTokenShape cfg] (nn.models.causalVocabularyShape cfg) := by
        rw [List.append_nil]
        exact fun {m} _ _ =>
          nn.models.causalTransformerTokenProgram cfg body (α := Float) (m := m)
      let evaluator ← TorchLean.Module.withParams forwardProgram
        opts module.trainer.params
      let predict := fun (x : Tensor.T Nat (nn.models.causalTokenShape cfg)) =>
        TorchLean.Module.evaluateT evaluator .nil (.cons x .nil)
      let promptIds := tok.encode train.prompt
      let allowId : Nat → Bool :=
        if train.asciiOnly then
          fun i => asciiAllowed (alphabetFull.getD i '?')
        else
          fun _ => true
      let outIds ←
        generateSampledFromIds batch seqLen vocab predict promptIds
          train.generate train.temperature train.topK train.seed train.repeatWindow train.repeatPenalty
          (allowId := allowId) (padId := 0)
      let sampled := escapeCharIdsForDisplay tok outIds
      match train.saveParams? with
      | none => pure ()
      | some path =>
          Checkpoint.saveModuleParams module path
          IO.println s!"  wrote params: {path}"
      IO.println s!"  vocab={vocab} (unique chars)"
      IO.println s!"  architecture=width {width}, heads {heads}, layers {layers}, dropout {dropout}"
      IO.println s!"  sampled={sampled}"
      text.writeGenerationTrainLog
        train.log "CharGPT (minGPT-style)" train.steps beforeLoss afterLoss
        train.toGenerationOptions sampled
        #[ModelZoo.deviceNote opts,
          s!"vocab={vocab}",
          s!"train_tokens={trainTokens.size}",
          s!"validation_tokens={valTokens.size}",
          ModelZoo.cudaMemWatchNote opts train.steps train.cudaMemWatch]
      pure 0)

end NN.Examples.Models.Sequence.CharGpt
