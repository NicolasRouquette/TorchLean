/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Device-agnostic example:
  lake exe torchlean lstm --device cpu
  lake -R -K cuda=true exe torchlean lstm --device cuda

This example trains a tiny byte-level LSTM on real text:
- load a corpus through `--tiny-shakespeare` or `--data-file`,
- turn the first few bytes into a next-token training window,
- train `nn.lstm` plus a time-distributed linear head.
-/

module

public import NN.API
public import NN.Examples.Models.Common.RealData

/-!
# LSTM Text Example

Runnable `torchlean lstm` example. It reads a local text corpus, takes a short byte window from the
front, and trains an LSTM plus a time-distributed linear head.

The model constructor is exposed as `TorchLean.nn.models.lstmWithLinearHead`. The local code names the
architecture, builds the text dataset, and trains through the public `Trainer` surface.

## Scope

This is the gated recurrent baseline. It keeps the text window short so the example stays focused on
the LSTM cell, the time-distributed head, and the public `Trainer` API. For generation and
longer-context language-model behavior, use one of:
- `torchlean chargpt` (Karpathy-style, single-file char-level GPT),
- `torchlean gpt2` (byte-level GPT-2-style model + save/reload),
- `torchlean text_gpt2` (CUDA corpus trainer).

```bash
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
lake -R -K cuda=true exe torchlean lstm --device cuda --tiny-shakespeare --steps 1
```
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Sequence.Lstm

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "torchlean lstm"

/-- Default JSON loss-curve path for this command. -/
def defaultLogJson : System.FilePath := ModelZoo.trainLogPath "lstm"

/-- Number of byte-level timesteps in the training window. -/
def seqLen : Nat := 2
/-- Tiny one-hot token width for the example dataset. -/
def inputSize : Nat := 4
/-- Hidden state width of the LSTM cell. -/
def hiddenSize : Nat := 2

/-- Shared shape/config record for the reusable LSTM-with-head constructor. -/
def cfg : nn.models.RecurrentConfig :=
  { seqLen := seqLen
    inputSize := inputSize
    hiddenSize := hiddenSize
    outputSize := inputSize }

/-- Input shape: one token vector per timestep. -/
abbrev σ :=
  nn.models.recurrentInShape cfg

/-- Output shape: one prediction row per timestep. -/
abbrev τ :=
  nn.models.recurrentOutShape cfg

/-- LSTM followed by a time-distributed linear output head. -/
def model : nn.M (nn.Sequential σ τ) :=
  nn.models.lstmWithLinearHead cfg

/-- Build one next-token training sample from the loaded corpus prefix. -/
def sample (corpus : String) : SupervisedSample Float σ τ :=
  let s := Data.textCausalSample (α := Float) seqLen inputSize (corpus.take (seqLen + 1)).toString
  Sample.mk (Spec.Tensor.materialize (Sample.x s)) (Spec.Tensor.materialize (Sample.y s))

/-- Train the LSTM with the public `Trainer` surface. -/
def train (opts : Options) (corpusFlags : RealData.TextCorpusFlags)
    (flags : CLI.Training.RunOptions) : IO Unit := do
  let corpus ← RealData.TextCorpusFlags.read exeName corpusFlags
  let trainer :=
    Trainer.new model <|
      Trainer.Config.fromRunConfig
        (Trainer.runConfig opts { optimizer := optim.sgd { lr := 1e-2 } })
        .regression
  let trainData := Data.floatSamples [sample corpus]
  let trained ← trainer.train
    trainData
    (CLI.Training.RunOptions.toTrainerOptions flags
      (title := "LSTM text training")
      (notes := #[s!"corpus={corpusFlags.path}"]))
  trained.printSummary

/-- CLI entrypoint for the LSTM text command. -/
def main (args : List String) : IO UInt32 := do
  TrainCommand.run
    { exeName := exeName
      defaultLogJson := defaultLogJson
      defaultSteps := 1
      description := "LSTM"
      dataOptions := RealData.TextCorpusFlags.help
      parseData := RealData.TextCorpusFlags.parse
      train := train }
    args

end NN.Examples.Models.Sequence.Lstm
