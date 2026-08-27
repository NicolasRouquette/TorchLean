/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Device-agnostic example:
  lake exe torchlean rnn --device cpu
  lake -R -K cuda=true exe torchlean rnn --device cuda

This example trains a tiny byte-level RNN on real text:
- load a corpus through `--tiny-shakespeare` or `--data-file`,
- turn the first few bytes into a next-token training window,
- train `nn.rnn` plus a time-distributed linear head.
-/

module

public import NN.API
public import NN.Examples.Models.Common.RealData

/-!
# RNN Text Example

Runnable `torchlean rnn` example. It reads a local text corpus, takes a short byte window from the
front, and trains a vanilla RNN plus a time-distributed linear head.

The model constructor is exposed as `TorchLean.nn.models.rnnWithLinearHead`. The local code names the
architecture, builds the text dataset, and trains through the public `Trainer` surface.

## Scope

This is the plain recurrent baseline. It keeps the text window short so the example stays focused on
the recurrent cell, the time-distributed head, and the public `Trainer` API. For generation and
longer contexts, use `chargpt`, `gpt2`, or `text_gpt2`.

```bash
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
lake -R -K cuda=true exe torchlean rnn --device cuda --tiny-shakespeare --steps 1
```
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Sequence.Rnn

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "torchlean rnn"

/-- Default JSON loss-curve path for this command. -/
def defaultLogJson : System.FilePath := ModelZoo.trainLogPath "rnn"

/-- Number of byte-level timesteps in the training window. -/
def seqLen : Nat := 2
/-- Tiny one-hot token width for the example dataset. -/
def inputSize : Nat := 4

/-- Toy byte bucketing: encode byte id `b` as `b % 4`; collisions are intentional. -/
def byteBucket (id : Nat) : Fin inputSize :=
  ⟨id % inputSize, Nat.mod_lt _ (by decide)⟩
/-- Hidden state width of the vanilla recurrent cell. -/
def hiddenSize : Nat := 2

/-- Shared shape/config record for the reusable RNN-with-head constructor. -/
def cfg : nn.models.RecurrentConfig :=
  { seqLen := seqLen
    inputSize := inputSize
    hiddenSize := hiddenSize
    outputSize := inputSize }

/-- Input shape: one token vector per timestep. -/
abbrev σ :=
  cfg.inputShape

/-- Output shape: one prediction row per timestep. -/
abbrev τ :=
  cfg.outputShape

/-- Vanilla RNN followed by a time-distributed linear output head. -/
def model : nn.Builder (nn.Sequential σ τ) :=
  nn.models.rnnWithLinearHead cfg

/-- Build one next-token training sample from the loaded corpus prefix. -/
def sample (corpus : String) : Sample.Supervised Float σ τ :=
  let s := Data.CausalLM.byteSample
    (α := Float) seqLen inputSize byteBucket (corpus.take (seqLen + 1)).toString
  Sample.mk (Spec.Tensor.materialize (Sample.x s)) (Spec.Tensor.materialize (Sample.y s))

/-- Train the vanilla RNN with the public `Trainer` surface. -/
def train (opts : Options) (corpusFlags : RealData.TextCorpusFlags)
    (flags : CLI.Training.OptimizerOptions) : IO Unit := do
  let corpus ← RealData.TextCorpusFlags.read exeName corpusFlags
  let trainer :=
    Trainer.new model <|
      Trainer.Config.fromRunConfig
        (Trainer.RunConfig.ofRuntimeOptions opts { optimizer := optim.sgd { lr := flags.lr } })
        .regression
  let trainData := Data.floatSamples #[sample corpus]
  let trained ← trainer.train
    trainData
    (CLI.Training.OptimizerOptions.toTrainerOptions flags
      (title := "RNN text training")
      (notes := #[s!"corpus={corpusFlags.path}"]))
  trained.printSummary

/-- CLI entrypoint for the vanilla RNN text command. -/
def main (args : List String) : IO UInt32 := do
  TrainCommand.run
    { exeName := exeName
      defaultLogJson := defaultLogJson
      defaultSteps := 1
      defaultLr := 1e-2
      description := "vanilla RNN"
      dataOptions := RealData.TextCorpusFlags.help
      parseData := RealData.TextCorpusFlags.parse
      train := train }
    args

end NN.Examples.Models.Sequence.Rnn
