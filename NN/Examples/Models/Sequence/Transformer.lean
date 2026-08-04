/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Real-data CUDA example:
  python3 scripts/datasets/download_example_data.py --tiny-shakespeare
  lake -R -K cuda=true exe torchlean transformer --device cuda --tiny-shakespeare --steps 1
-/

module


public import NN.API
public import NN.Examples.Models.Common.RealData

/-!
# Transformer Text Example

Runnable `torchlean transformer` example. It reads a local text corpus, builds a short sequence
reconstruction sample, and trains one transformer encoder block on that real text window.

The reusable model wiring is exposed as `TorchLean.nn.models.transformerEncoder`. This command stays
small so attention, normalization, the optimizer, logging, and CUDA execution remain easy to test
regularly.

```bash
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
lake -R -K cuda=true exe torchlean transformer --device cuda --tiny-shakespeare --steps 1
```
-/

@[expose] public section

open TorchLean

namespace NN.Examples.Models.Sequence.Transformer

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "torchlean transformer"

/-- Default JSON loss-curve path for this command. -/
def defaultLogJson : System.FilePath := ModelZoo.trainLogPath "transformer"

/-- Number of rows in the typed encoder batch. -/
def batch : Nat := 1

/-- Short reconstruction window for the quick encoder training run. -/
def seqLen : Nat := 1
/-- Transformer feature width. -/
def dModel : Nat := 2
/-- Number of attention heads. -/
def numHeads : Nat := 1
/-- Per-head width; $\mathtt{numHeads}\cdot\mathtt{headDim}$ matches `dModel`. -/
def headDim : Nat := 2
/-- Feed-forward hidden width inside the encoder block. -/
def ffnHidden : Nat := 4

/-- API-level encoder configuration shared by shapes and the constructor. -/
def cfg : nn.models.TransformerEncoderConfig :=
  { batch := batch
    seqLen := seqLen
    dModel := dModel
    numHeads := numHeads
    headDim := headDim
    ffnHidden := ffnHidden }

/-- Input shape: a batch of sequence rows with `dModel` features per token. -/
abbrev σ :=
  nn.models.transformerEncoderShape cfg

/-- Output shape matches the input because this command trains a reconstruction objective. -/
abbrev τ :=
  σ

/-- One reusable transformer encoder block from the public model API. -/
def model : nn.M (nn.Sequential σ τ) :=
  nn.models.transformerEncoder cfg (by decide) (by decide)

/-- Build one reconstruction sample from the loaded corpus prefix. -/
def sample (corpus : String) : SupervisedSample Float σ τ :=
  let s := Data.textCausalBatchSample (α := Float) batch seqLen dModel
    (corpus.take (seqLen + 1)).toString
  Sample.mk (Spec.Tensor.materialize (Sample.x s)) (Spec.Tensor.materialize (Sample.y s))

/-- Train the Transformer encoder with the public `Trainer` surface. -/
def train (opts : Options) (corpusFlags : RealData.TextCorpusFlags)
    (flags : CLI.Training.RunOptions) : IO Unit := do
  let corpus ← RealData.TextCorpusFlags.read exeName corpusFlags
  let trainer :=
    Trainer.new model <|
      Trainer.Config.fromRunConfig
        (Trainer.runConfig opts { optimizer := optim.sgd { lr := 1e-4 } })
        .regression
  let trainData := Data.floatSamples [sample corpus]
  let trained ← trainer.train
    trainData
    (CLI.Training.RunOptions.toTrainerOptions flags
      (title := "Transformer text training")
      (notes := #[s!"corpus={corpusFlags.path}"]))
  trained.printSummary

/-- CLI entrypoint for the Transformer encoder text command. -/
def main (args : List String) : IO UInt32 := do
  TrainCommand.run
    { exeName := exeName
      defaultLogJson := defaultLogJson
      defaultSteps := 1
      description := "Transformer encoder"
      dataOptions := RealData.TextCorpusFlags.help
      parseData := RealData.TextCorpusFlags.parse
      train := train }
    args

end NN.Examples.Models.Sequence.Transformer
