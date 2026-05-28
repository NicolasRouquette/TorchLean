/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Device-agnostic example:
  lake exe torchlean rnn --cpu
  lake build -R -K cuda=true && lake exe torchlean rnn --cuda

This is a real-data sequence run:
- reads a local text corpus (default: `data/real/text/tiny_shakespeare.txt`),
- builds a byte-level causal-LM one-hot window,
- trains `nn.rnn` plus a time-distributed linear head for one or more steps.
-/

module

public import NN
public import NN.API.Models.SimpleSeq
public import NN.Examples.Models.Sequence.SimpleText

/-!
# RNN Text Example

Runnable `torchlean rnn` example. It reads a local text corpus, creates a byte-level
causal-language-model window, and trains a vanilla RNN plus time-distributed linear head.

The model constructor lives in `NN.API.Models.SimpleSeq` so other examples can reuse it. This file
keeps only the architecture-specific declarations; the shared corpus loading, CLI parsing, logging,
and train loop live in `NN.Examples.Models.Sequence.SimpleText`.

## Scope

This command is the focused vanilla-RNN path: one real corpus window, one recurrent cell, and the
same training/logging interface used by the other sequence examples. For autoregressive sampling
and longer-context language-model behavior, use `chargpt`, `gpt2`, or `text_gpt2`.

```bash
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
lake build -R -K cuda=true && lake exe torchlean rnn --cuda --tiny-shakespeare --steps 1
```
-/

@[expose] public section

open Spec Tensor
open NN.API

namespace NN.Examples.Models.Sequence.Rnn

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "torchlean rnn"

/-- Default JSON loss-curve path for this command. -/
def defaultLogJson : System.FilePath := "data/model_zoo/rnn_trainlog.json"

/-- Byte-window length used by the typed recurrent sample. -/
def seqLen : Nat := 8
/--
Byte vocabulary size.

We use full byte tokens (`0..255`) so the sample is easy to interpret. If you want to make CPU runs
even faster, you can lower this to `32`, but you will be training on hashed bytes (`byte % vocab`).
-/
def inputSize : Nat := 256
/-- Hidden state width of the vanilla recurrent cell. -/
def hiddenSize : Nat := 64

/-- Shared shape/config record consumed by the reusable API constructor. -/
def cfg : nn.models.SeqRnnHeadConfig :=
  { seqLen := seqLen, inputSize := inputSize, hiddenSize := hiddenSize }

/-- Input shape: one byte-level one-hot vector per timestep. -/
abbrev σ : Shape :=
  nn.models.seqRnnHeadInShape cfg

/-- Output shape: one logit row per input timestep. -/
abbrev τ : Shape :=
  nn.models.seqRnnHeadOutShape cfg

/-- Vanilla RNN followed by a time-distributed linear output head. -/
def mkModel : nn.M (nn.Sequential σ τ) :=
  nn.models.rnnWithLinearHead cfg

/-- Convert corpus text into one supervised causal sequence window. -/
def mkSample {α : Type} [Semantics.Scalar α] [Runtime.Scalar α] (input : String) :
    API.sample.Supervised α σ τ :=
  RealData.textCausalSample (α := α) seqLen inputSize input

/-- Shared runner configuration for `torchlean rnn`. -/
def runner : SimpleText.RunnerConfig σ τ :=
  { exeName := exeName
    defaultLogJson := defaultLogJson
    modelName := "vanilla RNN"
    logTitle := "RNN text training"
    mkModel := mkModel
    mkSample := fun {α} _ _ input => mkSample (α := α) input
    lr := 1e-2 }

/-- CLI entrypoint for the vanilla RNN text command. -/
def main (args : List String) : IO UInt32 := do
  SimpleText.main runner args

end NN.Examples.Models.Sequence.Rnn
