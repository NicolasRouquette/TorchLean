/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team

Device-agnostic example:
  lake exe torchlean lstm --cpu
  lake build -R -K cuda=true && lake exe torchlean lstm --cuda

This is a real-data sequence run:
- reads a local text corpus selected by the shared `--tiny-shakespeare` / `--data-file` flags,
- builds a byte-level causal-LM one-hot window,
- trains `nn.lstm` plus a time-distributed linear head for one or more steps.
-/

module

public import NN
public import NN.API.Models.SimpleSeq
public import NN.Examples.Models.Sequence.SimpleText

/-!
# LSTM Text Example

Runnable `torchlean lstm` example. It reads a local text corpus, creates a byte-level
causal-language-model window, and trains an LSTM plus time-distributed linear head.

The model constructor lives in `NN.API.Models.SimpleSeq` so other examples can reuse it. This file
keeps only the architecture-specific declarations; the shared corpus loading, CLI parsing, logging,
and train loop live in `NN.Examples.Models.Sequence.SimpleText`.

## Scope

This command is the focused LSTM path: one real corpus window, one gated recurrent cell, and the same
training/logging interface used by the other sequence examples. For autoregressive sampling and
longer-context language-model behavior, use one of:
- `torchlean chargpt` (Karpathy-style, single-file char-level GPT),
- `torchlean gpt2` (byte-level GPT-2-style model + save/reload),
- `torchlean text_gpt2` (CUDA corpus trainer).

```bash
python3 scripts/datasets/download_example_data.py --tiny-shakespeare
lake build -R -K cuda=true && lake exe torchlean lstm --cuda --tiny-shakespeare --steps 1
```
-/

@[expose] public section

open Spec Tensor
open NN.API

namespace NN.Examples.Models.Sequence.Lstm

/-- CLI subcommand name used in terminal banners and error messages. -/
def exeName : String := "torchlean lstm"

/-- Default JSON loss-curve path for this command. -/
def defaultLogJson : System.FilePath := Common.modelZooTrainLog "lstm"

/-- Byte-window length used by the typed recurrent sample. -/
def seqLen : Nat := 8
/--
Byte vocabulary size.

This example uses byte-level tokens (`0..255`) rather than hashing bytes into a reduced bucket
count. The full byte vocabulary avoids unnecessary aliasing and keeps the sample semantics clear.
-/
def inputSize : Nat := 256
/-- Hidden state width of the LSTM cell. -/
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

/-- LSTM followed by a time-distributed linear output head. -/
def mkModel : nn.M (nn.Sequential σ τ) :=
  nn.models.lstmWithLinearHead cfg

/-- Convert corpus text into one supervised causal sequence window. -/
def mkSample {α : Type} [Semantics.Scalar α] [Runtime.Scalar α] (input : String) :
    API.sample.Supervised α σ τ :=
  RealData.textCausalSample (α := α) seqLen inputSize input

/-- Shared runner configuration for `torchlean lstm`. -/
def runner : SimpleText.RunnerConfig σ τ :=
  { exeName := exeName
    defaultLogJson := defaultLogJson
    modelName := "LSTM"
    logTitle := "LSTM text training"
    mkModel := mkModel
    mkSample := fun {α} _ _ input => mkSample (α := α) input
    lr := 1e-2 }

/-- CLI entrypoint for the LSTM text command. -/
def main (args : List String) : IO UInt32 := do
  SimpleText.main runner args

end NN.Examples.Models.Sequence.Lstm
