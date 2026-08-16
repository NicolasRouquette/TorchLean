/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded

/-!
# Recurrent Models

RNN and LSTM sequence models with a linear projection at every time step.
-/

@[expose] public section

namespace TorchLean


open Spec Tensor

namespace nn
namespace models

/--
Configuration for an RNN or LSTM followed by a time-distributed linear head.

Shapes follow the convention used by the runnable examples:
- input: `(seqLen × inputSize)`
- output: `(seqLen × outputSize)`
-/
structure RecurrentConfig where
  /-- Number of time steps. -/
  seqLen : Nat
  /-- Number of features presented at each time step. -/
  inputSize : Nat
  /-- Width of the recurrent state. -/
  hiddenSize : Nat
  /-- Number of features produced at each time step. -/
  outputSize : Nat
deriving Repr

/-- Input shape `(seqLen × inputSize)`. -/
abbrev recurrentInShape (cfg : RecurrentConfig) : Spec.Shape :=
  .dim cfg.seqLen (.dim cfg.inputSize .scalar)

/-- Output shape `(seqLen × outputSize)`. -/
abbrev recurrentOutShape (cfg : RecurrentConfig) : Spec.Shape :=
  .dim cfg.seqLen (.dim cfg.outputSize .scalar)

/--
Vanilla RNN core plus time-distributed linear head:

`rnn(seqLen, inputSize, hiddenSize) → linear(hiddenSize, outputSize)`.
-/
def rnnWithLinearHead (cfg : RecurrentConfig) :
    nn.Builder (nn.Sequential (recurrentInShape cfg) (recurrentOutShape cfg)) :=
  nn.Sequential![
    nn.rnn cfg.seqLen cfg.inputSize cfg.hiddenSize,
    linear cfg.hiddenSize cfg.outputSize (pfx := .dim cfg.seqLen .scalar)
  ]

/--
LSTM core plus time-distributed linear head:

`lstm(seqLen, inputSize, hiddenSize) → linear(hiddenSize, outputSize)`.
-/
def lstmWithLinearHead (cfg : RecurrentConfig) :
    nn.Builder (nn.Sequential (recurrentInShape cfg) (recurrentOutShape cfg)) :=
  nn.Sequential![
    nn.lstm cfg.seqLen cfg.inputSize cfg.hiddenSize,
    linear cfg.hiddenSize cfg.outputSize (pfx := .dim cfg.seqLen .scalar)
  ]

end models
end nn

end TorchLean
