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

The model consumes a fixed-length sequence. Constructors accept an arbitrary leading shape for
batches, ensembles, or other pointwise collections of sequences.
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

/-- Input shape `leading ++ (seqLen × inputSize)`. -/
abbrev RecurrentConfig.inputShape (cfg : RecurrentConfig)
    (leading : List Nat := []) : List Nat :=
  leading ++ [cfg.seqLen, cfg.inputSize]

/-- Output shape `leading ++ (seqLen × outputSize)`. -/
abbrev RecurrentConfig.outputShape (cfg : RecurrentConfig)
    (leading : List Nat := []) : List Nat :=
  leading ++ [cfg.seqLen, cfg.outputSize]

/--
Vanilla RNN core plus time-distributed linear head:

`rnn(seqLen, inputSize, hiddenSize) → linear(hiddenSize, outputSize)`.
-/
def rnnWithLinearHead (cfg : RecurrentConfig) (leading : List Nat := []) :
    nn.Builder (nn.Sequential (cfg.inputShape leading) (cfg.outputShape leading)) := do
  let recurrent ← nn.rnn cfg.seqLen cfg.inputSize cfg.hiddenSize (leading := leading)
  let headRaw ← linear cfg.hiddenSize cfg.outputSize
    (leading := leading ++ [cfg.seqLen])
  let head : nn.Sequential
      (leading ++ [cfg.seqLen, cfg.hiddenSize])
      (leading ++ [cfg.seqLen, cfg.outputSize]) := by
    simpa [List.append_assoc] using headRaw
  pure (recurrent >>> head)

/--
LSTM core plus time-distributed linear head:

`lstm(seqLen, inputSize, hiddenSize) → linear(hiddenSize, outputSize)`.
-/
def lstmWithLinearHead (cfg : RecurrentConfig) (leading : List Nat := []) :
    nn.Builder (nn.Sequential (cfg.inputShape leading) (cfg.outputShape leading)) := do
  let recurrent ← nn.lstm cfg.seqLen cfg.inputSize cfg.hiddenSize (leading := leading)
  let headRaw ← linear cfg.hiddenSize cfg.outputSize
    (leading := leading ++ [cfg.seqLen])
  let head : nn.Sequential
      (leading ++ [cfg.seqLen, cfg.hiddenSize])
      (leading ++ [cfg.seqLen, cfg.outputSize]) := by
    simpa [List.append_assoc] using headRaw
  pure (recurrent >>> head)

end models
end nn

end TorchLean
