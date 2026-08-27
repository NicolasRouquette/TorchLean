/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.CLI.Parser

/-!
# Training Flag Presets

Small parser compositions for the training flags shared by runnable examples.
-/

@[expose] public section

namespace TorchLean.CLI

/-- Common training flags for epoch-oriented loader and tutorial commands. -/
structure EpochBatch where
  /-- Number of epochs to train for. -/
  epochs : Nat
  /-- Batch size. -/
  batch : Nat

/-- Parse `--epochs` and `--batch`, requiring both selected values to be positive. -/
def takePositiveEpochBatch
    (args : List String)
    (exeName : String)
    (defaultEpochs defaultBatch : Nat) :
    Except String (EpochBatch × List String) := do
  let (epochs, args) ← takeNatFlagDefault args "epochs" defaultEpochs
  let (batch, args) ← takeNatFlagDefault args "batch" defaultBatch
  if epochs = 0 then
    throw s!"{exeName}: --epochs must be > 0"
  if batch = 0 then
    throw s!"{exeName}: --batch must be > 0"
  pure ({ epochs, batch }, args)

/-- Parse an optional `--steps` flag with the provided default. -/
def takeStepsFlagDefault (args : List String) (default : Nat) :
    Except String (Nat × List String) :=
  takeNatFlagDefault args "steps" default

end TorchLean.CLI
