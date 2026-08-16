/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.Examples.ModelZoo

/-!
# Quickstart Shared Parsing

Small command-line parsers shared by the first-tour examples.

These are example utilities, not part of the training API. User code should still start from
`Trainer.new` and `trainer.train`; this file only keeps repeated
quickstart flag parsing out of the tutorial bodies.
-/

@[expose] public section

namespace NN.Examples.Quickstart

open TorchLean

/-- Parsed runtime and training settings for quickstart commands. -/
structure RuntimeTrain where
  /-- Logged training flags parsed from `--steps`, `--log`, and related options. -/
  train : CLI.Training.RunOptions
  /-- Runtime settings parsed from scalar, execution-mode, and device flags. -/
  run : Trainer.RunConfig
  /-- Public trainer training options derived from the parsed flags. -/
  trainOptions : Trainer.TrainOptions

/--
Parse the common quickstart tail:

`--steps`, optional logging flags, and runtime flags such as `--scalar`, `--execution`, or `--device`.

Each quickstart still owns its model, dataset, task, and any tutorial-specific flags.
-/
def parseRuntimeTrain
    (exeName : String)
    (args : List String)
    (defaultLogJson : System.FilePath)
    (defaultSteps : Nat)
    (optimizer : optim.Optimizer)
    (logEvery : Nat := 0) :
    IO RuntimeTrain := do
  let (train, args) ← CLI.orThrow exeName <|
    CLI.Training.RunOptions.parse exeName args defaultLogJson defaultSteps
  let trainOptions :=
    CLI.Training.RunOptions.toTrainerOptionsWhenRequested train args
      (logEvery := logEvery)
  let run ← Trainer.RunConfig.parseRuntimeArgsOrThrow exeName args
    { optimizer := optimizer }
  pure { train := train, run := run, trainOptions := trainOptions }

end NN.Examples.Quickstart
