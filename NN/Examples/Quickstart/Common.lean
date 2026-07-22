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
  train : ModelZoo.LoggedTrainFlags
  /-- Runtime settings parsed from dtype/backend/device flags. -/
  run : Trainer.RunConfig
  /-- Public trainer training options derived from the parsed flags. -/
  trainOptions : Trainer.TrainOptions

/--
Parse the common quickstart tail:

`--steps`, optional logging flags, and runtime flags such as `--dtype`, `--backend`, or `--device`.

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
    ModelZoo.parseLoggedTrainFlags exeName args defaultLogJson defaultSteps
  let trainOptions :=
    ModelZoo.LoggedTrainFlags.trainOptionsWhenLogRequested train args
      (logEvery := logEvery)
  let run ← Trainer.RunConfig.parseRuntimeArgsOrThrow exeName args
    { optimizer := optimizer }
  pure { train := train, run := run, trainOptions := trainOptions }

end NN.Examples.Quickstart
