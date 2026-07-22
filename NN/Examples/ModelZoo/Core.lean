/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.Trainer
public import NN.Examples.ModelZoo.Base

/-!
# TorchLean Model-Zoo API

Flag and logging adapters used by runnable examples.

The ordinary training API is `Trainer.new` plus `trainer.train`. This module keeps repository
command plumbing in the `ModelZoo` namespace instead of mixing it into the trainer core.
-/

@[expose] public section

namespace NN.Examples.ModelZoo

open TorchLean

/-- Parse shared logged-training flags for model-zoo commands. -/
def parseLoggedTrainFlags (exeName : String) (args : List String)
    (defaultLogPath : System.FilePath) (defaultSteps : Nat := 1)
    (allowZeroSteps : Bool := false) :
    Except String (LoggedTrainFlags × List String) :=
  NN.API.Common.parseLoggedTrainFlags exeName args defaultLogPath defaultSteps allowZeroSteps

/-- Parse shared optimizer/training flags for model-zoo commands. -/
def parseTrainFlags (exeName : String) (args : List String)
    (defaultLogPath : System.FilePath) (defaultSteps : Nat := 1) (defaultLr : Float := 1e-3)
    (allowZeroSteps : Bool := false) :
    Except String (TrainFlags × List String) :=
  NN.API.Common.parseModelTrainFlags exeName args defaultLogPath defaultSteps defaultLr allowZeroSteps

namespace LoggedTrainFlags

/-- Build public trainer options from logged model-zoo flags. -/
def trainOptions (flags : ModelZoo.LoggedTrainFlags)
    (enableLog : Bool := true)
    (logEvery : Nat := 0)
    (title : String := "Training")
    (notes : Array String := #[]) :
    Trainer.TrainOptions :=
  { steps := flags.steps
    batchSize := flags.batchSize
    log := if enableLog then flags.log else .disabled
    logEvery := logEvery
    cudaMemWatch := flags.cudaMemWatch
    title := title
    notes := notes }

/--
Build training options that write a log only when the original CLI arguments include `--log`.

Tutorial commands use this when they want clean terminal output by default but still support the
standard TrainLog artifact without repeating the flag check in each file.
-/
def trainOptionsWhenLogRequested (flags : ModelZoo.LoggedTrainFlags)
    (args : List String)
    (logEvery : Nat := 0)
    (title : String := "Training")
    (notes : Array String := #[]) :
    Trainer.TrainOptions :=
  trainOptions flags (enableLog := CLI.hasFlagValue args "log")
    (logEvery := logEvery) (title := title) (notes := notes)

/-- Write a before/after loss artifact from logged model-zoo flags. -/
def writeBeforeAfterLossLog
    (flags : ModelZoo.LoggedTrainFlags)
    (title : String)
    (beforeLoss afterLoss : Float)
    (notes : Array String := #[]) : IO Unit :=
  ModelZoo.writeBeforeAfterLossLog flags.log title flags.steps beforeLoss afterLoss notes

end LoggedTrainFlags

namespace TrainFlags

/-- Build public trainer options from optimizer/training model-zoo flags. -/
def trainOptions (flags : ModelZoo.TrainFlags)
    (logEvery : Nat := 0)
    (title : String := "Training")
    (notes : Array String := #[]) :
    Trainer.TrainOptions :=
  let base : NN.API.Common.LoggedTrainFlags := flags.toLoggedTrainFlags
  { steps := base.steps
    batchSize := base.batchSize
    log := base.log
    logEvery := logEvery
    cudaMemWatch := base.cudaMemWatch
    title := title
    notes := notes }

/-- Write a before/after loss artifact from optimizer/training model-zoo flags. -/
def writeBeforeAfterLossLog
    (flags : ModelZoo.TrainFlags)
    (title : String)
    (beforeLoss afterLoss : Float)
    (notes : Array String := #[]) : IO Unit :=
  ModelZoo.LoggedTrainFlags.writeBeforeAfterLossLog
    flags.toLoggedTrainFlags title beforeLoss afterLoss notes

end TrainFlags

end NN.Examples.ModelZoo
