/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.CLI
public import NN.API.Trainer.Reporting

/-!
# Training Command-Line Options

Command-line options shared by runnable training programs. These records describe process-level
choices such as step counts, log destinations, and CUDA allocator sampling. Model definitions and
the `Trainer.train` API do not depend on them.
-/

@[expose] public section

namespace TorchLean.CLI.Training

/-- Step, batching, logging, and allocator-reporting options for a training command. -/
structure RunOptions where
  /-- Number of optimizer updates. -/
  steps : Nat
  /-- Number of in-memory samples consumed by one optimizer update. -/
  batchSize : Nat := 1
  /-- Destination for the JSON training log. -/
  log : _root_.Runtime.Training.LogDestination
  /-- Resolved log path, retained for command summaries. -/
  logPath : System.FilePath
  /-- Number of completed steps between CUDA allocator samples; `0` selects the default policy. -/
  cudaMemWatch : Nat := 0
deriving Repr

namespace RunOptions

/-- Parse the common options accepted by runnable training commands. -/
def parse
    (exeName : String)
    (args : List String)
    (defaultLogPath : System.FilePath)
    (defaultSteps : Nat := 1)
    (allowZeroSteps : Bool := false) :
    Except String (RunOptions × List String) := do
  let (logRaw?, args) ← CLI.takeFlagValueOnce args "log"
  let (steps, args) ← CLI.takeStepsFlagDefault args defaultSteps
  let (batchSize?, args) ← CLI.takeNatFlagOnce args "batch-size"
  let (cudaMemWatch?, args) ← CLI.takeNatFlagOnce args "cuda-mem-watch"
  if !allowZeroSteps && steps = 0 then
    throw s!"{exeName}: --steps must be > 0"
  let batchSize := batchSize?.getD 1
  if batchSize = 0 then
    throw s!"{exeName}: --batch-size must be > 0"
  let log := _root_.Runtime.Training.LogDestination.parse? defaultLogPath logRaw?
  pure
    ({ steps, batchSize, log, logPath := log.pathD defaultLogPath,
       cudaMemWatch := cudaMemWatch?.getD 0 }, args)

end RunOptions

/-- Training command options that also select a learning rate. -/
structure OptimizerOptions extends RunOptions where
  /-- Learning rate passed to the command's optimizer constructor. -/
  lr : Float
deriving Repr

namespace OptimizerOptions

/-- Parse run options followed by a positive `--lr` value. -/
def parse
    (exeName : String)
    (args : List String)
    (defaultLogPath : System.FilePath)
    (defaultSteps : Nat := 1)
    (defaultLr : Float := 1e-3)
    (allowZeroSteps : Bool := false) :
    Except String (OptimizerOptions × List String) := do
  let (run, args) ← RunOptions.parse exeName args defaultLogPath defaultSteps allowZeroSteps
  let (lr, args) ← CLI.takePositiveFloatFlag args exeName "lr" defaultLr
  pure ({ toRunOptions := run, lr }, args)

end OptimizerOptions

end TorchLean.CLI.Training
