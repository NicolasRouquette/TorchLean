/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.CLI
public import NN.Runtime.Training.Log

/-!
# RL Command-Line Options

TorchLean's runnable RL examples (`NN/Examples/Models/RL/*`) share one CLI shape:

- `--updates <n>`: how many update iterations to run,
- `--eval-every <n>`: evaluate every `n` updates,
- `--eval-episodes <n>`: number of evaluation episodes per checkpoint,
- `--eval-max-steps <n>`: max steps per evaluation episode,
- `--log <path|off|none|false>`: where to write the widget-friendly TrainLog JSON.

This module centralizes that parsing so we don't duplicate the same flag boilerplate across
CartPole/Pong/GridWorld examples.
-/

@[expose] public section

namespace TorchLean
namespace rl
namespace cli

/-- Parsed PPO-style training flags shared by multiple runnable examples. -/
structure PpoFlags where
  updates : Nat
  evalEvery : Nat
  evalEpisodes : Nat
  evalMaxSteps : Nat
  log : _root_.Runtime.Training.LogDestination
  logPath : System.FilePath
deriving Repr

/--
Parse PPO-style shared flags.

Notes:
- `--log off|none|false` disables writing the JSON artifact but still returns the resolved default
  `logPath` (useful for printing consistent banners).
- We treat `0` as invalid for the update/eval counts because a “no-op” run usually indicates a CLI
  mistake.
-/
def parsePpoFlags (exeName : String) (args : List String)
    (defaultLogPath : System.FilePath)
    (defaultUpdates defaultEvalEvery defaultEvalEpisodes defaultEvalMaxSteps : Nat) :
    Except String (PpoFlags × List String) := do
  let (logRaw?, args) ← TorchLean.CLI.takeFlagValueOnce args "log"
  let (updates, args) ←
    TorchLean.CLI.takePositiveNatFlag args exeName "updates" defaultUpdates
  let (evalEvery, args) ←
    TorchLean.CLI.takePositiveNatFlag args exeName "eval-every" defaultEvalEvery
  let (evalEpisodes, args) ←
    TorchLean.CLI.takePositiveNatFlag args exeName "eval-episodes" defaultEvalEpisodes
  let (evalMaxSteps, args) ←
    TorchLean.CLI.takePositiveNatFlag args exeName "eval-max-steps" defaultEvalMaxSteps

  let log := _root_.Runtime.Training.LogDestination.parse? defaultLogPath logRaw?
  pure ({ updates := updates
          evalEvery := evalEvery
          evalEpisodes := evalEpisodes
          evalMaxSteps := evalMaxSteps
          log := log
          logPath := log.pathD defaultLogPath }, args)

end cli
end rl
end TorchLean
