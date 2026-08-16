/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.Examples.Quickstart.Common

/-!
# Simple MLP training example (regression)

This is a focused end-to-end example of training a small MLP in TorchLean.

It mirrors the simplest PyTorch workflow:

1. build a small synthetic dataset (in-memory),
2. define an MLP (`Linear -> ReLU -> Linear`),
3. train with Adam,
4. report loss before/after, plus a few sample predictions.

Run:

- `lake exe torchlean quickstart_mlp`
- `lake exe torchlean quickstart_mlp --steps 200 --scalar ieee32-exec --execution eager`
- `lake exe torchlean quickstart_mlp --steps 200 --scalar float32 --execution eager`

Optional flags (tutorial-specific):

- `--seed S` (model init + any shuffling)
- `--steps N`
-/

@[expose] public section


namespace NN.Examples.Quickstart.SimpleMLPTrain

open TorchLean

/-- Default JSON log path used only when the user explicitly passes `--log`. -/
def defaultLogJson : System.FilePath := ModelZoo.trainLogPath "quickstart_simple_mlp"

def inDim : Nat := 2
def outDim : Nat := 1

/-- A small 2-layer MLP `2 -> 8 -> 1`. -/
def model : nn.Builder (nn.Sequential (.dim inDim .scalar) (.dim outDim .scalar)) :=
  nn.Sequential![
    nn.linear inDim 8,
    nn.relu,
    nn.linear 8 outDim
  ]

/--
Small piecewise-linear regression target:

$$
y=0.8\,\operatorname{ReLU}(x_1+x_2)
  -0.4\,\operatorname{ReLU}(x_2-x_1)+0.2.
$$

This is a natural fit for a small ReLU MLP, which keeps the command dependable.
-/
def target (x1 x2 : Float) : Float :=
  let relu (x : Float) := if x < 0.0 then 0.0 else x
  (0.8 * relu (x1 + x2)) - (0.4 * relu (x2 - x1)) + 0.2

/--
Build the tutorial dataset at the runtime-selected scalar type.

`Data.regressionGrid` keeps shape-indexed tensor slicing out of the first training example.
The underlying value is still a TorchLean supervised dataset with checked input/output shapes.
-/
def buildDataset : Trainer.DataSource (.dim inDim .scalar) (.dim outDim .scalar) :=
  Data.regressionGrid (-1.0) 1.0 5 target

/-- Command-line help for the simple MLP quickstart. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean simple MLP quickstart"
    , ""
    , "Usage:"
    , "  lake exe torchlean quickstart_mlp [options]"
    , ""
    , "Options:"
    , "  --seed N"
    , "  --steps N"
    , "  --scalar float32|ieee32-exec|complex64"
    , "  --execution eager|typed-graph"
    , "  --device auto|cpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external"
    , "  --show-backend                    print backend capsules as they execute"
    , "  --log PATH"
    , ""
    , "Top-level chooser:"
    , "  lake exe torchlean --choose quickstart_mlp --steps 20"
    ]

def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return
  let (seed, args) ← CLI.seed "SimpleMLPTrain" args
  let parsed ←
    _root_.NN.Examples.Quickstart.parseRuntimeTrain
      "SimpleMLPTrain" args defaultLogJson 200 (optim.adam { lr := 0.03 })
      (logEvery := 25)
  let trainer := Trainer.new model <|
    Trainer.Config.fromRunConfig parsed.run .regression (seed := seed)

  IO.println "== Quickstart: simple MLP training =="
  IO.println s!"seed  = {seed}"
  IO.println s!"steps = {parsed.train.steps}"

  let probes := [
    Trainer.Probe.point "center" 0.0 0.0 (some (toString (target 0.0 0.0))),
    Trainer.Probe.point "heldout" 0.25 (-0.75) (some (toString (target 0.25 (-0.75))))
  ]
  let trained ← trainer.train buildDataset parsed.trainOptions probes
  trained.printSummary
  let heldout : Tensor Float (.dim inDim .scalar) := tensorOfList! [2] [0.25, -0.75]
  trained.printPrediction "predict(heldout)" heldout

end NN.Examples.Quickstart.SimpleMLPTrain
