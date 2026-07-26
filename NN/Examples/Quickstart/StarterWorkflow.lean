/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API

/-!
# Quickstart: Starter Workflow

The smallest useful TorchLean training setup is ordinary model code:

```lean
public import NN.API
open TorchLean

def model :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]
```

No subsystem-specific imports are needed here. Model construction, data, training, prediction, and
the public robustness-checking entry point all come from `import NN.API`. Lower-level certificate
formats and proof developments use the focused `NN.Verification` and `NN.Proofs` imports.
-/

@[expose] public section

namespace NN.Examples.Quickstart.StarterWorkflow

open TorchLean

def model :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]

/-- Checks that KAN constructors are available from `NN.API`. -/
def kanModel : nn.M (nn.Sequential (.dim 4 (.dim 2 .scalar)) (.dim 4 (.dim 1 .scalar))) :=
  nn.models.KAN
    { batch := 4
      inDim := 2
      hidden := [8]
      outDim := 1 }

def target (x1 x2 : Float) : Float :=
  let relu (x : Float) := if x < 0.0 then 0.0 else x
  relu (x1 + x2) + 0.25

/--
Tiny in-memory regression dataset.

The important bit is the last line: `Data.tensorDataset xs ys` turns ordinary `Float` tensors into a
runtime-polymorphic dataset, so the trainer can still choose `Float`, executable IEEE32, CPU, CUDA,
eager, or compiled execution later.
-/
def data : Trainer.Dataset (.dim 2 .scalar) (.dim 1 .scalar) :=
  let xs : Tensor.T Float (shape![4, 2]) :=
    tensorOfList! [4, 2] [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
  let ys : Tensor.T Float (shape![4, 1]) :=
    tensorOfList! [4, 1] [target 0.0 0.0, target 0.0 1.0, target 1.0 0.0, target 1.0 1.0]
  Data.tensorDataset xs ys

def probes : List (Trainer.Probe (.dim 2 .scalar)) :=
  [ Trainer.Probe.point "origin" 0.0 0.0 (some (toString (target 0.0 0.0)))
  , Trainer.Probe.point "heldout" 0.5 (-0.25) (some (toString (target 0.5 (-0.25)))) ]

/-- Select an optimizer through the public API. -/
def optimizerChoiceExample : Except String (String × optim.Optimizer) := do
  let kind ← optim.Kind.parse "adamw"
  pure (kind.name, kind.toOptimizer 0.01)

/-- A LoRA parameter type exposed by the public adapter API. -/
def loraParamsExample : Type :=
  Adapters.LoRA.Params Float 2 1 1

/--
Run the public API example from another command or from `#eval` while developing.

The shape below is the user-facing training path:

- build the trainer from the model,
- attach optimizer/backend choices once,
- call `trainer.predict` for initial prediction,
- call `trainer.train`,
- use the returned trained handle for prediction.
- call `trained.verifyRobustLInf` on a small $\ell_\infty$ box.

The quickstart build only checks that these declarations typecheck; it does not train during
ordinary `lake build`, which keeps CI fast.
-/
def run (_args : List String := []) : IO Unit := do
  let trainer :=
    Trainer.new model
      { task := .regression
        optimizer := optim.adam { lr := 0.03 }
        backend := .compiled
        dtype := .float32 }
  let heldout : Tensor.T Float (.dim 2 .scalar) := tensorOfList! [2] [0.5, -0.25]
  let initial ← trainer.predict heldout
  IO.println s!"initial(heldout) = {Tensor.pretty initial}"
  let trained ← trainer.train data { steps := 25, batchSize := 4, logEvery := 10 } probes
  trained.printSummary
  trained.printPrediction "predict(heldout)" heldout
  let cert ← trained.verifyRobustLInf heldout 0.05
  cert.printSummary

end NN.Examples.Quickstart.StarterWorkflow
