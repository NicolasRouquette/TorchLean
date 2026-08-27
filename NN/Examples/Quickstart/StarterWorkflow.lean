/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.API.Verification

/-!
# Quickstart: Starter Workflow

The smallest useful TorchLean training setup is ordinary model code:

```lean
import NN.API
open TorchLean

def model :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]
```

Model construction, data, training, and prediction come from `NN.API`. This example also imports
`NN.API.Verification` because it retains an IBP verifier after training. Lower-level certificate
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

def target (x1 x2 : Float) : Float :=
  let relu (x : Float) := if x < 0.0 then 0.0 else x
  relu (x1 + x2) + 0.25

/--
Tiny in-memory regression dataset.

`Data.tensorDataset` infers the input and target shapes from the two tensors. The trainer remains
free to choose the scalar semantics and execution device later.
-/
def data :=
  let xs : Tensor Float [4, 2] :=
    tensor! [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
  let ys : Tensor Float [4, 1] :=
    tensor! [[target 0.0 0.0], [target 0.0 1.0], [target 1.0 0.0], [target 1.0 1.0]]
  Data.tensorDataset xs ys

def probes : Array (Trainer.Probe [2]) :=
  #[ Trainer.Probe.ofFloatTensor "origin" (tensor! (ty := Float) [0.0, 0.0])
      "x=(0.0,0.0)" (some (toString (target 0.0 0.0)))
  , Trainer.Probe.ofFloatTensor "heldout" (tensor! (ty := Float) [0.5, -0.25])
      "x=(0.5,-0.25)" (some (toString (target 0.5 (-0.25)))) ]

/--
Run the public API example from another command or from `#eval` while developing.

The shape below is the user-facing training path:

- build the trainer from the model,
- attach optimizer, execution, and device choices once,
- call `trainer.predict` for initial prediction,
- call `trainer.trainVerified`,
- use the returned verified training result for prediction,
- call `trained.verifyRobustLInf` on a small $\ell_\infty$ box.

The quickstart build only checks that these declarations typecheck; it does not train during
ordinary `lake build`, which keeps CI fast.
-/
def run (_args : List String := []) : IO Unit := do
  let trainer :=
    Trainer.new model
      { task := .regression
        optimizer := optim.adam { lr := 0.03 }
        execution := .typedGraph
        device := .cpu
        scalar := .ieee32Exec }
  let heldout : Tensor Float [2] := tensor! [0.5, -0.25]
  let initial ← trainer.predict heldout
  IO.println s!"initial(heldout) = {Tensor.pretty initial}"
  let trained ← trainer.trainVerified data { steps := 25, batchSize := 4, logEvery := 10 } probes
  trained.printSummary
  trained.printPrediction "predict(heldout)" heldout
  let cert ← trained.verifyRobustLInf heldout 0.05
  cert.printSummary

end NN.Examples.Quickstart.StarterWorkflow
