/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Shape
public import NN.API.Module
public import NN.API.Trainer.Manual.Core
public import NN.API.Optim
public import NN.API.Loss
public import NN.API.Trainer.Reporting
public import NN.API.Trainer.Summary
public import NN.API.Neural.Summary

/-!
# Trainer

Create a trainer from a checked model, choose its loss and optimizer, then train or predict:

```lean
let trainer := Trainer.new model
  { task := .regression
    optimizer := optim.adam { lr := 0.03 } }
let y0 ← trainer.predict x
let trained ← trainer.train data { steps := 200, batchSize := 16, logEvery := 25 }
trained.printSummary
```

`Trainer.Manual` provides direct access to runtime modules, tensor packs, and callbacks for custom
loops.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

universe u

/-- Runtime, device, scalar, and optimizer settings for a trainer or one training call. -/
structure RunConfig where
  /-- Optimizer used unless a training call supplies another run configuration. -/
  optimizer : optim.Optimizer := optim.sgd { lr := 0.01 }
  /-- Scalar semantics used for the run. -/
  scalar : Runtime.ScalarMode := .float32
  /-- Immediate tape execution or reusable typed-graph execution. -/
  execution : Runtime.ExecutionMode := .eager
  /-- Device used for execution. -/
  device : Runtime.Device := .cpu
  /-- Optional advanced override for provider, assurance, and VJP policy. -/
  backendProfile? : Option _root_.NN.Backend.BackendProfile := none
  /-- Print each accepted backend capsule when it is first used. -/
  showBackend : Bool := false

/--
Loss used to train a model.

The output shape belongs to the model. The task only decides how `(prediction, target)` becomes a
scalar objective; it does not need a separate input-shape index.
-/
inductive Task (outputShape : List Nat) where
  /-- Mean-squared-error supervised regression. -/
  | regression (reduction : Loss.Reduction := .mean)
  /-- One-hot cross entropy over a class or structured logit tensor. -/
  | oneHotCrossEntropy (axis : Nat)
      [axisInBounds : _root_.Spec.Shape.AxisInBounds axis (Shape.ofList outputShape)]
      (reduction : Loss.Reduction := .mean)
  /-- A checked TorchLean loss program supplied by the caller. -/
  | custom
      (loss : ∀ {α : Type}, [_root_.Context α] → [DecidableEq Shape] →
        _root_.Runtime.Autograd.TorchLean.Program α
          [Shape.ofList outputShape, Shape.ofList outputShape] ([] : List Nat))

/-- Model-independent options accepted by `Trainer.new`. -/
structure Config (inputShape outputShape : List Nat) extends RunConfig where
  /-- Task loss attached to this trainer. -/
  task : Task outputShape := .regression
  /-- Seed used when the model is still a seedable `TorchLean.nn.Builder` builder. -/
  seed : Nat := 0

end Trainer

/--
A checked model together with its loss, runtime settings, and initialization seed.

Construct trainers with `Trainer.new`. The resulting value supports prediction, training, and model
inspection directly through dot notation.
-/
structure Trainer (inputShape outputShape : List Nat) where
  /-- Checked TorchLean model. -/
  model : TorchLean.nn.Sequential inputShape outputShape
  /-- Supervised objective used by `train`. -/
  task : Trainer.Task outputShape
  /-- Runtime, backend-contract, and optimizer choices carried by this trainer. -/
  runtime : Trainer.RunConfig := {}
  /-- Seed used to build this trainer when the input was a `TorchLean.nn.Builder` model builder. -/
  seed : Nat := 0

namespace Trainer

/-- Checked model summary for this trainer. -/
def info {inputShape outputShape : List Nat}
    (trainer : TorchLean.Trainer inputShape outputShape) : String :=
  nn.info trainer.model

/-- Print the checked model summary under the heading `model`. -/
def printInfo {inputShape outputShape : List Nat}
    (trainer : TorchLean.Trainer inputShape outputShape) : IO Unit := do
  IO.println "model:"
  IO.println trainer.info

/-- Print the checked model summary with a caller-chosen heading. -/
def printInfoAs {inputShape outputShape : List Nat}
    (trainer : TorchLean.Trainer inputShape outputShape) (label : String) : IO Unit := do
  IO.println s!"{label}:"
  IO.println trainer.info

namespace Internal

/--
Internal selection of a checked model, its concrete runtime task, and its state-layout equality.

Regression and one-hot cross entropy differ only in how they construct `task`; the runtime runner,
state, and prediction machinery is otherwise identical.
-/
structure SelectedTask (inputShape outputShape : List Nat) where
  /-- The checked model selected for execution. -/
  model : TorchLean.nn.Sequential inputShape outputShape
  /-- Concrete supervised task passed to the manual runner. -/
  task : TorchLean.Trainer.Manual.SeqTask
    (Shape.ofList inputShape) (Shape.ofList outputShape)
  /-- Runtime settings inherited from the public trainer. -/
  runtime : RunConfig := {}
  /-- The selected task uses exactly the model's parameter and buffer layout. -/
  stateShapes_eq : task.stateShapes = nn.stateShapes model

namespace SelectedTask

/-- Select mean-squared-error training from a public trainer. -/
def regression {inputShape outputShape : List Nat}
    (trainer : TorchLean.Trainer inputShape outputShape) (reduction : Loss.Reduction) :
    SelectedTask inputShape outputShape :=
  { model := trainer.model
    task := TorchLean.Trainer.Manual.SeqTask.mse trainer.model reduction
    runtime := trainer.runtime
    stateShapes_eq := by rfl }

/-- Select one-hot cross-entropy training from a public trainer. -/
def oneHotCrossEntropy {inputShape outputShape : List Nat}
    (trainer : TorchLean.Trainer inputShape outputShape) (axis : Nat)
    [validAxis : _root_.Spec.Shape.AxisInBounds axis (Shape.ofList outputShape)]
    (reduction : Loss.Reduction) : SelectedTask inputShape outputShape :=
  { model := trainer.model
    task := TorchLean.Trainer.Manual.SeqTask.oneHotCrossEntropy trainer.model axis reduction
    runtime := trainer.runtime
    stateShapes_eq := by rfl }

end SelectedTask

end Internal

end Trainer

end TorchLean
