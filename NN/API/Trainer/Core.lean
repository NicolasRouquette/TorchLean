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
public import NN.API.Trainer.Manual
public import NN.API.Trainer.Summary

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

/-- Runtime settings stored with a trainer. -/
structure RuntimeSettings where
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

The shape parameters belong to the model. The task only decides how `(prediction, target)` becomes
a scalar objective.
-/
inductive Task (σ τ : Shape) where
  /-- Mean-squared-error supervised regression. -/
  | regression (reduction : Loss.Reduction := .mean)
  /-- One-hot cross entropy over a class or structured logit tensor. -/
  | oneHotCrossEntropy (axis : Nat) [axisInBounds : _root_.Spec.Shape.AxisInBounds axis τ]
      (reduction : Loss.Reduction := .mean)
  /-- A checked TorchLean loss program supplied by the caller. -/
  | custom
      (loss : ∀ {α : Type}, [_root_.Context α] → [DecidableEq Shape] →
        _root_.Runtime.Autograd.TorchLean.Program α [τ, τ] Shape.scalar)

/-- Model-independent options accepted by `Trainer.new`. -/
structure Config (σ τ : Shape) extends RuntimeSettings where
  /-- Task loss attached to this trainer. -/
  task : Task σ τ := .regression
  /-- Seed used when the model is still a seedable `TorchLean.nn.Builder` builder. -/
  seed : Nat := 0

end Trainer

/--
A checked model together with its loss, runtime settings, and initialization seed.

Construct trainers with `Trainer.new`. The resulting value supports prediction, training, and model
inspection directly through dot notation.
-/
structure Trainer (σ τ : Shape) where
  /-- Checked TorchLean model. -/
  model : TorchLean.nn.Sequential σ τ
  /-- Supervised objective used by `train`. -/
  task : Trainer.Task σ τ
  /-- Runtime, backend-contract, and optimizer choices carried by this trainer. -/
  runtime : Trainer.RuntimeSettings := {}
  /-- Seed used to build this trainer when the input was a `TorchLean.nn.Builder` model builder. -/
  seed : Nat := 0

namespace Trainer

/-- Checked model summary for this trainer. -/
def info {σ τ : Shape} (trainer : TorchLean.Trainer σ τ) : String :=
  nn.info trainer.model

/-- Print the checked model summary under the heading `model`. -/
def printInfo {σ τ : Shape} (trainer : TorchLean.Trainer σ τ) : IO Unit := do
  IO.println "model:"
  IO.println trainer.info

/-- Print the checked model summary with a caller-chosen heading. -/
def printInfoAs {σ τ : Shape} (trainer : TorchLean.Trainer σ τ) (label : String) : IO Unit := do
  IO.println s!"{label}:"
  IO.println trainer.info

namespace Implementation

/--
Typed dispatch record for supervised regression.

This record carries the equalities needed by the regression implementation. Users construct a
`TorchLean.Trainer` rather than this task-specific record.
-/
structure Regression (σ τ : Shape) where
  /-- The checked TorchLean model used by this trainer. -/
  model : TorchLean.nn.Sequential σ τ
  /-- Mean vs sum loss reduction for the built regression task. -/
  reduction : Loss.Reduction := .mean
  /-- Runtime, backend-contract, and optimizer choices carried by this trainer. -/
  runtime : RuntimeSettings := {}

/--
Typed dispatch record for general one-hot cross-entropy training.

Classification and sequence models use the same checked one-hot cross-entropy runtime path. Image
examples usually batch their dataset first with `Data.batchDataset`; text examples often train on a
whole matrix of one-hot next-token rows:

```lean
let trainer := Trainer.new model
  { task := .oneHotCrossEntropy 1, optimizer := optim.adam { lr := 1e-3 } }
let trained ← trainer.train tokenWindows { steps := 200 }
```

The trained result exposes prediction tensors, not class labels, because token decoding is
model-specific and belongs in the text example.
-/
structure OneHotCrossEntropy (σ τ : Shape) where
  /-- The checked TorchLean model used by this trainer. -/
  model : TorchLean.nn.Sequential σ τ
  /-- Tensor dimension containing the class logits. -/
  axis : Nat
  /-- Evidence that `axis` names a dimension of the output tensor. -/
  [validAxis : _root_.Spec.Shape.AxisInBounds axis τ]
  /-- Mean vs sum loss reduction for the one-hot cross-entropy task. -/
  reduction : Loss.Reduction := .mean
  /-- Runtime, backend-contract, and optimizer choices carried by this trainer. -/
  runtime : RuntimeSettings := {}

/--
Typed dispatch record for a checked custom scalar loss.

Custom losses cover masked language-model objectives, physics residuals, and algorithmic tasks
where the model is still an ordinary `TorchLean.nn.Sequential`, but the loss has task logic that
does not fit a canned reduction. The boundary stays precise: the loss is a TorchLean program over
`(prediction, target)`, so module construction and optimizer wiring remain inside the trainer
API.
-/
structure Custom (σ τ : Shape) where
  /-- The checked TorchLean model used by this trainer. -/
  model : TorchLean.nn.Sequential σ τ
  /-- Checked scalar loss program applied to `(modelOutput, target)`. -/
  loss : ∀ {α : Type}, [_root_.Context α] → [DecidableEq Shape] →
    _root_.Runtime.Autograd.TorchLean.Program α [τ, τ] Shape.scalar
  /-- Runtime/backend/optimizer choices carried by this trainer. -/
  runtime : RuntimeSettings := {}

namespace Regression

/-- Checked TorchLean task induced by this regression dispatch record. -/
def task {σ τ : Shape} (trainer : Regression σ τ) : TorchLean.Trainer.Manual.SeqTask σ τ :=
  TorchLean.Trainer.Manual.SeqTask.mse trainer.model trainer.reduction

/-- Checked model summary for this trainer. -/
def info {σ τ : Shape} (trainer : Regression σ τ) : String :=
  nn.info trainer.model

/--
Print the checked model summary with the standard public-example heading.

Examples use this instead of open-coding `IO.println "model:"; IO.println trainer.info`, so the
first thing users see is consistent across regression, classifier, sequence, and custom trainers.
-/
def printInfo {σ τ : Shape} (trainer : Regression σ τ) : IO Unit := do
  IO.println "model:"
  IO.println trainer.info

/--
Print this trainer's checked model summary with a caller-chosen heading.

Most examples should use `trainer.printInfo`; paired-model examples such as GANs use this labeled
variant so both summaries still go through the same formatting path.
-/
def printInfoAs {σ τ : Shape} (trainer : Regression σ τ) (label : String) : IO Unit := do
  IO.println s!"{label}:"
  IO.println trainer.info

/-- The runtime task induced by this trainer has exactly the model's state layout. -/
theorem task_state_shapes_eq {σ τ : Shape} (trainer : Regression σ τ) :
    TorchLean.Trainer.Manual.stateShapes trainer.task = nn.stateShapes trainer.model := by
  cases trainer with
  | mk model reduction =>
      rfl

end Regression

namespace OneHotCrossEntropy

/-- Checked TorchLean task induced by this cross-entropy dispatch record. -/
def task {σ τ : Shape} (trainer : OneHotCrossEntropy σ τ) : TorchLean.Trainer.Manual.SeqTask σ τ :=
  letI := trainer.validAxis
  TorchLean.Trainer.Manual.SeqTask.oneHotCrossEntropy trainer.model trainer.axis trainer.reduction

/-- Checked model summary for this trainer. -/
def info {σ τ : Shape} (trainer : OneHotCrossEntropy σ τ) : String :=
  nn.info trainer.model

/-- Print the checked model summary with the standard public-example heading. -/
def printInfo {σ τ : Shape} (trainer : OneHotCrossEntropy σ τ) : IO Unit := do
  IO.println "model:"
  IO.println trainer.info

/-- The runtime task induced by this trainer has exactly the model's state layout. -/
theorem task_state_shapes_eq {σ τ : Shape} (trainer : OneHotCrossEntropy σ τ) :
    TorchLean.Trainer.Manual.stateShapes trainer.task = nn.stateShapes trainer.model := by
  cases trainer with
  | mk model axis reduction runtime =>
      rfl

end OneHotCrossEntropy

namespace Custom

/-- Checked model summary for this custom-loss trainer. -/
def info {σ τ : Shape} (trainer : Custom σ τ) : String :=
  nn.info trainer.model

/-- Print the checked model summary with the standard public-example heading. -/
def printInfo {σ τ : Shape} (trainer : Custom σ τ) : IO Unit := do
  IO.println "model:"
  IO.println trainer.info

end Custom

end Implementation

end Trainer

end TorchLean
