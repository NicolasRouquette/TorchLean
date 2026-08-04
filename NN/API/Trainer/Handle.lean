/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Module
public import NN.API.Trainer.Manual.Core
public import NN.API.Optim
public import NN.API.Loss
public import NN.API.Trainer.Reporting
public import NN.API.Trainer.Manual
public import NN.API.Trainer.Summary

/-!
# Trainer Handle

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
  /-- Scalar dtype used for the run. -/
  dtype : Runtime.DType := .float
  /-- Eager or compiled execution backend. -/
  backend : Runtime.Backend := .eager
  /-- Contract profile controlling device, providers, assurance, and VJP ownership. -/
  executionProfile : _root_.NN.Backend.BackendProfile :=
    _root_.NN.Backend.BackendProfile.checkedCpu
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
  /-- One-hot cross entropy over the model output tensor. -/
  | classification (reduction : Loss.Reduction := .mean)
  /-- One-hot cross entropy for sequence or structured logit tensors. -/
  | crossEntropy (reduction : Loss.Reduction := .mean)
  /-- A checked TorchLean loss program supplied by the caller. -/
  | custom
      (loss : ∀ {α : Type}, [_root_.Context α] → [DecidableEq Shape] →
        _root_.Runtime.Autograd.TorchLean.Program α [τ, τ] Shape.scalar)

/-- Model-independent options accepted by `Trainer.new`. -/
structure Config (σ τ : Shape) extends RuntimeSettings where
  /-- Task loss attached to this trainer. -/
  task : Task σ τ := .regression
  /-- Seed used when the model is still a seedable `TorchLean.nn.M` builder. -/
  seed : Nat := 0

/--
A checked model together with its loss, runtime settings, and initialization seed.
-/
structure Handle (σ τ : Shape) where
  /-- Checked TorchLean model. -/
  model : TorchLean.nn.Sequential σ τ
  /-- Supervised objective used by `train`. -/
  task : Task σ τ
  /-- Runtime/backend/optimizer choices carried by this trainer. -/
  runtime : RuntimeSettings := {}
  /-- Seed used to build this trainer when the input was an `TorchLean.nn.M` model builder. -/
  seed : Nat := 0

namespace Handle

/-- Checked model summary for this unified trainer. -/
def info {σ τ : Shape} (trainer : Handle σ τ) : String :=
  nn.info trainer.model

/-- Print the checked model summary under the heading `model`. -/
def printInfo {σ τ : Shape} (trainer : Handle σ τ) : IO Unit := do
  IO.println "model:"
  IO.println trainer.info

/-- Print the checked model summary with a caller-chosen heading. -/
def printInfoAs {σ τ : Shape} (trainer : Handle σ τ) (label : String) : IO Unit := do
  IO.println s!"{label}:"
  IO.println trainer.info

end Handle

namespace Implementation

/--
Typed dispatch record for supervised regression.

This record carries the equalities needed by the regression implementation. Users construct a
`Trainer.Handle` rather than this task-specific record.
-/
structure Regression (σ τ : Shape) where
  /-- The checked TorchLean model used by this trainer. -/
  model : TorchLean.nn.Sequential σ τ
  /-- Mean vs sum loss reduction for the built regression task. -/
  reduction : Loss.Reduction := .mean
  /-- Runtime/backend/optimizer choices carried by this trainer. -/
  runtime : RuntimeSettings := {}

/--
Typed dispatch record for general one-hot cross-entropy training.

Classification and sequence models use the same checked one-hot cross-entropy runtime path. Image
examples usually batch their dataset first with `Data.batchDataset`; text examples often train on a
whole matrix of one-hot next-token rows:

```lean
let trainer := Trainer.new model { task := .crossEntropy, optimizer := optim.adam { lr := 1e-3 } }
let trained ← trainer.train tokenWindows { steps := 200 }
```

The output handle exposes prediction tensors, not class labels, because token decoding is
model-specific and belongs in the text example.
-/
structure CrossEntropy (σ τ : Shape) where
  /-- The checked TorchLean model used by this trainer. -/
  model : TorchLean.nn.Sequential σ τ
  /-- Mean vs sum loss reduction for the one-hot cross-entropy task. -/
  reduction : Loss.Reduction := .mean
  /-- Runtime/backend/optimizer choices carried by this trainer. -/
  runtime : RuntimeSettings := {}

/--
Typed dispatch record for a checked custom scalar loss.

Custom losses cover masked language-model objectives, physics residuals, and algorithmic tasks
where the model is still an ordinary `TorchLean.nn.Sequential`, but the loss has task logic that does not fit
a canned reduction. The boundary stays precise: the loss is a TorchLean program over
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

/-- The runtime task induced by this handle has exactly the model's parameter shapes. -/
theorem taskParamShapes_eq {σ τ : Shape} (trainer : Regression σ τ) :
    TorchLean.Trainer.Manual.paramShapes trainer.task = nn.paramShapes trainer.model := by
  cases trainer with
  | mk model reduction =>
      rfl

end Regression

namespace CrossEntropy

/-- Checked TorchLean task induced by this cross-entropy dispatch record. -/
def task {σ τ : Shape} (trainer : CrossEntropy σ τ) : TorchLean.Trainer.Manual.SeqTask σ τ :=
  TorchLean.Trainer.Manual.SeqTask.crossEntropyOneHot trainer.model trainer.reduction

/-- Checked model summary for this trainer. -/
def info {σ τ : Shape} (trainer : CrossEntropy σ τ) : String :=
  nn.info trainer.model

/-- Print the checked model summary with the standard public-example heading. -/
def printInfo {σ τ : Shape} (trainer : CrossEntropy σ τ) : IO Unit := do
  IO.println "model:"
  IO.println trainer.info

/-- The runtime task induced by this handle has exactly the model's parameter shapes. -/
theorem taskParamShapes_eq {σ τ : Shape} (trainer : CrossEntropy σ τ) :
    TorchLean.Trainer.Manual.paramShapes trainer.task = nn.paramShapes trainer.model := by
  cases trainer with
  | mk model reduction =>
      rfl

end CrossEntropy

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
