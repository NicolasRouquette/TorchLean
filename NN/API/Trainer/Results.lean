/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Trainer.Core
public import NN.API.Trainer.Manual.Evaluation

/-!
# Training Results

Regression, cross-entropy, and custom losses all return the same trained-model type.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

/--
A trained TorchLean model.

The result retains the live runtime state through its prediction closures. Import
`NN.API.Verification.Trainer` and use `Trainer.trainVerified` when the trained parameters should
also be retained for IBP verification.
-/
structure TrainResult (inputShape outputShape : List Nat) where
  /-- Before/after scalar summary for the completed run. -/
  report : TrainSummary
  /-- Run one `Float` input through the trained model. -/
  predict : Tensor Float inputShape → IO (Tensor Float outputShape)
  /-- Run several `Float` inputs through the trained model. -/
  predictMany : Array (Tensor Float inputShape) → IO (Array (Tensor Float outputShape))

namespace Internal
namespace SelectedTask

/-- Package an already-trained runner as the public prediction and reporting result. -/
def toTrainResult {inputShape outputShape : List Nat} {α : Type}
    [_root_.Context α] [DecidableEq Shape] [ToString α] [Runtime.FromFloat α]
    [Runtime.TensorTransfer α]
    (selected : SelectedTask inputShape outputShape)
    (runner : Manual.Runner α selected.task) (steps : Nat) (before after : α) :
    TrainResult inputShape outputShape :=
  let predict := fun (xFloat : Tensor Float inputShape) => do
    Manual.Runner.eval (task := selected.task) runner
    let x := Tensor.map (Runtime.ofFloat (α := α)) xFloat
    let yhat ← Manual.Runner.run (task := selected.task) runner x
    Runtime.toFloatTensor yhat
  { report := { steps := steps, before := toString before, after := toString after }
    predict := predict
    predictMany := fun xs => xs.mapM predict }

end SelectedTask
end Internal

namespace TrainResult

/-- One-line summary for the completed training run. -/
def summary {inputShape outputShape : List Nat}
    (result : TrainResult inputShape outputShape) : String :=
  result.report.summary

/-- Print the before/after training summary. -/
def printSummary {inputShape outputShape : List Nat}
    (result : TrainResult inputShape outputShape) : IO Unit :=
  IO.println result.summary

/-- Print one prediction with a caller-supplied label. -/
def printPrediction {inputShape outputShape : List Nat}
    (result : TrainResult inputShape outputShape) (label : String)
    (x : Tensor Float inputShape) : IO Unit := do
  let yhat ← result.predict x
  IO.println s!"{label} = {Tensor.pretty yhat}"

instance {inputShape outputShape : List Nat} :
    ToString (TrainResult inputShape outputShape) where
  toString := summary

end TrainResult

/--
A trained model returned by step-indexed stream training.

Generated or resampled workloads may not have one static dataset to summarize. The ordinary training
result is paired with the evaluation curve collected from a caller-provided sample.
-/
structure StreamTrainResult (inputShape outputShape : List Nat) where
  /-- Trained model result. -/
  result : TrainResult inputShape outputShape
  /-- Evaluation loss curve recorded during stream training. -/
  curve : Training.Curve

namespace StreamTrainResult

/-- One-line summary for the trained stream run. -/
def summary {inputShape outputShape : List Nat}
    (result : StreamTrainResult inputShape outputShape) : String :=
  result.result.summary

/-- Print the stream training summary. -/
def printSummary {inputShape outputShape : List Nat}
    (result : StreamTrainResult inputShape outputShape) : IO Unit :=
  IO.println result.summary

/-- Run one prediction through the trained stream result. -/
def predict {inputShape outputShape : List Nat}
    (result : StreamTrainResult inputShape outputShape) (x : Tensor Float inputShape) :
    IO (Tensor Float outputShape) :=
  result.result.predict x

/-- Run several predictions through the trained stream result. -/
def predictMany {inputShape outputShape : List Nat}
    (result : StreamTrainResult inputShape outputShape)
    (xs : Array (Tensor Float inputShape)) : IO (Array (Tensor Float outputShape)) :=
  result.result.predictMany xs

instance {inputShape outputShape : List Nat} :
    ToString (StreamTrainResult inputShape outputShape) where
  toString := summary

end StreamTrainResult

/-- Two trained regression models and the coupled metric recorded by an alternating stream. -/
structure PairStreamTrainResult
    (inputShape₁ outputShape₁ inputShape₂ outputShape₂ : List Nat) where
  /-- Trained result for the first model. -/
  first : TrainResult inputShape₁ outputShape₁
  /-- Trained result for the second model. -/
  second : TrainResult inputShape₂ outputShape₂
  /-- Task-specific curve recorded by the caller-provided evaluation function. -/
  curve : Training.Curve

namespace PairStreamTrainResult

/-- One-line summary for the two trained models. -/
def summary {inputShape₁ outputShape₁ inputShape₂ outputShape₂ : List Nat}
    (result : PairStreamTrainResult inputShape₁ outputShape₁ inputShape₂ outputShape₂) : String :=
  s!"first: {result.first.summary}; second: {result.second.summary}"

/-- Print the training summary for both models. -/
def printSummary {inputShape₁ outputShape₁ inputShape₂ outputShape₂ : List Nat}
    (result : PairStreamTrainResult inputShape₁ outputShape₁ inputShape₂ outputShape₂) : IO Unit :=
  IO.println result.summary

/-- Print the endpoints of the coupled metric curve. -/
def printCurveSummary {inputShape₁ outputShape₁ inputShape₂ outputShape₂ : List Nat}
    (result : PairStreamTrainResult inputShape₁ outputShape₁ inputShape₂ outputShape₂)
    (metric : String := "loss") : IO Unit := do
  let endpoints ←
    Training.Curve.endpoints result.curve "PairStreamTrainResult.printCurveSummary"
  let firstMetric := s!"{metric}0={endpoints.first}"
  let lastMetric := s!"{metric}{endpoints.finalStep}={endpoints.last}"
  IO.println s!"  steps={endpoints.finalStep} {firstMetric} {lastMetric}"

instance {inputShape₁ outputShape₁ inputShape₂ outputShape₂ : List Nat} :
    ToString (PairStreamTrainResult inputShape₁ outputShape₁ inputShape₂ outputShape₂) where
  toString := summary

end PairStreamTrainResult

end Trainer

end TorchLean
