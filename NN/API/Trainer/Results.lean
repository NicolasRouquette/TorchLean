/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Trainer.Handle

/-!
# Training Results

Regression, cross-entropy, and custom losses all return the same trained-model type. Capabilities
that do not apply to every task, such as the current IBP verifier, are stored as optional
operations.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

/-- Result of checking an output box with the trained-model IBP verifier. -/
structure VerificationReport where
  /-- Number of IR nodes in the compiled verifier graph. -/
  nodes : Nat
  /-- Flattened output dimension reported by the verifier. -/
  outputDim : Nat
  /-- Lower bound on the flattened output box. -/
  lo : String
  /-- Upper bound on the flattened output box. -/
  hi : String

namespace VerificationReport

/-- One-line verification summary. -/
def summary (report : VerificationReport) : String :=
  s!"IBP nodes={report.nodes} output_dim={report.outputDim} lo={report.lo} hi={report.hi}"

/-- Print the verification summary. -/
def printSummary (report : VerificationReport) : IO Unit :=
  IO.println report.summary

instance : ToString VerificationReport where
  toString := summary

end VerificationReport

/--
A trained TorchLean model.

The handle owns the live runtime state through its closures. `predict` and `predictBatch` are
available for every task. `verifyRobustLInf?` is present when the training path can compile the
trained parameters to the checked IBP verifier.
-/
structure TrainResult (σ τ : Shape) where
  /-- Before/after scalar summary for the completed run. -/
  report : TrainSummary
  /-- Run one `Float` input through the trained model. -/
  predict : Tensor.T Float σ → IO (Tensor.T Float τ)
  /-- Run several `Float` inputs through the trained model. -/
  predictBatch : List (Tensor.T Float σ) → IO (List (Tensor.T Float τ))
  /-- Optional verifier for a uniform $\ell_\infty$ input ball. -/
  verifyRobustLInf? :
    Option (Tensor.T Float σ → Float → IO VerificationReport) := none

namespace TrainResult

/-- One-line summary for the completed training run. -/
def summary {σ τ : Shape} (result : TrainResult σ τ) : String :=
  result.report.summary

/-- Print the before/after training summary. -/
def printSummary {σ τ : Shape} (result : TrainResult σ τ) : IO Unit :=
  IO.println result.summary

/-- Print one prediction with a caller-supplied label. -/
def printPrediction {σ τ : Shape}
    (result : TrainResult σ τ) (label : String) (x : Tensor.T Float σ) : IO Unit := do
  let yhat ← result.predict x
  IO.println s!"{label} = {Tensor.pretty yhat}"

/--
Verify a uniform $\ell_\infty$ ball around `center`.

The method fails explicitly when the training path did not attach the checked IBP capability.
-/
def verifyRobustLInf {σ τ : Shape}
    (result : TrainResult σ τ) (center : Tensor.T Float σ) (eps : Float) :
    IO VerificationReport :=
  match result.verifyRobustLInf? with
  | some verify => verify center eps
  | none =>
      throw <| IO.userError
        "Trainer.TrainResult.verifyRobustLInf: this trained result has no IBP verifier"

/-- Verify a uniform $\ell_\infty$ ball and print the resulting output interval. -/
def printRobustLInf {σ τ : Shape}
    (result : TrainResult σ τ) (center : Tensor.T Float σ) (eps : Float) : IO Unit := do
  let report ← result.verifyRobustLInf center eps
  report.printSummary

instance {σ τ : Shape} : ToString (TrainResult σ τ) where
  toString := summary

end TrainResult

/--
A trained model returned by step-indexed stream training.

Generated or resampled workloads may not have one static dataset to summarize. The ordinary trained
handle is paired with the evaluation curve collected from a caller-provided sample.
-/
structure StreamTrainResult (σ τ : Shape) where
  /-- Trained model handle. -/
  result : TrainResult σ τ
  /-- Evaluation loss curve recorded during stream training. -/
  curve : Training.Curve

namespace StreamTrainResult

/-- One-line summary for the trained stream run. -/
def summary {σ τ : Shape} (result : StreamTrainResult σ τ) : String :=
  result.result.summary

/-- Print the stream training summary. -/
def printSummary {σ τ : Shape} (result : StreamTrainResult σ τ) : IO Unit :=
  IO.println result.summary

/-- Run one prediction through the trained stream result. -/
def predict {σ τ : Shape}
    (result : StreamTrainResult σ τ) (x : Tensor.T Float σ) : IO (Tensor.T Float τ) :=
  result.result.predict x

/-- Run several predictions through the trained stream result. -/
def predictBatch {σ τ : Shape}
    (result : StreamTrainResult σ τ) (xs : List (Tensor.T Float σ)) :
    IO (List (Tensor.T Float τ)) :=
  result.result.predictBatch xs

instance {σ τ : Shape} : ToString (StreamTrainResult σ τ) where
  toString := summary

end StreamTrainResult

/-- Two trained regression models and the coupled metric recorded by an alternating stream. -/
structure PairStreamTrainResult (σ₁ τ₁ σ₂ τ₂ : Shape) where
  /-- Trained handle for the first model. -/
  first : TrainResult σ₁ τ₁
  /-- Trained handle for the second model. -/
  second : TrainResult σ₂ τ₂
  /-- Task-specific curve recorded by the caller-provided evaluation function. -/
  curve : Training.Curve

namespace PairStreamTrainResult

/-- One-line summary for the two trained models. -/
def summary {σ₁ τ₁ σ₂ τ₂ : Shape} (result : PairStreamTrainResult σ₁ τ₁ σ₂ τ₂) :
    String :=
  s!"first: {result.first.summary}; second: {result.second.summary}"

/-- Print the trained-handle summary for both models. -/
def printSummary {σ₁ τ₁ σ₂ τ₂ : Shape} (result : PairStreamTrainResult σ₁ τ₁ σ₂ τ₂) :
    IO Unit :=
  IO.println result.summary

/-- Print the endpoints of the coupled metric curve. -/
def printCurveSummary {σ₁ τ₁ σ₂ τ₂ : Shape}
    (result : PairStreamTrainResult σ₁ τ₁ σ₂ τ₂)
    (metric : String := "loss") : IO Unit := do
  let endpoints ←
    Training.Curve.endpoints result.curve "PairStreamTrainResult.printCurveSummary"
  IO.println
    s!"  steps={endpoints.finalStep} {metric}0={endpoints.first} {metric}{endpoints.finalStep}={endpoints.last}"

instance {σ₁ τ₁ σ₂ τ₂ : Shape} : ToString (PairStreamTrainResult σ₁ τ₁ σ₂ τ₂) where
  toString := summary

end PairStreamTrainResult

end Trainer

end TorchLean
