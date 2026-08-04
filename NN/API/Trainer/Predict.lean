/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Trainer.Train

/-!
# Prediction and Training

Prediction and training methods on `Trainer.Handle`.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

namespace Implementation

/-- Predict one Float input through a runtime runner and return a Float output. -/
def predictWithRunner {σ τ : Shape} {task : TorchLean.Trainer.Manual.SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Shape] [ToString α] [Runtime.FromFloat α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (runner : TorchLean.Trainer.Manual.Runner α task) (x : Tensor.T Float σ) :
    IO (Tensor.T Float τ) := do
  Manual.evalMode (task := task) runner
  let x' := Tensor.castFloat (Runtime.ofFloat (α := α)) x
  let y ← Manual.predict (task := task) runner x'
  Tensor.toFloatIO y

/-- Predict one input through a custom-loss trainer without first running training. -/
def predictCustomWithRunConfig {σ τ : Shape}
    (trainer : Custom σ τ) (run : RunConfig) (x : Tensor.T Float σ) :
    IO (Tensor.T Float τ) := do
  let opts := run.toOptions
  let runFor
      {α : Type} [_root_.Context α] [DecidableEq Shape] [ToString α] [Runtime.FromFloat α]
      [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α] :
      IO (Tensor.T Float τ) := do
    Module.withScalarLossModel
      (α := α) (mkModel := pure trainer.model) (opts := opts) (loss := trainer.loss)
      (k := fun model m => do
        let x' := Tensor.castFloat (Runtime.ofFloat (α := α)) x
        let y ← Module.predict (α := α) opts model m x'
        Tensor.toFloatIO y)
  if opts.usesCuda && run.dtype != .float then
    throw <| IO.userError
      "TorchLean.Trainer.predict: CUDA execution currently requires dtype Float"
  match (← Trainer.Implementation.withReadableRuntime run.dtype (fun {α} _ _ _ _ _ =>
      runFor (α := α))) with
  | .ok out => pure out
  | .error msg => throw <| IO.userError msg

/-- Build the regression dispatch record used by `Handle.train`. -/
def regressionHandle {σ τ : Shape} (trainer : Handle σ τ)
    (reduction : Loss.Reduction := .mean) : Regression σ τ :=
  { model := trainer.model
    reduction := reduction
    runtime := trainer.runtime }

/-- Build the cross-entropy dispatch record used by `Handle.train`. -/
def crossEntropyHandle {σ τ : Shape} (trainer : Handle σ τ)
    (reduction : Loss.Reduction := .mean) : CrossEntropy σ τ :=
  { model := trainer.model
    reduction := reduction
    runtime := trainer.runtime }

/-- Build the custom-loss dispatch record used by `Handle.train`. -/
def customHandle {σ τ : Shape} (trainer : Handle σ τ)
    (loss : ∀ {α : Type}, [_root_.Context α] → [DecidableEq Shape] →
      _root_.Runtime.Autograd.TorchLean.Program α [τ, τ] Shape.scalar) :
    Custom σ τ :=
  { model := trainer.model
    loss := loss
    runtime := trainer.runtime }

end Implementation

/--
Predict one input using the trainer's current model and runtime settings.

Inference before any training call. After training, use the returned trained handle's
`trained.predict` / `trained.predictBatch` methods to predict with the trained parameters.
-/
def Handle.predict {σ τ : Shape} (trainer : Handle σ τ) (x : Tensor.T Float σ) :
    IO (Tensor.T Float τ) := do
  match trainer.task with
  | .regression reduction =>
      let impl := Implementation.regressionHandle trainer reduction
      Implementation.Regression.Internal.withRunnerFromRunConfig impl impl.runConfig
        (fun {_} _ _ _ _ _ _ runner => Implementation.predictWithRunner runner x)
  | .classification reduction =>
      let impl := Implementation.crossEntropyHandle trainer reduction
      Implementation.CrossEntropy.Internal.withRunnerFromRunConfig impl impl.runConfig
        (fun {_} _ _ _ _ _ runner => Implementation.predictWithRunner runner x)
  | .crossEntropy reduction =>
      let impl := Implementation.crossEntropyHandle trainer reduction
      Implementation.CrossEntropy.Internal.withRunnerFromRunConfig impl impl.runConfig
        (fun {_} _ _ _ _ _ runner => Implementation.predictWithRunner runner x)
  | .custom loss =>
      let impl := Implementation.customHandle trainer loss
      Implementation.predictCustomWithRunConfig impl impl.runConfig x

/-- Predict a list of inputs using the trainer's current model and runtime settings. -/
def Handle.predictBatch {σ τ : Shape} (trainer : Handle σ τ) (xs : List (Tensor.T Float σ)) :
    IO (List (Tensor.T Float τ)) :=
  xs.mapM trainer.predict

/--
Train the model with the loss and runtime settings stored in `trainer`.

The result stores the trained parameters together with prediction and reporting methods.
-/
def Handle.train {σ τ : Shape} (trainer : Handle σ τ)
    (data : Dataset σ τ) (trainOptions : TrainOptions := {}) (probes : List (Probe σ) := []) :
    IO (TrainResult σ τ) := do
  match trainer.task with
  | .regression reduction =>
      (Implementation.regressionHandle trainer reduction).train data trainOptions probes
  | .classification reduction =>
      (Implementation.crossEntropyHandle trainer reduction).train data trainOptions probes
  | .crossEntropy reduction =>
      (Implementation.crossEntropyHandle trainer reduction).train data trainOptions probes
  | .custom loss =>
      (Implementation.customHandle trainer loss).train data trainOptions

/--
Train a regression model from a `Float` sample stream.

Generated-data examples use this when there is no fixed `Dataset` to hand to `trainer.train`.
-/
def Handle.trainStreamFloat {σ τ : Shape}
    (trainer : Handle σ τ)
    (opts : Options)
    (sampleAt : Nat → SupervisedSample Float σ τ)
    (evalSample : SupervisedSample Float σ τ)
    (trainOptions : TrainOptions := {})
    (curveEvery : Nat := 0)
    (cudaMemWatch : Nat := 0)
    (onEval : Nat → String → (Tensor.T Float σ → IO (Tensor.T Float τ)) → IO Unit :=
      fun _ _ _ => pure ()) :
    IO (StreamTrainResult σ τ) := do
  match trainer.task with
  | .regression reduction =>
      (Implementation.regressionHandle trainer reduction).trainStreamFloat opts sampleAt evalSample
        trainOptions
        (curveEvery := curveEvery) (cudaMemWatch := cudaMemWatch) (onEval := onEval)
  | _ =>
      throw <| IO.userError
        "Trainer.trainStreamFloat: stream training currently expects task := .regression"

/--
Train two regression models from coupled `Float` streams.

GAN-style examples use this path when two regression trainers have to step together, without opening
the lower-level runtime modules directly.
-/
def Handle.trainPairStreamFloat {σ₁ τ₁ σ₂ τ₂ : Shape}
    (first : Handle σ₁ τ₁)
    (second : Handle σ₂ τ₂)
    (opts : Options)
    (firstSampleAt : Nat → SupervisedSample Float σ₁ τ₁)
    (secondSamplesAt : Nat → List (SupervisedSample Float σ₂ τ₂))
    (evalTotal :
      (Tensor.T Float σ₁ → IO (Tensor.T Float τ₁)) →
      (Tensor.T Float σ₂ → IO (Tensor.T Float τ₂)) →
      IO Float)
    (trainOptions : TrainOptions := {})
    (curveEvery : Nat := 1)
    (cudaMemWatch : Nat := 0) :
    IO (PairStreamTrainResult σ₁ τ₁ σ₂ τ₂) := do
  match first.task, second.task with
  | .regression r1, .regression r2 =>
      Implementation.Regression.trainPairStreamFloat
        (Implementation.regressionHandle first r1) (Implementation.regressionHandle second r2) opts
        firstSampleAt secondSamplesAt evalTotal trainOptions
        (curveEvery := curveEvery) (cudaMemWatch := cudaMemWatch)
  | _, _ =>
      throw <| IO.userError
        "Trainer.trainPairStreamFloat: both trainers must use task := .regression"

/--
Train a cross-entropy model after the scalar type has already been selected.

Use this path from a runtime callback where Lean has already selected a concrete scalar `α`.
-/
def Handle.trainSelectedCrossEntropy {σ τ : Shape} {α : Type}
    [_root_.Context α] [DecidableEq Shape] [ToString α] [Runtime.FromFloat α]
    (trainer : Handle σ τ)
    (opts : Options) (data : Dataset σ τ) (trainOptions : TrainOptions := {})
    (probes : List (Probe σ) := []) :
    IO (TrainResult σ τ) := do
  match trainer.task with
  | .crossEntropy reduction =>
      (Implementation.crossEntropyHandle trainer reduction).trainSelected (α := α) opts data
        trainOptions probes
  | .classification reduction =>
      (Implementation.crossEntropyHandle trainer reduction).trainSelected (α := α) opts data
        trainOptions probes
  | _ =>
      throw <| IO.userError
        "Trainer.trainSelectedCrossEntropy: expected task := .crossEntropy or task := .classification"


end Trainer

end TorchLean
