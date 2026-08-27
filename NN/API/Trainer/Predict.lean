/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Trainer.Train

/-!
# Prediction and Training

Prediction and training methods on `TorchLean.Trainer`.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

/--
Predict one input using the trainer's current model and runtime settings.

Inference before any training call. After training, use the returned trained result's
`trained.predict` / `trained.predictMany` methods to predict with the trained parameters.
-/
def predict {inputShape outputShape : List Nat}
    (trainer : TorchLean.Trainer inputShape outputShape) (x : Tensor Float inputShape) :
    IO (Tensor Float outputShape) := do
  match trainer.task with
  | .regression reduction =>
      let impl := Internal.SelectedTask.regression trainer reduction
      Internal.SelectedTask.withRunner impl impl.runtime
        (fun {α} _ _ _ _ _ runner => do
          Manual.Runner.eval runner
          let x' := Spec.Tensor.map (Runtime.ofFloat (α := α)) x
          let y ← Manual.Runner.run runner x'
          Runtime.toFloatTensor y)
  | @Task.oneHotCrossEntropy _ axis validAxis reduction =>
      letI := validAxis
      let impl := Internal.SelectedTask.oneHotCrossEntropy trainer axis reduction
      Internal.SelectedTask.withRunner impl impl.runtime
        (fun {α} _ _ _ _ _ runner => do
          Manual.Runner.eval runner
          let x' := Spec.Tensor.map (Runtime.ofFloat (α := α)) x
          let y ← Manual.Runner.run runner x'
          Runtime.toFloatTensor y)
  | .custom loss =>
      let opts := trainer.runtime.toRuntimeOptions
      let runFor
          {α : Type} [_root_.Context α] [DecidableEq Shape] [ToString α]
          [Runtime.FromFloat α] [Runtime.TensorTransfer α] :
          IO (Tensor Float outputShape) := do
        let objectiveDef := nn.Objective.create trainer.model (loss := loss)
        let m ← Module.instantiate (α := α) opts objectiveDef
        let x' := Tensor.map (Runtime.ofFloat (α := α)) x
        let y ← Module.Supervised.predict (α := α) opts trainer.model m x'
        Runtime.toFloatTensor y
      if opts.usesCuda && trainer.runtime.scalar != .float32 then
        throw <| IO.userError
          "TorchLean.Trainer.predict: CUDA execution currently requires --scalar float32"
      match (← Internal.withReadableRuntime trainer.runtime.scalar
          (fun {α} _ _ _ _ _ => runFor (α := α))) with
      | .ok out => pure out
      | .error msg => throw <| IO.userError msg

/-- Predict an array of inputs using the trainer's current model and runtime settings. -/
def predictMany {inputShape outputShape : List Nat}
    (trainer : TorchLean.Trainer inputShape outputShape)
    (xs : Array (Tensor Float inputShape)) : IO (Array (Tensor Float outputShape)) :=
  xs.mapM trainer.predict

/--
Train the model with the loss and runtime settings stored in `trainer`.

The result stores the trained parameters together with prediction and reporting methods.
-/
def train {inputShape outputShape : List Nat}
    (trainer : TorchLean.Trainer inputShape outputShape)
    (data : Dataset inputShape outputShape) (trainOptions : TrainOptions := {})
    (probes : Array (Probe inputShape) := #[]) : IO (TrainResult inputShape outputShape) := do
  match trainer.task with
  | .regression reduction =>
      Internal.Regression.train
        (Internal.SelectedTask.regression trainer reduction) data trainOptions probes
  | @Task.oneHotCrossEntropy _ axis validAxis reduction =>
      letI := validAxis
      let impl := Internal.SelectedTask.oneHotCrossEntropy trainer axis reduction
      Internal.OneHotCrossEntropy.train impl data trainOptions probes
  | .custom loss =>
      Internal.Custom.train trainer loss data trainOptions

/--
Train a regression model from a `Float` sample stream.

Generated-data examples use this when there is no fixed `Dataset` to hand to `trainer.train`.
-/
def trainStream {inputShape outputShape : List Nat}
    (trainer : TorchLean.Trainer inputShape outputShape)
    (opts : Options)
    (sampleAt : Nat → Sample.Supervised Float inputShape outputShape)
    (evalSample : Sample.Supervised Float inputShape outputShape)
    (trainOptions : TrainOptions := {})
    (curveEvery : Nat := 0)
    (cudaMemWatch : Nat := 0)
    (onEval : Nat → String →
      (Tensor Float inputShape → IO (Tensor Float outputShape)) → IO Unit :=
      fun _ _ _ => pure ()) :
    IO (StreamTrainResult inputShape outputShape) := do
  match trainer.task with
  | .regression reduction =>
      Internal.Regression.trainStream
        (Internal.SelectedTask.regression trainer reduction) opts sampleAt evalSample
        trainOptions
        (curveEvery := curveEvery) (cudaMemWatch := cudaMemWatch) (onEval := onEval)
  | _ =>
      throw <| IO.userError
        "Trainer.trainStream: stream training currently expects task := .regression"

/--
Train two regression models from coupled `Float` streams.

GAN-style examples use this path when two regression trainers have to step together, without opening
the lower-level runtime modules directly.
-/
def trainPairStreams {inputShape₁ outputShape₁ inputShape₂ outputShape₂ : List Nat}
    (first : TorchLean.Trainer inputShape₁ outputShape₁)
    (second : TorchLean.Trainer inputShape₂ outputShape₂)
    (opts : Options)
    (firstSampleAt : Nat → Sample.Supervised Float inputShape₁ outputShape₁)
    (secondSamplesAt : Nat → Array (Sample.Supervised Float inputShape₂ outputShape₂))
    (evalTotal :
      (Tensor Float inputShape₁ → IO (Tensor Float outputShape₁)) →
      (Tensor Float inputShape₂ → IO (Tensor Float outputShape₂)) →
      IO Float)
    (trainOptions : TrainOptions := {})
    (curveEvery : Nat := 1)
    (cudaMemWatch : Nat := 0) :
    IO (PairStreamTrainResult inputShape₁ outputShape₁ inputShape₂ outputShape₂) := do
  match first.task, second.task with
  | .regression r1, .regression r2 =>
      Internal.Regression.trainPairStreams
        (Internal.SelectedTask.regression first r1)
        (Internal.SelectedTask.regression second r2) opts
        firstSampleAt secondSamplesAt evalTotal trainOptions
        (curveEvery := curveEvery) (cudaMemWatch := cudaMemWatch)
  | _, _ =>
      throw <| IO.userError
        "Trainer.trainPairStreams: both trainers must use task := .regression"

end Trainer

end TorchLean
