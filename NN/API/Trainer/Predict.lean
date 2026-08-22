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

namespace Implementation

/-- Predict one Float input through a runtime runner and return a Float output. -/
def predictWithRunner {σ τ : Shape} {task : TorchLean.Trainer.Manual.SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Shape] [ToString α] [Runtime.FromFloat α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (runner : TorchLean.Trainer.Manual.Runner α task) (x : Tensor Float σ) :
    IO (Tensor Float τ) := do
  Manual.Runner.eval (task := task) runner
  let x' := Tensor.map (Runtime.ofFloat (α := α)) x
  let y ← Manual.Runner.run (task := task) runner x'
  Tensor.toFloatIO y

/-- Predict one input through a custom-loss trainer without first running training. -/
def predictCustom {σ τ : Shape}
    (trainer : Custom σ τ) (run : RunConfig) (x : Tensor Float σ) :
    IO (Tensor Float τ) := do
  let opts := run.toRuntimeOptions
  let runFor
      {α : Type} [_root_.Context α] [DecidableEq Shape] [ToString α] [Runtime.FromFloat α]
      [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α] :
      IO (Tensor Float τ) := do
    let model := trainer.model
    let objectiveDef := nn.Objective.create model (loss := trainer.loss)
    let m ← Module.instantiate (α := α) opts objectiveDef
    let x' := Tensor.map (Runtime.ofFloat (α := α)) x
    let y ← Module.Supervised.predict (α := α) opts model m x'
    Tensor.toFloatIO y
  if opts.usesCuda && run.scalar != .float32 then
    throw <| IO.userError
      "TorchLean.Trainer.predict: CUDA execution currently requires --scalar float32"
  match (← Trainer.Implementation.withReadableRuntime run.scalar (fun {α} _ _ _ _ _ =>
      runFor (α := α))) with
  | .ok out => pure out
  | .error msg => throw <| IO.userError msg

/-- Convert a public trainer to the regression implementation selected by its task. -/
def toRegression {σ τ : Shape} (trainer : TorchLean.Trainer σ τ)
    (reduction : Loss.Reduction := .mean) : Regression σ τ :=
  { model := trainer.model
    reduction := reduction
    runtime := trainer.runtime }

/-- Convert a public trainer to the cross-entropy implementation selected by its task. -/
def toOneHotCrossEntropy {σ τ : Shape} (trainer : TorchLean.Trainer σ τ)
    (axis : Nat) [_root_.Spec.Shape.AxisInBounds axis τ]
    (reduction : Loss.Reduction := .mean) : OneHotCrossEntropy σ τ :=
  { model := trainer.model
    axis := axis
    reduction := reduction
    runtime := trainer.runtime }

/-- Convert a public trainer to the custom-loss implementation selected by its task. -/
def toCustom {σ τ : Shape} (trainer : TorchLean.Trainer σ τ)
    (loss : ∀ {α : Type}, [_root_.Context α] → [DecidableEq Shape] →
      _root_.Runtime.Autograd.TorchLean.Program α [τ, τ] Shape.scalar) :
    Custom σ τ :=
  { model := trainer.model
    loss := loss
    runtime := trainer.runtime }

end Implementation

/--
Predict one input using the trainer's current model and runtime settings.

Inference before any training call. After training, use the returned trained result's
`trained.predict` / `trained.predictMany` methods to predict with the trained parameters.
-/
def predict {σ τ : Shape} (trainer : TorchLean.Trainer σ τ) (x : Tensor Float σ) :
    IO (Tensor Float τ) := do
  match trainer.task with
  | .regression reduction =>
      let impl := Implementation.toRegression trainer reduction
      Implementation.Regression.Internal.withRunner impl impl.runConfig
        (fun {_} _ _ _ _ _ _ runner => Implementation.predictWithRunner runner x)
  | @Task.oneHotCrossEntropy _ _ axis validAxis reduction =>
      letI := validAxis
      let impl := Implementation.toOneHotCrossEntropy trainer axis reduction
      Implementation.OneHotCrossEntropy.Internal.withRunner impl impl.runConfig
        (fun {_} _ _ _ _ _ runner => Implementation.predictWithRunner runner x)
  | .custom loss =>
      let impl := Implementation.toCustom trainer loss
      Implementation.predictCustom impl impl.runConfig x

/-- Predict a list of inputs using the trainer's current model and runtime settings. -/
def predictMany {σ τ : Shape} (trainer : TorchLean.Trainer σ τ)
    (xs : List (Tensor Float σ)) :
    IO (List (Tensor Float τ)) :=
  xs.mapM trainer.predict

/--
Train the model with the loss and runtime settings stored in `trainer`.

The result stores the trained parameters together with prediction and reporting methods.
-/
def train {σ τ : Shape} (trainer : TorchLean.Trainer σ τ)
    (data : DataSource σ τ) (trainOptions : TrainOptions := {}) (probes : List (Probe σ) := []) :
    IO (TrainResult σ τ) := do
  match trainer.task with
  | .regression reduction =>
      (Implementation.toRegression trainer reduction).train data trainOptions probes
  | @Task.oneHotCrossEntropy _ _ axis validAxis reduction =>
      letI := validAxis
      (Implementation.toOneHotCrossEntropy trainer axis reduction).train data trainOptions probes
  | .custom loss =>
      (Implementation.toCustom trainer loss).train data trainOptions

/--
Train a regression model from a `Float` sample stream.

Generated-data examples use this when there is no fixed `Dataset` to hand to `trainer.train`.
-/
def trainStream {σ τ : Shape}
    (trainer : TorchLean.Trainer σ τ)
    (opts : Options)
    (sampleAt : Nat → Sample.Supervised Float σ τ)
    (evalSample : Sample.Supervised Float σ τ)
    (trainOptions : TrainOptions := {})
    (curveEvery : Nat := 0)
    (cudaMemWatch : Nat := 0)
    (onEval : Nat → String → (Tensor Float σ → IO (Tensor Float τ)) → IO Unit :=
      fun _ _ _ => pure ()) :
    IO (StreamTrainResult σ τ) := do
  match trainer.task with
  | .regression reduction =>
      (Implementation.toRegression trainer reduction).trainStream opts sampleAt evalSample
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
def trainPairStreams {σ₁ τ₁ σ₂ τ₂ : Shape}
    (first : TorchLean.Trainer σ₁ τ₁)
    (second : TorchLean.Trainer σ₂ τ₂)
    (opts : Options)
    (firstSampleAt : Nat → Sample.Supervised Float σ₁ τ₁)
    (secondSamplesAt : Nat → List (Sample.Supervised Float σ₂ τ₂))
    (evalTotal :
      (Tensor Float σ₁ → IO (Tensor Float τ₁)) →
      (Tensor Float σ₂ → IO (Tensor Float τ₂)) →
      IO Float)
    (trainOptions : TrainOptions := {})
    (curveEvery : Nat := 1)
    (cudaMemWatch : Nat := 0) :
    IO (PairStreamTrainResult σ₁ τ₁ σ₂ τ₂) := do
  match first.task, second.task with
  | .regression r1, .regression r2 =>
      Implementation.Regression.trainPairStreams
        (Implementation.toRegression first r1) (Implementation.toRegression second r2) opts
        firstSampleAt secondSamplesAt evalTotal trainOptions
        (curveEvery := curveEvery) (cudaMemWatch := cudaMemWatch)
  | _, _ =>
      throw <| IO.userError
        "Trainer.trainPairStreams: both trainers must use task := .regression"

end Trainer

end TorchLean
