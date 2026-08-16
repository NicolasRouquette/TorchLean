/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Trainer.Run
public import NN.API.Trainer.Results

/-!
# Custom-Loss Training

Custom checked supervised-loss training for the trainer API.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

namespace Implementation

namespace Custom

namespace Internal

/-- Mean loss for the custom trainer's concrete runtime module. -/
def meanModuleLoss {σ τ : Shape} {α : Type}
    [_root_.Context α] [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (model : TorchLean.nn.Sequential σ τ)
    (m : Module.Objective α (nn.stateShapes model) [σ, τ])
    (samples : List (SupervisedSample α σ τ)) : IO α := do
  match samples with
  | [] => pure 0
  | xs =>
      let vals ← xs.mapM (fun sample => Module.lossValue model m sample)
      pure (vals.foldl (· + ·) 0 / (xs.length : α))

/--
Shared custom-loss training core for already-parsed public runtime settings.

This opens an `Objective` for a custom supervised objective. The unified
`Trainer.new ... { task := .custom ... }` path uses the same checked module/loss/optimizer
machinery as the runtime trainer.
-/
def trainCore {σ τ : Shape} {β : Type}
    (trainer : Custom σ τ) (run : RunConfig) (data : DataSource σ τ)
    (trainOpts : TrainOptions)
    (afterTrain :
      {α : Type} → [_root_.Context α] → [DecidableEq Shape] → [ToString α] →
      [Runtime.FromFloat α] →
      (model : TorchLean.nn.Sequential σ τ) →
      Module.Objective α (nn.stateShapes model) [σ, τ] → IO β) :
    IO (TrainResult σ τ × β) := do
  let runtimeOpts := run.toRuntimeOptions
  let runFor
      {α : Type} [_root_.Context α] [DecidableEq Shape] [ToString α] [Runtime.FromFloat α]
      [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α] :
      IO (TrainResult σ τ × β) := do
    let model := trainer.model
    let objectiveDef := nn.Objective.create model (loss := trainer.loss)
    let m ← Module.instantiate (α := α) runtimeOpts objectiveDef
    let dataset ← data.build (α := α)
    let samples := dataset.toList
    IO.println s!"dataset size = {dataset.size}"
    let before ← meanModuleLoss model m samples
    let stepSample ← Module.makeSupervisedStep m run.optimizer
    let watchEvery := TorchLean.Trainer.Manual.CUDAMemory.cadence runtimeOpts trainOpts.steps
      trainOpts.cudaMemWatch
    let mut memWatch? ← TorchLean.Trainer.Manual.CUDAMemory.sample runtimeOpts watchEvery
      trainOpts.steps 0 none
    for stepIdx in [0:trainOpts.steps] do
      for sample in samples do
        stepSample sample
      memWatch? ← TorchLean.Trainer.Manual.CUDAMemory.sample runtimeOpts watchEvery
        trainOpts.steps (stepIdx + 1) memWatch?
      if trainOpts.logEvery > 0 && stepIdx % trainOpts.logEvery = 0 then
        let loss ← meanModuleLoss model m samples
        IO.println s!"step {stepIdx}: loss={loss}"
    let after ← meanModuleLoss model m samples
    let predict :=
      fun (xFloat : Tensor Float σ) => do
        let x := Tensor.castFloat (Runtime.ofFloat (α := α)) xFloat
        let yhat ← Module.Supervised.predict (α := α) runtimeOpts model m x
        Tensor.toFloatIO yhat
    let predictMany :=
      fun (xsFloat : List (Tensor Float σ)) => xsFloat.mapM predict
    let result : TrainResult σ τ :=
      { report :=
          { steps := trainOpts.steps
            before := toString before
            after := toString after }
        predict := predict
        predictMany := predictMany }
    let extra ← afterTrain (α := α) model m
    pure (result, extra)
  if runtimeOpts.usesCuda && run.scalar != .float32 then
    throw <| IO.userError
      "TorchLean.Trainer.train: CUDA execution currently requires --scalar float32"
  match (← Trainer.Implementation.withReadableRuntime run.scalar (fun {α} _ _ _ _ _ =>
      runFor (α := α))) with
  | .ok out => pure out
  | .error msg => throw <| IO.userError msg

end Internal

/--
Train on an in-memory dataset with a custom checked supervised loss and an explicit runtime override.

The shape matches the canned trainers: runtime choices come from `RunConfig`, training/logging
choices come from `TrainOptions`, and the returned result retains the trained model for prediction.
-/
def trainWithRun {σ τ : Shape} (trainer : Custom σ τ)
    (data : DataSource σ τ) (run : RunConfig := trainer.runConfig) (opts : TrainOptions := {}) :
    IO (TrainResult σ τ) := do
  let (report, _) ← Custom.Internal.trainCore trainer run data opts
    (fun {_} _ _ _ _ _ _ => pure ())
  report.report.writeLog opts.log opts.title opts.notes
  pure report

/-- Train on an in-memory dataset using this custom trainer's attached runtime settings. -/
def train {σ τ : Shape} (trainer : Custom σ τ)
    (data : DataSource σ τ) (opts : TrainOptions := {}) :
    IO (TrainResult σ τ) :=
  trainWithRun trainer data trainer.runConfig opts

end Custom

end Implementation

end Trainer

end TorchLean
