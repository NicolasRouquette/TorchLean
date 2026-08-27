/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Trainer.Run
public import NN.API.Trainer.Results
public import NN.API.Trainer.Memory

/-!
# Custom-Loss Training

Custom checked supervised-loss training for the trainer API.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

namespace Internal

namespace Custom

/-- Mean loss for the custom trainer's concrete runtime module. -/
def meanModuleLoss {inputShape targetShape : List Nat} {α : Type}
    [_root_.Context α] [DecidableEq Shape]
    [Runtime.TensorTransfer α]
    (model : TorchLean.nn.Sequential inputShape targetShape)
    (m : Module.Objective α Unit (nn.stateShapes model) [inputShape, targetShape])
    (samples : Data.SampleStream (Sample.Supervised α inputShape targetShape)) : IO α := do
  if samples.isEmpty then pure 0
  else
    let mut total : α := 0
    for h : i in [0:samples.size] do
      have hi : i < samples.size := h.2.1
      total := total + (← Module.Internal.lossValue model m (samples.get ⟨i, hi⟩))
    pure (total / (samples.size : α))

/--
Shared custom-loss training core for already-parsed public runtime settings.

This opens an `Objective` for a custom supervised objective. The unified
`Trainer.new ... { task := .custom ... }` path uses the same checked module/loss/optimizer
machinery as the runtime trainer.
-/
def trainCore {inputShape outputShape : List Nat} {β : Type}
    (trainer : TorchLean.Trainer inputShape outputShape)
    (loss : ∀ {α : Type}, [_root_.Context α] → [DecidableEq Shape] →
      _root_.Runtime.Autograd.TorchLean.Program α
        [Shape.ofList outputShape, Shape.ofList outputShape] ([] : List Nat))
    (run : RunConfig)
    (data : Dataset inputShape outputShape)
    (trainOpts : TrainOptions)
    (afterTrain :
      {α : Type} → [_root_.Context α] → [DecidableEq Shape] → [ToString α] →
      [Runtime.FromFloat α] →
      (model : TorchLean.nn.Sequential inputShape outputShape) →
      Module.Objective α Unit (nn.stateShapes model)
        [Shape.ofList inputShape, Shape.ofList outputShape] → IO β) :
    IO (TrainResult inputShape outputShape × β) := do
  let runtimeOpts := run.toRuntimeOptions
  let runFor
      {α : Type} [_root_.Context α] [DecidableEq Shape] [ToString α] [Runtime.FromFloat α]
    [Runtime.TensorTransfer α] :
      IO (TrainResult inputShape outputShape × β) := do
    let model := trainer.model
    let objectiveDef := nn.Objective.create model (loss := loss)
    let m ← Module.instantiate (α := α) runtimeOpts objectiveDef
    let dataset ← data.build (α := α)
    IO.println s!"dataset size = {dataset.size}"
    let before ← meanModuleLoss model m dataset
    let stepSample ← Module.makeSupervisedStep m run.optimizer
    let watchEvery := TorchLean.Trainer.Manual.CUDAMemory.cadence runtimeOpts trainOpts.steps
      trainOpts.cudaMemWatch
    let mut memWatch? ← TorchLean.Trainer.Manual.CUDAMemory.sample runtimeOpts watchEvery
      trainOpts.steps 0 none
    for stepIdx in [0:trainOpts.steps] do
      for h : i in [0:dataset.size] do
        have hi : i < dataset.size := h.2.1
        stepSample (dataset.get ⟨i, hi⟩)
      memWatch? ← TorchLean.Trainer.Manual.CUDAMemory.sample runtimeOpts watchEvery
        trainOpts.steps (stepIdx + 1) memWatch?
      if trainOpts.logEvery > 0 && stepIdx % trainOpts.logEvery = 0 then
        let loss ← meanModuleLoss model m dataset
        IO.println s!"step {stepIdx}: loss={loss}"
    let after ← meanModuleLoss model m dataset
    let predict :=
      fun (xFloat : Tensor Float inputShape) => do
        let x := Tensor.map (Runtime.ofFloat (α := α)) xFloat
        let yhat ← Module.Supervised.predict (α := α) runtimeOpts model m x
        Runtime.toFloatTensor yhat
    let predictMany :=
      fun (xsFloat : Array (Tensor Float inputShape)) => xsFloat.mapM predict
    let result : TrainResult inputShape outputShape :=
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
  match (← Trainer.Internal.withReadableRuntime run.scalar (fun {α} _ _ _ _ _ =>
      runFor (α := α))) with
  | .ok out => pure out
  | .error msg => throw <| IO.userError msg

/-- Train on an in-memory dataset using this custom trainer's attached runtime settings. -/
def train {inputShape outputShape : List Nat}
    (trainer : TorchLean.Trainer inputShape outputShape)
    (loss : ∀ {α : Type}, [_root_.Context α] → [DecidableEq Shape] →
      _root_.Runtime.Autograd.TorchLean.Program α
        [Shape.ofList outputShape, Shape.ofList outputShape] ([] : List Nat))
    (data : Dataset inputShape outputShape) (opts : TrainOptions := {}) :
    IO (TrainResult inputShape outputShape) := do
  let (result, _) ← Custom.trainCore trainer loss trainer.runtime data opts
    (fun {_} _ _ _ _ _ _ => pure ())
  result.report.writeLog opts.log opts.title opts.notes
  pure result

end Custom

end Internal

end Trainer

end TorchLean
