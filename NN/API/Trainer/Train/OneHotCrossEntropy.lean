/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Checkpoint
public import NN.API.Trainer.Run
public import NN.API.Trainer.Results
public import NN.API.Trainer.Manual.Loops

/-!
# One-Hot Cross-Entropy Training

One-hot cross-entropy training for classifiers, text windows, and structured logit tensors.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

namespace Internal

namespace OneHotCrossEntropy

/-- Cast exact-bit Float state to the runtime scalar selected for the current run. -/
def castStateBits {α : Type} [_root_.Context α] [Runtime.FromFloat α] :
    {ss : List Shape} → _root_.TorchLean.TensorPack Float ss → _root_.TorchLean.TensorPack α ss
  | [], .nil => .nil
  | _s :: ss, .cons x xs =>
      .cons (Spec.Tensor.map (Runtime.ofFloat (α := α)) x)
        (castStateBits (α := α) (ss := ss) xs)

/-- Convert runtime state back to the public Float checkpoint format. -/
def stateToFloatIO {α : Type} [_root_.Context α]
    [Runtime.TensorTransfer α] :
    {ss : List Shape} → _root_.TorchLean.TensorPack α ss → IO (_root_.TorchLean.TensorPack Float ss)
  | [], .nil => pure .nil
  | _s :: ss, .cons x xs => do
      let xF ← Runtime.toFloatTensor x
      let xsF ← stateToFloatIO (α := α) (ss := ss) xs
      pure (.cons xF xsF)

/--
Load optional checkpoint bits into a one-hot cross-entropy runner before training.

The saved file is checked against the model's state layout first, then cast into the selected
runtime scalar. That means stale checkpoints fail at the boundary instead of silently perturbing a
training run.
-/
def loadCheckpointIfSome {inputShape outputShape : List Nat} {α : Type}
    [_root_.Context α] [Runtime.FromFloat α] [DecidableEq Shape]
    (trainer : SelectedTask inputShape outputShape)
    (runner : TorchLean.Trainer.Manual.Runner α trainer.task)
    (path? : Option System.FilePath) : IO Unit := do
  match path? with
  | none => pure ()
  | some path =>
      let stateFloat ←
        TorchLean.Checkpoint.loadStateBits (stateShapes := nn.stateShapes trainer.model) path
      let state : _root_.TorchLean.TensorPack α trainer.task.stateShapes :=
        Eq.mpr
          (congrArg (_root_.TorchLean.TensorPack α) trainer.stateShapes_eq)
          (castStateBits (α := α) stateFloat)
      Module.loadState runner.module state

/-- Save optional trained checkpoint bits from a one-hot cross-entropy runner. -/
def saveCheckpointIfSome {inputShape outputShape : List Nat} {α : Type}
    [_root_.Context α] [Runtime.FromFloat α] [DecidableEq Shape]
    [Runtime.TensorTransfer α]
    (trainer : SelectedTask inputShape outputShape)
    (runner : TorchLean.Trainer.Manual.Runner α trainer.task)
    (path? : Option System.FilePath) : IO Unit := do
  match path? with
  | none => pure ()
  | some path =>
      let state ← Manual.Runner.state (task := trainer.task) runner
      let stateFloat ← stateToFloatIO (α := α) state
      let modelState : _root_.TorchLean.TensorPack Float (nn.stateShapes trainer.model) :=
        Eq.mp (congrArg (_root_.TorchLean.TensorPack Float) trainer.stateShapes_eq) stateFloat
      TorchLean.Checkpoint.writeStateBits (ss := nn.stateShapes trainer.model)
        path modelState
      IO.println s!"  wrote checkpoint: {path}"

/--
Shared one-hot cross-entropy training core for already-parsed public runtime settings.

This core is generic over shapes. It works for byte-level language-model windows,
sequence-to-sequence one-hot targets, and other supervised tasks whose target is already encoded as
a one-hot tensor with the same shape expected by the model loss.
-/
def trainSelectedCore {inputShape outputShape : List Nat} {β : Type} {α : Type}
    [_root_.Context α] [DecidableEq Shape] [ToString α] [Runtime.FromFloat α]
    [Runtime.TensorTransfer α]
    (trainer : SelectedTask inputShape outputShape)
    (runner : TorchLean.Trainer.Manual.Runner α trainer.task)
    (run : RunConfig) (data : Dataset inputShape outputShape)
    (opts : TrainOptions) (probes : Array (Probe inputShape) := #[])
    (afterTrain : {α : Type} → [_root_.Context α] → [DecidableEq Shape] → [ToString α] →
      [Runtime.FromFloat α] →
      TorchLean.Trainer.Manual.Runner α trainer.task → IO β) :
    IO (TrainResult inputShape outputShape × β) := do
  loadCheckpointIfSome (α := α) trainer runner opts.loadCheckpoint?
  let dataset ← data.build (α := α)
  IO.println s!"dataset size = {dataset.size}"

  let reportProbes := fun (title : String) => do
    unless probes.isEmpty do
      IO.println title
      for probe in probes do
        let yhat ← Manual.Runner.run (task := trainer.task) runner (probe.input (α := α))
        let expected :=
          match probe.expected with
          | some value => s!"  target={value}"
          | none => ""
        let inputText := if probe.inputText.isEmpty then "" else s!" {probe.inputText}"
        IO.println s!"  {probe.name}:{inputText}{expected}  pred={Tensor.pretty yhat}"

  TorchLean.Trainer.Manual.Report.meanLoss (task := trainer.task) runner dataset "before"
  reportProbes "predictions(before)"

  let cfg := opts.toTrainConfig run.optimizer
  let report ← TorchLean.Trainer.Manual.trainFinite (task := trainer.task) runner cfg dataset

  Manual.Runner.eval (task := trainer.task) runner
  TorchLean.Trainer.Manual.Report.meanLoss (task := trainer.task) runner dataset "after"
  reportProbes "predictions(after)"
  saveCheckpointIfSome (α := α) trainer runner opts.saveCheckpoint?
  let result := SelectedTask.toTrainResult (α := α) trainer runner cfg.steps
    report.before report.after
  let extra ← afterTrain (α := α) runner
  pure (result, extra)

/--
Shared one-hot cross-entropy training core for already-parsed runtime settings.

CLI commands may select the scalar type before calling into the public trainer. This entrypoint
keeps that path inside the trainer API instead of exposing manual module calls to examples.
-/
def trainCore {inputShape outputShape : List Nat} {β : Type}
    (trainer : SelectedTask inputShape outputShape) (run : RunConfig)
    (data : Dataset inputShape outputShape)
    (opts : TrainOptions) (probes : Array (Probe inputShape) := #[])
    (afterTrain : {α : Type} → [_root_.Context α] → [DecidableEq Shape] → [ToString α] →
      [Runtime.FromFloat α] →
      TorchLean.Trainer.Manual.Runner α trainer.task → IO β) :
    IO (TrainResult inputShape outputShape × β) := do
  SelectedTask.withRunner trainer run (fun {α} _ _ _ _ _ runner =>
    trainSelectedCore (α := α) trainer runner run data opts probes afterTrain)

/--
Train on an in-memory one-hot cross-entropy dataset using the trainer's attached runtime settings.

Persistent runtime choices live on the trainer, while step and logging choices live on
`TrainOptions`.
-/
def train {inputShape outputShape : List Nat}
    (trainer : SelectedTask inputShape outputShape)
    (data : Dataset inputShape outputShape) (opts : TrainOptions := {})
    (probes : Array (Probe inputShape) := #[]) : IO (TrainResult inputShape outputShape) := do
  let (result, _) ← OneHotCrossEntropy.trainCore trainer trainer.runtime data
    opts probes (fun {_} _ _ _ _ _ => pure ())
  result.report.writeLog opts.log opts.title opts.notes
  pure result

end OneHotCrossEntropy

end Internal

end Trainer

end TorchLean
