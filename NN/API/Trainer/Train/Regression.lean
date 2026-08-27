/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Trainer.Run
public import NN.API.Trainer.Results
public import NN.API.Trainer.Manual.Loops

/-!
# Regression Training

Regression dataset training for the trainer API.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

namespace Internal

namespace Regression

/-- Train using an already-instantiated regression runner. -/
def trainCoreWithRunner {inputShape outputShape : List Nat} {α β : Type}
    [_root_.Context α] [DecidableEq Shape] [ToString α] [Runtime.FromFloat α]
    [Runtime.TensorTransfer α]
    (trainer : SelectedTask inputShape outputShape)
    (data : Dataset inputShape outputShape)
    (cfg : TorchLean.Trainer.Manual.TrainConfig)
    (probes : Array (Probe inputShape))
    (runner : TorchLean.Trainer.Manual.Runner α trainer.task)
    (afterTrain : TorchLean.Trainer.Manual.Runner α trainer.task → IO β) :
    IO (TrainResult inputShape outputShape × β) := do
  let dataset ← data.build (α := α)
  IO.println s!"dataset size = {dataset.size}"

  Manual.Runner.train (task := trainer.task) runner

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

  let report ← TorchLean.Trainer.Manual.trainFinite (task := trainer.task) runner cfg dataset

  Manual.Runner.eval (task := trainer.task) runner
  TorchLean.Trainer.Manual.Report.meanLoss (task := trainer.task) runner dataset "after"
  reportProbes "predictions(after)"
  let result := SelectedTask.toTrainResult (α := α) trainer runner cfg.steps
    report.before report.after
  let extra ← afterTrain runner
  pure (result, extra)

/--
Shared regression training core for already-parsed public runtime settings.

This starts from `RunConfig` rather than CLI strings, so library calls do not parse or print runtime
settings twice.
-/
def trainCore {inputShape outputShape : List Nat} {β : Type}
    (trainer : SelectedTask inputShape outputShape) (run : RunConfig)
    (data : Dataset inputShape outputShape)
    (cfg : TorchLean.Trainer.Manual.TrainConfig)
    (probes : Array (Probe inputShape) := #[])
    (afterTrain : {α : Type} → [_root_.Context α] → [DecidableEq Shape] → [ToString α] →
      [Runtime.FromFloat α] →
      TorchLean.Trainer.Manual.Runner α trainer.task → IO β) :
    IO (TrainResult inputShape outputShape × β) := do
  SelectedTask.withRunner trainer run (fun {α} _ _ _ _ _ runner => do
    trainCoreWithRunner (α := α) trainer data cfg probes runner (afterTrain (α := α)))

/--
Train on an in-memory regression dataset using the trainer's attached runtime settings.

For ordinary training:

- put persistent optimizer, backend, scalar, or device choices on the trainer value itself,
- pass per-training-call knobs such as `steps` and `logEvery` here.
-/
def train {inputShape outputShape : List Nat} (trainer : SelectedTask inputShape outputShape)
    (data : Dataset inputShape outputShape) (opts : TrainOptions := {})
    (probes : Array (Probe inputShape) := #[]) : IO (TrainResult inputShape outputShape) := do
  let (result, _) ← Regression.trainCore trainer trainer.runtime data
    (opts.toTrainConfig trainer.runtime.optimizer) probes
    (fun {_} _ _ _ _ _ => pure ())
  result.report.writeLog opts.log opts.title opts.notes
  pure result

end Regression

end Internal

end Trainer

end TorchLean
