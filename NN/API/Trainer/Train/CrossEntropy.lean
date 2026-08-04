/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Checkpoint
public import NN.API.Trainer.Run
public import NN.API.Trainer.Results

/-!
# Cross-Entropy Training

One-hot cross-entropy training for classifiers, text windows, and structured logit tensors.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

namespace Implementation

namespace CrossEntropy

namespace Internal

/-- Cast a Float checkpoint payload to the runtime scalar selected for the current run. -/
def castParamBits {α : Type} [_root_.Context α] [Runtime.FromFloat α] :
    {ss : List Shape} → TensorPack Float ss → TensorPack α ss
  | [], .nil => .nil
  | _s :: ss, .cons x xs =>
      .cons (Tensor.castFloat (Runtime.ofFloat (α := α)) x) (castParamBits (α := α) (ss := ss) xs)

/-- Convert a runtime parameter payload back to the public Float checkpoint format. -/
def paramsToFloatIO {α : Type} [_root_.Context α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α] :
    {ss : List Shape} → TensorPack α ss → IO (TensorPack Float ss)
  | [], .nil => pure .nil
  | _s :: ss, .cons x xs => do
      let xF ← Tensor.toFloatIO x
      let xsF ← paramsToFloatIO (α := α) (ss := ss) xs
      pure (.cons xF xsF)

/--
Load optional checkpoint bits into a cross-entropy runner before training.

The saved file is checked against the model's parameter shapes first, then cast into the selected
runtime scalar. That means stale checkpoints fail at the boundary instead of silently perturbing a
training run.
-/
def loadCheckpointIfSome {σ τ : Shape} {α : Type}
    [_root_.Context α] [Runtime.FromFloat α] [DecidableEq Shape]
    (trainer : CrossEntropy σ τ)
    (runner : TorchLean.Trainer.Manual.Runner α trainer.task)
    (path? : Option System.FilePath) : IO Unit := do
  match path? with
  | none => pure ()
  | some path =>
      let psF ←
        TorchLean.Checkpoint.loadParamBits (paramShapes := nn.paramShapes trainer.model) path
      let psTask : TensorPack α (TorchLean.Trainer.Manual.paramShapes trainer.task) :=
        Eq.mpr (by rw [CrossEntropy.taskParamShapes_eq (trainer := trainer)])
          (castParamBits (α := α) psF)
      Module.setParams runner.module psTask

/-- Save optional trained checkpoint bits from a cross-entropy runner. -/
def saveCheckpointIfSome {σ τ : Shape} {α : Type}
    [_root_.Context α] [Runtime.FromFloat α] [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (trainer : CrossEntropy σ τ)
    (runner : TorchLean.Trainer.Manual.Runner α trainer.task)
    (path? : Option System.FilePath) : IO Unit := do
  match path? with
  | none => pure ()
  | some path =>
      let ps ← Manual.params (task := trainer.task) runner
      let psFTask ← paramsToFloatIO (α := α) ps
      let psFModel : TensorPack Float (nn.paramShapes trainer.model) :=
        Eq.mp (by rw [CrossEntropy.taskParamShapes_eq (trainer := trainer)]) psFTask
      TorchLean.Checkpoint.saveParamBits (paramShapes := nn.paramShapes trainer.model)
        path psFModel
      IO.println s!"  wrote params: {path}"

/--
Run a general cross-entropy trainer directly from a public `RunConfig`.

Same direct runtime path used by regression/classifier trainers. It does *not* serialize the config
back into CLI flags or expose a `Runner` callback to ordinary examples.
-/
def withRunnerFromRunConfig {σ τ : Shape} {β : Type}
    (trainer : CrossEntropy σ τ) (run : RunConfig)
    (k : {α : Type} → [_root_.Context α] → [DecidableEq Shape] → [ToString α] →
      [Runtime.FromFloat α] →
      [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α] →
      TorchLean.Trainer.Manual.Runner α trainer.task → IO β) :
    IO β := do
  let opts := run.toOptions
  if opts.usesCuda && run.dtype != .float then
    throw <| IO.userError
      "TorchLean.Trainer.trainSelectedCrossEntropy: CUDA execution currently requires dtype Float"
  match run.dtype with
  | .float =>
      let runner ← TorchLean.Trainer.Manual.instantiateConfiguredFloat trainer.task opts
      k (α := Float) runner
  | dtype =>
      match (← Trainer.Implementation.withReadableRuntime dtype (fun {α} _ _ _ _ _ => do
          let runner ←
            TorchLean.Trainer.Manual.instantiateWithRuntimeOptions
              (task := trainer.task) (α := α) (opts := opts)
          k (α := α) runner)) with
      | .ok out => pure out
      | .error msg => throw <| IO.userError msg

/--
Shared cross-entropy training core for already-parsed public runtime settings.

This core is generic over shapes. It works for byte-level language-model windows,
sequence-to-sequence one-hot targets, and other supervised tasks whose target is already encoded as
a one-hot tensor with the same shape expected by the model loss.
-/
def trainDatasetWithSelectedRunnerCore {σ τ : Shape} {β : Type} {α : Type}
    [_root_.Context α] [DecidableEq Shape] [ToString α] [Runtime.FromFloat α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (trainer : CrossEntropy σ τ)
    (runner : TorchLean.Trainer.Manual.Runner α trainer.task)
    (run : RunConfig) (data : Dataset σ τ)
    (opts : TrainOptions) (probes : List (Probe σ) := [])
    (afterTrain : {α : Type} → [_root_.Context α] → [DecidableEq Shape] → [ToString α] →
      [Runtime.FromFloat α] →
      TorchLean.Trainer.Manual.Runner α trainer.task → IO β) :
    IO (TrainResult σ τ × β) := do
  loadCheckpointIfSome (α := α) trainer runner opts.loadParams?
  let dataset ← data.build (α := α)
  IO.println s!"dataset size = {dataset.size}"

  let reportProbes := fun (title : String) => do
    unless probes.isEmpty do
      IO.println title
      for probe in probes do
        let yhat ← Manual.predict (task := trainer.task) runner (probe.input (α := α))
        let expected :=
          match probe.expected with
          | some value => s!"  target={value}"
          | none => ""
        let inputText := if probe.inputText.isEmpty then "" else s!" {probe.inputText}"
        IO.println s!"  {probe.name}:{inputText}{expected}  pred={Tensor.pretty yhat}"

  TorchLean.Trainer.Manual.Report.reportMeanLoss (task := trainer.task) runner dataset "before"
  reportProbes "predictions(before)"

  let cfg := opts.toTrainConfig run.optimizer
  let report ← TorchLean.Trainer.Manual.trainDataset (task := trainer.task) runner cfg dataset

  Manual.evalMode (task := trainer.task) runner
  TorchLean.Trainer.Manual.Report.reportMeanLoss (task := trainer.task) runner dataset "after"
  reportProbes "predictions(after)"
  saveCheckpointIfSome (α := α) trainer runner opts.saveParams?
  let predict :=
    fun (xFloat : Tensor.T Float σ) => do
      Manual.evalMode (task := trainer.task) runner
      let x := Tensor.castFloat (Runtime.ofFloat (α := α)) xFloat
      let yhat ← Manual.predict (task := trainer.task) runner x
      Tensor.toFloatIO yhat
  let predictBatch :=
    fun (xsFloat : List (Tensor.T Float σ)) => xsFloat.mapM predict
  let result : TrainResult σ τ :=
    { report :=
        { steps := cfg.steps
          before := toString report.before
          after := toString report.after }
      predict := predict
      predictBatch := predictBatch }
  let extra ← afterTrain (α := α) runner
  pure (result, extra)

/--
Shared cross-entropy training core for already-parsed runtime settings.

CLI commands may select the scalar type before calling into the public trainer. This entrypoint
keeps that path inside the trainer API instead of exposing manual module calls to examples.
-/
def trainDatasetWithRunConfigCore {σ τ : Shape} {β : Type}
    (trainer : CrossEntropy σ τ) (run : RunConfig) (data : Dataset σ τ)
    (opts : TrainOptions) (probes : List (Probe σ) := [])
    (afterTrain : {α : Type} → [_root_.Context α] → [DecidableEq Shape] → [ToString α] →
      [Runtime.FromFloat α] →
      TorchLean.Trainer.Manual.Runner α trainer.task → IO β) :
    IO (TrainResult σ τ × β) := do
  withRunnerFromRunConfig trainer run (fun {α} _ _ _ _ _ runner =>
    trainDatasetWithSelectedRunnerCore (α := α) trainer runner run data opts probes afterTrain)

end Internal

end CrossEntropy

namespace CrossEntropy

/--
Train on an in-memory one-hot cross-entropy dataset using an explicit runtime override.

Use this when one call should temporarily override the optimizer/backend/dtype/device settings
attached to the trainer.
-/
def trainWithRun {σ τ : Shape} (trainer : CrossEntropy σ τ)
    (data : Dataset σ τ) (run : RunConfig := trainer.runConfig) (opts : TrainOptions := {})
    (probes : List (Probe σ) := []) :
    IO (TrainResult σ τ) := do
  let (report, _) ← CrossEntropy.Internal.trainDatasetWithRunConfigCore trainer run data
    opts probes
    (fun {_} _ _ _ _ _ => pure ())
  report.report.writeLog opts.log opts.title opts.notes
  pure report

/--
Train with a runtime scalar that has already been selected by the caller.

Call `trainer.train` when the scalar should be chosen from `trainer.runConfig.dtype`.
This method exists for model-zoo dispatchers that already run inside a runtime callback. At that
point Lean has a concrete scalar type `α`, not merely a `Runtime.DType` tag. Even these deep-dive
examples still use the public trainer API, so they do not
have to open-code module creation, loss calls, or optimizer steps.
-/
def trainSelected {σ τ : Shape} {α : Type}
    [_root_.Context α] [DecidableEq Shape] [ToString α] [Runtime.FromFloat α]
    (trainer : CrossEntropy σ τ)
    (runtimeOpts : Options) (data : Dataset σ τ) (trainOpts : TrainOptions := {})
    (probes : List (Probe σ) := []) :
    IO (TrainResult σ τ) := do
  let run := (trainer.runConfig.withOptions runtimeOpts)
  let runner ←
    TorchLean.Trainer.Manual.instantiateWithRuntimeOptions
      (task := trainer.task) (α := α) (opts := runtimeOpts)
  let (report, _) ← CrossEntropy.Internal.trainDatasetWithSelectedRunnerCore
    (α := α) trainer runner run data trainOpts probes
    (fun {_} _ _ _ _ _ => pure ())
  report.report.writeLog trainOpts.log trainOpts.title trainOpts.notes
  pure report

/--
Train on an in-memory one-hot cross-entropy dataset using the trainer's attached runtime settings.

Sequence-model implementation behind `trainer.train`: persistent runtime choices live on the
trainer, while step/logging choices live on `TrainOptions`.
-/
def train {σ τ : Shape} (trainer : CrossEntropy σ τ)
    (data : Dataset σ τ) (opts : TrainOptions := {}) (probes : List (Probe σ) := []) :
    IO (TrainResult σ τ) :=
  trainWithRun trainer data trainer.runConfig opts probes

end CrossEntropy

end Implementation

end Trainer

end TorchLean
