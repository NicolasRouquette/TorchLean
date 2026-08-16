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
# One-Hot Cross-Entropy Training

One-hot cross-entropy training for classifiers, text windows, and structured logit tensors.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

namespace Implementation

namespace OneHotCrossEntropy

namespace Internal

/-- Cast exact-bit Float state to the runtime scalar selected for the current run. -/
def castStateBits {α : Type} [_root_.Context α] [Runtime.FromFloat α] :
    {ss : List Shape} → TensorPack Float ss → TensorPack α ss
  | [], .nil => .nil
  | _s :: ss, .cons x xs =>
      .cons (Tensor.castFloat (Runtime.ofFloat (α := α)) x) (castStateBits (α := α) (ss := ss) xs)

/-- Convert runtime state back to the public Float checkpoint format. -/
def stateToFloatIO {α : Type} [_root_.Context α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α] :
    {ss : List Shape} → TensorPack α ss → IO (TensorPack Float ss)
  | [], .nil => pure .nil
  | _s :: ss, .cons x xs => do
      let xF ← Tensor.toFloatIO x
      let xsF ← stateToFloatIO (α := α) (ss := ss) xs
      pure (.cons xF xsF)

/--
Load optional checkpoint bits into a one-hot cross-entropy runner before training.

The saved file is checked against the model's state layout first, then cast into the selected
runtime scalar. That means stale checkpoints fail at the boundary instead of silently perturbing a
training run.
-/
def loadCheckpointIfSome {σ τ : Shape} {α : Type}
    [_root_.Context α] [Runtime.FromFloat α] [DecidableEq Shape]
    (trainer : OneHotCrossEntropy σ τ)
    (runner : TorchLean.Trainer.Manual.Runner α trainer.task)
    (path? : Option System.FilePath) : IO Unit := do
  match path? with
  | none => pure ()
  | some path =>
      let stateFloat ←
        TorchLean.Checkpoint.loadStateBits (stateShapes := nn.stateShapes trainer.model) path
      let state : TensorPack α (TorchLean.Trainer.Manual.stateShapes trainer.task) :=
        Eq.mpr (by rw [OneHotCrossEntropy.task_state_shapes_eq (trainer := trainer)])
          (castStateBits (α := α) stateFloat)
      Module.loadState runner.module state

/-- Save optional trained checkpoint bits from a one-hot cross-entropy runner. -/
def saveCheckpointIfSome {σ τ : Shape} {α : Type}
    [_root_.Context α] [Runtime.FromFloat α] [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (trainer : OneHotCrossEntropy σ τ)
    (runner : TorchLean.Trainer.Manual.Runner α trainer.task)
    (path? : Option System.FilePath) : IO Unit := do
  match path? with
  | none => pure ()
  | some path =>
      let state ← Manual.Runner.state (task := trainer.task) runner
      let stateFloat ← stateToFloatIO (α := α) state
      let modelState : TensorPack Float (nn.stateShapes trainer.model) :=
        Eq.mp (by rw [OneHotCrossEntropy.task_state_shapes_eq (trainer := trainer)]) stateFloat
      TorchLean.Checkpoint.saveStateBits (stateShapes := nn.stateShapes trainer.model)
        path modelState
      IO.println s!"  wrote checkpoint: {path}"

/--
Run a one-hot cross-entropy trainer directly from a public `RunConfig`.

Same direct runtime path used by regression/classifier trainers. It does *not* serialize the config
back into CLI flags or expose a `Runner` callback to ordinary examples.
-/
def withRunner {σ τ : Shape} {β : Type}
    (trainer : OneHotCrossEntropy σ τ) (run : RunConfig)
    (k : {α : Type} → [_root_.Context α] → [DecidableEq Shape] → [ToString α] →
      [Runtime.FromFloat α] →
      [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α] →
      TorchLean.Trainer.Manual.Runner α trainer.task → IO β) :
    IO β := do
  let opts := run.toRuntimeOptions
  if opts.usesCuda && run.scalar != .float32 then
    throw <| IO.userError
      "TorchLean.Trainer.OneHotCrossEntropy: CUDA execution requires --scalar float32"
  match run.scalar with
  | .float32 =>
      let runner ←
        TorchLean.Trainer.Manual.Runner.instantiate
          (task := trainer.task) (α := Float32) (opts := opts)
      k (α := Float32) runner
  | scalar =>
      match (← Trainer.Implementation.withReadableRuntime scalar (fun {α} _ _ _ _ _ => do
          let runner ←
            TorchLean.Trainer.Manual.Runner.instantiate
              (task := trainer.task) (α := α) (opts := opts)
          k (α := α) runner)) with
      | .ok out => pure out
      | .error msg => throw <| IO.userError msg

/--
Shared one-hot cross-entropy training core for already-parsed public runtime settings.

This core is generic over shapes. It works for byte-level language-model windows,
sequence-to-sequence one-hot targets, and other supervised tasks whose target is already encoded as
a one-hot tensor with the same shape expected by the model loss.
-/
def trainSelectedCore {σ τ : Shape} {β : Type} {α : Type}
    [_root_.Context α] [DecidableEq Shape] [ToString α] [Runtime.FromFloat α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (trainer : OneHotCrossEntropy σ τ)
    (runner : TorchLean.Trainer.Manual.Runner α trainer.task)
    (run : RunConfig) (data : DataSource σ τ)
    (opts : TrainOptions) (probes : List (Probe σ) := [])
    (afterTrain : {α : Type} → [_root_.Context α] → [DecidableEq Shape] → [ToString α] →
      [Runtime.FromFloat α] →
      TorchLean.Trainer.Manual.Runner α trainer.task → IO β) :
    IO (TrainResult σ τ × β) := do
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
  let report ← TorchLean.Trainer.Manual.trainDataset (task := trainer.task) runner cfg dataset

  Manual.Runner.eval (task := trainer.task) runner
  TorchLean.Trainer.Manual.Report.meanLoss (task := trainer.task) runner dataset "after"
  reportProbes "predictions(after)"
  saveCheckpointIfSome (α := α) trainer runner opts.saveCheckpoint?
  let predict :=
    fun (xFloat : Tensor Float σ) => do
      Manual.Runner.eval (task := trainer.task) runner
      let x := Tensor.castFloat (Runtime.ofFloat (α := α)) xFloat
      let yhat ← Manual.Runner.run (task := trainer.task) runner x
      Tensor.toFloatIO yhat
  let predictMany :=
    fun (xsFloat : List (Tensor Float σ)) => xsFloat.mapM predict
  let result : TrainResult σ τ :=
    { report :=
        { steps := cfg.steps
          before := toString report.before
          after := toString report.after }
      predict := predict
      predictMany := predictMany }
  let extra ← afterTrain (α := α) runner
  pure (result, extra)

/--
Shared one-hot cross-entropy training core for already-parsed runtime settings.

CLI commands may select the scalar type before calling into the public trainer. This entrypoint
keeps that path inside the trainer API instead of exposing manual module calls to examples.
-/
def trainCore {σ τ : Shape} {β : Type}
    (trainer : OneHotCrossEntropy σ τ) (run : RunConfig) (data : DataSource σ τ)
    (opts : TrainOptions) (probes : List (Probe σ) := [])
    (afterTrain : {α : Type} → [_root_.Context α] → [DecidableEq Shape] → [ToString α] →
      [Runtime.FromFloat α] →
      TorchLean.Trainer.Manual.Runner α trainer.task → IO β) :
    IO (TrainResult σ τ × β) := do
  withRunner trainer run (fun {α} _ _ _ _ _ runner =>
    trainSelectedCore (α := α) trainer runner run data opts probes afterTrain)

end Internal

end OneHotCrossEntropy

namespace OneHotCrossEntropy

/--
Train on an in-memory one-hot cross-entropy dataset using an explicit runtime override.

Use this when one call should temporarily override the optimizer, backend, scalar, or device settings
attached to the trainer.
-/
def trainWithRun {σ τ : Shape} (trainer : OneHotCrossEntropy σ τ)
    (data : DataSource σ τ) (run : RunConfig := trainer.runConfig) (opts : TrainOptions := {})
    (probes : List (Probe σ) := []) :
    IO (TrainResult σ τ) := do
  let (report, _) ← OneHotCrossEntropy.Internal.trainCore trainer run data
    opts probes
    (fun {_} _ _ _ _ _ => pure ())
  report.report.writeLog opts.log opts.title opts.notes
  pure report

/--
Train on an in-memory one-hot cross-entropy dataset using the trainer's attached runtime settings.

Sequence-model implementation behind `trainer.train`: persistent runtime choices live on the
trainer, while step/logging choices live on `TrainOptions`.
-/
def train {σ τ : Shape} (trainer : OneHotCrossEntropy σ τ)
    (data : DataSource σ τ) (opts : TrainOptions := {}) (probes : List (Probe σ) := []) :
    IO (TrainResult σ τ) :=
  trainWithRun trainer data trainer.runConfig opts probes

end OneHotCrossEntropy

end Implementation

end Trainer

end TorchLean
