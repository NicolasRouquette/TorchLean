/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Trainer.Run
public import NN.API.Trainer.Results
public import NN.API.Verification

/-!
# Regression Training

Regression dataset training for the trainer API.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

namespace Implementation

namespace Regression

namespace Internal

/-!
This namespace contains the dependent runner machinery behind the public regression trainer. The
public API stays small: examples talk about models, datasets, and artifacts, while this file carries
the shape-indexed runtime details.
-/

/--
Run a regression trainer directly from a public `RunConfig`.

Direct non-CLI execution path for the public training API. It deliberately avoids
`Manual.run trainer.task run.toArgs`, because that CLI-oriented path is designed for executable
commands that parse and print runtime flags themselves. Public trainer methods already hold a
`RunConfig`, so they can instantiate the runner directly and keep the user-facing output clean.
-/
def withRunnerFromRunConfig {σ τ : Shape} {β : Type}
    (trainer : Regression σ τ) (run : RunConfig)
    (k : {α : Type} → [_root_.Context α] → [DecidableEq Shape] → [ToString α] →
      [Runtime.FromFloat α] → [NN.MLTheory.CROWN.BoundOps α] →
      [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α] →
      TorchLean.Trainer.Manual.Runner α trainer.task → IO β) :
    IO β := do
  let opts := run.toOptions
  if opts.usesCuda && run.dtype != .float then
    throw <| IO.userError
      "TorchLean.Trainer.train: CUDA execution currently requires dtype Float"
  match run.dtype with
  | .float =>
      let runner ← TorchLean.Trainer.Manual.instantiateConfiguredFloat trainer.task opts
      k (α := Float) runner
  | .float32 { mode := .ieee754Exec } =>
      let α := TorchLean.Floats.F32 .ieee754Exec
      let runner ←
        TorchLean.Trainer.Manual.instantiateWithRuntimeOptions
          (task := trainer.task) (α := α) (opts := opts)
      k (α := α) runner
  | .real | .float32 { mode := .fp32 } =>
      throw <| IO.userError
        "TorchLean.Trainer.train: the selected dtype is proof-only and cannot execute training"
  | .complex _ =>
      throw <| IO.userError
        "TorchLean.Trainer.train: regression verification currently supports real scalar backends"

/--
Build the public trained regression result from an already-trained runner.

Both dataset and stream training end at the same place: a runner whose parameters have been
updated. This operation packages that runner behind the stable public API:

- `predict` casts ordinary `Float` tensors into the selected runtime scalar,
- `predictBatch` runs the same prediction path on a batched tensor,
- `verify` compiles the trained model into verifier IR and runs the public IBP request.

Keeping this here prevents every training variant from re-copying the same trained-model closures.
-/
def trainedHandle {σ τ : Shape} {α : Type}
    [_root_.Context α] [DecidableEq Shape] [ToString α] [Runtime.FromFloat α]
    [NN.MLTheory.CROWN.BoundOps α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (trainer : Regression σ τ)
    (runner : TorchLean.Trainer.Manual.Runner α trainer.task)
    (steps : Nat) (before after : α) :
    TrainResult σ τ :=
  let predict :=
    fun (xFloat : Tensor.T Float σ) => do
      Manual.evalMode (task := trainer.task) runner
      let x := Tensor.castFloat (Runtime.ofFloat (α := α)) xFloat
      let yhat ← Manual.predict (task := trainer.task) runner x
      Tensor.toFloatIO yhat
  let predictBatch :=
    fun (xsFloat : List (Tensor.T Float σ)) => do
      xsFloat.mapM predict
  let verifyRobustLInf :=
    fun (centerFloat : Tensor.T Float σ) (eps : Float) => do
      Manual.evalMode (task := trainer.task) runner
      let params : TensorPack α
          (TorchLean.Trainer.Manual.paramShapes trainer.task) ←
        Manual.params (task := trainer.task) runner
      let params' : TensorPack α (nn.paramShapes trainer.model) :=
        Eq.mp (by rw [Regression.taskParamShapes_eq (trainer := trainer)]) params
      let compiled ←
        match Verification.compileForward (α := α) trainer.model params' with
        | .ok c => pure c
        | .error e => throw <| IO.userError e
      let center := Tensor.castFloat (Runtime.ofFloat (α := α)) centerFloat
      let ps := Verification.seedLInfBall compiled center (Runtime.ofFloat eps)
      let ibp := Verification.runIBP compiled ps
      let outB ←
        match Verification.outputBox? compiled ibp with
        | .ok box => pure box
        | .error msg => throw <| IO.userError msg
      pure
        { nodes := compiled.graph.nodes.size
          outputDim := outB.dim
          lo := Tensor.pretty outB.lo
          hi := Tensor.pretty outB.hi }
  { report :=
      { steps := steps
        before := toString before
        after := toString after }
    predict := predict
    predictBatch := predictBatch
    verifyRobustLInf? := some verifyRobustLInf }

/--
Shared regression training core for already-parsed public runtime settings.

Path used by `trainer.train` and the regression implementation handle. It mirrors the CLI-backed
trainer body, but starts from `RunConfig` instead of CLI strings, so public API calls do not print
or parse runtime settings twice.
-/
def trainDatasetWithRunConfigCore {σ τ : Shape} {β : Type}
    (trainer : Regression σ τ) (run : RunConfig) (data : Dataset σ τ)
    (cfg : TorchLean.Trainer.Manual.TrainConfig) (probes : List (Probe σ) := [])
    (afterTrain : {α : Type} → [_root_.Context α] → [DecidableEq Shape] → [ToString α] →
      [Runtime.FromFloat α] →
      TorchLean.Trainer.Manual.Runner α trainer.task → IO β) :
    IO (TrainResult σ τ × β) := do
  withRunnerFromRunConfig trainer run (fun {α} _ _ _ _ _ _ runner => do
    let dataset ← data.build (α := α)
    IO.println s!"dataset size = {dataset.size}"

    Manual.trainMode (task := trainer.task) runner

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

    let report ← TorchLean.Trainer.Manual.trainDataset (task := trainer.task) runner cfg dataset

    Manual.evalMode (task := trainer.task) runner
    TorchLean.Trainer.Manual.Report.reportMeanLoss (task := trainer.task) runner dataset "after"
    reportProbes "predictions(after)"
    let result := trainedHandle (α := α) trainer runner cfg.steps report.before report.after
    let extra ← afterTrain (α := α) runner
    pure (result, extra))

end Internal

end Regression

namespace Regression

/--
Train on an in-memory regression dataset using an explicit runtime override.

Use this when one call should temporarily override the optimizer/backend/dtype/device settings
attached to the trainer.
-/
def trainWithRun {σ τ : Shape} (trainer : Regression σ τ)
    (data : Dataset σ τ) (run : RunConfig := trainer.runConfig) (opts : TrainOptions := {})
    (probes : List (Probe σ) := []) :
    IO (TrainResult σ τ) := do
  let (report, _) ← Regression.Internal.trainDatasetWithRunConfigCore trainer run data
    (opts.toTrainConfig run.optimizer) probes
    (fun {_} _ _ _ _ _ => pure ())
  report.report.writeLog opts.log opts.title opts.notes
  pure report

/--
Train on an in-memory regression dataset using the trainer's attached runtime settings.

For ordinary training:

- put persistent optimizer/backend/dtype/device choices on the trainer value itself,
- pass per-training-call knobs such as `steps` and `logEvery` here.
-/
def train {σ τ : Shape} (trainer : Regression σ τ)
    (data : Dataset σ τ) (opts : TrainOptions := {}) (probes : List (Probe σ) := []) :
    IO (TrainResult σ τ) :=
  trainWithRun trainer data trainer.runConfig opts probes

end Regression

end Implementation

end Trainer

end TorchLean
