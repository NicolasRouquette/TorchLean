/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.CLI
public import NN.API.Data.Loaders
public import NN.API.Trainer.Manual.Core

/-!
# Manual Training Evaluation and Reports

Runner prediction, scalar-loss and one-hot accuracy evaluation, and shared training reports.
-/

@[expose] public section

namespace TorchLean
namespace Trainer
namespace Manual

open SeqTask

/-- Values measured before and after a manual training run. -/
structure TrainReport (α : Type) where
  /-- Metric before training. -/
  before : α
  /-- Metric after training. -/
  after : α

namespace Runner

/-- Run one input tensor using the active mode (`.train` or `.eval`). -/
def run {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (x : Tensor α σ) : IO (Tensor α τ) := do
  if runner.module.opts.usesCuda then
    let selectedMode ← currentMode runner
    _root_.Runtime.Autograd.TorchLean.NN.Seq.forwardNoGrad
      (α := α) (tensorTransfer := runner.module.tensorTransfer)
      runner.module.opts task.model runner.module.trainer.state x (mode := selectedMode)
  else
    let ps ← state runner
    let predictor :=
      match ← currentMode runner with
      | .train => runner.predictorTrain
      | .eval => runner.predictorEval
    let args : _root_.TorchLean.TensorPack α
        (_root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes task.model ++ [σ]) :=
      TorchLean.TensorPack.append
        (ss₁ := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes task.model)
        (ss₂ := [σ]) ps (.cons x .nil)
    pure (_root_.Runtime.Autograd.Torch.TypedGraph.forward predictor args)

/-- Run an array of inputs using the active mode. -/
def runBatch {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (xs : Array (Tensor α σ)) : IO (Array (Tensor α τ)) :=
  xs.mapM (run runner)

/-- Compute `(correct, total)` along a class axis for a one-hot classification dataset. -/
def accuracyOneHot {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (axis : Nat) [Spec.Shape.AxisInBounds axis τ]
    (samples : Array (_root_.TorchLean.TensorPack α [σ, τ])) : IO (Nat × Nat) := do
  let mut correct := 0
  let mut total := 0
  for sample in samples do
    let (x, y) :=
      match sample with
      | .cons x (.cons y .nil) => (x, y)
    let logits ← run runner x
    let (sampleCorrect, sampleTotal) :=
      _root_.TorchLean.Metrics.accuracyOneHotAxis (α := α) axis logits y
    correct := correct + sampleCorrect
    total := total + sampleTotal
  pure (correct, total)

/-- Mean scalar loss over an array of supervised samples using the runner's active mode. -/
def meanLoss {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    (runner : Runner α task) (samples : Array (_root_.TorchLean.TensorPack α [σ, τ])) :
    IO α := do
  let values ←
    if runner.module.opts.usesCuda then
      samples.mapM (fun sample => do
        let loss ← TorchLean.Module.loss runner.module sample .nil
        pure (Spec.Tensor.item loss))
    else do
      let lossGraph :=
        match ← currentMode runner with
        | .train => runner.lossTrain
        | .eval => runner.lossEval
      let ps ← state runner
      samples.mapM (fun sample => do
        let args : _root_.TorchLean.TensorPack α (stateShapes task ++ [σ, τ]) :=
          TorchLean.TensorPack.append
            (α := α) (ss₁ := stateShapes task) (ss₂ := [σ, τ]) ps sample
        pure (Spec.Tensor.item <|
          _root_.Runtime.Autograd.Torch.TypedScalarGraph.forward lossGraph args))
  if values.isEmpty then pure 0
  else pure (values.foldl (· + ·) 0 / (values.size : α))

/-- Mean scalar loss over a finite sample stream without materializing it. -/
def meanLossStream {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    (runner : Runner α task)
    (dataset : _root_.TorchLean.Data.SampleStream (_root_.TorchLean.TensorPack α [σ, τ])) :
    IO α := do
  if dataset.isEmpty then
    pure 0
  else if runner.module.opts.usesCuda then
    let mut total : α := 0
    for h : i in [0:dataset.size] do
      have hi : i < dataset.size := h.2.1
      let loss ← TorchLean.Module.loss runner.module (dataset.get ⟨i, hi⟩) .nil
      total := total + Spec.Tensor.item loss
    pure (total / (dataset.size : α))
  else
    let lossGraph ←
      match ← currentMode runner with
      | .train => pure runner.lossTrain
      | .eval => pure runner.lossEval
    let ps ← state runner
    let mut total : α := 0
    for h : i in [0:dataset.size] do
      have hi : i < dataset.size := h.2.1
      let sample := dataset.get ⟨i, hi⟩
      let args : _root_.TorchLean.TensorPack α (stateShapes task ++ [σ, τ]) :=
        TorchLean.TensorPack.append
          (α := α) (ss₁ := stateShapes task) (ss₂ := [σ, τ]) ps sample
      total := total + Spec.Tensor.item
        (_root_.Runtime.Autograd.Torch.TypedScalarGraph.forward lossGraph args)
    pure (total / (dataset.size : α))

/-- Scalar loss for one sample through the instantiated runtime module. -/
def moduleLoss {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (runner : Runner α task) (sample : _root_.TorchLean.TensorPack α [σ, τ]) : IO α := do
  let loss ← TorchLean.Module.loss runner.module sample .nil
  pure (Spec.Tensor.item loss)

end Runner

/-- Evaluate mean scalar loss over one deterministic, drop-last loader epoch. -/
def Objective.meanLoss {inputShape targetShape : List Nat} {n : Nat}
    {stateShapes : List Shape} {α : Type} [_root_.Context α] [DecidableEq Shape]
    (module : TorchLean.Module.Objective α Unit stateShapes
      [Shape.ofList (n :: inputShape), Shape.ofList (n :: targetShape)])
    (loader : TorchLean.Data.SupervisedEpochs α n inputShape targetShape) : IO α := do
  let evaluationLoader : TorchLean.Data.EpochLoader
      (TorchLean.Sample.Supervised α inputShape targetShape) :=
    { loader.loader with shuffle := false, dropLast := true }
  let (_, rawBatches) ←
    match TorchLean.Data.EpochLoader.epoch "Objective.meanLoss" evaluationLoader with
    | .ok result => pure result
    | .error message => throw <| IO.userError s!"Objective.meanLoss: {message}"
  let mut total : α := 0
  let mut count : Nat := 0
  for rawBatch in rawBatches do
    let sample ← TorchLean.CLI.orThrow "Objective.meanLoss" <|
      TorchLean.Data.collateSupervised (α := α) (inputShape := inputShape)
        (targetShape := targetShape) n rawBatch
    total := total + Spec.Tensor.item (← TorchLean.Module.loss module sample .nil)
    count := count + 1
  if count = 0 then pure 0 else pure (total / (count : α))

/-- Evaluate one-hot classification accuracy over one deterministic, drop-last loader epoch. -/
def Runner.accuracyOneHotLoader
    {inputShape targetShape : List Nat} {batch : Nat}
    {task : SeqTask (Shape.ofList (batch :: inputShape))
      (Shape.ofList (batch :: targetShape))}
    {α : Type} [_root_.Context α] [DecidableEq Shape]
    (runner : Runner α task) (axis : Nat)
    [Shape.AxisInBounds axis (Shape.ofList targetShape)]
    (loader : TorchLean.Data.SupervisedEpochs α batch inputShape targetShape) :
    IO (Nat × Nat) := do
  let evaluationLoader : TorchLean.Data.EpochLoader
      (TorchLean.Sample.Supervised α inputShape targetShape) :=
    { loader.loader with shuffle := false, dropLast := true }
  let (_, rawBatches) ←
    match TorchLean.Data.EpochLoader.epoch "Runner.accuracyOneHotLoader" evaluationLoader with
    | .ok result => pure result
    | .error message => throw <| IO.userError s!"Runner.accuracyOneHotLoader: {message}"
  let mut correct := 0
  let mut total := 0
  for rawBatch in rawBatches do
    let sample ← TorchLean.CLI.orThrow "Runner.accuracyOneHotLoader" <|
      TorchLean.Data.collateSupervised (α := α) (inputShape := inputShape)
        (targetShape := targetShape) batch rawBatch
    let (batchCorrect, batchTotal) ← runner.accuracyOneHot (axis + 1) #[sample]
    correct := correct + batchCorrect
    total := total + batchTotal
  pure (correct, total)

namespace Report

/-- Print the mean loss of a finite supervised sample stream with a label. -/
def meanLoss
    {inputShape targetShape : List Nat} {task : SeqTask inputShape targetShape}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    (runner : Runner α task)
    (dataset : _root_.TorchLean.Data.SampleStream
      (TorchLean.Sample.Supervised α inputShape targetShape))
    (label : String) : IO Unit := do
  let loss ← Runner.meanLossStream (task := task) runner dataset
  IO.println s!"mean_loss({label}) = {loss}"

end Report

end Manual
end Trainer
end TorchLean
