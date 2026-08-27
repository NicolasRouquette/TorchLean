/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Trainer.Run
public import NN.API.Trainer.Results
public import NN.API.Trainer.Train.Regression
public import NN.API.Trainer.Manual.Stepper

/-!
# Stream Training

Regression stream and paired-stream training for generated or resampled workloads.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

namespace Internal

namespace Regression

/-- Train a regression model from a step-indexed stream of generated or resampled examples. -/
def trainStream {inputShape outputShape : List Nat}
    (trainer : SelectedTask inputShape outputShape)
    (runtimeOpts : Options)
    (sampleAt : Nat → Sample.Supervised Float inputShape outputShape)
    (evalSample : Sample.Supervised Float inputShape outputShape)
    (trainOpts : TrainOptions := {})
    (curveEvery : Nat := 0)
    (cudaMemWatch : Nat := 0)
    (onEval : Nat → String →
      (Tensor Float inputShape → IO (Tensor Float outputShape)) → IO Unit :=
      fun _ _ _ => pure ()) :
    IO (StreamTrainResult inputShape outputShape) := do
  let run := trainer.runtime.withRuntimeOptions runtimeOpts
  SelectedTask.withRunner trainer run fun {α} _ _ _ _ _ runner => do
    let castSample (sample : Sample.Supervised Float inputShape outputShape) :
        Sample.Supervised α inputShape outputShape :=
      Sample.mk
        (Tensor.map (Runtime.ofFloat (α := α)) (Sample.x sample))
        (Tensor.map (Runtime.ofFloat (α := α)) (Sample.y sample))
    let scalarToFloat (value : α) : IO Float := do
      let valueFloat ← Runtime.toFloatTensor (Spec.Tensor.scalar value)
      pure (Spec.Tensor.item valueFloat)
    let cfg := trainOpts.toTrainConfig run.optimizer
    let stepper ← TorchLean.Trainer.Manual.stepper
      (task := trainer.task) runner cfg.optimizer cfg.scheduler
    let predict :=
      fun (xFloat : Tensor Float inputShape) => do
        Manual.Runner.eval (task := trainer.task) runner
        let x := Tensor.map (Runtime.ofFloat (α := α)) xFloat
        let yhat ← Manual.Runner.run (task := trainer.task) runner x
        Runtime.toFloatTensor yhat
    let evalLoss := do
      Manual.Runner.eval (task := trainer.task) runner
      TorchLean.Trainer.Manual.Runner.moduleLoss (task := trainer.task) runner
        (castSample evalSample)
    let beforeLossα ← evalLoss
    let beforeLoss ← scalarToFloat beforeLossα
    let mut curve : Training.Curve := {}
    curve := curve.push 0 beforeLoss
    onEval 0 "before" predict
    let mut lastα := beforeLossα
    let mut last := beforeLoss
    let every : Nat := if curveEvery = 0 then Nat.max 1 (cfg.steps / 50) else curveEvery
    let watchEvery :=
      TorchLean.Trainer.Manual.CUDAMemory.cadence runtimeOpts cfg.steps cudaMemWatch
    let mut memWatch? ←
      TorchLean.Trainer.Manual.CUDAMemory.sample runtimeOpts watchEvery cfg.steps 0 none
    for step in [0:cfg.steps] do
      let _ ← stepper.stepSample (castSample (sampleAt step))
      let done := step + 1
      memWatch? ← TorchLean.Trainer.Manual.CUDAMemory.sample runtimeOpts watchEvery cfg.steps done
        memWatch?
      if TorchLean.Training.shouldReport every done then
        lastα ← evalLoss
        last ← scalarToFloat lastα
        curve := curve.push done last
        onEval done s!"step {done}" predict
    if cfg.steps % every != 0 then
      lastα ← evalLoss
      last ← scalarToFloat lastα
      curve := curve.push cfg.steps last
    onEval cfg.steps "after" predict
    let trainResult := SelectedTask.toTrainResult (α := α) trainer runner cfg.steps
      beforeLossα lastα
    if trainOpts.log.isEnabled then
      Training.writeLogTo trainOpts.log
        (curve.toTrainLog trainOpts.title "loss" (notes := trainOpts.notes))
    pure { result := trainResult, curve := curve }

/--
Train two regression trainers with an alternating Float sample stream.

Public paired-model training path. A GAN is the motivating case: the generator receives one
supervised warm-up sample per step, while the discriminator may receive both real and fake score
samples. The trainer API handles the alternating optimizer mechanics and lets the example provide
only the domain-specific pieces:

- `firstSampleAt step` for the first model,
- `secondSamplesAt step` for one or more second-model updates,
- `evalTotal predictFirst predictSecond` for the scalar curve to record.

The callback sees only prediction functions, never modules or optimizer states. That boundary lets
examples define meaningful metrics without becoming miniature copies of the runtime trainer.

The trained results use the paired `evalTotal` value for their before/after summaries. For coupled
models, the generator and discriminator are judged by one task-level scalar, not by two unrelated
dataset losses. If a future caller needs separate reports, it should expose them through the
curve/history artifact rather than reopening the modules.
-/
def trainPairStreams {inputShape₁ outputShape₁ inputShape₂ outputShape₂ : List Nat}
    (first : SelectedTask inputShape₁ outputShape₁)
    (second : SelectedTask inputShape₂ outputShape₂)
    (runtimeOpts : Options)
    (firstSampleAt : Nat → Sample.Supervised Float inputShape₁ outputShape₁)
    (secondSamplesAt : Nat → Array (Sample.Supervised Float inputShape₂ outputShape₂))
    (evalTotal :
      (Tensor Float inputShape₁ → IO (Tensor Float outputShape₁)) →
      (Tensor Float inputShape₂ → IO (Tensor Float outputShape₂)) →
      IO Float)
    (trainOpts : TrainOptions := {})
    (curveEvery : Nat := 1)
    (cudaMemWatch : Nat := 0) :
    IO (PairStreamTrainResult inputShape₁ outputShape₁ inputShape₂ outputShape₂) := do
  let firstRun := RunConfig.withRuntimeOptions first.runtime runtimeOpts
  let secondRun := RunConfig.withRuntimeOptions second.runtime runtimeOpts
  if firstRun.scalar != secondRun.scalar then
    throw <| IO.userError
      "Trainer.trainPairStreams: both trainers must use the same scalar semantics"
  SelectedTask.withRunner first firstRun fun {α} _ _ _ _ _ firstRunner => do
    let castFirst (sample : Sample.Supervised Float inputShape₁ outputShape₁) :
        Sample.Supervised α inputShape₁ outputShape₁ :=
      Sample.mk
        (Tensor.map (Runtime.ofFloat (α := α)) (Sample.x sample))
        (Tensor.map (Runtime.ofFloat (α := α)) (Sample.y sample))
    let castSecond (sample : Sample.Supervised Float inputShape₂ outputShape₂) :
        Sample.Supervised α inputShape₂ outputShape₂ :=
      Sample.mk
        (Tensor.map (Runtime.ofFloat (α := α)) (Sample.x sample))
        (Tensor.map (Runtime.ofFloat (α := α)) (Sample.y sample))
    let secondRunner ←
      TorchLean.Trainer.Manual.Runner.instantiate
        (task := second.task) (α := α) (opts := secondRun.toRuntimeOptions)
    let firstCfg := trainOpts.toTrainConfig firstRun.optimizer
    let secondCfg := trainOpts.toTrainConfig secondRun.optimizer
    let firstStepper ← TorchLean.Trainer.Manual.stepper
      (task := first.task) firstRunner firstCfg.optimizer firstCfg.scheduler
    let secondStepper ← TorchLean.Trainer.Manual.stepper
      (task := second.task) secondRunner secondCfg.optimizer secondCfg.scheduler
    let predictFirst :=
      fun (xFloat : Tensor Float inputShape₁) => do
        Manual.Runner.eval (task := first.task) firstRunner
        let x := Tensor.map (Runtime.ofFloat (α := α)) xFloat
        let yhat ← Manual.Runner.run (task := first.task) firstRunner x
        Runtime.toFloatTensor yhat
    let predictSecond :=
      fun (xFloat : Tensor Float inputShape₂) => do
        Manual.Runner.eval (task := second.task) secondRunner
        let x := Tensor.map (Runtime.ofFloat (α := α)) xFloat
        let yhat ← Manual.Runner.run (task := second.task) secondRunner x
        Runtime.toFloatTensor yhat
    let beforeLoss ← evalTotal predictFirst predictSecond
    let mut curve : Training.Curve := {}
    curve := curve.push 0 beforeLoss
    let mut last := beforeLoss
    let every := Nat.max 1 curveEvery
    let watchEvery :=
      TorchLean.Trainer.Manual.CUDAMemory.cadence runtimeOpts trainOpts.steps cudaMemWatch
    let mut memWatch? ←
      TorchLean.Trainer.Manual.CUDAMemory.sample runtimeOpts watchEvery trainOpts.steps 0 none
    for step in [0:trainOpts.steps] do
      let _ ← firstStepper.stepSample (castFirst (firstSampleAt step))
      for sample in secondSamplesAt step do
        let _ ← secondStepper.stepSample (castSecond sample)
        pure ()
      let done := step + 1
      memWatch? ← TorchLean.Trainer.Manual.CUDAMemory.sample runtimeOpts watchEvery trainOpts.steps
        done memWatch?
      if TorchLean.Training.shouldReport every done then
        last ← evalTotal predictFirst predictSecond
        curve := curve.push done last
    if trainOpts.steps % every != 0 then
      last ← evalTotal predictFirst predictSecond
      curve := curve.push trainOpts.steps last
    let firstResult :=
      SelectedTask.toTrainResult (α := α) first firstRunner trainOpts.steps
        (Runtime.ofFloat beforeLoss) (Runtime.ofFloat last)
    let secondResult := SelectedTask.toTrainResult (α := α) second secondRunner
      trainOpts.steps (Runtime.ofFloat beforeLoss) (Runtime.ofFloat last)
    if trainOpts.log.isEnabled then
      Training.writeLogTo trainOpts.log
        (curve.toTrainLog trainOpts.title "loss" (notes := trainOpts.notes))
    pure { first := firstResult, second := secondResult, curve := curve }

end Regression

end Internal

end Trainer

end TorchLean
