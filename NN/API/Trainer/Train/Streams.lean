/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Trainer.Run
public import NN.API.Trainer.Results
public import NN.API.Trainer.Train.Regression

/-!
# Stream Training

Regression stream and paired-stream training for generated or resampled workloads.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

namespace Implementation

namespace Regression

/--
Train a regression trainer from a step-indexed Float sample stream.

Use this when the "dataset" is really a recipe:

- diffusion draws a fresh noised image at each step,
- PDE examples resample collocation points,
- operator-learning demos cycle generated batches while evaluating on one fixed probe.

The public contract is still trainer-shaped. The caller supplies `sampleAt step`, TorchLean owns
the optimizer and runner state, and the returned value is the same trained model result used by
ordinary static-dataset training, plus a curve of evaluation loss on `evalSample`.
-/
def trainStreamWithRun {σ τ : Shape}
    (trainer : Regression σ τ)
    (runtimeOpts : Options)
    (sampleAt : Nat → Sample.Supervised Float σ τ)
    (evalSample : Sample.Supervised Float σ τ)
    (run : RunConfig := trainer.runConfig)
    (trainOpts : TrainOptions := {})
    (curveEvery : Nat := 0)
    (cudaMemWatch : Nat := 0)
    (onEval : Nat → String → (Tensor Float σ → IO (Tensor Float τ)) → IO Unit :=
      fun _ _ _ => pure ()) :
    IO (StreamTrainResult σ τ) := do
  let run := run.withRuntimeOptions runtimeOpts
  Regression.Internal.withRunner trainer run fun {α} _ _ _ _ _ _ runner => do
    let castSample (sample : Sample.Supervised Float σ τ) : Sample.Supervised α σ τ :=
      Sample.mk
        (Tensor.map (Runtime.ofFloat (α := α)) (Sample.x sample))
        (Tensor.map (Runtime.ofFloat (α := α)) (Sample.y sample))
    let scalarToFloat (value : α) : IO Float := do
      let valueFloat ← Tensor.toFloatIO (Spec.Tensor.scalar value)
      pure (Spec.Tensor.item valueFloat)
    let cfg := trainOpts.toTrainConfig run.optimizer
    let stepper ← TorchLean.Trainer.Manual.stepper
      (task := trainer.task) runner cfg.optimizer cfg.scheduler
    let predict :=
      fun (xFloat : Tensor Float σ) => do
        Manual.Runner.eval (task := trainer.task) runner
        let x := Tensor.map (Runtime.ofFloat (α := α)) xFloat
        let yhat ← Manual.Runner.run (task := trainer.task) runner x
        Tensor.toFloatIO yhat
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
    let trainResult := Regression.Internal.mkTrainResult (α := α) trainer runner cfg.steps
      beforeLossα lastα
    if trainOpts.log.isEnabled then
      Training.writeLogTo trainOpts.log
        (curve.toTrainLog trainOpts.title "loss" (notes := trainOpts.notes))
    pure { result := trainResult, curve := curve }

/--
Train a regression trainer from a Float sample stream using the trainer's attached runtime settings.

Stream analogue of `trainer.train`: most static datasets should use the unified method, while
generated or resampled workloads should use this entrypoint so they do not hand-roll module loops.
-/
def trainStream {σ τ : Shape}
    (trainer : Regression σ τ)
    (runtimeOpts : Options)
    (sampleAt : Nat → Sample.Supervised Float σ τ)
    (evalSample : Sample.Supervised Float σ τ)
    (trainOpts : TrainOptions := {})
    (curveEvery : Nat := 0)
    (cudaMemWatch : Nat := 0)
    (onEval : Nat → String → (Tensor Float σ → IO (Tensor Float τ)) → IO Unit :=
      fun _ _ _ => pure ()) :
    IO (StreamTrainResult σ τ) :=
  trainStreamWithRun trainer runtimeOpts sampleAt evalSample trainer.runConfig trainOpts curveEvery
    cudaMemWatch onEval

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
def trainPairStreams {σ₁ τ₁ σ₂ τ₂ : Shape}
    (first : Regression σ₁ τ₁)
    (second : Regression σ₂ τ₂)
    (runtimeOpts : Options)
    (firstSampleAt : Nat → Sample.Supervised Float σ₁ τ₁)
    (secondSamplesAt : Nat → List (Sample.Supervised Float σ₂ τ₂))
    (evalTotal :
      (Tensor Float σ₁ → IO (Tensor Float τ₁)) →
      (Tensor Float σ₂ → IO (Tensor Float τ₂)) →
      IO Float)
    (trainOpts : TrainOptions := {})
    (curveEvery : Nat := 1)
    (cudaMemWatch : Nat := 0) :
    IO (PairStreamTrainResult σ₁ τ₁ σ₂ τ₂) := do
  let firstRun := first.runConfig.withRuntimeOptions runtimeOpts
  let secondRun := second.runConfig.withRuntimeOptions runtimeOpts
  if firstRun.scalar != secondRun.scalar then
    throw <| IO.userError
      "Trainer.trainPairStreams: both trainers must use the same scalar semantics"
  Regression.Internal.withRunner first firstRun fun {α} _ _ _ _ _ _ firstRunner => do
    let castFirst (sample : Sample.Supervised Float σ₁ τ₁) : Sample.Supervised α σ₁ τ₁ :=
      Sample.mk
        (Tensor.map (Runtime.ofFloat (α := α)) (Sample.x sample))
        (Tensor.map (Runtime.ofFloat (α := α)) (Sample.y sample))
    let castSecond (sample : Sample.Supervised Float σ₂ τ₂) : Sample.Supervised α σ₂ τ₂ :=
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
      fun (xFloat : Tensor Float σ₁) => do
        Manual.Runner.eval (task := first.task) firstRunner
        let x := Tensor.map (Runtime.ofFloat (α := α)) xFloat
        let yhat ← Manual.Runner.run (task := first.task) firstRunner x
        Tensor.toFloatIO yhat
    let predictSecond :=
      fun (xFloat : Tensor Float σ₂) => do
        Manual.Runner.eval (task := second.task) secondRunner
        let x := Tensor.map (Runtime.ofFloat (α := α)) xFloat
        let yhat ← Manual.Runner.run (task := second.task) secondRunner x
        Tensor.toFloatIO yhat
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
      Regression.Internal.mkTrainResult (α := α) first firstRunner trainOpts.steps
        (Runtime.ofFloat beforeLoss) (Runtime.ofFloat last)
    let secondResult := Regression.Internal.mkTrainResult (α := α) second secondRunner
      trainOpts.steps (Runtime.ofFloat beforeLoss) (Runtime.ofFloat last)
    if trainOpts.log.isEnabled then
      Training.writeLogTo trainOpts.log
        (curve.toTrainLog trainOpts.title "loss" (notes := trainOpts.notes))
    pure { first := firstResult, second := secondResult, curve := curve }

end Regression

end Implementation

end Trainer

end TorchLean
