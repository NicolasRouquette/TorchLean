/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.Trainer.Run
public import NN.API.Public.Facade.Trainer.Results
public import NN.API.Public.Facade.Trainer.Train.Regression

/-!
# TorchLean Stream Trainer Implementation

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
the optimizer and runner state, and the returned value is the same trained model handle used by
ordinary static-dataset training, plus a curve of evaluation loss on `evalSample`.
-/
def trainStreamFloatWithRun {σ τ : Shape}
    (trainer : Regression σ τ)
    (runtimeOpts : Options)
    (sampleAt : Nat → SupervisedSample Float σ τ)
    (evalSample : SupervisedSample Float σ τ)
    (run : RunConfig := trainer.runConfig)
    (trainOpts : TrainOptions := {})
    (curveEvery : Nat := 0)
    (cudaMemWatch : Nat := 0)
    (onEval : Nat → String → (Tensor.T Float σ → IO (Tensor.T Float τ)) → IO Unit :=
      fun _ _ _ => pure ()) :
    IO (StreamTrainResult σ τ) := do
  let run := run.withOptions runtimeOpts
  let runner ←
    NN.API.TorchLean.Trainer.instantiateConfigured
      (task := trainer.task) (α := Float) (opts := runtimeOpts)
  let cfg := trainOpts.toTrainConfig run.optimizer
  let stepper ← NN.API.TorchLean.Trainer.stepper
    (task := trainer.task) runner cfg.optimizer cfg.scheduler
  let predict :=
    fun (xFloat : Tensor.T Float σ) => do
      Manual.evalMode (task := trainer.task) runner
      Manual.predict (task := trainer.task) runner xFloat
  let evalLoss := do
    Manual.evalMode (task := trainer.task) runner
    NN.API.TorchLean.Supervised.moduleLoss (task := trainer.task) runner evalSample
  let beforeLoss ← evalLoss
  let mut curve : Training.Curve := {}
  curve := curve.push 0 beforeLoss
  onEval 0 "before" predict
  let mut last := beforeLoss
  let every : Nat := if curveEvery = 0 then Nat.max 1 (cfg.steps / 50) else curveEvery
  let watchEvery := NN.API.Common.effectiveCudaMemWatch runtimeOpts cfg.steps cudaMemWatch
  let mut memWatch? ← NN.API.Common.reportCudaMemWatch runtimeOpts watchEvery cfg.steps 0 none
  for step in [0:cfg.steps] do
    let _ ← stepper.stepSample (sampleAt step)
    let done := step + 1
    memWatch? ← NN.API.Common.reportCudaMemWatch runtimeOpts watchEvery cfg.steps done memWatch?
    if NN.API.Common.shouldLogStep every done then
      last ← evalLoss
      curve := curve.push done last
      onEval done s!"step {done}" predict
  if cfg.steps % every != 0 then
    last ← evalLoss
    curve := curve.push cfg.steps last
  onEval cfg.steps "after" predict
  let trainResult := Regression.Internal.trainedHandle (α := Float) trainer runner cfg.steps beforeLoss last
  if trainOpts.log.isEnabled then
    NN.API.Common.writeTrainLogTo trainOpts.log
      (curve.toTrainLog trainOpts.title "loss" (notes := trainOpts.notes))
  pure { result := trainResult, curve := curve }

/--
Train a regression trainer from a Float sample stream using the trainer's attached runtime settings.

Stream analogue of `trainer.train`: most static datasets should use the unified method, while
generated or resampled workloads should use this entrypoint so they do not hand-roll module loops.
-/
def trainStreamFloat {σ τ : Shape}
    (trainer : Regression σ τ)
    (runtimeOpts : Options)
    (sampleAt : Nat → SupervisedSample Float σ τ)
    (evalSample : SupervisedSample Float σ τ)
    (trainOpts : TrainOptions := {})
    (curveEvery : Nat := 0)
    (cudaMemWatch : Nat := 0)
    (onEval : Nat → String → (Tensor.T Float σ → IO (Tensor.T Float τ)) → IO Unit :=
      fun _ _ _ => pure ()) :
    IO (StreamTrainResult σ τ) :=
  trainStreamFloatWithRun trainer runtimeOpts sampleAt evalSample trainer.runConfig trainOpts curveEvery
    cudaMemWatch onEval

/--
Train two regression trainers with an alternating Float sample stream.

Public paired-model training path. A GAN is the motivating case: the generator receives one
supervised warm-up sample per step, while the discriminator may receive both real and fake score
samples. The trainer API handles the alternating optimizer mechanics and lets the example provide only the
domain-specific pieces:

- `firstSampleAt step` for the first model,
- `secondSamplesAt step` for one or more second-model updates,
- `evalTotal predictFirst predictSecond` for the scalar curve to record.

The callback sees only prediction functions, never modules or optimizer states. That boundary lets
examples define meaningful metrics without becoming miniature copies of the runtime trainer.

The trained handles use the paired `evalTotal` value for their before/after summaries. For coupled
models, the generator and discriminator are judged by one task-level scalar, not by two unrelated
dataset losses. If a future caller needs separate reports, it should expose them through the
curve/history artifact rather than reopening the modules.
-/
def trainPairStreamFloat {σ₁ τ₁ σ₂ τ₂ : Shape}
    (first : Regression σ₁ τ₁)
    (second : Regression σ₂ τ₂)
    (runtimeOpts : Options)
    (firstSampleAt : Nat → SupervisedSample Float σ₁ τ₁)
    (secondSamplesAt : Nat → List (SupervisedSample Float σ₂ τ₂))
    (evalTotal :
      (Tensor.T Float σ₁ → IO (Tensor.T Float τ₁)) →
      (Tensor.T Float σ₂ → IO (Tensor.T Float τ₂)) →
      IO Float)
    (trainOpts : TrainOptions := {})
    (curveEvery : Nat := 1)
    (cudaMemWatch : Nat := 0) :
    IO (PairStreamTrainResult σ₁ τ₁ σ₂ τ₂) := do
  let firstRun := first.runConfig.withOptions runtimeOpts
  let secondRun := second.runConfig.withOptions runtimeOpts
  let firstRunner ←
    NN.API.TorchLean.Trainer.instantiateConfigured
      (task := first.task) (α := Float) (opts := runtimeOpts)
  let secondRunner ←
    NN.API.TorchLean.Trainer.instantiateConfigured
      (task := second.task) (α := Float) (opts := runtimeOpts)
  let firstCfg := trainOpts.toTrainConfig firstRun.optimizer
  let secondCfg := trainOpts.toTrainConfig secondRun.optimizer
  let firstStepper ← NN.API.TorchLean.Trainer.stepper
    (task := first.task) firstRunner firstCfg.optimizer firstCfg.scheduler
  let secondStepper ← NN.API.TorchLean.Trainer.stepper
    (task := second.task) secondRunner secondCfg.optimizer secondCfg.scheduler
  let predictFirst :=
    fun (x : Tensor.T Float σ₁) => do
      Manual.evalMode (task := first.task) firstRunner
      Manual.predict (task := first.task) firstRunner x
  let predictSecond :=
    fun (x : Tensor.T Float σ₂) => do
      Manual.evalMode (task := second.task) secondRunner
      Manual.predict (task := second.task) secondRunner x
  let beforeLoss ← evalTotal predictFirst predictSecond
  let mut curve : Training.Curve := {}
  curve := curve.push 0 beforeLoss
  let mut last := beforeLoss
  let every := Nat.max 1 curveEvery
  let watchEvery := NN.API.Common.effectiveCudaMemWatch runtimeOpts trainOpts.steps cudaMemWatch
  let mut memWatch? ← NN.API.Common.reportCudaMemWatch runtimeOpts watchEvery trainOpts.steps 0 none
  for step in [0:trainOpts.steps] do
    let _ ← firstStepper.stepSample (firstSampleAt step)
    for s in secondSamplesAt step do
      let _ ← secondStepper.stepSample s
      pure ()
    let done := step + 1
    memWatch? ← NN.API.Common.reportCudaMemWatch runtimeOpts watchEvery trainOpts.steps done memWatch?
    if NN.API.Common.shouldLogStep every done then
      last ← evalTotal predictFirst predictSecond
      curve := curve.push done last
  if trainOpts.steps % every != 0 then
    last ← evalTotal predictFirst predictSecond
    curve := curve.push trainOpts.steps last
  let firstResult := Regression.Internal.trainedHandle (α := Float) first firstRunner trainOpts.steps beforeLoss last
  let secondResult := Regression.Internal.trainedHandle (α := Float) second secondRunner trainOpts.steps beforeLoss last
  if trainOpts.log.isEnabled then
    NN.API.Common.writeTrainLogTo trainOpts.log
      (curve.toTrainLog trainOpts.title "loss" (notes := trainOpts.notes))
  pure { first := firstResult, second := secondResult, curve := curve }

end Regression

end Implementation

end Trainer

end TorchLean
