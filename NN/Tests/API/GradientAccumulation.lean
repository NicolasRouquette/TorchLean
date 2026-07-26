/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Runtime.Training.Loops

/-!
# Gradient Accumulation

These checks cover mean-gradient updates, loader scheduler cadence, native-hook selection, and
partial loader batches. The probe optimizer records which route each update used without coupling
the test to a particular backend implementation.
-/

@[expose] public section

namespace NN.Tests.API.GradientAccumulation

open NN.API.TorchLean

def vector1 (x : Float) : Spec.Tensor Float (.dim 1 .scalar) :=
  .dim (fun _ => .scalar x)

def model :=
  API.TorchLean.Layers.linear 1 1 17 29

def task : Supervised.SeqTask (.dim 1 .scalar) (.dim 1 .scalar) :=
  Supervised.SeqTask.mse model

def sample (x y : Float) :
    API.TorchLean.TensorPack Float [.dim 1 .scalar, .dim 1 .scalar] :=
  .cons (vector1 x) (.cons (vector1 y) .nil)

def readLinearParams
    (ps : API.TorchLean.TensorPack Float (Supervised.paramShapes task)) :
    Float × Float :=
  match ps with
  | .cons weight (.cons bias .nil) =>
      ( Spec.Tensor.toScalar (Spec.Tensor.get (Spec.Tensor.get weight 0) 0)
      , Spec.Tensor.toScalar (Spec.Tensor.get bias 0) )

def close (x y : Float) : Bool :=
  Float.abs (x - y) ≤ 1e-5

/-- One-channel image used by the BatchNorm buffer regression. -/
def constantImage (value : Float) :
    Spec.Tensor Float (.dim 1 (.dim 1 (.dim 2 .scalar))) :=
  .dim fun _ => .dim fun _ => .dim fun _ => .scalar value

def batchNormModel :
    API.TorchLean.LayerCore.Seq
      (.dim 1 (.dim 1 (.dim 2 .scalar)))
      (.dim 1 (.dim 1 (.dim 2 .scalar))) :=
  API.TorchLean.LayerCore.singleLayer <|
    API.TorchLean.LayerCore.batchnormChannelFirstMode 1 1 2
      (h_c := by decide) (h_h := by decide) (h_w := by decide) (momentum := 0.5)

def batchNormTask :
    Supervised.SeqTask (.dim 1 (.dim 1 (.dim 2 .scalar)))
      (.dim 1 (.dim 1 (.dim 2 .scalar))) :=
  Supervised.SeqTask.mse batchNormModel

def batchNormSample (value : Float) :
    API.TorchLean.TensorPack Float
      [.dim 1 (.dim 1 (.dim 2 .scalar)), .dim 1 (.dim 1 (.dim 2 .scalar))] :=
  .cons (constantImage value) (.cons (constantImage 0.0) .nil)

def readBatchNormBuffers
    (ps : API.TorchLean.TensorPack Float (Supervised.paramShapes batchNormTask)) :
    Float × Float :=
  match ps with
  | .cons _gamma (.cons _beta (.cons mean (.cons variance (.cons _momentum .nil)))) =>
      ( Spec.Tensor.toScalar (Spec.Tensor.get mean 0)
      , Spec.Tensor.toScalar (Spec.Tensor.get variance 0) )

def noOpOptimizer (shapes : List Spec.Shape) :
    API.TorchLean.Optim.Optimizer Float shapes where
  State := Unit
  init := fun _ => pure ()
  step := fun _ _ _ => pure ()

/-- Counters shared by the probe optimizer and the assertions that follow a run. -/
structure ProbeCounters where
  nativeSteps : IO.Ref Nat
  lossSteps : IO.Ref Nat
  genericSteps : IO.Ref Nat
  scheduledValues : IO.Ref (List Nat)

/-- Optimizer state used to verify that a schedule value remains fixed across an epoch. -/
structure ProbeState where
  scheduleValue : Nat
  counters : ProbeCounters

def newProbeCounters : IO ProbeCounters := do
  pure {
    nativeSteps := ← IO.mkRef 0
    lossSteps := ← IO.mkRef 0
    genericSteps := ← IO.mkRef 0
    scheduledValues := ← IO.mkRef []
  }

def recordProbeStep (state : ProbeState) (counter : IO.Ref Nat) : IO Unit := do
  counter.modify (· + 1)
  state.counters.scheduledValues.modify (· ++ [state.scheduleValue])

/--
Optimizer that records native, loss-returning, and generic dispatch separately.

The no-loss hook completes the update itself. The loss hook returns `none` after recording the
attempt, allowing the normal same-tape fallback to produce the scalar when a test enables logging.
-/
def probeOptimizer (counters : ProbeCounters) :
    API.TorchLean.Optim.Optimizer Float (Supervised.paramShapes task) where
  State := ProbeState
  init := fun _ => pure { scheduleValue := 0, counters }
  step := fun state _ _ => do
    recordProbeStep state counters.genericSteps
    pure state
  trainerStep? := fun _ state _ => do
    recordProbeStep state counters.nativeSteps
    pure (some state)
  trainerStepWithLoss? := fun _ _state _ => do
    counters.lossSteps.modify (· + 1)
    pure none

def expectNat (label : String) (actual expected : Nat) : IO Unit :=
  unless actual == expected do
    throw <| IO.userError s!"{label}: got {actual}, expected {expected}"

def checkClosedFormMeanGradient : IO Unit := do
  let runner ← Supervised.instantiateConfiguredFloat task { backend := .compiled }
  let (weight, bias) := readLinearParams (← Supervised.params runner)
  let x₁ := 1.0
  let y₁ := 0.0
  let x₂ := 3.0
  let y₂ := 1.0
  let residual₁ := weight * x₁ + bias - y₁
  let residual₂ := weight * x₂ + bias - y₂
  let gradWeight := (2.0 * residual₁ * x₁ + 2.0 * residual₂ * x₂) / 2.0
  let gradBias := (2.0 * residual₁ + 2.0 * residual₂) / 2.0
  let lr := 0.1
  let _ ← Supervised.trainSamples runner
    { steps := 1
      batchSize := 2
      optimizer := .sgd lr
      logEvery := 0 }
    [sample x₁ y₁, sample x₂ y₂]
  let (weight', bias') := readLinearParams (← Supervised.params runner)
  unless close weight' (weight - lr * gradWeight) && close bias' (bias - lr * gradBias) do
    throw <| IO.userError <|
      s!"minibatch update mismatch: got ({weight'}, {bias'}), expected "
        ++ s!"({weight - lr * gradWeight}, {bias - lr * gradBias})"

/-- A batch size of one keeps the native no-loss route when progress logging is disabled. -/
def checkNoLossFastPath : IO Unit := do
  let runner ← Supervised.instantiateConfiguredFloat task { backend := .compiled }
  let counters ← newProbeCounters
  let opt := probeOptimizer counters
  let state ← API.TorchLean.Module.initOptim runner.module opt
  let batches ← IO.mkRef [[sample 1.0 0.0]]
  let nextBatch := do
    let remaining ← batches.get
    match remaining with
    | batch :: rest =>
        batches.set rest
        pure batch
    | [] =>
        throw <| IO.userError "no-loss probe exhausted"
  Supervised.Internal.runSampleSteps runner
    { steps := 1, batchSize := 1, logEvery := 0 }
    nextBatch (fun _ => pure ()) opt state (fun _ current => current)
  expectNat "native no-loss steps" (← counters.nativeSteps.get) 1
  expectNat "loss-returning steps" (← counters.lossSteps.get) 0
  expectNat "generic steps" (← counters.genericSteps.get) 0

/--
A loader with multi-sample batches keeps every update on the generic optimizer state, including a
final singleton batch.
-/
def checkPartialBatchStateRoute : IO Unit := do
  let runner ← Supervised.instantiateConfiguredFloat task { backend := .compiled }
  let counters ← newProbeCounters
  let opt := probeOptimizer counters
  let state ← API.TorchLean.Module.initOptim runner.module opt
  let loader : _root_.Runtime.Autograd.Train.DataLoader
      (API.TorchLean.TensorPack Float [.dim 1 .scalar, .dim 1 .scalar]) := {
    dataset := _root_.Runtime.Autograd.Train.Dataset.ofList
      [sample 1.0 0.0, sample 2.0 0.0, sample 3.0 0.0]
    batchSize := 2
  }
  let _ ← Supervised.Internal.runLoaderEpochs runner
    { epochs := 1, logEvery := 0 }
    loader opt state (fun _ current => current)
  expectNat "partial-loader native steps" (← counters.nativeSteps.get) 0
  expectNat "partial-loader generic steps" (← counters.genericSteps.get) 2

/-- Loader schedules advance once per epoch while logging keeps a global update counter. -/
def checkEpochSchedulerCadence : IO Unit := do
  let runner ← Supervised.instantiateConfiguredFloat task { backend := .compiled }
  let counters ← newProbeCounters
  let opt := probeOptimizer counters
  let state ← API.TorchLean.Module.initOptim runner.module opt
  let loader : _root_.Runtime.Autograd.Train.DataLoader
      (API.TorchLean.TensorPack Float [.dim 1 .scalar, .dim 1 .scalar]) := {
    dataset := _root_.Runtime.Autograd.Train.Dataset.ofList
      [sample 1.0 0.0, sample 2.0 0.0, sample 3.0 0.0, sample 4.0 0.0]
    batchSize := 2
  }
  let _ ← Supervised.Internal.runLoaderEpochs runner
    { epochs := 2, logEvery := 0 }
    loader opt state (fun epoch current => { current with scheduleValue := epoch })
  let observed ← counters.scheduledValues.get
  unless observed == [0, 0, 1, 1] do
    throw <| IO.userError
      s!"loader scheduler cadence: got {observed}, expected [0, 0, 1, 1]"

/--
BatchNorm buffers advance once per accumulated item, and requesting a logged loss does not apply a
second buffer update.
-/
def checkBatchNormBuffers : IO Unit := do
  let batch := [batchNormSample 2.0, batchNormSample 4.0]
  let runner ← Supervised.instantiateConfiguredFloat batchNormTask { backend := .compiled }
  Supervised.trainMode runner
  let opt := noOpOptimizer (Supervised.paramShapes batchNormTask)
  let state ← API.TorchLean.Module.initOptim runner.module opt
  let _ ← Supervised.Internal.stepBatch runner opt state false batch
  let noLossBuffers := readBatchNormBuffers (← Supervised.params runner)

  let loggedRunner ← Supervised.instantiateConfiguredFloat batchNormTask { backend := .compiled }
  Supervised.trainMode loggedRunner
  let loggedState ← API.TorchLean.Module.initOptim loggedRunner.module opt
  let _ ← Supervised.Internal.stepBatchAndLoss loggedRunner opt loggedState false batch
  let loggedBuffers := readBatchNormBuffers (← Supervised.params loggedRunner)

  unless close noLossBuffers.1 2.5 && close noLossBuffers.2 0.25 do
    throw <| IO.userError <|
      s!"BatchNorm buffers: got mean={noLossBuffers.1}, variance={noLossBuffers.2}; "
        ++ "expected mean=2.5, variance=0.25"
  unless close loggedBuffers.1 noLossBuffers.1 && close loggedBuffers.2 noLossBuffers.2 do
    throw <| IO.userError <|
      s!"BatchNorm logging changed buffer updates: no-loss={noLossBuffers}, logged={loggedBuffers}"

def run : IO Unit := do
  checkClosedFormMeanGradient
  checkNoLossFastPath
  checkPartialBatchStateRoute
  checkEpochSchedulerCadence
  checkBatchNormBuffers

end NN.Tests.API.GradientAccumulation
