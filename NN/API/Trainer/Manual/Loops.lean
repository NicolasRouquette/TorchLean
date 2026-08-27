/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Trainer.Memory
public import NN.API.Trainer.Manual.Evaluation
public import NN.API.Trainer.Manual.Execution

@[expose] public section

namespace TorchLean
namespace Trainer
namespace Manual

open SeqTask

/-!
# Runtime Supervised Training

Fixed-step training over finite sample streams with gradient accumulation.
-/

namespace Internal

/-- Take the next cyclic group from a finite stream, evaluating only the requested samples. -/
def nextCyclicStreamBatch {α : Type} (context : String)
    (samples : _root_.TorchLean.Data.SampleStream α)
    (cursorRef : IO.Ref Nat) (batchSize : Nat) : IO (Array α) := do
  if hEmpty : samples.size = 0 then
    throw <| IO.userError s!"{context}: empty sample stream"
  else
    have hSize : 0 < samples.size := Nat.pos_of_ne_zero hEmpty
    let mut cursor ← cursorRef.get
    let mut batch : Array α := #[]
    for _ in [0:batchSize] do
      let i := cursor % samples.size
      batch := batch.push (samples.get ⟨i, Nat.mod_lt _ hSize⟩)
      cursor := cursor + 1
    cursorRef.set cursor
    pure batch

end Internal

/-! ## Finite sample streams -/

/--
Train on a finite sample stream for a fixed number of optimizer updates.

`cfg.batchSize` controls how many dataset items contribute to each update. It does not multiply the
number of optimizer steps. If an item is already a batched tensor, use `batchSize := 1`
unless you intend to accumulate gradients across several such minibatches.
-/
def trainFinite {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    [_root_.TorchLean.Runtime.FromFloat α]
    (runner : Runner α task)
    (cfg : TrainConfig)
    (samples : _root_.TorchLean.Data.SampleStream
      (_root_.TorchLean.TensorPack α [σ, τ])) :
    IO (TrainReport α) := do
  Runner.train runner
  let before ← Runner.meanLossStream runner samples
  let watchEvery := CUDAMemory.cadence runner.module.opts cfg.steps cfg.cudaMemWatch
  let memWatchRef ← IO.mkRef (none : Option CUDAMemory.State)
  let watchMemory (done : Nat) : IO Unit := do
    let state ← memWatchRef.get
    let state ← CUDAMemory.sample runner.module.opts watchEvery cfg.steps done state
    memWatchRef.set state
  watchMemory 0
  unless samples.isEmpty do
    let cursorRef ← IO.mkRef 0
    let nextBatch :=
      Internal.nextCyclicStreamBatch "Manual.trainFinite" samples cursorRef
        (Internal.effectiveTrainBatchSize cfg.batchSize)
    Internal.withBoundOptimizer runner cfg.optimizer cfg.scheduler fun opt state scheduleState =>
      Internal.runSampleSteps runner cfg nextBatch watchMemory opt state scheduleState
  let after ← Runner.meanLossStream runner samples
  pure { before := before, after := after }

/-- Train on an in-memory array by exposing it as a finite sample stream. -/
def trainSamples {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    [_root_.TorchLean.Runtime.FromFloat α]
    (runner : Runner α task)
    (cfg : TrainConfig)
    (samples : Array (_root_.TorchLean.TensorPack α [σ, τ])) :
    IO (TrainReport α) :=
  trainFinite runner cfg (_root_.TorchLean.Data.SampleStream.ofArray samples)

end Manual
end Trainer
end TorchLean
