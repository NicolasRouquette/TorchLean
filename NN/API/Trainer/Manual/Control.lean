/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Trainer.Manual.Evaluation
public import NN.API.Trainer.Memory

/-!
# Manual Training Control

Lifecycle events, composable callbacks, CUDA memory callbacks, and scoped runner mode changes for
manual training loops.
-/

@[expose] public section

namespace TorchLean
namespace Trainer
namespace Manual

/-- Information reported after one optimizer step. -/
structure StepEvent (α : Type) where
  /-- Current epoch number. -/
  epoch : Nat
  /-- Global optimizer-step counter. -/
  step : Nat
  /-- Loss reported for this step. -/
  loss : α

/-- Information reported after one epoch. -/
structure EpochEvent where
  /-- Epoch number that just completed. -/
  epoch : Nat
  /-- Number of steps executed in the epoch. -/
  steps : Nat

/-- Stop a training loop after a number of complete epochs or optimizer steps. -/
inductive TrainingLimit where
  /-- Run the requested number of complete loader epochs. -/
  | epochs (count : Nat)
  /-- Run exactly the requested number of optimizer steps. -/
  | steps (count : Nat)
deriving Repr

/-- Hooks run at the lifecycle boundaries of a manual training loop. -/
structure Callbacks (α : Type) where
  /-- Called once before training starts. -/
  onTrainStart : IO Unit := pure ()
  /-- Called after each training step. -/
  onStep : StepEvent α → IO Unit := fun _ => pure ()
  /-- Called after each epoch. -/
  onEpochEnd : EpochEvent → IO Unit := fun _ => pure ()
  /-- Called once after training finishes. -/
  onTrainEnd : TrainReport α → IO Unit := fun _ => pure ()

namespace Callbacks

/-- Compose callback collections in execution order. -/
def append {α : Type} (first second : Callbacks α) : Callbacks α :=
  { onTrainStart := do
      first.onTrainStart
      second.onTrainStart
    onStep := fun event => do
      first.onStep event
      second.onStep event
    onEpochEnd := fun event => do
      first.onEpochEnd event
      second.onEpochEnd event
    onTrainEnd := fun report => do
      first.onTrainEnd report
      second.onTrainEnd report }

instance {α : Type} : Append (Callbacks α) where
  append := append

/-- Sample CUDA allocator state at a fixed completed-step cadence. -/
def cudaMemory {α : Type} (opts : _root_.Runtime.Autograd.Torch.Options)
    (watchEvery totalSteps : Nat) : IO (Callbacks α) := do
  let stateRef ← IO.mkRef (none : Option CUDAMemory.State)
  pure
    { onStep := fun event => do
        let state ← stateRef.get
        let state ← CUDAMemory.sample opts watchEvery totalSteps (event.step + 1) state
        stateRef.set state }

end Callbacks

end Manual
end Trainer
end TorchLean
