/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.ScalarTrainer
public import NN.Runtime.Autograd.TorchLean.Optim

/-!
# TorchLean training-loop helpers

This module contains training-loop utilities that need the high-level `TorchLean.Optim` optimizer
interface. The lower-level `Runtime.Autograd.Torch.Utils` file stops at
`trainCycleSGD`, because that helper only depends on the `ScalarTrainer` update bundled by the
low-level session layer.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor

/--
Train `steps` updates with an arbitrary TorchLean optimizer, cycling through `samples`.

PyTorch comparison: analogous to using a `torch.optim.Optimizer` and calling
`loss.backward(); opt.step()` in a loop, except here `opt.step` consumes an explicit gradient
`_root_.TorchLean.TensorPack` aligned with `paramShapes`.
-/
def trainCycleOptim
    {α : Type} [Context α] [ToString α]
    {paramShapes inputShapes : List Shape}
    (tr : _root_.Runtime.Autograd.Torch.ScalarTrainer α Unit paramShapes inputShapes)
    (opt : Optim.Optimizer α paramShapes)
    (st0 : opt.State)
    (steps : Nat) (samples : Array (TorchLean.TensorPack α inputShapes))
    (logEvery : Nat := 1) : IO opt.State := do
  match samples[0]? with
  | none =>
      throw <| IO.userError "trainCycleOptim: empty dataset"
  | some first =>
      let mut st := st0
      for step in [0:steps] do
        let xs := samples.getD (step % samples.size) first
        let (st', lossTensor) ←
          match ← opt.trainerStepWithLoss? tr st xs .nil with
          | some result =>
              pure result
          | none => do
              let (lossTensor, grads) ←
                _root_.Runtime.Autograd.Torch.ScalarTrainer.runLossAndGradState
                  (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes) tr xs
                  .nil
              let _ ← tr.getState
              let st' ← opt.step st tr.state grads
              pure (st', lossTensor)
        st := st'
        if logEvery != 0 && step % logEvery = 0 then
          IO.println s!"step {step}: loss={lossTensor.item}"
      pure st

end TorchLean
end Autograd
end Runtime
