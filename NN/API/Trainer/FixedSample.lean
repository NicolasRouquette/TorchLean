/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.CLI
public import NN.API.Module
public import NN.API.Trainer.Memory
public import NN.API.Seeded
public import NN.Runtime.Training.Log

/-!
# Fixed-Sample Training

Some runnable examples train repeatedly on one caller-supplied sample:

1. build a model with `TorchLean.nn.withModel`,
2. wrap it as an `ObjectiveDef` (model + supervised loss),
3. load or synthesize one supervised sample `(x, y)`,
4. run `steps` optimizer updates on that fixed sample, and
5. either print before/after loss or write a TrainLog curve.

This module provides that loop without tying it to a particular model family.

Scope:
- it trains against one fixed sample supplied by the caller;
- it is model-agnostic: callers supply the loss wrapper and optimizer constructor;
- it is backend-agnostic: callers can use it on CPU or CUDA via `API.Runtime.Options`.

For dataset-backed training, use the `TorchLean.Trainer` API exported by `NN` or the shared model-zoo
loader helpers.
-/

@[expose] public section

namespace TorchLean

open Spec Tensor

namespace Trainer
namespace FixedSample

/-- Before/after scalar losses for a fixed-sample training run. -/
structure LossPair (α : Type) where
  beforeLoss : α
  afterLoss : α
deriving Repr

/-- One fixed-sample run for an arbitrary scalar backend. -/
def steps
    {α : Type} [_root_.Context α] [DecidableEq Shape] [ToString α] [_root_.TorchLean.Runtime.FromFloat α]
    [Runtime.TensorTransfer α]
    {inputShape targetShape : List Nat}
    (mkModel : TorchLean.nn.Builder
      (TorchLean.nn.Sequential inputShape targetShape))
    (mkModuleDef :
      (model : TorchLean.nn.Sequential inputShape targetShape) →
        TorchLean.Module.ObjectiveDef Unit (TorchLean.nn.stateShapes model)
          [Shape.ofList inputShape, Shape.ofList targetShape])
    (mkOptim :
      (cast : Float → α) → (paramShapes : List Shape) →
        _root_.Runtime.Autograd.TorchLean.Optim.Optimizer α paramShapes)
    (cast : Float → α)
    (opts : _root_.Runtime.Autograd.Torch.Options)
    (sample : TorchLean.Sample.Supervised α inputShape targetShape)
    (steps : Nat)
    (cudaMemWatch : Nat := 0) :
    IO (LossPair α) := do
  TorchLean.nn.withModel mkModel fun model => do
    let modDef := mkModuleDef model
    let m ← TorchLean.Module.instantiateAs (α := α) modDef cast opts
    let initialLossTensor ← TorchLean.Module.loss (α := α) m sample .nil
    let beforeLoss := _root_.Spec.Tensor.item initialLossTensor
    let opt := mkOptim cast (TorchLean.nn.stateShapes model)
    let bound ← _root_.Runtime.Autograd.TorchLean.Module.bindOptimizer (α := α) m opt
    let watchEvery := TorchLean.Trainer.Manual.CUDAMemory.cadence opts steps cudaMemWatch
    let mut memWatch? ← TorchLean.Trainer.Manual.CUDAMemory.sample opts watchEvery steps 0 none
    for step in [0:steps] do
      bound.step sample
      memWatch? ←
        TorchLean.Trainer.Manual.CUDAMemory.sample
          opts watchEvery steps (step + 1) memWatch?
    let finalLossTensor ← TorchLean.Module.loss (α := α) m sample .nil
    let afterLoss := _root_.Spec.Tensor.item finalLossTensor
    pure { beforeLoss := beforeLoss, afterLoss := afterLoss }

/-- Fixed-sample run over Lean's `Float`, returning a full per-step loss curve. -/
def curve
    {inputShape targetShape : List Nat}
    (mkModel : TorchLean.nn.Builder
      (TorchLean.nn.Sequential inputShape targetShape))
    (mkModuleDef :
      (model : TorchLean.nn.Sequential inputShape targetShape) →
        TorchLean.Module.ObjectiveDef Unit (TorchLean.nn.stateShapes model)
          [Shape.ofList inputShape, Shape.ofList targetShape])
    (mkOptim :
      (paramShapes : List Shape) → _root_.Runtime.Autograd.TorchLean.Optim.Optimizer Float paramShapes)
    (opts : _root_.Runtime.Autograd.Torch.Options)
    (sample : TorchLean.Sample.Supervised Float inputShape targetShape)
    (steps : Nat)
    (cudaMemWatch : Nat := 0) :
    IO _root_.Runtime.Training.Curve := do
  TorchLean.nn.withModel mkModel fun model => do
    let modDef := mkModuleDef model
    let m ← TorchLean.Module.instantiateAs (α := Float) modDef id opts
    let initialLossTensor ← TorchLean.Module.loss (α := Float) m sample .nil
    let initialLoss := _root_.Spec.Tensor.item initialLossTensor
    let opt := mkOptim (TorchLean.nn.stateShapes model)
    let bound ← _root_.Runtime.Autograd.TorchLean.Module.bindOptimizer (α := Float) m opt
    let mut curve : _root_.Runtime.Training.Curve := {}
    curve := curve.push 0 initialLoss
    let mut last := initialLoss
    let watchEvery := TorchLean.Trainer.Manual.CUDAMemory.cadence opts steps cudaMemWatch
    let mut memWatch? ← TorchLean.Trainer.Manual.CUDAMemory.sample opts watchEvery steps 0 none
    for step in [0:steps] do
      bound.step sample
      memWatch? ←
        TorchLean.Trainer.Manual.CUDAMemory.sample
          opts watchEvery steps (step + 1) memWatch?
      let loss ← TorchLean.Module.loss (α := Float) m sample .nil
      last := _root_.Spec.Tensor.item loss
      curve := curve.push (step + 1) last
    pure curve

end FixedSample
end Trainer
end TorchLean
