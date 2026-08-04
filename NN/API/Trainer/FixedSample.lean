/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.CLI
public import NN.API.Trainer.Manual.Core
public import NN.API.Seeded
public import NN.Runtime.Training.Log

/-!
# Fixed-Sample Training

Some runnable examples train repeatedly on one caller-supplied sample:

1. build a model with `TorchLean.nn.withModel`,
2. wrap it as a `ScalarModuleDef` (model + supervised loss),
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
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α] [_root_.TorchLean.Runtime.FromFloat α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    {σ τ : Spec.Shape}
    (mkModel : TorchLean.nn.M (TorchLean.nn.Sequential σ τ))
    (mkModuleDef :
      (model : TorchLean.nn.Sequential σ τ) →
        TorchLean.Module.ScalarModuleDef (TorchLean.nn.paramShapes model) [σ, τ])
    (mkOptim :
      (cast : Float → α) → (paramShapes : List Spec.Shape) →
        _root_.Runtime.Autograd.TorchLean.Optim.Optimizer α paramShapes)
    (cast : Float → α)
    (opts : _root_.Runtime.Autograd.Torch.Options)
    (sample : TorchLean.Sample.Supervised α σ τ)
    (steps : Nat)
    (cudaMemWatch : Nat := 0) :
    IO (LossPair α) := do
  TorchLean.nn.withModel mkModel fun model => do
    let modDef := mkModuleDef model
    let m ← TorchLean.Module.instantiateConfigured (α := α) modDef cast opts
    let initialLossTensor ← TorchLean.Module.forward (α := α) m sample .nil
    let beforeLoss := _root_.Spec.Tensor.toScalar initialLossTensor
    let opt := mkOptim cast (TorchLean.nn.paramShapes model)
    let optH ← _root_.Runtime.Autograd.TorchLean.Module.optimizerHandle (α := α) m opt
    let watchEvery := TorchLean.Trainer.Manual.effectiveCudaMemWatch opts steps cudaMemWatch
    let mut memWatch? ← TorchLean.Trainer.Manual.reportCudaMemWatch opts watchEvery steps 0 none
    for step in [0:steps] do
      optH.step sample
      memWatch? ← TorchLean.Trainer.Manual.reportCudaMemWatch opts watchEvery steps (step + 1) memWatch?
    let finalLossTensor ← TorchLean.Module.forward (α := α) m sample .nil
    let afterLoss := _root_.Spec.Tensor.toScalar finalLossTensor
    pure { beforeLoss := beforeLoss, afterLoss := afterLoss }

/-- Fixed-sample run specialized to `Float`, returning a full per-step curve. -/
def curveFloat
    {σ τ : Spec.Shape}
    (mkModel : TorchLean.nn.M (TorchLean.nn.Sequential σ τ))
    (mkModuleDef :
      (model : TorchLean.nn.Sequential σ τ) →
        TorchLean.Module.ScalarModuleDef (TorchLean.nn.paramShapes model) [σ, τ])
    (mkOptim :
      (paramShapes : List Spec.Shape) → _root_.Runtime.Autograd.TorchLean.Optim.Optimizer Float paramShapes)
    (opts : _root_.Runtime.Autograd.Torch.Options)
    (sample : TorchLean.Sample.Supervised Float σ τ)
    (steps : Nat)
    (cudaMemWatch : Nat := 0) :
    IO _root_.Runtime.Training.Curve := do
  TorchLean.nn.withModel mkModel fun model => do
    let modDef := mkModuleDef model
    let m ← TorchLean.Module.instantiateConfigured (α := Float) modDef id opts
    let initialLossTensor ← TorchLean.Module.forward (α := Float) m sample .nil
    let initialLoss := _root_.Spec.Tensor.toScalar initialLossTensor
    let opt := mkOptim (TorchLean.nn.paramShapes model)
    let optH ← _root_.Runtime.Autograd.TorchLean.Module.optimizerHandle (α := Float) m opt
    let mut curve : _root_.Runtime.Training.Curve := {}
    curve := curve.push 0 initialLoss
    let mut last := initialLoss
    let watchEvery := TorchLean.Trainer.Manual.effectiveCudaMemWatch opts steps cudaMemWatch
    let mut memWatch? ← TorchLean.Trainer.Manual.reportCudaMemWatch opts watchEvery steps 0 none
    for step in [0:steps] do
      optH.step sample
      memWatch? ← TorchLean.Trainer.Manual.reportCudaMemWatch opts watchEvery steps (step + 1) memWatch?
      let loss ← TorchLean.Module.forward (α := Float) m sample .nil
      last := _root_.Spec.Tensor.toScalar loss
      curve := curve.push (step + 1) last
    pure curve

end FixedSample
end Trainer
end TorchLean
