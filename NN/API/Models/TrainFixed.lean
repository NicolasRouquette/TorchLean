/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public

/-!
# Fixed-Sample Training Helpers (API)

Many runnable examples in `NN/Examples/Models/*` follow the same pattern:

1. build a model with `nn.withModel`,
2. wrap it as a `ScalarModuleDef` (model + supervised loss),
3. load or synthesize one supervised sample `(x, y)`,
4. run `steps` optimizer updates on that fixed sample, and
5. either print `loss0 -> loss1` or write a TrainLog curve.

This module keeps that loop in one place so examples stay short and consistent.

Scope:
- it trains against one fixed sample supplied by the caller;
- it is model-agnostic: callers supply the loss wrapper and optimizer constructor;
- it is backend-agnostic: callers can use it on CPU or CUDA via `TorchLean.Options`.

For dataset-backed training, use the loader/stream helpers in `NN.API.train`.
-/

@[expose] public section

namespace NN
namespace API

open Spec Tensor

namespace Models
namespace TrainFixed

/-- Before/after scalar losses for a fixed-sample training run. -/
structure LossPair (α : Type) where
  loss0 : α
  loss1 : α
deriving Repr

/-- One fixed-sample run for an arbitrary scalar backend. -/
def steps
    {α : Type} [Semantics.Scalar α] [DecidableEq Shape] [ToString α] [Runtime.Scalar α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    {σ τ : Shape}
    (mkModel : nn.M (nn.Sequential σ τ))
    (mkModuleDef :
      (model : nn.Sequential σ τ) →
        TorchLean.Module.ScalarModuleDef (nn.paramShapes model) [σ, τ])
    (mkOptim :
      (cast : Float → α) → (paramShapes : List Shape) → TorchLean.Optim.Optimizer α paramShapes)
    (cast : Float → α)
    (opts : TorchLean.Options)
    (sample : sample.Supervised α σ τ)
    (steps : Nat)
    (cudaMemWatch : Nat := 0) :
    IO (LossPair α) := do
  nn.withModel mkModel fun model => do
    let modDef := mkModuleDef model
    let m ← TorchLean.Module.instantiateWithOptions (α := α) modDef cast opts
    let loss0 ← TorchLean.Module.forward (α := α) m sample
    let L0 := _root_.Spec.Tensor.toScalar loss0
    let opt := mkOptim cast (nn.paramShapes model)
    let optH ← TorchLean.Optim.handle (α := α) m opt
    let watchEvery := Common.effectiveCudaMemWatch opts steps cudaMemWatch
    let mut memWatch? ← Common.reportCudaMemWatch opts watchEvery steps 0 none
    for step in [0:steps] do
      optH.step sample
      memWatch? ← Common.reportCudaMemWatch opts watchEvery steps (step + 1) memWatch?
    let loss1 ← TorchLean.Module.forward (α := α) m sample
    let L1 := _root_.Spec.Tensor.toScalar loss1
    pure { loss0 := L0, loss1 := L1 }

/-- Fixed-sample run specialized to `Float`, returning a full per-step curve. -/
def curveFloat
    {σ τ : Shape}
    (mkModel : nn.M (nn.Sequential σ τ))
    (mkModuleDef :
      (model : nn.Sequential σ τ) →
        TorchLean.Module.ScalarModuleDef (nn.paramShapes model) [σ, τ])
    (mkOptim :
      (paramShapes : List Shape) → TorchLean.Optim.Optimizer Float paramShapes)
    (opts : TorchLean.Options)
    (sample : sample.Supervised Float σ τ)
    (steps : Nat)
    (cudaMemWatch : Nat := 0) :
    IO _root_.Runtime.Training.Curve := do
  nn.withModel mkModel fun model => do
    let modDef := mkModuleDef model
    let m ← TorchLean.Module.instantiateWithOptions (α := Float) modDef id opts
    let loss0 ← TorchLean.Module.forward (α := Float) m sample
    let L0 := _root_.Spec.Tensor.toScalar loss0
    let opt := mkOptim (nn.paramShapes model)
    let optH ← TorchLean.Optim.handle (α := Float) m opt
    let mut curve : _root_.Runtime.Training.Curve := {}
    curve := curve.push 0 L0
    let mut last := L0
    let watchEvery := Common.effectiveCudaMemWatch opts steps cudaMemWatch
    let mut memWatch? ← Common.reportCudaMemWatch opts watchEvery steps 0 none
    for step in [0:steps] do
      optH.step sample
      memWatch? ← Common.reportCudaMemWatch opts watchEvery steps (step + 1) memWatch?
      let loss ← TorchLean.Module.forward (α := Float) m sample
      last := _root_.Spec.Tensor.toScalar loss
      curve := curve.push (step + 1) last
    pure curve

end TrainFixed
end Models

end API
end NN
