/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Trainer.Attention

/-!
# Trainer Parameters

Mutable, shape-indexed parameter storage for the Torch-style trainer. Parameter values may retain a
CUDA mirror; reads and writes keep host and device ownership explicit through the underlying
`Param` operations.
-/

@[expose] public section

namespace Runtime.Autograd.Torch

open Spec Tensor Proofs.Autograd.Algebra

/-- A heterogeneous list of mutable parameters indexed by their tensor shapes. -/
inductive ParamList (α : Type) : List Shape → Type where
  | nil : ParamList α []
  | cons {s : Shape} {ss : List Shape} : Param α s → ParamList α ss → ParamList α (s :: ss)

namespace ParamList

/-- Materialize `value - rate * gradient` in one traversal. -/
def subScaleMaterialize {α : Type} [Sub α] [Mul α] :
    {s : Shape} → Tensor α s → Tensor α s → α → Tensor α s
  | .scalar, .scalar value, .scalar gradient, rate =>
      Tensor.scalar (value - rate * gradient)
  | .dim n shape, .dim values, .dim gradients, rate =>
      let entries : Array (Tensor α shape) := Array.ofFn fun i : Fin n =>
        subScaleMaterialize (s := shape) (values i) (gradients i) rate
      Tensor.dim fun i =>
        let hsize : entries.size = n := by
          simp [entries]
        let hindex : i.1 < entries.size :=
          Eq.ndrec (motive := fun m => i.1 < m) i.2 hsize.symm
        entries[i.1]'hindex

/-- Allocate mutable parameters from an ordered tensor pack. -/
def ofTList {α : Type} : {ss : List Shape} → TList α ss → IO (ParamList α ss)
  | [], .nil => pure .nil
  | _ :: _, .cons value values => do
      let hostValue ← IO.mkRef value
      let cudaValue ← IO.mkRef (none : Option Runtime.Autograd.Cuda.AnyBuffer)
      let hostCurrent ← IO.mkRef true
      let param : Param α _ := { value := hostValue, cudaValue, hostCurrent }
      pure (.cons param (← ofTList (α := α) values))

/--
Allocate mutable parameters with an explicit trainability mask.

The mask follows parameter order and must have exactly the same length as the shape list.
-/
def ofTListWithRequiresGrad {α : Type} :
    {ss : List Shape} → TList α ss → List Bool → IO (ParamList α ss)
  | [], .nil, [] => pure .nil
  | _ :: shapes, .cons value values, requiresGrad :: rest => do
      let hostValue ← IO.mkRef value
      let cudaValue ← IO.mkRef (none : Option Runtime.Autograd.Cuda.AnyBuffer)
      let hostCurrent ← IO.mkRef true
      let param : Param α _ := { value := hostValue, cudaValue, hostCurrent, requiresGrad }
      pure (.cons param (← ofTListWithRequiresGrad (α := α) (ss := shapes) values rest))
  | [], .nil, _ =>
      throw <| IO.userError "torch: requiresGrad list longer than parameter list"
  | _ :: _, .cons _ _, [] =>
      throw <| IO.userError "torch: requiresGrad list shorter than parameter list"

/-- Read the trainability mask in parameter order. -/
def requiresGradList {α : Type} : {ss : List Shape} → ParamList α ss → List Bool
  | [], .nil => []
  | _ :: _, .cons param params => param.requiresGrad :: requiresGradList params

/-- Read current host parameter values without synchronizing stale CUDA mirrors. -/
def values {α : Type} : {ss : List Shape} → ParamList α ss → IO (TList α ss)
  | [], .nil => pure .nil
  | _ :: shapes, .cons param params => do
      pure (.cons (← param.value.get) (← values (α := α) (ss := shapes) params))

/-- Read parameter values after synchronizing any current CUDA mirrors to the host. -/
def valuesSynced {α : Type} [Internal.CudaBridge.TensorConv α] [DecidableEq Shape] :
    {ss : List Shape} → ParamList α ss → IO (TList α ss)
  | [], .nil => pure .nil
  | shape :: shapes, .cons param params => do
      Internal.syncParamCudaToHost (α := α) (sh := shape) param
      pure (.cons (← param.value.get) (← valuesSynced (α := α) (ss := shapes) params))

/-- Replace host parameter values from an ordered tensor pack. -/
def setValues {α : Type} : {ss : List Shape} → ParamList α ss → TList α ss → IO Unit
  | [], .nil, .nil => pure ()
  | shape :: shapes, .cons param params, .cons value values => do
      Internal.setParamHostValue (α := α) (sh := shape) param value
      setValues (α := α) (ss := shapes) params values

/-- Apply `param := param - rate * gradient` to every trainable parameter. -/
def sgdStep {α : Type} [Context α] :
    {ss : List Shape} → ParamList α ss → α → TList α ss → IO Unit
  | [], .nil, _, .nil => pure ()
  | shape :: shapes, .cons param params, rate, .cons gradient gradients => do
      if param.requiresGrad then
        let value ← param.value.get
        let updated := subScaleMaterialize (s := shape) value gradient rate
        Internal.setParamHostValue (α := α) (sh := shape) param updated
      sgdStep (α := α) (ss := shapes) params rate gradients

end ParamList
end Runtime.Autograd.Torch
