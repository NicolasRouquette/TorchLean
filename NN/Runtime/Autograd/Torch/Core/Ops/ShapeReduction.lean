/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Ops.Dispatch

/-!
# Eager Tensor Operations

PyTorch-style tensor operations backed by the eager CPU/CUDA tapes. These wrappers record runtime
nodes, dispatch CUDA kernels when requested, and preserve the typed `TensorRef` surface.
-/


@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec
open Tensor
open Proofs.Autograd.Algebra

namespace Internal

namespace EagerSession

/-! ## Shape and reduction operations -/

/-- Sum-reduce all elements to a scalar. PyTorch: `x.sum()`. -/
def sum {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α Shape.scalar) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.sum (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.sum (t := t0) (s := sh) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .reduceSum cpu cuda

/-- Flatten a tensor to a 1D vector. PyTorch: `torch.flatten`. -/
def flatten {α : Type} (s : EagerSession α) [Inhabited α] [DecidableEq Shape] {sh : Shape}
  (x : TensorRef α sh) : IO (TensorRef α (.dim (Spec.Shape.size sh) .scalar)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.flatten (t := t0) (s := sh) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.flatten (t := t0) (s := sh) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .reshape cpu cuda

/--
Reshape a tensor while preserving total number of elements.

PyTorch comparison: `torch.reshape` / `view` (when valid).
-/
def reshape {α : Type} (s : EagerSession α) [Inhabited α] [DecidableEq Shape] {sh1 sh2 : Shape}
  (x : TensorRef α sh1) (h : Spec.Shape.size sh1 = Spec.Shape.size sh2) : IO (TensorRef α sh2) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.reshape (t := t0) (s₁ := sh1) (s₂ := sh2) x.id h)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.reshape (t := t0) (s₁ := sh1) (s₂ := sh2) x.id h
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .reshape cpu cuda

/-- Transpose a 2D matrix. PyTorch: `x.t()` / `x.transpose(0,1)`. -/
def transpose2d {α : Type} (s : EagerSession α) [DecidableEq Shape] {m n : Nat}
  (x : TensorRef α (.dim m (.dim n .scalar))) : IO (TensorRef α (.dim n (.dim m .scalar))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.transpose2d (t := t0) (m := m) (n := n) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.transpose2d (t := t0) (m := m) (n := n) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .permute cpu cuda

/-- Swap two adjacent axes at a given depth. PyTorch analogue: `x.transpose(dim, dim+1)`. -/
def swapAdjacentAtDepth {α : Type} (s : EagerSession α) [DecidableEq Shape] {sh : Shape}
  (depth : Nat) (x : TensorRef α sh) : IO (TensorRef α (sh.swapAdjacentAtDepth depth)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.swapAdjacentAtDepth (t := t0) (s := sh) depth
      x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.swapAdjacentAtDepth (t := t0) (s := sh) depth x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .permute cpu cuda

/-- Permute a 3D tensor `(a,b,c) → (b,c,a)`. PyTorch: `x.permute(1,2,0)`. -/
def transpose3dFirstToLast {α : Type} (s : EagerSession α) [DecidableEq Shape] {a b c : Nat}
  (x : TensorRef α (.dim a (.dim b (.dim c .scalar)))) :
  IO (TensorRef α (.dim b (.dim c (.dim a .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.transpose3dFirstToLast (t := t0) (a := a) (b :=
      b) (c := c) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.transpose3dFirstToLast (t := t0) (a := a) (b := b) (c := c) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .permute cpu cuda

/-- Permute a 3D tensor `(a,b,c) → (c,a,b)`. PyTorch: `x.permute(2,0,1)`. -/
def transpose3dLastToFirst {α : Type} (s : EagerSession α) [DecidableEq Shape] {a b c : Nat}
  (x : TensorRef α (.dim a (.dim b (.dim c .scalar)))) :
  IO (TensorRef α (.dim c (.dim a (.dim b .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.transpose3dLastToFirst (t := t0) (a := a) (b :=
      b) (c := c) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.transpose3dLastToFirst (t := t0) (a := a) (b := b) (c := c) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .permute cpu cuda

/-- Swap the last two axes of a 3D tensor `(a,b,c) → (a,c,b)`. PyTorch: `x.transpose(1,2)`. -/
def transpose3dLastTwo {α : Type} (s : EagerSession α) [DecidableEq Shape] {a b c : Nat}
  (x : TensorRef α (.dim a (.dim b (.dim c .scalar)))) :
  IO (TensorRef α (.dim a (.dim c (.dim b .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.transpose3dLastTwo (t := t0) (a := a) (b := b)
      (c := c) x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.transpose3dLastTwo (t := t0) (a := a) (b := b) (c := c) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .permute cpu cuda

/-- Broadcast a tensor to a larger shape. PyTorch: implicit broadcasting / `expand`. -/
def broadcastTo {α : Type} (s : EagerSession α) [Inhabited α] [Add α] [Zero α] [DecidableEq Shape]
  {sh1 sh2 : Shape} (cb : Shape.CanBroadcastTo sh1 sh2) (x : TensorRef α sh1) : IO (TensorRef α sh2)
    := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.broadcastTo (α := α) (t := t0) (s₁ := sh1) (s₂ :=
      sh2) cb x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.broadcastTo (t := t0) (s₁ := sh1) (s₂ := sh2) cb x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .broadcast cpu cuda

/-- Sum-reduce along `axis`. PyTorch: `torch.sum(x, dim=axis)`. -/
def reduceSum {α : Type} (s : EagerSession α) [Add α] [Zero α] [Inhabited α] [DecidableEq Shape]
  {sh : Shape} (axis : Nat) [valid : Shape.HasNonemptyAxis axis sh] [wf : Shape.WellFormed sh]
  (x : TensorRef α sh) : IO (TensorRef α (shapeAfterSum sh axis)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.reduceSum (t := t0) (s := sh) axis x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.reduceSum (s := sh) axis (t := t0) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .reduceSum cpu cuda

/-- Mean-reduce along `axis`. PyTorch: `torch.mean(x, dim=axis)`. -/
def reduceMean {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (axis : Nat) [valid : Shape.HasNonemptyAxis axis sh] [wf : Shape.WellFormed sh]
  (x : TensorRef α sh) : IO (TensorRef α (shapeAfterSum sh axis)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.reduceMean (t := t0) (s := sh) axis x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.reduceMean (s := sh) axis (t := t0) x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .reduceMean cpu cuda

end EagerSession

end Internal
end Torch
end Autograd
end Runtime
