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

/-! ## Linear algebra and concatenation -/

/-- 2D matrix multiplication. PyTorch: `torch.matmul` for 2D tensors. -/
def matmul {α : Type} (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {m n p : Nat}
  (a : TensorRef α (.dim m (.dim n .scalar)))
  (b : TensorRef α (.dim n (.dim p .scalar))) :
  IO (TensorRef α (.dim m (.dim p .scalar))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ←
      okOrThrow (Runtime.Autograd.Tape.matmul (t := t0) (m := m) (n := n) (p := p) a.id b.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.matmul (t := t0) (m := m) (n := n) (p := p) a.id b.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .matmul cpu cuda

/-- Batched matrix multiplication. PyTorch: `torch.bmm`. -/
def bmm {α : Type} (s : EagerSession α) [Add α] [Mul α] [Zero α] [DecidableEq Shape]
  {batch m n p : Nat}
  (a : TensorRef α (.dim batch (.dim m (.dim n .scalar))))
  (b : TensorRef α (.dim batch (.dim n (.dim p .scalar)))) :
  IO (TensorRef α (.dim batch (.dim m (.dim p .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.bmm (α := α) (t := t0) (batch := batch) (m := m)
      (n := n) (p := p) a.id b.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.bmm (t := t0) (batch := batch) (m := m) (n := n) (p := p) a.id b.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .matmul cpu cuda

/-- Concatenate two vectors along dim 0. PyTorch: `torch.cat([a,b], dim=0)`. -/
def concatVectors {α : Type} (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {n m : Nat}
  (a : TensorRef α (.dim n .scalar))
  (b : TensorRef α (.dim m .scalar)) :
  IO (TensorRef α (.dim (n + m) .scalar)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.concatVectors (t := t0) (n := n) (m := m) a.id
      b.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.concatVectors (t := t0) (n := n) (m := m) a.id b.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .concat cpu cuda

/-- Concatenate along dim 0 for tensors with leading dimension. PyTorch: `torch.cat(..., dim=0)`. -/
def concatLeadingAxis {α : Type} (s : EagerSession α) [DecidableEq Shape]
  {n m : Nat} {sh : Shape}
  (a : TensorRef α (.dim n sh))
  (b : TensorRef α (.dim m sh)) :
  IO (TensorRef α (.dim (n + m) sh)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.concatLeadingAxis (α := α) (t := t0) (n := n) (m := m)
      (s := sh) a.id b.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.concatLeadingAxis (t := t0) (n := n) (m := m) (s := sh) a.id b.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .concat cpu cuda

/-- Slice along dim 0: `x[start:start+len]`. PyTorch: standard slicing. -/
def sliceLeadingAxisRange {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape]
  {n : Nat} {sh : Shape}
  (x : TensorRef α (.dim n sh)) (start len : Nat) (h : len + start ≤ n) :
  IO (TensorRef α (.dim len sh)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.sliceLeadingAxisRange (α := α) (t := t0) (n := n) (s := sh)
      x.id start len h)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.sliceLeadingAxisRange (t := t0) (n := n) (s := sh) x.id start len h
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .slice cpu cuda

end EagerSession

end Internal
end Torch
end Autograd
end Runtime
