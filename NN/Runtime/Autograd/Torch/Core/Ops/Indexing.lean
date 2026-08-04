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

/-! ## Indexing operations -/

/-- Gather a scalar from a 1D vector with a `Fin n` index. PyTorch: `x[i]`. -/
def gatherScalar {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape]
  {n : Nat} (x : TensorRef α (.dim n .scalar)) (i : Fin n) : IO (TensorRef α Shape.scalar) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.gatherScalar (t := t0) (n := n) x.id i)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.gatherScalar (t := t0) (n := n) x.id i
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .gather cpu cuda

/-- Gather a row from a 2D tensor with a `Fin rows` index. PyTorch: `x[i]` for 2D tensors. -/
def gatherRow {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape]
  {rows cols : Nat} (x : TensorRef α (.dim rows (.dim cols .scalar))) (i : Fin rows) :
  IO (TensorRef α (.dim cols .scalar)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.gatherRow (t := t0) (rows := rows) (cols := cols)
      x.id i)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.gatherRow (t := t0) (rows := rows) (cols := cols) x.id i
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .gather cpu cuda

/-- Gather a scalar from a 1D vector with a raw `Nat` index (totalized by the tape op). -/
def gatherScalarNat {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape]
  {n : Nat} (x : TensorRef α (.dim n .scalar)) (i : Nat) : IO (TensorRef α Shape.scalar) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.gatherScalarNat (t := t0) (n := n) x.id i)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.gatherScalarNat (t := t0) (n := n) x.id i
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .gather cpu cuda

/-- Dynamic gather scalar using an index stored in `NatRef`. -/
def gatherScalarRef {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape]
  {n : Nat} (x : TensorRef α (.dim n .scalar)) (i : NatRef) : IO (TensorRef α Shape.scalar) := do
  let idx ← getNat (α := α) s i
  gatherScalarNat (α := α) s (n := n) x idx

/-- Dynamic gather row using an index stored in `NatRef` (out-of-range gives a zero row). -/
def gatherRowRef {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α) [Zero α] [DecidableEq Shape]
  {rows cols : Nat} (x : TensorRef α (.dim rows (.dim cols .scalar))) (i : NatRef) :
  IO (TensorRef α (.dim cols .scalar)) := do
  let idx ← getNat (α := α) s i
  if h : idx < rows then
    gatherRow (α := α) s (rows := rows) (cols := cols) x ⟨idx, h⟩
  else
    -- total: out-of-bounds labels map to a zero row
    const (α := α) s (sh := .dim cols .scalar) (fill (0 : α) (.dim cols .scalar)) (name := none)

/-- Gather `k` scalars using an explicit index tensor. PyTorch analogue: `gather` / advanced
  indexing. -/
def gatherVecNat {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {n k : Nat} (x : TensorRef α (.dim n .scalar)) (idx : Tensor Nat (.dim k .scalar)) :
  IO (TensorRef α (.dim k .scalar)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.gatherVecNat (t := t0) (n := n) (k := k) x.id
      idx)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.gatherVecNat (t := t0) (n := n) (k := k) x.id idx
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .gather cpu cuda

/-- Gather `k` rows using an explicit index tensor. PyTorch: `index_select(dim=0, index=...)`. -/
def gatherRowsNat {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {rows cols k : Nat} (x : TensorRef α (.dim rows (.dim cols .scalar))) (idx : Tensor Nat (.dim k
    .scalar)) :
  IO (TensorRef α (.dim k (.dim cols .scalar))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.gatherRowsNat (t := t0) (rows := rows) (cols :=
      cols) (k := k) x.id idx)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.gatherRowsNat (t := t0) (rows := rows) (cols := cols) (k := k) x.id idx
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .gather cpu cuda

/-- Gather `k` scalars using indices stored in the nat-environment (`NatVecRef`). -/
def gatherVecRef {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {n k : Nat} (x : TensorRef α (.dim n .scalar)) (idx : NatVecRef k) :
  IO (TensorRef α (.dim k .scalar)) := do
  let it ← getNatVec (α := α) (k := k) s idx
  gatherVecNat (α := α) s (n := n) (k := k) x it

/-- Gather `k` rows using indices stored in the nat-environment (`NatVecRef`). -/
def gatherRowsRef {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {rows cols k : Nat} (x : TensorRef α (.dim rows (.dim cols .scalar))) (idx : NatVecRef k) :
  IO (TensorRef α (.dim k (.dim cols .scalar))) := do
  let it ← getNatVec (α := α) (k := k) s idx
  gatherRowsNat (α := α) s (rows := rows) (cols := cols) (k := k) x it

/-- Scatter-add into a vector: return a copy of `x` with `x[i] += v`. -/
def scatterAddVec {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {n : Nat} (x : TensorRef α (.dim n .scalar)) (v : TensorRef α Shape.scalar) (i : Fin n) :
  IO (TensorRef α (.dim n .scalar)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.scatterAddVec (t := t0) (n := n) x.id v.id i)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.scatterAddVec (t := t0) (n := n) x.id v.id i
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .scatterAdd cpu cuda

/-- Scatter-add into a matrix row: return a copy of `x` with `x[i,:] += v`. -/
def scatterAddRow {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {rows cols : Nat}
  (x : TensorRef α (.dim rows (.dim cols .scalar))) (v : TensorRef α (.dim cols .scalar)) (i : Fin
    rows) :
  IO (TensorRef α (.dim rows (.dim cols .scalar))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.scatterAddRow (t := t0) (rows := rows) (cols :=
      cols) x.id v.id i)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.scatterAddRow (t := t0) (rows := rows) (cols := cols) x.id v.id i
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .scatterAdd cpu cuda

end EagerSession

end Internal
end Torch
end Autograd
end Runtime
