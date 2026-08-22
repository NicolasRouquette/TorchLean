/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.Runtime.Autograd.Engine.Core.Shape

/-!
# Core Tape Indexing Operations

This file implements gather and scatter-style tape nodes. The forward rules expose typed indexing
operations, and the backward rules route upstream gradients back to the selected source coordinates.
-/

@[expose] public section

namespace Runtime
namespace Autograd

open Spec
open Tensor

namespace Tape

/--
Gather a scalar from a 1D vector using a compile-time index `Fin n`.

PyTorch comparison: `x[i]` (1D indexing).
-/
def gatherScalar {α : Type} [Zero α] [DecidableEq Shape]
  {n : Nat} (t : Tape α) (xId : Nat) (i : Fin n) : Result (Tape α × Nat) := do
  let x ← requireValue (α := α) (t := t) (s := .dim n .scalar) xId
  let y : Tensor α Shape.scalar := getAtSpec x i
  let node : Node α :=
    { name := some s!"gather_scalar[{i.val}]"
      value := Spec.PackedTensor.ofTensor y
      requiresGrad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := Shape.scalar) dLdyAny
        let g : α := Tensor.item dLdy
        let dx : Tensor α (.dim n .scalar) :=
          Tensor.dim (fun j => Tensor.scalar (if decide (j = i) then g else 0))
        pure [(xId, Spec.PackedTensor.ofTensor dx)]
    }
  pure (t.addNode node)

/--
Gather a row from a 2D matrix using a compile-time index `Fin rows`.

PyTorch comparison: `x[i]` for 2D tensors (row indexing).
-/
def gatherRow {α : Type} [Zero α] [DecidableEq Shape]
  {rows cols : Nat} (t : Tape α) (xId : Nat) (i : Fin rows) : Result (Tape α × Nat) := do
  let x ← requireValue (α := α) (t := t) (s := .dim rows (.dim cols .scalar)) xId
  let y : Tensor α (.dim cols .scalar) := getAtSpec x i
  let node : Node α :=
    { name := some s!"gather_row[{i.val}]"
      value := Spec.PackedTensor.ofTensor y
      requiresGrad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := .dim cols .scalar) dLdyAny
        let dx : Tensor α (.dim rows (.dim cols .scalar)) :=
          Tensor.dim (fun r =>
            if decide (r = i) then
              dLdy
            else
              fill (0 : α) (.dim cols .scalar))
        pure [(xId, Spec.PackedTensor.ofTensor dx)]
    }
  pure (t.addNode node)

/--
Gather a scalar from a 1D vector using a runtime `Nat` index.

Out-of-bounds indices are totalized to return `0`.
PyTorch comparison: `x[i]` would raise on out-of-range; here we return `0` to keep the op total.
-/
def gatherScalarNatOrZero {α : Type} [Zero α] [DecidableEq Shape]
  {n : Nat} (t : Tape α) (xId : Nat) (i : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α := α) (t := t) (s := .dim n .scalar) xId
  let y : Tensor α Shape.scalar :=
    if h : i < n then
      getAtSpec x ⟨i, h⟩
    else
      Tensor.scalar 0
  let node : Node α :=
    { name := some s!"gather_scalar_nat_or_zero[{i}]"
      value := Spec.PackedTensor.ofTensor y
      requiresGrad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := Shape.scalar) dLdyAny
        let g : α := Tensor.item dLdy
        let dx : Tensor α (.dim n .scalar) :=
          Tensor.dim (fun j =>
            if i < n then
              if decide (j.val = i) then Tensor.scalar g else Tensor.scalar 0
            else
              Tensor.scalar 0)
        pure [(xId, Spec.PackedTensor.ofTensor dx)]
    }
  pure (t.addNode node)

/--
Gather `k` scalars from a 1D vector using an explicit index tensor.

Out-of-bounds indices are totalized to `0`. In the backward pass, gradients are accumulated for
repeated indices (scatter-add semantics).
PyTorch comparison: related to `torch.gather` / advanced indexing.
-/
def gatherVecNatOrZero {α : Type} [Add α] [Zero α] [DecidableEq Shape]
  {n k : Nat} (t : Tape α) (xId : Nat) (idx : Tensor Nat (.dim k .scalar)) :
  Result (Tape α × Nat) := do
  let x ← requireValue (α := α) (t := t) (s := .dim n .scalar) xId
  let y : Tensor α (.dim k .scalar) :=
    match idx with
    | Tensor.dim f =>
        Tensor.dim (fun j =>
          match f j with
          | Tensor.scalar ij =>
              if h : ij < n then
                getAtSpec x ⟨ij, h⟩
              else
                Tensor.scalar 0)
  let node : Node α :=
    { name := some "gather_vec_nat_or_zero"
      value := Spec.PackedTensor.ofTensor y
      requiresGrad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := .dim k .scalar) dLdyAny
        let dx : Tensor α (.dim n .scalar) :=
          Tensor.dim (fun iFin =>
            let sum : α :=
              (List.finRange k).foldl (fun acc j =>
                let ij :=
                  match idx with
                  | Tensor.dim f =>
                      match f j with
                      | Tensor.scalar v => v
                if ij < n then
                  if decide (ij = iFin.val) then
                    let gj : α :=
                      match getAtSpec dLdy j with
                      | Tensor.scalar v => v
                    acc + gj
                  else acc
                else acc
              ) 0
            Tensor.scalar sum)
        pure [(xId, Spec.PackedTensor.ofTensor dx)]
    }
  pure (t.addNode node)

/--
Gather `k` rows from a 2D matrix using an explicit index tensor.

Out-of-bounds indices are totalized to zero rows; backward accumulates gradients into selected
rows (scatter-add), including repeated indices.
-/
def gatherRowsNatOrZero {α : Type} [Add α] [Zero α] [DecidableEq Shape]
  {rows cols k : Nat} (t : Tape α) (xId : Nat) (idx : Tensor Nat (.dim k .scalar)) :
  Result (Tape α × Nat) := do
  let x ← requireValue (α := α) (t := t) (s := .dim rows (.dim cols .scalar)) xId
  let y : Tensor α (.dim k (.dim cols .scalar)) :=
    match idx with
    | Tensor.dim f =>
        Tensor.dim (fun j =>
          match f j with
          | Tensor.scalar ij =>
              if h : ij < rows then
                getAtSpec x ⟨ij, h⟩
              else
                fill (0 : α) (.dim cols .scalar))
  let node : Node α :=
    { name := some "gather_rows_nat_or_zero"
      value := Spec.PackedTensor.ofTensor y
      requiresGrad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := .dim k (.dim cols .scalar)) dLdyAny
        let dx : Tensor α (.dim rows (.dim cols .scalar)) :=
          Tensor.dim (fun rFin =>
            let rowGrad : Tensor α (.dim cols .scalar) :=
              (List.finRange k).foldl (fun acc j =>
                let ij :=
                  match idx with
                  | Tensor.dim f =>
                      match f j with
                      | Tensor.scalar v => v
                if ij < rows then
                  if decide (ij = rFin.val) then
                    addSpec acc (getAtSpec dLdy j)
                  else acc
                else acc
              ) (fill (0 : α) (.dim cols .scalar))
            rowGrad)
        pure [(xId, Spec.PackedTensor.ofTensor dx)]
    }
  pure (t.addNode node)

/--
Scatter-add into a vector: return a copy of `x` with `x[i] += v`.

Backward: gradient w.r.t. `x` is the upstream `dL/dy`, and gradient w.r.t. `v` is the gathered
scalar `dL/dy[i]`.
-/
def scatterAddVec {α : Type} [Add α] [Zero α] [DecidableEq Shape]
  {n : Nat} (t : Tape α) (xId vId : Nat) (i : Fin n) : Result (Tape α × Nat) := do
  let x ← requireValue (α := α) (t := t) (s := .dim n .scalar) xId
  let vT ← requireValue (α := α) (t := t) (s := Shape.scalar) vId
  let v : α := Tensor.item vT
  let xiT : Tensor α Shape.scalar := getAtSpec x i
  let xi : α := Tensor.item xiT
  let y : Tensor α (.dim n .scalar) := updateSpec x [i.val] (xi + v)
  let node : Node α :=
    { name := some s!"scatter_add_vec[{i.val}]"
      value := Spec.PackedTensor.ofTensor y
      requiresGrad := true
      parents := [xId, vId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := .dim n .scalar) dLdyAny
        let dv : Tensor α Shape.scalar := getAtSpec dLdy i
        pure [(xId, Spec.PackedTensor.ofTensor dLdy), (vId, Spec.PackedTensor.ofTensor dv)]
    }
  pure (t.addNode node)

/--
Scatter-add into a matrix row: return a copy of `x` with `x[i,:] += v`.

Backward: gradient w.r.t. `v` is the gathered row `dL/dy[i,:]`.
-/
def scatterAddRow {α : Type} [Add α] [Zero α] [DecidableEq Shape]
  {rows cols : Nat} (t : Tape α) (xId vId : Nat) (i : Fin rows) : Result (Tape α × Nat) := do
  let x ← requireValue (α := α) (t := t) (s := .dim rows (.dim cols .scalar)) xId
  let v ← requireValue (α := α) (t := t) (s := .dim cols .scalar) vId
  let y : Tensor α (.dim rows (.dim cols .scalar)) :=
    Tensor.dim (fun r =>
      if decide (r = i) then
        addSpec (getAtSpec x r) v
      else
        getAtSpec x r)
  let node : Node α :=
    { name := some s!"scatter_add_row[{i.val}]"
      value := Spec.PackedTensor.ofTensor y
      requiresGrad := true
      parents := [xId, vId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := .dim rows (.dim cols .scalar)) dLdyAny
        let dv : Tensor α (.dim cols .scalar) := getAtSpec dLdy i
        pure [(xId, Spec.PackedTensor.ofTensor dLdy), (vId, Spec.PackedTensor.ofTensor dv)]
    }
  pure (t.addNode node)
