/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape.LinearAlgebra

@[expose] public section


namespace Spec
namespace Tensor

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-!
# Concatenation and Slicing

Concatenation, slicing, singleton dimensions, and layout transforms.
-/

/-- Runtime check that a tensor value matches a runtime `Shape`.

We use this in a few “dynamic” utilities where we have a runtime shape value and want to guard
access/casts in a total way.
-/
def matchShape {s : Shape} (t : Tensor α s) : Shape → Prop :=
  match t with
  | .scalar _ =>
      fun
      | .scalar => True
      | .dim _ _ => False
  | .dim (n := n) f =>
      fun
      | .scalar => False
      | .dim n' s' => n = n' ∧ ∀ i : Fin n, (f i).matchShape s'

/-- Concatenate along axis 0 (append `t2` after `t1`). -/
def concatLeadingAxisSpec {α : Type} {n m : Nat} {s : Shape}
  (t1 : Tensor α (.dim n s))
  (t2 : Tensor α (.dim m s)) :
  Tensor α (.dim (n + m) s) :=
  match t1, t2 with
  | .dim f1, .dim f2 =>
    .dim fun i =>
      if h : i.val < n then
        f1 ⟨i.val, h⟩
      else
        let j : Fin m :=
          ⟨i.val - n, Nat.sub_lt_left_of_lt_add (Nat.not_lt.mp h) i.is_lt⟩
        f2 j

/-!
## Slicing / concatenation on the leading axis

`concatLeadingAxisSpec` is the "append on axis 0" primitive that powers many higher-level utilities
(sequence concatenation, channel skip connections, etc.).

For backprop and for "undoing" concatenations, it is convenient to have an explicit slice operation.
We keep the API compact and index-safe:

- `sliceLeadingAxisRangeSpec start len` selects `len` consecutive entries starting at `start` along axis 0.
- `concatLeadingAxisBackwardSpec` is the adjoint of `concatLeadingAxisSpec` (splits a gradient tensor).
-/

/-- Slice `len` entries along axis 0, starting at `start`.

This is the simplest "range slice" one typically needs to express:
- taking the first `n` channels/tokens,
- extracting the skip-connection half after a concat,
- implementing `take`/`drop` without changing the inner shape.

The proof `start + len ≤ n` makes the slice total (no out-of-bounds behavior). -/
def sliceLeadingAxisRangeSpec {α : Type} {n : Nat} {s : Shape}
  (start len : Nat) (h : start + len ≤ n)
  (t : Tensor α (.dim n s)) : Tensor α (.dim len s) :=
  match t with
  | .dim f =>
      .dim fun i =>
        let idx : Nat := start + i.val
        have h1 : idx < start + len := by
          simp [idx]
        have h2 : start + len ≤ n := h
        f ⟨idx, lt_of_lt_of_le h1 h2⟩

/-- Backward (adjoint) of `concatLeadingAxisSpec`.

If `y = concat_leading_axis_spec x1 x2`, then in reverse-mode we split the upstream gradient `δy` into:

- $\delta x_1$: the first $n$ entries of $\delta y$,
- $\delta x_2$: the last $m$ entries of $\delta y$. -/
def concatLeadingAxisBackwardSpec {α : Type} {n m : Nat} {s : Shape}
  (δ : Tensor α (.dim (n + m) s)) :
  Tensor α (.dim n s) × Tensor α (.dim m s) :=
  let δ₁ := sliceLeadingAxisRangeSpec (α := α) (n := n + m) (s := s) 0 n (by simp) δ
  let δ₂ :=
    sliceLeadingAxisRangeSpec (α := α) (n := n + m) (s := s) n m
      (by simp) δ
  (δ₁, δ₂)

/--
Backward (adjoint) of `sliceLeadingAxisRangeSpec`.

If `y = sliceLeadingAxisRangeSpec start len x`, then `sliceLeadingAxisRangeBackwardSpec start len δy` re-inserts
the gradient into the original shape and fills everything outside the slice with zeros.
-/
def sliceLeadingAxisRangeBackwardSpec {α : Type} [Zero α] {n : Nat} {s : Shape}
  (start len : Nat) (_h : start + len ≤ n)
  (δ : Tensor α (.dim len s)) : Tensor α (.dim n s) :=
  -- This is the adjoint of `slice_leading_axis_range_spec`: the gradient is re-inserted into the
  -- original shape and everything outside the slice is filled with zeros.
  Tensor.dim (fun i =>
    if h1 : i.val < start then
      fill (0 : α) s
    else if h2 : i.val < start + len then
      let j : Fin len :=
        ⟨i.val - start, Nat.sub_lt_left_of_lt_add (Nat.not_lt.mp h1) h2⟩
      getAtSpec δ j
    else
      fill (0 : α) s)

/-- Expand a `(n, s)` tensor into `(n, 1, s)` by inserting a trailing dimension of size 1.

PyTorch analogy: `t.unsqueeze(-1)` for a rank-1 outer dimension (or `unsqueeze(dim=1)` in 2D terms).
  -/
def expandToColSpec {n s} (t : Tensor α (.dim n s)) : Tensor α (.dim n (.dim 1 s)) :=
  Tensor.dim (fun i => Tensor.dim (fun _ => getAtSpec t i))

/-- Squeeze a `(n,1,s)` tensor back into `(n,s)` by dropping the singleton dimension. -/
def squeezeColSpec {n s} (t : Tensor α (.dim n (.dim 1 s))) : Tensor α (.dim n s) :=
  Tensor.dim (fun i => getAtSpec (getAtSpec t i) 0)
end Tensor
end Spec
