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

/-- Concatenate at the axis following an arbitrary leading shape. -/
def concatAxisSpec {α : Type} :
    (leading : Shape) → {n m : Nat} → {suffix : Shape} →
      Tensor α (leading.concat (.dim n suffix)) →
      Tensor α (leading.concat (.dim m suffix)) →
      Tensor α (leading.concat (.dim (n + m) suffix))
  | .scalar, n, _m, _, .dim left, .dim right =>
      .dim fun i =>
        if h : i.val < n then
          left ⟨i.val, h⟩
        else
          right ⟨i.val - n, Nat.sub_lt_left_of_lt_add (Nat.not_lt.mp h) i.isLt⟩
  | .dim _extent rest, n, m, suffix, .dim left, .dim right =>
      .dim fun i => concatAxisSpec rest (left i) (right i)

/-- Insert a singleton dimension at any valid axis. -/
def unsqueezeSpec {α : Type} {shape : Shape} (tensor : Tensor α shape)
    (axis : Nat) (hAxis : axis ≤ shape.rank) : Tensor α (shape.insertAxis axis 1) :=
  match axis with
  | 0 =>
      match shape, tensor with
      | .scalar, tensor => .dim fun _ => tensor
      | .dim _ _, tensor => .dim fun _ => tensor
  | axis + 1 =>
      match shape, tensor with
      | .dim _ rest, .dim entries =>
          .dim fun i => unsqueezeSpec (entries i) axis (by
            apply Nat.lt_succ_iff.mp
            simpa [Shape.rank, Nat.add_comm] using hAxis)
      | .scalar, .scalar value =>
          False.elim (Nat.not_succ_le_zero axis hAxis)
termination_by axis

end Tensor
end Spec
