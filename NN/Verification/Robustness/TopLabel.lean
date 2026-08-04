/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor.API

/-!
# Certified label checks

Shared predicates for certified classification from output bounds.

The checker uses one rule: the claimed label's lower bound must be strictly above every other
class upper bound. Tensor bounds come from in-memory IBP/CROWN runs. Array bounds come from JSON
artifacts.
-/

@[expose] public section

namespace NN.Verification.Robustness.TopLabel

open _root_.Spec
open _root_.Spec.Tensor

/-- Strict label certificate over any indexed lower/upper bounds. -/
def strictTopLabelBy {α : Type} [Max α] (n label : Nat)
    (lo hi : Fin n → α) (gt : α → α → Bool) : Bool :=
  if h : label < n then
    let y : Fin n := ⟨label, h⟩
    let loY := lo y
    let maxOther? :=
      (List.finRange n).foldl (fun (acc : Option α) i =>
        if i = y then acc
        else
          match acc with
          | none => some (hi i)
          | some m => some (max m (hi i))) none
    match maxOther? with
    | none => true
    | some m => gt loY m
  else
    false

/-- Check a label directly from tensor lower/upper bounds. -/
def certifiesLabelFromTensorBounds {α : Type} [Context α] {n : Nat}
    (lo hi : Tensor α (.dim n .scalar)) (label : Nat) : Bool :=
  strictTopLabelBy n label
    (fun i => Tensor.vecGet lo i)
    (fun i => Tensor.vecGet hi i)
    Context.gtBool

/-- Check a label from JSON-style array lower/upper bounds. -/
def certifiesLabelFromArrayBounds (lo hi : Array Float) (label : Nat) : Bool :=
  if hSize : lo.size = hi.size then
    strictTopLabelBy lo.size label
      (fun i => lo[i.1]'i.2)
      (fun i =>
        have h : i.1 < hi.size := by
          simp [← hSize, i.2]
        hi[i.1]'h)
      (fun x y => decide (x > y))
  else
    false

end NN.Verification.Robustness.TopLabel
