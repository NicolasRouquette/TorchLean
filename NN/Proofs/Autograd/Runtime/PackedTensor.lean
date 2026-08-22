/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Algebra.Soundness
public import NN.Runtime.Autograd.Engine.Core

/-!
# Packed Tensors

Bridge lemmas between the proved typed context (`TList`) and the runtime tape representation
(`Spec.PackedTensor` stored in `Array`s).

The conversion preserves order and stores each erased shape in the resulting `PackedTensor`.
-/

@[expose] public section

namespace Proofs
namespace Autograd
namespace Algebra

open Spec
open Tensor

namespace TList

variable {α : Type}

/--
Convert a typed context into a list of shape-tagged tensors.
-/
def toPackedList : {ss : List Shape} → TList α ss → List (Spec.PackedTensor α)
  | [], .nil => []
  | _ :: ss, .cons t ts => Spec.PackedTensor.ofTensor t :: toPackedList (ss := ss) ts

/-- `toPackedList` but as an `Array` (the container used by the runtime tape). -/
def toPackedArray {ss : List Shape} (ts : TList α ss) : Array (Spec.PackedTensor α) :=
  (toPackedList (α := α) (ss := ss) ts).toArray

/--
Like `toPackedList`, but tags each element with the absolute runtime node id it should correspond to,
starting from `start`.
-/
def toIndexedPackedList : {ss : List Shape} → TList α ss → Nat → List (Nat × Spec.PackedTensor α)
  | [], .nil, _ => []
  | _ :: ss, .cons t ts, i => (i, Spec.PackedTensor.ofTensor t) :: toIndexedPackedList (ss := ss) ts
    (i + 1)

/--
Every `(pid, tensor)` produced by `toIndexedPackedList ts start` satisfies `pid < start + ss.length`.

This is bookkeeping used to prove runtime backward only references earlier nodes.
-/
theorem mem_toIndexedPackedList_lt :
    {ss : List Shape} → (ts : TList α ss) → (start : Nat) →
      ∀ {pid : Nat} {pg : Spec.PackedTensor α},
        (pid, pg) ∈ toIndexedPackedList (α := α) (ss := ss) ts start → pid < start + ss.length
  | [], .nil, start, pid, pg, hmem => by
      cases hmem
  | _ :: ss, .cons _t ts, start, pid, pg, hmem => by
      simp [toIndexedPackedList] at hmem
      cases hmem with
      | inl h =>
          -- head element: `pid = start`
          have hpid : pid = start := h.1
          cases hpid
          -- After rewriting `(_ :: ss).length`, this is `start < start + Nat.succ ss.length`.
          simp
      | inr h =>
          have := mem_toIndexedPackedList_lt (ss := ss) ts (start + 1) (pid := pid) (pg := pg) h
          -- `start + 1 + ss.length = start + (ss.length + 1)`
          simpa [Nat.add_assoc, Nat.add_left_comm, Nat.add_comm] using this

/--
`toPackedList` ignores type-level casts of the shape list.

This is a convenience lemma for rewriting across reassociation/cast steps in proofs.
-/
@[simp] theorem toPackedList_cast {ss₁ ss₂ : List Shape} (h : ss₁ = ss₂) (xs : TList α ss₁) :
    toPackedList (α := α) (ss := ss₂) (TList.cast (α := α) h xs) = toPackedList (α := α) (ss := ss₁) xs :=
      by
  cases h
  simp [TList.cast]

/--
`toPackedArray` ignores type-level casts of the shape list.

This is a direct corollary of `toPackedList_cast`.
-/
@[simp] theorem toPackedArray_cast {ss₁ ss₂ : List Shape} (h : ss₁ = ss₂) (xs : TList α ss₁) :
    toPackedArray (α := α) (ss := ss₂) (TList.cast (α := α) h xs) = toPackedArray (α := α) (ss := ss₁) xs
      := by
  cases h
  simp [toPackedArray]

/-- `toPackedList` has the same length as the underlying shape list. -/
@[simp] theorem length_toPackedList :
    {ss : List Shape} → (xs : TList α ss) → (toPackedList (α := α) (ss := ss) xs).length = ss.length
  | [], .nil => by simp [toPackedList]
  | _ :: ss, .cons _t ts => by
      simpa [toPackedList] using (congrArg Nat.succ (length_toPackedList (ss := ss) ts))

/-- `toPackedArray` has the same size as the underlying shape list. -/
@[simp] theorem size_toPackedArray :
    {ss : List Shape} → (xs : TList α ss) → (toPackedArray (α := α) (ss := ss) xs).size = ss.length
  | ss, xs => by
      simp [toPackedArray, List.size_toArray, length_toPackedList (ss := ss) xs]

-- Tell `grind` about the most common cast/length normalization lemmas for these conversions.
attribute [grind =] toPackedList_cast toPackedArray_cast length_toPackedList size_toPackedArray

/-!
### Shape-erasing conversions

The lemmas below show that these conversions preserve length and order and interact well with
`TList.snoc` and `TList.get`. They are used later to relate runtime node ids to positions in the
typed proof context `Γ ++ ss`.
-/

/--
`toPackedList` commutes with appending a value: converting `snoc xs t` is `toPackedList xs ++ [t]`.
-/
theorem toPackedList_snoc {ss : List Shape} {τ : Shape} :
    (xs : TList α ss) → (t : Tensor α τ) →
      toPackedList (α := α) (ss := ss ++ [τ]) (TList.snoc (α := α) (ss := ss) (τ := τ) xs t) =
        toPackedList (α := α) (ss := ss) xs ++ [Spec.PackedTensor.ofTensor t] := by
  intro xs t
  induction ss with
  | nil =>
      cases xs
      simp [TList.snoc, toPackedList]
  | cons s ss ih =>
      cases xs with
      | cons x xs =>
          simp [TList.snoc, toPackedList, ih (xs := xs)]

/-- Array-form version of `toPackedList_snoc`. -/
@[simp] theorem toPackedArray_snoc {ss : List Shape} {τ : Shape} (xs : TList α ss) (t : Tensor α τ) :
    toPackedArray (α := α) (ss := ss ++ [τ]) (TList.snoc (α := α) (ss := ss) (τ := τ) xs t) =
      (toPackedArray (α := α) (ss := ss) xs).push (Spec.PackedTensor.ofTensor t) := by
  simp [toPackedArray, toPackedList_snoc (α := α) (ss := ss) (τ := τ) xs t]

/-- `toPackedArray` of a cons context is array cons/append of the head element. -/
theorem toPackedArray_cons {α : Type} {s : Shape} {ss : List Shape} (x : Tensor α s) (xs : TList α ss)
  :
    toPackedArray (α := α) (ss := s :: ss) (TList.cons x xs) =
      #[Spec.PackedTensor.ofTensor x] ++ toPackedArray (α := α) (ss := ss) xs := by
  simp [toPackedArray, toPackedList]

attribute [grind =] toPackedList_snoc toPackedArray_snoc toPackedArray_cons

/--
Array lookup through `toPackedArray` corresponds to `TList.get` after packing the result.

This is the key lemma that lets us connect runtime indexing (`arr[i]`) to proof indexing (`get xs
  i`).
-/
theorem get_toPackedArray {α : Type} :
    {ss : List Shape} → (xs : TList α ss) → (i : Fin ss.length) →
      let arr := toPackedArray (α := α) (ss := ss) xs
      arr[i.1]'(by
        dsimp [arr]
        exact Nat.lt_of_lt_of_eq i.2 (size_toPackedArray (α := α) (ss := ss) xs).symm) =
        Spec.PackedTensor.ofTensor (TList.get (α := α) (ss := ss) xs i)
  | [], .nil, i => by
      cases i with
      | mk val isLt =>
        exact False.elim ((Nat.not_lt_zero val) isLt)
  | _ :: ss, .cons x xs, ⟨0, hi⟩ => by
      -- `0 : Fin (Nat.succ ss.length)` is elaborated with a canonical proof,
      -- which is not definitionally equal to `hi`. Normalize the index so that
      -- `TList.get` reduces by `rfl`.
      have h0 : (0 : Fin (Nat.succ ss.length)) = ⟨0, hi⟩ := by
        ext
        rfl
      cases h0
      simp [toPackedArray, toPackedList]
      rfl
  | _ :: ss, .cons x xs, ⟨Nat.succ i, hi⟩ => by
      have : i < ss.length := Nat.lt_of_succ_lt_succ hi
      simpa [toPackedArray, toPackedList, TList.get] using
        (get_toPackedArray (α := α) (ss := ss) xs ⟨i, this⟩)

end TList

end Algebra
end Autograd
end Proofs
