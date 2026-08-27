/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import Mathlib.Init

/-!
# Masking primitives for self-supervised objectives

This file gives a small finite-index vocabulary for masked prediction objectives. It stays
independent of any particular image or transformer implementation: a patch/token collection is just
`Fin n → α`, and a mask is a Boolean predicate on `Fin n`.

The definitions make MAE/JEPA-style objectives precise enough for local invariants before they are
connected to larger executable models.
-/

@[expose] public section

namespace NN.MLTheory.SelfSupervised

/-- A finite mask over `n` patches/tokens. `true` means the index is selected. -/
abbrev Mask (n : Nat) := Fin n → Bool

/-- Proposition stating that index `i` is selected by the Boolean mask `m`. -/
def selected {n : Nat} (m : Mask n) (i : Fin n) : Prop :=
  m i = true

/-- The all-visible/all-target mask. -/
def allMask (n : Nat) : Mask n :=
  fun _ => true

/-- Mask selecting no positions. -/
def emptyMask (n : Nat) : Mask n :=
  fun _ => false

/-- Pointwise Boolean complement of a mask. -/
def complement {n : Nat} (m : Mask n) : Mask n :=
  fun i => !(m i)

@[simp] theorem allMask_selected {n : Nat} (i : Fin n) :
    selected (allMask n) i := by
  simp [selected, allMask]

@[simp] theorem emptyMask_not_selected {n : Nat} (i : Fin n) :
    ¬ selected (emptyMask n) i := by
  simp [selected, emptyMask]

@[simp] theorem complement_selected_iff {n : Nat} (m : Mask n) (i : Fin n) :
    selected (complement m) i ↔ ¬ selected m i := by
  simp [selected, complement]

/--
Generic masked loss over an explicit array of selected indices.

For theory files we keep the scalar loss as `Nat`; concrete runtime losses can instantiate this
with squared-error bins, quantized patch losses, or any other executable per-patch score.
-/
def maskedLoss {n : Nat} (idxs : Array (Fin n)) (perPatchLoss : Fin n → Nat) : Nat :=
  (idxs.map perPatchLoss).sum

@[simp] theorem maskedLoss_nil {n : Nat} (perPatchLoss : Fin n → Nat) :
    maskedLoss (#[] : Array (Fin n)) perPatchLoss = 0 := by
  simp [maskedLoss]

@[simp] theorem maskedLoss_push {n : Nat} (idxs : Array (Fin n)) (i : Fin n)
    (perPatchLoss : Fin n → Nat) :
    maskedLoss (idxs.push i) perPatchLoss =
      maskedLoss idxs perPatchLoss + perPatchLoss i := by
  simp [maskedLoss, Array.map_push]

theorem maskedLoss_append {n : Nat} (xs ys : Array (Fin n)) (perPatchLoss : Fin n → Nat) :
    maskedLoss (xs ++ ys) perPatchLoss =
      maskedLoss xs perPatchLoss + maskedLoss ys perPatchLoss := by
  simp [maskedLoss, Array.map_append, Array.sum_append]

theorem maskedLoss_reverse {n : Nat} (idxs : Array (Fin n)) (perPatchLoss : Fin n → Nat) :
    maskedLoss idxs.reverse perPatchLoss = maskedLoss idxs perPatchLoss := by
  simp [maskedLoss, Array.map_reverse, Array.sum_reverse]

theorem maskedLoss_eq_zero_of_all_zero {n : Nat} (idxs : Array (Fin n))
    (perPatchLoss : Fin n → Nat) (h : ∀ i ∈ idxs, perPatchLoss i = 0) :
    maskedLoss idxs perPatchLoss = 0 := by
  have hmap : idxs.map perPatchLoss = Array.replicate idxs.size 0 := by
    apply Array.ext <;> simp only [Array.size_map, Array.size_replicate]
    intro i hiMap hiReplicate
    rw [Array.getElem_map, Array.getElem_replicate]
    exact h idxs[i] (Array.getElem_mem hiMap)
  rw [maskedLoss, hmap]
  simp

end NN.MLTheory.SelfSupervised
