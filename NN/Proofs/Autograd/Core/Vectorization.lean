/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Basic

public import Mathlib.Algebra.BigOperators.Ring.Finset
public import Mathlib.Analysis.InnerProductSpace.PiL2

/-!
# Vectorization

Shared Euclidean-space vectorization utilities for analytic autograd proofs.

This module centralizes the `Vec` alias (`EuclideanSpace ℝ (Fin n)`) and the basic
`Tensor ℝ [n]` ↔ `Vec n` conversions used across multiple proof files.

## PyTorch correspondence / citations
This plays the same role as treating a length-`n` tensor as an element of $\mathbb R^n$ when using
standard analysis results (mean value theorem, operator norms, etc.).
https://pytorch.org/docs/stable/linalg.html
-/

@[expose] public section


namespace Proofs
namespace Autograd

open Spec
open Tensor

open scoped BigOperators

noncomputable section

/-- Euclidean vectors over `ℝ`. -/
abbrev Vec (n : Nat) := EuclideanSpace ℝ (Fin n)

/--
Convert a rank-one tensor (`Tensor ℝ [n]`) into a Euclidean vector `Vec n`.

This is the “analysis-friendly” view of a length-`n` tensor as an element of $\mathbb R^n$.
-/
def getScalarE {n : Nat} (t : Tensor ℝ [n]) : Vec n :=
  (EuclideanSpace.equiv (𝕜 := ℝ) (ι := Fin n)).symm (Spec.Tensor.getScalar t)

/-- Coordinate evaluation of the Euclidean view of a rank-one tensor. -/
@[simp] lemma getScalarE_ofLp {n : Nat} (t : Tensor ℝ [n]) (i : Fin n) :
    (getScalarE t).ofLp i = Spec.Tensor.getScalar t i := by
  simp [getScalarE, EuclideanSpace.equiv]

/--
Convert a Euclidean vector `Vec n` back into a rank-one tensor (`Tensor ℝ [n]`).

This is the inverse direction of `getScalarE`.
-/
def ofFnE {n : Nat} (v : Vec n) : Tensor ℝ [n] :=
  Spec.Tensor.ofFn ((EuclideanSpace.equiv (𝕜 := ℝ) (ι := Fin n)) v)

/-- `getScalarE` is a left inverse of `ofFnE`. -/
@[simp] lemma getScalarE_ofFnE {n : Nat} (v : Vec n) : getScalarE (ofFnE v) = v := by
  classical
  let e := EuclideanSpace.equiv (𝕜 := ℝ) (ι := Fin n)
  change e.symm (fun i => (Spec.Tensor.ofFn (e v)).getScalar i) = v
  rw [show (fun i => (Spec.Tensor.ofFn (e v)).getScalar i) = e v by
    funext i
    simp]
  exact e.symm_apply_apply v

/-- `ofFnE` is a left inverse of `getScalarE`. -/
@[simp] lemma ofFnE_getScalarE {n : Nat} (t : Tensor ℝ [n]) : ofFnE (getScalarE t) = t := by
  classical
  let e := EuclideanSpace.equiv (𝕜 := ℝ) (ι := Fin n)
  change Spec.Tensor.ofFn (e (e.symm (fun i => t.getScalar i))) = t
  rw [e.apply_symm_apply]
  exact Spec.Tensor.ofFn_getScalar t

/--
Coordinate formula for the Euclidean inner product on `Vec n`.

This is the statement $\langle x,y\rangle=\sum_i x_i y_i$ specialized to
`EuclideanSpace ℝ (Fin n)`.
-/
lemma inner_eq_sum_mul {n : Nat} (x y : Vec n) :
    inner ℝ x y = ∑ i : Fin n, x i * y i := by
  classical
  simpa [Vec, dotProduct, mul_comm] using
    (EuclideanSpace.inner_eq_star_dotProduct (x := x) (y := y))

end
end Autograd
end Proofs
