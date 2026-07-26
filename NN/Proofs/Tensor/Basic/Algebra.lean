/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Basic.BoundsNorms

/-!
# Tensor Algebra Lemmas

This file collects foundational algebraic lemmas for `Spec.Tensor`: extensionality, map/fold
rewrites, and pointwise arithmetic facts used throughout the autograd and runtime-correctness proof
files.
-/

@[expose] public section

namespace Spec

open Tensor
open scoped BigOperators

/-- Tensor extensionality over a generic element type: equal `getSpec` views imply equal tensors. -/
theorem tensor_ext {α : Type} {s : Shape} {x y : Tensor α s} :
  (∀ idxs : List Nat, getSpec x idxs = getSpec y idxs) → x = y := by
  intro h
  -- do induction on the shape, generalizing x and y so ih can be applied to sub-tensors
  induction s with
  | scalar =>
    -- both x and y must be Tensor.scalar a, Tensor.scalar b
    cases x with
    | scalar a =>
      cases y with
      | scalar b =>
        -- use the index [] to get equality of stored scalars
        have : getSpec (Tensor.scalar a) [] = getSpec (Tensor.scalar b) [] := h []
        have hab : a = b := by
          simpa using this
        simp [hab]

  | dim n s ih =>
    -- x and y must be Tensor.dim fx and Tensor.dim fy
    cases x with
    | dim fx =>
      cases y with
      | dim fy =>
        -- if dimension is zero, both are dim 0 s and are equal by definition
        by_cases hn : n = 0
        · -- n = 0
          -- both constructors are `Tensor.dim` with first argument `Fin 0 → _`
          -- there is only one inhabitant of `Fin 0 → _` so fx = fy definitionally;
          -- we can finish by refl
          have : fx = fy := by
            apply funext
            intro i
            rw[hn] at i
            exact Fin.elim0 i  -- impossible, Fin 0 has no elements
          rw [this]

        · -- n = Nat.succ n'
          -- show pointwise equality of the component functions fx and fy
          have pointwise : ∀ i : Fin n, fx i = fy i := by
            intro i
            -- to apply ih, we must show get_spec (fx i) idxs = get_spec (fy i) idxs for all idxs
            apply ih
            intro idxs
            -- by the computation rule for get_spec on `Tensor.dim`, we can rewrite
            -- get_spec (fx i) idxs  = get_spec (Tensor.dim fx) (i.val :: idxs)
            -- and similarly for fy; then use the hypothesis `h`
            calc
              getSpec (fx i) idxs = getSpec (Tensor.dim fx) (i.val :: idxs) := by
                simp [i.isLt]
              _ = getSpec (Tensor.dim fy) (i.val :: idxs) := by
                apply h
              _ = getSpec (fy i) idxs := by
                simp [i.isLt]

          -- now lift pointwise equality of the functions to equality of functions
          have eq_funcs : fx = fy := by
            apply funext
            intro i
            exact pointwise i

          -- rewrite and finish
          rw [eq_funcs]
/-- Elementwise addition is associative (over $\mathbb R$ tensors). -/
theorem add_spec_assoc {s : Shape}
  (a b c : Tensor ℝ s) :
  addSpec (addSpec a b) c = addSpec a (addSpec b c) := by
  -- Structural recursion on `s` and use associativity of ℝ addition
  induction s with
  | scalar =>
    cases a; cases b; cases c
    simp [addSpec, map2Spec]
    ring  -- Uses associativity of real addition
  | dim n s ih =>
    cases a with | dim fa =>
    cases b with | dim fb =>
    cases c with | dim fc =>
    simp [addSpec, map2Spec]
    funext i
    exact ih (fa i) (fb i) (fc i)

/-- Elementwise subtraction distributes over addition on the right. -/
theorem sub_spec_add_right {s : Shape}
  (a b c : Tensor ℝ s) :
  subSpec a (addSpec b c) = addSpec (subSpec a b) (negSpec c) := by
  -- Expand definitions using map2_spec and use ring properties
  induction s with
  | scalar =>
    cases a; cases b; cases c
    simp [subSpec, addSpec, negSpec, map2Spec, mapSpec]
    ring  -- Uses distributivity: a - (b + c) = (a - b) + (-c)
  | dim n s ih =>
    cases a with | dim fa =>
    cases b with | dim fb =>
    cases c with | dim fc =>
    simp [subSpec, addSpec, negSpec, map2Spec, mapSpec]
    funext i
    exact ih (fa i) (fb i) (fc i)

/-- Elementwise multiplication distributes over addition on the right. -/
theorem mul_spec_add_right {s : Shape}
  (a b c : Tensor ℝ s) :
  mulSpec a (addSpec b c) = addSpec (mulSpec a b) (mulSpec a c) := by
  -- Structural recursion on `s` and use distributivity of ℝ
  induction s with
  | scalar =>
    cases a; cases b; cases c
    simp [mulSpec, addSpec, map2Spec]
    ring  -- Uses distributivity of real multiplication
  | dim n s ih =>
    cases a with | dim fa =>
    cases b with | dim fb =>
    cases c with | dim fc =>
    simp [mulSpec, addSpec, map2Spec]
    funext i
    exact ih (fa i) (fb i) (fc i)

/-- Elementwise multiplication distributes over addition on the left. -/
theorem mul_spec_add_left {s : Shape}
  (a b c : Tensor ℝ s) :
  mulSpec (addSpec a b) c = addSpec (mulSpec a c) (mulSpec b c) := by
  -- Structural recursion on `s` and use distributivity of ℝ
  induction s with
  | scalar =>
    cases a; cases b; cases c
    simp [mulSpec, addSpec, map2Spec]
    ring  -- Uses distributivity of real multiplication
  | dim n s ih =>
    cases a with | dim fa =>
    cases b with | dim fb =>
    cases c with | dim fc =>
    simp [mulSpec, addSpec, map2Spec]
    funext i
    exact ih (fa i) (fb i) (fc i)

/-- Bias cancellation for tensor subtraction: `(a + c) - (b + c) = a - b`. -/
theorem sub_spec_bias_cancel {s : Shape} (a b c : Tensor ℝ s) :
  subSpec (addSpec a c) (addSpec b c) = subSpec a b := by
  -- Key lemma: (a + c) - (b + c) = a - b
  induction s with
  | scalar =>
    cases a; cases b; cases c
    simp [subSpec, addSpec, map2Spec]
  | dim n s ih =>
    cases a with | dim fa =>
    cases b with | dim fb =>
    cases c with | dim fc =>
    simp [subSpec, addSpec, map2Spec]
    funext i
    exact ih (fa i) (fb i) (fc i)

/-- Linearity of matrix-vector multiplication in the vector argument (addition). -/
theorem mat_vec_add {m n : Nat}
  (W : Tensor ℝ (.dim m (.dim n .scalar)))
  (x y : Tensor ℝ (.dim n .scalar)) :
  matVecMulSpec W (addSpec x y) =
  addSpec (matVecMulSpec W x) (matVecMulSpec W y) := by
  classical
  have hToVec :
      toVec (matVecMulSpec W (addSpec x y)) =
        toVec (addSpec (matVecMulSpec W x) (matVecMulSpec W y)) := by
    funext i
    -- Rewrite all mat-vec outputs as sums.
    rw [toVec_mat_vec_mul_spec (A := W) (v := addSpec x y) (i := i)]
    -- Expand the elementwise addition on the right (without unfolding `toVec` itself).
    simp [toVec_add_spec]
    rw [toVec_mat_vec_mul_spec (A := W) (v := x) (i := i)]
    rw [toVec_mat_vec_mul_spec (A := W) (v := y) (i := i)]
    -- Distribute `*` over `+` inside the sum and split the sum.
    simp [mul_add, Finset.sum_add_distrib]

  have hTensor :
      ofVec (toVec (matVecMulSpec W (addSpec x y))) =
        ofVec (toVec (addSpec (matVecMulSpec W x) (matVecMulSpec W y))) :=
    congrArg ofVec hToVec

  -- `ofVec ∘ toVec` is identity.
  simpa using
    (Eq.trans (ofVec_toVec (t := matVecMulSpec W (addSpec x y))).symm
      (Eq.trans hTensor (ofVec_toVec (t := addSpec (matVecMulSpec W x) (matVecMulSpec W
        y)))))

/-- Linearity of matrix-vector multiplication in the vector argument (scaling). -/
theorem mat_vec_scale {m n : Nat}
  (W : Tensor ℝ (.dim m (.dim n .scalar)))
  (x : Tensor ℝ (.dim n .scalar)) (c : ℝ) :
  matVecMulSpec W (scaleSpec x c) =
  scaleSpec (matVecMulSpec W x) c := by
  classical
  have hToVec :
      toVec (matVecMulSpec W (scaleSpec x c)) =
        toVec (scaleSpec (matVecMulSpec W x) c) := by
    funext i
    rw [toVec_mat_vec_mul_spec (A := W) (v := scaleSpec x c) (i := i)]
    -- `toVec (scale_spec _ c)` is pointwise scaling.
    simp [toVec_scale_spec]
    rw [toVec_mat_vec_mul_spec (A := W) (v := x) (i := i)]
    -- Pull out the scalar `c` from the sum.
    -- (Reassociate `*` so `Finset.sum_mul` applies.)
    have hassoc :
        (∑ k : Fin n, get2 W i k * (toVec x k * c)) =
          ∑ k : Fin n, (get2 W i k * toVec x k) * c := by
      refine Finset.sum_congr rfl ?_
      intro k _
      ring
    -- Now use `Finset.sum_mul` to factor `c` to the right.
    -- (`Finset.sum_mul` gives the reverse direction, so use symmetry.)
    simpa [hassoc, mul_assoc] using
      (Finset.sum_mul (s := (Finset.univ : Finset (Fin n)))
        (f := fun k : Fin n => get2 W i k * toVec x k) (a := c)).symm

  have hTensor :
      ofVec (toVec (matVecMulSpec W (scaleSpec x c))) =
        ofVec (toVec (scaleSpec (matVecMulSpec W x) c)) :=
    congrArg ofVec hToVec

  simpa using
    (Eq.trans (ofVec_toVec (t := matVecMulSpec W (scaleSpec x c))).symm
      (Eq.trans hTensor (ofVec_toVec (t := scaleSpec (matVecMulSpec W x) c))))

/-- Full linearity of matrix-vector multiplication in the vector argument. -/
theorem mat_vec_linear_combination {m n : Nat}
  (W : Tensor ℝ (.dim m (.dim n .scalar)))
  (x y : Tensor ℝ (.dim n .scalar)) (a b : ℝ) :
  matVecMulSpec W (addSpec (scaleSpec x a) (scaleSpec y b)) =
  addSpec (scaleSpec (matVecMulSpec W x) a)
           (scaleSpec (matVecMulSpec W y) b) := by
  -- Combine mat_vec_add and mat_vec_scale
  rw [mat_vec_add, mat_vec_scale, mat_vec_scale]

/--
Simplification lemmas for common patterns.
Useful for automated proof tactics.
-/
@[simp]
theorem get_dim_scalar {n : Nat} (f : Fin n → Tensor ℝ .scalar) (i : Fin n) :
  get (Tensor.dim f) i = f i := by rfl

/-- Mapping `0 + ·` over an `Option` is the identity. -/
lemma option_zero_add (o : Option ℝ) : o.map (fun x => 0 + x) = o := by
  cases o
  · rfl
  · simp [zero_add]

-- add_spec with zero tensor on the left
/-- Left identity for `add_spec`: adding the all-zero tensor does nothing. -/
@[simp]
theorem add_spec_zero_left {s : Shape} : ∀ (t : Tensor ℝ s),
  addSpec (fill 0 s) t = t
| Tensor.scalar a => by simp [addSpec, map2Spec, fill, zero_add]
| Tensor.dim fx => by
    simp [addSpec, map2Spec, fill]
    apply funext
    intro i
    -- recursively call theorem on component fx i
    exact add_spec_zero_left (fx i)

-- add_spec with zero tensor on the right
/-- Right identity for `add_spec`: adding the all-zero tensor does nothing. -/
@[simp]
theorem add_spec_zero_right {s : Shape} : ∀ (t : Tensor ℝ s),
  addSpec t (fill (0 : ℝ) s) = t
  | Tensor.scalar a => by simp [addSpec, map2Spec, fill]
  | Tensor.dim fx => by
    simp [addSpec, map2Spec, fill]
    apply funext
    intro i
    -- recursively call theorem on component fx i
    exact add_spec_zero_right (fx i)

-- mul_spec with one tensor on the left
/-- Left identity for `mul_spec`: multiplying by the all-ones tensor does nothing. -/
@[simp]
theorem mul_spec_one_left {s : Shape} : ∀ (t : Tensor ℝ s),
  mulSpec (fill (1 : ℝ) s) t = t
| Tensor.scalar a => by
    -- scalar case: 1 * a = a
    simp [mulSpec, map2Spec, fill]
| Tensor.dim fx => by
    -- dim case: function extensionality over components
    simp [mulSpec, map2Spec, fill]
    apply funext
    intro i
    -- recursively apply theorem to component
    exact mul_spec_one_left (fx i)

-- mul_spec with one tensor on the right
/-- Right identity for `mul_spec`: multiplying by the all-ones tensor does nothing. -/
@[simp]
theorem mul_spec_one_right {s : Shape} : ∀ (t : Tensor ℝ s),
  mulSpec t (fill (1 : ℝ) s) = t
  | Tensor.scalar a => by
    -- scalar case: a * 1 = a
    simp [mulSpec, map2Spec, fill]
  | Tensor.dim fx => by
      -- dim case: function extensionality over components
      simp [mulSpec, map2Spec, fill]
      apply funext
      intro i
      -- recursively apply theorem to component
      exact mul_spec_one_right (fx i)

end Spec
