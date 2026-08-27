/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.BigOperators.Ring.Finset
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Analysis.Normed.Group.Basic
public import Mathlib.Data.Real.Basic
public import Mathlib.Analysis.Real.Sqrt
public import NN.MLTheory.LearningTheory.Robustness.Spec
public import NN.Proofs.Tensor.Basic
public import NN.Spec.Core.Context
public import NN.Spec.Core.Tensor
public import NN.Spec.Core.TensorOps

/-!
# Real-valued tensor norm facts

This module owns the proof-oriented `Tensor` norm and distance definitions over `ℝ`, together with
their algebraic properties. Neural-network Lipschitz bounds build on these facts in
`NN.Proofs.Analysis.Lipschitz.Network`.

## Scope and conventions
- Everything here is **spec-level** and **real-valued** (`ℝ`), so we can freely use Mathlib’s
  analysis and order theory.
- The main `L2` norm here is proof-oriented: it is defined from `Spec.tensorNormSquared`, the same
  dot-product/sum-of-squares object used throughout tensor algebra proofs.
- `NN.MLTheory.Robustness.Spec` also has scalar-polymorphic norm definitions for runtime and
  verification statements. This file does **not** duplicate that API surface; it proves real-valued
  theorems and includes bridge lemmas where those polymorphic specs need theorem-level support.

## PyTorch correspondence / citations
- $\ell_2$/$\ell_1$/$\ell_\infty$ norms correspond to PyTorch’s `torch.linalg.*_norm` /
  `torch.linalg.norm` APIs.
  https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html
  https://pytorch.org/docs/stable/generated/torch.linalg.norm.html

Import `NN.Proofs.Analysis.Lipschitz` for both the norm foundation and network bounds, or this module
when only the norm theory is needed.
-/

@[expose] public section

namespace Proofs

open Spec
open Tensor
open scoped BigOperators

-- Reuse the tensor-algebra definitions from `Spec`/`Proofs.Tensor.Basic` instead of restating dot
-- products or squared norms locally.
open Spec (dot tensorNormSquared tensor_norm_squared_nonneg
           tensor_norm_squared_zero_iff mul_spec_comm add_spec_comm dot_comm
           sum_spec_add_distrib mul_spec_add_left mul_spec_add_right
           add_spec_assoc)

-- ====================================================================
-- TENSOR NORMS AND DISTANCE FUNCTIONS
-- ====================================================================

/--
$\ell_2$ norm (Euclidean norm) for tensors.
Fundamental for measuring tensor magnitudes and distances.
-/
noncomputable def tensorL2Norm {s : Shape} (t : Tensor ℝ s) : ℝ :=
  Real.sqrt (tensorNormSquared t)

/--
$\ell_\infty$ norm (maximum norm) for tensors.
Important for uniform convergence and pointwise bounds.
-/
noncomputable def tensorLInftyNorm {s : Shape} (t : Tensor ℝ s) : ℝ :=
  tensorFoldlSpec (fun acc x => max acc (|x|)) (0 : ℝ) t

/--
$\ell_1$ norm (Manhattan norm) for tensors.
Useful for sparsity-inducing regularization.
-/
noncomputable def tensorL1Norm {s : Shape} (t : Tensor ℝ s) : ℝ :=
  sumSpec (absSpec t)

/--
Distance function based on the $\ell_2$ norm.
-/
noncomputable def tensorL2Dist {s : Shape} (x y : Tensor ℝ s) : ℝ :=
  tensorL2Norm (subSpec x y)

/-- Distance function based on the $\ell_\infty$ norm. -/
noncomputable def tensorLInftyDist {s : Shape} (x y : Tensor ℝ s) : ℝ :=
  tensorLInftyNorm (subSpec x y)

/-!
## Cross-library norm facts

`NN.MLTheory.Robustness.Spec` defines a scalar-polymorphic `tensor_linf_norm`. In this file we work
over $\mathbb R$ and often use `tensor_l2_norm`. The key inequality
$\lVert v\rVert_\infty\le\lVert v\rVert_2$ is what lets $\ell_2$-based
Lipschitz proofs feed directly into the $\ell_\infty$-robustness lemmas.
-/

/--
For a real vector-valued tensor, the $\ell_\infty$ norm from
`NN.MLTheory.Robustness.Spec` is bounded by the $\ell_2$ norm from this file:

$\lVert v\rVert_\infty\le\lVert v\rVert_2$.
-/
theorem tensor_linf_norm_le_tensor_l2_norm {n : Nat} (y : Tensor ℝ [n]) :
    NN.MLTheory.Robustness.Spec.tensorLinfNorm (α := ℝ) y ≤ tensorL2Norm y := by
  classical
  cases y with
  | dim values =>
    -- Coordinate bound: `|y[i]| ≤ ‖y‖₂`.
    have habs_getScalar_le : ∀ i : Fin n, |getScalar (Tensor.dim values) i| ≤ tensorL2Norm (Tensor.dim values) := by
      intro i
      have hle_sum :
          getScalar (Tensor.dim values) i * getScalar (Tensor.dim values) i ≤
            ∑ j : Fin n, getScalar (Tensor.dim values) j * getScalar (Tensor.dim values) j := by
        have h_nonneg :
            ∀ j : Fin n, 0 ≤ getScalar (Tensor.dim values) j * getScalar (Tensor.dim values) j := by
          intro j
          exact mul_self_nonneg (getScalar (Tensor.dim values) j)
        have h' :
            getScalar (Tensor.dim values) i * getScalar (Tensor.dim values) i ≤
              (Finset.univ : Finset (Fin n)).sum
                (fun j => getScalar (Tensor.dim values) j * getScalar (Tensor.dim values) j) :=
          Finset.single_le_sum (fun j _ => h_nonneg j) (by simp)
        simpa using h'

      have hnormsq :
          tensorNormSquared (Tensor.dim values : Tensor ℝ [n]) =
            ∑ j : Fin n, getScalar (Tensor.dim values) j * getScalar (Tensor.dim values) j := by
        simpa [tensorNormSquared] using
          (dot_vec_eq_sum (a := (Tensor.dim values)) (b := (Tensor.dim values)))

      have hle_normsq :
          getScalar (Tensor.dim values) i * getScalar (Tensor.dim values) i ≤
            tensorNormSquared (Tensor.dim values : Tensor ℝ [n]) := by
        simpa [hnormsq] using hle_sum

      have hsqrt :
          Real.sqrt (getScalar (Tensor.dim values) i * getScalar (Tensor.dim values) i) ≤
            Real.sqrt (tensorNormSquared (Tensor.dim values : Tensor ℝ [n])) :=
        Real.sqrt_le_sqrt hle_normsq

      have hsqrt_lhs :
          Real.sqrt (getScalar (Tensor.dim values) i * getScalar (Tensor.dim values) i) =
            |getScalar (Tensor.dim values) i| := by
        simpa [pow_two] using (Real.sqrt_sq_eq_abs (getScalar (Tensor.dim values) i))

      rw [hsqrt_lhs] at hsqrt
      simpa only [tensorL2Norm] using hsqrt

    -- Each slice (scalar) `values i` has `L∞` norm equal to `|getScalar y i|`, hence is ≤ `‖y‖₂`.
    have hval :
        ∀ i : Fin n,
          NN.MLTheory.Robustness.Spec.tensorLinfNorm (α := ℝ) (values i) ≤
            tensorL2Norm (Tensor.dim values : Tensor ℝ [n]) := by
      intro i
      cases hvi : values i with
      | scalar v =>
        have hi := habs_getScalar_le i
        simpa [NN.MLTheory.Robustness.Spec.tensorLinfNorm, MathFunctions.abs, getScalar, hvi] using hi

    have h0 : (0 : ℝ) ≤ tensorL2Norm (Tensor.dim values : Tensor ℝ [n]) := by
      -- Note: `tensor_l2_norm_nonneg` is defined later in this file; avoid forward references.
      simp [tensorL2Norm]

    have hfold :
        (List.finRange n).foldl
            (fun acc i =>
              max acc (NN.MLTheory.Robustness.Spec.tensorLinfNorm (α := ℝ) (values i)))
            0
          ≤ tensorL2Norm (Tensor.dim values : Tensor ℝ [n]) := by
      exact List.foldl_max_le_of_le (List.finRange n)
        (fun i => NN.MLTheory.Robustness.Spec.tensorLinfNorm (α := ℝ) (values i)) h0
        (by
          intro i _hi
          exact hval i)

    -- Unfold the `tensor_linf_norm` definition on vectors.
    simpa [NN.MLTheory.Robustness.Spec.tensorLinfNorm] using hfold

-- Basic norm properties used throughout the Lipschitz development.

/--
The $\ell_2$ norm is nonnegative.
-/
theorem tensor_l2_norm_nonneg {s : Shape} (t : Tensor ℝ s) :
  tensorL2Norm t ≥ (0 : ℝ) := by
  simp [tensorL2Norm]

/--
The $\ell_2$ norm is zero if and only if the tensor is zero.
-/
theorem tensor_l2_norm_zero_iff {s : Shape} (t : Tensor ℝ s) :
  tensorL2Norm t = (0 : ℝ) ↔ t = fill (0 : ℝ) s := by
  rw [show tensorL2Norm t = 0 ↔ Real.sqrt (tensorNormSquared t) = 0 by rfl]
  rw [Real.sqrt_eq_zero]
  exact tensor_norm_squared_zero_iff t
  exact tensor_norm_squared_nonneg t

/--
Basic lemma: dot product with zero tensor is zero.
-/
theorem dot_zero_right {s : Shape} (x : Tensor ℝ s) :
  dot x (fill (0 : ℝ) s) = (0 : ℝ) := by
  -- By induction on the tensor `Shape`; the `dim` case reduces to “folding `(+ )` over zeros”.
  induction s with
  | scalar =>
    cases x with | scalar a =>
    simp only [dot, mulSpec, map2Spec, fill, sumSpec, tensorFoldlSpec]
    ring
  | dim n s ih =>
    cases x with | dim fx =>
    simp only [dot, mulSpec, map2Spec, fill, sumSpec, tensorFoldlSpec]
    -- Use induction hypothesis on each component
    have h : ∀ i : Fin n, dot (fx i) (fill (0 : ℝ) s) = (0 : ℝ) := by
      intro i
      exact ih (fx i)
    -- The goal shows the expanded form of sum_spec for a dim tensor
    -- We need to prove: tensor_foldl_spec.go (· + ·) n s (fun i => mul_spec (fx i) (fill 0 s)) 0 0
    -- = 0

    -- Key insight: each component mul_spec (fx i) (fill 0 s) has sum 0
    have component_sum_zero : ∀ i : Fin n, sumSpec (mulSpec (fx i) (fill (0 : ℝ) s)) = (0 : ℝ) :=
      by
      intro i
      rw [← dot]
      exact h i

    -- Now we prove that the fold starting from 0, adding 0 at each step, gives 0
    -- We'll use strong induction on the starting index
    suffices ∀ k, k ≤ n → tensorFoldlSpec.go (· + ·) n s (fun i => mulSpec (fx i) (fill (0 : ℝ)
      s)) k (0 : ℝ) = (0 : ℝ) by
      exact this 0 (Nat.zero_le n)

    intro k hk
    -- We'll prove by induction on n - k
    induction h_ind : n - k generalizing k with
    | zero =>
      -- Base case: n - k = 0, so k = n
      have k_eq_n : k = n := by grind
      subst k
      -- Since `k = n`, the `k < n` loop condition is false and `go` returns the accumulator.
      have hgo :
          tensorFoldlSpec.go (· + ·) n s (fun i => mulSpec (fx i) (fill (0 : ℝ) s)) n (0 : ℝ) =
            (0 : ℝ) := by
        simpa using
          (Spec.tensor_foldl_spec_go_of_not_lt (f := (· + ·))
              (values := fun i => mulSpec (fx i) (fill (0 : ℝ) s)) (k := n) (acc := (0 : ℝ))
              (by simp))
      simp [hgo]
    | succ m ih =>
      have hlt : k < n := by grind
      rw [Spec.tensor_foldl_spec_go_of_lt (f := (· + ·))
        (values := fun i => mulSpec (fx i) (fill (0 : ℝ) s)) (k := k) (acc := (0 : ℝ)) hlt]
      have sum_zero : tensorFoldlSpec (· + ·) (0 : ℝ) (mulSpec (fx ⟨k, hlt⟩) (fill (0 : ℝ) s)) =
        (0 : ℝ) := by
        rw [← sumSpec]
        exact component_sum_zero ⟨k, hlt⟩
      rw [sum_zero]
      have h_next : n - (k + 1) = m := by grind
      have k_plus_one_le : k + 1 ≤ n := Nat.succ_le_of_lt hlt
      exact ih (k + 1) k_plus_one_le h_next


/--
Bilinearity of dot product over addition (distributive property).
-/
theorem dot_add_add {s : Shape} (x y : Tensor ℝ s) :
  dot (addSpec x y) (addSpec x y) =
  dot x x + 2 * dot x y + dot y y := by
  -- This is the key bilinearity property: (x + y) · (x + y) = x·x + 2x·y + y·y
  -- It follows from distributivity of dot product over addition
  simp [dot]
  -- First expand mul_spec (add_spec x y) (add_spec x y) using distributivity
  rw [mul_spec_add_left]
  -- Now we have add_spec (mul_spec x (add_spec x y)) (mul_spec y (add_spec x y))
  -- Expand each term using mul_spec_add_right
  rw [mul_spec_add_right x x y, mul_spec_add_right y x y]
  -- Now we have add_spec (add_spec (mul_spec x x) (mul_spec x y)) (add_spec (mul_spec y x)
  -- (mul_spec y y))
  -- Use commutativity of multiplication
  rw [mul_spec_comm y x]
  -- Now rearrange the nested add_spec operations
  -- We have sum_spec of: add_spec (add_spec (mul_spec x x) (mul_spec x y)) (add_spec (mul_spec x y)
  -- (mul_spec y y))
  -- First, use associativity and commutativity of add_spec to rearrange
  have rearrange : addSpec (addSpec (mulSpec x x) (mulSpec x y)) (addSpec (mulSpec x y)
    (mulSpec y y)) =
                   addSpec (addSpec (mulSpec x x) (mulSpec y y)) (addSpec (mulSpec x y)
                     (mulSpec x y)) := by
    -- Rearrange `(a + b) + (b + c)` into `(a + c) + (b + b)` using associativity/commutativity.
    rw [add_spec_assoc (mulSpec x x) (mulSpec x y) (addSpec (mulSpec x y) (mulSpec y y))]
    rw [← add_spec_assoc (mulSpec x y) (mulSpec x y) (mulSpec y y)]
    rw [add_spec_comm (addSpec (mulSpec x y) (mulSpec x y)) (mulSpec y y)]
    rw [← add_spec_assoc (mulSpec x x) (mulSpec y y) (addSpec (mulSpec x y) (mulSpec x y))]
  rw [rearrange]
  -- Apply sum_spec_add_distrib twice
  rw [sum_spec_add_distrib]
  rw [sum_spec_add_distrib (mulSpec x x) (mulSpec y y)]
  rw [sum_spec_add_distrib (mulSpec x y) (mulSpec x y)]
  -- Now we have: sum_spec (mul_spec x x) + sum_spec (mul_spec y y) + (sum_spec (mul_spec x y) +
  -- sum_spec (mul_spec x y))
  -- Now we need to show: sum_spec (mul_spec x x) + sum_spec (mul_spec y y) + (sum_spec (mul_spec x
  -- y) + sum_spec (mul_spec x y)) =
  -- sum_spec (mul_spec x x) + 2 * sum_spec (mul_spec x y) + sum_spec (mul_spec y y)
  -- This follows from the fact that a + a = 2 * a
  ring

/-- Bilinearity of the dot product:
$\operatorname{dot}(x+ty,x+ty)=\lVert x\rVert^2+2t\langle x,y\rangle+t^2\lVert y\rVert^2$. -/
theorem dot_quadratic_expand {s : Shape} (x y : Tensor ℝ s) (t : ℝ) :
  dot (addSpec x (scaleSpec y t)) (addSpec x (scaleSpec y t)) =
  dot x x + 2 * t * dot x y + t^2 * dot y y := by
  rw [dot_add_add]
  have h1 : dot x (scaleSpec y t) = t * dot x y := by
    -- Scale in the second argument via commutativity + `Spec.dot_scale_left`.
    calc
      dot x (scaleSpec y t) = dot (scaleSpec y t) x := by simp [dot_comm]
      _ = t * dot y x := by simpa using (Spec.dot_scale_left (a := y) (b := x) (k := t))
      _ = t * dot x y := by simp [dot_comm]
  have h2 : dot (scaleSpec y t) (scaleSpec y t) = t^2 * dot y y := by
    -- Scale both arguments by repeated use of `Spec.dot_scale_left` + commutativity.
    have hy : dot y (scaleSpec y t) = t * dot y y := by
      calc
        dot y (scaleSpec y t) = dot (scaleSpec y t) y := by simp [dot_comm]
        _ = t * dot y y := by simpa using (Spec.dot_scale_left (a := y) (b := y) (k := t))
    calc
      dot (scaleSpec y t) (scaleSpec y t)
          = t * dot y (scaleSpec y t) := by
              simpa using (Spec.dot_scale_left (a := y) (b := scaleSpec y t) (k := t))
      _ = t * (t * dot y y) := by simp [hy]
      _ = t^2 * dot y y := by ring
  rw [h1, h2]
  ring

/--
Cauchy-Schwarz inequality for tensors.
For any tensors $x$ and $y$,
$|\langle x,y\rangle|\le\lVert x\rVert\,\lVert y\rVert$.
This is a fundamental inequality in inner product spaces.
-/
theorem tensor_cauchy_schwarz {s : Shape} (x y : Tensor ℝ s) :
  |dot x y| ≤ tensorL2Norm x * tensorL2Norm y := by
  unfold tensorL2Norm

  -- Handle the degenerate case where y = 0
  by_cases hy : tensorNormSquared y = 0
  · -- Case: y = 0, so dot x y = 0 and the inequality becomes |0| ≤ ||x|| * 0 = 0
    have y_zero : y = fill (0 : ℝ) s := by
      rw [← tensor_norm_squared_zero_iff]
      exact hy
    rw [y_zero, dot_zero_right]
    -- |0| = 0 and ||x|| * ||0|| = ||x|| * 0 = 0, so 0 ≤ 0
    simp
    have h : tensorNormSquared (fill (0 : ℝ) s) = 0 := by
      rw [tensor_norm_squared_zero_iff]
    rw [h, Real.sqrt_zero, mul_zero]

  · -- Main case: y ≠ 0
    -- We use the discriminant method: for any t ∈ ℝ, ||x + ty||² ≥ 0
    -- This gives us a quadratic in t: ||x||² + 2t⟨x,y⟩ + t²||y||² ≥ 0
    -- Since this quadratic is always non-negative, its discriminant ≤ 0

    -- For any real t, we have tensor_norm_squared (add_spec x (scale_spec y t)) ≥ 0
    have quad_nonneg : ∀ t : ℝ, tensorNormSquared (addSpec x (scaleSpec y t)) ≥ 0 := by
      intro t
      exact tensor_norm_squared_nonneg (addSpec x (scaleSpec y t))

    -- The quadratic expansion
    have quad_expand : ∀ t : ℝ,
      tensorNormSquared (addSpec x (scaleSpec y t)) =
      tensorNormSquared x + 2 * t * dot x y + t^2 * tensorNormSquared y := by
      intro t
      unfold tensorNormSquared
      exact dot_quadratic_expand x y t

    -- For the quadratic at² + bt + c ≥ 0 for all t, we need discriminant b² - 4ac ≤ 0
    -- Here: a = tensor_norm_squared y, b = 2 * dot x y, c = tensor_norm_squared x
    have discriminant_nonpos : (2 * dot x y)^2 ≤ 4 * tensorNormSquared x * tensorNormSquared y
      := by
      -- The discriminant of the quadratic t²||y||² + 2t⟨x,y⟩ + ||x||² must be ≤ 0
      -- since the quadratic is always non-negative
      have quad_form : ∀ t, tensorNormSquared y * t^2 + 2 * dot x y * t + tensorNormSquared x ≥
        0 := by
        intro t
        -- Rewrite using quad_expand
        calc tensorNormSquared y * t^2 + 2 * dot x y * t + tensorNormSquared x
          = tensorNormSquared x + 2 * t * dot x y + t^2 * tensorNormSquared y := by ring
          _ = tensorNormSquared (addSpec x (scaleSpec y t)) := by rw [← quad_expand t]
          _ ≥ 0 := quad_nonneg t

      -- For a quadratic at² + bt + c ≥ 0 for all t, we need b² - 4ac ≤ 0
      -- Here a = tensor_norm_squared y ≠ 0, b = 2 * dot x y, c = tensor_norm_squared x
      have a_pos : tensorNormSquared y > 0 := by
        -- `tensor_norm_squared y` is nonnegative, so if it is not `0` it must be strictly positive.
        have hle : 0 ≤ tensorNormSquared y := tensor_norm_squared_nonneg y
        cases lt_or_eq_of_le hle with
        | inl hlt =>
          exact hlt
        | inr heq0 =>
          -- `heq0 : 0 = tensor_norm_squared y` contradicts `hy : tensor_norm_squared y ≠ 0`.
          exact False.elim (hy (by simpa using heq0.symm))

      -- Use the discriminant inequality for quadratics
      -- If at² + bt + c ≥ 0 for all t and a > 0, then b² ≤ 4ac
      -- This is a standard result in analysis
      have discriminant : (2 * dot x y)^2 - 4 * tensorNormSquared y * tensorNormSquared x ≤ 0 :=
        by
        -- The quadratic p(t) = at² + bt + c achieves its minimum at t = -b/(2a)
        -- If p(t) ≥ 0 for all t, then p(-b/(2a)) ≥ 0
        -- This gives us the discriminant condition
        let a := tensorNormSquared y
        let b := 2 * dot x y
        let c := tensorNormSquared x
        let t_min := -b / (2 * a)
        have p_min : a * t_min^2 + b * t_min + c ≥ 0 := quad_form t_min
        -- Expand p(t_min) = c - b²/(4a)
        have expand_p_min : a * t_min^2 + b * t_min + c = c - b^2 / (4 * a) := by
          -- Substitute t_min = -b/(2a) into the quadratic
          simp only [t_min]
          field_simp [a_pos.ne']
          ring
        rw [expand_p_min] at p_min
        -- From c - b²/(4a) ≥ 0 we get b² ≤ 4ac
        have : b^2 / (4 * a) ≤ c := by linarith
        have : b^2 ≤ 4 * a * c := by
          -- From this : b^2 / (4 * a) ≤ c
          -- Multiply both sides by 4 * a (which is positive)
          have h : 0 < 4 * a := by linarith
          -- We want to show b^2 ≤ 4 * a * c
          -- We have b^2 / (4 * a) ≤ c
          -- Multiply both sides by 4 * a
          have h4a : (4 * a) ≠ 0 := by
            exact mul_ne_zero (by norm_num) a_pos.ne'
          calc b^2 = b^2 / (4 * a) * (4 * a) := by
                -- `b^2 / (4a) * (4a) = b^2` since `4a ≠ 0`.
                have : b^2 / (4 * a) * (4 * a) = b^2 := by
                  calc
                    b^2 / (4 * a) * (4 * a) = (b^2 * (4 * a)) / (4 * a) := by
                      simp [div_mul_eq_mul_div]
                    _ = b^2 := by
                      simpa [mul_assoc] using (mul_div_cancel_right₀ (b^2) h4a)
                simpa using this.symm
               _ ≤ c * (4 * a) := by exact mul_le_mul_of_nonneg_right this (le_of_lt h)
               _ = 4 * a * c := by ring
        -- Substitute back
        simp only [a, b, c] at this
        -- Need to show the right multiplication order
        calc (2 * dot x y)^2 - 4 * tensorNormSquared y * tensorNormSquared x
          = b^2 - 4 * a * c := by simp only [a, b, c]
          _ ≤ 0 := by linarith

      -- From discriminant: (2 * dot x y)^2 - 4 * tensor_norm_squared y * tensor_norm_squared x ≤ 0
      -- We need: (2 * dot x y)^2 ≤ 4 * tensor_norm_squared x * tensor_norm_squared y
      -- Since multiplication is commutative: 4 * tensor_norm_squared y * tensor_norm_squared x = 4
      -- * tensor_norm_squared x * tensor_norm_squared y
      have comm : 4 * tensorNormSquared y * tensorNormSquared x = 4 * tensorNormSquared x *
        tensorNormSquared y := by ring
      linarith

    -- Simplify the discriminant inequality to get |⟨x,y⟩| ≤ ||x||||y||
    have cs_squared : (dot x y)^2 ≤ tensorNormSquared x * tensorNormSquared y := by
      -- From discriminant_nonpos: (2 * dot x y)² ≤ 4 * tensor_norm_squared x * tensor_norm_squared
      -- y
      -- Simplify: 4 * (dot x y)² ≤ 4 * tensor_norm_squared x * tensor_norm_squared y
      -- Hence: (dot x y)² ≤ tensor_norm_squared x * tensor_norm_squared y
      have h : 4 * (dot x y)^2 ≤ 4 * tensorNormSquared x * tensorNormSquared y := by
        -- (2 * dot x y)^2 = 4 * (dot x y)^2
        have expand : (2 * dot x y)^2 = 4 * (dot x y)^2 := by ring
        rw [expand] at discriminant_nonpos
        exact discriminant_nonpos
      linarith

    -- Take square roots to get the final result
    have sqrt_ineq : |dot x y| ≤ Real.sqrt (tensorNormSquared x) * Real.sqrt (tensorNormSquared
      y) := by
      -- |a|² = a² and √(a²) ≤ √b iff a² ≤ b when b ≥ 0
      have abs_sq : |dot x y|^2 = (dot x y)^2 := by
        -- |a|² = a² for any real number a
        exact sq_abs (dot x y)

      -- We want to show: |dot x y| ≤ √(tensor_norm_squared x) * √(tensor_norm_squared y)
      -- Square both sides: |dot x y|² ≤ (√(tensor_norm_squared x) * √(tensor_norm_squared y))²
      -- This becomes: (dot x y)² ≤ tensor_norm_squared x * tensor_norm_squared y
      -- which we have as cs_squared

      have rhs_sq : (Real.sqrt (tensorNormSquared x) * Real.sqrt (tensorNormSquared y))^2 =
        tensorNormSquared x * tensorNormSquared y := by
        rw [mul_pow]
        simp [Real.sq_sqrt (tensor_norm_squared_nonneg x), Real.sq_sqrt
          (tensor_norm_squared_nonneg y)]

      -- Use the fact that if a² ≤ b² and a,b ≥ 0, then a ≤ b
      have lhs_nonneg : 0 ≤ |dot x y| := by
        -- |·| is non-negative
        exact abs_nonneg (dot x y)
      have rhs_nonneg : 0 ≤ Real.sqrt (tensorNormSquared x) * Real.sqrt (tensorNormSquared y) :=
        by
        exact mul_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _)

      -- Apply square root monotonicity
      -- We have: |dot x y|² = (dot x y)² ≤ tensor_norm_squared x * tensor_norm_squared y
      -- Taking square roots: |dot x y| ≤ √(tensor_norm_squared x * tensor_norm_squared y)
      have h : |dot x y| ^ 2 ≤ (Real.sqrt (tensorNormSquared x) * Real.sqrt (tensorNormSquared
        y))^2 := by
        rw [abs_sq, rhs_sq]
        exact cs_squared
      -- Since sqrt is monotone on non-negative reals
      rw [← Real.sqrt_sq lhs_nonneg, ← Real.sqrt_sq rhs_nonneg]
      exact Real.sqrt_le_sqrt h

    exact sqrt_ineq

/--
Triangle inequality for the $\ell_2$ norm.
-/
theorem tensor_l2_norm_triangle {s : Shape} (x y : Tensor ℝ s) :
  tensorL2Norm (addSpec x y) ≤ tensorL2Norm x + tensorL2Norm y := by
  -- We prove this by showing the squared version and taking square roots
  -- Since all norms are non-negative, ||a|| ≤ ||b|| + ||c|| iff ||a||² ≤ (||b|| + ||c||)²

  -- The strategy: show tensor_norm_squared (add_spec x y) ≤ (tensor_l2_norm x + tensor_l2_norm y)²
  -- then use properties of square roots

  have squared_ineq : tensorNormSquared (addSpec x y) ≤ (tensorL2Norm x + tensorL2Norm y)^2
    := by
    -- Expand both sides
    have expand_left : tensorNormSquared (addSpec x y) =
      tensorNormSquared x + 2 * dot x y + tensorNormSquared y := by
      unfold tensorNormSquared
      exact dot_add_add x y

    have expand_right : (tensorL2Norm x + tensorL2Norm y)^2 =
      tensorNormSquared x + 2 * tensorL2Norm x * tensorL2Norm y + tensorNormSquared y := by
      unfold tensorL2Norm
      ring_nf
      simp only [Real.sq_sqrt (tensor_norm_squared_nonneg x), Real.sq_sqrt
        (tensor_norm_squared_nonneg y)]

    rw [expand_left, expand_right]

    -- Reduce to: dot x y ≤ tensor_l2_norm x * tensor_l2_norm y
    suffices h : dot x y ≤ tensorL2Norm x * tensorL2Norm y by linarith

    -- This follows from Cauchy-Schwarz: |⟨x,y⟩| ≤ ||x||||y|| implies dot x y ≤ ||x||||y||
    have cs := tensor_cauchy_schwarz x y
    have le_abs := le_abs_self (dot x y)
    exact le_trans le_abs (le_trans cs (le_refl _))

  -- Now convert the squared inequality back to the original form
  -- We have: tensor_norm_squared (add_spec x y) ≤ (tensor_l2_norm x + tensor_l2_norm y)²
  -- We want: tensor_l2_norm (add_spec x y) ≤ tensor_l2_norm x + tensor_l2_norm y
  -- This follows from the monotonicity of square root and the fact that both sides are non-negative

  unfold tensorL2Norm

  -- Apply sqrt to both sides of the squared inequality
  have rhs_nonneg : 0 ≤ tensorL2Norm x + tensorL2Norm y := by
    exact add_nonneg (tensor_l2_norm_nonneg x) (tensor_l2_norm_nonneg y)

  have sqrt_rhs : Real.sqrt ((tensorL2Norm x + tensorL2Norm y)^2) = tensorL2Norm x +
    tensorL2Norm y := by
    exact Real.sqrt_sq rhs_nonneg

  -- Use calc to chain the inequalities
  calc Real.sqrt (tensorNormSquared (addSpec x y))
    ≤ Real.sqrt ((tensorL2Norm x + tensorL2Norm y)^2) := Real.sqrt_le_sqrt squared_ineq
    _ = tensorL2Norm x + tensorL2Norm y := sqrt_rhs
    _ = Real.sqrt (tensorNormSquared x) + Real.sqrt (tensorNormSquared y) := by
      unfold tensorL2Norm; rfl
/--
Homogeneity of the $\ell_2$ norm.
-/
theorem tensor_l2_norm_scale {s : Shape} (t : Tensor ℝ s) (c : ℝ) :
  tensorL2Norm (scaleSpec t c) = |c| * tensorL2Norm t := by
  -- The homogeneity property follows from the bilinearity of the dot product
  -- ||c*t||² = ⟨c*t, c*t⟩ = c² * ⟨t, t⟩ = c² * ||t||²
  -- Taking square roots: ||c*t|| = |c| * ||t||
  unfold tensorL2Norm tensorNormSquared
  -- Goal: Real.sqrt (dot (scale_spec t c) (scale_spec t c)) = |c| * Real.sqrt (dot t t)

  -- First, we need to show that dot (scale_spec t c) (scale_spec t c) = c² * dot t t
  have h_dot_scale : dot (scaleSpec t c) (scaleSpec t c) = c * c * dot t t := by
    have h_right : dot t (scaleSpec t c) = c * dot t t := by
      calc
        dot t (scaleSpec t c) = dot (scaleSpec t c) t := by simp [dot_comm]
        _ = c * dot t t := by simpa using (Spec.dot_scale_left (a := t) (b := t) (k := c))
    calc
      dot (scaleSpec t c) (scaleSpec t c)
          = c * dot t (scaleSpec t c) := by
              simpa using (Spec.dot_scale_left (a := t) (b := scaleSpec t c) (k := c))
      _ = c * (c * dot t t) := by simp [h_right]
      _ = c * c * dot t t := by ring

  rw [h_dot_scale]
  -- Goal: Real.sqrt (c * c * dot t t) = |c| * Real.sqrt (dot t t)

  -- Key fact: c * c = |c|²
  have c_sq : c * c = |c|^2 := by
    -- c * c = c² = |c|²
    -- This follows from c² = |c|²
    rw [← sq]
    rw [sq_abs]

  rw [c_sq]
  -- Goal: Real.sqrt (|c|² * dot t t) = |c| * Real.sqrt (dot t t)

  -- Use the fact that sqrt(a² * b) = a * sqrt(b) when a ≥ 0 and b ≥ 0
  have h_nonneg : 0 ≤ dot t t := tensor_norm_squared_nonneg t
  have abs_nonneg : 0 ≤ |c| := by
    exact abs_nonneg c

  -- Apply the square root property: sqrt(a² * b) = a * sqrt(b) when a ≥ 0 and b ≥ 0
  rw [Real.sqrt_mul (sq_nonneg |c|)]
  rw [Real.sqrt_sq abs_nonneg]

end Proofs
