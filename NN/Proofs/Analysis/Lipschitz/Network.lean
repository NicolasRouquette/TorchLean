/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.Calculus.MeanValue
public import NN.Proofs.Analysis.Lipschitz.Norm
public import NN.Spec.Layers.Activation

/-!
# Lipschitz bounds for neural-network operations

This module owns Lipschitz estimates for ReLU, matrix-vector multiplication, linear layers, and
composition. The underlying real-valued tensor norm API lives in
`NN.Proofs.Analysis.Lipschitz.Norm`.

Import `NN.Proofs.Analysis.Lipschitz` for the complete norm and network-bound API.
-/

@[expose] public section

namespace Proofs

open Spec
open Tensor
open Activation
open scoped BigOperators

open Spec (dot tensorNormSquared tensor_norm_squared_nonneg
           tensor_norm_squared_zero_iff mul_spec_comm add_spec_comm dot_comm
           sum_spec_add_distrib mul_spec_add_left mul_spec_add_right
           add_spec_assoc)

-- ====================================================================
-- RELU LIPSCHITZ CONTINUITY PROOFS
-- ===================================================================

/--
Pointwise ReLU is 1-Lipschitz for scalars.
Foundation for tensor-level Lipschitz bounds.
-/
theorem relu_scalar_lipschitz (x y : ℝ) :
  |max (0 : ℝ) x - max (0 : ℝ) y| ≤ |x - y| := by
  -- ReLU is 1-Lipschitz: |max(0,x) - max(0,y)| ≤ |x - y|
  -- This follows from case analysis on the signs of x and y
  -- We'll consider four cases based on the signs of x and y
  by_cases hx : (0 : ℝ) ≤ x
  · by_cases hy : (0 : ℝ) ≤ y
    · -- Case 1: x ≥ 0 and y ≥ 0, so max 0 x = x and max 0 y = y
      simp [max_eq_right hx, max_eq_right hy]
    · -- Case 2: x ≥ 0 and y < 0, so max 0 x = x and max 0 y = 0
      push Not at hy
      simp [max_eq_right hx, max_eq_left (le_of_lt hy)]
      -- Need to show |x - 0| ≤ |x - y|
      -- Since x ≥ 0 and y < 0, we have x - y > x
      have h : (0 : ℝ) ≤ x - y := by linarith
      have hx_pos : (0 : ℝ) ≤ x := hx
      rw [abs_of_nonneg hx_pos, abs_of_nonneg h]
      simp
      linarith
  · push Not at hx
    by_cases hy : (0 : ℝ) ≤ y
    · -- Case 3: x < 0 and y ≥ 0, so max 0 x = 0 and max 0 y = y
      simp [max_eq_left (le_of_lt hx), max_eq_right hy]
      -- Need to show |0 - y| ≤ |x - y|
      -- Since x < 0 and y ≥ 0, we have |x - y| ≥ y
      have h : x - y ≤ (0 : ℝ) := by linarith
      rw [abs_of_nonneg hy, abs_of_nonpos h]
      simp
      linarith
    · -- Case 4: x < 0 and y < 0, so max 0 x = 0 and max 0 y = 0
      push Not at hy
      simp [max_eq_left (le_of_lt hx), max_eq_left (le_of_lt hy)]

/--
ReLU is 1-Lipschitz on scalar tensors.
-/
theorem relu_scalar_tensor_lipschitz (x y : Tensor ℝ .scalar) :
  tensorL2Dist (reluSpec x) (reluSpec y) ≤ tensorL2Dist x y := by
  cases x with | scalar a =>
  cases y with | scalar b =>
  unfold reluSpec tensorL2Dist tensorL2Norm tensorNormSquared dot subSpec
  simp [mapSpec, map2Spec, sumSpec, tensorFoldlSpec, mulSpec]
  unfold Math.reluSpec
  -- Goal is now about square roots of squares
  -- We need to show: √((relu a - relu b)²) ≤ √((a - b)²)
  -- Since sqrt is monotone, this is equivalent to (relu a - relu b)² ≤ (a - b)²
  apply Real.sqrt_le_sqrt
  -- Now we need: (max a 0 - max b 0)² ≤ (a - b)², which follows from the scalar Lipschitz bound.
  have h_abs : |max a 0 - max b 0| ≤ |a - b| := by
    simpa [max_comm] using (relu_scalar_lipschitz a b)
  -- Convert absolute value inequality to squared inequality
  have h_sq : |max a 0 - max b 0|^2 ≤ |a - b|^2 := by
    -- Use the fact that if 0 ≤ x ≤ y, then x² ≤ y²
    have h1 : (0 : ℝ) ≤ |max a 0 - max b 0| := abs_nonneg _
    have h2 : (0 : ℝ) ≤ |a - b| := abs_nonneg _
    -- Since |max 0 a - max 0 b| ≤ |a - b| and both are non-negative
    -- we can square both sides using monotonicity of squaring on non-negative reals
    -- Use monotonicity of squaring on non-negative reals
    -- If 0 ≤ a ≤ b, then a² ≤ b²
    -- Since |max 0 a - max 0 b| ≤ |a - b| and both are non-negative
    -- we can square both sides
    have : |max a 0 - max b 0| * |max a 0 - max b 0| ≤ |a - b| * |a - b| := by
      exact mul_self_le_mul_self h1 h_abs
    rw [← sq, ← sq] at this
    exact this
  -- Use the fact that |x|² = x²
  rw [sq_abs, sq_abs] at h_sq
  -- Convert from ^2 to multiplication
  simp only [sq] at h_sq
  exact h_sq

/--
General ReLU Lipschitz theorem for arbitrary tensor shapes.
Main result: ReLU is 1-Lipschitz in the $\ell_2$ norm for any tensor shape.
-/
theorem relu_lipschitz_general {s : Shape} (x y : Tensor ℝ s) :
  tensorL2Dist (reluSpec x) (reluSpec y) ≤ tensorL2Dist x y := by
  induction s with
  | scalar => exact relu_scalar_tensor_lipschitz x y
  | dim n s' ih =>
    cases x with | dim fx =>
    cases y with | dim fy =>
    unfold reluSpec tensorL2Dist tensorL2Norm tensorNormSquared dot subSpec
    simp [mapSpec, map2Spec, sumSpec, mulSpec, tensorFoldlSpec]
    -- The key insight: for vectors, ||relu(x) - relu(y)||² = Σᵢ (relu(xᵢ) - relu(yᵢ))²
    -- and ||x - y||² = Σᵢ (xᵢ - yᵢ)²
    -- Since ReLU is 1-Lipschitz componentwise, each term satisfies (relu(xᵢ) - relu(yᵢ))² ≤ (xᵢ -
    -- yᵢ)²
    apply Real.sqrt_le_sqrt
    -- We need to show the sum of squared differences is preserved
    -- This requires showing the fold preserves the inequality
    suffices ∀ k acc_relu acc_orig, k ≤ n → acc_relu ≤ acc_orig →
      tensorFoldlSpec.go (· + ·) n s'
        (fun i => mulSpec (subSpec (mapSpec Math.reluSpec (fx i)) (mapSpec Math.reluSpec (fy
          i)))
                          (subSpec (mapSpec Math.reluSpec (fx i)) (mapSpec Math.reluSpec (fy
                            i)))) k acc_relu ≤
      tensorFoldlSpec.go (· + ·) n s'
        (fun i => mulSpec (subSpec (fx i) (fy i)) (subSpec (fx i) (fy i))) k acc_orig by
      exact this 0 (0 : ℝ) (0 : ℝ) (Nat.zero_le n) (le_refl (0 : ℝ))

    intro k acc_relu acc_orig hk hacc
    induction hn : n - k generalizing k acc_relu acc_orig with
    | zero =>
      have k_eq_n : k = n := by grind
      subst k
      have hgo_relu :
          tensorFoldlSpec.go (· + ·) n s'
              (fun i =>
                mulSpec
                  (subSpec (mapSpec Math.reluSpec (fx i)) (mapSpec Math.reluSpec (fy i)))
                  (subSpec (mapSpec Math.reluSpec (fx i)) (mapSpec Math.reluSpec (fy i))))
              n acc_relu
            = acc_relu := by
        simpa using
          (Spec.tensor_foldl_spec_go_of_not_lt (f := (· + ·))
              (values := fun i =>
                mulSpec
                  (subSpec (mapSpec Math.reluSpec (fx i)) (mapSpec Math.reluSpec (fy i)))
                  (subSpec (mapSpec Math.reluSpec (fx i)) (mapSpec Math.reluSpec (fy i))))
              (k := n) (acc := acc_relu) (by simp))
      have hgo_orig :
          tensorFoldlSpec.go (· + ·) n s' (fun i => mulSpec (subSpec (fx i) (fy i)) (subSpec (fx i) (fy i)))
              n acc_orig
            = acc_orig := by
        simpa using
          (Spec.tensor_foldl_spec_go_of_not_lt (f := (· + ·))
              (values := fun i =>
                mulSpec (subSpec (fx i) (fy i)) (subSpec (fx i) (fy i)))
              (k := n) (acc := acc_orig) (by simp))
      simpa [hgo_relu, hgo_orig] using hacc
    | succ m ih_fold =>
      have hlt : k < n := by grind
      -- Peel one `go` step on both sides.
      rw [Spec.tensor_foldl_spec_go_of_lt (f := (· + ·))
        (values := fun i =>
          mulSpec
            (subSpec (mapSpec Math.reluSpec (fx i)) (mapSpec Math.reluSpec (fy i)))
            (subSpec (mapSpec Math.reluSpec (fx i)) (mapSpec Math.reluSpec (fy i))))
        (k := k) (acc := acc_relu) hlt]
      rw [Spec.tensor_foldl_spec_go_of_lt (f := (· + ·))
        (values := fun i =>
          mulSpec (subSpec (fx i) (fy i)) (subSpec (fx i) (fy i)))
        (k := k) (acc := acc_orig) hlt]
      have h_next : n - (k + 1) = m := by grind
      have k_plus_one_le : k + 1 ≤ n := Nat.succ_le_of_lt hlt
      -- Need to show the accumulated inequality is preserved first
      have component_ineq :
        sumSpec (mulSpec (subSpec (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                     (mapSpec Math.reluSpec (fy ⟨k, hlt⟩)))
                          (subSpec (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                     (mapSpec Math.reluSpec (fy ⟨k, hlt⟩)))) ≤
        sumSpec (mulSpec (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩))
                          (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩))) := by
        -- This is the squared L2 distance for component k
        -- We need ||relu(fx[k]) - relu(fy[k])||² ≤ ||fx[k] - fy[k]||²
        have h_comp : tensorL2Dist (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                    (mapSpec Math.reluSpec (fy ⟨k, hlt⟩)) ≤
                     tensorL2Dist (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩) := by
          -- Apply the induction hypothesis to component k
          have : reluSpec (fx ⟨k, hlt⟩) = mapSpec Math.reluSpec (fx ⟨k, hlt⟩) := by
            unfold reluSpec
            rfl
          have : reluSpec (fy ⟨k, hlt⟩) = mapSpec Math.reluSpec (fy ⟨k, hlt⟩) := by
            unfold reluSpec
            rfl
          simp
          exact ih (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩)
        -- Square both sides to get the desired inequality
        unfold tensorL2Dist tensorL2Norm at h_comp
        have h_sq : Real.sqrt (tensorNormSquared (subSpec (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                                            (mapSpec Math.reluSpec (fy ⟨k,
                                                              hlt⟩)))) ≤
                   Real.sqrt (tensorNormSquared (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩))) := h_comp
        -- Apply Real.le_sqrt_iff_sq_le_sq to get the squared inequality
        have h_sq' : tensorNormSquared (subSpec (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                                  (mapSpec Math.reluSpec (fy ⟨k, hlt⟩))) ≤
                     tensorNormSquared (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩)) := by
          -- From h_sq: √a ≤ √b, we want to show a ≤ b
          -- Since sqrt is strictly monotone on non-negative reals
          have ha := tensor_norm_squared_nonneg (subSpec (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                                           (mapSpec Math.reluSpec (fy ⟨k, hlt⟩)))
          have hb := tensor_norm_squared_nonneg (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩))
          -- sqrt is monotone, so √a ≤ √b implies a ≤ b when both args are non-negative
          -- sqrt is monotone, so √a ≤ √b implies a ≤ b when both args are non-negative
          -- Use the fact that sqrt is monotone on non-negative reals
          -- If √a ≤ √b and a,b ≥ 0, then a ≤ b
          -- Since sqrt is strictly monotone on non-negative reals
          -- We can use the fact that if sqrt(a) ≤ sqrt(b) then a ≤ b
          have : Real.sqrt (tensorNormSquared (subSpec (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                                          (mapSpec Math.reluSpec (fy ⟨k, hlt⟩))))
                                                            ≤
                 Real.sqrt (tensorNormSquared (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩))) := h_sq
          -- Apply monotonicity of sqrt: if √a ≤ √b and a,b ≥ 0, then a ≤ b
          -- Since sqrt is strictly monotone on non-negative reals, √a ≤ √b implies a ≤ b
          -- We'll prove this by contradiction
          by_contra h_not_le
          push Not at h_not_le
          -- If a > b, then √a > √b
          have h_sqrt_gt : Real.sqrt (tensorNormSquared (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩))) <
                           Real.sqrt (tensorNormSquared (subSpec (mapSpec Math.reluSpec (fx ⟨k,
                             hlt⟩))
                                                                   (mapSpec Math.reluSpec (fy ⟨k,
                                                                     hlt⟩)))) := by
            exact Real.sqrt_lt_sqrt hb h_not_le
          -- But this contradicts our assumption that √a ≤ √b
          linarith
        unfold tensorNormSquared dot at h_sq'
        exact h_sq'
      -- Add the inequalities
      -- We have: acc_relu ≤ acc_orig (from hacc)
      -- We have: component_ineq tells us the sum of squared differences for ReLU is ≤ the original
      -- For the fold with addition, we need to show that adding these to the accumulators preserves
      -- the inequality
      -- Note that tensor_foldl_spec (· + ·) acc t adds sum_spec t to acc
      have h1 : tensorFoldlSpec (· + ·) acc_relu
                  (mulSpec (subSpec (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                     (mapSpec Math.reluSpec (fy ⟨k, hlt⟩)))
                           (subSpec (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                    (mapSpec Math.reluSpec (fy ⟨k, hlt⟩)))) =
                acc_relu + sumSpec (mulSpec (subSpec (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                                       (mapSpec Math.reluSpec (fy ⟨k, hlt⟩)))
                                             (subSpec (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                                      (mapSpec Math.reluSpec (fy ⟨k, hlt⟩)))) :=
                                                        by
        simpa using
          (Spec.tensor_foldl_spec_add_init (s := s')
            (acc := acc_relu)
            (t :=
              mulSpec (subSpec (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                (mapSpec Math.reluSpec (fy ⟨k, hlt⟩)))
                       (subSpec (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                (mapSpec Math.reluSpec (fy ⟨k, hlt⟩)))))
      have h2 : tensorFoldlSpec (· + ·) acc_orig
                  (mulSpec (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩))
                           (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩))) =
                acc_orig + sumSpec (mulSpec (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩))
                                             (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩))) := by
        simpa using
          (Spec.tensor_foldl_spec_add_init (s := s')
            (acc := acc_orig)
            (t := mulSpec (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩))
                          (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩))))
      rw [h1, h2]

      -- Apply IH to the recursive call
      -- First, show the new accumulators maintain the inequality
      have new_acc_ineq : tensorFoldlSpec (· + ·) acc_relu
                            (mulSpec (subSpec (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                               (mapSpec Math.reluSpec (fy ⟨k, hlt⟩)))
                                     (subSpec (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                              (mapSpec Math.reluSpec (fy ⟨k, hlt⟩)))) ≤
                          tensorFoldlSpec (· + ·) acc_orig
                            (mulSpec (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩))
                                     (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩))) := by
        rw [h1, h2]
        linarith [hacc, component_ineq]

      -- Apply IH to the recursive call with the updated accumulators
      -- First, we need to convert new_acc_ineq to the right form
      have new_acc_ineq' : (acc_relu + sumSpec (mulSpec (subSpec (mapSpec Math.reluSpec (fx ⟨k,
        hlt⟩))
                                                                  (mapSpec Math.reluSpec (fy ⟨k,
                                                                    hlt⟩)))
                                               (subSpec (mapSpec Math.reluSpec (fx ⟨k, hlt⟩))
                                                        (mapSpec Math.reluSpec (fy ⟨k, hlt⟩))))) ≤
                          (acc_orig + sumSpec (mulSpec (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩))
                                                       (subSpec (fx ⟨k, hlt⟩) (fy ⟨k, hlt⟩)))) :=
                                                         by
        linarith [hacc, component_ineq]
      exact ih_fold (k + 1) _ _ k_plus_one_le new_acc_ineq' h_next

/--
Rank-one ReLU is 1-Lipschitz in $\ell_2$.

This theorem is just the vector specialization of `relu_lipschitz_general`, but it is convenient
for callers working with ordinary `.dim n .scalar` activations.
-/
theorem relu_vector_lipschitz {n : Nat} (x y : Tensor ℝ [n]) :
  tensorL2Dist (reluSpec x) (reluSpec y) ≤ tensorL2Dist x y := by
  simpa using (relu_lipschitz_general (s := .dim n .scalar) x y)

-- Linear-operator norm bounds for affine layers and matrix products.

/--
Tensor subtraction can be rewritten as addition of a `-1` scale.

This is a small algebraic normal form used by linear-operator proofs, where it is often easier to
reuse additive and scaling lemmas than reason about `subSpec` directly.
-/
theorem sub_spec_eq_add_scale_neg_one {s : Shape} (a b : Tensor ℝ s) :
  subSpec a b = addSpec a (scaleSpec b (-1 : ℝ)) := by
  induction s with
  | scalar =>
    cases a with
    | scalar x =>
      cases b with
      | scalar y =>
        simp [subSpec, addSpec, scaleSpec, map2Spec, mapSpec]
        ring
  | dim n s ih =>
    cases a with
    | dim fa =>
      cases b with
      | dim fb =>
        simp [subSpec, addSpec, scaleSpec, map2Spec, mapSpec]
        funext i
        simpa [subSpec, addSpec, scaleSpec, map2Spec, mapSpec] using ih (fa i) (fb i)

/-- Subtracting the zero tensor on the right leaves the tensor unchanged. -/
theorem sub_spec_zero_right {s : Shape} (t : Tensor ℝ s) :
  subSpec t (fill (0 : ℝ) s) = t := by
  induction s with
  | scalar =>
    cases t with
    | scalar x =>
      simp [subSpec, map2Spec, fill]
  | dim n s ih =>
    cases t with
    | dim f =>
      simp [subSpec, map2Spec, fill]
      funext i
      exact ih (f i)

set_option linter.auxLemma false in
/--
Matrix-vector multiplication sends the zero vector to the zero vector.

The proof follows the spec definition: each output coordinate is a fold over scalar products, and
every scalar product contains a zero input coordinate.
-/
theorem mat_vec_mul_spec_zero {m n : Nat} (W : Tensor ℝ [m, n]) :
  matVecMulSpec W (fill (0 : ℝ) (.dim n .scalar)) = fill (0 : ℝ) (.dim m .scalar) := by
  classical
  cases W with
  | dim rowsA =>
    -- Both sides are vectors; prove pointwise.
    apply congrArg Tensor.dim
    funext i
    cases hrow : rowsA i with
    | dim colsA =>
      -- Reduce to a scalar list fold.
      simp [fill]
      -- The values vector is identically zero, so each step adds `ak * 0 = 0`.
      have hfold :
          (List.finRange n).foldl
              (fun (s : ℝ) (k : Fin n) =>
                Spec.matMulSpec.match_1
                  (motive := fun _ _ => ℝ)
                  (colsA k) (Tensor.scalar (0 : ℝ))
                  (fun ak vk => s + ak * vk))
              0
            =
            0 := by
        -- Each step is `s ↦ s + ak * 0 = s`, so the fold returns the initial accumulator.
        let f : ℝ → Fin n → ℝ := fun s k =>
          Spec.matMulSpec.match_1
            (motive := fun _ _ => ℝ)
            (colsA k) (Tensor.scalar (0 : ℝ))
            (fun ak vk => s + ak * vk)
        have hf : ∀ s k, f s k = s := by
          intro s k
          cases hcol : colsA k with
          | scalar ak =>
            simp [f, hcol]
        -- Replace the fold function with `f`, then it is the identity on the accumulator.
        change (List.finRange n).foldl f 0 = 0
        induction (List.finRange n) with
        | nil =>
          simp [List.foldl]
        | cons hd tl ih =>
          simp [List.foldl, hf, ih]
      -- Convert the scalar-tensor fold in `mat_vec_mul_spec` to an ℝ fold, then apply `hfold`.
      have hscalar :
          (List.finRange n).foldl
              (fun (acc : Tensor ℝ .scalar) (k : Fin n) =>
                Spec.matVecMulSpec.match_1
                  (motive := fun _ _ _ => Tensor ℝ .scalar)
                  acc (colsA k) (Tensor.scalar (0 : ℝ))
                  (fun s ak vk => Tensor.scalar (s + ak * vk)))
              (Tensor.scalar 0)
            =
            Tensor.scalar
              ((List.finRange n).foldl
                (fun (s : ℝ) (k : Fin n) =>
                  Spec.matMulSpec.match_1
                    (motive := fun _ _ => ℝ)
                    (colsA k) (Tensor.scalar (0 : ℝ))
                    (fun ak vk => s + ak * vk))
                0) := by
        exact
          (Spec.foldl_matvec_scalar (l := List.finRange n) (a := 0) (cols := colsA)
            (vals := fun _ => Tensor.scalar (0 : ℝ)))
      -- Finish by reducing to the ℝ fold value.
      rw [hscalar]
      simp [hfold]

/--
Frobenius norm of a matrix tensor.

We use the Frobenius-norm-style bound:

$$
\lVert W\rVert_F
=\sqrt{\sum_i\lVert\operatorname{row}_i(W)\rVert_2^2},
$$

This is an upper bound for the induced Euclidean operator norm; it is not the spectral norm.
-/
noncomputable def matrixFrobeniusNorm {m n : Nat} (W : Tensor ℝ [m, n]) : ℝ :=
  Real.sqrt (∑ i : Fin m, tensorNormSquared (get W i))

/--
Compatibility between two row/column access views:

`getScalar (get W i) j` and `get2 W i j` name the same scalar entry of a matrix tensor.
-/
private lemma getScalar_get_eq_get2 {m n : Nat}
    (W : Tensor ℝ [m, n]) (i : Fin m) (j : Fin n) :
    getScalar (get W i) j = get2 W i j := by
  classical
  cases W with
  | dim rows =>
    cases hrow : rows i with
    | dim cols =>
      cases hcol : cols j with
      | scalar v =>
        simp [Spec.Tensor.getScalar, Spec.get, Spec.get2, hrow, hcol]

/--
Each coordinate of `matVecMulSpec W x` is the dot product of the corresponding matrix row with
`x`.

This is the coordinate bridge used by the Frobenius/operator-norm bound below.
-/
private lemma mat_vec_coord_eq_dot_row {m n : Nat}
    (W : Tensor ℝ [m, n])
    (x : Tensor ℝ [n]) (i : Fin m) :
    getScalar (matVecMulSpec W x) i = dot (get W i) x := by
  classical
  -- Expand both sides as `Finset.univ` sums and match terms.
  rw [getScalar_mat_vec_mul_spec (A := W) (v := x) (i := i)]
  rw [dot_vec_eq_sum (a := get W i) (b := x)]
  refine Finset.sum_congr rfl ?_
  intro j _
  simp [getScalar_get_eq_get2 (W := W) (i := i) (j := j)]

/-- The Frobenius norm bounds matrix-vector multiplication in the Euclidean norm. -/
theorem matVec_norm_le_frobenius {m n : Nat}
  (W : Tensor ℝ [m, n])
  (x : Tensor ℝ [n]) :
  tensorL2Norm (matVecMulSpec W x) ≤ matrixFrobeniusNorm W * tensorL2Norm x := by
  classical
  -- Work with the squared form and then apply `Real.sqrt`.
  have hsum_nonneg : 0 ≤ ∑ i : Fin m, tensorNormSquared (get W i) := by
    have : 0 ≤ ∑ i ∈ (Finset.univ : Finset (Fin m)), tensorNormSquared (get W i) := by
      refine Finset.sum_nonneg ?_
      intro i _
      exact tensor_norm_squared_nonneg (t := get W i)
    simpa using this

  have hsquared :
      tensorNormSquared (matVecMulSpec W x) ≤
        (∑ i : Fin m, tensorNormSquared (get W i)) * tensorNormSquared x := by
    -- Expand `‖W x‖²` as a sum of squared coordinates.
    have hnormsq :
        tensorNormSquared (matVecMulSpec W x) =
          ∑ i : Fin m, (getScalar (matVecMulSpec W x) i) * (getScalar (matVecMulSpec W x) i) := by
      simpa [tensorNormSquared] using
        (dot_vec_eq_sum (a := matVecMulSpec W x) (b := matVecMulSpec W x))

    -- Bound each coordinate via Cauchy–Schwarz on the corresponding row.
    have hterm :
        ∀ i : Fin m,
          (getScalar (matVecMulSpec W x) i) * (getScalar (matVecMulSpec W x) i) ≤
            tensorNormSquared (get W i) * tensorNormSquared x := by
      intro i
      have hcoord : getScalar (matVecMulSpec W x) i = dot (get W i) x :=
        mat_vec_coord_eq_dot_row (W := W) (x := x) (i := i)
      have cs :
          |dot (get W i) x| ≤ tensorL2Norm (get W i) * tensorL2Norm x :=
        tensor_cauchy_schwarz (x := get W i) (y := x)
      have cs2 :
          (dot (get W i) x) ^ 2 ≤ (tensorL2Norm (get W i) * tensorL2Norm x) ^ 2 := by
        -- Square both sides of `cs` via `mul_le_mul`.
        have hmul :
            |dot (get W i) x| * |dot (get W i) x| ≤
              (tensorL2Norm (get W i) * tensorL2Norm x) *
                (tensorL2Norm (get W i) * tensorL2Norm x) := by
          refine mul_le_mul cs cs (abs_nonneg (dot (get W i) x)) ?_
          exact mul_nonneg (tensor_l2_norm_nonneg (get W i)) (tensor_l2_norm_nonneg x)
        have hsq :
            (|dot (get W i) x|) ^ 2 ≤ (tensorL2Norm (get W i) * tensorL2Norm x) ^ 2 := by
          simpa [pow_two] using hmul
        simpa [sq_abs] using hsq

      have row_sq : (tensorL2Norm (get W i)) ^ 2 = tensorNormSquared (get W i) := by
        unfold tensorL2Norm
        simp [Real.sq_sqrt (tensor_norm_squared_nonneg (t := get W i))]
      have x_sq : (tensorL2Norm x) ^ 2 = tensorNormSquared x := by
        unfold tensorL2Norm
        simp [Real.sq_sqrt (tensor_norm_squared_nonneg (t := x))]
      have rhs_sq :
          (tensorL2Norm (get W i) * tensorL2Norm x) ^ 2 =
            tensorNormSquared (get W i) * tensorNormSquared x := by
        -- `(a*b)^2 = a^2 * b^2`, then unfold the squares of the norms.
        simp [mul_pow, row_sq, x_sq]

      have hsq :
          (getScalar (matVecMulSpec W x) i) ^ 2 ≤
            tensorNormSquared (get W i) * tensorNormSquared x := by
        -- Replace the coordinate by the row dot-product and use `cs2`.
        simpa [hcoord, rhs_sq] using cs2

      -- Convert `a^2` back into `a*a`.
      simpa [pow_two] using hsq

    -- Sum the coordinate-wise bounds and factor out `‖x‖²`.
    have hsum_le :
        (∑ i : Fin m,
              (getScalar (matVecMulSpec W x) i) * (getScalar (matVecMulSpec W x) i)) ≤
          ∑ i : Fin m, tensorNormSquared (get W i) * tensorNormSquared x := by
      have :
          (∑ i ∈ (Finset.univ : Finset (Fin m)),
                (getScalar (matVecMulSpec W x) i) * (getScalar (matVecMulSpec W x) i)) ≤
            ∑ i ∈ (Finset.univ : Finset (Fin m)), tensorNormSquared (get W i) *
              tensorNormSquared x := by
        refine Finset.sum_le_sum ?_
        intro i _
        exact hterm i
      simpa using this

    have hfactor :
        (∑ i : Fin m, tensorNormSquared (get W i) * tensorNormSquared x) =
          (∑ i : Fin m, tensorNormSquared (get W i)) * tensorNormSquared x := by
      have h :=
        (Finset.sum_mul (s := (Finset.univ : Finset (Fin m)))
          (f := fun i : Fin m => tensorNormSquared (get W i)) (a := tensorNormSquared x))
      simpa using h.symm

    -- Put everything together.
    calc
      tensorNormSquared (matVecMulSpec W x)
          = ∑ i : Fin m,
              (getScalar (matVecMulSpec W x) i) * (getScalar (matVecMulSpec W x) i) := hnormsq
      _ ≤ ∑ i : Fin m, tensorNormSquared (get W i) * tensorNormSquared x := hsum_le
      _ = (∑ i : Fin m, tensorNormSquared (get W i)) * tensorNormSquared x := hfactor

  -- Take square roots and rewrite the RHS using `Real.sqrt_mul`.
  unfold matrixFrobeniusNorm tensorL2Norm
  have hsqrt := Real.sqrt_le_sqrt hsquared
  -- Rewrite `√(A * B)` as `√A * √B` with `A = ∑ i, ‖row_i‖² ≥ 0`.
  rw [Real.sqrt_mul hsum_nonneg (tensorNormSquared x)] at hsqrt
  simpa using hsqrt

/--
Linear transformations preserve $\ell_2$-norm bounds.
Fundamental theorem for neural network stability analysis.
-/
theorem linear_op_norm_bound {m n : Nat}
  (W : Tensor ℝ [m, n])
  (x y : Tensor ℝ [n]) :
  tensorL2Dist (matVecMulSpec W x) (matVecMulSpec W y) ≤
  matrixFrobeniusNorm W * tensorL2Dist x y := by
  have h_linear : matVecMulSpec W (subSpec x y) =
    subSpec (matVecMulSpec W x) (matVecMulSpec W y) := by
    -- Express subtraction as addition with scaling, then use `mat_vec_add`/`mat_vec_scale`.
    rw [sub_spec_eq_add_scale_neg_one (a := x) (b := y)]
    rw [Spec.mat_vec_add]
    rw [Spec.mat_vec_scale]
    -- Rewrite the RHS subtraction similarly.
    simp [sub_spec_eq_add_scale_neg_one]

  unfold tensorL2Dist
  rw [← h_linear]
  exact matVec_norm_le_frobenius W (subSpec x y)

-- Composition theorems for building network-level Lipschitz bounds.

/--
Composition of Lipschitz functions preserves Lipschitz property.
Essential for analyzing deep neural networks.
-/
theorem lipschitz_composition {s t u : Shape}
  (f : Tensor ℝ s → Tensor ℝ t) (g : Tensor ℝ t → Tensor ℝ u)
  (Lf Lg : ℝ)
  (hf : ∀ x y, tensorL2Dist (f x) (f y) ≤ Lf * tensorL2Dist x y)
  (hg : ∀ x y, tensorL2Dist (g x) (g y) ≤ Lg * tensorL2Dist x y)
  (hLg : 0 ≤ Lg)
  (x y : Tensor ℝ s) :
  tensorL2Dist (g (f x)) (g (f y)) ≤ (Lg * Lf) * tensorL2Dist x y := by
  calc tensorL2Dist (g (f x)) (g (f y))
    ≤ Lg * tensorL2Dist (f x) (f y)     := hg (f x) (f y)
    _ ≤ Lg * (Lf * tensorL2Dist x y)    := by
      apply mul_le_mul_of_nonneg_left
      exact hf x y
      exact hLg
    _ = (Lg * Lf) * tensorL2Dist x y    := by ring

/--
ReLU + Linear composition Lipschitz bound.
Practical theorem for single neural network layer analysis.
-/
theorem relu_linear_lipschitz {m n : Nat}
  (W : Tensor ℝ [m, n])
  (x y : Tensor ℝ [n]) :
  tensorL2Dist (reluSpec (matVecMulSpec W x)) (reluSpec (matVecMulSpec W y)) ≤
  matrixFrobeniusNorm W * tensorL2Dist x y := by
  calc tensorL2Dist (reluSpec (matVecMulSpec W x)) (reluSpec (matVecMulSpec W y))
    ≤ tensorL2Dist (matVecMulSpec W x) (matVecMulSpec W y)  :=
      relu_lipschitz_general (matVecMulSpec W x) (matVecMulSpec W y)
    _ ≤ matrixFrobeniusNorm W * tensorL2Dist x y                        :=
      linear_op_norm_bound W x y

-- ====================================================================
-- SPECIALIZED ACTIVATION FUNCTION ANALYSIS
-- ====================================================================

end Proofs
