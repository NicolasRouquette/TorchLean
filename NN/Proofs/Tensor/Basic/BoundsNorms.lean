/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Basic.LinearAlgebra

/-!
Bounds and norm facts for dependent tensors.

The lemmas here support proof developments that need componentwise bounds, norm estimates, and
simple inequalities over tensor-shaped data.
-/

@[expose] public section

namespace Spec

open Tensor
open scoped BigOperators

/-- Sum distributes over elementwise addition. -/
theorem sum_spec_add_distrib {s : Shape} (a b : Tensor ℝ s) :
  sumSpec (addSpec a b) = sumSpec a + sumSpec b := by
  unfold sumSpec addSpec
  induction s with
  | scalar =>
    cases a; cases b
    simp [tensorFoldlSpec, map2Spec]
  | dim n s ih =>
    cases a with | dim fa =>
    cases b with | dim fb =>
    simp [tensorFoldlSpec, map2Spec]
    -- We need to show that folding over the sum equals sum of the folds
    -- This uses properties of fold with addition
    -- tensor_foldl_spec.go (· + ·) n s (fun i => add_spec (fa i) (fb i)) 0 0 =
    -- tensor_foldl_spec.go (· + ·) n s fa 0 0 + tensor_foldl_spec.go (· + ·) n s fb 0 0
    have h : ∀ k acc1 acc2 acc3, k ≤ n → acc3 = acc1 + acc2 →
      tensorFoldlSpec.go (· + ·) n s (fun i => addSpec (fa i) (fb i)) k acc3 =
      tensorFoldlSpec.go (· + ·) n s fa k acc1 + tensorFoldlSpec.go (· + ·) n s fb k acc2 := by
      intro k acc1 acc2 acc3 hk hacc
      induction hn : n - k generalizing k acc1 acc2 acc3 with
      | zero =>
        have k_eq_n : k = n := by
          exact Nat.le_antisymm hk (Nat.sub_eq_zero_iff_le.mp hn)
        subst k
        simp [tensor_foldl_spec_go_of_not_lt, hacc]
      | succ m ih_fold =>
        have hlt : k < n := by
          exact Nat.sub_pos_iff_lt.mp (by simp [hn])
        -- Peel one loop step at index `k` for each `go`.
        rw [tensor_foldl_spec_go_of_lt (f := (· + ·)) (values := fun i => addSpec (fa i) (fb i))
          (k := k) (acc := acc3) hlt]
        rw [tensor_foldl_spec_go_of_lt (f := (· + ·)) (values := fa) (k := k) (acc := acc1) hlt]
        rw [tensor_foldl_spec_go_of_lt (f := (· + ·)) (values := fb) (k := k) (acc := acc2) hlt]
        have h_next : n - (k + 1) = m := by
          simp [Nat.sub_succ, hn]
        -- Apply IH with updated accumulators
        let new_acc1 := tensorFoldlSpec (· + ·) acc1 (fa ⟨k, hlt⟩)
        let new_acc2 := tensorFoldlSpec (· + ·) acc2 (fb ⟨k, hlt⟩)
        let new_acc3 := tensorFoldlSpec (· + ·) acc3 (addSpec (fa ⟨k, hlt⟩) (fb ⟨k, hlt⟩))
        -- Need to show new_acc3 = new_acc1 + new_acc2
        have acc_eq : new_acc3 = new_acc1 + new_acc2 := by
          simp only [new_acc1, new_acc2, new_acc3]
          -- tensor_foldl_spec with addition just adds sum_spec to accumulator
          have h1 : tensorFoldlSpec (· + ·) acc3 (addSpec (fa ⟨k, hlt⟩) (fb ⟨k, hlt⟩)) =
                    acc3 + sumSpec (addSpec (fa ⟨k, hlt⟩) (fb ⟨k, hlt⟩)) := by
            simpa using (tensor_foldl_spec_add_init (s := s) (acc := acc3)
              (t := addSpec (fa ⟨k, hlt⟩) (fb ⟨k, hlt⟩)))
          have h2 : tensorFoldlSpec (· + ·) acc1 (fa ⟨k, hlt⟩) = acc1 + sumSpec (fa ⟨k, hlt⟩) :=
            by
            simpa using (tensor_foldl_spec_add_init (s := s) (acc := acc1) (t := fa ⟨k, hlt⟩))
          have h3 : tensorFoldlSpec (· + ·) acc2 (fb ⟨k, hlt⟩) = acc2 + sumSpec (fb ⟨k, hlt⟩) :=
            by
            simpa using (tensor_foldl_spec_add_init (s := s) (acc := acc2) (t := fb ⟨k, hlt⟩))
          rw [h1, h2, h3]
          -- Now we have: acc3 + sum_spec (add_spec (fa ⟨k, hlt⟩) (fb ⟨k, hlt⟩)) =
          --              acc1 + sum_spec (fa ⟨k, hlt⟩) + (acc2 + sum_spec (fb ⟨k, hlt⟩))
          -- We can use ih: sum_spec (add_spec a b) = sum_spec a + sum_spec b
          -- First show that sum_spec (add_spec (fa ⟨k, hlt⟩) (fb ⟨k, hlt⟩)) = sum_spec (fa ⟨k,
          -- hlt⟩) + sum_spec (fb ⟨k, hlt⟩)
          have sum_add : sumSpec (addSpec (fa ⟨k, hlt⟩) (fb ⟨k, hlt⟩)) = sumSpec (fa ⟨k, hlt⟩) +
            sumSpec (fb ⟨k, hlt⟩) := by
            -- Apply the induction hypothesis
            unfold sumSpec
            unfold addSpec
            exact ih (fa ⟨k, hlt⟩) (fb ⟨k, hlt⟩)
          rw [sum_add, hacc]
          ring
        exact ih_fold (k + 1) new_acc1 new_acc2 new_acc3 (Nat.succ_le_of_lt hlt) acc_eq h_next
    exact h 0 0 0 0 (Nat.zero_le n) (by ring : (0 : ℝ) = 0 + 0)

/--
Dot product properties.
Essential for gradient computations and neural network training.
-/
theorem dot_comm {s : Shape} (a b : Tensor ℝ s) :
  dot a b = dot b a := by
  simp [dot, mul_spec_comm]

/-- Dot-product distributes over addition in the left argument. -/
theorem dot_add_left {s : Shape} (a b c : Tensor ℝ s) :
  dot (addSpec a b) c = dot a c + dot b c := by
  simp [dot]
  -- Reduce to distributivity of `mul_spec` and `sum_spec_add_distrib`.
  have hmul : mulSpec (addSpec a b) c = addSpec (mulSpec a c) (mulSpec b c) := by
    -- Structural recursion on `s` and use distributivity of ℝ.
    induction s with
    | scalar =>
      cases a; cases b; cases c
      simp [mulSpec, addSpec, map2Spec]
      ring
    | dim n s ih =>
      cases a with | dim fa =>
      cases b with | dim fb =>
      cases c with | dim fc =>
      simp [mulSpec, addSpec, map2Spec]
      funext i
      exact ih (fa i) (fb i) (fc i)
  rw [hmul]
  simpa using sum_spec_add_distrib (a := mulSpec a c) (b := mulSpec b c)

/-- Scaling a tensor scales its dot-product: `dot (scale_spec a k) b = k * dot a b`. -/
theorem dot_scale_left {s : Shape} (a b : Tensor ℝ s) (k : ℝ) :
  dot (scaleSpec a k) b = k * dot a b := by
  -- Induction on the tensor shape.
  induction s with
  | scalar =>
    cases a with
    | scalar x =>
      cases b with
      | scalar y =>
        simp [dot, scaleSpec, sumSpec, tensorFoldlSpec, mulSpec, mapSpec, map2Spec]
        ring
  | dim n s ih =>
    cases a with
    | dim fa =>
      cases b with
      | dim fb =>
        -- Reduce both sides to scaling of the outer fold.
        let scaled : Fin n → Tensor ℝ s := fun i => mulSpec (scaleSpec (fa i) k) (fb i)
        let unscaled : Fin n → Tensor ℝ s := fun i => mulSpec (fa i) (fb i)

        have component : ∀ i : Fin n, sumSpec (scaled i) = k * sumSpec (unscaled i) := by
          intro i
          simpa [dot, scaled, unscaled] using ih (a := fa i) (b := fb i)

        have go_scale : ∀ j acc,
            tensorFoldlSpec.go (· + ·) n s scaled j (k * acc) =
              k * tensorFoldlSpec.go (· + ·) n s unscaled j acc := by
          intro j acc
          induction hn : n - j generalizing j acc with
          | zero =>
            have hnot : ¬ j < n := by
              exact Nat.not_lt.mpr (Nat.sub_eq_zero_iff_le.mp hn)
            -- Both `go` loops terminate and return the accumulator.
            simp
              [tensor_foldl_spec_go_of_not_lt (f := (· + ·)) (values := scaled) (k := j) (acc := k * acc) hnot,
                tensor_foldl_spec_go_of_not_lt (f := (· + ·)) (values := unscaled) (k := j) (acc := acc) hnot]
          | succ m ihj =>
            have hlt : j < n := by
              exact Nat.sub_pos_iff_lt.mp (by simp [hn])
            rw [tensor_foldl_spec_go_of_lt (f := (· + ·)) (values := scaled) (k := j) (acc := k * acc) hlt]
            rw [tensor_foldl_spec_go_of_lt (f := (· + ·)) (values := unscaled) (k := j) (acc := acc) hlt]
            have h_next : n - (j + 1) = m := by
              simp [Nat.sub_succ, hn]
            -- Show that the accumulator update scales correctly.
            have step :
                tensorFoldlSpec (· + ·) (k * acc) (scaled ⟨j, hlt⟩) =
                  k * tensorFoldlSpec (· + ·) acc (unscaled ⟨j, hlt⟩) := by
              have h_scaled :
                  tensorFoldlSpec (· + ·) (k * acc) (scaled ⟨j, hlt⟩) =
                    (k * acc) + sumSpec (scaled ⟨j, hlt⟩) := by
                simpa using
                  (tensor_foldl_spec_add_init (s := s) (acc := k * acc) (t := scaled ⟨j, hlt⟩))
              have h_unscaled :
                  tensorFoldlSpec (· + ·) acc (unscaled ⟨j, hlt⟩) =
                    acc + sumSpec (unscaled ⟨j, hlt⟩) := by
                simpa using
                  (tensor_foldl_spec_add_init (s := s) (acc := acc) (t := unscaled ⟨j, hlt⟩))
              have h_comp : sumSpec (scaled ⟨j, hlt⟩) = k * sumSpec (unscaled ⟨j, hlt⟩) :=
                component ⟨j, hlt⟩
              calc
                tensorFoldlSpec (· + ·) (k * acc) (scaled ⟨j, hlt⟩)
                    = (k * acc) + sumSpec (scaled ⟨j, hlt⟩) := h_scaled
                _ = (k * acc) + (k * sumSpec (unscaled ⟨j, hlt⟩)) := by
                      simp [h_comp]
                _ = k * (acc + sumSpec (unscaled ⟨j, hlt⟩)) := by
                      ring
                _ = k * tensorFoldlSpec (· + ·) acc (unscaled ⟨j, hlt⟩) := by
                      simp [h_unscaled]
            -- Apply IH to the tail with the correctly-scaled accumulator.
            simpa [step] using
              (ihj (j := j + 1) (acc := tensorFoldlSpec (· + ·) acc (unscaled ⟨j, hlt⟩)) h_next)

        -- Finish by unfolding `dot`/`sum_spec` and reducing to `go_scale` at `j=0, acc=0`.
        simpa [dot, sumSpec, tensorFoldlSpec, mulSpec, scaleSpec, mapSpec, map2Spec, scaled,
          unscaled] using
          (go_scale (j := 0) (acc := 0))

/-!
## Flatten / unflatten round-trips

The round-trip lemmas for the spec definitions live with the definitions themselves:
`NN/Spec/Core/TensorReductionShape.lean` provides

- `Spec.Tensor.flatten_unflatten_inverse`
- `Spec.Tensor.unflatten_flatten_inverse`

so downstream proof files do not need to re-prove the index arithmetic here.
-/

/--
Size preservation under operations.
Essential for proving tensor operations maintain expected dimensions.
-/
theorem shape_size_add {s : Shape} (a b : Tensor ℝ s) :
  Spec.Shape.size (shapeOf (addSpec a b)) = Spec.Shape.size s := by
  rw [shapeOf_eq_shape]

/-- Size preservation for `mul_spec`: elementwise multiplication does not change shape size. -/
theorem shape_size_mul {s : Shape} (a b : Tensor ℝ s) :
  Spec.Shape.size (shapeOf (mulSpec a b)) = Spec.Shape.size s := by
  rw [shapeOf_eq_shape]

-- Error and approximation theorems
/-
Numerical stability: helper bounds for `safediv_spec`.

The actual user-facing statement in this file is `safediv_bound`; the helpers below exist only to
make that proof easy to maintain.
-/
-- Helper: a uniform bound for all entries of a tensor (used in `safediv_bound`).
/-- Predicate `tensorAllLE b t`: all entries of `t` are `≤ b`. -/
private def tensorAllLE (bound : ℝ) : ∀ {s : Shape}, Tensor ℝ s → Prop
  | .scalar, .scalar x => x ≤ bound
  | .dim _ _, .dim values => ∀ j, tensorAllLE bound (values j)

-- Maximum entry value over a tensor (using `0` as a safe initial accumulator).
/-- A maximum-like bound for tensor entries, computed structurally (used only for bounding lemmas).
  -/
private def tensorMax : ∀ {s : Shape}, Tensor ℝ s → ℝ
  | .scalar, .scalar x => x
  | .dim n _, .dim values =>
      ((List.finRange n).map (fun j => tensorMax (values j))).foldl max 0

/-- Each slice maximum is bounded by the tensor maximum of a `.dim` tensor. -/
private lemma tensorMax_le_dim {n : Nat} {inner : Shape} (values : Fin n → Tensor ℝ inner) (j : Fin
  n) :
    tensorMax (values j) ≤ tensorMax (Tensor.dim values) := by
  -- `tensorMax (Tensor.dim values)` is a `foldl max` over the list of slice maxima.
  have hj : j ∈ List.finRange n := by simp
  have hmem :
      tensorMax (values j) ∈ (List.finRange n).map (fun k => tensorMax (values k)) :=
    List.mem_map_of_mem (f := fun k => tensorMax (values k)) hj
  -- Use the generic list bound lemma.
  simpa [tensorMax] using
    (List.le_foldl_max_of_mem
      (l := (List.finRange n).map (fun k => tensorMax (values k))) (f := id)
      (acc := 0) (i := tensorMax (values j)) hmem)

/-- Monotonicity: if `tensorAllLE b₁ t` and `b₁ ≤ b₂`, then `tensorAllLE b₂ t`. -/
private lemma tensorAllLE_mono {bnd₁ bnd₂ : ℝ} :
    ∀ {s : Shape} (t : Tensor ℝ s), tensorAllLE bnd₁ t → bnd₁ ≤ bnd₂ → tensorAllLE bnd₂ t := by
  intro s t ht hle
  induction s with
  | scalar =>
    cases t with
    | scalar x =>
      simpa [tensorAllLE] using le_trans ht hle
  | dim n inner ih =>
    cases t with
    | dim values =>
      intro j
      exact ih (t := values j) (ht j)

/-- Every tensor is bounded by its computed maximum: `tensorAllLE (tensorMax t) t`. -/
private lemma tensorAllLE_tensorMax : ∀ {s : Shape} (t : Tensor ℝ s), tensorAllLE (tensorMax t) t :=
  by
  intro s t
  induction s with
  | scalar =>
    cases t with
    | scalar x =>
      simp [tensorAllLE, tensorMax]
  | dim n inner ih =>
    cases t with
    | dim values =>
      intro j
      have h_sub : tensorAllLE (tensorMax (values j)) (values j) := ih (t := values j)
      have h_le : tensorMax (values j) ≤ tensorMax (Tensor.dim values) :=
        tensorMax_le_dim values j
      exact tensorAllLE_mono (t := values j) h_sub h_le

/-- If all entries are `≤ bound`, then clamping by `min · bound` is the identity. -/
private lemma map_min_eq_self_of_tensorAllLE {bound : ℝ} :
    ∀ {s : Shape} (t : Tensor ℝ s), tensorAllLE bound t →
      mapSpec (fun x => min x bound) t = t := by
  intro s t ht
  induction s with
  | scalar =>
    cases t with
    | scalar x =>
      have hx : x ≤ bound := by simpa [tensorAllLE] using ht
      simp [mapSpec, hx]
  | dim n inner ih =>
    cases t with
    | dim values =>
      apply congrArg Tensor.dim
      funext j
      exact ih (t := values j) (ht j)

/-- Existence of a uniform bound for `safediv_spec`, expressed via an idempotent `min`-clamp. -/
theorem safediv_bound {s : Shape} (a b : Tensor ℝ s) :
  ∀ _i : Fin s.size, (Numbers.epsilon : ℝ) > 0 →
  ∃ bound, absSpec (safedivSpec a b) = mapSpec (fun x => min x bound) (absSpec (safedivSpec a
    b)) := by
  -- The `i` and positivity hypothesis are irrelevant: the tensor has finitely many entries, so we
  -- can take `bound` to be a maximum over all entries (then `min x bound = x` everywhere).
  intro _ _
  let t := absSpec (safedivSpec a b)
  refine ⟨tensorMax t, ?_⟩
  simpa [t] using
    (Eq.symm (map_min_eq_self_of_tensorAllLE (t := t) (tensorAllLE_tensorMax (t := t))))

/--
Tensor norm properties.
Essential for regularization and optimization proofs.
-/
noncomputable def tensorNormSquared {s : Shape} (t : Tensor ℝ s) : ℝ :=
  dot t t

/-- `tensor_norm_squared t` is nonnegative, since it is a sum of squares. -/
theorem tensor_norm_squared_nonneg {s : Shape} (t : Tensor ℝ s) :
  0 ≤ tensorNormSquared t := by
  -- `tensor_norm_squared t = dot t t = sum_spec (mul_spec t t)` is a sum of squares.
  -- We prove non-negativity by structural induction on the shape.
  -- (The recursion follows the definition of `tensor_foldl_spec` used by `sum_spec`.)
  induction s with
  | scalar =>
    cases t with
    | scalar x =>
      -- dot (scalar x) (scalar x) = x * x
      simp [tensorNormSquared, dot, sumSpec, mulSpec, map2Spec, tensorFoldlSpec,
        mul_self_nonneg]
  | dim n s ih =>
    cases t with
    | dim values =>
      -- Square each sub-tensor and sum all entries via `tensor_foldl_spec.go`.
      let valuesSq : Fin n → Tensor ℝ s := fun i => mulSpec (values i) (values i)
      have term_nonneg : ∀ i : Fin n, 0 ≤ sumSpec (valuesSq i) := by
        intro i
        -- `sum_spec (valuesSq i) = tensor_norm_squared (values i)` by definition.
        simpa [valuesSq, tensorNormSquared, dot] using (ih (t := values i))

      -- Show the fold accumulator stays nonnegative.
      have go_nonneg :
          ∀ k acc, k ≤ n → 0 ≤ acc →
            0 ≤ tensorFoldlSpec.go (· + ·) n s valuesSq k acc := by
        intro k acc hk hacc
        -- Induct on the remaining length `n - k`.
        induction hn : n - k generalizing k acc with
        | zero =>
          have hk' : k = n := by
            have : n ≤ k := Nat.sub_eq_zero_iff_le.mp hn
            exact Nat.le_antisymm hk this
          subst k
          -- `k = n` so the loop stops immediately.
          have hgo :
              tensorFoldlSpec.go (· + ·) n s valuesSq n acc = acc := by
            simpa using
              (tensor_foldl_spec_go_of_not_lt (f := (· + ·)) (values := valuesSq) (k := n) (acc := acc)
                (by simp))
          simp [hgo, hacc]
        | succ m ih_go =>
          have hlt : k < n := by
            have : 0 < n - k := by simp [hn]
            exact Nat.sub_pos_iff_lt.mp this
          have hk1 : k + 1 ≤ n := Nat.succ_le_of_lt hlt
          -- Unfold one loop step at index `k`.
          rw [tensor_foldl_spec_go_of_lt (f := (· + ·)) (values := valuesSq) (k := k) (acc := acc) hlt]
          -- The next accumulator is `acc + sum_spec (valuesSq k)`.
          have hstep :
              tensorFoldlSpec (· + ·) acc (valuesSq ⟨k, hlt⟩) =
                acc + sumSpec (valuesSq ⟨k, hlt⟩) := by
            simpa using
              (tensor_foldl_spec_add_init (s := s) (acc := acc) (t := valuesSq ⟨k, hlt⟩))
          have hacc' : 0 ≤ tensorFoldlSpec (· + ·) acc (valuesSq ⟨k, hlt⟩) := by
            have hterm : 0 ≤ sumSpec (valuesSq ⟨k, hlt⟩) := term_nonneg ⟨k, hlt⟩
            simpa [hstep] using add_nonneg hacc hterm
          -- Reduce `n - (k+1)` to `m` and apply IH.
          have h_next : n - (k + 1) = m := by
            rw [Nat.sub_succ, hn]
            rfl
          have := ih_go (k := k + 1)
            (acc := tensorFoldlSpec (· + ·) acc (valuesSq ⟨k, hlt⟩)) hk1 hacc'
          simpa [h_next] using this

      -- Finish by unfolding `tensor_norm_squared` and applying `go_nonneg` at `k=0, acc=0`.
      simpa [tensorNormSquared, dot, sumSpec, mulSpec, map2Spec, tensorFoldlSpec, valuesSq]
        using
        (go_nonneg (k := 0) (acc := (0 : ℝ)) (by exact Nat.zero_le n) (by simp))

/-- `tensor_norm_squared t = 0` iff `t` is the all-zero tensor. -/
theorem tensor_norm_squared_zero_iff {s : Shape} (t : Tensor ℝ s) :
  tensorNormSquared t = 0 ↔ t = fill (0 : ℝ) s := by
  -- Both directions by induction on the shape.
  induction s with
  | scalar =>
    cases t with
    | scalar x =>
      -- `tensor_norm_squared (scalar x) = x*x`.
      simp [tensorNormSquared, dot, sumSpec, mulSpec, map2Spec, tensorFoldlSpec, fill]
  | dim n s ih =>
    cases t with
    | dim values =>
      let valuesSq : Fin n → Tensor ℝ s := fun i => mulSpec (values i) (values i)
      have term_nonneg : ∀ i : Fin n, 0 ≤ sumSpec (valuesSq i) := by
        intro i
        simpa [ge_iff_le, valuesSq, tensorNormSquared, dot] using
          tensor_norm_squared_nonneg (t := values i)

      -- Monotonicity of the `go` loop: accumulator is always ≤ final result.
      have go_ge :
          ∀ k acc, k ≤ n →
            acc ≤ tensorFoldlSpec.go (· + ·) n s valuesSq k acc := by
        intro k acc hk
        induction hn : n - k generalizing k acc with
        | zero =>
          have hk' : k = n := by
            have : n ≤ k := Nat.sub_eq_zero_iff_le.mp hn
            exact Nat.le_antisymm hk this
          subst k
          have hgo :
              tensorFoldlSpec.go (· + ·) n s valuesSq n acc = acc := by
            simpa using
              (tensor_foldl_spec_go_of_not_lt (f := (· + ·)) (values := valuesSq) (k := n) (acc := acc)
                (by simp))
          simp [hgo]
        | succ m ih_go =>
          have hlt : k < n := by
            have : 0 < n - k := by simp [hn]
            exact Nat.sub_pos_iff_lt.mp this
          have hk1 : k + 1 ≤ n := Nat.succ_le_of_lt hlt
          rw [tensor_foldl_spec_go_of_lt (f := (· + ·)) (values := valuesSq) (k := k) (acc := acc) hlt]
          -- `acc ≤ acc + term` and then apply IH to the remainder.
          have hstep :
              tensorFoldlSpec (· + ·) acc (valuesSq ⟨k, hlt⟩) =
                acc + sumSpec (valuesSq ⟨k, hlt⟩) := by
            simpa using
              (tensor_foldl_spec_add_init (s := s) (acc := acc) (t := valuesSq ⟨k, hlt⟩))
          have hacc_le :
              acc ≤ tensorFoldlSpec (· + ·) acc (valuesSq ⟨k, hlt⟩) := by
            have hterm : 0 ≤ sumSpec (valuesSq ⟨k, hlt⟩) := term_nonneg ⟨k, hlt⟩
            -- `acc ≤ acc + term`
            have : acc ≤ acc + sumSpec (valuesSq ⟨k, hlt⟩) :=
              le_add_of_nonneg_right (a := acc) hterm
            simpa [hstep] using this
          have h_next : n - (k + 1) = m := by
            rw [Nat.sub_succ, hn]
            rfl
          have hrest :=
            ih_go (k := k + 1)
              (acc := tensorFoldlSpec (· + ·) acc (valuesSq ⟨k, hlt⟩)) hk1
          -- `acc ≤ newAcc ≤ go (k+1) newAcc`.
          exact le_trans hacc_le (by simpa [h_next] using hrest)

      -- Main equivalence.
      constructor
      · -- → direction: if the sum of squares is 0, all components are 0.
        intro h0
        -- Unfold `tensor_norm_squared` to get a statement about `go`.
        have hgo0 :
            tensorFoldlSpec.go (· + ·) n s valuesSq 0 0 = 0 := by
          simpa [tensorNormSquared, dot, sumSpec, mulSpec, map2Spec, tensorFoldlSpec,
            valuesSq] using h0

        -- Prove each `values i` is zero by iterating the `go` loop.
        have go_all_zero :
            ∀ k acc, k ≤ n → 0 ≤ acc →
              tensorFoldlSpec.go (· + ·) n s valuesSq k acc = 0 →
                acc = 0 ∧ ∀ i : Fin n, i.val ≥ k → values i = fill (0 : ℝ) s := by
          intro k acc hk hacc hgo
          induction hn : n - k generalizing k acc with
          | zero =>
            have hk' : k = n := by
              have : n ≤ k := Nat.sub_eq_zero_iff_le.mp hn
              exact Nat.le_antisymm hk this
            subst k
            -- loop stops: go n acc = acc
            have hgo_stop : acc = 0 := by
              simpa
                [tensor_foldl_spec_go_of_not_lt (f := (· + ·)) (values := valuesSq) (k := n) (acc := acc)
                  (by simp)]
                using hgo
            exact ⟨hgo_stop, by intro i hi; exfalso; exact Nat.not_lt_of_ge hi i.isLt⟩
          | succ m ih_go =>
            have hlt : k < n := by
              have : 0 < n - k := by simp [hn]
              exact Nat.sub_pos_iff_lt.mp this
            have hk1 : k + 1 ≤ n := Nat.succ_le_of_lt hlt
            -- Peel one loop step from the hypothesis `hgo`.
            have hgo_step :
                tensorFoldlSpec.go (· + ·) n s valuesSq (k + 1)
                    (tensorFoldlSpec (· + ·) acc (valuesSq ⟨k, hlt⟩))
                  = 0 := by
              simpa
                [tensor_foldl_spec_go_of_lt (f := (· + ·)) (values := valuesSq) (k := k) (acc := acc) hlt]
                using hgo
            -- New accumulator after processing index `k`.
            let nextAcc := tensorFoldlSpec (· + ·) acc (valuesSq ⟨k, hlt⟩)
            have hstep :
                nextAcc = acc + sumSpec (valuesSq ⟨k, hlt⟩) := by
              -- expand `nextAcc` and use `tensor_foldl_spec_add_init`
              simp [nextAcc, tensor_foldl_spec_add_init (s := s) (acc := acc) (t := valuesSq ⟨k,
                hlt⟩)]
            -- `go (k+1) nextAcc = 0`
            have hgo' : tensorFoldlSpec.go (· + ·) n s valuesSq (k + 1) nextAcc = 0 := by
              simpa [nextAcc] using hgo_step
            -- `nextAcc ≤ go (k+1) nextAcc = 0`
            have hnext_le0 : nextAcc ≤ 0 := by
              have hge := go_ge (k := k + 1) (acc := nextAcc) hk1
              exact le_trans hge (le_of_eq hgo')
            -- `0 ≤ nextAcc`
            have hterm : 0 ≤ sumSpec (valuesSq ⟨k, hlt⟩) := term_nonneg ⟨k, hlt⟩
            have hnext_ge0 : 0 ≤ nextAcc := by
              -- nextAcc = acc + term
              have : 0 ≤ acc + sumSpec (valuesSq ⟨k, hlt⟩) := add_nonneg hacc hterm
              simpa [hstep] using this
            -- hence `nextAcc = 0`
            have hnext0 : nextAcc = 0 := le_antisymm hnext_le0 hnext_ge0
            -- From `nextAcc = acc + term`, deduce `acc = 0` and `term = 0`.
            have hacc0 : acc = 0 := by
              have : acc ≤ nextAcc := by
                -- acc ≤ acc + term
                have : acc ≤ acc + sumSpec (valuesSq ⟨k, hlt⟩) := le_add_of_nonneg_right hterm
                simpa [hstep] using this
              exact le_antisymm (le_trans this (le_of_eq hnext0)) hacc
            have hterm0 : sumSpec (valuesSq ⟨k, hlt⟩) = 0 := by
              -- rewrite `nextAcc = acc + term` and use `acc = 0`, `nextAcc = 0`
              have : acc + sumSpec (valuesSq ⟨k, hlt⟩) = 0 := by simpa [hstep] using hnext0
              simpa [hacc0] using this
            -- Now apply IH on the remainder with `acc = 0` and `nextAcc = 0`.
            have h_next : n - (k + 1) = m := by
              rw [Nat.sub_succ, hn]
              rfl
            have ih_res :=
              (ih_go (k := k + 1) (acc := nextAcc) hk1 (by simp [hnext0]) hgo') h_next
            rcases ih_res with ⟨_acc0, htail⟩
            -- Deduce `values ⟨k, hlt⟩ = fill 0` from `term = 0` and IH on the inner shape.
            have hhead : values ⟨k, hlt⟩ = fill (0 : ℝ) s := by
              -- `term = tensor_norm_squared (values k)`:
              have : tensorNormSquared (values ⟨k, hlt⟩) = 0 := by
                simpa [valuesSq, tensorNormSquared, dot] using hterm0
              exact (ih (t := values ⟨k, hlt⟩)).1 this
            refine ⟨hacc0, ?_⟩
            intro i hi
            -- Split on whether `i.val = k` or `i.val ≥ k+1`.
            have hcase : i.val = k ∨ i.val ≥ k + 1 := by
              have hk' : k = i.val ∨ k < i.val := Nat.eq_or_lt_of_le hi
              cases hk' with
              | inl hk_eq => exact Or.inl hk_eq.symm
              | inr hk_lt => exact Or.inr (Nat.succ_le_of_lt hk_lt)
            cases hcase with
            | inl hk' =>
              -- `i = ⟨k, _⟩`
              have : i = ⟨k, hlt⟩ := by
                apply Fin.ext
                exact hk'
              subst this
              exact hhead
            | inr hge =>
              exact htail i hge

        have hall := go_all_zero (k := 0) (acc := (0 : ℝ)) (by exact Nat.zero_le n) (by simp) hgo0
        rcases hall with ⟨_acc0, hvals⟩
        -- Rebuild the tensor from pointwise equalities.
        apply congrArg Tensor.dim
        funext i
        exact hvals i (by exact Nat.zero_le _)
      · -- ← direction: the zero tensor has zero norm-squared.
        intro ht0
        rw [ht0]
        -- `fill 0` has `tensor_norm_squared = 0` because it contains only zeros.
        let innerZero : Tensor ℝ s := fill (0 : ℝ) s

        -- Inner zero tensor has zero norm-squared by IH.
        have hnorm_inner : tensorNormSquared innerZero = 0 :=
          (ih (t := innerZero)).2 rfl

        -- And elementwise multiplication preserves `fill 0`.
        have hm_inner : mulSpec innerZero innerZero = innerZero := by
          simpa [innerZero] using (mul_spec_fill_zero (s := s))

        -- Hence `sum_spec innerZero = 0`.
        have hsum_inner : sumSpec innerZero = 0 := by
          simpa [tensorNormSquared, dot, hm_inner] using hnorm_inner

        -- Outer elementwise product is also `fill 0`.
        have hm_outer :
            mulSpec (fill (0 : ℝ) (Shape.dim n s)) (fill (0 : ℝ) (Shape.dim n s)) =
              fill (0 : ℝ) (Shape.dim n s) := by
          simpa using (mul_spec_fill_zero (s := Shape.dim n s))

        -- Show `sum_spec (fill 0 (dim n s)) = 0` by iterating the `go` loop.
        have go_zero :
            ∀ k, k ≤ n →
              tensorFoldlSpec.go (· + ·) n s (fun _ : Fin n => innerZero) k 0 = 0 := by
          intro k hk
          induction hn : n - k generalizing k with
          | zero =>
            have hk' : k = n := by
              have : n ≤ k := Nat.sub_eq_zero_iff_le.mp hn
              exact Nat.le_antisymm hk this
            subst k
            simpa using
              (tensor_foldl_spec_go_of_not_lt (f := (· + ·)) (values := fun _ : Fin n => innerZero)
                (k := n) (acc := (0 : ℝ)) (by simp))
          | succ m ih_go =>
            have hlt : k < n := by
              have : 0 < n - k := by simp [hn]
              exact Nat.sub_pos_iff_lt.mp this
            have hk1 : k + 1 ≤ n := Nat.succ_le_of_lt hlt
            rw [tensor_foldl_spec_go_of_lt (f := (· + ·)) (values := fun _ : Fin n => innerZero) (k := k)
              (acc := (0 : ℝ)) hlt]
            -- `tensor_foldl_spec (+) 0 innerZero = 0` since `sum_spec innerZero = 0`.
            have hstep0 : tensorFoldlSpec (· + ·) 0 innerZero = 0 := by
              simpa [hsum_inner] using
                (tensor_foldl_spec_add_init (s := s) (acc := (0 : ℝ)) (t := innerZero))
            -- Reduce to the tail and apply IH.
            have h_next : n - (k + 1) = m := by
              rw [Nat.sub_succ, hn]
              rfl
            have := ih_go (k := k + 1) hk1
            simpa [hstep0, h_next] using this

        have hsum_outer : sumSpec (fill (0 : ℝ) (Shape.dim n s)) = 0 := by
          -- unfold `sum_spec`/`tensor_foldl_spec` and apply `go_zero` at `k=0`.
          simpa [sumSpec, tensorFoldlSpec, fill, innerZero] using
            (go_zero (k := 0) (by exact Nat.zero_le n))

        -- Put it together: `tensor_norm_squared = dot = sum_spec (mul_spec _)`.
        simp [tensorNormSquared, dot, hm_outer, hsum_outer]

/-! ## Extensionality and structural algebra -/


end Spec
