/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.CertificateStep

/-!
# Interval Soundness Lemmas

Scalar interval arithmetic, box-cast lemmas, and point-box facts used by the graph IBP soundness
induction.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN

namespace CertSoundness

noncomputable section

/-!
## Op-level soundness lemmas (enclosure for each supported step)

These lemmas are the building blocks for the final “certificate ⇒ semantics enclosure” theorem.

This proof reuses the following existing components:

* Linear IBP soundness over `ℝ` is already proved in `NN.MLTheory.CROWN.mlp` as
  `NN.MLTheory.CROWN.Theorems.ibp_linear_sound_real`.
* For add/sub/relu on `FlatBox`, the graph file already contains enclosure lemmas in
  `NN.MLTheory.CROWN.Graph.Theorems.Semantics`.
-/

/-- Monotonicity of real ReLU, used by interval enclosure proofs. -/
theorem relu_mono_real : ∀ {a b : ℝ}, a ≤ b →
    Activation.Math.reluSpec (α := ℝ) a ≤ Activation.Math.reluSpec (α := ℝ) b := by
  intro a b hab
  -- `relu_spec x = max x 0`
  simpa [Activation.Math.reluSpec] using max_le_max hab (le_rfl : (0:ℝ) ≤ 0)

/-- Addition is monotone in both operands. -/
theorem add_mono_real : ∀ {a b c d : ℝ}, a ≤ b → c ≤ d → a + c ≤ b + d := by
  intro a b c d hab hcd
  exact add_le_add hab hcd

/-- Subtraction is monotone in the minuend and antitone in the subtrahend. -/
theorem sub_mono_real : ∀ {a b c d : ℝ}, a ≤ b → d ≤ c → a - c ≤ b - d := by
  intro a b c d hab hdc
  have hneg : -c ≤ -d := neg_le_neg hdc
  have : a + (-c) ≤ b + (-d) := add_le_add hab hneg
  simpa [sub_eq_add_neg] using this

lemma if_lt_eq_min (a b : ℝ) :
    (if a < b then a else b) = min a b := by
  by_cases h : a < b
  · simp [h, min_eq_left (le_of_lt h)]
  · have h' : b ≤ a := le_of_not_gt h
    simp [h, min_eq_right h']

lemma if_gt_eq_max (a b : ℝ) :
    (if a > b then a else b) = max a b := by
  by_cases h : a > b
  · simp [h, max_eq_left (le_of_lt h)]
  · have h' : a ≤ b := le_of_not_gt h
    simp [h, max_eq_right h']

lemma mul_const_bounds {a ly uy y : ℝ} (hy : ly ≤ y) (hy' : y ≤ uy) :
    min (a * ly) (a * uy) ≤ a * y ∧ a * y ≤ max (a * ly) (a * uy) := by
  by_cases ha : 0 ≤ a
  · have hlo : a * ly ≤ a * y := mul_le_mul_of_nonneg_left hy ha
    have hhi : a * y ≤ a * uy := mul_le_mul_of_nonneg_left hy' ha
    refine ⟨le_trans (min_le_left _ _) hlo, le_trans hhi (le_max_right _ _)⟩
  · have ha' : a ≤ 0 := le_of_not_ge ha
    have hlo : a * uy ≤ a * y := mul_le_mul_of_nonpos_left hy' ha'
    have hhi : a * y ≤ a * ly := mul_le_mul_of_nonpos_left hy ha'
    refine ⟨le_trans (min_le_right _ _) hlo, le_trans hhi (le_max_left _ _)⟩

lemma mul_var_bounds {lx ux x y : ℝ} (hx : lx ≤ x) (hx' : x ≤ ux) :
    min (lx * y) (ux * y) ≤ x * y ∧ x * y ≤ max (lx * y) (ux * y) := by
  by_cases hy : 0 ≤ y
  · have hlo : lx * y ≤ x * y := mul_le_mul_of_nonneg_right hx hy
    have hhi : x * y ≤ ux * y := mul_le_mul_of_nonneg_right hx' hy
    refine ⟨le_trans (min_le_left _ _) hlo, le_trans hhi (le_max_right _ _)⟩
  · have hy' : y ≤ 0 := le_of_not_ge hy
    have hlo : ux * y ≤ x * y := mul_le_mul_of_nonpos_right hx' hy'
    have hhi : x * y ≤ lx * y := mul_le_mul_of_nonpos_right hx hy'
    refine ⟨le_trans (min_le_right _ _) hlo, le_trans hhi (le_max_left _ _)⟩

lemma interval_mul_bounds
    {lx ux ly uy x y : ℝ} (hx : lx ≤ x) (hx' : x ≤ ux) (hy : ly ≤ y) (hy' : y ≤ uy) :
    min (min (lx * ly) (lx * uy)) (min (ux * ly) (ux * uy)) ≤ x * y ∧
      x * y ≤ max (max (lx * ly) (lx * uy)) (max (ux * ly) (ux * uy)) := by
  have h_lx : min (lx * ly) (lx * uy) ≤ lx * y ∧ lx * y ≤ max (lx * ly) (lx * uy) :=
    mul_const_bounds (a := lx) hy hy'
  have h_ux : min (ux * ly) (ux * uy) ≤ ux * y ∧ ux * y ≤ max (ux * ly) (ux * uy) :=
    mul_const_bounds (a := ux) hy hy'
  have h_x : min (lx * y) (ux * y) ≤ x * y ∧ x * y ≤ max (lx * y) (ux * y) :=
    mul_var_bounds (lx := lx) (ux := ux) (x := x) (y := y) hx hx'
  -- Lower bound: corners ≤ each endpoint product, hence ≤ min endpoint product, hence ≤ x*y.
  have hC_lx : min (min (lx * ly) (lx * uy)) (min (ux * ly) (ux * uy)) ≤ lx * y := by
    exact le_trans (min_le_left _ _) h_lx.1
  have hC_ux : min (min (lx * ly) (lx * uy)) (min (ux * ly) (ux * uy)) ≤ ux * y := by
    exact le_trans (min_le_right _ _) h_ux.1
  have hC_to_min : min (min (lx * ly) (lx * uy)) (min (ux * ly) (ux * uy)) ≤ min (lx * y) (ux * y)
    :=
    le_min hC_lx hC_ux
  have hlo : min (min (lx * ly) (lx * uy)) (min (ux * ly) (ux * uy)) ≤ x * y :=
    le_trans hC_to_min h_x.1
  -- Upper bound: x*y ≤ max endpoint product ≤ max corner maxes.
  let C : ℝ := max (max (lx * ly) (lx * uy)) (max (ux * ly) (ux * uy))
  have hmax_lx : lx * y ≤ C := le_trans h_lx.2 (le_max_left _ _)
  have hmax_ux : ux * y ≤ C := le_trans h_ux.2 (le_max_right _ _)
  have hmax_to_C : max (lx * y) (ux * y) ≤ C := max_le hmax_lx hmax_ux
  have hhi : x * y ≤ C := le_trans h_x.2 hmax_to_C
  simpa [C] using And.intro hlo hhi

/-! Helpers: our bound propagation uses `BoundOps.min2/max2`, which are defined via `decide (a >
  b)`.
For `ℝ` these coincide with `min/max`. -/

lemma min2_eq_min (a b : ℝ) : NN.MLTheory.CROWN.BoundOps.min2 a b = min a b := by
  by_cases h : a > b
  · have hab : b ≤ a := le_of_lt h
    simp [NN.MLTheory.CROWN.BoundOps.min2, h, min_eq_right hab]
  · have hab : a ≤ b := le_of_not_gt h
    simp [NN.MLTheory.CROWN.BoundOps.min2, h, min_eq_left hab]

lemma max2_eq_max (a b : ℝ) : NN.MLTheory.CROWN.BoundOps.max2 a b = max a b := by
  by_cases h : a > b
  · have hab : b ≤ a := le_of_lt h
    simp [NN.MLTheory.CROWN.BoundOps.max2, h, max_eq_left hab]
  · have hab : a ≤ b := le_of_not_gt h
    simp [NN.MLTheory.CROWN.BoundOps.max2, h, max_eq_right hab]

theorem box_mul_elem_sound_real (n : Nat)
    (lo1 hi1 lo2 hi2 x y : Tensor ℝ (.dim n .scalar))
    (hx : encloses { dim := n, lo := lo1, hi := hi1 } x)
    (hy : encloses { dim := n, lo := lo2, hi := hi2 } y) :
    ∀ {B : FlatBox ℝ},
      boxMulElem (α := ℝ)
          { dim := n, lo := lo1, hi := hi1 }
          { dim := n, lo := lo2, hi := hi2 } = some B →
        EnclosesBox B ⟨n, Tensor.mulSpec (α := ℝ) x y⟩ := by
  classical
  cases lo1 with
  | dim l1 =>
    cases hi1 with
    | dim u1 =>
      cases lo2 with
      | dim l2 =>
        cases hi2 with
        | dim u2 =>
          cases x with
          | dim fx =>
            cases y with
            | dim fy =>
              intro B hB
              unfold boxMulElem at hB
              simp at hB
              symm at hB
              rw [hB]
              refine ⟨rfl, ?_⟩
              dsimp [encloses, NN.MLTheory.CROWN.Graph.Theorems.Semantics.encloses, getDimScalarFn,
                castDimScalar]
              intro i
              have hx_i := hx i
              have hy_i := hy i
              cases hLx : l1 i with
              | scalar lx =>
                cases hUx : u1 i with
                | scalar ux =>
                  cases hLy : l2 i with
                  | scalar ly =>
                    cases hUy : u2 i with
                    | scalar uy =>
                      cases hX : fx i with
                      | scalar xv =>
                        cases hY : fy i with
                        | scalar yv =>
                          have hx' : lx ≤ xv ∧ xv ≤ ux := by
                            simpa [encloses, NN.MLTheory.CROWN.Graph.Theorems.Semantics.encloses,
                              getDimScalarFn,
                              hLx, hUx, hX] using hx_i
                          have hy' : ly ≤ yv ∧ yv ≤ uy := by
                            simpa [encloses, NN.MLTheory.CROWN.Graph.Theorems.Semantics.encloses,
                              getDimScalarFn,
                              hLy, hUy, hY] using hy_i
                          have hMul :=
                            interval_mul_bounds (lx := lx) (ux := ux) (ly := ly) (uy := uy)
                              (x := xv) (y := yv) (hx := hx'.1) (hx' := hx'.2) (hy := hy'.1) (hy' :=
                                hy'.2)
                          simpa [Tensor.mulSpec, Tensor.map2Spec, min2_eq_min, max2_eq_max,
                            BoundOps.mulDown, BoundOps.mulUp, hLx, hUx, hLy, hUy, hX, hY]
                            using hMul

/-!
### Casting lemmas (avoid `cases` on `B.dim = v.n`)

`FlatBox` and `FlatVec` carry their dimensions in dependent types, so it is tempting to
`cases` equalities like `h : B.dim = v.n` to “align” types. In Lean this can easily trigger
dependent elimination failures when the equality mentions fields of dependent records.

Instead, we keep such equalities as *data* and move tensors/boxes across them using
`castDimScalar` / `castBoxDim`. The following small lemmas are proved once (by `cases` on
*fresh* Nat equalities) and then used throughout the main proof without ever `cases`-ing on
`B.dim = v.n` directly.
 -/

lemma castDimScalar_trans {n n' n'' : Nat}
    (h₁ : n = n') (h₂ : n' = n'') (t : Tensor ℝ (.dim n .scalar)) :
    castDimScalar (α := ℝ) (Eq.trans h₁ h₂) t
      = castDimScalar (α := ℝ) h₂ (castDimScalar (α := ℝ) h₁ t) := by
  cases h₁
  cases h₂
  rfl

lemma castDimScalar_map_spec {n n' : Nat}
    (h : n = n') (f : ℝ → ℝ) (t : Tensor ℝ (.dim n .scalar)) :
    castDimScalar (α := ℝ) h (Tensor.mapSpec (α := ℝ) f t)
      = Tensor.mapSpec (α := ℝ) f (castDimScalar (α := ℝ) h t) := by
  cases h
  rfl

lemma castDimScalar_add_spec {n n' : Nat}
    (h : n = n') (x y : Tensor ℝ (.dim n .scalar)) :
    castDimScalar (α := ℝ) h (Tensor.addSpec (α := ℝ) x y)
      = Tensor.addSpec (α := ℝ) (castDimScalar (α := ℝ) h x) (castDimScalar (α := ℝ) h y) := by
  cases h
  rfl

lemma castDimScalar_sub_spec {n n' : Nat}
    (h : n = n') (x y : Tensor ℝ (.dim n .scalar)) :
    castDimScalar (α := ℝ) h (Tensor.subSpec (α := ℝ) x y)
      = Tensor.subSpec (α := ℝ) (castDimScalar (α := ℝ) h x) (castDimScalar (α := ℝ) h y) := by
  cases h
  rfl

lemma castDimScalar_mul_spec {n n' : Nat}
    (h : n = n') (x y : Tensor ℝ (.dim n .scalar)) :
    castDimScalar (α := ℝ) h (Tensor.mulSpec (α := ℝ) x y)
      = Tensor.mulSpec (α := ℝ) (castDimScalar (α := ℝ) h x) (castDimScalar (α := ℝ) h y) := by
  cases h
  rfl

lemma contains_castBoxDim_iff {n n' : Nat}
    (h : n = n') (B : Box ℝ (.dim n .scalar)) (x : Tensor ℝ (.dim n .scalar)) :
    Box.contains (α := ℝ) (castBoxDim (α := ℝ) h B) (castDimScalar (α := ℝ) h x)
      ↔ Box.contains (α := ℝ) B x := by
  cases h
  simp [castBoxDim, castDimScalar]

lemma encloses_castDim {B : FlatBox ℝ} {n' : Nat}
    (h : B.dim = n') (x : Tensor ℝ (.dim B.dim .scalar)) :
    encloses B x →
      encloses { dim := n'
                 lo := castDimScalar (α := ℝ) h B.lo
                 hi := castDimScalar (α := ℝ) h B.hi }
        (castDimScalar (α := ℝ) h x) := by
  intro hx
  subst n'
  convert hx using 1 <;>
    simp only [NN.MLTheory.CROWN.Graph.castDimScalar_self]

theorem encloses_of_contains {n : Nat}
    (B : Box ℝ (.dim n .scalar)) (x : Tensor ℝ (.dim n .scalar)) :
    Box.contains (α := ℝ) B x → encloses (toFlatBox (α := ℝ) n B) x := by
  intro hx
  cases B with
  | mk lo hi =>
    cases lo with
    | dim flo =>
      cases hi with
      | dim fhi =>
        cases x with
        | dim fx =>
          dsimp only [encloses, NN.MLTheory.CROWN.Graph.Theorems.Semantics.encloses]
          rw [NN.MLTheory.CROWN.Box.contains.eq_def] at hx
          intro i
          have hx_i := hx i
          cases hL : flo i with
          | scalar l =>
            cases hU : fhi i with
            | scalar u =>
              cases hX : fx i with
              | scalar v =>
                simpa [toFlatBox, getDimScalarFn, Box.contains, hL, hU, hX] using hx_i

theorem contains_of_encloses
    (B : FlatBox ℝ) (x : Tensor ℝ (.dim B.dim .scalar)) :
    encloses B x → Box.contains (α := ℝ) (ofFlatBox (α := ℝ) B) x := by
  intro hx
  cases B with
  | mk n' lo hi =>
    cases lo with
    | dim flo =>
      cases hi with
      | dim fhi =>
        cases x with
        | dim fx =>
          rw [encloses] at hx
          rw [NN.MLTheory.CROWN.Box.contains.eq_def]
          intro i
          have hx_i := hx i
          cases hL : flo i with
          | scalar l =>
            cases hU : fhi i with
            | scalar u =>
              cases hX : fx i with
              | scalar v =>
                simpa [ofFlatBox, getDimScalarFn, Box.contains, hL, hU, hX] using hx_i

/-!
### Point Boxes Always Enclose Their Point

This is used in the `.const` case, where a constant node certifies a point box
`[v,v]` and the semantics returns exactly the same `v`.
-/

theorem encloses_point_self_real {n : Nat} (x : Tensor ℝ (.dim n .scalar)) :
    NN.MLTheory.CROWN.Graph.Theorems.Semantics.encloses (α := ℝ) { dim := n, lo := x, hi := x } x :=
      by
  cases x with
  | dim fx =>
      rw [NN.MLTheory.CROWN.Graph.Theorems.Semantics.encloses]
      intro i
      cases h : fx i with
      | scalar v =>
          simp [NN.MLTheory.CROWN.Graph.getDimScalarFn, h]

end

end CertSoundness

end NN.MLTheory.CROWN.Graph
