/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.IntervalLemmas
public import NN.MLTheory.CROWN.Extras.IntervalLemmas

/-!
# Nonlinear IBP Soundness Lemmas

Monotonicity and Lipschitz facts for the nonlinear graph operations handled by the IBP certificate
soundness theorem.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN

namespace CertSoundness

noncomputable section

/-!
### Soundness of `Runtime.Ops.IBP.map_minmax` for monotone scalar functions

`Runtime.Ops.IBP.sigmoid` and `Runtime.Ops.IBP.tanh` are defined using `map_minmax`.
If the activation is monotone, then the min/max of the endpoints is a correct enclosure.
-/

theorem map_minmax_sound_real {n : Nat} (f : ℝ → ℝ) (hf : Monotone f)
    (xB : Box ℝ (.dim n .scalar)) (x : Tensor ℝ [n])
    (hx : Box.contains (α := ℝ) xB x) :
    Box.contains (α := ℝ) (NN.MLTheory.CROWN.Runtime.Ops.IBP.mapMinmax (α := ℝ) (n := n) f xB)
      (Tensor.mapSpec (α := ℝ) (s := .dim n .scalar) f x) := by
  -- This is a pointwise proof over coordinates of the vector.
  cases xB with
  | mk lo hi =>
    cases lo with
    | dim flo =>
      cases hi with
      | dim fhi =>
        cases x with
        | dim fx =>
          intro i
          have hx_i := hx i
          cases hL : flo i with
          | scalar l =>
            cases hU : fhi i with
            | scalar u =>
              cases hX : fx i with
              | scalar v =>
                have hv : l ≤ v ∧ v ≤ u := by
                  simpa [NN.MLTheory.CROWN.Box.contains, hL, hU, hX] using hx_i
                have hlu : l ≤ u := le_trans hv.1 hv.2
                have hflfu : f l ≤ f u := hf hlu
                -- `map_minmax` chooses endpoint min/max by comparing `f l` and `f u`.
                -- With monotonicity we know `f l ≤ f u`, so lower is `f l` and upper is `f u`.
                have hlo : f l ≤ f v := hf hv.1
                have hhi : f v ≤ f u := hf hv.2
                have hnot : ¬ f u < f l := not_lt_of_ge hflfu
                -- Unfold `map_minmax` and reduce to the scalar goal:
                -- `lo ≤ f v ∧ f v ≤ hi`.
                simpa [NN.MLTheory.CROWN.Runtime.Ops.IBP.mapMinmax, Tensor.mapSpec,
                  hL, hU, hX, NN.MLTheory.CROWN.Box.contains, hnot] using And.intro hlo hhi

/-!
### Soundness of the 1-Lipschitz `sin`/`cos` enclosures

`Runtime.Ops.IBP.sin` / `Runtime.Ops.IBP.cos` use a midpoint enclosure with radius `r=(u-l)/2`,
clamped to `[-1,1]`. This avoids periodic case splits while remaining sound.
-/

lemma sin_lipschitz_real (x y : ℝ) : |Real.sin x - Real.sin y| ≤ |x - y| := by
  have h := Real.sin_sub_sin x y
  calc
    |Real.sin x - Real.sin y|
        = |2 * Real.sin ((x - y) / 2) * Real.cos ((x + y) / 2)| := by
            simp [h, mul_left_comm, mul_comm]
    _ = 2 * |Real.sin ((x - y) / 2)| * |Real.cos ((x + y) / 2)| := by
          simp [abs_mul, mul_left_comm, mul_comm]
    _ ≤ 2 * |(x - y) / 2| * 1 := by
          have hsin : |Real.sin ((x - y) / 2)| ≤ |(x - y) / 2| := by
            simpa using (Real.abs_sin_le_abs (x := (x - y) / 2))
          have hcos : |Real.cos ((x + y) / 2)| ≤ 1 := by
            simpa using Real.abs_cos_le_one ((x + y) / 2)
          -- Multiply the two bounds, keeping track of nonnegativity.
          have h2 : (2 : ℝ) * |Real.sin ((x - y) / 2)| ≤ 2 * |(x - y) / 2| :=
            mul_le_mul_of_nonneg_left hsin (by norm_num)
          have hstep1 :
              (2 * |Real.sin ((x - y) / 2)|) * |Real.cos ((x + y) / 2)|
                ≤ (2 * |(x - y) / 2|) * |Real.cos ((x + y) / 2)| :=
            mul_le_mul_of_nonneg_right h2 (abs_nonneg _)
          have hstep2 :
              (2 * |(x - y) / 2|) * |Real.cos ((x + y) / 2)|
                ≤ (2 * |(x - y) / 2|) * 1 :=
            mul_le_mul_of_nonneg_left hcos (mul_nonneg (by norm_num) (abs_nonneg _))
          -- Reassociate back into `2 * |sin| * |cos|`.
          simpa [mul_assoc, mul_left_comm, mul_comm] using le_trans hstep1 hstep2
    _ = |x - y| := by
          -- `2 * |(x-y)/2| = |x-y|`.
          have htwo : (2 : ℝ) ≠ 0 := by norm_num
          calc
            2 * |(x - y) / 2| * 1 = 2 * (|x - y| / 2) := by
              simp [div_eq_mul_inv, mul_left_comm]
            _ = |x - y| := by nlinarith

lemma cos_lipschitz_real (x y : ℝ) : |Real.cos x - Real.cos y| ≤ |x - y| := by
  have h := Real.cos_sub_cos x y
  calc
    |Real.cos x - Real.cos y|
        = |(-2) * Real.sin ((x + y) / 2) * Real.sin ((x - y) / 2)| := by
            simp [h, mul_assoc]
    _ = 2 * |Real.sin ((x + y) / 2)| * |Real.sin ((x - y) / 2)| := by
          simp [abs_mul, mul_assoc]
    _ ≤ 2 * 1 * |(x - y) / 2| := by
          have hsin1 : |Real.sin ((x + y) / 2)| ≤ 1 := by
            simpa using Real.abs_sin_le_one ((x + y) / 2)
          have hsin2 : |Real.sin ((x - y) / 2)| ≤ |(x - y) / 2| := by
            simpa using (Real.abs_sin_le_abs (x := (x - y) / 2))
          have h2 : (2 : ℝ) * |Real.sin ((x + y) / 2)| ≤ 2 * 1 :=
            mul_le_mul_of_nonneg_left hsin1 (by norm_num)
          have hstep1 :
              (2 * |Real.sin ((x + y) / 2)|) * |Real.sin ((x - y) / 2)|
                ≤ (2 * 1) * |Real.sin ((x - y) / 2)| :=
            mul_le_mul_of_nonneg_right h2 (abs_nonneg _)
          have hstep2 :
              (2 * 1) * |Real.sin ((x - y) / 2)| ≤ (2 * 1) * |(x - y) / 2| :=
            mul_le_mul_of_nonneg_left hsin2 (by norm_num)
          simpa [mul_assoc, mul_left_comm, mul_comm] using le_trans hstep1 hstep2
    _ = |x - y| := by
          have htwo : (2 : ℝ) ≠ 0 := by norm_num
          calc
            2 * 1 * |(x - y) / 2| = 2 * (|x - y| / 2) := by
              simp [div_eq_mul_inv, mul_left_comm, mul_comm]
            _ = |x - y| := by nlinarith

lemma ibp_sin_sound_real {n : Nat} (xB : Box ℝ (.dim n .scalar)) (x : Tensor ℝ [n])
    (hx : Box.contains (α := ℝ) xB x) :
    Box.contains (α := ℝ) (NN.MLTheory.CROWN.Runtime.Ops.IBP.sin (α := ℝ) (n := n) xB)
      (Tensor.mapSpec (α := ℝ) (s := .dim n .scalar) Real.sin x) := by
  cases xB with
  | mk lo hi =>
    cases lo with
    | dim flo =>
      cases hi with
      | dim fhi =>
        cases x with
        | dim fx =>
          intro i
          have hx_i := hx i
          cases hL : flo i with
          | scalar l =>
            cases hU : fhi i with
            | scalar u =>
              cases hX : fx i with
              | scalar v =>
                have hv : l ≤ v ∧ v ≤ u := by
                  simpa [NN.MLTheory.CROWN.Box.contains, hL, hU, hX] using hx_i
                have hlu : l ≤ u := le_trans hv.1 hv.2
                let m : ℝ := (l + u) / 2
                let r : ℝ := (u - l) / 2
                have hxm : |v - m| ≤ r := by
                  have hlo : -r ≤ v - m := by
                    dsimp [m, r]
                    nlinarith [hv.1, hv.2]
                  have hhi : v - m ≤ r := by
                    dsimp [m, r]
                    nlinarith [hv.1, hv.2]
                  exact abs_le.2 ⟨hlo, hhi⟩
                have hLip : |Real.sin v - Real.sin m| ≤ r := by
                  exact le_trans (sin_lipschitz_real v m) hxm
                have hdiff : -r ≤ Real.sin v - Real.sin m ∧ Real.sin v - Real.sin m ≤ r :=
                  abs_le.1 hLip
                have hmidLo : Real.sin m - r ≤ Real.sin v := by linarith [hdiff.1]
                have hmidHi : Real.sin v ≤ Real.sin m + r := by linarith [hdiff.2]
                have hsinRange : (-1 : ℝ) ≤ Real.sin v ∧ Real.sin v ≤ (1 : ℝ) := by
                  have habs : |Real.sin v| ≤ (1 : ℝ) := by simpa using Real.abs_sin_le_one v
                  exact abs_le.1 habs
                have hlo : max (-1 : ℝ) (Real.sin m - r) ≤ Real.sin v :=
                  max_le_iff.2 ⟨hsinRange.1, hmidLo⟩
                have hhi : Real.sin v ≤ min (1 : ℝ) (Real.sin m + r) :=
                  le_min_iff.2 ⟨hsinRange.2, hmidHi⟩
                have hmidLo' :
                    Real.sin ((l + u) / 2) ≤ Real.sin v + (u - l) / 2 := by
                  dsimp [m, r] at hmidLo
                  linarith
                have hmidHi' :
                    Real.sin v ≤ Real.sin ((l + u) / 2) + (u - l) / 2 := by
                  dsimp [m, r] at hmidHi
                  exact hmidHi
                simp [NN.MLTheory.CROWN.Box.contains,
                  MathFunctions.sin, Numbers.negOne, Numbers.one, Numbers.two,
                  Context.negOne, Context.two, One.one,
                  hL, hU, hX]
                exact ⟨⟨hsinRange.1, hmidLo'⟩,
                  ⟨by
                    change Real.sin v ≤ (1 : ℝ)
                    exact hsinRange.2,
                   hmidHi'⟩⟩

lemma ibp_cos_sound_real {n : Nat} (xB : Box ℝ (.dim n .scalar)) (x : Tensor ℝ [n])
    (hx : Box.contains (α := ℝ) xB x) :
    Box.contains (α := ℝ) (NN.MLTheory.CROWN.Runtime.Ops.IBP.cos (α := ℝ) (n := n) xB)
      (Tensor.mapSpec (α := ℝ) (s := .dim n .scalar) Real.cos x) := by
  cases xB with
  | mk lo hi =>
    cases lo with
    | dim flo =>
      cases hi with
      | dim fhi =>
        cases x with
        | dim fx =>
          intro i
          have hx_i := hx i
          cases hL : flo i with
          | scalar l =>
            cases hU : fhi i with
            | scalar u =>
              cases hX : fx i with
              | scalar v =>
                have hv : l ≤ v ∧ v ≤ u := by
                  simpa [NN.MLTheory.CROWN.Box.contains, hL, hU, hX] using hx_i
                have hlu : l ≤ u := le_trans hv.1 hv.2
                let m : ℝ := (l + u) / 2
                let r : ℝ := (u - l) / 2
                have hxm : |v - m| ≤ r := by
                  have hlo : -r ≤ v - m := by
                    dsimp [m, r]
                    nlinarith [hv.1, hv.2]
                  have hhi : v - m ≤ r := by
                    dsimp [m, r]
                    nlinarith [hv.1, hv.2]
                  exact abs_le.2 ⟨hlo, hhi⟩
                have hLip : |Real.cos v - Real.cos m| ≤ r := by
                  exact le_trans (cos_lipschitz_real v m) hxm
                have hdiff : -r ≤ Real.cos v - Real.cos m ∧ Real.cos v - Real.cos m ≤ r :=
                  abs_le.1 hLip
                have hmidLo : Real.cos m - r ≤ Real.cos v := by linarith [hdiff.1]
                have hmidHi : Real.cos v ≤ Real.cos m + r := by linarith [hdiff.2]
                have hcosRange : (-1 : ℝ) ≤ Real.cos v ∧ Real.cos v ≤ (1 : ℝ) := by
                  have habs : |Real.cos v| ≤ (1 : ℝ) := by simpa using Real.abs_cos_le_one v
                  exact abs_le.1 habs
                have hlo : max (-1 : ℝ) (Real.cos m - r) ≤ Real.cos v :=
                  max_le_iff.2 ⟨hcosRange.1, hmidLo⟩
                have hhi : Real.cos v ≤ min (1 : ℝ) (Real.cos m + r) :=
                  le_min_iff.2 ⟨hcosRange.2, hmidHi⟩
                have hmidLo' :
                    Real.cos ((l + u) / 2) ≤ Real.cos v + (u - l) / 2 := by
                  dsimp [m, r] at hmidLo
                  linarith
                have hmidHi' :
                    Real.cos v ≤ Real.cos ((l + u) / 2) + (u - l) / 2 := by
                  dsimp [m, r] at hmidHi
                  exact hmidHi
                simp [NN.MLTheory.CROWN.Box.contains,
                  MathFunctions.cos, Numbers.negOne, Numbers.one, Numbers.two,
                  Context.negOne, Context.two, One.one,
                  hL, hU, hX]
                exact ⟨⟨hcosRange.1, hmidLo'⟩,
                  ⟨by
                    change Real.cos v ≤ (1 : ℝ)
                    exact hcosRange.2,
                   hmidHi'⟩⟩

end

end CertSoundness

end NN.MLTheory.CROWN.Graph
