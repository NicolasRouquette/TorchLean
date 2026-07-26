/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Core
import Mathlib.Analysis.SpecialFunctions.Log.Base

/-!
# Radix Magnitude

The magnitude of a nonzero real $x$ is the unique integer $e$ for which
$\beta^{e-1}\le|x|<\beta^e$. These bounds are the basic bridge between logarithmic magnitude,
canonical exponents, and generic-format rounding.
-/

@[expose] public section

namespace TorchLean.Floats

/-- The logarithmic definition of `neuralMagnitude` satisfies the standard Flocq magnitude bounds. -/
theorem neuralMagnitude_spec (β : NeuralRadix) (x : ℝ) (hx : x ≠ 0) :
    neuralBpow β (neuralMagnitude β x - 1) ≤ abs x ∧
      abs x < neuralBpow β (neuralMagnitude β x) := by
  let q : ℝ := Real.logb β.toReal (abs x)
  have habs : 0 < abs x := abs_pos.mpr hx
  have hb : 1 < β.toReal := NeuralRadix.gt_one β
  have hmag : neuralMagnitude β x = ⌊q⌋ + 1 := by
    simp [neuralMagnitude, hx, q, Real.logb]
  constructor
  · have hfloor : (⌊q⌋ : ℝ) ≤ Real.logb β.toReal (abs x) := by
      simpa [q] using (Int.floor_le q)
    have hpow :=
      (Real.le_logb_iff_rpow_le (b := β.toReal) (x := (⌊q⌋ : ℝ))
        (y := abs x) hb habs).mp hfloor
    simpa [hmag, neuralBpow, Real.rpow_intCast] using hpow
  · have hfloor : Real.logb β.toReal (abs x) < ((⌊q⌋ + 1 : ℤ) : ℝ) := by
      simp [q, Int.lt_floor_add_one]
    have hpow :=
      (Real.logb_lt_iff_lt_rpow (b := β.toReal) (x := abs x)
        (y := ((⌊q⌋ + 1 : ℤ) : ℝ)) hb habs).mp hfloor
    have hpow' : abs x < β.toReal ^ (⌊q⌋ + 1 : ℤ) := by
      rw [← Real.rpow_intCast]
      exact hpow
    simpa [hmag, neuralBpow] using hpow'

/-- Lower magnitude bound for a nonzero real. -/
theorem neuralBpow_magnitude_sub_one_le (β : NeuralRadix) (x : ℝ) (hx : x ≠ 0) :
    neuralBpow β (neuralMagnitude β x - 1) ≤ abs x :=
  (neuralMagnitude_spec β x hx).1

/-- Strict upper magnitude bound for a nonzero real. -/
theorem abs_lt_neuralBpow_magnitude (β : NeuralRadix) (x : ℝ) (hx : x ≠ 0) :
    abs x < neuralBpow β (neuralMagnitude β x) :=
  (neuralMagnitude_spec β x hx).2

/-- The magnitude of $\beta^e$ is $e+1$. -/
@[simp] theorem neuralMagnitude_bpow (β : NeuralRadix) (e : ℤ) :
    neuralMagnitude β (neuralBpow β e) = e + 1 := by
  have hbpos : 0 < β.toReal := NeuralRadix.pos β
  have hblog : Real.log β.toReal ≠ 0 := ne_of_gt (Real.log_pos (NeuralRadix.gt_one β))
  simp [neuralMagnitude, neuralBpow, zpow_ne_zero _ (NeuralRadix.ne_zero β),
    abs_of_pos (zpow_pos hbpos e), Real.log_zpow, hblog]

/-- Radix powers preserve and reflect exponent order. -/
@[simp] theorem neuralBpow_le_neuralBpow_iff (β : NeuralRadix) (e₁ e₂ : ℤ) :
    neuralBpow β e₁ ≤ neuralBpow β e₂ ↔ e₁ ≤ e₂ := by
  change β.toReal ^ e₁ ≤ β.toReal ^ e₂ ↔ e₁ ≤ e₂
  exact zpow_le_zpow_iff_right₀ (G₀ := ℝ) (a := β.toReal)
    (m := e₁) (n := e₂) (NeuralRadix.gt_one β)

/-- Radix powers preserve and reflect strict exponent order. -/
@[simp] theorem neuralBpow_lt_neuralBpow_iff (β : NeuralRadix) (e₁ e₂ : ℤ) :
    neuralBpow β e₁ < neuralBpow β e₂ ↔ e₁ < e₂ := by
  change β.toReal ^ e₁ < β.toReal ^ e₂ ↔ e₁ < e₂
  exact zpow_lt_zpow_iff_right₀ (G₀ := ℝ) (a := β.toReal)
    (m := e₁) (n := e₂) (NeuralRadix.gt_one β)

/-- Radix-power bounds uniquely determine magnitude. -/
theorem neuralMagnitude_eq_of_bpow_bounds (β : NeuralRadix) (x : ℝ) (e : ℤ)
    (hx : x ≠ 0) (hlower : neuralBpow β (e - 1) ≤ abs x)
    (hupper : abs x < neuralBpow β e) : neuralMagnitude β x = e := by
  have hspec := neuralMagnitude_spec β x hx
  have hleft : e - 1 < neuralMagnitude β x := by
    exact (neuralBpow_lt_neuralBpow_iff β (e - 1) (neuralMagnitude β x)).mp
      (hlower.trans_lt hspec.2)
  have hright : neuralMagnitude β x - 1 < e := by
    exact (neuralBpow_lt_neuralBpow_iff β (neuralMagnitude β x - 1) e).mp
      (hspec.1.trans_lt hupper)
  linarith

/-- Any strict radix-power upper bound is also an upper bound on magnitude. -/
theorem neuralMagnitude_le_of_abs_lt_bpow (β : NeuralRadix) (x : ℝ) (e : ℤ)
    (hx : x ≠ 0) (hupper : abs x < neuralBpow β e) : neuralMagnitude β x ≤ e := by
  have hlower := neuralBpow_magnitude_sub_one_le β x hx
  have hexp : neuralMagnitude β x - 1 < e :=
    (neuralBpow_lt_neuralBpow_iff β (neuralMagnitude β x - 1) e).mp
      (hlower.trans_lt hupper)
  linarith

/-- Magnitude is monotone on positive real inputs. -/
theorem neuralMagnitude_mono_pos (β : NeuralRadix) {x y : ℝ}
    (hx : 0 < x) (hxy : x ≤ y) : neuralMagnitude β x ≤ neuralMagnitude β y := by
  have hy : 0 < y := hx.trans_le hxy
  have hxLower := neuralBpow_magnitude_sub_one_le β x hx.ne'
  have hyUpper := abs_lt_neuralBpow_magnitude β y hy.ne'
  have hpowers : neuralBpow β (neuralMagnitude β x - 1) <
      neuralBpow β (neuralMagnitude β y) := by
    exact hxLower.trans (by simpa [abs_of_pos hx, abs_of_pos hy] using hxy) |>.trans_lt hyUpper
  have hexp : neuralMagnitude β x - 1 < neuralMagnitude β y :=
    (neuralBpow_lt_neuralBpow_iff β _ _).mp hpowers
  linarith

/-- Magnitude is monotone with respect to absolute value. -/
theorem neuralMagnitude_mono_abs (β : NeuralRadix) {x y : ℝ}
    (hx : x ≠ 0) (hxy : abs x ≤ abs y) :
    neuralMagnitude β x ≤ neuralMagnitude β y := by
  by_cases hy : y = 0
  · subst y
    have hxy0 : abs x ≤ 0 := by simpa using hxy
    have habsx : abs x = 0 := le_antisymm hxy0 (abs_nonneg x)
    exact (hx (abs_eq_zero.mp habsx)).elim
  have hxLower := neuralBpow_magnitude_sub_one_le β x hx
  have hyUpper := abs_lt_neuralBpow_magnitude β y hy
  have hpowers : neuralBpow β (neuralMagnitude β x - 1) <
      neuralBpow β (neuralMagnitude β y) :=
    lt_of_le_of_lt (hxLower.trans hxy) hyUpper
  have hexp : neuralMagnitude β x - 1 < neuralMagnitude β y :=
    (neuralBpow_lt_neuralBpow_iff β _ _).mp hpowers
  linarith

/-- A monotone exponent format preserves absolute-value order at canonical exponents. -/
theorem neuralCexp_mono_abs (β : NeuralRadix)
    {fexp : ℤ → ℤ} [NeuralValidExp fexp] [NeuralMonotoneExp fexp]
    {x y : ℝ} (hx : x ≠ 0) (hxy : abs x ≤ abs y) :
    neuralCexp β fexp x ≤ neuralCexp β fexp y := by
  apply NeuralMonotoneExp.monotone
  exact neuralMagnitude_mono_abs β hx hxy

/-- The magnitude of a nonzero product is at most the sum of operand magnitudes. -/
theorem neuralMagnitude_mul_le_add (β : NeuralRadix) {x y : ℝ}
    (hx : x ≠ 0) (hy : y ≠ 0) :
    neuralMagnitude β (x * y) ≤ neuralMagnitude β x + neuralMagnitude β y := by
  apply neuralMagnitude_le_of_abs_lt_bpow β (x * y)
  · exact mul_ne_zero hx hy
  · rw [abs_mul, neuralBpow.add_exp]
    exact mul_lt_mul
      (abs_lt_neuralBpow_magnitude β x hx)
      (abs_lt_neuralBpow_magnitude β y hy).le
      (abs_pos.mpr hy)
      (neuralBpow.nonneg β _)

/-- Multiplication by a radix power shifts magnitude by its exponent. -/
theorem neuralMagnitude_mul_bpow (β : NeuralRadix) (x : ℝ) (e : ℤ) (hx : x ≠ 0) :
    neuralMagnitude β (x * neuralBpow β e) = neuralMagnitude β x + e := by
  apply neuralMagnitude_eq_of_bpow_bounds β
  · exact mul_ne_zero hx (neuralBpow.ne_zero β e)
  · rw [abs_mul, abs_of_pos (neuralBpow.pos β e)]
    calc
      neuralBpow β (neuralMagnitude β x + e - 1) =
          neuralBpow β (neuralMagnitude β x - 1) * neuralBpow β e := by
        rw [← neuralBpow.add_exp]
        congr 1
        ring
      _ ≤ abs x * neuralBpow β e := mul_le_mul_of_nonneg_right
        (neuralBpow_magnitude_sub_one_le β x hx) (neuralBpow.nonneg β e)
  · rw [abs_mul, abs_of_pos (neuralBpow.pos β e)]
    calc
      abs x * neuralBpow β e <
          neuralBpow β (neuralMagnitude β x) * neuralBpow β e :=
        mul_lt_mul_of_pos_right
          (abs_lt_neuralBpow_magnitude β x hx) (neuralBpow.pos β e)
      _ = neuralBpow β (neuralMagnitude β x + e) :=
        (neuralBpow.add_exp β _ _).symm

/-- A radix power with nonnegative exponent is the cast of a natural number. -/
theorem neuralBpow_eq_natCast_of_nonneg (β : NeuralRadix) (e : ℤ) (he : 0 ≤ e) :
    ∃ n : ℕ, neuralBpow β e = n := by
  obtain ⟨n, rfl⟩ := Int.eq_ofNat_of_zero_le he
  exact ⟨β.base ^ n, by simp [neuralBpow, NeuralRadix.toReal]⟩

end TorchLean.Floats
