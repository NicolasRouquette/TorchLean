/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Format.Theorems

/-!
# Exact Subtraction

The fixed-point family is an additive lattice, so subtraction is exact without a ratio
restriction.  This is the fixed-grid foundation for the floating-point Sterbenz theorem, where
the ratio hypotheses are needed to align the operands' effective exponents.
-/

@[expose] public section

namespace TorchLean.Floats

variable {β : NeuralRadix}

/-- Values on a fixed grid are closed under negation. -/
theorem FIXFormat_neg {emin : ℤ} {x : ℝ} (hx : FIXFormat (β := β) emin x) :
    FIXFormat (β := β) emin (-x) := by
  obtain ⟨f, hxf, hfe⟩ := hx
  refine ⟨{ mantissa := -f.mantissa, exponent := emin }, ?_, rfl⟩
  rw [hxf, neuralToReal, hfe]
  simp [neuralToReal]

/-- Values on a fixed grid are closed under addition. -/
theorem FIXFormat_add {emin : ℤ} {x y : ℝ}
    (hx : FIXFormat (β := β) emin x) (hy : FIXFormat (β := β) emin y) :
    FIXFormat (β := β) emin (x + y) := by
  obtain ⟨fx, hxf, hxe⟩ := hx
  obtain ⟨fy, hyf, hye⟩ := hy
  refine ⟨{ mantissa := fx.mantissa + fy.mantissa, exponent := emin }, ?_, rfl⟩
  rw [hxf, hyf, neuralToReal, neuralToReal, hxe, hye]
  simp only [neuralToReal]
  push_cast
  ring

/-- Values on a fixed grid are closed under subtraction. -/
theorem FIXFormat_sub {emin : ℤ} {x y : ℝ}
    (hx : FIXFormat (β := β) emin x) (hy : FIXFormat (β := β) emin y) :
    FIXFormat (β := β) emin (x - y) := by
  rw [sub_eq_add_neg]
  exact FIXFormat_add hx (FIXFormat_neg hy)

/-- Generic FIX representability is closed under exact subtraction. -/
theorem neural_generic_format_FIX_sub (emin : ℤ) {x y : ℝ}
    (hx : neuralGenericFormat β (FIXExp emin) x)
    (hy : neuralGenericFormat β (FIXExp emin) y) :
    neuralGenericFormat β (FIXExp emin) (x - y) := by
  rw [generic_format_FIX_iff] at hx hy ⊢
  exact FIXFormat_sub hx hy

/--
Sterbenz's lemma for the unbounded-exponent format: if two positive representable values differ by
at most a factor of two, then their subtraction is exactly representable at the same precision.

The proof does not appeal to rounding.  The ratio bound shows that the operands' magnitudes differ
by at most one radix bin.  Their integer scaled mantissas can therefore be aligned on the smaller
canonical exponent, and the exact integer difference is fine enough for the result's canonical
exponent.
-/
theorem neural_generic_format_FLX_sub_of_le_two_mul (prec : ℤ) (hprec : 0 < prec)
    {x y : ℝ} (hy : 0 < y) (hyx : y ≤ x) (hx2y : x ≤ 2 * y)
    (hxFmt : @neuralGenericFormat β (FLXExp prec) (flxValidExp prec hprec) x)
    (hyFmt : @neuralGenericFormat β (FLXExp prec) (flxValidExp prec hprec) y) :
    @neuralGenericFormat β (FLXExp prec) (flxValidExp prec hprec) (x - y) := by
  let _ : NeuralValidExp (FLXExp prec) := flxValidExp prec hprec
  by_cases hxy : x = y
  · rw [hxy, sub_self]
    exact neural_generic_format_zero
  have hx : 0 < x := hy.trans_le hyx
  have hdpos : 0 < x - y := sub_pos.mpr (lt_of_le_of_ne hyx (Ne.symm hxy))
  let ex := neuralMagnitude β x
  let ey := neuralMagnitude β y
  let e0 := ey - prec
  have heyex : ey ≤ ex := neuralMagnitude_mono_pos β hy hyx
  have hexb : ex ≤ ey + 1 := by
    have hxLower := neuralBpow_magnitude_sub_one_le β x hx.ne'
    have hyUpper := abs_lt_neuralBpow_magnitude β y hy.ne'
    have hbaseTwo : (2 : ℝ) ≤ β.toReal := by
      change (2 : ℝ) ≤ (β.base : ℝ)
      exact_mod_cast β.base_valid
    have hpow : neuralBpow β (ex - 1) < neuralBpow β (ey + 1) := by
      calc
        neuralBpow β (ex - 1) ≤ x := by simpa [ex, abs_of_pos hx] using hxLower
        _ ≤ 2 * y := hx2y
        _ ≤ β.toReal * y := mul_le_mul_of_nonneg_right hbaseTwo hy.le
        _ < β.toReal * neuralBpow β ey :=
          mul_lt_mul_of_pos_left (by simpa [ey, abs_of_pos hy] using hyUpper)
            (NeuralRadix.pos β)
        _ = neuralBpow β 1 * neuralBpow β ey := by simp [neuralBpow]
        _ = neuralBpow β (1 + ey) := (neuralBpow.add_exp β 1 ey).symm
        _ = neuralBpow β (ey + 1) := by congr 1; linarith
    have : ex - 1 < ey + 1 := (neuralBpow_lt_neuralBpow_iff β _ _).mp hpow
    linarith
  let shift := ex - ey
  have hshift0 : 0 ≤ shift := sub_nonneg.mpr heyex
  have hshift1 : shift ≤ 1 := by simp [shift]; linarith
  obtain ⟨scale, hscale⟩ := neuralBpow_eq_natCast_of_nonneg β shift hshift0
  obtain ⟨nx, hnx⟩ := neural_scaled_mantissa_int_of_generic
    (β := β) (fexp := FLXExp prec) x hxFmt
  obtain ⟨ny, hny⟩ := neural_scaled_mantissa_int_of_generic
    (β := β) (fexp := FLXExp prec) y hyFmt
  have hxRepr := neural_scaled_mantissa_mul_bpow
    (β := β) (fexp := FLXExp prec) x
  have hyRepr := neural_scaled_mantissa_mul_bpow
    (β := β) (fexp := FLXExp prec) y
  have hscaleReal : ((Int.ofNat scale : ℤ) : ℝ) = neuralBpow β shift := by
    simpa using hscale.symm
  have hxAligned : x = ((nx * Int.ofNat scale : ℤ) : ℝ) * neuralBpow β e0 := by
    rw [← hxRepr, hnx]
    rw [Int.cast_mul, hscaleReal]
    rw [mul_assoc, ← neuralBpow.add_exp]
    congr 2
    simp [neuralCexp, FLXExp, shift, ex, ey, e0]
  have hyAligned : y = (ny : ℝ) * neuralBpow β e0 := by
    rw [← hyRepr, hny]
    simp [neuralCexp, FLXExp, ey, e0]
  have hdUpper : x - y < neuralBpow β ey := by
    calc
      x - y ≤ y := by linarith
      _ < neuralBpow β ey := by simpa [ey, abs_of_pos hy] using
        (abs_lt_neuralBpow_magnitude β y hy.ne')
  have hmagd : neuralMagnitude β (x - y) ≤ ey :=
    neuralMagnitude_le_of_abs_lt_bpow β (x - y) ey hdpos.ne'
      (by simpa [abs_of_pos hdpos] using hdUpper)
  apply neural_generic_format_of_toReal_of_cexp_le
    ({ mantissa := nx * Int.ofNat scale - ny, exponent := e0 } : NeuralFloat β)
  · rw [hxAligned, hyAligned, neuralToReal]
    push_cast
    ring
  · simp [neuralCexp, FLXExp, e0]
    linarith

/--
Symmetric Sterbenz lemma for FLX.  If two positive representable values are within a factor of two,
their exact difference is representable, whichever operand is larger.
-/
theorem neural_generic_format_FLX_sterbenz (prec : ℤ) (hprec : 0 < prec)
    {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hx2y : x ≤ 2 * y) (hy2x : y ≤ 2 * x)
    (hxFmt : @neuralGenericFormat β (FLXExp prec) (flxValidExp prec hprec) x)
    (hyFmt : @neuralGenericFormat β (FLXExp prec) (flxValidExp prec hprec) y) :
    @neuralGenericFormat β (FLXExp prec) (flxValidExp prec hprec) (x - y) := by
  let _ : NeuralValidExp (FLXExp prec) := flxValidExp prec hprec
  rcases le_total y x with hyx | hxy
  · exact neural_generic_format_FLX_sub_of_le_two_mul
      prec hprec hy hyx hx2y hxFmt hyFmt
  · have hdiff := neural_generic_format_FLX_sub_of_le_two_mul
      prec hprec hx hxy hy2x hyFmt hxFmt
    have hneg := neural_generic_format_neg (β := β) (fexp := FLXExp prec) (y - x) hdiff
    simpa only [neg_sub] using hneg

end TorchLean.Floats
