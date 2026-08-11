/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: Nicolas Rouquette, TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Analysis.Sterbenz
public import NN.Floats.NeuralFloat.Error.Addition
public import NN.Floats.NeuralFloat.Error.Multiplication

/-!
# Sterbenz's Lemma for Gradual Underflow

This file extends the exact-subtraction result for the unbounded-exponent family `FLX` to the
gradual-underflow family `FLT`. The latter is the format used by the rounded-real binary32 model.

The proof separates two regimes. Below the normal threshold, `FLT` values lie on a fixed-exponent
grid that is closed under subtraction. Above that threshold, the existing `FLX` Sterbenz theorem
applies, and the resulting normal value can be transported back to `FLT`.

## Reference

- P. H. Sterbenz, *Floating-Point Computation*, Prentice-Hall, 1974.
-/

@[expose] public section

namespace TorchLean.Floats

variable {β : NeuralRadix}

/--
An `FLX` value in the normal range is representable in the corresponding `FLT` format.

This is the normal-range converse of `neural_generic_format_FLT_to_FLX`.
-/
theorem neural_generic_format_FLX_to_FLT_of_normal (emin prec : ℤ) (hprec : 0 < prec)
    {x : ℝ} (hxFLX : @neuralGenericFormat β (FLXExp prec) (flxValidExp prec hprec) x)
    (hnorm : neuralBpow β (emin + prec - 1) ≤ abs x) :
    @neuralGenericFormat β (FLTExp emin prec) (fltValidExp emin prec hprec) x := by
  let _ : NeuralValidExp (FLXExp prec) := flxValidExp prec hprec
  let _ : NeuralValidExp (FLTExp emin prec) := fltValidExp emin prec hprec
  obtain ⟨_, f, hxf, hmant⟩ := (generic_format_FLX_iff (β := β) prec hprec x).mp hxFLX
  refine (generic_format_FLT_iff (β := β) emin prec hprec x).mpr ⟨hprec, f, hxf, hmant, ?_⟩
  have hmabs : abs (f.mantissa : ℝ) = (f.mantissa.natAbs : ℝ) := by
    cases f.mantissa with
    | ofNat n => simp
    | negSucc n =>
        rw [Int.cast_negSucc, abs_of_neg]
        · norm_num
        · exact neg_neg_of_pos (by positivity : (0 : ℝ) < (n + 1 : ℕ))
  have hmantR : abs (f.mantissa : ℝ) < neuralBpow β prec := by
    rw [hmabs, neuralBpow_eq_natPow (β := β) prec hprec.le]
    exact_mod_cast hmant
  have habsx : abs x < neuralBpow β (f.exponent + prec) := by
    rw [hxf, neuralToReal, abs_mul, abs_of_pos (neuralBpow.pos β f.exponent)]
    calc
      abs (f.mantissa : ℝ) * neuralBpow β f.exponent <
          neuralBpow β prec * neuralBpow β f.exponent :=
        mul_lt_mul_of_pos_right hmantR (neuralBpow.pos β f.exponent)
      _ = neuralBpow β (f.exponent + prec) := by
        rw [← neuralBpow.add_exp]
        congr 1
        linarith
  have hlt : neuralBpow β (emin + prec - 1) < neuralBpow β (f.exponent + prec) :=
    lt_of_le_of_lt hnorm habsx
  have hexp : emin + prec - 1 < f.exponent + prec :=
    (neuralBpow_lt_neuralBpow_iff β _ _).mp hlt
  linarith

/--
Directed Sterbenz lemma for `FLT`: if `0 < y ≤ x ≤ 2y` and both operands are representable, then
their exact difference is representable.
-/
theorem neural_generic_format_FLT_sub_of_le_two_mul (emin prec : ℤ) (hprec : 0 < prec)
    {x y : ℝ} (hy : 0 < y) (hyx : y ≤ x) (hx2y : x ≤ 2 * y)
    (hxFmt : @neuralGenericFormat β (FLTExp emin prec) (fltValidExp emin prec hprec) x)
    (hyFmt : @neuralGenericFormat β (FLTExp emin prec) (fltValidExp emin prec hprec) y) :
    @neuralGenericFormat β (FLTExp emin prec) (fltValidExp emin prec hprec) (x - y) := by
  let _ : NeuralValidExp (FLTExp emin prec) := fltValidExp emin prec hprec
  let _ : NeuralValidExp (FLXExp prec) := flxValidExp prec hprec
  by_cases hsmall : abs (x - y) ≤ neuralBpow β (emin + prec - 1)
  · have hxFix := neural_generic_format_FLT_to_FIX (β := β) emin prec hprec hxFmt
    have hyFix := neural_generic_format_FLT_to_FIX (β := β) emin prec hprec hyFmt
    have hdFix := neural_generic_format_FIX_sub (β := β) emin hxFix hyFix
    apply neural_generic_format_FIX_to_FLT_of_abs_le (β := β) emin prec hprec hdFix
    exact hsmall.trans ((neuralBpow_le_neuralBpow_iff β _ _).2 (by linarith))
  · rw [not_le] at hsmall
    have hxFLX := neural_generic_format_FLT_to_FLX (β := β) emin prec hprec hxFmt
    have hyFLX := neural_generic_format_FLT_to_FLX (β := β) emin prec hprec hyFmt
    have hdFLX := neural_generic_format_FLX_sub_of_le_two_mul (β := β) prec hprec
      hy hyx hx2y hxFLX hyFLX
    exact neural_generic_format_FLX_to_FLT_of_normal (β := β) emin prec hprec
      hdFLX (le_of_lt hsmall)

/--
Sterbenz's lemma for `FLT`: if two positive representable values are within a factor of two, then
their exact difference is representable, including across the subnormal boundary.
-/
theorem neural_generic_format_FLT_sterbenz (emin prec : ℤ) (hprec : 0 < prec)
    {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hx2y : x ≤ 2 * y) (hy2x : y ≤ 2 * x)
    (hxFmt : @neuralGenericFormat β (FLTExp emin prec) (fltValidExp emin prec hprec) x)
    (hyFmt : @neuralGenericFormat β (FLTExp emin prec) (fltValidExp emin prec hprec) y) :
    @neuralGenericFormat β (FLTExp emin prec) (fltValidExp emin prec hprec) (x - y) := by
  let _ : NeuralValidExp (FLTExp emin prec) := fltValidExp emin prec hprec
  rcases le_total y x with hyx | hxy
  · exact neural_generic_format_FLT_sub_of_le_two_mul emin prec hprec hy hyx hx2y hxFmt hyFmt
  · have hdiff :=
      neural_generic_format_FLT_sub_of_le_two_mul emin prec hprec hx hxy hy2x hyFmt hxFmt
    have hneg := neural_generic_format_neg (β := β) (fexp := FLTExp emin prec) (y - x) hdiff
    simpa only [neg_sub] using hneg

end TorchLean.Floats
