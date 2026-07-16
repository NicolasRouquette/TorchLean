/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Analysis.Sterbenz
public import NN.Floats.NeuralFloat.Error.Addition
public import NN.Floats.NeuralFloat.Error.Multiplication

/-!
# Sterbenz's lemma for the gradual-underflow format `FLT`

`Analysis/Sterbenz.lean` proves Sterbenz's lemma for the unbounded-exponent family `FLX`.  Real
IEEE-style formats (in particular binary32, `FLTExp (-149) 24`) are `FLT`: a fixed underflow floor
`emin` with `max (e - prec) emin`.  This file lifts Sterbenz's lemma from `FLX` to `FLT`, so that the
exact-subtraction guarantee applies to the format that `FP32` actually uses.

The proof splits on the magnitude of the result relative to the underflow boundary
`β^(prec + emin)`:

- **Subnormal regime** (`|x - y| ≤ β^(prec + emin)`): every `FLT` value lies on its minimum-exponent
  `FIX` grid (`neural_generic_format_FLT_to_FIX`), that grid is closed under subtraction
  (`neural_generic_format_FIX_sub`), and a small `FIX` value is `FLT`
  (`neural_generic_format_FIX_to_FLT_of_abs_le`).  No ratio hypothesis is needed here — the fixed
  grid subtracts exactly.

- **Normal regime** (`β^(prec + emin) < |x - y|`): both operands are `FLX`
  (`neural_generic_format_FLT_to_FLX`), the `FLX` Sterbenz lemma gives an exact `FLX` difference,
  and a normal-range `FLX` value is `FLT` (`neural_generic_format_FLX_to_FLT_of_normal`, proved
  below).  This is where the ratio hypotheses `x ≤ 2y`, `y ≤ 2x` are used.
-/

@[expose] public section

namespace TorchLean.Floats

variable {β : NeuralRadix}

/--
A normal-range `FLX` value is `FLT`.

`FLX` and `FLT` agree on every value whose magnitude sits in the normal range.  Concretely: an `FLX`
value with `β^(prec + emin) ≤ |x|` has an effective exponent at least `emin`, so it also meets the
`FLT` underflow floor.  This is the converse, restricted to the normal range, of the unconditional
inclusion `neural_generic_format_FLT_to_FLX`.
-/
theorem neural_generic_format_FLX_to_FLT_of_normal (emin prec : ℤ) (hprec : 0 < prec)
    {x : ℝ} (hxFLX : @neuralGenericFormat β (FLXExp prec) (flxValidExp prec hprec) x)
    (hnorm : neuralBpow β (prec + emin) ≤ abs x) :
    @neuralGenericFormat β (FLTExp emin prec) (fltValidExp emin prec hprec) x := by
  letI : NeuralValidExp (FLXExp prec) := flxValidExp prec hprec
  letI : NeuralValidExp (FLTExp emin prec) := fltValidExp emin prec hprec
  obtain ⟨f, hxf, hmant⟩ := (generic_format_FLX_iff (β := β) prec hprec x).mp hxFLX
  -- It suffices to promote the `FLX` witness to an `FLT` witness by establishing the floor.
  refine (generic_format_FLT_iff (β := β) emin prec hprec x).mpr ⟨f, hxf, hmant, ?_⟩
  -- Turn the mantissa bound into a real bound `|f.mantissa| < β^prec`.
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
  -- Hence `|x| < β^(f.exponent + prec)`.
  have habsx : abs x < neuralBpow β (f.exponent + prec) := by
    rw [hxf, neuralToReal, abs_mul, abs_of_pos (neuralBpow.pos β f.exponent)]
    calc
      abs (f.mantissa : ℝ) * neuralBpow β f.exponent <
          neuralBpow β prec * neuralBpow β f.exponent :=
        mul_lt_mul_of_pos_right hmantR (neuralBpow.pos β f.exponent)
      _ = neuralBpow β (f.exponent + prec) := by
        rw [← neuralBpow.add_exp]; congr 1; linarith
  -- Combined with the normal-range lower bound, the exponent clears `emin`.
  have hlt : neuralBpow β (prec + emin) < neuralBpow β (f.exponent + prec) :=
    lt_of_le_of_lt hnorm habsx
  have hexp : prec + emin < f.exponent + prec := (neuralBpow_lt_neuralBpow_iff β _ _).mp hlt
  linarith

/--
Sterbenz's lemma for `FLT`, directed form: if `0 < y ≤ x ≤ 2y` and both are `FLT`-representable,
their exact difference is `FLT`-representable.
-/
theorem neural_generic_format_FLT_sub_of_le_two_mul (emin prec : ℤ) (hprec : 0 < prec)
    {x y : ℝ} (hy : 0 < y) (hyx : y ≤ x) (hx2y : x ≤ 2 * y)
    (hxFmt : @neuralGenericFormat β (FLTExp emin prec) (fltValidExp emin prec hprec) x)
    (hyFmt : @neuralGenericFormat β (FLTExp emin prec) (fltValidExp emin prec hprec) y) :
    @neuralGenericFormat β (FLTExp emin prec) (fltValidExp emin prec hprec) (x - y) := by
  letI : NeuralValidExp (FLTExp emin prec) := fltValidExp emin prec hprec
  letI : NeuralValidExp (FLXExp prec) := flxValidExp prec hprec
  by_cases hxy : x = y
  · rw [hxy, sub_self]; exact neural_generic_format_zero
  have hx : 0 < x := hy.trans_le hyx
  have hdpos : 0 < x - y := sub_pos.mpr (lt_of_le_of_ne hyx (Ne.symm hxy))
  by_cases hsmall : abs (x - y) ≤ neuralBpow β (prec + emin)
  · -- Subnormal regime: the fixed underflow grid subtracts exactly.
    have hxFix := neural_generic_format_FLT_to_FIX (β := β) emin prec hprec hxFmt
    have hyFix := neural_generic_format_FLT_to_FIX (β := β) emin prec hprec hyFmt
    have hdFix := neural_generic_format_FIX_sub (β := β) emin hxFix hyFix
    exact neural_generic_format_FIX_to_FLT_of_abs_le (β := β) emin prec hprec hdFix hsmall
  · -- Normal regime: transport through the unbounded `FLX` family.
    rw [not_le] at hsmall
    have hxFLX := neural_generic_format_FLT_to_FLX (β := β) emin prec hprec hxFmt
    have hyFLX := neural_generic_format_FLT_to_FLX (β := β) emin prec hprec hyFmt
    have hdFLX := neural_generic_format_FLX_sub_of_le_two_mul (β := β) prec hprec
      hy hyx hx2y hxFLX hyFLX
    exact neural_generic_format_FLX_to_FLT_of_normal (β := β) emin prec hprec
      hdFLX (le_of_lt hsmall)

/--
Symmetric Sterbenz lemma for `FLT`.  If two positive `FLT`-representable values are within a factor
of two, their exact difference is `FLT`-representable, whichever operand is larger.

This is the format-faithful analogue of `neural_generic_format_FLX_sterbenz`: it holds for the
gradual-underflow format `FLTExp emin prec` that real binary formats (e.g. `fexp32`) use, across the
whole exponent range including the subnormal boundary.
-/
theorem neural_generic_format_FLT_sterbenz (emin prec : ℤ) (hprec : 0 < prec)
    {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hx2y : x ≤ 2 * y) (hy2x : y ≤ 2 * x)
    (hxFmt : @neuralGenericFormat β (FLTExp emin prec) (fltValidExp emin prec hprec) x)
    (hyFmt : @neuralGenericFormat β (FLTExp emin prec) (fltValidExp emin prec hprec) y) :
    @neuralGenericFormat β (FLTExp emin prec) (fltValidExp emin prec hprec) (x - y) := by
  letI : NeuralValidExp (FLTExp emin prec) := fltValidExp emin prec hprec
  rcases le_total y x with hyx | hxy
  · exact neural_generic_format_FLT_sub_of_le_two_mul emin prec hprec hy hyx hx2y hxFmt hyFmt
  · have hdiff := neural_generic_format_FLT_sub_of_le_two_mul emin prec hprec hx hxy hy2x hyFmt hxFmt
    have hneg := neural_generic_format_neg (β := β) (fexp := FLTExp emin prec) (y - x) hdiff
    simpa only [neg_sub] using hneg

end TorchLean.Floats
