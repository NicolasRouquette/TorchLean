/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Error.Exactness
public import NN.Floats.NeuralFloat.Error.Directed

/-!
# Exactness of Multiplication Errors

In the unbounded-exponent FLX format, the residual of a rounded product of two representable
operands is itself representable.  The rounding mode may be any valid monotone integer rounding.
-/

@[expose] public section

namespace TorchLean.Floats

variable {β : NeuralRadix}

/-- Multiplication by a radix power preserves FLX representability. -/
theorem neural_generic_format_FLX_mul_bpow (prec : ℤ) (hprec : 0 < prec)
    {x : ℝ} (hx : @neuralGenericFormat β (FLXExp prec) (flxValidExp prec hprec) x)
    (e : ℤ) :
    @neuralGenericFormat β (FLXExp prec) (flxValidExp prec hprec)
      (x * neuralBpow β e) := by
  let _ : NeuralValidExp (FLXExp prec) := flxValidExp prec hprec
  by_cases hx0 : x = 0
  · subst x
    simp
  obtain ⟨f, hxf, hf⟩ := neural_canonical_exists_of_generic hx
  let shifted : NeuralFloat β := { mantissa := f.mantissa, exponent := f.exponent + e }
  have hvalue : x * neuralBpow β e = neuralToReal shifted := by
    rw [hxf]
    simp [shifted, neuralToReal, neuralBpow.add_exp]
    ring
  apply neural_generic_format_of_toReal_of_cexp_le shifted _ hvalue
  have hmag : neuralMagnitude β (x * neuralBpow β e) = neuralMagnitude β x + e := by
    rw [neuralMagnitude_mul_bpow β x e hx0]
  have hfexp : f.exponent = neuralCexp β (FLXExp prec) x := by
    rw [hxf]
    exact hf
  change neuralCexp β (FLXExp prec) (x * neuralBpow β e) ≤ f.exponent + e
  rw [hfexp]
  simp [neuralCexp, FLXExp, hmag]
  linarith

/--
A nonzero rounded-product residual has a representation at the sum of the operand canonical
exponents. This is the exponent-carrying form needed by FLT underflow proofs.
-/
theorem neural_mul_round_error_FLX_exists_repr (prec : ℤ) (hprec : 0 < prec)
    (rnd : ℝ → ℤ) [NeuralValidRnd rnd] {x y : ℝ}
    (hx : @neuralGenericFormat β (FLXExp prec) (flxValidExp prec hprec) x)
    (hy : @neuralGenericFormat β (FLXExp prec) (flxValidExp prec hprec) y)
    (herr0 : @neuralRound β (FLXExp prec) (flxValidExp prec hprec) rnd (x * y) - x * y ≠ 0) :
    ∃ f : NeuralFloat β,
      @neuralRound β (FLXExp prec) (flxValidExp prec hprec) rnd (x * y) - x * y =
        neuralToReal f ∧
      @neuralCexp β (FLXExp prec) (flxValidExp prec hprec)
          (@neuralRound β (FLXExp prec) (flxValidExp prec hprec) rnd (x * y) - x * y) ≤
        f.exponent ∧
      f.exponent = @neuralCexp β (FLXExp prec) (flxValidExp prec hprec) x +
        @neuralCexp β (FLXExp prec) (flxValidExp prec hprec) y := by
  let _ : NeuralValidExp (FLXExp prec) := flxValidExp prec hprec
  let product := x * y
  let rounded := neuralRound (β := β) (fexp := FLXExp prec) rnd product
  let err := rounded - product
  have herr0' : err ≠ 0 := herr0
  have hp0 : product ≠ 0 := by
    intro hp0
    have hr0 : rounded = 0 := by
      unfold rounded
      rw [hp0]
      exact neural_round_preserves_generic rnd 0 neural_generic_format_zero
    exact herr0' (by simp [err, hr0, hp0])
  have hx0 : x ≠ 0 := by
    intro h
    exact hp0 (by simp [product, h])
  have hy0 : y ≠ 0 := by
    intro h
    exact hp0 (by simp [product, h])
  let ex := neuralCexp β (FLXExp prec) x
  let ey := neuralCexp β (FLXExp prec) y
  obtain ⟨mx, hmx⟩ :=
    (neural_generic_format_iff_scaled_mantissa_int
      (β := β) (fexp := FLXExp prec) x).mp hx
  obtain ⟨my, hmy⟩ :=
    (neural_generic_format_iff_scaled_mantissa_int
      (β := β) (fexp := FLXExp prec) y).mp hy
  have hxrepr : x = (mx : ℝ) * neuralBpow β ex := by
    have h := neural_scaled_mantissa_mul_bpow (β := β) (fexp := FLXExp prec) x
    rw [hmx] at h
    exact h.symm
  have hyrepr : y = (my : ℝ) * neuralBpow β ey := by
    have h := neural_scaled_mantissa_mul_bpow (β := β) (fexp := FLXExp prec) y
    rw [hmy] at h
    exact h.symm
  let exactProduct : NeuralFloat β := { mantissa := mx * my, exponent := ex + ey }
  have hproduct : product = neuralToReal exactProduct := by
    unfold product exactProduct neuralToReal
    rw [hxrepr, hyrepr, neuralBpow.add_exp]
    push_cast
    ring
  obtain ⟨mr, hrounded⟩ :=
    neuralRound_toReal_exists_same_exponent
      (β := β) (fexp := FLXExp prec) rnd exactProduct
  let errorFloat : NeuralFloat β :=
    { mantissa := mr - exactProduct.mantissa, exponent := ex + ey }
  have herr : err = neuralToReal errorFloat := by
    unfold err rounded errorFloat neuralToReal
    rw [hproduct, hrounded]
    simp only [exactProduct]
    simp only [neuralToReal]
    push_cast
    ring
  have hinexact : rounded ≠ product := sub_ne_zero.mp herr0'
  have herrUlp : abs err < neuralUlp β (FLXExp prec) product := by
    exact neural_round_abs_error_lt_ulp_of_inexact rnd hinexact
  have herrBpow : abs err < neuralBpow β (neuralCexp β (FLXExp prec) product) := by
    simpa [neuralUlp, hp0] using herrUlp
  have hmagErr : neuralMagnitude β err ≤ neuralCexp β (FLXExp prec) product :=
    neuralMagnitude_le_of_abs_lt_bpow β err _ herr0' herrBpow
  have hmagProduct : neuralMagnitude β product ≤
      neuralMagnitude β x + neuralMagnitude β y := by
    simpa [product] using neuralMagnitude_mul_le_add β hx0 hy0
  have hcexp : neuralCexp β (FLXExp prec) err ≤ ex + ey := by
    change neuralMagnitude β err - prec ≤
      (neuralMagnitude β x - prec) + (neuralMagnitude β y - prec)
    simp [neuralCexp, FLXExp] at hmagErr
    linarith
  exact ⟨errorFloat, herr, hcexp, rfl⟩

/-- The residual of a valid rounded FLX product is exactly FLX-representable. -/
theorem neural_mul_round_error_FLX (prec : ℤ) (hprec : 0 < prec)
    (rnd : ℝ → ℤ) [NeuralValidRnd rnd] {x y : ℝ}
    (hx : @neuralGenericFormat β (FLXExp prec) (flxValidExp prec hprec) x)
    (hy : @neuralGenericFormat β (FLXExp prec) (flxValidExp prec hprec) y) :
    @neuralGenericFormat β (FLXExp prec) (flxValidExp prec hprec)
      (@neuralRound β (FLXExp prec) (flxValidExp prec hprec) rnd (x * y) - x * y) := by
  let _ : NeuralValidExp (FLXExp prec) := flxValidExp prec hprec
  let err := neuralRound (β := β) (fexp := FLXExp prec) rnd (x * y) - x * y
  by_cases herr0 : err = 0
  · change neuralGenericFormat β (FLXExp prec) err
    rw [herr0]
    exact neural_generic_format_zero
  obtain ⟨f, herr, hcexp, _⟩ :=
    neural_mul_round_error_FLX_exists_repr (β := β) prec hprec rnd hx hy herr0
  apply neural_generic_format_of_toReal_of_cexp_le f err herr
  exact hcexp

/-- Every FLT value is representable in the corresponding unbounded FLX format. -/
theorem neural_generic_format_FLT_to_FLX (emin prec : ℤ) (hprec : 0 < prec) {x : ℝ}
    (hx : @neuralGenericFormat β (FLTExp emin prec) (fltValidExp emin prec hprec) x) :
    @neuralGenericFormat β (FLXExp prec) (flxValidExp prec hprec) x := by
  let _ : NeuralValidExp (FLTExp emin prec) := fltValidExp emin prec hprec
  let _ : NeuralValidExp (FLXExp prec) := flxValidExp prec hprec
  apply neural_generic_inclusion (fexp₁ := FLTExp emin prec) (fexp₂ := FLXExp prec)
  · intro e
    simp [FLTExp, FLXExp]
  · exact hx

/--
The residual of an FLT rounded product is FLT-representable when the exact product is above the
underflow threshold `β^(emin + 2*prec - 1)`.
-/
theorem neural_mul_round_error_FLT (emin prec : ℤ) (hprec : 0 < prec)
    (rnd : ℝ → ℤ) [NeuralValidRnd rnd] {x y : ℝ}
    (hx : @neuralGenericFormat β (FLTExp emin prec) (fltValidExp emin prec hprec) x)
    (hy : @neuralGenericFormat β (FLTExp emin prec) (fltValidExp emin prec hprec) y)
    (hproduct : x * y ≠ 0 →
      neuralBpow β (emin + 2 * prec - 1) ≤ abs (x * y)) :
    @neuralGenericFormat β (FLTExp emin prec) (fltValidExp emin prec hprec)
      (@neuralRound β (FLTExp emin prec) (fltValidExp emin prec hprec) rnd (x * y) - x * y) := by
  let _ : NeuralValidExp (FLTExp emin prec) := fltValidExp emin prec hprec
  let _ : NeuralValidExp (FLXExp prec) := flxValidExp prec hprec
  let product := x * y
  let roundFLT := neuralRound (β := β) (fexp := FLTExp emin prec) rnd product
  let roundFLX := neuralRound (β := β) (fexp := FLXExp prec) rnd product
  let err := roundFLT - product
  by_cases hp0 : product = 0
  · have hr0 : roundFLT = 0 := by
      unfold roundFLT
      rw [hp0]
      exact neural_round_preserves_generic rnd 0 neural_generic_format_zero
    change neuralGenericFormat β (FLTExp emin prec) err
    simp [err, hr0, hp0]
  have hx0 : x ≠ 0 := by
    intro h
    exact hp0 (by simp [product, h])
  have hy0 : y ≠ 0 := by
    intro h
    exact hp0 (by simp [product, h])
  have hmagProduct : emin + 2 * prec ≤ neuralMagnitude β product := by
    have hlower := hproduct hp0
    have hupper := abs_lt_neuralBpow_magnitude β product hp0
    have hpowers : neuralBpow β (emin + 2 * prec - 1) <
        neuralBpow β (neuralMagnitude β product) := hlower.trans_lt hupper
    have hexp := (neuralBpow_lt_neuralBpow_iff β _ _).mp hpowers
    linarith
  have hcexpEq : neuralCexp β (FLTExp emin prec) product =
      neuralCexp β (FLXExp prec) product := by
    simp [neuralCexp, FLTExp, FLXExp, max_eq_left (by linarith :
      emin ≤ neuralMagnitude β product - prec)]
  have hroundEq : roundFLT = roundFLX := by
    unfold roundFLT roundFLX neuralRound neuralScaledMantissa
    rw [hcexpEq]
  have hxFLX := neural_generic_format_FLT_to_FLX (β := β) emin prec hprec hx
  have hyFLX := neural_generic_format_FLT_to_FLX (β := β) emin prec hprec hy
  by_cases herr0 : err = 0
  · change neuralGenericFormat β (FLTExp emin prec) err
    rw [herr0]
    exact neural_generic_format_zero
  have herrFLX0 : roundFLX - product ≠ 0 := by simpa [err, hroundEq] using herr0
  obtain ⟨f, herr, hcexpFLX, hfexp⟩ :=
    neural_mul_round_error_FLX_exists_repr
      (β := β) prec hprec rnd hxFLX hyFLX herrFLX0
  have hmagMul := neuralMagnitude_mul_le_add β hx0 hy0
  have hemin : emin ≤ f.exponent := by
    rw [hfexp]
    simp [neuralCexp, FLXExp]
    have : emin + 2 * prec ≤ neuralMagnitude β x + neuralMagnitude β y := by
      exact hmagProduct.trans (by simpa [product] using hmagMul)
    linarith
  have hcexpFLT : neuralCexp β (FLTExp emin prec) err ≤ f.exponent := by
    have herrEq : err = roundFLX - product := by simp [err, hroundEq]
    have hflx : neuralCexp β (FLXExp prec) err ≤ f.exponent := by
      simpa [herrEq] using hcexpFLX
    change max (neuralMagnitude β err - prec) emin ≤ f.exponent
    apply max_le
    · simpa [neuralCexp, FLXExp] using hflx
    · exact hemin
  apply neural_generic_format_of_toReal_of_cexp_le f err
  · simpa [err, hroundEq] using herr
  · exact hcexpFLT

/-- FLT multiplication by a radix power is exact when the shift stays above `emin`. -/
theorem neural_generic_format_FLT_mul_bpow (emin prec : ℤ) (hprec : 0 < prec)
    {x : ℝ} (hx : @neuralGenericFormat β (FLTExp emin prec)
      (fltValidExp emin prec hprec) x) (e : ℤ)
    (hshift : emin + prec - neuralMagnitude β x ≤ e) :
    @neuralGenericFormat β (FLTExp emin prec) (fltValidExp emin prec hprec)
      (x * neuralBpow β e) := by
  let _ : NeuralValidExp (FLTExp emin prec) := fltValidExp emin prec hprec
  by_cases hx0 : x = 0
  · subst x
    simp
  obtain ⟨f, hxf, hf⟩ := neural_canonical_exists_of_generic hx
  let shifted : NeuralFloat β := { mantissa := f.mantissa, exponent := f.exponent + e }
  have hvalue : x * neuralBpow β e = neuralToReal shifted := by
    rw [hxf]
    simp [shifted, neuralToReal, neuralBpow.add_exp]
    ring
  apply neural_generic_format_of_toReal_of_cexp_le shifted _ hvalue
  have hmag := neuralMagnitude_mul_bpow β x e hx0
  have hfexp : f.exponent = neuralCexp β (FLTExp emin prec) x := by
    rw [hxf]
    exact hf
  change max (neuralMagnitude β (x * neuralBpow β e) - prec) emin ≤ f.exponent + e
  rw [hmag, hfexp]
  simp only [neuralCexp, FLTExp]
  apply max_le
  · have hleft := add_le_add_right
      (le_max_left (neuralMagnitude β x - prec) emin) e
    linarith
  · have hbase : neuralMagnitude β x - prec ≤
        max (neuralMagnitude β x - prec) emin := le_max_left _ _
    linarith

/-- Nonnegative radix shifts preserve every FLT-representable value. -/
theorem neural_generic_format_FLT_mul_bpow_of_nonneg (emin prec : ℤ) (hprec : 0 < prec)
    {x : ℝ} (hx : @neuralGenericFormat β (FLTExp emin prec)
      (fltValidExp emin prec hprec) x) (e : ℤ) (he : 0 ≤ e) :
    @neuralGenericFormat β (FLTExp emin prec) (fltValidExp emin prec hprec)
      (x * neuralBpow β e) := by
  let _ : NeuralValidExp (FLTExp emin prec) := fltValidExp emin prec hprec
  by_cases hx0 : x = 0
  · subst x
    simp
  obtain ⟨f, hxf, hf⟩ := neural_canonical_exists_of_generic hx
  let shifted : NeuralFloat β := { mantissa := f.mantissa, exponent := f.exponent + e }
  have hvalue : x * neuralBpow β e = neuralToReal shifted := by
    rw [hxf]
    simp [shifted, neuralToReal, neuralBpow.add_exp]
    ring
  apply neural_generic_format_of_toReal_of_cexp_le shifted _ hvalue
  have hmag := neuralMagnitude_mul_bpow β x e hx0
  have hfexp : f.exponent = neuralCexp β (FLTExp emin prec) x := by
    rw [hxf]
    exact hf
  change max (neuralMagnitude β (x * neuralBpow β e) - prec) emin ≤ f.exponent + e
  rw [hmag, hfexp]
  simp only [neuralCexp, FLTExp]
  apply max_le
  · have hleft := add_le_add_right
      (le_max_left (neuralMagnitude β x - prec) emin) e
    linarith
  · exact (le_max_right _ _).trans (le_add_of_nonneg_right he)

end TorchLean.Floats
