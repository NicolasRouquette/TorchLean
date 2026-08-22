/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.Calc.Bracket
public import NN.Floats.Calc.Operations
public import NN.Floats.NeuralFloat.Format.Digits
public import NN.Floats.NeuralFloat.Rounding.Nearest

/-!
# Rounding from a Certified Bracket

The effective calculation layer identifies a unit interval `[m, m + 1)` and a location inside it.
This file turns that finite location data into the integer selected by directed or nearest rounding
and proves agreement with the rounded-real definitions.
-/

@[expose] public section

namespace TorchLean.Floats

/-- Unit-interval specialization of `NeuralInbetween`. -/
abbrev NeuralInbetweenInt (m : ℤ) (x : ℝ) (location : NeuralLocation) : Prop :=
  NeuralInbetween (m : ℝ) ((m + 1 : ℤ) : ℝ) x location

/-- Canonical location obtained from the floor bracket of a real value. -/
noncomputable def neuralIntegerLocation (x : ℝ) : NeuralLocation :=
  neuralInbetweenLocation (⌊x⌋ : ℝ) ((⌊x⌋ + 1 : ℤ) : ℝ) x

/-- Every real value satisfies its canonical floor bracket. -/
theorem neuralIntegerLocation_spec (x : ℝ) :
    NeuralInbetweenInt ⌊x⌋ x (neuralIntegerLocation x) := by
  apply neuralInbetweenLocation_spec
  · norm_num
  · refine ⟨Int.floor_le x, ?_⟩
    norm_num only [Int.cast_add, Int.cast_one]
    exact Int.lt_floor_add_one x

/-- Increment an integer when the supplied decision is true. -/
def neuralConditionalIncrement (increment : Bool) (m : ℤ) : ℤ :=
  if increment then m + 1 else m

/-- A conditional increment always selects one of the two bracket endpoints. -/
theorem neuralConditionalIncrement_bounds (increment : Bool) (m : ℤ) :
    m ≤ neuralConditionalIncrement increment m ∧
      neuralConditionalIncrement increment m ≤ m + 1 := by
  cases increment <;> simp [neuralConditionalIncrement]

/-- Upward rounding increments exactly when the location is inexact. -/
def neuralRoundUpLocation : NeuralLocation → Bool
  | .exact => false
  | .inexact _ => true

/-- Nearest rounding increments above the midpoint and delegates exact ties to `chooseUp`. -/
def neuralRoundNearestLocation (chooseUp : ℤ → Bool) (m : ℤ) : NeuralLocation → Bool
  | .exact => false
  | .inexact .lt => false
  | .inexact .eq => chooseUp m
  | .inexact .gt => true

/-- Tie decision used by nearest-even rounding: increment exactly when the lower integer is odd. -/
def neuralNearestEvenChoice (m : ℤ) : Bool := decide (¬Even m)

/-- Choice-based nearest rounding with the parity decision is TorchLean's nearest-even mode. -/
theorem neuralNearestChoice_even_eq (x : ℝ) :
    neuralNearestChoice neuralNearestEvenChoice x = neuralNearestEven x := by
  simp only [neuralNearestChoice, neuralNearestEven, neuralNearestEvenChoice]
  rw [Int.fract]
  split
  · rfl
  · split
    · rfl
    · by_cases heven : Even ⌊x⌋
      · simp [heven]
      · simp [heven]

/-- A certified unit bracket determines floor exactly. -/
theorem neuralInbetweenInt_floor {m : ℤ} {x : ℝ} {location : NeuralLocation}
    (hl : NeuralInbetweenInt m x location) : neuralFloorRound x = m := by
  unfold neuralFloorRound
  cases hl with
  | exact hx => simp [hx]
  | inexact _ hx _ =>
      exact Int.floor_eq_iff.mpr ⟨by exact_mod_cast hx.1.le, by simpa using hx.2⟩

/-- A certified unit bracket determines ceiling from exactness alone. -/
theorem neuralInbetweenInt_ceil {m : ℤ} {x : ℝ} {location : NeuralLocation}
    (hl : NeuralInbetweenInt m x location) :
    neuralCeilRound x = neuralConditionalIncrement (neuralRoundUpLocation location) m := by
  unfold neuralCeilRound
  cases hl with
  | exact hx => simp [hx, neuralConditionalIncrement, neuralRoundUpLocation]
  | inexact order hx _ =>
      have hceil : ⌈x⌉ = m + 1 :=
        Int.ceil_eq_iff.mpr ⟨by simpa using hx.1, by simpa using hx.2.le⟩
      simpa [neuralConditionalIncrement, neuralRoundUpLocation] using hceil

/-- A certified unit bracket computes arbitrary-tie nearest rounding. -/
theorem neuralInbetweenInt_nearestChoice (chooseUp : ℤ → Bool)
    {m : ℤ} {x : ℝ} {location : NeuralLocation}
    (hl : NeuralInbetweenInt m x location) :
    neuralNearestChoice chooseUp x =
      neuralConditionalIncrement (neuralRoundNearestLocation chooseUp m location) m := by
  have hfloor := neuralInbetweenInt_floor hl
  cases hl with
  | exact hx =>
      subst x
      simp [neuralNearestChoice, neuralConditionalIncrement, neuralRoundNearestLocation]
  | inexact order hx hmid =>
      change ⌊x⌋ = m at hfloor
      have hfract : Int.fract x = x - m := by
        rw [Int.fract]
        rw [hfloor]
      cases order with
      | lt =>
          have hlt : Int.fract x < (2⁻¹ : ℝ) := by
            rw [cmp_eq_lt_iff] at hmid
            rw [hfract]
            norm_num only [Int.cast_add, Int.cast_one] at hmid
            norm_num at hmid ⊢
            linarith
          simp [neuralNearestChoice, hfloor, hlt, neuralConditionalIncrement,
            neuralRoundNearestLocation]
      | eq =>
          have heq : Int.fract x = (2⁻¹ : ℝ) := by
            rw [cmp_eq_eq_iff] at hmid
            rw [hfract]
            norm_num only [Int.cast_add, Int.cast_one] at hmid
            norm_num at hmid ⊢
            linarith
          simp [neuralNearestChoice, hfloor, heq, neuralConditionalIncrement,
            neuralRoundNearestLocation]
          rfl
      | gt =>
          have hgt : Int.fract x > (2⁻¹ : ℝ) := by
            rw [cmp_eq_gt_iff] at hmid
            rw [hfract]
            norm_num only [Int.cast_add, Int.cast_one] at hmid
            norm_num at hmid ⊢
            linarith
          have hnlt : ¬Int.fract x < (2⁻¹ : ℝ) := not_lt.mpr hgt.le
          simp [neuralNearestChoice, hfloor, hnlt, hgt, neuralConditionalIncrement,
            neuralRoundNearestLocation]

/-- A certified unit bracket computes nearest-even rounding. -/
theorem neuralInbetweenInt_nearestEven {m : ℤ} {x : ℝ} {location : NeuralLocation}
    (hl : NeuralInbetweenInt m x location) :
    neuralNearestEven x =
      neuralConditionalIncrement
        (neuralRoundNearestLocation neuralNearestEvenChoice m location) m := by
  rw [← neuralNearestChoice_even_eq]
  exact neuralInbetweenInt_nearestChoice neuralNearestEvenChoice hl

/-! ## Mantissa truncation -/

/-- Mantissa, exponent, and location carried by an effective rounding calculation. -/
structure NeuralTruncationState where
  /-- Lower-endpoint mantissa. -/
  mantissa : ℤ
  /-- Shared radix exponent. -/
  exponent : ℤ
  /-- Input location within the represented unit interval. -/
  location : NeuralLocation
  deriving DecidableEq, Repr

/-- Real interval and location denoted by a truncation state. -/
abbrev NeuralTruncationState.Brackets (β : NeuralRadix)
    (state : NeuralTruncationState) (x : ℝ) : Prop :=
  NeuralInbetween
    (neuralToReal (β := β) { mantissa := state.mantissa, exponent := state.exponent })
    (neuralToReal (β := β) { mantissa := state.mantissa + 1, exponent := state.exponent })
    x state.location

/-- A positive value bracketed by a truncation state forces a nonnegative lower mantissa. -/
theorem NeuralTruncationState.mantissa_nonneg_of_brackets
    (β : NeuralRadix) (state : NeuralTruncationState) {x : ℝ}
    (hx : 0 < x) (hl : state.Brackets β x) : 0 ≤ state.mantissa := by
  have hstep : 0 < neuralBpow β state.exponent := neuralBpow.pos β state.exponent
  have hordered :
      neuralToReal (β := β) { mantissa := state.mantissa, exponent := state.exponent } <
        neuralToReal (β := β) { mantissa := state.mantissa + 1, exponent := state.exponent } := by
    simp only [neuralToReal]
    exact mul_lt_mul_of_pos_right (by norm_num) hstep
  have hupper := (neuralInbetween_bounds hordered hl).2
  have hsuccPos : (0 : ℝ) < state.mantissa + 1 := by
    have : 0 < ((state.mantissa + 1 : ℤ) : ℝ) * neuralBpow β state.exponent := by
      simpa [neuralToReal] using hx.trans hupper
    norm_num only [Int.cast_add, Int.cast_one] at this
    exact pos_of_mul_pos_left this hstep.le
  have hsuccPosZ : (0 : ℤ) < state.mantissa + 1 := by exact_mod_cast hsuccPos
  linarith

/--
A positive bracket determines the canonical exponent from the lower mantissa's digit count. This is
the representation-level counterpart of Flocq's `cexp_inbetween_float`.
-/
theorem neuralCexp_eq_fexp_digits_of_brackets
    {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]
    (state : NeuralTruncationState) {x : ℝ} (hx : 0 < x) (hl : state.Brackets β x)
    (hexp : state.exponent ≤ neuralCexp β fexp x ∨
      state.exponent ≤ fexp ((neuralDigits β state.mantissa : ℕ) + state.exponent)) :
    neuralCexp β fexp x =
      fexp ((neuralDigits β state.mantissa : ℕ) + state.exponent) := by
  have hstep : 0 < neuralBpow β state.exponent := neuralBpow.pos β state.exponent
  have hordered :
      neuralToReal (β := β) { mantissa := state.mantissa, exponent := state.exponent } <
        neuralToReal (β := β) { mantissa := state.mantissa + 1, exponent := state.exponent } := by
    simp only [neuralToReal]
    exact mul_lt_mul_of_pos_right (by norm_num) hstep
  have hbounds := neuralInbetween_bounds hordered hl
  have hmNonneg := state.mantissa_nonneg_of_brackets β hx hl
  rcases hmNonneg.eq_or_lt with hmZero | hmPos
  · have hmZero' : state.mantissa = 0 := hmZero.symm
    have hmagLe : neuralMagnitude β x ≤ state.exponent := by
      apply neuralMagnitude_le_of_abs_lt_bpow β x state.exponent hx.ne'
      simpa [hmZero', neuralToReal, abs_of_pos hx] using hbounds.2
    change fexp (neuralMagnitude β x) =
      fexp ((neuralDigits β state.mantissa : ℕ) + state.exponent)
    simp only [hmZero', neuralDigits_zero, Nat.cast_zero, zero_add]
    rcases hexp with hcexp | htarget
    · have hsmall : neuralMagnitude β x ≤ fexp (neuralMagnitude β x) :=
        hmagLe.trans hcexp
      exact (((NeuralValidExp.flocq_valid (fexp := fexp) (neuralMagnitude β x)).2
        hsmall).2 state.exponent hcexp).symm
    · have htarget' : state.exponent ≤ fexp state.exponent := by
        simpa [hmZero'] using htarget
      exact ((NeuralValidExp.flocq_valid (fexp := fexp) state.exponent).2 htarget').2
        (neuralMagnitude β x) (hmagLe.trans htarget')
  · have hmR : (0 : ℝ) < state.mantissa := by exact_mod_cast hmPos
    have hlowerPos : 0 < neuralToReal (β := β)
        { mantissa := state.mantissa, exponent := state.exponent } := by
      simp only [neuralToReal]
      exact mul_pos hmR hstep
    have hmagLower :
        (neuralDigits β state.mantissa : ℤ) + state.exponent ≤ neuralMagnitude β x := by
      have hmono := neuralMagnitude_mono_pos β hlowerPos hbounds.1
      change neuralMagnitude β
        ((state.mantissa : ℝ) * neuralBpow β state.exponent) ≤ neuralMagnitude β x at hmono
      rw [neuralMagnitude_mul_bpow β (state.mantissa : ℝ) state.exponent
        (by exact_mod_cast (ne_of_gt hmPos)), neuralMagnitude_intCast_of_pos β hmPos] at hmono
      exact hmono
    have hupperPower :
        neuralToReal (β := β)
            { mantissa := state.mantissa + 1, exponent := state.exponent } ≤
          neuralBpow β ((neuralDigits β state.mantissa : ℤ) + state.exponent) := by
      simp only [neuralToReal, neuralBpow.add_exp]
      exact mul_le_mul_of_nonneg_right
        (intCast_add_one_le_neuralBpow_digits β hmPos) hstep.le
    have hmagUpper : neuralMagnitude β x ≤
        (neuralDigits β state.mantissa : ℤ) + state.exponent := by
      apply neuralMagnitude_le_of_abs_lt_bpow β x _ hx.ne'
      simpa [abs_of_pos hx] using hbounds.2.trans_le hupperPower
    have hmag : neuralMagnitude β x =
        (neuralDigits β state.mantissa : ℤ) + state.exponent :=
      le_antisymm hmagUpper hmagLower
    simp [neuralCexp, hmag]

/-- A bracket stored at the canonical exponent is a unit bracket for the scaled mantissa. -/
theorem NeuralTruncationState.scaledBrackets_of_exponent_eq
    {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]
    (state : NeuralTruncationState) {x : ℝ}
    (hl : state.Brackets β x) (hexponent : state.exponent = neuralCexp β fexp x) :
    NeuralInbetweenInt state.mantissa (neuralScaledMantissa β fexp x) state.location := by
  rcases state with ⟨mantissa, exponent, location⟩
  have hp : 0 < neuralBpow β exponent := neuralBpow.pos β exponent
  have hp0 : neuralBpow β exponent ≠ 0 := ne_of_gt hp
  rw [neuralScaledMantissa_eq_div, ← hexponent]
  cases hl with
  | exact hx =>
      apply NeuralInbetween.exact
      rw [hx]
      simp [neuralToReal, hp0]
  | inexact order hx hmid =>
      apply NeuralInbetween.inexact order
      · constructor
        · apply (lt_div_iff₀ hp).2
          simpa [neuralToReal] using hx.1
        · apply (div_lt_iff₀ hp).2
          simpa [neuralToReal] using hx.2
      · cases order with
        | lt =>
            rw [cmp_eq_lt_iff] at hmid ⊢
            change x < ((mantissa : ℝ) * neuralBpow β exponent +
              ((mantissa + 1 : ℤ) : ℝ) * neuralBpow β exponent) / 2 at hmid
            change x / neuralBpow β exponent <
              ((mantissa : ℝ) + ((mantissa + 1 : ℤ) : ℝ)) / 2
            calc
              x / neuralBpow β exponent <
                  (((mantissa : ℝ) * neuralBpow β exponent +
                    ((mantissa + 1 : ℤ) : ℝ) * neuralBpow β exponent) / 2) /
                    neuralBpow β exponent := div_lt_div_of_pos_right hmid hp
              _ = ((mantissa : ℝ) + ((mantissa + 1 : ℤ) : ℝ)) / 2 := by
                field_simp
        | eq =>
            rw [cmp_eq_eq_iff] at hmid ⊢
            change x = ((mantissa : ℝ) * neuralBpow β exponent +
              ((mantissa + 1 : ℤ) : ℝ) * neuralBpow β exponent) / 2 at hmid
            change x / neuralBpow β exponent =
              ((mantissa : ℝ) + ((mantissa + 1 : ℤ) : ℝ)) / 2
            calc
              x / neuralBpow β exponent =
                  (((mantissa : ℝ) * neuralBpow β exponent +
                    ((mantissa + 1 : ℤ) : ℝ) * neuralBpow β exponent) / 2) /
                    neuralBpow β exponent := congrArg (fun y => y / neuralBpow β exponent) hmid
              _ = ((mantissa : ℝ) + ((mantissa + 1 : ℤ) : ℝ)) / 2 := by
                field_simp
        | gt =>
            rw [cmp_eq_gt_iff] at hmid ⊢
            change ((mantissa : ℝ) * neuralBpow β exponent +
              ((mantissa + 1 : ℤ) : ℝ) * neuralBpow β exponent) / 2 < x at hmid
            change ((mantissa : ℝ) + ((mantissa + 1 : ℤ) : ℝ)) / 2 <
              x / neuralBpow β exponent
            calc
              ((mantissa : ℝ) + ((mantissa + 1 : ℤ) : ℝ)) / 2 =
                  (((mantissa : ℝ) * neuralBpow β exponent +
                    ((mantissa + 1 : ℤ) : ℝ) * neuralBpow β exponent) / 2) /
                    neuralBpow β exponent := by
                field_simp
              _ < x / neuralBpow β exponent := div_lt_div_of_pos_right hmid hp

/--
Discard `shift` low radix digits and transfer their information into the refined location.
Positive shifts are the intended use; correctness theorems state that premise explicitly.
-/
noncomputable def neuralTruncateBy (β : NeuralRadix) (state : NeuralTruncationState)
    (shift : ℤ) : NeuralTruncationState :=
  let power := neuralIntPower β shift
  let remainder := state.mantissa % power
  { mantissa := state.mantissa / power
    exponent := state.exponent + shift
    location := neuralRefineLocation power remainder state.location }

/-- A positive shift produces a radix power strictly larger than one. -/
theorem neuralIntPower_one_lt (β : NeuralRadix) {shift : ℤ} (hshift : 0 < shift) :
    1 < neuralIntPower β shift := by
  obtain ⟨n, hn⟩ := Int.eq_ofNat_of_zero_le hshift.le
  subst shift
  cases n with
  | zero => simp at hshift
  | succ n =>
      rw [neuralIntPower_of_nonneg β (Int.natCast_nonneg _)]
      simp only [Int.toNat_natCast, pow_succ]
      have hbase : 2 ≤ β.base := β.base_valid
      have hpow : 0 < β.base ^ n := pow_pos (Nat.zero_lt_of_lt hbase) n
      exact Int.ofNat_lt.mpr (show 1 < β.base ^ n * β.base by nlinarith)

/-- The truncated remainder is a valid cell index in the discarded radix block. -/
theorem neuralTruncateBy_remainder_bounds (β : NeuralRadix)
    (state : NeuralTruncationState) {shift : ℤ} (hshift : 0 < shift) :
    let power := neuralIntPower β shift
    0 ≤ state.mantissa % power ∧ state.mantissa % power < power := by
  dsimp only
  have hpower : 0 < neuralIntPower β shift := lt_trans Int.zero_lt_one
    (neuralIntPower_one_lt β hshift)
  exact ⟨Int.emod_nonneg _ hpower.ne', Int.emod_lt_of_pos _ hpower⟩

/-- Mantissa reconstruction after one truncation step. -/
theorem neuralTruncateBy_mantissa_decomposition (β : NeuralRadix)
    (state : NeuralTruncationState) {shift : ℤ} (hshift : 0 < shift) :
    state.mantissa =
      state.mantissa % neuralIntPower β shift +
        neuralIntPower β shift * (neuralTruncateBy β state shift).mantissa := by
  have hpower : neuralIntPower β shift ≠ 0 := ne_of_gt
    (lt_trans Int.zero_lt_one (neuralIntPower_one_lt β hshift))
  simpa [neuralTruncateBy, add_comm] using
    (Int.emod_add_mul_ediv state.mantissa (neuralIntPower β shift)).symm

/-- One positive truncation step preserves the represented real bracket and its refined location. -/
theorem neuralTruncateBy_brackets (β : NeuralRadix) (state : NeuralTruncationState)
    {shift : ℤ} (hshift : 0 < shift) {x : ℝ}
    (hl : state.Brackets β x) :
    (neuralTruncateBy β state shift).Brackets β x := by
  let power := neuralIntPower β shift
  let quotient := state.mantissa / power
  let remainder := state.mantissa % power
  let step := neuralBpow β state.exponent
  let start := (quotient : ℝ) * neuralBpow β (state.exponent + shift)
  have hpowerOne : 1 < power := neuralIntPower_one_lt β hshift
  have hpowerPos : 0 < power := lt_trans Int.zero_lt_one hpowerOne
  have hpowerReal : (power : ℝ) = neuralBpow β shift := by
    exact neuralIntPower_cast_eq_bpow β hshift.le
  have hrem := neuralTruncateBy_remainder_bounds β state hshift
  have hdecomp := neuralTruncateBy_mantissa_decomposition β state hshift
  have hdecomp' : state.mantissa = remainder + power * quotient := by
    simpa [power, quotient, remainder, neuralTruncateBy] using hdecomp
  have hstepPos : 0 < step := neuralBpow.pos β state.exponent
  have hlower :
      start + (remainder : ℝ) * step =
        neuralToReal (β := β) {
          mantissa := state.mantissa
          exponent := state.exponent } := by
    unfold start step neuralToReal
    rw [neuralBpow.add_exp, ← hpowerReal]
    rw [hdecomp', Int.cast_add, Int.cast_mul]
    ring
  have hupper :
      start + ((remainder + 1 : ℤ) : ℝ) * step =
        neuralToReal (β := β) {
          mantissa := state.mantissa + 1
          exponent := state.exponent } := by
    unfold start step
    change
      (quotient : ℝ) * neuralBpow β (state.exponent + shift) +
          ((remainder + 1 : ℤ) : ℝ) * neuralBpow β state.exponent =
        ((state.mantissa + 1 : ℤ) : ℝ) * neuralBpow β state.exponent
    rw [neuralBpow.add_exp, ← hpowerReal]
    rw [hdecomp']
    norm_num only [Int.cast_add, Int.cast_mul, Int.cast_one]
    ring
  have hglobalLower :
      start = neuralToReal (β := β) {
        mantissa := quotient
        exponent := state.exponent + shift } := by
    rfl
  have hglobalUpper :
      start + (power : ℝ) * step =
        neuralToReal (β := β) {
          mantissa := quotient + 1
          exponent := state.exponent + shift } := by
    unfold start step neuralToReal
    rw [neuralBpow.add_exp, ← hpowerReal]
    rw [Int.cast_add, Int.cast_one]
    ring
  have hlCell : NeuralInbetween
      (start + (remainder : ℝ) * step)
      (start + ((remainder + 1 : ℤ) : ℝ) * step)
      x state.location := by
    rw [hlower, hupper]
    exact hl
  have hrefined := neuralRefineLocation_correct
    (start := start) (step := step) (steps := power) (k := remainder)
    hstepPos hpowerOne hrem hlCell
  have hglobalUpper' :
      neuralToReal (β := β) {
          mantissa := quotient
          exponent := state.exponent + shift } +
          (power : ℝ) * step =
        neuralToReal (β := β) {
          mantissa := quotient + 1
          exponent := state.exponent + shift } := by
    rw [← hglobalLower]
    exact hglobalUpper
  rw [hglobalLower, hglobalUpper'] at hrefined
  change NeuralInbetween
    (neuralToReal (β := β) {
      mantissa := (neuralTruncateBy β state shift).mantissa
      exponent := (neuralTruncateBy β state shift).exponent })
    (neuralToReal (β := β) {
      mantissa := (neuralTruncateBy β state shift).mantissa + 1
      exponent := (neuralTruncateBy β state shift).exponent })
    x (neuralTruncateBy β state shift).location
  simpa [neuralTruncateBy, power, quotient, remainder] using hrefined

/-- Number of low radix digits discarded to reach the exponent selected by `fexp`. -/
def neuralTruncationShift (β : NeuralRadix) (fexp : ℤ → ℤ)
    (state : NeuralTruncationState) : ℤ :=
  fexp ((neuralDigits β state.mantissa : ℕ) + state.exponent) - state.exponent

/-- Truncate only when the target exponent lies strictly above the stored exponent. -/
noncomputable def neuralTruncate (β : NeuralRadix) (fexp : ℤ → ℤ)
    (state : NeuralTruncationState) : NeuralTruncationState :=
  let shift := neuralTruncationShift β fexp state
  if 0 < shift then neuralTruncateBy β state shift else state

/-- Format-driven truncation preserves the represented real bracket. -/
theorem neuralTruncate_brackets (β : NeuralRadix) (fexp : ℤ → ℤ)
    (state : NeuralTruncationState) {x : ℝ} (hl : state.Brackets β x) :
    (neuralTruncate β fexp state).Brackets β x := by
  by_cases hshift : 0 < neuralTruncationShift β fexp state
  · rw [neuralTruncate, if_pos hshift]
    exact neuralTruncateBy_brackets β state hshift hl
  · rw [neuralTruncate, if_neg hshift]
    exact hl

/--
For a positive input with sufficient initial precision, truncation both preserves the bracket and
selects the canonical format exponent.
-/
theorem neuralTruncate_brackets_and_exponent
    {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]
    (state : NeuralTruncationState) {x : ℝ} (hx : 0 < x) (hl : state.Brackets β x)
    (hexp : state.exponent ≤ neuralCexp β fexp x ∨
      state.exponent ≤ fexp ((neuralDigits β state.mantissa : ℕ) + state.exponent)) :
    (neuralTruncate β fexp state).Brackets β x ∧
      (neuralTruncate β fexp state).exponent = neuralCexp β fexp x := by
  have hcanonical := neuralCexp_eq_fexp_digits_of_brackets state hx hl hexp
  constructor
  · exact neuralTruncate_brackets β fexp state hl
  · by_cases hshift : 0 < neuralTruncationShift β fexp state
    · rw [neuralTruncate, if_pos hshift]
      simp only [neuralTruncateBy, neuralTruncationShift]
      linarith
    · rw [neuralTruncate, if_neg hshift]
      simp only [neuralTruncationShift] at hshift
      have hle : state.exponent ≤ neuralCexp β fexp x := by
        rcases hexp with h | h
        · exact h
        · rw [hcanonical]
          exact h
      rw [hcanonical] at hle ⊢
      linarith

/-- Nearest-even result selected from a format-truncated bracket. -/
noncomputable def neuralRoundTruncatedNearestEven
    (β : NeuralRadix) (fexp : ℤ → ℤ) (state : NeuralTruncationState) : NeuralFloat β :=
  let truncated := neuralTruncate β fexp state
  { mantissa := neuralConditionalIncrement
      (neuralRoundNearestLocation neuralNearestEvenChoice
        truncated.mantissa truncated.location) truncated.mantissa
    exponent := truncated.exponent }

/-- Truncation never changes a zero mantissa into a nonzero one. -/
@[simp] theorem neuralTruncate_zero_mantissa (β : NeuralRadix) (fexp : ℤ → ℤ)
    (exponent : ℤ) (location : NeuralLocation) :
    (neuralTruncate β fexp {
      mantissa := 0, exponent := exponent, location := location }).mantissa = 0 := by
  simp only [neuralTruncate, neuralTruncationShift, neuralDigits_zero, Nat.cast_zero,
    zero_add, sub_pos]
  split_ifs
  · simp [neuralTruncateBy]
  · rfl

section GenericFormat

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]

/-- Canonical mantissa/exponent representation produced by a rounding mode. -/
noncomputable def neuralRoundedFloat (rnd : ℝ → ℤ) (x : ℝ) : NeuralFloat β :=
  { mantissa := rnd (neuralScaledMantissa β fexp x)
    exponent := neuralCexp β fexp x }

/--
If a finite decision procedure computes the rounded scaled mantissa, `neuralRound` is exactly the
real value of the corresponding mantissa/exponent pair.
-/
theorem neuralRound_eq_toReal_of_scaled_round_eq (rnd : ℝ → ℤ) (x : ℝ) (mantissa : ℤ)
    (hmantissa : rnd (neuralScaledMantissa β fexp x) = mantissa) :
    neuralRound (β := β) (fexp := fexp) rnd x =
      neuralToReal (β := β) {
        mantissa := mantissa
        exponent := neuralCexp β fexp x } := by
  simp [neuralRound, neuralToReal, hmantissa]

/-- A scaled-mantissa bracket computes format-level downward rounding. -/
theorem neuralRound_floor_of_scaledBracket (x : ℝ) {m : ℤ} {location : NeuralLocation}
    (hl : NeuralInbetweenInt m (neuralScaledMantissa β fexp x) location) :
    neuralRound (β := β) (fexp := fexp) neuralFloorRound x =
      neuralToReal (β := β) {
        mantissa := m
        exponent := neuralCexp β fexp x } := by
  apply neuralRound_eq_toReal_of_scaled_round_eq
  exact neuralInbetweenInt_floor hl

/-- A scaled-mantissa bracket computes format-level upward rounding. -/
theorem neuralRound_ceil_of_scaledBracket (x : ℝ) {m : ℤ} {location : NeuralLocation}
    (hl : NeuralInbetweenInt m (neuralScaledMantissa β fexp x) location) :
    neuralRound (β := β) (fexp := fexp) neuralCeilRound x =
      neuralToReal (β := β) {
        mantissa := neuralConditionalIncrement (neuralRoundUpLocation location) m
        exponent := neuralCexp β fexp x } := by
  apply neuralRound_eq_toReal_of_scaled_round_eq
  exact neuralInbetweenInt_ceil hl

/-- A scaled-mantissa bracket computes format-level nearest-even rounding. -/
theorem neuralRound_nearestEven_of_scaledBracket (x : ℝ) {m : ℤ}
    {location : NeuralLocation}
    (hl : NeuralInbetweenInt m (neuralScaledMantissa β fexp x) location) :
    neuralRound (β := β) (fexp := fexp) neuralNearestEven x =
      neuralToReal (β := β) {
        mantissa := neuralConditionalIncrement
          (neuralRoundNearestLocation neuralNearestEvenChoice m location) m
        exponent := neuralCexp β fexp x } := by
  apply neuralRound_eq_toReal_of_scaled_round_eq
  exact neuralInbetweenInt_nearestEven hl

/-- Canonical mantissa selected by effective nearest-even rounding. -/
noncomputable def neuralNearestEvenMantissa (x : ℝ) : ℤ :=
  neuralConditionalIncrement
    (neuralRoundNearestLocation neuralNearestEvenChoice ⌊x⌋ (neuralIntegerLocation x)) ⌊x⌋

/-- Every nearest-even format rounding has a canonical computed mantissa/exponent representation. -/
theorem neuralRound_nearestEven_computed (x : ℝ) :
    neuralRound (β := β) (fexp := fexp) neuralNearestEven x =
      neuralToReal (β := β) {
        mantissa := neuralNearestEvenMantissa (neuralScaledMantissa β fexp x)
        exponent := neuralCexp β fexp x } := by
  exact neuralRound_nearestEven_of_scaledBracket x
    (neuralIntegerLocation_spec (neuralScaledMantissa β fexp x))

/--
Nearest-even selection after canonical truncation agrees with generic rounded-real semantics.
-/
theorem neuralRoundTruncatedNearestEven_correct
    (state : NeuralTruncationState) {x : ℝ} (hx : 0 < x) (hl : state.Brackets β x)
    (hexp : state.exponent ≤ neuralCexp β fexp x ∨
      state.exponent ≤ fexp ((neuralDigits β state.mantissa : ℕ) + state.exponent)) :
    neuralToReal (neuralRoundTruncatedNearestEven β fexp state) =
      neuralRound (β := β) (fexp := fexp) neuralNearestEven x := by
  let truncated := neuralTruncate β fexp state
  have ht := neuralTruncate_brackets_and_exponent state hx hl hexp
  have hs : NeuralInbetweenInt truncated.mantissa
      (neuralScaledMantissa β fexp x) truncated.location :=
    truncated.scaledBrackets_of_exponent_eq ht.1 ht.2
  have hr := neuralRound_nearestEven_of_scaledBracket (β := β) (fexp := fexp) x hs
  rw [hr]
  simp [neuralRoundTruncatedNearestEven, truncated, ht.2]

end GenericFormat

end TorchLean.Floats
