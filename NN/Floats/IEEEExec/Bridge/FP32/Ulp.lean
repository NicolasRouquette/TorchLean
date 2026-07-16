/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Bridge.FP32.Ops

/-!
# IEEE32Exec and FP32: executable ULP and absorption

The op-level bridge theorems (`toReal_add_eq_fp32Round`, …) already show that each executable
`IEEE32Exec` operation refines the real-level `FP32` rounding model (`fp32Round`, definitionally the
`round32`/`ulp32` configuration in `NN.Floats.FP32.Notation`). What is still missing on the
executable side is the *unit in the last place*: `neuralUlp` / `ulp32` exist only as real-valued
`noncomputable` specifications, whereas an executable analysis needs to compute a ULP directly from a
bit pattern.

This file closes that gap:

* `IEEE32Exec.ulpExp` computes, from the decoded dyadic payload, the exponent `k` such that
  `2^k` is the ULP of the value — using only `Nat.log2` and the integer exponent selector `fexp32`.
  It is fully executable.
* `neuralBpow_ulpExp_eq_ulp32` certifies it: on the finite fragment, `2^(ulpExp x)` is exactly the
  real ULP `ulp32 (toReal x)` of the decoded value.

We also expose the *absorption* test that an executable adequacy analysis performs — "does adding
`b` to `a` change `a` at all?" — and prove it sound against the spec: if the executable sum equals
`a`, then the exact real result rounds back to `a` under `round32`. This is the executable check
that a swamped contribution has been lost, certified equal to the rounding specification.
-/

@[expose] public section

namespace TorchLean.Floats.IEEE754

open TorchLean.Floats

namespace IEEE32Exec

/-!
## Executable unit in the last place
-/

/--
The binary32 ULP *exponent* of a float, computed directly from its decoded dyadic payload.

For a finite nonzero value decoding to mantissa `m` and exponent `e`, its magnitude is
`⌊log₂ m⌋ + e + 1`, and the canonical exponent selected by `fexp32` at that magnitude is the ULP
exponent, so the ULP itself is `2^(ulpExp x)`. Zero (and, defensively, NaN/Inf) map to the smallest
grid step `2⁻¹⁴⁹`.

This is the executable counterpart of the real-valued, `noncomputable` `ulp32`.
-/
def ulpExp (x : IEEE32Exec) : Int :=
  match toDyadic? x with
  | some d => if d.mant = 0 then -149 else fexp32 (Int.ofNat (Nat.log 2 d.mant) + d.exp + 1)
  | none => -149

/--
`ulpExp` is certified against the specification: on the finite, nonzero fragment,
`2^(ulpExp x)` is exactly the real ULP `ulp₃₂ (toReal x)`.
-/
theorem neuralBpow_ulpExp_eq_ulp32_of_ne_zero (x : IEEE32Exec) {d : Dyadic}
    (hx : toDyadic? x = some d) (hm : d.mant ≠ 0) :
    neuralBpow binaryRadix (ulpExp x) = ulp₃₂ (toReal x) := by
  have hxreal : toReal x = dyadicToReal d := by rw [toReal_eq, hx]
  have hpos : 0 < _root_.abs (dyadicToReal d) := by
    rw [abs_dyadicToReal]
    exact mul_pos (by exact_mod_cast Nat.pos_of_ne_zero hm) (neuralBpow.pos binaryRadix d.exp)
  have hne : dyadicToReal d ≠ 0 := abs_pos.mp hpos
  have hue : ulpExp x = fexp32 (Int.ofNat (Nat.log 2 d.mant) + d.exp + 1) := by
    simp [ulpExp, hx, hm]
  rw [hue, hxreal]
  show neuralBpow binaryRadix (fexp32 (Int.ofNat (Nat.log 2 d.mant) + d.exp + 1))
      = neuralUlp binaryRadix fexp32 (dyadicToReal d)
  rw [neuralUlp.of_ne_zero binaryRadix fexp32 (dyadicToReal d) hne]
  simp only [neuralCexp]
  rw [neural_magnitude_dyadic d hm]

/--
`ulpExp` is certified on the whole finite fragment (including zero): if `x` decodes to a dyadic,
`2^(ulpExp x) = ulp₃₂ (toReal x)`.
-/
theorem neuralBpow_ulpExp_eq_ulp32 (x : IEEE32Exec) {d : Dyadic}
    (hx : toDyadic? x = some d) :
    neuralBpow binaryRadix (ulpExp x) = ulp₃₂ (toReal x) := by
  by_cases hm : d.mant = 0
  · -- Zero: the decoded value is `0`, and both sides collapse to `2⁻¹⁴⁹`.
    have hz : dyadicToReal d = 0 := by
      have h := abs_dyadicToReal d
      rw [hm, Nat.cast_zero, zero_mul] at h
      exact abs_eq_zero.mp h
    have hxreal : toReal x = 0 := by rw [toReal_eq, hx]; exact hz
    have hue : ulpExp x = -149 := by simp [ulpExp, hx, hm]
    rw [hue, hxreal]
    show neuralBpow binaryRadix (-149) = ulp32 0
    rw [ulp32_zero]
  · exact neuralBpow_ulpExp_eq_ulp32_of_ne_zero x hx hm

/-!
## Executable absorption test

An executable adequacy analysis flags a "swamped" (fully absorbed) contribution by checking whether
adding it changes the accumulator at all. `absorbs a b` is exactly that check, and it is sound: when
it fires, the exact real sum rounds back to `a` under the `FP32` specification.
-/

/-- The executable absorption test: `b` is absorbed into `a` when the float32 sum equals `a`. -/
def absorbs (a b : IEEE32Exec) : Bool := decide (add a b = a)

/--
Soundness of the executable sum's fixed point against the rounding specification: if the executable
float32 sum of `a` and `b` is `a` itself, then the *exact* real sum rounds to `toReal a` under
`round₃₂`. In other words, when the kernel reports that `b` was absorbed, the specification agrees
that the contribution was lost.
-/
theorem round32_add_eq_left_of_add_eq {a b : IEEE32Exec} {da db : Dyadic}
    (ha : toDyadic? a = some da) (hb : toDyadic? b = some db)
    (hfin : isFinite (add a b) = true) (habs : add a b = a) :
    round₃₂ (toReal a + toReal b) = toReal a := by
  have h := toReal_add_eq_fp32Round a b ha hb hfin
  rw [habs] at h
  exact h.symm

/-- Soundness of the executable `absorbs` test against the `round₃₂` specification. -/
theorem round32_add_eq_left_of_absorbs {a b : IEEE32Exec} {da db : Dyadic}
    (ha : toDyadic? a = some da) (hb : toDyadic? b = some db)
    (hfin : isFinite (add a b) = true) (habs : absorbs a b = true) :
    round₃₂ (toReal a + toReal b) = toReal a :=
  round32_add_eq_left_of_add_eq ha hb hfin (of_decide_eq_true habs)

end IEEE32Exec

end TorchLean.Floats.IEEE754
