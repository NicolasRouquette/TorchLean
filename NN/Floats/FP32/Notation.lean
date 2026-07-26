/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.FP32.Core
import Mathlib.Algebra.Order.Algebra

/-!
# Notation for TorchLean's FP32 model

TorchLean uses a proof-oriented float32 model (`FP32`) defined by:

- a radix $\beta=2$,
- the canonical IEEE-754 binary32 exponent function (`fexp32`), and
- round-to-nearest, ties-to-even (`rnd32`).

This file provides small, ergonomic aliases for the corresponding real-level operators:

- `round₃₂ x` (or ASCII `round32 x`): round $x\in\mathbb{R}$ to the binary32 grid.
- `ulp₃₂ x` and `eps₃₂ x`: the ULP scale (and half-ULP) associated with $x$.

We keep these under `TorchLean.Floats` so they are available where float semantics are in focus,
without polluting unrelated namespaces.
-/

@[expose] public section

namespace TorchLean.Floats

noncomputable section

/--
Real-level binary32 rounding operator for the canonical `fexp32`/`rnd32` configuration.

This is definitionally the same rounding operator used in the `NF`/`FP32` semantics, but phrased as
a function $\mathbb{R}\to\mathbb{R}$ (useful for bridge theorems and error bounds).
-/
noncomputable abbrev round32 (x : ℝ) : ℝ :=
  neuralRound (β := binaryRadix) (fexp := fexp32) rnd32 x

/-- One ULP at `x` for the canonical binary32 exponent configuration. -/
noncomputable abbrev ulp32 (x : ℝ) : ℝ :=
  neuralUlp binaryRadix fexp32 x

/-- Convenience abbreviation: half an ULP at `x`. -/
noncomputable abbrev eps32 (x : ℝ) : ℝ := ulp32 x / 2

/-- Binary32 has a smallest grid step, so its ULP at zero is $2^{-149}$. -/
@[simp] theorem ulp32_zero : ulp32 0 = neuralBpow binaryRadix (-149) := by
  exact neuralUlp_zero_FLT (-149) 24 (by norm_num)

/-- Unicode alias for `round32` (useful in error-bound statements). -/
noncomputable abbrev round₃₂ (x : ℝ) : ℝ :=
  round32 x

/-- Unicode alias for `ulp32` (useful in error-bound statements). -/
noncomputable abbrev ulp₃₂ (x : ℝ) : ℝ := ulp32 x

/-- Unicode alias for `eps32` (half-ULP). -/
noncomputable abbrev eps₃₂ (x : ℝ) : ℝ := eps32 x

/-
`round₃₂`/`ulp₃₂`/`eps₃₂` are purely ergonomic unicode aliases.

We keep the simp lemmas one-way (unicode → ASCII) to avoid accidental simp loops.
-/

/-- `round₃₂` is definitionally equal to `round32` (unicode → ASCII simp). -/
@[simp] lemma round₃₂_eq_round32 (x : ℝ) : round₃₂ x = round32 x := rfl

/-- `ulp₃₂` is definitionally equal to `ulp32` (unicode → ASCII simp). -/
@[simp] lemma ulp₃₂_eq_ulp32 (x : ℝ) : ulp₃₂ x = ulp32 x := rfl

/-- `eps₃₂` is definitionally equal to `eps32` (unicode → ASCII simp). -/
@[simp] lemma eps₃₂_eq_eps32 (x : ℝ) : eps₃₂ x = eps32 x := rfl

end

end TorchLean.Floats
