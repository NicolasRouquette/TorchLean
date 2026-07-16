/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.FP32.Notation
public import NN.Floats.NeuralFloat.Analysis.SterbenzFLT

/-!
# `FP32` Sterbenz: exact subtraction of near-equal binary32 values

Sterbenz's lemma at the binary32 configuration `fexp32 = FLTExp (-149) 24`.  If two representable
binary32 values are within a factor of two of each other, their difference is *exactly*
representable, so the binary32 subtraction incurs **no rounding error at all**.

Two forms are provided:

- `round32_sub_exact_of_sterbenz` — the spec-level statement about the rounding operator `round₃₂`:
  under the Sterbenz hypotheses, `round₃₂ (u - v) = u - v`.  This is Sterbenz's lemma as a theorem
  about `round₃₂`.
- `FP32.sub_exact_of_sterbenz` — the corollary phrased on the `FP32` scalar type:
  `(a - b).val = a.val - b.val` for representable `a`, `b`.

The proof composes the `FLT` Sterbenz lemma (`neural_generic_format_FLT_sterbenz`, which the
binary32 exponent function `fexp32` is an instance of) with the fact that rounding is the identity on
already-representable reals (`neural_round_preserves_generic`).
-/

@[expose] public section

namespace TorchLean.Floats

/--
Sterbenz's lemma for binary32, stated on the rounding operator: if `u` and `v` are representable
binary32 values with `0 < u`, `0 < v`, `u ≤ 2v`, and `v ≤ 2u`, then rounding their exact difference
is a no-op — `u - v` is already on the binary32 grid.
-/
theorem round32_sub_exact_of_sterbenz {u v : ℝ}
    (hu : neuralGenericFormat binaryRadix fexp32 u)
    (hv : neuralGenericFormat binaryRadix fexp32 v)
    (hupos : 0 < u) (hvpos : 0 < v) (huv : u ≤ 2 * v) (hvu : v ≤ 2 * u) :
    round₃₂ (u - v) = u - v := by
  -- `fexp32` is `FLTExp (-149) 24`; the `FLT` Sterbenz lemma gives a representable difference.
  have hfmt : neuralGenericFormat binaryRadix fexp32 (u - v) :=
    neural_generic_format_FLT_sterbenz (-149) 24 (by decide) hupos hvpos huv hvu hu hv
  -- Rounding is the identity on representable reals; `round₃₂` reduces to `neuralRound`.
  exact neural_round_preserves_generic (β := binaryRadix) (fexp := fexp32) rnd32 (u - v) hfmt

namespace FP32

/--
Sterbenz's lemma on the `FP32` scalar type: for representable `a`, `b` within a factor of two,
subtraction is exact, `(a - b).val = a.val - b.val`.  The result carries the operational meaning
that a near-equal binary32 subtraction is lossless.
-/
theorem sub_exact_of_sterbenz {a b : FP32}
    (ha : a.IsRepresentable) (hb : b.IsRepresentable)
    (hapos : 0 < a.val) (hbpos : 0 < b.val)
    (hab : a.val ≤ 2 * b.val) (hba : b.val ≤ 2 * a.val) :
    (a - b).val = a.val - b.val := by
  -- `(a - b).val` is by definition `round₃₂ (a.val - b.val)`.
  have hround : (a - b).val = round₃₂ (a.val - b.val) := rfl
  rw [hround]
  exact round32_sub_exact_of_sterbenz ha hb hapos hbpos hab hba

end FP32

end TorchLean.Floats
