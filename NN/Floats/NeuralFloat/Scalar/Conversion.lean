/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.NeuralFloat.Core
public import NN.Floats.NeuralFloat.Metadata
public import NN.Floats.NeuralFloat.Rounding.Core

/-!
# NeuralFloat Conversion Helpers

This module contains small utilities for *moving values between*:

- real semantics (`ℝ`), and
- an annotated rounded value whose mathematical payload is `NeuralFloat β`.

Important scope note:

- The canonical rounding semantics is `neuralRound`
  (see `NN/Floats/NeuralFloat/Rounding/Core.lean`), which
  models **one** “compute in `ℝ`, then round” step and comes with the key half-ULP theorem
  `neural_error_bound_ulp` (under round-to-nearest assumptions).
- The helpers in this file are conversion utilities and examples: packaging the rounded result into a
  `AnnotatedNeuralFloat` and attaching explicit error metadata.

If you are doing proofs about rounding error, prefer theorems in:

- `NN/Floats/NeuralFloat/Rounding/Core.lean` (core half-ULP bound), and
- `NN/Floats/NeuralFloat/Error/Bounds.lean` (small derived bounds).
-/

@[expose] public section


namespace TorchLean.Floats.Conversion

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]

/--
Conversion result with error tracking.
-/
structure NeuralConversionResult (α : Type) where
  /-- Converted value. -/
  value : α
  /--
  Absolute error bound for the conversion step.

  For conversions that only *repackage* an already-rounded value, this may be a metadata field
  rather than a proved bound.
  -/
  errorBound : ℝ
  /-- The attached absolute error bound is nonnegative. -/
  errorBound_nonneg : 0 ≤ errorBound
  /--
  Optional ULP-scale metadata at the input point.

  This is not itself an error bound. For round-to-nearest, a typical one-step error bound is
  `ulp(x)/2`, proved separately (see `neural_error_bound_ulp`).
  -/
  ulpScale : Option ℝ
  /-- Training phase, when all values in the result have one unambiguous provenance phase. -/
  phase : Option TrainingPhase

namespace NeuralConversionResult

/-- A conversion result is exact when its attached absolute error is zero. -/
def IsExact {α : Type} (result : NeuralConversionResult α) : Prop :=
  result.errorBound = 0

end NeuralConversionResult

/--
Convert an annotated float to its real semantics while retaining its conversion metadata.

In words: the returned `value` is exactly `neuralToReal f`.
The `errorBound` field is the metadata stored in the input record; it is not recomputed here.
-/
noncomputable def neuralFloatToReal (f : AnnotatedNeuralFloat β) : NeuralConversionResult ℝ :=
  { value := f.toReal,
    errorBound := f.metadata.errorBound,
    errorBound_nonneg := f.metadata.errorBound_nonneg,
    ulpScale := none,
    phase := some f.metadata.phase }

/--
Round a real number `x` onto the `(β,fexp,rnd)` grid and attach conversion metadata.

In words:
The returned payload is the same one used by `neuralRound` (it uses
`mantissa := rnd (neuralScaledMantissa β fexp x)` and `exponent := neuralCexp β fexp x`), and
`errorBound = |neuralToReal(value) - x|` is the actual absolute error for this rounding step.

If you additionally assume round-to-nearest (`NeuralValidRndToNearest`), the theorem
`neural_error_bound_ulp` bounds this error by `ulp(x)/2`.
-/
noncomputable def realToNeuralFloat (precision : NeuralPrecision) (rnd : ℝ → ℤ)
    [NeuralValidRnd rnd] (x : ℝ) (phase : TrainingPhase := TrainingPhase.forward) :
    NeuralConversionResult (AnnotatedNeuralFloat β) := by
  let m : ℤ := rnd (neuralScaledMantissa β fexp x)
  let e : ℤ := neuralCexp β fexp x
  let f : NeuralFloat β :=
    { mantissa := m
      exponent := e }
  let y : ℝ := neuralToReal (β := β) f
  let err : ℝ := abs (y - x)
  exact
    { value :=
        { value := f
          metadata :=
            { precision := precision
              errorBound := err
              errorBound_nonneg := abs_nonneg _
              phase := phase } }
      errorBound := err
      errorBound_nonneg := abs_nonneg _
      ulpScale := some (neuralUlp β fexp x)
      phase := some phase }

/-- Convert an array of real values to annotated neural floats. -/
noncomputable def arrayToNeuralFloats (precision : NeuralPrecision) (rnd : ℝ → ℤ)
    [NeuralValidRnd rnd] (values : Array ℝ) (phase : TrainingPhase := TrainingPhase.forward) :
    NeuralConversionResult (Array (AnnotatedNeuralFloat β)) := by
  let conv := values.map (fun x =>
    realToNeuralFloat (β := β) (fexp := fexp) precision rnd x phase)
  let converted := conv.map (·.value)
  let maxError := (conv.map (·.errorBound)).foldl max 0
  let errorBound := max 0 maxError
  let maxUlp := (values.map (neuralUlp β fexp)).foldl max 0
  exact
    { value := converted
      errorBound := errorBound
      errorBound_nonneg := le_max_left 0 maxError
      ulpScale := some maxUlp
      phase := some phase }

/-- Convert an array of annotated neural floats back to real values. -/
noncomputable def neuralFloatsToArray (neuralFloats : Array (AnnotatedNeuralFloat β)) :
    NeuralConversionResult (Array ℝ) := by
  let converted := neuralFloats.map AnnotatedNeuralFloat.toReal
  let maxError := (neuralFloats.map (·.metadata.errorBound)).foldl max 0
  let errorBound := max 0 maxError
  exact {
    value := converted,
    errorBound := errorBound,
    errorBound_nonneg := le_max_left 0 maxError,
    ulpScale := none,
    phase := none
  }

end TorchLean.Floats.Conversion
