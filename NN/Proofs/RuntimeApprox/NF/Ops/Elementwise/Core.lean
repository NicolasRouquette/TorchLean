/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.Ops.Sum

/-!
# NF Elementwise Bounds: Core Arithmetic Budgets
-/

@[expose] public section

namespace Proofs
namespace RuntimeApprox

open Spec
open Tensor
open NN.MLTheory.Robustness.Spec

noncomputable section

namespace NFBackend

open TorchLean.Floats
open Proofs.RuntimeRoundingApprox

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]
variable {rnd : ℝ → ℤ} [NeuralValidRndToNearest rnd]

local notation "R" => TorchLean.Floats.NF β fexp rnd

-- Elementwise bounds lifted to tensors via `linf_norm`.

/--
Per-entry bound tensor for addition.

`add_bound_tensor epsx epsy xR yR` computes an elementwise error budget for `xR + yR`. Its
  `linf_norm`
is used as the output epsilon in `approxTensor_add_spec`.
-/
def addBoundTensor {s : Shape} (epsx epsy : ℝ) (xR yR : Tensor R s) : SpecTensor s :=
  map2Spec
    (fun a b =>
      epsx + epsy + neuralUlp β fexp (a + b) / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) yR)

/--
Per-entry bound tensor for subtraction.

Analogous to `add_bound_tensor`, but for `xR - yR` (and the corresponding spec subtraction).
-/
def subBoundTensor {s : Shape} (epsx epsy : ℝ) (xR yR : Tensor R s) : SpecTensor s :=
  map2Spec
    (fun a b =>
      epsx + epsy + neuralUlp β fexp (a - b) / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) yR)

/--
Per-entry bound tensor for multiplication.

This is the elementwise lifting of the scalar bound `approx_mul_nf`, tracking first-order error
propagation plus one rounding term.
-/
def mulBoundTensor {s : Shape} (epsx epsy : ℝ) (xR yR : Tensor R s) : SpecTensor s :=
  map2Spec
    (fun a b =>
      (abs a + epsx) * epsy + (abs b + epsy) * epsx + neuralUlp β fexp (a * b) / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) yR)

/--
Per-entry bound tensor for scaling by a runtime constant.

`scale_bound_tensor eps c xR` bounds the error of `xR * c` assuming the input is approximated within
`eps` and treating `c` as exact (relative to its own `toSpec` value).
-/
def scaleBoundTensor {s : Shape} (eps : ℝ) (c : R) (xR : Tensor R s) : SpecTensor s :=
  mapSpec
    (fun a =>
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) c) * eps +
        neuralUlp β fexp (a * toSpec (β := β) (fexp := fexp) (rnd := rnd) c)
          / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)

/-- Per-entry budget for scaling when the scalar coefficient is itself approximate.

`scaleBoundTensor` is the zero-coefficient-error specialization. This general form is required by
constants such as `1 / sqrt(d)` in attention, where constructing the runtime coefficient already
incurs rounding.
-/
def scaleApproxBoundTensor {s : Shape} (eps epsC : ℝ) (c : R)
    (xR : Tensor R s) : SpecTensor s :=
  mapSpec
    (fun a =>
      (abs a + eps) * epsC +
        (abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) c) + epsC) * eps +
        neuralUlp β fexp
          (a * toSpec (β := β) (fexp := fexp) (rnd := rnd) c) / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)

/-- Per-entry bound tensor for negation. -/
def negBoundTensor {s : Shape} (eps : ℝ) (xR : Tensor R s) : SpecTensor s :=
  mapSpec
    (fun a => eps + neuralUlp β fexp (-a) / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)

/-- Per-entry bound tensor for absolute value. -/
def absBoundTensor {s : Shape} (eps : ℝ) (xR : Tensor R s) : SpecTensor s :=
  mapSpec
    (fun a => eps + neuralUlp β fexp (abs a) / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)

/--
Per-entry bound tensor for exponentiation (`exp`).

This matches `approx_exp_nf`: a mean-value-theorem bound on the real `exp` plus one rounding term.
-/
def expBoundTensor {s : Shape} (eps : ℝ) (xR : Tensor R s) : SpecTensor s :=
  mapSpec (fun a => expErrorBound (β := β) (fexp := fexp) a eps)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)

/-- Per-entry square-root budget on a domain with exact lower bound `η`. -/
def sqrtPosBoundTensor {s : Shape} (η eps : ℝ) (xR : Tensor R s) : SpecTensor s :=
  mapSpec
    (fun a => eps / Real.sqrt η + neuralUlp β fexp (Real.sqrt (max a 0)) / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)

/--
Per-entry bound tensor for hyperbolic tangent (`tanh`).

Currently uses the coarse unconditional bound from `approx_tanh_nf` (boundedness of `tanh`).
-/
def tanhBoundTensor {s : Shape} (_eps : ℝ) (xR : Tensor R s) : SpecTensor s :=
  mapSpec
    (fun a => (2 : ℝ) + neuralUlp β fexp (Real.tanh a) / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)

-- Safe log (clamped) bound.

/--
Per-entry bound tensor for `safeLog`.

`safeLog_bound_tensor ε eps xR` is the elementwise bound used by `approxTensor_safeLog_spec`, combining a
`(1/ε)` Lipschitz propagation term with one rounding-ULP term.
-/
def safeLogBoundTensor {s : Shape} (ε eps : ℝ) (xR : Tensor R s) : SpecTensor s :=
  mapSpec
    (fun a =>
      (1 / ε) * eps +
        neuralUlp β fexp (safeLog (ε := ε) a) / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)

/--
`approxTensor` bound for clamped log (`safeLog`) lifted to arbitrary tensor shapes.

This is the tensor-level wrapper around `approx_safeLog_nf`, using
  `approxTensor_map_spec_of_scalar_bound`.
-/
theorem approxTensor_safeLog_spec {s : Shape} (ε : ℝ) (hε : 0 < ε) :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (mapSpec (s := s) (safeLog (ε := ε)) xS)
          (mapSpec (s := s) (safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε) xR)
          (linfNorm (safeLogBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) ε eps xR))
            := by
  intro xS xR eps hx
  induction s with
  | scalar =>
      cases xS with
      | scalar x =>
          cases xR with
          | scalar xR =>
              have hx' :=
                (approxTensor_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                  (x := x) (xR := xR) (eps := eps)).1 hx
              have h :=
                approx_safeLog_nf (β := β) (fexp := fexp) (rnd := rnd) (x := x) (xR := xR) (eps :=
                  eps) (ε := ε)
                  hε hx'
              have hle :
                  abs
                      (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                          (safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε xR) -
                        safeLog (ε := ε) x) ≤
                    linfNorm
                      (safeLogBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := Shape.scalar)
                        ε eps
                        (Tensor.scalar xR)) := by
                refine le_trans h ?_
                -- `linf_norm` of a scalar tensor is the `abs` of its entry.
                simpa [safeLogBoundTensor, tensorToSpec, Spec.mapTensor, mapSpec, linfNorm,
                  RuntimeApprox.linfNorm, tensorDistance,
                    NN.MLTheory.Robustness.Spec.tensorDistance.tensor_sub,
                  tensorLinfNorm, MathFunctions.abs, safeLog] using le_abs_self _
              -- Wrap back into `approxTensor`.
              exact
                (approxTensor_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                  (x := safeLog (ε := ε) x)
                  (xR := safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε xR)
                  (eps := linfNorm
                    (safeLogBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := Shape.scalar) ε
                      eps
                      (Tensor.scalar xR)))).2 hle
  | dim n s ih =>
      cases xS with
      | dim xSf =>
          cases xR with
          | dim xRf =>
              let B : ℝ :=
                linfNorm
                  (safeLogBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := Shape.dim n s) ε
                    eps
                    (Tensor.dim xRf))
              have hB_nonneg : 0 ≤ B := by
                simpa [B] using
                  (linf_norm_nonneg (t := safeLogBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                    (s := Shape.dim n s) ε eps (Tensor.dim xRf)))
              have hcomp :
                  ∀ i : Fin n,
                tensorDistance (α := SpecScalar) linfNorm
                        (mapSpec (s := s) (safeLog (ε := ε)) (xSf i))
                        (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                          rnd))
                          (mapSpec (s := s) (safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε) (xRf
                            i)))
                      ≤ B := by
                intro i
                have hx_i := approxTensor_dim_get (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd
                  := rnd)) hx i
                have hih := ih (xS := xSf i) (xR := xRf i) hx_i
                have hB_ge :
                    linfNorm
                        (safeLogBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) ε eps
                          (xRf i)) ≤
                      B := by
                    simpa [B, safeLogBoundTensor, tensorToSpec, Spec.mapTensor, mapSpec] using
                      (linf_norm_le_get_dim
                        (t := safeLogBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                          (s := Shape.dim n s) ε eps (Tensor.dim xRf)) i)
                have hdist :
                    tensorDistance (α := SpecScalar) linfNorm
                        (mapSpec (s := s) (safeLog (ε := ε)) (xSf i))
                        (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                          rnd))
                          (mapSpec (s := s) (safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε) (xRf
                            i)))
                      ≤ linfNorm
                        (safeLogBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) ε eps
                          (xRf i)) := by
                  simpa [approxTensor, approxWith] using hih
                exact le_trans hdist hB_ge
              have hf :
                  ∀ i ∈ List.finRange n,
                tensorDistance (α := SpecScalar) linfNorm
                        (mapSpec (s := s) (safeLog (ε := ε)) (xSf i))
                        (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                          rnd))
                          (mapSpec (s := s) (safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε) (xRf
                            i)))
                      ≤ B := by
                intro i _hi
                exact hcomp i
              have hfold :=
                List.foldl_max_le_of_le (List.finRange n)
                  (fun i =>
                    tensorDistance (α := SpecScalar) linfNorm
                      (mapSpec (s := s) (safeLog (ε := ε)) (xSf i))
                      (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                        rnd))
                        (mapSpec (s := s) (safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε) (xRf
                          i))))
                  (acc := (0 : ℝ)) (eps := B) hB_nonneg hf
              have :
                  tensorDistance (α := SpecScalar) linfNorm
                      (mapSpec (s := Shape.dim n s) (safeLog (ε := ε)) (Tensor.dim xSf))
                      (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                        rnd))
                        (mapSpec (s := Shape.dim n s) (safeLogR (β := β) (fexp := fexp) (rnd :=
                          rnd) ε)
                          (Tensor.dim xRf)))
                      ≤ B := by
                    change
                      List.foldl
                        (fun a i =>
                          max a
                            (tensorDistance (α := SpecScalar) linfNorm
                              (mapSpec (s := s) (safeLog (ε := ε)) (xSf i))
                              (tensorToSpec (α := R)
                                (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                                (mapSpec (s := s)
                                  (safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε)
                                  (xRf i)))))
                        0 (List.finRange n) ≤ B
                    exact hfold
              simpa [approxTensor, approxWith, B] using this
end NFBackend

end

end RuntimeApprox
end Proofs
