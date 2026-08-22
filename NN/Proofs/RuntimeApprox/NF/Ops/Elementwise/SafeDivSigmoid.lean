/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.Ops.Elementwise.SoftplusSafeLog

/-!
# NF Elementwise Bounds: Safe Division and Sigmoid
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

-- ---------------------------------------------------------------------------
-- Safe division (clamped): `x / max y ε`
-- ---------------------------------------------------------------------------

/-- Spec-side safe division with a clamped denominator. -/
def safeDiv (ε : ℝ) (x y : ℝ) : ℝ :=
  x / max y ε

/-- Runtime implementation of `safeDiv` as a single rounded primitive. -/
def safeDivR (ε : ℝ) (xR yR : R) : R :=
  TorchLean.Floats.NF.ofReal (β := β) (fexp := fexp) (rnd := rnd)
    (safeDiv (ε := ε)
      (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR)
      (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR))

/--
Forward approximation bound for `safeDiv` in `NF`.

`safeDiv ε x y = x / max y ε` clamps the denominator away from 0. For `ε > 0`, this yields an
unconditional bound with explicit `(1/ε)` and `(1/ε^2)` sensitivity terms plus one rounding-ULP
  term.
-/
lemma approx_safeDiv_nf {x y : ℝ} {xR yR : R} {epsx epsy ε : ℝ}
    (hε : 0 < ε)
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ epsx)
    (hy : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR - y) ≤ epsy) :
    abs
        (toSpec (β := β) (fexp := fexp) (rnd := rnd)
            (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε xR yR) -
          safeDiv (ε := ε) x y) ≤
      (1 / ε) * epsx +
        (abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) + epsx) * (epsy / (ε * ε)) +
        neuralUlp β fexp
            (safeDiv (ε := ε)
              (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR)
              (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR)) / 2 := by
  set xhat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) xR
  set yhat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) yR
  set uhat : ℝ := max yhat ε
  set u : ℝ := max y ε

  have uhat_ge : ε ≤ uhat := le_max_right _ _
  have u_ge : ε ≤ u := le_max_right _ _
  have uhat_pos : 0 < uhat := lt_of_lt_of_le hε uhat_ge
  have u_pos : 0 < u := lt_of_lt_of_le hε u_ge

  have hx' : abs (xhat - x) ≤ epsx := by
    simpa [xhat, abs_sub_comm] using hx
  have hy' : abs (yhat - y) ≤ epsy := by
    simpa [yhat, abs_sub_comm] using hy

  have hx_abs : abs x ≤ abs xhat + epsx := by
    have h0 : abs x ≤ abs (x - xhat) + abs xhat := by
      simpa using (abs_sub_le x xhat 0)
    have h1 : abs (x - xhat) = abs (xhat - x) := by simp [abs_sub_comm]
    have h2 : abs (x - xhat) ≤ epsx := by simpa [h1] using hx'
    have := le_trans h0 (by
      simpa [add_assoc, add_left_comm, add_comm] using add_le_add_right h2 (abs xhat))
    simpa [add_assoc, add_left_comm, add_comm] using this

  have hmax : abs (uhat - u) ≤ epsy := by
    have hLip : abs (max yhat ε - max y ε) ≤ abs (yhat - y) := by
      simpa [abs_sub_comm] using (abs_max_sub_max_le_abs yhat y ε)
    exact le_trans (by simpa [uhat, u, abs_sub_comm] using hLip) hy'

  have hround :
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε xR yR) -
            safeDiv (ε := ε) xhat yhat) ≤
        neuralUlp β fexp (safeDiv (ε := ε) xhat yhat) / 2 := by
    simpa [safeDivR, safeDiv, xhat, yhat, toSpec, TorchLean.Floats.NF.toReal,
      TorchLean.Floats.NF.ofReal,
      TorchLean.Floats.NF.roundR, Proofs.RuntimeRoundingApprox.roundR] using
        (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd)
          (safeDiv (ε := ε) xhat yhat))

  have hdiff :
      abs (safeDiv (ε := ε) xhat yhat - safeDiv (ε := ε) x y) ≤
        (1 / ε) * epsx + (abs xhat + epsx) * (epsy / (ε * ε)) := by
    -- Split numerator and denominator effects.
    have hsplit :
        abs (xhat / uhat - x / u) ≤ abs (xhat / uhat - x / uhat) + abs (x / uhat - x / u) := by
      -- `|a-c| ≤ |a-b| + |b-c|` with `b = x/uhat`.
      simpa [sub_eq_add_neg, add_assoc] using
        abs_sub_le (xhat / uhat) (x / uhat) (x / u)

    have hnum :
        abs (xhat / uhat - x / uhat) ≤ (1 / ε) * epsx := by
      have hsub : xhat / uhat - x / uhat = (xhat - x) / uhat := by
        simpa using (sub_div xhat x uhat).symm
      have hinv : (1 : ℝ) / uhat ≤ (1 : ℝ) / ε := by
        simpa [one_div] using (one_div_le_one_div_of_le hε uhat_ge)
      have hcoef : 0 ≤ (1 : ℝ) / ε := by exact le_of_lt (one_div_pos.2 hε)
      calc
        abs (xhat / uhat - x / uhat)
            = abs ((xhat - x) / uhat) := by simp [hsub]
        _ = abs (xhat - x) * ((1 : ℝ) / uhat) := by
                simp [div_eq_mul_inv, abs_mul, abs_inv, abs_of_pos uhat_pos]
        _ ≤ abs (xhat - x) * ((1 : ℝ) / ε) := by
              exact mul_le_mul_of_nonneg_left hinv (abs_nonneg _)
        _ ≤ epsx * ((1 : ℝ) / ε) := by
              exact mul_le_mul_of_nonneg_right hx' hcoef
        _ = (1 / ε) * epsx := by ring

    have hden :
        abs (x / uhat - x / u) ≤ (abs xhat + epsx) * (epsy / (ε * ε)) := by
      have hsub : x / uhat - x / u = x * ((1 : ℝ) / uhat - (1 : ℝ) / u) := by
        simp [div_eq_mul_inv, sub_eq_add_neg, mul_add, mul_comm]
      -- Bound `|1/uhat - 1/u|` using algebra and the `max` Lipschitz bound.
      have h_inv :
          abs ((1 : ℝ) / uhat - (1 : ℝ) / u) ≤ epsy / (ε * ε) := by
        have hu0 : uhat ≠ 0 := ne_of_gt uhat_pos
        have hv0 : u ≠ 0 := ne_of_gt u_pos
        have hiden :
            (1 : ℝ) / uhat - (1 : ℝ) / u = (u - uhat) / (uhat * u) := by
          field_simp [hu0, hv0]
        have hprod_ge : (ε * ε) ≤ uhat * u := by
          have : ε ≤ uhat := uhat_ge
          have : ε ≤ u := u_ge
          nlinarith
        have hprod_pos : 0 < uhat * u := mul_pos uhat_pos u_pos
        have hprod_inv :
            (1 : ℝ) / (uhat * u) ≤ (1 : ℝ) / (ε * ε) := by
          simpa [one_div] using (one_div_le_one_div_of_le (mul_pos hε hε) hprod_ge)
        have hprod_inv_nonneg : 0 ≤ (1 : ℝ) / (uhat * u) := by
          exact le_of_lt (one_div_pos.2 hprod_pos)
        calc
          abs ((1 : ℝ) / uhat - (1 : ℝ) / u)
              = abs ((u - uhat) / (uhat * u)) := by
                  simpa using congrArg abs hiden
          _ = abs (u - uhat) / (uhat * u) := by
                  simpa [abs_of_pos hprod_pos] using (abs_div (u - uhat) (uhat * u))
          _ = abs (u - uhat) * ((1 : ℝ) / (uhat * u)) := by
                  simp [div_eq_mul_inv]
          _ ≤ abs (u - uhat) * ((1 : ℝ) / (ε * ε)) := by
                exact mul_le_mul_of_nonneg_left hprod_inv (abs_nonneg _)
          _ ≤ epsy * ((1 : ℝ) / (ε * ε)) := by
                have : abs (u - uhat) ≤ epsy := by simpa [abs_sub_comm] using hmax
                exact mul_le_mul_of_nonneg_right this (by
                  have : 0 < (1 : ℝ) / (ε * ε) := by
                    exact one_div_pos.2 (mul_pos hε hε)
                  exact le_of_lt this)
          _ = epsy / (ε * ε) := by
                simp [div_eq_mul_inv, mul_comm]

      calc
        abs (x / uhat - x / u)
            = abs (x * ((1 : ℝ) / uhat - (1 : ℝ) / u)) := by simp [hsub]
        _ = abs x * abs ((1 : ℝ) / uhat - (1 : ℝ) / u) := by
              simp [abs_mul]
        _ ≤ (abs xhat + epsx) * abs ((1 : ℝ) / uhat - (1 : ℝ) / u) := by
              exact mul_le_mul_of_nonneg_right hx_abs (abs_nonneg _)
        _ ≤ (abs xhat + epsx) * (epsy / (ε * ε)) := by
              have epsx_nonneg : 0 ≤ epsx := le_trans (abs_nonneg _) hx'
              have hsum_nonneg : 0 ≤ abs xhat + epsx := add_nonneg (abs_nonneg _) epsx_nonneg
              exact mul_le_mul_of_nonneg_left h_inv hsum_nonneg

    -- Combine.
    have hadd :=
      calc
        abs (xhat / uhat - x / u)
            ≤ abs (xhat / uhat - x / uhat) + abs (x / uhat - x / u) := hsplit
        _ ≤ (1 / ε) * epsx + (abs xhat + epsx) * (epsy / (ε * ε)) := by
              exact add_le_add hnum hden
        _ = (1 / ε) * epsx + (abs xhat + epsx) * (epsy / (ε * ε)) := by rfl
    -- Rewrite `safeDiv`.
    simpa [safeDiv, uhat, u] using hadd

  -- Final triangle inequality: rounding + input sensitivity.
  have :=
    calc
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd)
              (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε xR yR) -
            safeDiv (ε := ε) x y)
          ≤
        abs
            (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε xR yR) -
              safeDiv (ε := ε) xhat yhat) +
          abs (safeDiv (ε := ε) xhat yhat - safeDiv (ε := ε) x y) := by
            simpa [sub_eq_add_neg, add_assoc] using
              abs_sub_le
                (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                  (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε xR yR))
                (safeDiv (ε := ε) xhat yhat)
                (safeDiv (ε := ε) x y)
      _ ≤
        neuralUlp β fexp (safeDiv (ε := ε) xhat yhat) / 2 +
          ((1 / ε) * epsx + (abs xhat + epsx) * (epsy / (ε * ε))) := by
            exact add_le_add hround hdiff
      _ = (1 / ε) * epsx + (abs xhat + epsx) * (epsy / (ε * ε)) +
        neuralUlp β fexp (safeDiv (ε := ε) xhat yhat) / 2 := by
            ring

  simpa [xhat, yhat] using this

/-- Error budget for division with exact denominator lower bound `η` and denominator approximation
error `epsy`. The caller must separately establish `epsy < η`; otherwise the rounded denominator
may cross zero and no finite perturbation bound follows.
-/
def divPosErrorBound (η epsx epsy xhat yhat : ℝ) : ℝ :=
  (1 / (η - epsy)) * epsx +
    (abs xhat + epsx) * (epsy / ((η - epsy) * (η - epsy))) +
    neuralUlp β fexp (xhat / yhat) / 2

/-- Forward error for ordinary division when the exact denominator stays positively separated
from zero and its approximation budget is smaller than that separation.

The effective runtime margin is `η - epsy`: from `η ≤ y` and `|ŷ - y| ≤ epsy` we obtain
`η - epsy ≤ ŷ`. The result is proved through the shared clamped-division analysis, after showing
that neither the exact nor rounded denominator activates the clamp. This is the form needed by
stable softmax and normalization, where a mathematical lower bound on a reduction must survive
rounding before division is allowed.
-/
lemma approx_div_nf_of_pos_lb {x y : ℝ} {xR yR : R} {epsx epsy η : ℝ}
    (hyLower : η ≤ y) (hbudget : epsy < η)
    (hx : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR - x) ≤ epsx)
    (hy : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR - y) ≤ epsy) :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (xR / yR) - x / y) ≤
      divPosErrorBound (β := β) (fexp := fexp) η epsx epsy
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR)
        (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR) := by
  let margin : ℝ := η - epsy
  let yhat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) yR
  have hepsy : 0 ≤ epsy :=
    le_trans (abs_nonneg (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR - y)) hy
  have hmargin : 0 < margin := by
    dsimp [margin]
    linarith
  have hyhatLower : margin ≤ yhat := by
    have hdiff : y - yhat ≤ epsy := by
      calc
        y - yhat ≤ abs (y - yhat) := le_abs_self _
        _ = abs (yhat - y) := abs_sub_comm _ _
        _ ≤ epsy := by simpa [yhat] using hy
    dsimp [margin]
    linarith
  have hyMargin : margin ≤ y := by
    dsimp [margin]
    linarith
  have hmaxHat : max (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR) margin =
      toSpec (β := β) (fexp := fexp) (rnd := rnd) yR :=
    max_eq_left (by simpa [yhat] using hyhatLower)
  have hmaxReal : max y margin = y := max_eq_left hyMargin
  have hruntime :
      safeDivR (β := β) (fexp := fexp) (rnd := rnd) margin xR yR = xR / yR := by
    simp only [safeDivR, safeDiv, hmaxHat]
    rfl
  have hspec : safeDiv (ε := margin) x y = x / y := by
    simp [safeDiv, hmaxReal]
  have hsafe :=
    approx_safeDiv_nf (β := β) (fexp := fexp) (rnd := rnd)
      (x := x) (y := y) (xR := xR) (yR := yR)
      (epsx := epsx) (epsy := epsy) (ε := margin) hmargin hx hy
  rw [hruntime, hspec] at hsafe
  simpa [safeDiv, hmaxHat, margin, divPosErrorBound] using hsafe

/--
Per-entry bound tensor for `safeDiv`.

This is the elementwise lifting of `approx_safeDiv_nf`'s bound (with a max-clamped denominator).
-/
def safeDivBoundTensor {s : Shape} (ε epsx epsy : ℝ) (xR yR : Tensor R s) : SpecTensor s :=
  map2Spec
    (fun a b =>
      (1 / ε) * epsx +
        (abs a + epsx) * (epsy / (ε * ε)) +
        neuralUlp β fexp (safeDiv (ε := ε) a b) / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) yR)

/-- Per-entry budget for ordinary division on a certified positive denominator domain.

Unlike `safeDivBoundTensor`, this definition does not change the operation by clamping its
denominator. The accompanying theorem therefore requires `epsy < η`, ensuring that an exact lower
bound `η ≤ y` remains positive after the denominator is rounded.
-/
def divPosBoundTensor {s : Shape} (η epsx epsy : ℝ)
    (xR yR : Tensor R s) : SpecTensor s :=
  map2Spec
    (divPosErrorBound (β := β) (fexp := fexp) η epsx epsy)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) yR)

/-- Shape-generic forward error for ordinary elementwise division by positive denominators.

The domain condition is stated over the exact tensor, while `epsy < η` certifies that every
runtime denominator remains separated from zero. This is the reusable division rule for softmax,
normalization, and positive quantization scales; callers do not need a rank-specific theorem.
-/
theorem approxTensor_div_spec_of_pos_lb {s : Shape} (η : ℝ) :
    ∀ {xS yS : SpecTensor s} {xR yR : Tensor R s} {epsx epsy : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR epsx →
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) yS yR epsy →
      Tensor.Forall (fun y : ℝ => η ≤ y) yS →
      epsy < η →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (divSpec xS yS) (divSpec xR yR)
          (linfNorm (divPosBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) η epsx epsy xR yR)) := by
  induction s with
  | scalar =>
      intro xS yS xR yR epsx epsy hx hy hdom hmargin
      cases xS with
      | scalar x =>
          cases yS with
          | scalar y =>
              cases xR with
              | scalar xR =>
                  cases yR with
                  | scalar yR =>
                      have hx' := (approxTensor_scalar_iff (α := R)
                        (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))).mp hx
                      have hy' := (approxTensor_scalar_iff (α := R)
                        (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))).mp hy
                      have hdiv := approx_div_nf_of_pos_lb
                        (β := β) (fexp := fexp) (rnd := rnd)
                        (by simpa using hdom) hmargin hx' hy'
                      apply (approxTensor_scalar_iff (α := R)
                        (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))).mpr
                      change
                        abs
                            (toSpec (β := β) (fexp := fexp) (rnd := rnd) (xR / yR) -
                              x / y) ≤
                          abs
                            (divPosErrorBound (β := β) (fexp := fexp) η epsx epsy
                              (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR)
                              (toSpec (β := β) (fexp := fexp) (rnd := rnd) yR))
                      exact le_trans hdiv (le_abs_self _)
  | dim n inner ih =>
      intro xS yS xR yR epsx epsy hx hy hdom hmargin
      cases xS with
      | dim valuesXS =>
          cases yS with
          | dim valuesYS =>
              cases xR with
              | dim valuesXR =>
                  cases yR with
                  | dim valuesYR =>
                      let bound := linfNorm
                        (divPosBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                          (s := .dim n inner) η epsx epsy
                          (Tensor.dim valuesXR) (Tensor.dim valuesYR))
                      have hbound : 0 ≤ bound := by
                        simpa [bound] using
                          (linf_norm_nonneg
                            (t := divPosBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                              (s := .dim n inner) η epsx epsy
                              (Tensor.dim valuesXR) (Tensor.dim valuesYR)))
                      refine approxTensor_dim_of_forall
                        (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                        (xS := divSpec (Tensor.dim valuesXS) (Tensor.dim valuesYS))
                        (xR := divSpec (Tensor.dim valuesXR) (Tensor.dim valuesYR))
                        (eps := bound) hbound ?_
                      intro i
                      have hxI := approxTensor_dim_get (α := R)
                        (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hx i
                      have hyI := approxTensor_dim_get (α := R)
                        (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hy i
                      have hlocal := ih hxI hyI (by simpa using hdom i) hmargin
                      have hle :
                          linfNorm
                              (divPosBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                                (s := inner) η epsx epsy (valuesXR i) (valuesYR i)) ≤ bound := by
                        have h := linf_norm_le_get_dim
                          (t := divPosBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                            (s := .dim n inner) η epsx epsy
                            (Tensor.dim valuesXR) (Tensor.dim valuesYR)) i
                        change
                          linfNorm
                              (divPosBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                                (s := inner) η epsx epsy (valuesXR i) (valuesYR i)) ≤ bound at h
                        exact h
                      exact approxTensor_mono hlocal hle

/--
`approxTensor` bound for `safeDiv` lifted to arbitrary tensor shapes.

This is the tensor-level wrapper around `approx_safeDiv_nf`, built via
  `approxTensor_map2_spec_of_scalar_bound`.
-/
theorem approxTensor_safeDiv_spec {s : Shape} (ε : ℝ) (hε : 0 < ε) :
    ∀ {xS yS : SpecTensor s} {xR yR : Tensor R s} {epsx epsy : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR epsx →
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) yS yR epsy →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (map2Spec (s := s) (safeDiv (ε := ε)) xS yS)
          (map2Spec (s := s) (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε) xR yR)
          (linfNorm (safeDivBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) ε epsx epsy
            xR yR)) := by
  intro xS yS xR yR epsx epsy hx hy
  have h :=
    approxTensor_map2_spec_of_scalar_bound (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
      rnd))
      (s := s)
      (fS := safeDiv (ε := ε))
      (fR := safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε)
      (bnd := fun a b epsx epsy =>
        (1 / ε) * epsx +
          (abs a + epsx) * (epsy / (ε * ε)) +
          neuralUlp β fexp (safeDiv (ε := ε) a b) / 2)
      (xS := xS) (yS := yS) (xR := xR) (yR := yR) (epsx := epsx) (epsy := epsy) hx hy (by
        intro x y xR yR hx hy
        simpa using
          (approx_safeDiv_nf (β := β) (fexp := fexp) (rnd := rnd)
            (x := x) (y := y) (xR := xR) (yR := yR) (epsx := epsx) (epsy := epsy) (ε := ε) hε hx
              hy))
  simpa [safeDivBoundTensor] using h

-- Sigmoid / Softmax (elementwise logistic) bounds.

/--
Scalar forward bound for `sigmoid` in `NF`.

`sigmoid(x) = 1 / (1 + exp(-x))` is implemented using the existing bounds for `exp`, `+`, and
division (with the denominator lower-bounded by 1).
-/
def sigmoidBoundScalar (xR : R) : ℝ :=
  let oneR : R := (1 : R)
  let denomR : R := oneR + MathFunctions.exp (-xR)
  let oneHat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) oneR
  let denomHat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) denomR
  let qhat : ℝ := oneHat / denomHat
  neuralUlp β fexp qhat / 2 +
    abs oneHat * abs (1 / denomHat) + abs oneHat + oneEps (β := β) (fexp := fexp)

/-- Per-entry bound tensor for `sigmoid`. -/
def sigmoidBoundTensor {s : Shape} (_eps : ℝ) (xR : Tensor R s) : SpecTensor s :=
  Spec.mapTensor (sigmoidBoundScalar (β := β) (fexp := fexp) (rnd := rnd)) xR

/--
`approxTensor` bound for `sigmoid` lifted to arbitrary tensor shapes.

This is the tensor-level wrapper around `sigmoid_bound_scalar` (scalar case) and the usual
componentwise `linf_norm` lifting (dimension case).
-/
theorem approxTensor_sigmoid_spec {s : Shape} :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) xS)
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) xR)
          (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR)) :=
            by
  intro xS xR eps hx
  induction s with
  | scalar =>
      cases xS with
      | scalar x =>
          cases xR with
          | scalar xR =>
              -- `sigmoid` is just a division `1 / (1 + exp (-x))`.
              let oneR : R := (1 : R)
              let denomR : R := oneR + MathFunctions.exp (-xR)
              let y : ℝ := (1 : ℝ) + Real.exp (-x)
              have hy : (1 : ℝ) ≤ y := by
                have : 0 ≤ Real.exp (-x) := by simpa using (Real.exp_nonneg (-x))
                linarith
              have hone :
                  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) oneR - (1 : ℝ)) ≤
                    oneEps (β := β) (fexp := fexp) := by
                change
                  abs ((TorchLean.Floats.NF.ofReal (β := β) (fexp := fexp) (rnd := rnd)
                    (1 : ℝ)).val - (1 : ℝ)) ≤ oneEps (β := β) (fexp := fexp)
                simpa [oneEps, NFBackend.toSpec, TorchLean.Floats.NF.toReal,
                  Proofs.RuntimeRoundingApprox.roundR, TorchLean.Floats.NF.roundR,
                  TorchLean.Floats.NF.ofReal] using
                  (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd :=
                    rnd) (1 : ℝ))
              have hdiv :=
                approx_div_nf_of_one_le (β := β) (fexp := fexp) (rnd := rnd)
                  (x := (1 : ℝ)) (y := y) (xR := oneR) (yR := denomR)
                  (epsx := oneEps (β := β) (fexp := fexp)) hy hone
              have hb :
                  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                          (Activation.Math.sigmoidSpec (α := R) xR) -
                        Activation.Math.sigmoidSpec (α := ℝ) x) ≤
                    sigmoidBoundScalar (β := β) (fexp := fexp) (rnd := rnd) xR := by
                simpa [Activation.Math.sigmoidSpec, sigmoidBoundScalar, oneEps, oneR, denomR, y,
                  MathFunctions.exp]
                  using hdiv
              have hle :
                  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                          (Activation.Math.sigmoidSpec (α := R) xR) -
                        Activation.Math.sigmoidSpec (α := ℝ) x) ≤
                    linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                      (s := Shape.scalar) eps (Tensor.scalar xR)) := by
                refine le_trans hb ?_
                -- `linf_norm` of a scalar tensor is `abs` of its entry.
                simpa [sigmoidBoundTensor, Spec.mapTensor, linfNorm, RuntimeApprox.linfNorm,
                  tensorLinfNorm, MathFunctions.abs] using
                  (le_abs_self (sigmoidBoundScalar (β := β) (fexp := fexp) (rnd := rnd) xR))
              exact
                (approxTensor_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                  (x := Activation.Math.sigmoidSpec (α := ℝ) x)
                  (xR := Activation.Math.sigmoidSpec (α := R) xR)
                  (eps := linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                    (s := Shape.scalar) eps (Tensor.scalar xR)))).2 (by
                      simpa using hle)
  | dim n s ih =>
      cases xS with
      | dim xSf =>
          cases xR with
          | dim xRf =>
              let B : ℝ :=
                linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                  (s := Shape.dim n s) eps (Tensor.dim xRf))
              have hB_nonneg : 0 ≤ B := by
                simpa [B] using (linf_norm_nonneg
                  (t := sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                    (s := Shape.dim n s) eps (Tensor.dim xRf)))
              have hcomp :
                  ∀ i : Fin n,
                    tensorDistance (α := SpecScalar) linfNorm
                        (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (xSf i))
                        (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                          rnd))
                          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (xRf i)))
                      ≤ B := by
                intro i
                have hx_i :=
                  approxTensor_dim_get (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                    (xS := Tensor.dim xSf) (xR := Tensor.dim xRf) (eps := eps) hx i
                have hih := ih (xS := xSf i) (xR := xRf i) hx_i
                have hB_ge :
                    linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                        (s := s) eps (xRf i)) ≤ B := by
                  simpa [B, sigmoidBoundTensor, Spec.mapTensor] using
                    (linf_norm_le_get_dim
                      (t := sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                        (s := Shape.dim n s) eps (Tensor.dim xRf)) i)
                have hdist :
                    tensorDistance (α := SpecScalar) linfNorm
                        (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (xSf i))
                        (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                          rnd))
                          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (xRf i)))
                      ≤ linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                        (s := s) eps (xRf i)) := by
                  simpa [approxTensor, approxWith] using hih
                exact le_trans hdist hB_ge
              have hf :
                  ∀ i ∈ List.finRange n,
                    tensorDistance (α := SpecScalar) linfNorm
                        (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (xSf i))
                        (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                          rnd))
                          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (xRf i)))
                      ≤ B := by
                intro i _hi
                exact hcomp i
              have hfold :=
                List.foldl_max_le_of_le (List.finRange n)
                  (fun i =>
                    tensorDistance (α := SpecScalar) linfNorm
                      (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (xSf i))
                      (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                        rnd))
                        (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (xRf i))))
                  (acc := (0 : ℝ)) (eps := B) hB_nonneg hf
              have :
                  tensorDistance (α := SpecScalar) linfNorm
                      (mapSpec (s := Shape.dim n s) (Activation.Math.sigmoidSpec (α := ℝ))
                        (Tensor.dim xSf))
                      (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                        rnd))
                        (mapSpec (s := Shape.dim n s) (Activation.Math.sigmoidSpec (α := R))
                          (Tensor.dim xRf)))
                    ≤ B := by
                change
                  List.foldl
                    (fun a i =>
                      max a
                        (tensorDistance (α := SpecScalar) linfNorm
                          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (xSf i))
                          (tensorToSpec (α := R)
                            (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (xRf i)))))
                    0 (List.finRange n) ≤ B
                exact hfold
              simpa [approxTensor, approxWith, B] using this
end NFBackend

end

end RuntimeApprox
end Proofs
