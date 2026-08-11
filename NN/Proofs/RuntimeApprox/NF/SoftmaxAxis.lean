/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Analysis.Softmax
public import NN.Proofs.Autograd.FDeriv.Softmax
public import NN.Proofs.RuntimeApprox.NF.ShapeOps
public import NN.Spec.Layers.Attention

/-!
# Axis softmax

This file collects the mathematical facts needed by numerical bounds for the coupled, vector-valued
softmax used in attention. It is deliberately separate from the older NF scalar logistic helper:
axis softmax has a shared denominator and a dense Jacobian, while logistic acts independently on
each tensor entry.

The stable spec implementation is `Activation.softmaxVecSpec`. `Proofs.Analysis.Softmax` proves
that its entries are positive, sum to one, and lie in `[0,1]`. The analytic derivative is
`Proofs.Autograd.softmaxJvp`; its Jacobian is self-adjoint, so the same formula implements the VJP.
The theorem below adds the conservation law needed for backward error analysis: every softmax JVP
has coordinate sum zero.

References:

* A. Griewank and A. Walther, *Evaluating Derivatives*, 2nd ed., 2008, for forward/reverse
  differentiation of coupled maps.
* A. A. Baydin et al., "Automatic Differentiation in Machine Learning: a Survey," JMLR 2018.
* PyTorch `torch.nn.functional.softmax` documentation for the runtime axis convention.
-/

@[expose] public section

namespace Proofs
namespace RuntimeApprox
namespace AxisSoftmax

open scoped BigOperators

open _root_.Proofs.Autograd
open Spec
open Tensor
open NN.MLTheory.Robustness.Spec
open TorchLean.Floats

noncomputable section

variable {β : NeuralRadix} {fexp : ℤ -> ℤ} [NeuralValidExp fexp]
variable {rnd : ℝ -> ℤ} [NeuralValidRndToNearest rnd]

local notation "R" => TorchLean.Floats.NF β fexp rnd

/-! ## Maximum and max shift -/

omit [NeuralValidRndToNearest rnd] in
/-- Forgetting the `NF` format commutes with the nonempty-vector maximum exactly.

`NF.max` only selects one operand; it performs no arithmetic and therefore introduces no rounding
error. The proof uses one fold homomorphism rather than repeating a coordinate induction in every
stable normalization operator.
-/
theorem toSpec_maxVecSpec {n : Nat} (xR : Tensor R (.dim (Nat.succ n) .scalar)) :
    NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)
        (Tensor.toScalar (Activation.maxVecSpec xR)) =
      Tensor.toScalar
        (Activation.maxVecSpec
          (Spec.mapTensor (NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)) := by
  cases xR with
  | dim values =>
      let first : Fin (Nat.succ n) := ⟨0, Nat.succ_pos n⟩
      let runtimeValue : Fin (Nat.succ n) -> R := fun i => Tensor.toScalar (values i)
      let realValue : Fin (Nat.succ n) -> ℝ := fun i =>
        NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd) (runtimeValue i)
      have hfold := List.foldl_hom
        (f := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (g₁ := fun acc i => max acc (runtimeValue i))
        (g₂ := fun acc i => max acc (realValue i))
        (l := List.finRange (Nat.succ n))
        (init := runtimeValue first)
        (fun acc i => by
          simpa [realValue] using
            (NFBackend.toSpec_max (β := β) (fexp := fexp) (rnd := rnd)
              acc (runtimeValue i)).symm)
      simpa only [Activation.maxVecSpec, Spec.mapTensor, Spec.toScalar_mapTensor,
        Tensor.toScalar_scalar, runtimeValue, realValue, first] using hfold.symm

omit [NeuralValidRndToNearest rnd] in
/-- The maximum of a rounded vector approximates the real maximum with the same infinity-norm
budget as the vector itself. No additional ULP term appears because maximum is a selection.
-/
theorem approxT_maxVecSpec {n : Nat}
    {xS : SpecTensor (.dim (Nat.succ n) .scalar)}
    {xR : Tensor R (.dim (Nat.succ n) .scalar)} {eps : ℝ}
    (hx : approxT (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps) :
    approxT (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Activation.maxVecSpec xS) (Activation.maxVecSpec xR) eps := by
  classical
  cases xS with
  | dim specValues =>
  cases xR with
  | dim runtimeValues =>
  let xS : SpecTensor (.dim (Nat.succ n) .scalar) := Tensor.dim specValues
  let xR : Tensor R (.dim (Nat.succ n) .scalar) := Tensor.dim runtimeValues
  let xHat : SpecTensor (.dim (Nat.succ n) .scalar) :=
    Spec.mapTensor (NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR
  let mS : ℝ := Tensor.toScalar (Activation.maxVecSpec xS)
  let mHat : ℝ := Tensor.toScalar (Activation.maxVecSpec xHat)
  have hpoint : ∀ i, |Spec.toVec xHat i - Spec.toVec xS i| <= eps := by
    intro i
    have hi := approxT_dim_get (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) hx i
    cases hspec : specValues i with
    | scalar specValue =>
        cases hruntime : runtimeValues i with
        | scalar runtimeValue =>
            have hi' := (approxT_scalar_iff (α := R)
              (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))).mp
                (by simpa [hspec, hruntime] using hi)
            simpa [xS, xHat, xR, Spec.mapTensor, Spec.toVec, hspec, hruntime] using hi'
  rcases Proofs.exists_toVec_eq_maxVecSpec xS with ⟨iS, hiS⟩
  rcases Proofs.exists_toVec_eq_maxVecSpec xHat with ⟨iHat, hiHat⟩
  have hmS_le : mS <= mHat + eps := by
    have hcoord := (abs_sub_le_iff.mp (hpoint iS)).2
    have hmax := Proofs.toVec_le_maxVecSpec xHat iS
    dsimp [mS, mHat] at *
    linarith
  have hmHat_le : mHat <= mS + eps := by
    have hcoord := (abs_sub_le_iff.mp (hpoint iHat)).1
    have hmax := Proofs.toVec_le_maxVecSpec xS iHat
    dsimp [mS, mHat] at *
    linarith
  have hmaxError : |mHat - mS| <= eps := by
    rw [abs_le]
    constructor <;> linarith
  have hbridge := toSpec_maxVecSpec (β := β) (fexp := fexp) (rnd := rnd) xR
  cases hspecMax : Activation.maxVecSpec xS with
  | scalar specMaximum =>
      cases hruntimeMax : Activation.maxVecSpec xR with
      | scalar runtimeMaximum =>
          apply (approxT_scalar_iff (α := R)
            (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))).mpr
          have hbridge' :
              NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd) runtimeMaximum = mHat := by
            simpa [mHat, hruntimeMax] using hbridge
          have hmS : specMaximum = mS := by simp [mS, hspecMax]
          rw [hbridge', hmS]
          exact hmaxError

/-! ## Rounded stable softmax -/

/-- Error after subtracting the rounded maximum from every logit. -/
def shiftErrorBound {n : Nat} (eps : ℝ)
    (xR : Tensor R (.dim (Nat.succ n) .scalar)) : ℝ :=
  let maxR := Activation.maxVecSpec xR
  let maxRepR : Tensor R (.dim (Nat.succ n) .scalar) := Tensor.replicate maxR
  linfNorm (NFBackend.subBoundTensor (β := β) (fexp := fexp) eps eps xR maxRepR)

/-- Error after exponentiating the max-shifted logits. -/
def exponentErrorBound {n : Nat} (eps : ℝ)
    (xR : Tensor R (.dim (Nat.succ n) .scalar)) : ℝ :=
  let maxRepR : Tensor R (.dim (Nat.succ n) .scalar) :=
    Tensor.replicate (Activation.maxVecSpec xR)
  let shiftedR := subSpec xR maxRepR
  linfNorm
    (NFBackend.expBoundTensor (β := β) (fexp := fexp)
      (shiftErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR) shiftedR)

/-- Error in the sequentially rounded denominator reduction. -/
def denominatorErrorBound {n : Nat} (eps : ℝ)
    (xR : Tensor R (.dim (Nat.succ n) .scalar)) : ℝ :=
  NFBackend.sumBound (β := β) (fexp := fexp) (rnd := rnd)
    (exponentErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR)
    (Activation.maxShiftedExpVecSpec xR)

/-- Per-coordinate output budget for stable softmax.

The exact denominator is at least one. The checker must additionally establish
`denominatorErrorBound eps xR < 1`; this prevents the rounded denominator from crossing zero and
turns the division condition into an explicit, checkable certificate obligation.
-/
def softmaxBoundTensor {n : Nat} (eps : ℝ)
    (xR : Tensor R (.dim (Nat.succ n) .scalar)) : SpecTensor (.dim (Nat.succ n) .scalar) :=
  let exR := Activation.maxShiftedExpVecSpec xR
  let denomR : R := sumSpec exR
  let epsNum := exponentErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR
  let epsDenom := denominatorErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR
  Spec.mapTensor
    (fun numR => NFBackend.divPosErrorBound (β := β) (fexp := fexp)
      1 epsNum epsDenom
      (NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd) numR)
      (NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd) denomR))
    exR

/-- Infinity-norm forward-error budget for stable vector softmax. -/
def softmaxErrorBound {n : Nat} (eps : ℝ)
    (xR : Tensor R (.dim (Nat.succ n) .scalar)) : ℝ :=
  linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd) eps xR)

/-- The max-shifted `NF` implementation approximates real vector softmax.

Unlike a blanket continuity statement, the theorem follows the executable stages: maximum,
subtraction, exponential, sequential sum, and division. The sole side condition is the numerical
certificate check that the denominator error remains below its proved real lower bound `1`.
-/
theorem approxT_softmaxVecSpec {n : Nat}
    {xS : SpecTensor (.dim (Nat.succ n) .scalar)}
    {xR : Tensor R (.dim (Nat.succ n) .scalar)} {eps : ℝ}
    (hx : approxT (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps)
    (hdenom : denominatorErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR < 1) :
    approxT (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Activation.softmaxVecSpec xS) (Activation.softmaxVecSpec xR)
      (softmaxErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR) := by
  classical
  cases xS with
  | dim valuesS =>
  cases xR with
  | dim valuesR =>
  let xS : SpecTensor (.dim (Nat.succ n) .scalar) := Tensor.dim valuesS
  let xR : Tensor R (.dim (Nat.succ n) .scalar) := Tensor.dim valuesR
  let maxS := Activation.maxVecSpec xS
  let maxR := Activation.maxVecSpec xR
  let maxRepS : SpecTensor (.dim (Nat.succ n) .scalar) := Tensor.replicate maxS
  let maxRepR : Tensor R (.dim (Nat.succ n) .scalar) := Tensor.replicate maxR
  let shiftedS := subSpec xS maxRepS
  let shiftedR := subSpec xR maxRepR
  let exS := Activation.maxShiftedExpVecSpec xS
  let exR := Activation.maxShiftedExpVecSpec xR
  let denomS : ℝ := sumSpec exS
  let denomR : R := sumSpec exR
  let epsShift := shiftErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR
  let epsNum := exponentErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR
  let epsDenom := denominatorErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR
  let outBound := softmaxErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR

  have hmax := approxT_maxVecSpec (β := β) (fexp := fexp) (rnd := rnd) hx
  have hmaxRep : approxT (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      maxRepS maxRepR eps := by
    simpa [maxRepS, maxRepR, maxS, maxR] using
      (NFBackend.approxT_replicate (β := β) (fexp := fexp) (rnd := rnd)
        (s := .dim (Nat.succ n) .scalar) hmax)
  have hshift : approxT (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      shiftedS shiftedR epsShift := by
    have h := NFBackend.approxT_sub_spec (β := β) (fexp := fexp) (rnd := rnd) hx hmaxRep
    simpa [shiftedS, shiftedR, epsShift, shiftErrorBound, maxRepR, maxR] using h
  have hexp : approxT (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      exS exR epsNum := by
    have h := NFBackend.approxT_exp_spec (β := β) (fexp := fexp) (rnd := rnd) hshift
    simpa [exS, exR, shiftedS, shiftedR, epsNum, exponentErrorBound,
      Activation.maxShiftedExpVecSpec, maxRepS, maxRepR, maxS, maxR] using h
  have hsum : approxT (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Tensor.scalar denomS) (Tensor.scalar denomR) epsDenom := by
    have h := NFBackend.approxT_sum_spec (β := β) (fexp := fexp) (rnd := rnd) hexp
    simpa [denomS, denomR, epsDenom, denominatorErrorBound, epsNum, exR] using h
  have hdenomLower : (1 : ℝ) ≤ denomS := by
    simpa [denomS, exS] using (Proofs.softmax_shift_denom_bounds xS).1
  have hbudget : epsDenom < (1 : ℝ) := by
    simpa [epsDenom] using hdenom
  have hOutNonneg : 0 ≤ outBound := by
    simpa [outBound, softmaxErrorBound] using
      (linf_norm_nonneg
        (t := softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd) eps xR))

  refine approxT_dim_of_forall
    (xS := Activation.softmaxVecSpec xS)
    (xR := Activation.softmaxVecSpec xR)
    (eps := outBound) hOutNonneg ?_
  intro i
  cases hExS : exS with
  | dim exValuesS =>
      cases hExR : exR with
      | dim exValuesR =>
          have hexp' : approxT (α := R)
              (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
              (Tensor.dim exValuesS) (Tensor.dim exValuesR) epsNum := by
            simpa [← hExS, ← hExR] using hexp
          have hnumI' : approxT (α := R)
              (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
              (exValuesS i) (exValuesR i) epsNum := by
            simpa using
              (approxT_dim_get (α := R)
                (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) hexp' i)
          cases hNumS : exValuesS i with
          | scalar numS =>
              cases hNumR : exValuesR i with
              | scalar numR =>
                  have hnumScalar := (approxT_scalar_iff (α := R)
                    (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))).mp
                      (by simpa [hNumS, hNumR] using hnumI')
                  have hdenomScalar := (approxT_scalar_iff (α := R)
                    (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))).mp hsum
                  have hdiv := NFBackend.approx_div_nf_of_pos_lb
                    (β := β) (fexp := fexp) (rnd := rnd) (η := (1 : ℝ))
                    hdenomLower hbudget hnumScalar hdenomScalar
                  have hcoord := linf_norm_le_get_dim
                    (t := softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd) eps xR) i
                  have hdivBound :
                      NFBackend.divPosErrorBound (β := β) (fexp := fexp) 1 epsNum epsDenom
                          (NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd) numR)
                          (NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd) denomR) ≤
                        outBound := by
                    refine le_trans (le_abs_self _) ?_
                    simpa [outBound, softmaxErrorBound, softmaxBoundTensor, exR, denomR,
                      epsNum, epsDenom, hExR, hNumR, Spec.mapTensor, linfNorm,
                      RuntimeApprox.linfNorm, tensorLinfNorm, MathFunctions.abs] using hcoord
                  have hscalarOut : approxT (α := R)
                      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
                      (Tensor.scalar (numS / denomS)) (Tensor.scalar (numR / denomR)) outBound :=
                    (approxT_scalar_iff (α := R)
                      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))).mpr
                        (le_trans hdiv hdivBound)
                  have hSoftS : Activation.softmaxVecSpec xS =
                      divSpec exS (Tensor.replicate (Tensor.scalar denomS)) := by
                    rfl
                  have hSoftR : Activation.softmaxVecSpec xR =
                      divSpec exR (Tensor.replicate (Tensor.scalar denomR)) := by
                    rfl
                  rw [hSoftS, hSoftR]
                  simpa [hExS, hExR, hNumS, hNumR, Spec.Tensor.divSpec,
                    Spec.Tensor.map2Spec, Spec.Tensor.replicate] using hscalarOut

/-- Row-wise error tensor for last-axis softmax on a matrix. -/
def softmaxRowsBoundTensor {m n : Nat} (eps : ℝ)
    (xR : Tensor R (.dim m (.dim (Nat.succ n) .scalar))) :
    SpecTensor (.dim m (.dim (Nat.succ n) .scalar)) :=
  match xR with
  | Tensor.dim rows => Tensor.dim (fun i =>
      softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd) eps (rows i))

/-- Global infinity-norm budget for row-wise last-axis softmax. -/
def softmaxRowsErrorBound {m n : Nat} (eps : ℝ)
    (xR : Tensor R (.dim m (.dim (Nat.succ n) .scalar))) : ℝ :=
  linfNorm (softmaxRowsBoundTensor (β := β) (fexp := fexp) (rnd := rnd) eps xR)

/-- Matrix-level stable softmax theorem, obtained by applying the vector theorem independently to
each row.

The denominator obligation remains row-specific: a certificate may accept well-conditioned rows
without replacing them by a single pessimistic analytic assumption. The output uses one global
infinity-norm budget because that is the contract consumed by matrix multiplication and graph
composition.
-/
theorem approxT_softmaxRowsSpec {m n : Nat}
    {xS : SpecTensor (.dim m (.dim (Nat.succ n) .scalar))}
    {xR : Tensor R (.dim m (.dim (Nat.succ n) .scalar))} {eps : ℝ}
    (hx : approxT (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps)
    (hdenom : ∀ i : Fin m,
      denominatorErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps (Spec.get xR i) < 1) :
    approxT (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Activation.softmaxSpec xS) (Activation.softmaxSpec xR)
      (softmaxRowsErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR) := by
  classical
  cases xS with
  | dim rowsS =>
      cases xR with
      | dim rowsR =>
          let bound := softmaxRowsErrorBound (β := β) (fexp := fexp) (rnd := rnd)
            eps (Tensor.dim rowsR)
          have hbound : 0 ≤ bound := by
            simpa [bound, softmaxRowsErrorBound] using
              (linf_norm_nonneg
                (t := softmaxRowsBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                  eps (Tensor.dim rowsR)))
          refine approxT_dim_of_forall
            (xS := Activation.softmaxSpec (Tensor.dim rowsS))
            (xR := Activation.softmaxSpec (Tensor.dim rowsR))
            (eps := bound) hbound ?_
          intro i
          have hrow := approxT_dim_get (α := R)
            (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) hx i
          have hsoft := approxT_softmaxVecSpec (β := β) (fexp := fexp) (rnd := rnd) hrow
            (by simpa [Spec.get, getAtSpec] using hdenom i)
          have hrowLe :
              softmaxErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps (rowsR i) ≤ bound := by
            have h := linf_norm_le_get_dim
              (t := softmaxRowsBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                eps (Tensor.dim rowsR)) i
            simpa [bound, softmaxRowsErrorBound, softmaxRowsBoundTensor,
              softmaxErrorBound] using h
          have hsoft' := approxT_mono hsoft hrowLe
          simpa [Activation.softmaxSpec] using hsoft'

/-! ## Exact hard-masked softmax -/

/-- Stable softmax numerators for a row whose allowed maximum is already known.

Blocked coordinates are set to literal zero after exponentiation. This expression is equivalent to
the `some rowMax` branch of `Spec.hardMaskedSoftmaxVecSpec` and never introduces a finite masking
sentinel.
-/
def hardMaskedNumerators {α : Type} [Context α] {n : Nat}
    (scores : Tensor α (.dim n .scalar))
    (mask : Tensor Bool (.dim n .scalar)) (rowMax : α) :
    Tensor α (.dim n .scalar) :=
  let maxRep : Tensor α (.dim n .scalar) := Tensor.replicate (Tensor.scalar rowMax)
  let exponentials := expSpec (subSpec scores maxRep)
  map2Spec
    (fun value allowed => if allowed then value else 0) exponentials mask

/-- The staged numerator computation equals the fused expression used by the public spec. -/
theorem hardMaskedNumerators_eq_fused {α : Type} [Context α] {n : Nat}
    (scores : Tensor α (.dim n .scalar))
    (mask : Tensor Bool (.dim n .scalar)) (rowMax : α) :
    hardMaskedNumerators scores mask rowMax =
      map2Spec
        (fun score allowed => if allowed then MathFunctions.exp (score - rowMax) else 0)
        scores mask := by
  cases scores with
  | dim scoreValues =>
      cases mask with
      | dim maskValues =>
          apply congrArg Tensor.dim
          funext i
          cases hscore : scoreValues i with
          | scalar score =>
              cases hmask : maskValues i with
              | scalar allowed =>
                  simp [map2Spec, Tensor.replicate, hscore]

/-- Error after subtracting an approximate allowed-row maximum from every score. -/
def hardMaskedShiftError {n : Nat} (epsScores epsMax : ℝ)
    (scoresR : Tensor R (.dim n .scalar)) (rowMaxR : R) : ℝ :=
  let maxRepR : Tensor R (.dim n .scalar) := Tensor.replicate (Tensor.scalar rowMaxR)
  linfNorm
    (NFBackend.subBoundTensor (β := β) (fexp := fexp)
      epsScores epsMax scoresR maxRepR)

/-- Error in the hard-masked numerator vector; applying the mask adds no rounding error. -/
def hardMaskedNumeratorError {n : Nat} (epsScores epsMax : ℝ)
    (scoresR : Tensor R (.dim n .scalar))
    (_mask : Tensor Bool (.dim n .scalar)) (rowMaxR : R) : ℝ :=
  let maxRepR : Tensor R (.dim n .scalar) := Tensor.replicate (Tensor.scalar rowMaxR)
  let shiftedR := subSpec scoresR maxRepR
  let epsShift := hardMaskedShiftError (β := β) (fexp := fexp) (rnd := rnd)
    epsScores epsMax scoresR rowMaxR
  linfNorm
    (NFBackend.expBoundTensor (β := β) (fexp := fexp) epsShift shiftedR)

/-- Error in the sequentially rounded sum of the allowed numerators. -/
def hardMaskedDenominatorError {n : Nat} (epsScores epsMax : ℝ)
    (scoresR : Tensor R (.dim n .scalar))
    (mask : Tensor Bool (.dim n .scalar)) (rowMaxR : R) : ℝ :=
  NFBackend.sumBound (β := β) (fexp := fexp) (rnd := rnd)
    (hardMaskedNumeratorError (β := β) (fexp := fexp) (rnd := rnd)
      epsScores epsMax scoresR mask rowMaxR)
    (hardMaskedNumerators scoresR mask rowMaxR)

/-- Final per-coordinate budget for a nonempty hard-masked softmax row. -/
def hardMaskedSoftmaxBoundTensor {n : Nat} (η epsScores epsMax : ℝ)
    (scoresR : Tensor R (.dim n .scalar))
    (mask : Tensor Bool (.dim n .scalar)) (rowMaxR : R) :
    SpecTensor (.dim n .scalar) :=
  let numeratorsR := hardMaskedNumerators scoresR mask rowMaxR
  let denominatorR : R := sumSpec numeratorsR
  let denominatorRepR : Tensor R (.dim n .scalar) :=
    Tensor.replicate (Tensor.scalar denominatorR)
  NFBackend.divPosBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
    η
    (hardMaskedNumeratorError (β := β) (fexp := fexp) (rnd := rnd)
      epsScores epsMax scoresR mask rowMaxR)
    (hardMaskedDenominatorError (β := β) (fexp := fexp) (rnd := rnd)
      epsScores epsMax scoresR mask rowMaxR)
    numeratorsR denominatorRepR

/-- Numerical certificate for the nonempty branch of hard-masked vector softmax.

`hmaxS` and `hmaxR` identify the selected allowed-row maxima. The theorem does not trust those
values blindly: `hmax` must relate them numerically, and `hdenomLower` supplies the exact positive
lower bound used by division. For the canonical selected maximum this lower bound is `1`, because
one allowed shifted score is zero and contributes `exp 0 = 1`.
-/
theorem approxT_hardMaskedSoftmaxVecSpec_of_max {n : Nat}
    {scoresS : SpecTensor (.dim n .scalar)}
    {scoresR : Tensor R (.dim n .scalar)}
    (mask : Tensor Bool (.dim n .scalar))
    {rowMaxS : ℝ} {rowMaxR : R} {epsScores epsMax η : ℝ}
    (hscores : approxT (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      scoresS scoresR epsScores)
    (hmax : abs
      (NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd) rowMaxR - rowMaxS) ≤ epsMax)
    (hmaxS : Spec.hardMaskedMax? scoresS mask = some rowMaxS)
    (hmaxR : Spec.hardMaskedMax? scoresR mask = some rowMaxR)
    (hdenomLower : η ≤ sumSpec (hardMaskedNumerators scoresS mask rowMaxS))
    (hdenomMargin :
      hardMaskedDenominatorError (β := β) (fexp := fexp) (rnd := rnd)
        epsScores epsMax scoresR mask rowMaxR < η) :
    approxT (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Spec.hardMaskedSoftmaxVecSpec scoresS mask)
      (Spec.hardMaskedSoftmaxVecSpec scoresR mask)
      (linfNorm
        (hardMaskedSoftmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          η epsScores epsMax scoresR mask rowMaxR)) := by
  let maxS : SpecTensor .scalar := Tensor.scalar rowMaxS
  let maxR : Tensor R .scalar := Tensor.scalar rowMaxR
  let maxRepS : SpecTensor (.dim n .scalar) := Tensor.replicate maxS
  let maxRepR : Tensor R (.dim n .scalar) := Tensor.replicate maxR
  let shiftedS := subSpec scoresS maxRepS
  let shiftedR := subSpec scoresR maxRepR
  let expS := expSpec shiftedS
  let expR := expSpec shiftedR
  let numeratorsS := hardMaskedNumerators scoresS mask rowMaxS
  let numeratorsR := hardMaskedNumerators scoresR mask rowMaxR
  let denominatorS : ℝ := sumSpec numeratorsS
  let denominatorR : R := sumSpec numeratorsR
  let denominatorRepS : SpecTensor (.dim n .scalar) :=
    Tensor.replicate (Tensor.scalar denominatorS)
  let denominatorRepR : Tensor R (.dim n .scalar) :=
    Tensor.replicate (Tensor.scalar denominatorR)
  let epsShift := hardMaskedShiftError (β := β) (fexp := fexp) (rnd := rnd)
    epsScores epsMax scoresR rowMaxR
  let epsNumerator := hardMaskedNumeratorError (β := β) (fexp := fexp) (rnd := rnd)
    epsScores epsMax scoresR mask rowMaxR
  let epsDenominator := hardMaskedDenominatorError (β := β) (fexp := fexp) (rnd := rnd)
    epsScores epsMax scoresR mask rowMaxR

  have hmaxTensor : approxT (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      maxS maxR epsMax :=
    (approxT_scalar_iff (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))).2 hmax
  have hmaxRep := NFBackend.approxT_replicate
    (β := β) (fexp := fexp) (rnd := rnd) (s := .dim n .scalar) hmaxTensor
  have hshift := NFBackend.approxT_sub_spec
    (β := β) (fexp := fexp) (rnd := rnd) hscores hmaxRep
  have hexp := NFBackend.approxT_exp_spec
    (β := β) (fexp := fexp) (rnd := rnd) hshift
  have hnumerators := NFBackend.approxT_applyBoolMask
    (β := β) (fexp := fexp) (rnd := rnd) mask hexp
  have hsum := NFBackend.approxT_sum_spec
    (β := β) (fexp := fexp) (rnd := rnd) hnumerators
  have hdenominatorRep := NFBackend.approxT_replicate
    (β := β) (fexp := fexp) (rnd := rnd)
    (s := .dim n .scalar) hsum
  have hdenominatorDomain :
      Tensor.Forall (fun z : ℝ => η ≤ z) denominatorRepS := by
    exact Tensor.forall_replicate (by simpa [denominatorS, numeratorsS] using hdenomLower)
  have hout := NFBackend.approxT_div_spec_of_pos_lb
    (β := β) (fexp := fexp) (rnd := rnd) η
    hnumerators hdenominatorRep hdenominatorDomain hdenomMargin
  simp only [Spec.hardMaskedSoftmaxVecSpec, hmaxS, hmaxR]
  rw [← hardMaskedNumerators_eq_fused scoresS mask rowMaxS,
    ← hardMaskedNumerators_eq_fused scoresR mask rowMaxR]
  simpa [
    hardMaskedSoftmaxBoundTensor, hardMaskedNumerators,
    hardMaskedShiftError, hardMaskedNumeratorError, hardMaskedDenominatorError,
    maxS, maxR, maxRepS, maxRepR, shiftedS, shiftedR, expS, expR,
    numeratorsS, numeratorsR, denominatorS, denominatorR,
    denominatorRepS, denominatorRepR, epsShift, epsNumerator, epsDenominator] using hout

/-- If both semantics find no allowed key, hard-masked softmax agrees exactly on the zero row. -/
theorem approxT_hardMaskedSoftmaxVecSpec_allBlocked {n : Nat}
    (scoresS : SpecTensor (.dim n .scalar))
    (scoresR : Tensor R (.dim n .scalar))
    (mask : Tensor Bool (.dim n .scalar))
    (hmaxS : Spec.hardMaskedMax? scoresS mask = none)
    (hmaxR : Spec.hardMaskedMax? scoresR mask = none) :
    approxT (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Spec.hardMaskedSoftmaxVecSpec scoresS mask)
      (Spec.hardMaskedSoftmaxVecSpec scoresR mask) 0 := by
  simp only [Spec.hardMaskedSoftmaxVecSpec, hmaxS, hmaxR]
  change approxT (α := R)
    (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
    (Spec.fill (0 : ℝ) (.dim n .scalar)) (Spec.fill (0 : R) (.dim n .scalar)) 0
  exact NFBackend.approxT_fill_zero (β := β) (fexp := fexp) (rnd := rnd)
    (s := .dim n .scalar)

/-- Checkable evidence for nonempty hard-masked softmax rows.

The data fields record the maxima and margins used by execution; the proposition fields establish
that they describe the exact and rounded rows. Keeping this evidence together prevents a caller
from accidentally pairing a denominator check with a different score matrix or mask.
-/
structure HardMaskedRowsEvidence {m n : Nat}
    (scoresS : SpecTensor (.dim m (.dim n .scalar)))
    (scoresR : Tensor R (.dim m (.dim n .scalar)))
    (mask : Tensor Bool (.dim m (.dim n .scalar)))
    (epsScores : ℝ) where
  rowMaxS : Fin m → ℝ
  rowMaxR : Fin m → R
  epsMax : Fin m → ℝ
  eta : Fin m → ℝ
  maxApprox : ∀ i,
    abs (NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd) (rowMaxR i) -
      rowMaxS i) ≤ epsMax i
  specMax : ∀ i,
    Spec.hardMaskedMax? (Spec.get scoresS i) (Spec.get mask i) = some (rowMaxS i)
  runtimeMax : ∀ i,
    Spec.hardMaskedMax? (Spec.get scoresR i) (Spec.get mask i) = some (rowMaxR i)
  denominatorLower : ∀ i,
    eta i ≤ sumSpec
      (hardMaskedNumerators (Spec.get scoresS i) (Spec.get mask i) (rowMaxS i))
  denominatorMargin : ∀ i,
    hardMaskedDenominatorError (β := β) (fexp := fexp) (rnd := rnd)
      epsScores (epsMax i) (Spec.get scoresR i) (Spec.get mask i) (rowMaxR i) < eta i

/-- Rowwise hard-masked softmax bounds with one independently certified maximum per row. -/
def hardMaskedRowsBoundTensor {m n : Nat}
    (η : Fin m → ℝ) (epsScores : ℝ)
    (scoresR : Tensor R (.dim m (.dim n .scalar)))
    (mask : Tensor Bool (.dim m (.dim n .scalar)))
    (rowMaxR : Fin m → R) (epsMax : Fin m → ℝ) :
    SpecTensor (.dim m (.dim n .scalar)) :=
  match scoresR, mask with
  | Tensor.dim scoreRows, Tensor.dim maskRows =>
      Tensor.dim (fun i =>
        hardMaskedSoftmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (η i) epsScores (epsMax i) (scoreRows i) (maskRows i) (rowMaxR i))

/-- Matrix-level hard-masked softmax when every row has at least one allowed coordinate.

The selected maxima and denominator checks remain row-local. This matches causal attention, where
row `i` always admits key `i`, and avoids replacing all rows by the worst intermediate scale before
the final infinity norm is taken.
-/
theorem approxT_hardMaskedSoftmaxRowsSpec_of_max {m n : Nat}
    {scoresS : SpecTensor (.dim m (.dim n .scalar))}
    {scoresR : Tensor R (.dim m (.dim n .scalar))}
    (mask : Tensor Bool (.dim m (.dim n .scalar)))
    {epsScores : ℝ}
    (hscores : approxT (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      scoresS scoresR epsScores)
    (evidence : HardMaskedRowsEvidence (β := β) (fexp := fexp) (rnd := rnd)
      scoresS scoresR mask epsScores) :
    approxT (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Spec.hardMaskedSoftmaxSpec scoresS mask)
      (Spec.hardMaskedSoftmaxSpec scoresR mask)
      (linfNorm
        (hardMaskedRowsBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          evidence.eta epsScores scoresR mask evidence.rowMaxR evidence.epsMax)) := by
  cases scoresS with
  | dim scoreRowsS =>
      cases scoresR with
      | dim scoreRowsR =>
          cases mask with
          | dim maskRows =>
              let bound := linfNorm
                (hardMaskedRowsBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                  evidence.eta epsScores (Tensor.dim scoreRowsR) (Tensor.dim maskRows)
                  evidence.rowMaxR evidence.epsMax)
              have hbound : 0 ≤ bound := by
                simpa [bound] using
                  (linf_norm_nonneg
                    (t := hardMaskedRowsBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                      evidence.eta epsScores (Tensor.dim scoreRowsR) (Tensor.dim maskRows)
                      evidence.rowMaxR evidence.epsMax))
              refine approxT_dim_of_forall
                (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
                (xS := Spec.hardMaskedSoftmaxSpec (Tensor.dim scoreRowsS) (Tensor.dim maskRows))
                (xR := Spec.hardMaskedSoftmaxSpec (Tensor.dim scoreRowsR) (Tensor.dim maskRows))
                (eps := bound) hbound ?_
              intro i
              have hscoresI := approxT_dim_get (α := R)
                (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) hscores i
              have hrow := approxT_hardMaskedSoftmaxVecSpec_of_max
                (β := β) (fexp := fexp) (rnd := rnd)
                (maskRows i) hscoresI (evidence.maxApprox i)
                (by simpa [Spec.get, getAtSpec] using evidence.specMax i)
                (by simpa [Spec.get, getAtSpec] using evidence.runtimeMax i)
                (by simpa [Spec.get, getAtSpec] using evidence.denominatorLower i)
                (by simpa [Spec.get, getAtSpec] using evidence.denominatorMargin i)
              have hrowLe :
                  linfNorm
                      (hardMaskedSoftmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                        (evidence.eta i) epsScores (evidence.epsMax i)
                        (scoreRowsR i) (maskRows i) (evidence.rowMaxR i)) ≤
                    bound := by
                have h := linf_norm_le_get_dim
                  (t := hardMaskedRowsBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                    evidence.eta epsScores (Tensor.dim scoreRowsR) (Tensor.dim maskRows)
                    evidence.rowMaxR evidence.epsMax) i
                simpa [bound, hardMaskedRowsBoundTensor] using h
              have hrow' := approxT_mono hrow hrowLe
              simpa [Spec.hardMaskedSoftmaxSpec] using hrow'

/-! ## Rounded softmax backward -/

/-- Error in the `dY * softmax(x)` product used by the softmax VJP. -/
def vjpProductError {n : Nat} (epsDY epsY : ℝ)
    (dYR yR : Tensor R (.dim (Nat.succ n) .scalar)) : ℝ :=
  linfNorm (NFBackend.mulBoundTensor (β := β) (fexp := fexp) epsDY epsY dYR yR)

/-- Error in the rounded dot product `sum (dY * softmax(x))`. -/
def vjpDotError {n : Nat} (epsDY epsY : ℝ)
    (dYR yR : Tensor R (.dim (Nat.succ n) .scalar)) : ℝ :=
  let productR := mulSpec dYR yR
  NFBackend.sumBound (β := β) (fexp := fexp) (rnd := rnd)
    (vjpProductError (β := β) (fexp := fexp) epsDY epsY dYR yR) productR

/-- Error after subtracting the replicated rounded softmax dot product from `dY`. -/
def vjpCenteredError {n : Nat} (epsDY epsY : ℝ)
    (dYR yR : Tensor R (.dim (Nat.succ n) .scalar)) : ℝ :=
  let dotR : R := sumSpec (mulSpec dYR yR)
  let dotRepR : Tensor R (.dim (Nat.succ n) .scalar) := Tensor.replicate (Tensor.scalar dotR)
  linfNorm (NFBackend.subBoundTensor (β := β) (fexp := fexp)
    epsDY (vjpDotError (β := β) (fexp := fexp) (rnd := rnd) epsDY epsY dYR yR)
    dYR dotRepR)

/-- End-to-end infinity-norm budget for the rounded softmax VJP. -/
def softmaxVjpErrorBound {n : Nat} (epsDY epsY : ℝ)
    (dYR yR : Tensor R (.dim (Nat.succ n) .scalar)) : ℝ :=
  let dotR : R := sumSpec (mulSpec dYR yR)
  let centeredR := subSpec dYR
    (Tensor.replicate (Tensor.scalar dotR) : Tensor R (.dim (Nat.succ n) .scalar))
  linfNorm (NFBackend.mulBoundTensor (β := β) (fexp := fexp)
    epsY (vjpCenteredError (β := β) (fexp := fexp) (rnd := rnd) epsDY epsY dYR yR)
    yR centeredR)

/-- Rounded VJP theorem for any already-certified softmax weight vector.

This formulation is shared by ordinary and hard-masked softmax. In the masked case blocked weights
are exactly zero, so the common formula also returns exactly zero gradient at blocked logits; no
separate finite-sentinel derivative rule is required.
-/
theorem approxT_softmaxBackwardFromWeightsVecSpec {n : Nat}
    {yS dYS : SpecTensor (.dim (Nat.succ n) .scalar)}
    {yR dYR : Tensor R (.dim (Nat.succ n) .scalar)} {epsY epsDY : ℝ}
    (hy : approxT (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) yS yR epsY)
    (hdY : approxT (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) dYS dYR epsDY) :
    approxT (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Spec.softmaxBackwardFromWeightsSpec yS dYS)
      (Spec.softmaxBackwardFromWeightsSpec yR dYR)
      (softmaxVjpErrorBound (β := β) (fexp := fexp) (rnd := rnd)
        epsDY epsY dYR yR) := by
  let productS := mulSpec dYS yS
  let productR := mulSpec dYR yR
  let epsProduct := vjpProductError (β := β) (fexp := fexp) epsDY epsY dYR yR
  let dotS : ℝ := sumSpec productS
  let dotR : R := sumSpec productR
  let epsDot := vjpDotError (β := β) (fexp := fexp) (rnd := rnd) epsDY epsY dYR yR
  let dotRepS : SpecTensor (.dim (Nat.succ n) .scalar) := Tensor.replicate (Tensor.scalar dotS)
  let dotRepR : Tensor R (.dim (Nat.succ n) .scalar) := Tensor.replicate (Tensor.scalar dotR)
  let centeredS := subSpec dYS dotRepS
  let centeredR := subSpec dYR dotRepR
  let epsCentered :=
    vjpCenteredError (β := β) (fexp := fexp) (rnd := rnd) epsDY epsY dYR yR

  have hproduct : approxT (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      productS productR epsProduct := by
    have h := NFBackend.approxT_mul_spec (β := β) (fexp := fexp) (rnd := rnd) hdY hy
    simpa [productS, productR, epsProduct, vjpProductError] using h
  have hdot : approxT (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Tensor.scalar dotS) (Tensor.scalar dotR) epsDot := by
    have h := NFBackend.approxT_sum_spec (β := β) (fexp := fexp) (rnd := rnd) hproduct
    simpa [dotS, dotR, epsDot, vjpDotError, productR, epsProduct] using h
  have hdotRep : approxT (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      dotRepS dotRepR epsDot := by
    simpa [dotRepS, dotRepR] using
      (NFBackend.approxT_replicate (β := β) (fexp := fexp) (rnd := rnd)
        (s := .dim (Nat.succ n) .scalar) hdot)
  have hcentered : approxT (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      centeredS centeredR epsCentered := by
    have h := NFBackend.approxT_sub_spec (β := β) (fexp := fexp) (rnd := rnd) hdY hdotRep
    simpa [centeredS, centeredR, epsCentered, vjpCenteredError, dotRepR, dotR, productR,
      epsDot] using h
  have hout := NFBackend.approxT_mul_spec (β := β) (fexp := fexp) (rnd := rnd) hy hcentered
  simpa [Spec.softmaxBackwardFromWeightsSpec, productS, productR, dotS, dotR,
    dotRepS, dotRepR, centeredS, centeredR, epsCentered, softmaxVjpErrorBound] using hout

/-- Forward-error theorem for the executable softmax VJP.

This is the training counterpart of `approxT_softmaxVecSpec`. It follows the implementation's
factorization `y * (dY - sum (dY * y))`; the proof never materializes a dense Jacobian and reuses
the same rounded multiplication, reduction, replication, and subtraction contracts as ordinary
model execution.
-/
theorem approxT_softmaxBackwardVecSpec {n : Nat}
    {xS dYS : SpecTensor (.dim (Nat.succ n) .scalar)}
    {xR dYR : Tensor R (.dim (Nat.succ n) .scalar)} {epsX epsDY : ℝ}
    (hx : approxT (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR epsX)
    (hdY : approxT (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) dYS dYR epsDY)
    (hdenom : denominatorErrorBound (β := β) (fexp := fexp) (rnd := rnd) epsX xR < 1) :
    let yR := Activation.softmaxVecSpec xR
    approxT (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Activation.softmaxBackwardSpec xS dYS) (Activation.softmaxBackwardSpec xR dYR)
      (softmaxVjpErrorBound (β := β) (fexp := fexp) (rnd := rnd)
        epsDY (softmaxErrorBound (β := β) (fexp := fexp) (rnd := rnd) epsX xR) dYR yR) := by
  dsimp only
  let yS := Activation.softmaxVecSpec xS
  let yR := Activation.softmaxVecSpec xR
  let epsY := softmaxErrorBound (β := β) (fexp := fexp) (rnd := rnd) epsX xR
  have hy : approxT (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) yS yR epsY := by
    simpa [yS, yR, epsY] using
      (approxT_softmaxVecSpec (β := β) (fexp := fexp) (rnd := rnd) hx hdenom)
  have hout := approxT_softmaxBackwardFromWeightsVecSpec
    (β := β) (fexp := fexp) (rnd := rnd) hy hdY
  simpa [Activation.softmaxBackwardSpec, Spec.softmaxBackwardFromWeightsSpec, yS, yR, epsY]
    using hout

/-- The analytic softmax on a nonempty vector sums to one. -/
theorem sum_softmaxVec {n : Nat} (x : Vec (Nat.succ n)) :
    (∑ i, softmaxVec x i) = 1 := by
  classical
  have hpos : 0 < sumExp x := by
    simpa [sumExp] using
      Finset.sum_pos (fun i (_ : i ∈ (Finset.univ : Finset (Fin (Nat.succ n)))) =>
        Real.exp_pos (x i)) Finset.univ_nonempty
  have hne : sumExp x ≠ 0 := ne_of_gt hpos
  have hne' : (∑ i, Real.exp (x i)) ≠ 0 := by
    simpa [sumExp] using hne
  simp only [softmaxVec, softmaxVecOfFun_apply]
  calc
    (∑ i, Real.exp (x i) / sumExp x) = (∑ i, Real.exp (x i)) / sumExp x := by
      simpa using
        (Finset.sum_div (s := (Finset.univ : Finset (Fin (Nat.succ n))))
          (f := fun i => Real.exp (x i)) (a := sumExp x)).symm
    _ = 1 := div_self hne'

/-- A softmax JVP is tangent to the probability simplex: its coordinates sum to zero. -/
theorem sum_softmaxJvp {n : Nat} (x dx : Vec (Nat.succ n)) :
    (∑ i, softmaxJvp x dx i) = 0 := by
  classical
  let y : Vec (Nat.succ n) := softmaxVec x
  let s : Real := dotCLM y dx
  have hy : (∑ i, y i) = 1 := sum_softmaxVec x
  have hs : s = ∑ i, y i * dx i := by
    simp [s, dotCLM_apply]
  calc
    (∑ i, softmaxJvp x dx i) = ∑ i, y i * (dx i - s) := by
      simp [softmaxJvp, y, s]
    _ = (∑ i, y i * dx i) - s * (∑ i, y i) := by
      calc
        (∑ i, y i * (dx i - s)) = ∑ i, (y i * dx i - s * y i) := by
          refine Finset.sum_congr rfl ?_
          intro i _
          ring
        _ = (∑ i, y i * dx i) - ∑ i, s * y i := by
          rw [Finset.sum_sub_distrib]
        _ = (∑ i, y i * dx i) - s * (∑ i, y i) := by
          rw [Finset.mul_sum]
    _ = 0 := by rw [hy, hs]; ring

/-- Coordinatewise VJP/JVP bound in the infinity norm.

If every upstream coordinate has magnitude at most `G`, then every softmax input gradient has
magnitude at most `2G`. The estimate is dimension-free because softmax weights are nonnegative and
sum to one. It is intentionally conservative; tighter certificates may retain the factor
`2 * y_i * (1 - y_i)` for each coordinate.
-/
theorem abs_softmaxJvp_le_two_mul {n : Nat} (x dx : Vec (Nat.succ n)) (G : Real)
    (hdx : ∀ i, abs (dx i) <= G) (i : Fin (Nat.succ n)) :
    abs (softmaxJvp x dx i) <= 2 * G := by
  classical
  let y : Vec (Nat.succ n) := softmaxVec x
  let s : Real := dotCLM y dx
  have hyPos : ∀ j, 0 < y j := by
    intro j
    simp only [y, softmaxVec, softmaxVecOfFun_apply]
    exact div_pos (Real.exp_pos (x j)) <| by
      simpa [sumExp] using
        Finset.sum_pos (fun k (_ : k ∈ (Finset.univ : Finset (Fin (Nat.succ n)))) =>
          Real.exp_pos (x k)) Finset.univ_nonempty
  have hySum : (∑ j, y j) = 1 := sum_softmaxVec x
  have hs : s = ∑ j, y j * dx j := by
    simp [s, dotCLM_apply]
  have hsAbs : abs s <= G := by
    rw [hs]
    calc
      abs (∑ j, y j * dx j) <= ∑ j, abs (y j * dx j) :=
        Finset.abs_sum_le_sum_abs _ _
      _ = ∑ j, y j * abs (dx j) := by
        refine Finset.sum_congr rfl ?_
        intro j _
        rw [abs_mul, abs_of_pos (hyPos j)]
      _ <= ∑ j, y j * G := by
        refine Finset.sum_le_sum ?_
        intro j _
        exact mul_le_mul_of_nonneg_left (hdx j) (le_of_lt (hyPos j))
      _ = G := by rw [← Finset.sum_mul, hySum, one_mul]
  have hyLeOne : y i <= 1 := by
    calc
      y i <= ∑ j, y j :=
        Finset.single_le_sum (fun j _ => le_of_lt (hyPos j)) (Finset.mem_univ i)
      _ = 1 := hySum
  have hdiff : abs (dx i - s) <= 2 * G := by
    calc
      abs (dx i - s) <= abs (dx i) + abs s := abs_sub _ _
      _ <= G + G := add_le_add (hdx i) hsAbs
      _ = 2 * G := by ring
  simp only [softmaxJvp, softmaxVecOfFun_apply]
  change abs (y i * (dx i - s)) <= 2 * G
  rw [abs_mul, abs_of_pos (hyPos i)]
  calc
    y i * abs (dx i - s) <= 1 * abs (dx i - s) :=
      mul_le_mul_of_nonneg_right hyLeOne (abs_nonneg _)
    _ <= 1 * (2 * G) := mul_le_mul_of_nonneg_left hdiff zero_le_one
    _ = 2 * G := one_mul _

end

end AxisSoftmax
end RuntimeApprox
end Proofs
