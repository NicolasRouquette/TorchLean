/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.Ops.Elementwise.SafeDivSigmoid

/-!
# NF Elementwise Bounds: Scalar Logistic Node
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

/--
Scalar forward bound for the scalar logistic-form NF `softmax` node.

Here the node computes the scalar logistic-like function
$\exp(x)/(\exp(x)+1)$, implemented using `exp`, `+`, and division (the denominator is $\ge 1$).
We keep the public node name stable for the NF graph, but the mathematical function is
`Activation.Math.logisticSpec`, not axis softmax.
-/
def softmaxBoundScalar (eps : ℝ) (xR : R) : ℝ :=
  let numR : R := MathFunctions.exp xR
  let denomR : R := numR + (1 : R)
  let numHat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) numR
  let denomHat : ℝ := toSpec (β := β) (fexp := fexp) (rnd := rnd) denomR
  let qhat : ℝ := numHat / denomHat
  let epsNum : ℝ :=
    expErrorBound (β := β) (fexp := fexp)
      (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) eps
  neuralUlp β fexp qhat / 2 +
    abs numHat * abs (1 / denomHat) + abs numHat + epsNum

/-- Per-entry bound tensor for the scalar logistic NF node. -/
def softmaxBoundTensor {s : Shape} (eps : ℝ) (xR : Tensor R s) : SpecTensor s :=
  Spec.Tensor.map (softmaxBoundScalar (β := β) (fexp := fexp) (rnd := rnd) eps) xR

/--
`approxTensor` bound for the scalar logistic NF node lifted to arbitrary tensor shapes.

This is the tensor-level wrapper around the scalar `exp`/`+`/`div` bound, using the usual
  `linf_norm`
lifting for dimensioned tensors.
-/
theorem approxTensor_softmax_spec {s : Shape} :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) xS)
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) xR)
          (linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR)) :=
            by
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
              let numR : R := MathFunctions.exp xR
              let denomR : R := numR + (1 : R)
              let y : ℝ := Real.exp x + 1
              have hy : (1 : ℝ) ≤ y := by
                have : 0 ≤ Real.exp x := by simpa using (Real.exp_nonneg x)
                linarith
              have hnum :=
                approx_exp_nf (β := β) (fexp := fexp) (rnd := rnd) (x := x) (xR := xR) (eps := eps)
                  hx'
              -- Apply the division bound with numerator = exp and denominator = exp + 1.
              have hdiv :=
                approx_div_nf_of_one_le (β := β) (fexp := fexp) (rnd := rnd)
                  (x := Real.exp x) (y := y) (xR := numR) (yR := denomR)
                  (epsx := expErrorBound (β := β) (fexp := fexp)
                    (toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) eps)
                  hy hnum
              have hb :
                  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                          (Activation.Math.logisticSpec (α := R) xR) -
                        Activation.Math.logisticSpec (α := ℝ) x) ≤
                    softmaxBoundScalar (β := β) (fexp := fexp) (rnd := rnd) eps xR := by
                simpa [Activation.Math.logisticSpec, softmaxBoundScalar, numR, denomR, y,
                  MathFunctions.exp] using
                  hdiv
              have hle :
                  abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                          (Activation.Math.logisticSpec (α := R) xR) -
                        Activation.Math.logisticSpec (α := ℝ) x) ≤
                    linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                      (s := Shape.scalar) eps (Tensor.scalar xR)) := by
                refine le_trans hb ?_
                simpa [softmaxBoundTensor, Spec.Tensor.map, linfNorm, RuntimeApprox.linfNorm,
                  tensorLinfNorm, MathFunctions.abs] using
                  (le_abs_self (softmaxBoundScalar (β := β) (fexp := fexp) (rnd := rnd) eps xR))
              exact
                (approxTensor_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                  (x := Activation.Math.logisticSpec (α := ℝ) x)
                  (xR := Activation.Math.logisticSpec (α := R) xR)
                  (eps := linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                    (s := Shape.scalar) eps (Tensor.scalar xR)))).2 (by
                      simpa using hle)
  | dim n s ih =>
      cases xS with
      | dim xSf =>
          cases xR with
          | dim xRf =>
              let B : ℝ :=
                linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                  (s := Shape.dim n s) eps (Tensor.dim xRf))
              have hB_nonneg : 0 ≤ B := by
                simpa [B] using (linf_norm_nonneg
                  (t := softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                    (s := Shape.dim n s) eps (Tensor.dim xRf)))
              have hcomp :
                  ∀ i : Fin n,
                    tensorDistance (α := SpecScalar) linfNorm
                        (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (xSf i))
                        (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                          rnd))
                          (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (xRf i)))
                      ≤ B := by
                intro i
                have hx_i :=
                  approxTensor_dim_get (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                    (xS := Tensor.dim xSf) (xR := Tensor.dim xRf) (eps := eps) hx i
                have hih := ih (xS := xSf i) (xR := xRf i) hx_i
                have hB_ge :
                    linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                        (s := s) eps (xRf i)) ≤ B := by
                  simpa [B, softmaxBoundTensor, Spec.Tensor.map] using
                    (linf_norm_le_get_dim
                      (t := softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                        (s := Shape.dim n s) eps (Tensor.dim xRf)) i)
                have hdist :
                    tensorDistance (α := SpecScalar) linfNorm
                        (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (xSf i))
                        (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                          rnd))
                          (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (xRf i)))
                      ≤ linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                        (s := s) eps (xRf i)) := by
                  simpa [approxTensor, approxWith] using hih
                exact le_trans hdist hB_ge
              have hf :
                  ∀ i ∈ List.finRange n,
                    tensorDistance (α := SpecScalar) linfNorm
                        (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (xSf i))
                        (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                          rnd))
                          (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (xRf i)))
                      ≤ B := by
                intro i _hi
                exact hcomp i
              have hfold :=
                List.foldl_max_le_of_le (List.finRange n)
                  (fun i =>
                    tensorDistance (α := SpecScalar) linfNorm
                      (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (xSf i))
                      (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                        rnd))
                        (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (xRf i))))
                  (acc := (0 : ℝ)) (eps := B) hB_nonneg hf
              have :
                  tensorDistance (α := SpecScalar) linfNorm
                      (mapSpec (s := Shape.dim n s) (Activation.Math.logisticSpec (α := ℝ))
                        (Tensor.dim xSf))
                      (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
                        rnd))
                        (mapSpec (s := Shape.dim n s) (Activation.Math.logisticSpec (α := R))
                          (Tensor.dim xRf)))
                    ≤ B := by
                change
                  List.foldl
                    (fun a i =>
                      max a
                        (tensorDistance (α := SpecScalar) linfNorm
                          (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (xSf i))
                          (tensorToSpec (α := R)
                            (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                            (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (xRf i)))))
                    0 (List.finRange n) ≤ B
                exact hfold
              simpa [approxTensor, approxWith, B] using this
end NFBackend

end

end RuntimeApprox
end Proofs
