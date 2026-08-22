/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.Ops
public import NN.Proofs.RuntimeApprox.NF.FoldLemmas
public import NN.Proofs.RuntimeApprox.NF.Ops
public import NN.Spec.Core.TensorReductionShape

/-!
# NF Shape Operators

NF (rounded) backend: approximation lemmas for shape-only tensor operators.

These operators do not perform arithmetic on scalars (they only permute/replicate entries), so
they preserve existing `approxTensor` error bounds.

Shape-only operations should not introduce extra rounding error. Their proofs
are mostly transport/indexing arguments rather than numerical analysis.

## PyTorch correspondence / citations
These are the proof analogues of “view-like”/index-rearrangement ops in PyTorch which do not change
floating-point values, only their arrangement:
https://pytorch.org/docs/stable/generated/torch.reshape.html
https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
https://pytorch.org/docs/stable/generated/torch.permute.html
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

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]
variable {rnd : ℝ → ℤ} [NeuralValidRndToNearest rnd]

local notation "R" => TorchLean.Floats.NF β fexp rnd

omit [NeuralValidExp fexp] [NeuralValidRndToNearest rnd] in
/-- Filling a tensor preserves a scalar approximation budget at every shape.

Although this fact is used heavily when constructing reverse-mode zero contexts, it is a shape
fact rather than a backward-mode fact. Keeping it here also makes rounded constants available to
normalization, attention, and quantization without importing the reverse-mode implementation.
-/
lemma approxTensor_fill_const {cS : ℝ} {cR : R} {eps : ℝ}
    (h : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) cR - cS) ≤ eps) :
    ∀ {s : Shape},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.fill cS s) (Spec.fill cR s) eps := by
  intro s
  induction s with
  | scalar =>
      exact (approxTensor_scalar_iff (α := R)
        (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))).2 h
  | dim n inner ih =>
      have hε : 0 ≤ eps := le_trans (abs_nonneg _) h
      refine approxTensor_dim_of_forall
        (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (xS := Spec.fill cS (.dim n inner)) (xR := Spec.fill cR (.dim n inner))
        (eps := eps) hε ?_
      intro i
      simpa [Spec.fill] using ih

private lemma toSpec_one_bound :
    abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (1 : R) - (1 : ℝ)) ≤
      neuralUlp β fexp (1 : ℝ) / 2 := by
  convert
    (Proofs.RuntimeRoundingApprox.roundR_abs_error
      (β := β) (fexp := fexp) (rnd := rnd) (1 : ℝ)) using 1
  · simp [NFBackend.toSpec, TorchLean.Floats.NF.toReal,
      Proofs.RuntimeRoundingApprox.roundR]
    exact congrArg (fun x => abs (x - (1 : ℝ)))
      (show (1 : R).val = neuralRound (β := β) (fexp := fexp) rnd 1 from rfl)

/-- A tensor filled with runtime one differs from exact one by at most one construction rounding. -/
lemma approxTensor_fill_one :
    ∀ {s : Shape},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.fill (1 : ℝ) s) (Spec.fill (1 : R) s) (neuralUlp β fexp (1 : ℝ) / 2) := by
  intro s
  apply approxTensor_fill_const (β := β) (fexp := fexp) (rnd := rnd)
  exact toSpec_one_bound (β := β) (fexp := fexp) (rnd := rnd)

/-- Zero is exactly representable in every valid neural floating-point format. -/
lemma approxTensor_fill_zero :
    ∀ {s : Shape},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.fill (0 : ℝ) s) (Spec.fill (0 : R) s) 0 := by
  intro s
  apply approxTensor_fill_const (β := β) (fexp := fexp) (rnd := rnd)
  simp

omit [NeuralValidExp fexp] [NeuralValidRndToNearest rnd] in
theorem approxTensor_replicate {s : Shape}
    {xS : SpecTensor .scalar} {xR : Tensor R .scalar} {eps : ℝ}
    (hx : approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps) :
    approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Spec.Tensor.replicate (α := SpecScalar) (s := s) xS)
      (Spec.Tensor.replicate (α := R) (s := s) xR)
      eps := by
  classical
  cases xS with
  | scalar x =>
      cases xR with
      | scalar xR =>
          have hscalar :=
            (approxTensor_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
              (x := x) (xR := xR) (eps := eps)).1 hx
          have hε : 0 ≤ eps := le_trans (abs_nonneg _) hscalar
          induction s with
          | scalar =>
              simpa [Spec.Tensor.replicate] using hx
          | dim n inner ih =>
              refine approxTensor_dim_of_forall
                (n := n) (s := inner) (xS := Spec.Tensor.replicate (α := SpecScalar) (s := .dim n
                  inner) (.scalar x))
                (xR := Spec.Tensor.replicate (α := R) (s := .dim n inner) (.scalar xR))
                (eps := eps) hε ?_
              intro i
              simpa [Spec.Tensor.replicate] using ih

omit [NeuralValidRndToNearest rnd] in
theorem approxTensor_broadcastTo
    {s₁ s₂ : Shape} (cb : Shape.CanBroadcastTo s₁ s₂)
    {xS : SpecTensor s₁} {xR : Tensor R s₁} {eps : ℝ}
    (hx : approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps) :
    approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Spec.Tensor.broadcastTo (α := SpecScalar) (s₁ := s₁) (s₂ := s₂) cb xS)
      (Spec.Tensor.broadcastTo (α := R) (s₁ := s₁) (s₂ := s₂) cb xR)
      eps := by
  classical
  have hε : 0 ≤ eps := approxTensor_eps_nonneg (s := s₁) hx
  induction cb with
  | scalar =>
      cases xS with
      | scalar _ =>
          cases xR with
          | scalar _ =>
              simpa [Spec.Tensor.broadcastTo] using hx
  | dim_eq tail ih =>
      cases xS with
      | dim fS =>
          cases xR with
          | dim fR =>
              refine approxTensor_dim_of_forall
                (n := _) (s := _)
                (xS := Spec.Tensor.broadcastTo (α := SpecScalar) (Shape.CanBroadcastTo.dim_eq tail)
                  (Tensor.dim fS))
                (xR := Spec.Tensor.broadcastTo (α := R) (Shape.CanBroadcastTo.dim_eq tail)
                  (Tensor.dim fR))
                (eps := eps) hε ?_
              intro i
              have hx_i :=
                approxTensor_dim_get (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hx i
              simpa [Spec.Tensor.broadcastTo] using ih (xS := fS i) (xR := fR i) hx_i
  | dim_1_to_n tail ih =>
      cases xS with
      | dim fS =>
          cases xR with
          | dim fR =>
              refine approxTensor_dim_of_forall
                (n := _) (s := _)
                (xS := Spec.Tensor.broadcastTo (α := SpecScalar) (Shape.CanBroadcastTo.dim_1_to_n
                  tail) (Tensor.dim fS))
                (xR := Spec.Tensor.broadcastTo (α := R) (Shape.CanBroadcastTo.dim_1_to_n tail)
                  (Tensor.dim fR))
                (eps := eps) hε ?_
              intro i
              have hx0 :=
                approxTensor_dim_get (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hx (0 : Fin
                  1)
              simpa [Spec.Tensor.broadcastTo] using ih (xS := fS 0) (xR := fR 0) hx0
  | expand_dims tail ih =>
      refine approxTensor_dim_of_forall
        (n := _) (s := _)
        (xS := Spec.Tensor.broadcastTo (α := SpecScalar) (Shape.CanBroadcastTo.expand_dims tail) xS)
        (xR := Spec.Tensor.broadcastTo (α := R) (Shape.CanBroadcastTo.expand_dims tail) xR)
        (eps := eps) hε ?_
      intro i
      simpa [Spec.Tensor.broadcastTo] using ih (xS := xS) (xR := xR) hx

/-- Applying the same Boolean mask to exact and rounded tensors preserves the error budget.

Masking is a selection operation, not arithmetic: allowed entries are unchanged and blocked
entries are exactly zero in both semantics. In particular, this theorem does not model a finite
negative sentinel and introduces no extra ULP term.
-/
theorem approxTensor_applyBoolMask {s : Shape}
    {xS : SpecTensor s} {xR : Tensor R s} (mask : Tensor Bool s) {eps : ℝ}
    (hx : approxTensor (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps) :
    approxTensor (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (map2Spec (fun x allowed => if allowed then x else 0) xS mask)
      (map2Spec (fun x allowed => if allowed then x else 0) xR mask) eps := by
  induction s with
  | scalar =>
      cases xS with
      | scalar x =>
          cases xR with
          | scalar xR =>
              cases mask with
              | scalar allowed =>
                  cases allowed with
                  | false =>
                      apply (approxTensor_scalar_iff (α := R)
                        (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))).2
                      simpa using approxTensor_eps_nonneg hx
                  | true => simpa [map2Spec] using hx
  | dim n inner ih =>
      cases xS with
      | dim valuesS =>
          cases xR with
          | dim valuesR =>
              cases mask with
              | dim masks =>
                  have hε := approxTensor_eps_nonneg hx
                  refine approxTensor_dim_of_forall
                    (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                    (xS := map2Spec (fun x allowed => if allowed then x else 0)
                      (Tensor.dim valuesS) (Tensor.dim masks))
                    (xR := map2Spec (fun x allowed => if allowed then x else 0)
                      (Tensor.dim valuesR) (Tensor.dim masks))
                    (eps := eps) hε ?_
                  intro i
                  exact ih (masks i) (approxTensor_dim_get (α := R)
                    (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hx i)

end NFBackend

end
end RuntimeApprox
end Proofs
