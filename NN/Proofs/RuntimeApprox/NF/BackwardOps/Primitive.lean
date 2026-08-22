/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.BackwardOps.Backend

/-!
# Primitive NF Reverse Nodes

Reverse-mode approximation nodes for scalar, elementwise, reduction, and activation operations.
Each node packages the forward operation, the spec VJP, the rounded runtime VJP, and the local error
bound proof used by graph-level backpropagation.
-/
@[expose] public section

namespace Proofs
namespace RuntimeApprox

open Spec
open Tensor
open NN.MLTheory.Robustness.Spec
open Proofs.Autograd.Algebra

noncomputable section

namespace NFBackend

open TorchLean.Floats
open Proofs.RuntimeRoundingApprox

variable {β : NeuralRadix} {fexp : ℤ → ℤ} [NeuralValidExp fexp]
variable {rnd : ℝ → ℤ} [NeuralValidRndToNearest rnd]

local notation "R" => TorchLean.Floats.NF β fexp rnd

-- ---------------------------------------------------------------------------
-- Reverse nodes (RevNode constructors)
-- ---------------------------------------------------------------------------

/--
Reverse node for addition: `z = a + b`.

VJP is `(δ ↦ (δ, δ))`, with a special-case when `a` and `b` are the same context index.
-/
def addRevNode {Γ : List Shape} {s : Shape} (a b : Idx Γ s) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { toFwdNode := addNode (β := β) (fexp := fexp) (rnd := rnd) a b
      vjpSpec := fun _ctx δ =>
        if h : a.i = b.i then
          SparseContext.setIdx (α := SpecScalar) (Γ := Γ) (s := s) a (addSpec δ δ)
        else
          SparseContext.set2Idx (α := SpecScalar) (Γ := Γ) (s₁ := s) (s₂ := s) a δ b δ
      vjpRuntime := fun _ctx δ =>
        if h : a.i = b.i then
          SparseContext.setIdx (α := R) (Γ := Γ) (s := s) a (addSpec δ δ)
        else
          SparseContext.set2Idx (α := R) (Γ := Γ) (s₁ := s) (s₂ := s) a δ b δ
      vjpBound := fun _epsCtx _ctxR epsδ δR =>
        if h : a.i = b.i then
          let epsBoth :=
            linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsδ epsδ δR
              δR)
          EList.setIdx (Γ := Γ) (s := s) a epsBoth
        else
          EList.set2Idx (Γ := Γ) (s₁ := s) (s₂ := s) a epsδ b epsδ 0
      vjpSound := ?_ }
  intro _ctxS _ctxR _epsCtx δS δR epsδ _hctx hδ
  classical
  by_cases hEq : a.i = b.i
  · have hsum :
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (addSpec δS δS) (addSpec δR δR)
          (linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsδ epsδ δR
            δR)) := by
      simpa using
        (approxTensor_add_spec (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (xS := δS) (yS := δS) (xR := δR) (yR := δR)
          (epsx := epsδ) (epsy := epsδ) hδ hδ)
    have hctx' :=
      approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
        (Γ := Γ) (s := s) a
        (tS := addSpec δS δS) (tR := addSpec δR δR)
        (eps := linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsδ epsδ
          δR δR)) hsum
    simpa [hEq] using hctx'
  · have hctx' :=
      approxCtx_set2Idx_ne (β := β) (fexp := fexp) (rnd := rnd)
        (Γ := Γ) (s₁ := s) (s₂ := s) a b
        (t₁S := δS) (t₁R := δR) (eps₁ := epsδ)
        (t₂S := δS) (t₂R := δR) (eps₂ := epsδ)
        hδ hδ hEq
    simpa [hEq] using hctx'

/--
Reverse node for subtraction: `z = a - b`.

VJP is `(δ ↦ (δ, -δ))`, with a special-case when `a` and `b` are the same context index.
-/
def subRevNode {Γ : List Shape} {s : Shape} (a b : Idx Γ s) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { toFwdNode := subNode (β := β) (fexp := fexp) (rnd := rnd) a b
      vjpSpec := fun _ctx δ =>
        if h : a.i = b.i then
          SparseContext.setIdx (α := SpecScalar) (Γ := Γ) (s := s) a (Spec.fill (0 : SpecScalar) s)
        else
          SparseContext.set2Idx (α := SpecScalar) (Γ := Γ) (s₁ := s) (s₂ := s) a δ b (negSpec δ)
      vjpRuntime := fun _ctx δ =>
        if h : a.i = b.i then
          SparseContext.setIdx (α := R) (Γ := Γ) (s := s) a (Spec.fill (0 : R) s)
        else
          SparseContext.set2Idx (α := R) (Γ := Γ) (s₁ := s) (s₂ := s) a δ b (negSpec δ)
      vjpBound := fun _epsCtx _ctxR epsδ δR =>
        if h : a.i = b.i then
          EList.setIdx (Γ := Γ) (s := s) a 0
        else
          let epsNeg :=
            linfNorm (negBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsδ δR)
          EList.set2Idx (Γ := Γ) (s₁ := s) (s₂ := s) a epsδ b epsNeg 0
      vjpSound := ?_ }
  intro _ctxS _ctxR _epsCtx δS δR epsδ _hctx hδ
  classical
  by_cases hEq : a.i = b.i
  · have h0 :=
      approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
        (Γ := Γ) (s := s) a
        (tS := Spec.fill (0 : SpecScalar) s) (tR := Spec.fill (0 : R) s)
        (eps := 0) (approxTensor_fill_zero (β := β) (fexp := fexp) (rnd := rnd) (s := s))
    simpa [hEq] using h0
  · have hneg :
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (negSpec δS) (negSpec δR)
          (linfNorm (negBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsδ δR)) := by
      simpa using
        (approxTensor_neg_spec (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (xS := δS) (xR := δR) (eps := epsδ) hδ)
    have hctx' :=
      approxCtx_set2Idx_ne (β := β) (fexp := fexp) (rnd := rnd)
        (Γ := Γ) (s₁ := s) (s₂ := s) a b
        (t₁S := δS) (t₁R := δR) (eps₁ := epsδ)
        (t₂S := negSpec δS) (t₂R := negSpec δR)
        (eps₂ := linfNorm (negBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsδ δR))
        hδ hneg hEq
    simpa [hEq] using hctx'

/--
Reverse node for multiplication: `z = a * b`.

VJP is `(δ ↦ (δ*b, δ*a))`, with rounding-aware bounds produced by the NF backend.
-/
def mulRevNode {Γ : List Shape} {s : Shape} (a b : Idx Γ s) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { toFwdNode := mulNode (β := β) (fexp := fexp) (rnd := rnd) a b
      vjpSpec := fun ctx δ =>
        if h : a.i = b.i then
          let x := getIdx (α := SpecScalar) ctx a
          let u := mulSpec δ x
          SparseContext.setIdx (α := SpecScalar) (Γ := Γ) (s := s) a (addSpec u u)
        else
          let xa := getIdx (α := SpecScalar) ctx a
          let xb := getIdx (α := SpecScalar) ctx b
          SparseContext.set2Idx (α := SpecScalar) (Γ := Γ) (s₁ := s) (s₂ := s) a (mulSpec δ xb) b (mulSpec
            δ xa)
      vjpRuntime := fun ctx δ =>
        if h : a.i = b.i then
          let x := getIdx (α := R) ctx a
          let u := mulSpec δ x
          SparseContext.setIdx (α := R) (Γ := Γ) (s := s) a (addSpec u u)
        else
          let xa := getIdx (α := R) ctx a
          let xb := getIdx (α := R) ctx b
          SparseContext.set2Idx (α := R) (Γ := Γ) (s₁ := s) (s₂ := s) a (mulSpec δ xb) b (mulSpec δ xa)
      vjpBound := fun epsCtx ctxR epsδ δR =>
        if h : a.i = b.i then
          let xR := getIdx (α := R) ctxR a
          let uR := mulSpec δR xR
          let epsU :=
            linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) epsδ (getIdxEps (Γ := Γ) (s := s) epsCtx a) δR xR)
          let epsBoth :=
            linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsU epsU uR
              uR)
          EList.setIdx (Γ := Γ) (s := s) a epsBoth
        else
          let xaR := getIdx (α := R) ctxR a
          let xbR := getIdx (α := R) ctxR b
          let epsA :=
            linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) epsδ (getIdxEps (Γ := Γ) (s := s) epsCtx b) δR xbR)
          let epsB :=
            linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) epsδ (getIdxEps (Γ := Γ) (s := s) epsCtx a) δR xaR)
          EList.set2Idx (Γ := Γ) (s₁ := s) (s₂ := s) a epsA b epsB 0
      vjpSound := ?_ }
  intro ctxS ctxR epsCtx δS δR epsδ hctx hδ
  classical
  have ha :=
    approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx a
  have hb :=
    approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx b
  by_cases hEq : a.i = b.i
  · -- x*x case: contributions add
    have hu :
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (mulSpec δS (getIdx (α := SpecScalar) ctxS a))
          (mulSpec δR (getIdx (α := R) ctxR a))
          (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsδ (getIdxEps (Γ := Γ) (s := s) epsCtx a) δR (getIdx (α := R) ctxR a))) := by
      simpa using
        (approxTensor_mul_spec (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (xS := δS) (yS := getIdx (α := SpecScalar) ctxS a)
          (xR := δR) (yR := getIdx (α := R) ctxR a)
          (epsx := epsδ) (epsy := getIdxEps (Γ := Γ) (s := s) epsCtx a) hδ ha)
    have hsum :
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (addSpec (mulSpec δS (getIdx (α := SpecScalar) ctxS a)) (mulSpec δS (getIdx (α :=
            SpecScalar) ctxS a)))
          (addSpec (mulSpec δR (getIdx (α := R) ctxR a)) (mulSpec δR (getIdx (α := R) ctxR a)))
          (linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s)
            (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) epsδ (getIdxEps (Γ := Γ) (s := s) epsCtx a) δR (getIdx (α := R) ctxR a)))
            (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) epsδ (getIdxEps (Γ := Γ) (s := s) epsCtx a) δR (getIdx (α := R) ctxR a)))
            (mulSpec δR (getIdx (α := R) ctxR a)) (mulSpec δR (getIdx (α := R) ctxR a)))) := by
      simpa using
        (approxTensor_add_spec (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (xS := mulSpec δS (getIdx (α := SpecScalar) ctxS a))
          (yS := mulSpec δS (getIdx (α := SpecScalar) ctxS a))
          (xR := mulSpec δR (getIdx (α := R) ctxR a))
          (yR := mulSpec δR (getIdx (α := R) ctxR a))
          (epsx := linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsδ (getIdxEps (Γ := Γ) (s := s) epsCtx a) δR (getIdx (α := R) ctxR a)))
          (epsy := linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsδ (getIdxEps (Γ := Γ) (s := s) epsCtx a) δR (getIdx (α := R) ctxR a)))
          hu hu)
    have hctx' :=
      approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
        (Γ := Γ) (s := s) a
        (tS := addSpec (mulSpec δS (getIdx (α := SpecScalar) ctxS a)) (mulSpec δS (getIdx (α :=
          SpecScalar) ctxS a)))
        (tR := addSpec (mulSpec δR (getIdx (α := R) ctxR a)) (mulSpec δR (getIdx (α := R) ctxR
          a)))
        (eps := linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s)
          (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsδ (getIdxEps (Γ := Γ) (s := s) epsCtx a) δR (getIdx (α := R) ctxR a)))
          (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsδ (getIdxEps (Γ := Γ) (s := s) epsCtx a) δR (getIdx (α := R) ctxR a)))
          (mulSpec δR (getIdx (α := R) ctxR a)) (mulSpec δR (getIdx (α := R) ctxR a)))) hsum
    simpa [hEq] using hctx'
  · have hA :
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (mulSpec δS (getIdx (α := SpecScalar) ctxS b))
          (mulSpec δR (getIdx (α := R) ctxR b))
          (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsδ (getIdxEps (Γ := Γ) (s := s) epsCtx b) δR (getIdx (α := R) ctxR b))) := by
      simpa using
        (approxTensor_mul_spec (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (xS := δS) (yS := getIdx (α := SpecScalar) ctxS b)
          (xR := δR) (yR := getIdx (α := R) ctxR b)
          (epsx := epsδ) (epsy := getIdxEps (Γ := Γ) (s := s) epsCtx b) hδ hb)
    have hB :
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (mulSpec δS (getIdx (α := SpecScalar) ctxS a))
          (mulSpec δR (getIdx (α := R) ctxR a))
          (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsδ (getIdxEps (Γ := Γ) (s := s) epsCtx a) δR (getIdx (α := R) ctxR a))) := by
      simpa using
        (approxTensor_mul_spec (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (xS := δS) (yS := getIdx (α := SpecScalar) ctxS a)
          (xR := δR) (yR := getIdx (α := R) ctxR a)
          (epsx := epsδ) (epsy := getIdxEps (Γ := Γ) (s := s) epsCtx a) hδ ha)
    have hctx' :=
      approxCtx_set2Idx_ne (β := β) (fexp := fexp) (rnd := rnd)
        (Γ := Γ) (s₁ := s) (s₂ := s) a b
        (t₁S := mulSpec δS (getIdx (α := SpecScalar) ctxS b))
        (t₁R := mulSpec δR (getIdx (α := R) ctxR b))
        (eps₁ := linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) epsδ (getIdxEps (Γ := Γ) (s := s) epsCtx b) δR (getIdx (α := R) ctxR b)))
        (t₂S := mulSpec δS (getIdx (α := SpecScalar) ctxS a))
        (t₂R := mulSpec δR (getIdx (α := R) ctxR a))
        (eps₂ := linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) epsδ (getIdxEps (Γ := Γ) (s := s) epsCtx a) δR (getIdx (α := R) ctxR a)))
        hA hB hEq
    simpa [hEq] using hctx'

/--
Reverse node for scaling by a constant: `z = c * a`.
-/
def scaleRevNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) (c : R) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { toFwdNode := scaleNode (β := β) (fexp := fexp) (rnd := rnd) a c
      vjpSpec := fun _ctx δ =>
        SparseContext.setIdx (α := SpecScalar) (Γ := Γ) (s := s) a
          (scaleSpec (α := SpecScalar) (s := s) δ (toSpec (β := β) (fexp := fexp) (rnd := rnd) c))
      vjpRuntime := fun _ctx δ =>
        SparseContext.setIdx (α := R) (Γ := Γ) (s := s) a (scaleSpec (α := R) (s := s) δ c)
      vjpBound := fun _epsCtx _ctxR epsδ δR =>
        EList.setIdx (Γ := Γ) (s := s) a
          (linfNorm (scaleBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsδ c δR))
      vjpSound := ?_ }
  intro _ctxS _ctxR _epsCtx δS δR epsδ _hctx hδ
  have hscale :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (scaleSpec (α := SpecScalar) (s := s) δS (toSpec (β := β) (fexp := fexp) (rnd := rnd) c))
        (scaleSpec (α := R) (s := s) δR c)
        (linfNorm (scaleBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsδ c δR)) :=
          by
    simpa using
      (approxTensor_scale_spec (β := β) (fexp := fexp) (rnd := rnd) (c := c)
        (s := s) (xS := δS) (xR := δR) (eps := epsδ) hδ)
  have hctx' :=
    approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
      (Γ := Γ) (s := s) a
      (tS := scaleSpec (α := SpecScalar) (s := s) δS (toSpec (β := β) (fexp := fexp) (rnd := rnd)
        c))
      (tR := scaleSpec (α := R) (s := s) δR c)
      (eps := linfNorm (scaleBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsδ c
        δR)) hscale
  simpa using hctx'

/--
Reverse node for negation: `z = -a`.
-/
def negRevNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { toFwdNode := negNode (β := β) (fexp := fexp) (rnd := rnd) a
      vjpSpec := fun _ctx δ =>
        SparseContext.setIdx (α := SpecScalar) (Γ := Γ) (s := s) a (negSpec δ)
      vjpRuntime := fun _ctx δ =>
        SparseContext.setIdx (α := R) (Γ := Γ) (s := s) a (negSpec δ)
      vjpBound := fun _epsCtx _ctxR epsδ δR =>
        EList.setIdx (Γ := Γ) (s := s) a (linfNorm (negBoundTensor (β := β) (fexp := fexp) (rnd
          := rnd) (s := s) epsδ δR))
      vjpSound := ?_ }
  intro _ctxS _ctxR _epsCtx δS δR epsδ _hctx hδ
  have hneg :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (negSpec δS) (negSpec δR)
        (linfNorm (negBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsδ δR)) := by
    simpa using
      (approxTensor_neg_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s) (xS := δS) (xR := δR) (eps := epsδ) hδ)
  have hctx' :=
    approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
      (Γ := Γ) (s := s) a (tS := negSpec δS) (tR := negSpec δR)
      (eps := linfNorm (negBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsδ δR))
        hneg
  simpa using hctx'

/--
Reverse node for `exp`.
-/
def expRevNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { toFwdNode := expNode (β := β) (fexp := fexp) (rnd := rnd) a
      vjpSpec := fun ctx δ =>
        let x := getIdx (α := SpecScalar) ctx a
        let ex := mapSpec (s := s) MathFunctions.exp x
        SparseContext.setIdx (α := SpecScalar) (Γ := Γ) (s := s) a (mulSpec ex δ)
      vjpRuntime := fun ctx δ =>
        let x := getIdx (α := R) ctx a
        let ex := mapSpec (s := s) MathFunctions.exp x
        SparseContext.setIdx (α := R) (Γ := Γ) (s := s) a (mulSpec ex δ)
      vjpBound := fun epsCtx ctxR epsδ δR =>
        let xR := getIdx (α := R) ctxR a
        let exR := mapSpec (s := s) MathFunctions.exp xR
        let epsEx :=
          linfNorm (expBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) xR)
        let epsOut :=
          linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsEx epsδ exR δR)
        EList.setIdx (Γ := Γ) (s := s) a epsOut
      vjpSound := ?_ }
  intro ctxS ctxR epsCtx δS δR epsδ hctx hδ
  have hx :=
    approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx a
  have hex :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mapSpec (s := s) MathFunctions.exp (getIdx (α := SpecScalar) ctxS a))
        (mapSpec (s := s) MathFunctions.exp (getIdx (α := R) ctxR a))
        (linfNorm (expBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a))) := by
    simpa [Spec.Tensor.expSpec] using
      (approxTensor_exp_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := getIdx (α := SpecScalar) ctxS a)
        (xR := getIdx (α := R) ctxR a)
        (eps := getIdxEps (Γ := Γ) (s := s) epsCtx a) hx)
  have hout :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mulSpec (mapSpec (s := s) MathFunctions.exp (getIdx (α := SpecScalar) ctxS a)) δS)
        (mulSpec (mapSpec (s := s) MathFunctions.exp (getIdx (α := R) ctxR a)) δR)
        (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (expBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          epsδ
          (mapSpec (s := s) MathFunctions.exp (getIdx (α := R) ctxR a))
          δR)) := by
    simpa using
      (approxTensor_mul_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := mapSpec (s := s) MathFunctions.exp (getIdx (α := SpecScalar) ctxS a))
        (yS := δS)
        (xR := mapSpec (s := s) MathFunctions.exp (getIdx (α := R) ctxR a))
        (yR := δR)
        (epsx := linfNorm (expBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
        (epsy := epsδ)
        hex hδ)
  have hctx' :=
    approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
      (Γ := Γ) (s := s) a
      (tS := mulSpec (mapSpec (s := s) MathFunctions.exp (getIdx (α := SpecScalar) ctxS a)) δS)
      (tR := mulSpec (mapSpec (s := s) MathFunctions.exp (getIdx (α := R) ctxR a)) δR)
      (eps := linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (linfNorm (expBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
        epsδ
        (mapSpec (s := s) MathFunctions.exp (getIdx (α := R) ctxR a))
        δR)) hout
  simpa using hctx'

/--
Reverse node for `tanh`.
-/
def tanhRevNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { toFwdNode := tanhNode (β := β) (fexp := fexp) (rnd := rnd) a
      vjpSpec := fun ctx δ =>
        let x := getIdx (α := SpecScalar) ctx a
        let t := mapSpec (s := s) MathFunctions.tanh x
        let df := subSpec (Spec.fill (1 : ℝ) s) (mulSpec t t)
        SparseContext.setIdx (α := SpecScalar) (Γ := Γ) (s := s) a (mulSpec df δ)
      vjpRuntime := fun ctx δ =>
        let x := getIdx (α := R) ctx a
        let t := mapSpec (s := s) MathFunctions.tanh x
        let df := subSpec (Spec.fill (1 : R) s) (mulSpec t t)
        SparseContext.setIdx (α := R) (Γ := Γ) (s := s) a (mulSpec df δ)
      vjpBound := fun epsCtx ctxR epsδ δR =>
        let xR := getIdx (α := R) ctxR a
        let tR := mapSpec (s := s) MathFunctions.tanh xR
        let epsT :=
          linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) xR)
        let epsSq :=
          linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsT epsT tR tR)
        let onesR : Tensor R s := Spec.fill (1 : R) s
        let epsDf :=
          linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (neuralUlp β fexp (1 : ℝ) / 2) epsSq onesR (mulSpec tR
              tR))
        let dfR := subSpec onesR (mulSpec tR tR)
        let epsOut :=
          linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsDf epsδ dfR δR)
        EList.setIdx (Γ := Γ) (s := s) a epsOut
      vjpSound := ?_ }
  intro ctxS ctxR epsCtx δS δR epsδ hctx hδ
  have hx :=
    approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx a
  have ht :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctxS a))
        (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
        (linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a))) := by
    simpa using
      (approxTensor_tanh_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := getIdx (α := SpecScalar) ctxS a)
        (xR := getIdx (α := R) ctxR a)
        (eps := getIdxEps (Γ := Γ) (s := s) epsCtx a) hx)
  have hsq :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctxS a))
          (mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctxS a)))
        (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
          (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a)))
        (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
          (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a)))) := by
    simpa using
      (approxTensor_mul_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctxS a))
        (yS := mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctxS a))
        (xR := mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
        (yR := mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
        (epsx := linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
        (epsy := linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
        ht ht)
  have hones :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.fill (1 : ℝ) s) (Spec.fill (1 : R) s)
        (neuralUlp β fexp (1 : ℝ) / 2) :=
    approxTensor_fill_one (β := β) (fexp := fexp) (rnd := rnd) (s := s)
  have hdf :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (subSpec (Spec.fill (1 : ℝ) s)
          (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctxS a))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctxS a))))
        (subSpec (Spec.fill (1 : R) s)
          (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))))
        (linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (neuralUlp β fexp (1 : ℝ) / 2)
          (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s)
            (linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            (linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))))
          (Spec.fill (1 : R) s)
          (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))))) := by
    simpa using
      (approxTensor_sub_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := Spec.fill (1 : ℝ) s)
        (yS := mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctxS a))
          (mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctxS a)))
        (xR := Spec.fill (1 : R) s)
        (yR := mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
          (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a)))
        (epsx := neuralUlp β fexp (1 : ℝ) / 2)
        (epsy := linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
          (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))))
        hones hsq)
  have hout :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mulSpec
          (subSpec (Spec.fill (1 : ℝ) s)
            (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctxS a))
              (mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctxS a))))
          δS)
        (mulSpec
          (subSpec (Spec.fill (1 : R) s)
            (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
              (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))))
          δR)
        (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s)
            (neuralUlp β fexp (1 : ℝ) / 2)
            (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s)
              (linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
              (linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
              (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
              (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))))
            (Spec.fill (1 : R) s)
            (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
              (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a)))))
          epsδ
          (subSpec (Spec.fill (1 : R) s)
            (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
              (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))))
          δR)) := by
    simpa using
      (approxTensor_mul_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := subSpec (Spec.fill (1 : ℝ) s)
          (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctxS a))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctxS a))))
        (yS := δS)
        (xR := subSpec (Spec.fill (1 : R) s)
          (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))))
        (yR := δR)
        (epsx := linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (neuralUlp β fexp (1 : ℝ) / 2)
          (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s)
            (linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            (linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))))
          (Spec.fill (1 : R) s)
          (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a)))))
        (epsy := epsδ)
        hdf hδ)
  have hctx' :=
    approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
      (Γ := Γ) (s := s) a
      (tS := mulSpec
        (subSpec (Spec.fill (1 : ℝ) s)
          (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctxS a))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := SpecScalar) ctxS a))))
        δS)
      (tR := mulSpec
        (subSpec (Spec.fill (1 : R) s)
          (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))))
        δR)
      (eps := linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (neuralUlp β fexp (1 : ℝ) / 2)
          (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s)
            (linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            (linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))))
          (Spec.fill (1 : R) s)
          (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a)))))
        epsδ
        (subSpec (Spec.fill (1 : R) s)
          (mulSpec (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))
            (mapSpec (s := s) MathFunctions.tanh (getIdx (α := R) ctxR a))))
        δR)) hout
  simpa using hctx'

/--
Reverse node for `sigmoid`.
-/
def sigmoidRevNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { toFwdNode := sigmoidNode (β := β) (fexp := fexp) (rnd := rnd) a
      vjpSpec := fun ctx δ =>
        let x := getIdx (α := SpecScalar) ctx a
        let sS := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) x
        let df := mulSpec sS (subSpec (Spec.fill (1 : ℝ) s) sS)
        SparseContext.setIdx (α := SpecScalar) (Γ := Γ) (s := s) a (mulSpec df δ)
      vjpRuntime := fun ctx δ =>
        let x := getIdx (α := R) ctx a
        let sR := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) x
        let df := mulSpec sR (subSpec (Spec.fill (1 : R) s) sR)
        SparseContext.setIdx (α := R) (Γ := Γ) (s := s) a (mulSpec df δ)
      vjpBound := fun epsCtx ctxR epsδ δR =>
        let xR := getIdx (α := R) ctxR a
        let sR := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) xR
        let epsS :=
          linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) xR)
        let epsOne : ℝ := neuralUlp β fexp (1 : ℝ) / 2
        let epsOneMinus :=
          linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsOne epsS (Spec.fill (1 : R) s) sR)
        let oneMinusR := subSpec (Spec.fill (1 : R) s) sR
        let epsDf :=
          linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsS epsOneMinus sR oneMinusR)
        let dfR := mulSpec sR oneMinusR
        let epsOut :=
          linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsDf epsδ dfR δR)
        EList.setIdx (Γ := Γ) (s := s) a epsOut
      vjpSound := ?_ }
  intro ctxS ctxR epsCtx δS δR epsδ hctx hδ
  have hx :=
    approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx a
  have hs :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
          a))
        (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
        (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a))) := by
    simpa using
      (approxTensor_sigmoid_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := getIdx (α := SpecScalar) ctxS a)
        (xR := getIdx (α := R) ctxR a)
        (eps := getIdxEps (Γ := Γ) (s := s) epsCtx a) hx)
  have hones :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.fill (1 : ℝ) s) (Spec.fill (1 : R) s)
        (neuralUlp β fexp (1 : ℝ) / 2) :=
    approxTensor_fill_one (β := β) (fexp := fexp) (rnd := rnd) (s := s)
  have honeMinus :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (subSpec (Spec.fill (1 : ℝ) s)
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
            a)))
        (subSpec (Spec.fill (1 : R) s)
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a)))
        (linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (neuralUlp β fexp (1 : ℝ) / 2)
          (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (Spec.fill (1 : R) s)
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a)))) :=
            by
    simpa using
      (approxTensor_sub_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := Spec.fill (1 : ℝ) s)
        (yS := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar)
          ctxS a))
        (xR := Spec.fill (1 : R) s)
        (yR := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
        (epsx := neuralUlp β fexp (1 : ℝ) / 2)
        (epsy := linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
        hones hs)
  have hdf :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mulSpec
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
            a))
          (subSpec (Spec.fill (1 : ℝ) s)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar)
              ctxS a))))
        (mulSpec
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
          (subSpec (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))))
        (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s)
            (neuralUlp β fexp (1 : ℝ) / 2)
            (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))))
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
          (subSpec (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a)))))
              := by
    simpa using
      (approxTensor_mul_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar)
          ctxS a))
        (yS := subSpec (Spec.fill (1 : ℝ) s)
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
            a)))
        (xR := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
        (yR := subSpec (Spec.fill (1 : R) s)
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a)))
        (epsx := linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
        (epsy := linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (neuralUlp β fexp (1 : ℝ) / 2)
          (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (Spec.fill (1 : R) s)
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))))
        hs honeMinus)
  have hout :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mulSpec
          (mulSpec
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar)
              ctxS a))
            (subSpec (Spec.fill (1 : ℝ) s)
              (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar)
                ctxS a))))
          δS)
        (mulSpec
          (mulSpec
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
            (subSpec (Spec.fill (1 : R) s)
              (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))))
          δR)
        (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s)
            (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            (linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s)
              (neuralUlp β fexp (1 : ℝ) / 2)
              (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
              (Spec.fill (1 : R) s)
              (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))))
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
            (subSpec (Spec.fill (1 : R) s)
              (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR
                a)))))
          epsδ
          (mulSpec
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
            (subSpec (Spec.fill (1 : R) s)
              (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))))
          δR)) := by
    simpa using
      (approxTensor_mul_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := mulSpec
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
            a))
          (subSpec (Spec.fill (1 : ℝ) s)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar)
              ctxS a))))
        (yS := δS)
        (xR := mulSpec
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
          (subSpec (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))))
        (yR := δR)
        (epsx := linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s)
            (neuralUlp β fexp (1 : ℝ) / 2)
            (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))))
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
          (subSpec (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a)))))
        (epsy := epsδ)
        hdf hδ)
  have hctx' :=
    approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
      (Γ := Γ) (s := s) a
      (tS := mulSpec
        (mulSpec
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
            a))
          (subSpec (Spec.fill (1 : ℝ) s)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar)
              ctxS a))))
        δS)
      (tR := mulSpec
        (mulSpec
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
          (subSpec (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))))
        δR)
      (eps := linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s)
            (neuralUlp β fexp (1 : ℝ) / 2)
            (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))))
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
          (subSpec (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a)))))
        epsδ
        (mulSpec
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
          (subSpec (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))))
        δR)) hout
  simpa using hctx'

/--
Reverse node for `softplus`.
-/
def softplusRevNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { toFwdNode := softplusNode (β := β) (fexp := fexp) (rnd := rnd) a
      vjpSpec := fun ctx δ =>
        let x := getIdx (α := SpecScalar) ctx a
        let sig := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) x
        SparseContext.setIdx (α := SpecScalar) (Γ := Γ) (s := s) a (mulSpec sig δ)
      vjpRuntime := fun ctx δ =>
        let x := getIdx (α := R) ctx a
        let sig := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) x
        SparseContext.setIdx (α := R) (Γ := Γ) (s := s) a (mulSpec sig δ)
      vjpBound := fun epsCtx ctxR epsδ δR =>
        let xR := getIdx (α := R) ctxR a
        let sigR := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) xR
        let epsSig :=
          linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) xR)
        let epsOut :=
          linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsSig epsδ sigR δR)
        EList.setIdx (Γ := Γ) (s := s) a epsOut
      vjpSound := ?_ }
  intro ctxS ctxR epsCtx δS δR epsδ hctx hδ
  have hx :=
    approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx a
  have hsig :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
          a))
        (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
        (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a))) := by
    simpa using
      (approxTensor_sigmoid_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := getIdx (α := SpecScalar) ctxS a)
        (xR := getIdx (α := R) ctxR a)
        (eps := getIdxEps (Γ := Γ) (s := s) epsCtx a) hx)
  have hout :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mulSpec (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α :=
          SpecScalar) ctxS a)) δS)
        (mulSpec (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR
          a)) δR)
        (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          epsδ
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
          δR)) := by
    simpa using
      (approxTensor_mul_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar)
          ctxS a))
        (yS := δS)
        (xR := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
        (yR := δR)
        (epsx := linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
        (epsy := epsδ)
        hsig hδ)
  have hctx' :=
    approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
      (Γ := Γ) (s := s) a
      (tS := mulSpec
        (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
          a))
        δS)
      (tR := mulSpec
        (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
        δR)
      (eps := linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
        epsδ
        (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
        δR)) hout
  simpa using hctx'

/--
Reverse node for a log with an explicit stabilization parameter `ε` (to avoid `log 0`-style issues).
-/
def safeLogRevNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) (ε : ℝ) (hε : 0 < ε) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  let epsR : R := TorchLean.Floats.NF.ofReal (β := β) (fexp := fexp) (rnd := rnd) ε
  let epsErr : ℝ := neuralUlp β fexp ε / 2
  refine
    { toFwdNode := safeLogSoftplusNode (β := β) (fexp := fexp) (rnd := rnd) a ε hε
      vjpSpec := fun ctx δ =>
        let x := getIdx (α := SpecScalar) ctx a
        let num := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) x
        let sp := mapSpec (s := s) (Activation.Math.softplusSpec (α := ℝ)) x
        let denom := addSpec sp (Spec.fill ε s)
        let df := map2Spec (s := s) (safeDiv (ε := ε)) num denom
        SparseContext.setIdx (α := SpecScalar) (Γ := Γ) (s := s) a (mulSpec df δ)
      vjpRuntime := fun ctx δ =>
        let xR := getIdx (α := R) ctx a
        let numR := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) xR
        let spR := mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) xR
        let denomR := addSpec spR (Spec.fill epsR s)
        let dfR := map2Spec (s := s) (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε) numR denomR
        SparseContext.setIdx (α := R) (Γ := Γ) (s := s) a (mulSpec dfR δ)
      vjpBound := fun epsCtx ctxR epsδ δR =>
        let xR := getIdx (α := R) ctxR a
        let numR := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) xR
        let spR := mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) xR
        let epsNum :=
          linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) xR)
        let epsSp :=
          linfNorm (softplusBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) xR)
        let denomR := addSpec spR (Spec.fill epsR s)
        let epsDen :=
          linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsSp epsErr spR (Spec.fill epsR s))
        let epsDf :=
          linfNorm (safeDivBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) ε epsNum epsDen numR denomR)
        let dfR := map2Spec (s := s) (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε) numR denomR
        let epsOut :=
          linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsDf epsδ dfR δR)
        EList.setIdx (Γ := Γ) (s := s) a epsOut
      vjpSound := ?_ }
  intro ctxS ctxR epsCtx δS δR epsδ hctx hδ
  have hx :=
    approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx a

  have hnum :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
          a))
        (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
        (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a))) := by
    simpa using
      (approxTensor_sigmoid_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := getIdx (α := SpecScalar) ctxS a)
        (xR := getIdx (α := R) ctxR a)
        (eps := getIdxEps (Γ := Γ) (s := s) epsCtx a) hx)

  have hsp :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mapSpec (s := s) (Activation.Math.softplusSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
          a))
        (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R) ctxR
          a))
        (linfNorm (softplusBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a))) := by
    simpa using
      (approxTensor_softplus_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := getIdx (α := SpecScalar) ctxS a)
        (xR := getIdx (α := R) ctxR a)
        (eps := getIdxEps (Γ := Γ) (s := s) epsCtx a) hx)

  have heps_val :
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) epsR - ε) ≤ epsErr := by
    simpa [epsR, epsErr, toSpec, TorchLean.Floats.NF.toReal, TorchLean.Floats.NF.ofReal,
      TorchLean.Floats.NF.roundR, Proofs.RuntimeRoundingApprox.roundR] using
        (Proofs.RuntimeRoundingApprox.roundR_abs_error (β := β) (fexp := fexp) (rnd := rnd) ε)
  have heps :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.fill ε s) (Spec.fill epsR s) epsErr := by
    simpa [epsErr] using
      (approxTensor_fill_const (β := β) (fexp := fexp) (rnd := rnd)
        (cS := ε) (cR := epsR) (eps := epsErr) heps_val (s := s))

  have hden :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (addSpec
          (mapSpec (s := s) (Activation.Math.softplusSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
            a))
          (Spec.fill ε s))
        (addSpec
          (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R) ctxR
            a))
          (Spec.fill epsR s))
        (linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (softplusBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          epsErr
          (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R) ctxR
            a))
          (Spec.fill epsR s))) := by
    simpa using
      (approxTensor_add_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := mapSpec (s := s) (Activation.Math.softplusSpec (α := ℝ)) (getIdx (α := SpecScalar)
          ctxS a))
        (yS := Spec.fill ε s)
        (xR := mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R)
          ctxR a))
        (yR := Spec.fill epsR s)
        (epsx := linfNorm (softplusBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
        (epsy := epsErr)
        hsp heps)

  have hdf :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (map2Spec (s := s) (safeDiv (ε := ε))
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
            a))
          (addSpec
            (mapSpec (s := s) (Activation.Math.softplusSpec (α := ℝ)) (getIdx (α := SpecScalar)
              ctxS a))
            (Spec.fill ε s)))
        (map2Spec (s := s) (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε)
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
          (addSpec
            (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R)
              ctxR a))
            (Spec.fill epsR s)))
        (linfNorm (safeDivBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) ε
          (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s)
            (linfNorm (softplusBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            epsErr
            (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R)
              ctxR a))
            (Spec.fill epsR s)))
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
          (addSpec
            (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R)
              ctxR a))
            (Spec.fill epsR s)))) := by
    simpa using
      (approxTensor_safeDiv_spec (β := β) (fexp := fexp) (rnd := rnd) (s := s) (ε := ε) hε
        (xS := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar)
          ctxS a))
        (yS := addSpec
          (mapSpec (s := s) (Activation.Math.softplusSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
            a))
          (Spec.fill ε s))
        (xR := mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
        (yR := addSpec
          (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R) ctxR
            a))
          (Spec.fill epsR s))
        (epsx := linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
        (epsy := linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (softplusBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          epsErr
          (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R) ctxR
            a))
          (Spec.fill epsR s)))
        hnum hden)

  have hout :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mulSpec
          (map2Spec (s := s) (safeDiv (ε := ε))
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar)
              ctxS a))
            (addSpec
              (mapSpec (s := s) (Activation.Math.softplusSpec (α := ℝ)) (getIdx (α := SpecScalar)
                ctxS a))
              (Spec.fill ε s)))
          δS)
        (mulSpec
          (map2Spec (s := s) (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
            (addSpec
              (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R)
                ctxR a))
              (Spec.fill epsR s)))
          δR)
        (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (safeDivBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) ε
            (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            (linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s)
              (linfNorm (softplusBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
              epsErr
              (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R)
                ctxR a))
              (Spec.fill epsR s)))
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
            (addSpec
              (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R)
                ctxR a))
              (Spec.fill epsR s))))
          epsδ
          (map2Spec (s := s) (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
            (addSpec
              (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R)
                ctxR a))
              (Spec.fill epsR s)))
          δR)) := by
    simpa using
      (approxTensor_mul_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS :=
          map2Spec (s := s) (safeDiv (ε := ε))
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar)
              ctxS a))
            (addSpec
              (mapSpec (s := s) (Activation.Math.softplusSpec (α := ℝ)) (getIdx (α := SpecScalar)
                ctxS a))
              (Spec.fill ε s)))
        (yS := δS)
        (xR :=
          map2Spec (s := s) (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε)
            (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
            (addSpec
              (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R)
                ctxR a))
              (Spec.fill epsR s)))
        (yR := δR)
        (epsx := linfNorm (safeDivBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) ε
          (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s)
            (linfNorm (softplusBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            epsErr
            (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R)
              ctxR a))
            (Spec.fill epsR s)))
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
          (addSpec
            (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R)
              ctxR a))
            (Spec.fill epsR s))))
        (epsy := epsδ)
        hdf hδ)

  have hctx' :=
    approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
      (Γ := Γ) (s := s) a
      (tS := mulSpec
        (map2Spec (s := s) (safeDiv (ε := ε))
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
            a))
          (addSpec
            (mapSpec (s := s) (Activation.Math.softplusSpec (α := ℝ)) (getIdx (α := SpecScalar)
              ctxS a))
            (Spec.fill ε s)))
        δS)
      (tR := mulSpec
        (map2Spec (s := s) (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε)
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
          (addSpec
            (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R)
              ctxR a))
            (Spec.fill epsR s)))
        δR)
      (eps := linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (linfNorm (safeDivBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) ε
          (linfNorm (sigmoidBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (linfNorm (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s)
            (linfNorm (softplusBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            epsErr
            (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R)
              ctxR a))
            (Spec.fill epsR s)))
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
          (addSpec
            (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R)
              ctxR a))
            (Spec.fill epsR s))))
        epsδ
        (map2Spec (s := s) (safeDivR (β := β) (fexp := fexp) (rnd := rnd) ε)
          (mapSpec (s := s) (Activation.Math.sigmoidSpec (α := R)) (getIdx (α := R) ctxR a))
          (addSpec
            (mapSpec (s := s) (softplusR (β := β) (fexp := fexp) (rnd := rnd)) (getIdx (α := R)
              ctxR a))
            (Spec.fill epsR s)))
        δR)) hout
  simpa using hctx'

/-- Reverse node for the scalar `softmax` node, using the analytic $\mathbb R$ derivative plus NF
error bounds. -/
def softmaxRevNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { toFwdNode := softmaxNode (β := β) (fexp := fexp) (rnd := rnd) a
      vjpSpec := fun ctx δ =>
        let x := getIdx (α := SpecScalar) ctx a
        let sS := mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) x
        let df := mulSpec sS (subSpec (Spec.fill (1 : ℝ) s) sS)
        SparseContext.setIdx (α := SpecScalar) (Γ := Γ) (s := s) a (mulSpec df δ)
      vjpRuntime := fun ctx δ =>
        let x := getIdx (α := R) ctx a
        let sR := mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) x
        let df := mulSpec sR (subSpec (Spec.fill (1 : R) s) sR)
        SparseContext.setIdx (α := R) (Γ := Γ) (s := s) a (mulSpec df δ)
      vjpBound := fun epsCtx ctxR epsδ δR =>
        let xR := getIdx (α := R) ctxR a
        let sR := mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) xR
        let epsS :=
          linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) xR)
        let epsOne : ℝ := neuralUlp β fexp (1 : ℝ) / 2
        let epsOneMinus :=
          linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsOne epsS (Spec.fill (1 : R) s) sR)
        let oneMinusR := subSpec (Spec.fill (1 : R) s) sR
        let epsDf :=
          linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsS epsOneMinus sR oneMinusR)
        let dfR := mulSpec sR oneMinusR
        let epsOut :=
          linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) epsDf epsδ dfR δR)
        EList.setIdx (Γ := Γ) (s := s) a epsOut
      vjpSound := ?_ }
  intro ctxS ctxR epsCtx δS δR epsδ hctx hδ
  have hx :=
    approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx a
  have hs :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
          a))
        (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))
        (linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a))) := by
    simpa using
      (approxTensor_softmax_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := getIdx (α := SpecScalar) ctxS a)
        (xR := getIdx (α := R) ctxR a)
        (eps := getIdxEps (Γ := Γ) (s := s) epsCtx a) hx)
  have hones :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Spec.fill (1 : ℝ) s) (Spec.fill (1 : R) s)
        (neuralUlp β fexp (1 : ℝ) / 2) :=
    approxTensor_fill_one (β := β) (fexp := fexp) (rnd := rnd) (s := s)
  have honeMinus :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (subSpec (Spec.fill (1 : ℝ) s)
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
            a)))
        (subSpec (Spec.fill (1 : R) s)
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a)))
        (linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (neuralUlp β fexp (1 : ℝ) / 2)
          (linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (Spec.fill (1 : R) s)
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a)))) :=
            by
    simpa using
      (approxTensor_sub_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := Spec.fill (1 : ℝ) s)
        (yS := mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (getIdx (α := SpecScalar)
          ctxS a))
        (xR := Spec.fill (1 : R) s)
        (yR := mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))
        (epsx := neuralUlp β fexp (1 : ℝ) / 2)
        (epsy := linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
        hones hs)
  have hdf :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mulSpec
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
            a))
          (subSpec (Spec.fill (1 : ℝ) s)
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (getIdx (α := SpecScalar)
              ctxS a))))
        (mulSpec
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))
          (subSpec (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))))
        (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s)
            (neuralUlp β fexp (1 : ℝ) / 2)
            (linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))))
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))
          (subSpec (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a)))))
              := by
    simpa using
      (approxTensor_mul_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (getIdx (α := SpecScalar)
          ctxS a))
        (yS := subSpec (Spec.fill (1 : ℝ) s)
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
            a)))
        (xR := mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))
        (yR := subSpec (Spec.fill (1 : R) s)
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a)))
        (epsx := linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
        (epsy := linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (neuralUlp β fexp (1 : ℝ) / 2)
          (linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (Spec.fill (1 : R) s)
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))))
        hs honeMinus)
  have hout :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (mulSpec
          (mulSpec
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (getIdx (α := SpecScalar)
              ctxS a))
            (subSpec (Spec.fill (1 : ℝ) s)
              (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (getIdx (α := SpecScalar)
                ctxS a))))
          δS)
        (mulSpec
          (mulSpec
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))
            (subSpec (Spec.fill (1 : R) s)
              (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))))
          δR)
        (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s)
            (linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            (linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s)
              (neuralUlp β fexp (1 : ℝ) / 2)
              (linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
                (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
              (Spec.fill (1 : R) s)
              (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))))
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))
            (subSpec (Spec.fill (1 : R) s)
              (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR
                a)))))
          epsδ
          (mulSpec
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))
            (subSpec (Spec.fill (1 : R) s)
              (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))))
          δR)) := by
    simpa using
      (approxTensor_mul_spec (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (xS := mulSpec
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
            a))
          (subSpec (Spec.fill (1 : ℝ) s)
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (getIdx (α := SpecScalar)
              ctxS a))))
        (yS := δS)
        (xR := mulSpec
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))
          (subSpec (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))))
        (yR := δR)
        (epsx := linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s)
            (neuralUlp β fexp (1 : ℝ) / 2)
            (linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))))
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))
          (subSpec (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a)))))
        (epsy := epsδ)
        hdf hδ)
  have hctx' :=
    approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
      (Γ := Γ) (s := s) a
      (tS := mulSpec
        (mulSpec
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (getIdx (α := SpecScalar) ctxS
            a))
          (subSpec (Spec.fill (1 : ℝ) s)
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := ℝ)) (getIdx (α := SpecScalar)
              ctxS a))))
        δS)
      (tR := mulSpec
        (mulSpec
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))
          (subSpec (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))))
        δR)
      (eps := linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
        (s := s)
        (linfNorm (mulBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          (s := s)
          (linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
          (linfNorm (subBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s)
            (neuralUlp β fexp (1 : ℝ) / 2)
            (linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
              (s := s) (getIdxEps (Γ := Γ) (s := s) epsCtx a) (getIdx (α := R) ctxR a)))
            (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))))
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))
          (subSpec (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a)))))
        epsδ
        (mulSpec
          (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))
          (subSpec (Spec.fill (1 : R) s)
            (mapSpec (s := s) (Activation.Math.logisticSpec (α := R)) (getIdx (α := R) ctxR a))))
        δR)) hout
  simpa using hctx'

/--
Reverse node for ReLU, using the standard piecewise derivative/VJP.
-/
def reluRevNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ s :=
by
  classical
  refine
    { toFwdNode := reluNode (β := β) (fexp := fexp) (rnd := rnd) a
      vjpSpec := fun ctx δ =>
        let x := getIdx (α := SpecScalar) ctx a
        let gated := map2Spec (fun d x => if x > 0 then d else 0) δ x
        SparseContext.setIdx (α := SpecScalar) (Γ := Γ) (s := s) a gated
      vjpRuntime := fun ctx δ =>
        let x := getIdx (α := R) ctx a
        let gated := map2Spec (fun d x => if x > 0 then d else 0) δ x
        SparseContext.setIdx (α := R) (Γ := Γ) (s := s) a gated
      vjpBound := fun _epsCtx ctxR epsδ δR =>
        let xR := getIdx (α := R) ctxR a
        let bndT : SpecTensor s :=
          map2Spec (fun a _b => abs a + epsδ)
            (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) δR)
            (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)
        EList.setIdx (Γ := Γ) (s := s) a (linfNorm bndT)
      vjpSound := ?_ }
  intro ctxS ctxR epsCtx δS δR epsδ hctx hδ
  have hx :=
    approxCtx_getIdx (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hctx a
  have hgate :
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (map2Spec (fun d x => if x > 0 then d else 0) δS (getIdx (α := SpecScalar) ctxS a))
        (map2Spec (fun d x => if x > 0 then d else 0) δR (getIdx (α := R) ctxR a))
        (linfNorm
          (map2Spec (fun a _b => abs a + epsδ)
            (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) δR)
            (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) (getIdx
              (α := R) ctxR a)))) := by
    -- Use the generic `map2` lifting lemma with a conservative scalar bound.
    simpa using
      (approxTensor_map2_spec_of_scalar_bound (α := R)
        (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) (s := s)
        (fS := fun d x => if x > 0 then d else 0)
        (fR := fun d x => if x > 0 then d else 0)
        (bnd := fun a _b epsδ _epsx => abs a + epsδ)
        (xS := δS) (yS := getIdx (α := SpecScalar) ctxS a)
        (xR := δR) (yR := getIdx (α := R) ctxR a)
        (epsx := epsδ) (epsy := getIdxEps (Γ := Γ) (s := s) epsCtx a)
        hδ hx (by
          intro d x dR xR hd hx'
          by_cases hxS : x > 0 <;> by_cases hxR : xR > 0
          · -- both on
            simpa [hxS, hxR] using le_trans hd (le_add_of_nonneg_left (abs_nonneg _))
          · -- spec on, runtime off
            have hδ' : abs d ≤ abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) dR) + epsδ := by
              have hdiff : abs (d - toSpec (β := β) (fexp := fexp) (rnd := rnd) dR) ≤ epsδ := by
                simpa [abs_sub_comm] using hd
              calc
                abs d = abs ((d - toSpec (β := β) (fexp := fexp) (rnd := rnd) dR) + toSpec (β := β)
                  (fexp := fexp) (rnd := rnd) dR) := by ring_nf
                _ ≤ abs (d - toSpec (β := β) (fexp := fexp) (rnd := rnd) dR) + abs (toSpec (β := β)
                  (fexp := fexp) (rnd := rnd) dR) := abs_add_le _ _
                _ ≤ epsδ + abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) dR) := add_le_add hdiff
                  (le_rfl)
                _ = abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) dR) + epsδ := by ring_nf
            -- output diff is |0 - d| = |d|
            have : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (if xR > 0 then dR else 0) - (if
              x > 0 then d else 0))
                = abs d := by
                  simp [hxS, hxR]
            -- bound by |toSpec dR| + epsδ
            have : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (if xR > 0 then dR else 0) - (if
              x > 0 then d else 0))
                ≤ abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) dR) + epsδ := by
                  simpa [this] using hδ'
            simpa [hxS, hxR] using this
          · -- spec off, runtime on
            -- output diff is |toSpec dR - 0| = |toSpec dR|
            have heps : 0 ≤ epsδ := le_trans (abs_nonneg _) hd
            have : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (if xR > 0 then dR else 0) - (if
              x > 0 then d else 0))
                = abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) dR) := by
                  simp [hxS, hxR]
            have : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (if xR > 0 then dR else 0) - (if
              x > 0 then d else 0))
                ≤ abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) dR) + epsδ := by
                  simpa [this] using
                    (le_add_of_nonneg_right (a := abs (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                      dR)) heps)
            simpa [hxS, hxR] using this
          · -- both off
            have heps : 0 ≤ epsδ := le_trans (abs_nonneg _) hd
            have : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (if xR > 0 then dR else 0) - (if
              x > 0 then d else 0))
                = 0 := by simp [hxS, hxR]
            have : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (if xR > 0 then dR else 0) - (if
              x > 0 then d else 0))
                ≤ abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) dR) + epsδ := by
                  simpa [this] using add_nonneg (abs_nonneg _) heps
            simpa [hxS, hxR] using this))
  have hctx' :=
    approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
      (Γ := Γ) (s := s) a
      (tS := map2Spec (fun d x => if x > 0 then d else 0) δS (getIdx (α := SpecScalar) ctxS a))
      (tR := map2Spec (fun d x => if x > 0 then d else 0) δR (getIdx (α := R) ctxR a))
      (eps := linfNorm
        (map2Spec (fun a _b => abs a + epsδ)
          (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) δR)
          (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) (getIdx
            (α := R) ctxR a)))) hgate
  simpa using hctx'

/--
Reverse node for reduction `sum`, sending the upstream gradient back along the broadcasted shape.
-/
def sumRevNode {Γ : List Shape} {s : Shape} (a : Idx Γ s) :
    RevNode (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) Γ Shape.scalar :=
by
  classical
  refine
    { toFwdNode := sumNode (β := β) (fexp := fexp) (rnd := rnd) a
      vjpSpec := fun _ctx δ =>
        match δ with
        | Tensor.scalar d =>
            SparseContext.setIdx (α := SpecScalar) (Γ := Γ) (s := s) a (Spec.fill d s)
      vjpRuntime := fun _ctx δ =>
        match δ with
        | Tensor.scalar d =>
            SparseContext.setIdx (α := R) (Γ := Γ) (s := s) a (Spec.fill d s)
      vjpBound := fun _epsCtx _ctxR epsδ _δR =>
        EList.setIdx (Γ := Γ) (s := s) a epsδ
      vjpSound := ?_ }
  intro _ctxS _ctxR _epsCtx δS δR epsδ _hctx hδ
  cases δS with
  | scalar dS =>
      cases δR with
      | scalar dR =>
          have hd :
              abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) dR - dS) ≤ epsδ := by
            simpa using
              (approxTensor_scalar_iff (α := R)
                (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                (x := dS) (xR := dR) (eps := epsδ)).1 hδ
          have hfill :
              approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
                (Spec.fill dS s) (Spec.fill dR s) epsδ :=
            approxTensor_fill_const (β := β) (fexp := fexp) (rnd := rnd) (cS := dS) (cR := dR) (eps :=
              epsδ) hd
              (s := s)
          have hctx' :=
            approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
              (Γ := Γ) (s := s) a (tS := Spec.fill dS s) (tR := Spec.fill dR s) (eps := epsδ) hfill
          simpa using hctx'


end NFBackend

end

end RuntimeApprox
end Proofs
