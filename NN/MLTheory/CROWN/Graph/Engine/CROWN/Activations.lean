/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.CROWN.Linear

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN
open NN.IR

variable {α : Type} [Context α]
variable [BoundOps α]

open BoundOps

/-!
# CROWN Activation Relaxations

Affine transfer rules for nonlinear scalar activations.
-/

/--
One-dimensional sigmoid linear bounds on an interval `[l,u]`.

Returns `(a_lo, b_lo, a_hi, b_hi)` for lower and upper affine lines.
-/
def sigmoidLineBounds (l u : α) : α × α × α × α :=
  let σ (x : α) := Activation.Math.sigmoidSpec (α := α) x
  let σ' (x : α) := Activation.Math.sigmoidDerivSpec (α := α) x
  if u < Numbers.zero then
    -- Convex region: secant is an upper bound, tangent is a lower bound.
    let den := u - l
    let a_hi := if den > Numbers.epsilon then (σ u - σ l) / den else σ' l
    let b_hi := σ l - a_hi * l
    let a_lo := σ' u
    let b_lo := σ u - a_lo * u
    (a_lo, b_lo, a_hi, b_hi)
  else if l > Numbers.zero then
    -- Concave region: tangent is an upper bound, secant is a lower bound.
    let a_hi := σ' l
    let b_hi := σ l - a_hi * l
    let den := u - l
    let a_lo := if den > Numbers.epsilon then (σ u - σ l) / den else σ' l
    let b_lo := σ l - a_lo * l
    (a_lo, b_lo, a_hi, b_hi)
  else
    -- Crossing the inflection: fall back to constant bounds.
    (Numbers.zero, σ l, Numbers.zero, σ u)

/-- Propagate CROWN bounds through ReLU using standard per-neuron triangle relaxations. -/
def propagateReluBounds
  (preB : FlatBox α) (xB : FlatAffineBounds α) (hout : xB.outDim = preB.dim) :
  FlatAffineBounds α := by
  let relaxHi0 := NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector (α:=α) (n:=preB.dim) preB.lo
    preB.hi
  let relaxLo0 := NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVectorLower (α:=α) (n:=preB.dim) preB.lo
    preB.hi
  let xLo : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.loAff
  let xHi : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.hiAff
  let loAff := NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α:=α)
    (inDim:=xB.inDim) (hidDim:=preB.dim) relaxLo0 xLo
  let hiAff := NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α:=α)
    (inDim:=xB.inDim) (hidDim:=preB.dim) relaxHi0 xHi
  exact
    { inDim := xB.inDim
      outDim := preB.dim
      loAff := materializeAffineVec (α := α) (inDim := xB.inDim) (outDim := preB.dim) loAff
      hiAff := materializeAffineVec (α := α) (inDim := xB.inDim) (outDim := preB.dim) hiAff }

/-- Propagate ReLU bounds with externally supplied α slopes for crossing neurons. -/
def propagateReluBoundsWithAlpha
  (preB : FlatBox α) (xB : FlatAffineBounds α) (hout : xB.outDim = preB.dim)
  (alpha : Tensor α [preB.dim]) : FlatAffineBounds α := by
  -- Upper relaxation: standard secant/tight bounds (independent of α).
  let relaxHi0 := NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxVector (α:=α) (n:=preB.dim) preB.lo
    preB.hi
  -- Lower relaxation: for crossing bounds l < 0 < u, use a provided per-neuron α ∈ [0,1]
  -- (line y ≥ α x), which is always sound for ReLU; stable regions override α.
  let clamp01 (x : α) : α :=
    let x0 := if x > Numbers.zero then x else Numbers.zero
    if x0 > Numbers.one then Numbers.one else x0
  let relaxLo0 : Tensor (NN.MLTheory.CROWN.Runtime.Ops.ReLURelax α) [preB.dim] :=
    match preB.lo, preB.hi, alpha with
    | .dim lF, .dim uF, .dim aF =>
      Tensor.dim (fun i =>
        match lF i, uF i, aF i with
        | .scalar l, .scalar u, .scalar a =>
          if u > Numbers.zero then
            if l > Numbers.zero then
              Tensor.scalar { slope := Numbers.one, bias := Numbers.zero }
            else
              Tensor.scalar { slope := clamp01 a, bias := Numbers.zero }
          else
            Tensor.scalar { slope := Numbers.zero, bias := Numbers.zero })
  let xLo : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.loAff
  let xHi : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.hiAff
  let loAff := NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α:=α)
    (inDim:=xB.inDim) (hidDim:=preB.dim) relaxLo0 xLo
  let hiAff := NN.MLTheory.CROWN.Runtime.Ops.ReLU.propagateAffine (α:=α)
    (inDim:=xB.inDim) (hidDim:=preB.dim) relaxHi0 xHi
  exact
    { inDim := xB.inDim
      outDim := preB.dim
      loAff := materializeAffineVec (α := α) (inDim := xB.inDim) (outDim := preB.dim) loAff
      hiAff := materializeAffineVec (α := α) (inDim := xB.inDim) (outDim := preB.dim) hiAff }

/-- Propagate CROWN bounds through `exp` using tangent/secant relaxations. -/
def propagateExpBounds
  (preB : FlatBox α) (xB : FlatAffineBounds α) (hout : xB.outDim = preB.dim) :
  FlatAffineBounds α :=
  let xLo : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.loAff
  let xHi : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.hiAff
  -- Per-component [l,u]
  let flo := getDimScalarFn (α:=α) preB.lo
  let fhi := getDimScalarFn (α:=α) preB.hi
  let slopes_hi : Tensor α [preB.dim] :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let den := u - l
        let a :=
          if den > Numbers.epsilon then (MathFunctions.exp u - MathFunctions.exp l) / den
          else MathFunctions.exp l
        Tensor.scalar a)
  let bias_hi : Tensor α [preB.dim] :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let den := u - l
        let a :=
          if den > Numbers.epsilon then (MathFunctions.exp u - MathFunctions.exp l) / den
          else MathFunctions.exp l
        let b := MathFunctions.exp l - a * l
        Tensor.scalar b)
  -- Lower: tangent at l
  let slopes_lo : Tensor α [preB.dim] :=
    Tensor.dim (fun i =>
      match flo i with
      | .scalar l =>
        Tensor.scalar (MathFunctions.exp l))
  let bias_lo : Tensor α [preB.dim] :=
    Tensor.dim (fun i =>
      match flo i with
      | .scalar l =>
        let a := MathFunctions.exp l
        Tensor.scalar (MathFunctions.exp l - a * l))
  let loAff := affApplyDiag (α:=α) (inDim:=xB.inDim) (outDim:=preB.dim) slopes_lo bias_lo xLo
  let hiAff := affApplyDiag (α:=α) (inDim:=xB.inDim) (outDim:=preB.dim) slopes_hi bias_hi xHi
  { inDim := xB.inDim, outDim := preB.dim, loAff := loAff, hiAff := hiAff }

/-- Propagate CROWN bounds through log on the positive-domain convention used by the verifier. -/
def propagateLogBounds
  (preB : FlatBox α) (xB : FlatAffineBounds α) (hout : xB.outDim = preB.dim) :
  FlatAffineBounds α :=
  let xLo : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.loAff
  let xHi : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.hiAff
  let flo := getDimScalarFn (α:=α) preB.lo
  let fhi := getDimScalarFn (α:=α) preB.hi
  -- Clamp to positive domain.
  let loSafe : Tensor α [preB.dim] :=
    Tensor.dim (fun i =>
      match flo i with
      | .scalar v => Tensor.scalar (if v > Numbers.epsilon then v else Numbers.epsilon))
  let hiSafe : Tensor α [preB.dim] :=
    Tensor.dim (fun i =>
      match fhi i with
      | .scalar v => Tensor.scalar (if v > Numbers.epsilon then v else Numbers.epsilon))
  let floS := getDimScalarFn (α:=α) loSafe
  let fhiS := getDimScalarFn (α:=α) hiSafe
  -- Upper: tangent at loSafe (concave ⇒ tangent is over-approx).
  let slopes_hi : Tensor α [preB.dim] :=
    Tensor.dim (fun i =>
      match floS i with
      | .scalar l => Tensor.scalar (Numbers.one / l))
  let bias_hi : Tensor α [preB.dim] :=
    Tensor.dim (fun i =>
      match floS i with
      | .scalar l =>
        let a := Numbers.one / l
        Tensor.scalar (MathFunctions.log l - a * l))
  -- Lower: secant on [loSafe, hiSafe] (concave ⇒ secant is under-approx).
  let slopes_lo : Tensor α [preB.dim] :=
    Tensor.dim (fun i =>
      match floS i, fhiS i with
      | .scalar l, .scalar u =>
        let den := u - l
        let a :=
          if den > Numbers.epsilon then (MathFunctions.log u - MathFunctions.log l) / den
          else Numbers.one / l
        Tensor.scalar a)
  let bias_lo : Tensor α [preB.dim] :=
    Tensor.dim (fun i =>
      match floS i, fhiS i with
      | .scalar l, .scalar u =>
        let den := u - l
        let a :=
          if den > Numbers.epsilon then (MathFunctions.log u - MathFunctions.log l) / den
          else Numbers.one / l
        let b := MathFunctions.log l - a * l
        Tensor.scalar b)
  let loAff := affApplyDiag (α:=α) (inDim:=xB.inDim) (outDim:=preB.dim) slopes_lo bias_lo xLo
  let hiAff := affApplyDiag (α:=α) (inDim:=xB.inDim) (outDim:=preB.dim) slopes_hi bias_hi xHi
  { inDim := xB.inDim, outDim := preB.dim, loAff := loAff, hiAff := hiAff }

/-- Propagate CROWN bounds through sigmoid with convex/concave one-dimensional relaxations. -/
def propagateSigmoidBounds
  (preB : FlatBox α) (xB : FlatAffineBounds α) (hout : xB.outDim = preB.dim) :
  FlatAffineBounds α :=
  let xLo : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.loAff
  let xHi : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.hiAff
  let flo := getDimScalarFn (α:=α) preB.lo
  let fhi := getDimScalarFn (α:=α) preB.hi
  let σ (x : α) := Activation.Math.sigmoidSpec (α:=α) x
  let σ' (x : α) := Activation.Math.sigmoidDerivSpec (α:=α) x
  let slopes_hi : Tensor α [preB.dim] :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        -- Convex for x ≤ 0, concave for x ≥ 0; crossing uses constant bounds.
        let a :=
          if u < Numbers.zero then
            let den := u - l
            if den > Numbers.epsilon then (σ u - σ l) / den else σ' l
          else if l > Numbers.zero then
            σ' l
          else
            Numbers.zero
        Tensor.scalar a)
  let bias_hi : Tensor α [preB.dim] :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let b :=
          if u < Numbers.zero then
            let den := u - l
            let a := if den > Numbers.epsilon then (σ u - σ l) / den else σ' l
            σ l - a * l
          else if l > Numbers.zero then
            let a := σ' l
            σ l - a * l
          else
            -- Crossing: constant upper bound = σ(u)
            σ u
        Tensor.scalar b)
  let slopes_lo : Tensor α [preB.dim] :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let a :=
          if u < Numbers.zero then
            σ' u
          else if l > Numbers.zero then
            let den := u - l
            if den > Numbers.epsilon then (σ u - σ l) / den else σ' l
          else
            Numbers.zero
        Tensor.scalar a)
  let bias_lo : Tensor α [preB.dim] :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let b :=
          if u < Numbers.zero then
            let a := σ' u
            σ u - a * u
          else if l > Numbers.zero then
            let den := u - l
            let a := if den > Numbers.epsilon then (σ u - σ l) / den else σ' l
            σ l - a * l
          else
            -- Crossing: constant lower bound = σ(l)
            σ l
        Tensor.scalar b)
  let loAff := affApplyDiag (α:=α) (inDim:=xB.inDim) (outDim:=preB.dim) slopes_lo bias_lo xLo
  let hiAff := affApplyDiag (α:=α) (inDim:=xB.inDim) (outDim:=preB.dim) slopes_hi bias_hi xHi
  { inDim := xB.inDim, outDim := preB.dim, loAff := loAff, hiAff := hiAff }

/-- Propagate CROWN bounds through tanh with convex/concave one-dimensional relaxations. -/
def propagateTanhBounds
  (preB : FlatBox α) (xB : FlatAffineBounds α) (hout : xB.outDim = preB.dim) :
  FlatAffineBounds α :=
  let xLo : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.loAff
  let xHi : AffineVec α xB.inDim preB.dim :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=preB.dim) hout xB.hiAff
  let flo := getDimScalarFn (α:=α) preB.lo
  let fhi := getDimScalarFn (α:=α) preB.hi
  let t (x : α) := Activation.Math.tanhSpec (α:=α) x
  let t' (x : α) := Activation.Math.tanhDerivSpec (α:=α) x
  let slopes_hi : Tensor α [preB.dim] :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let a :=
          if u < Numbers.zero then
            let den := u - l
            if den > Numbers.epsilon then (t u - t l) / den else t' l
          else if l > Numbers.zero then
            t' l
          else
            Numbers.zero
        Tensor.scalar a)
  let bias_hi : Tensor α [preB.dim] :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let b :=
          if u < Numbers.zero then
            let den := u - l
            let a := if den > Numbers.epsilon then (t u - t l) / den else t' l
            t l - a * l
          else if l > Numbers.zero then
            let a := t' l
            t l - a * l
          else
            t u
        Tensor.scalar b)
  let slopes_lo : Tensor α [preB.dim] :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let a :=
          if u < Numbers.zero then
            t' u
          else if l > Numbers.zero then
            let den := u - l
            if den > Numbers.epsilon then (t u - t l) / den else t' l
          else
            Numbers.zero
        Tensor.scalar a)
  let bias_lo : Tensor α [preB.dim] :=
    Tensor.dim (fun i =>
      match flo i, fhi i with
      | .scalar l, .scalar u =>
        let b :=
          if u < Numbers.zero then
            let a := t' u
            t u - a * u
          else if l > Numbers.zero then
            let den := u - l
            let a := if den > Numbers.epsilon then (t u - t l) / den else t' l
            t l - a * l
          else
            t l
        Tensor.scalar b)
  let loAff := affApplyDiag (α:=α) (inDim:=xB.inDim) (outDim:=preB.dim) slopes_lo bias_lo xLo
  let hiAff := affApplyDiag (α:=α) (inDim:=xB.inDim) (outDim:=preB.dim) slopes_hi bias_hi xHi
  { inDim := xB.inDim, outDim := preB.dim, loAff := loAff, hiAff := hiAff }


end NN.MLTheory.CROWN.Graph
