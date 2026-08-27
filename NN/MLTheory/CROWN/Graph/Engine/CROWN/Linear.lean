/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.Affine

/-!
# Forward CROWN Bounds

This module computes lower and upper affine bounds for each graph node. The pass is
Linear nodes compose affine lower and upper forms. Operator-specific rules may retain more
dependence, while unsupported or numerically unjustified cases fall back to constant affine bounds
derived from the already-computed IBP box.
-/

public section

namespace NN.MLTheory.CROWN.Graph

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN
open NN.IR

variable {α : Type} [Context α]
variable [BoundOps α]

open BoundOps

/-!
For a chosen flattened input node `ctx.inputId`, the pass computes a pair of affine forms
`loAff(x) ≤ node(x) ≤ hiAff(x)` for each supported node.  The transfer rules below use the usual
CROWN/DeepPoly ingredients:

- Linear layers use sign-splitting (`W⁺/W⁻`) to combine parent bounds.
- ReLU uses the standard triangle upper bound and a simple evidence-based lower choice (0 vs x).
- Exp/log use secant/tangent bounds (convex/concave).
- Softmax and LayerNorm use conservative last-axis relaxations.

Unsupported axes or shape mismatches fall back to constant affine bounds derived from the IBP box.
-/

/-- Exact lower and upper affine bounds for the identity node. -/
@[expose] def boundsIdentity (n : Nat) : FlatAffineBounds α :=
  { inDim := n, outDim := n, loAff := affIdentity (α:=α) n, hiAff := affIdentity (α:=α) n }

/-- Constant affine bounds with zero coefficient matrix and explicit lower/upper offsets. -/
@[expose]
def boundsConst (inputDim outDim : Nat) (lo hi : Tensor α [outDim]) :
  FlatAffineBounds α :=
  let zA := Spec.fill (α:=α) 0 (.dim outDim (.dim inputDim .scalar))
  { inDim := inputDim
    outDim := outDim
    loAff := { A := zA, c := lo }
    hiAff := { A := zA, c := hi } }

/-- Materialize an affine vector so later graph passes do not accumulate deep tensor closures. -/
def materializeAffineVec {inDim outDim : Nat} (a : AffineVec α inDim outDim) : AffineVec α
  inDim outDim :=
  { A := Tensor.materialize a.A, c := Tensor.materialize a.c }

/-- Propagate affine lower/upper bounds through an affine layer `W*x + b`. -/
def propagateLinearBounds
  {n m : Nat}
  (W : Tensor α [m, n])
  (b : Tensor α [m])
  (xB : FlatAffineBounds α)
  (hout : xB.outDim = n) : FlatAffineBounds α := by
  -- Align parent affines to outDim=n.
  let xLo : AffineVec α xB.inDim n :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=n) hout xB.loAff
  let xHi : AffineVec α xB.inDim n :=
    castAffineOut (α:=α) (n:=xB.inDim) (m:=xB.outDim) (m':=n) hout xB.hiAff
  let xLo := materializeAffineVec (α := α) (inDim := xB.inDim) (outDim := n) xLo
  let xHi := materializeAffineVec (α := α) (inDim := xB.inDim) (outDim := n) xHi
  let Wpos := IBP.matPos (α:=α) (m:=m) (n:=n) W
  let Wneg := IBP.matNeg (α:=α) (m:=m) (n:=n) W
  let A_hi :=
    Tensor.materialize <|
      Tensor.addSpec (Spec.matMulSpec (α:=α) Wpos xHi.A) (Spec.matMulSpec (α:=α) Wneg xLo.A)
  let c_hi :=
    Tensor.materialize <|
      Tensor.addSpec
        (Tensor.addSpec (Spec.matVecMulSpec (α:=α) Wpos xHi.c) (Spec.matVecMulSpec (α:=α)
          Wneg xLo.c))
        b
  let A_lo :=
    Tensor.materialize <|
      Tensor.addSpec (Spec.matMulSpec (α:=α) Wpos xLo.A) (Spec.matMulSpec (α:=α) Wneg xHi.A)
  let c_lo :=
    Tensor.materialize <|
      Tensor.addSpec
        (Tensor.addSpec (Spec.matVecMulSpec (α:=α) Wpos xLo.c) (Spec.matVecMulSpec (α:=α)
          Wneg xHi.c))
        b
  exact
    { inDim := xB.inDim
      outDim := m
      loAff := { A := A_lo, c := c_lo }
      hiAff := { A := A_hi, c := c_hi } }

/-- Compose an affine form with a diagonal affine relaxation `slope * x + bias`. -/
def affApplyDiag {inDim outDim : Nat}
  (slopes bias : Tensor α [outDim])
  (aff : AffineVec α inDim outDim) : AffineVec α inDim outDim :=
  match slopes, bias, aff.A, aff.c with
  | .dim sF, .dim bF, .dim rows, .dim cF =>
    let A' :=
      Tensor.dim (fun i =>
        match sF i, rows i with
        | .scalar si, .dim cols =>
          Tensor.dim (fun j =>
            match cols j with
            | .scalar aij => Tensor.scalar (si * aij)))
    let c' :=
      Tensor.dim (fun i =>
        match sF i, cF i, bF i with
        | .scalar si, .scalar ci, .scalar bi => Tensor.scalar (si * ci + bi))
    { A := A', c := c' }

/-- Apply a diagonal relaxation for an upper bound, selecting parent rows by slope sign. -/
def affApplyDiagSignedUpper {inDim outDim : Nat}
  (slopes bias : Tensor α [outDim])
  (xLo xHi : AffineVec α inDim outDim) : AffineVec α inDim outDim :=
  match slopes, bias, xLo.A, xHi.A, xLo.c, xHi.c with
  | .dim sF, .dim bF, .dim rowsL, .dim rowsU, .dim cL, .dim cU =>
    let A' :=
      Tensor.dim (fun i =>
        match sF i with
        | .scalar si =>
          let row := if decide (si > Numbers.zero) then rowsU i else rowsL i
          match row with
          | .dim cols =>
            Tensor.dim (fun j =>
              match cols j with
              | .scalar aij => Tensor.scalar (si * aij)))
    let c' :=
      Tensor.dim (fun i =>
        match sF i, bF i with
        | .scalar si, .scalar bi =>
          let ci := if decide (si > Numbers.zero) then cU i else cL i
          match ci with
          | .scalar cv => Tensor.scalar (si * cv + bi))
    { A := A', c := c' }

/-- Apply a diagonal relaxation for a lower bound, selecting parent rows by slope sign. -/
def affApplyDiagSignedLower {inDim outDim : Nat}
  (slopes bias : Tensor α [outDim])
  (xLo xHi : AffineVec α inDim outDim) : AffineVec α inDim outDim :=
  match slopes, bias, xLo.A, xHi.A, xLo.c, xHi.c with
  | .dim sF, .dim bF, .dim rowsL, .dim rowsU, .dim cL, .dim cU =>
    let A' :=
      Tensor.dim (fun i =>
        match sF i with
        | .scalar si =>
          let row := if decide (si > Numbers.zero) then rowsL i else rowsU i
          match row with
          | .dim cols =>
            Tensor.dim (fun j =>
              match cols j with
              | .scalar aij => Tensor.scalar (si * aij)))
    let c' :=
      Tensor.dim (fun i =>
        match sF i, bF i with
        | .scalar si, .scalar bi =>
          let ci := if decide (si > Numbers.zero) then cL i else cU i
          match ci with
          | .scalar cv => Tensor.scalar (si * cv + bi))
    { A := A', c := c' }


end NN.MLTheory.CROWN.Graph
