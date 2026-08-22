/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

/-
Arithmetic operators for CROWN bound propagation.

This file implements IBP and affine bounds for:
- Power: f(x) = x^n
- Sqrt: f(x) = √x
- Neg: f(x) = -x
- Reciprocal: f(x) = 1/x
- Abs: f(x) = |x|
- Min/Max: f(x,y) = min(x,y) / max(x,y)
-/

module

public import NN.MLTheory.CROWN.Core

/-!
# `NN.MLTheory.CROWN.Operators.Arithmetic`

IBP and affine transfer rules for arithmetic primitives (negation, absolute value, reciprocal,
square root, powers, min/max) used by the CROWN bound propagation engine.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Operators.Arithmetic

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN

variable {α : Type} [Context α]

/-! ### Negation -/

/-- Negation, $f(x)=-x$, is the simplest linear operation. -/
def neg (x : α) : α := -x

/-- IBP for negation. Just swaps and negates bounds. -/
def ibpNegScalar (l u : α) : α × α :=
  (-u, -l)

/-- IBP for negation on boxes. -/
def ibpNeg (n : Nat) (B : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match B.lo, B.hi with
  | .dim lo, .dim hi =>
    let outLo := Tensor.dim (fun i =>
      match hi i with
      | .scalar u => Tensor.scalar (-u))
    let outHi := Tensor.dim (fun i =>
      match lo i with
      | .scalar l => Tensor.scalar (-l))
    { lo := outLo, hi := outHi }

/-- Affine bounds for negation (exact). -/
def affNeg : α × α × α × α :=
  (-Numbers.one, Numbers.zero, -Numbers.one, Numbers.zero)

/-- Derivative of negation (constant -1). -/
def derivNeg : α × α := (-Numbers.one, -Numbers.one)

/-! ### Absolute Value -/

/-- Absolute value: $f(x)=|x|$. -/
def abs (x : α) : α :=
  if x > Numbers.zero then x else -x

/-- Interval propagation rule for scalar absolute value over `[l,u]`. -/
def ibpAbsScalar (l u : α) : α × α :=
  if l > Numbers.zero then
    -- All positive: |x| = x
    (l, u)
  else if u < Numbers.zero then
    -- All negative: |x| = -x
    (-u, -l)
  else
    -- Spans zero: min is 0, max is max(|l|, u)
    let absL := -l
    (Numbers.zero, if absL > u then absL else u)

/-- Apply the scalar absolute-value interval rule coordinatewise to a vector box. -/
def ibpAbs (n : Nat) (B : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match B.lo, B.hi with
  | .dim lo, .dim hi =>
    let outLo := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        Tensor.scalar (ibpAbsScalar l u).1)
    let outHi := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        Tensor.scalar (ibpAbsScalar l u).2)
    { lo := outLo, hi := outHi }

/--
Affine lower and upper bounds for absolute value on an ordered interval `[l, u]`.

On an interval crossing zero, the lower bound is the zero line and the upper bound is the secant
through `(l, -l)` and `(u, u)`. The degenerate interval `[0, 0]` is represented by the zero line.
-/
def affAbs (l u : α) : α × α × α × α :=
  if l > Numbers.zero then
    (Numbers.one, Numbers.zero, Numbers.one, Numbers.zero)
  else if u < Numbers.zero then
    (-Numbers.one, Numbers.zero, -Numbers.one, Numbers.zero)
  else
    if u > l then
      let slope := (u + l) / (u - l)
      let bias := u - slope * u
      (Numbers.zero, Numbers.zero, slope, bias)
    else
      (Numbers.zero, Numbers.zero, Numbers.zero, Numbers.zero)

/-! ### Reciprocal -/

/-- Reciprocal: $f(x)=1/x$. -/
def reciprocal (x : α) : α := Numbers.one / x

/-- IBP for reciprocal on boxes, defined only when every coordinate interval excludes zero. -/
def ibpReciprocal? (n : Nat) (B : Box α (.dim n .scalar)) :
    Option (Box α (.dim n .scalar)) :=
  match B.lo, B.hi with
  | .dim lo, .dim hi =>
    if (List.finRange n).all (fun i =>
        match lo i, hi i with
        | .scalar l, .scalar u => l > Numbers.zero || u < Numbers.zero) then
      let outLo := Tensor.dim (fun i =>
        match lo i, hi i with
        | .scalar _l, .scalar u => Tensor.scalar (Numbers.one / u))
      let outHi := Tensor.dim (fun i =>
        match lo i, hi i with
        | .scalar l, .scalar _u => Tensor.scalar (Numbers.one / l))
      some { lo := outLo, hi := outHi }
    else
      none

/-! ### Power -/

/-- Helper for positive integer power. -/
def posPow (base : α) (exp : Nat) : α :=
  match exp with
  | 0 => Numbers.one
  | k + 1 => base * posPow base k

/-- Integer power: $f(x)=x^n$. -/
def powerInt (x : α) (n : Int) : α :=
  if n == 0 then Numbers.one
  else if n > 0 then
    posPow x n.toNat
  else
    -- Negative power: 1/x^|n|
    Numbers.one / posPow x (-n).toNat

/-- IBP for x². -/
def ibpSquareScalar (l u : α) : α × α :=
  let l2 := l * l
  let u2 := u * u
  if l > Numbers.zero then
    (l2, u2)
  else if u < Numbers.zero then
    (u2, l2)
  else
    -- Spans zero
    (Numbers.zero, if l2 > u2 then l2 else u2)

/-- IBP for x² on boxes. -/
def ibpSquare (n : Nat) (B : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match B.lo, B.hi with
  | .dim lo, .dim hi =>
    let outLo := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        Tensor.scalar (ibpSquareScalar l u).1)
    let outHi := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar u =>
        Tensor.scalar (ibpSquareScalar l u).2)
    { lo := outLo, hi := outHi }

/-- Affine bounds for x². -/
def affSquare (l u : α) : α × α × α × α :=
  -- x² is convex, so secant for upper, tangent for lower
  let slope_sec := l + u
  let bias_sec := -(l * u)  -- Secant: y = (l+u)x - lu
  -- Tangent at midpoint
  let mid := (l + u) * Numbers.half
  let slope_tan := Numbers.two * mid
  let bias_tan := -(mid * mid)
  (slope_tan, bias_tan, slope_sec, bias_sec)

/-! ### Min/Max -/

/-- Elementwise minimum of two boxes. -/
def ibpMin (n : Nat) (B1 B2 : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match B1.lo, B1.hi, B2.lo, B2.hi with
  | .dim lo1, .dim hi1, .dim lo2, .dim hi2 =>
    let outLo := Tensor.dim (fun i =>
      match lo1 i, lo2 i with
      | .scalar l1, .scalar l2 =>
        Tensor.scalar (if l1 < l2 then l1 else l2))
    let outHi := Tensor.dim (fun i =>
      match hi1 i, hi2 i with
      | .scalar u1, .scalar u2 =>
        -- max is min of upper bounds
        Tensor.scalar (if u1 < u2 then u1 else u2))
    { lo := outLo, hi := outHi }

/-- Elementwise maximum of two boxes. -/
def ibpMax (n : Nat) (B1 B2 : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match B1.lo, B1.hi, B2.lo, B2.hi with
  | .dim lo1, .dim hi1, .dim lo2, .dim hi2 =>
    let outLo := Tensor.dim (fun i =>
      match lo1 i, lo2 i with
      | .scalar l1, .scalar l2 =>
        -- min is max of lower bounds
        Tensor.scalar (if l1 > l2 then l1 else l2))
    let outHi := Tensor.dim (fun i =>
      match hi1 i, hi2 i with
      | .scalar u1, .scalar u2 =>
        Tensor.scalar (if u1 > u2 then u1 else u2))
    { lo := outLo, hi := outHi }

/--
Clamp one scalar with the same composition used by `Spec.clampSpec`:
`min clamp_hi (max clamp_lo x)`.

In particular, when `clamp_lo > clamp_hi`, the result is `clamp_hi`. This agrees with PyTorch's
documented behavior instead of silently switching the two bounds.
-/
def clampScalar (x clamp_lo clamp_hi : α) : α :=
  let floored := if x > clamp_lo then x else clamp_lo
  if floored < clamp_hi then floored else clamp_hi

/-- Clamp operation: `clamp(x, lo, hi) = min(hi, max(lo, x))`. -/
def ibpClampScalar (x_lo x_hi clamp_lo clamp_hi : α) : α × α :=
  (clampScalar x_lo clamp_lo clamp_hi, clampScalar x_hi clamp_lo clamp_hi)

/-- Interval propagation for `clamp`, applied coordinatewise to a vector box. -/
def ibpClamp (n : Nat) (B : Box α (.dim n .scalar)) (clamp_lo clamp_hi : α) : Box α (.dim n
  .scalar) :=
  match B.lo, B.hi with
  | .dim lo, .dim hi =>
    let outLo := Tensor.dim (fun i =>
      match lo i with
      | .scalar l =>
        Tensor.scalar (clampScalar l clamp_lo clamp_hi))
    let outHi := Tensor.dim (fun i =>
      match hi i with
      | .scalar u =>
        Tensor.scalar (clampScalar u clamp_lo clamp_hi))
    { lo := outLo, hi := outHi }

end NN.MLTheory.CROWN.Operators.Arithmetic
