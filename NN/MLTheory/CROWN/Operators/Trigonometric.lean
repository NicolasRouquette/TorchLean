/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Core
public import NN.MLTheory.CROWN.Runtime.Ops
public import NN.Spec.Core.Context
public import NN.Spec.Core.Tensor
public import NN.Spec.Core.TensorOps

/-!
# Trigonometric operator bounds (optional)

This file provides simple IBP and affine transfer rules for trigonometric functions used in
physics-informed objectives (PINNs): `sin`, `cos`, `tan`, and `atan`.

This module is not imported by the default `NN.MLTheory.CROWN.Operators` index because `tan` and
`atan` require an extra scalar-function interface. Import it explicitly when needed.

Note: `tan`/`atan` are not part of `Context` today. The corresponding transfer rules below are
therefore gated behind an extra `TanAtan α` typeclass.

## Soundness status

The transfer rules in this file are **optional** and have different proof status by operator:

- `sin`/`cos` delegate to the conservative 1-Lipschitz enclosure rules in
  `NN.MLTheory.CROWN.Runtime.Ops.IBP.sin/cos`. This avoids endpoint-only periodic reasoning, which
  can miss internal extrema.
- `tan`/`atan` require the extra `TanAtan α` class. Their endpoint rules are named with an explicit
  caller-side monotonicity assumption; for `tan`, the interval must additionally avoid poles.
- Affine transfer rules here are executable relaxation candidates. A downstream theorem should
  provide or assume the corresponding enclosure property for the scalar backend being used.

## References

- PINNs: Raissi, Perdikaris, Karniadakis, "Physics-informed neural networks", JCP 2019:
  https://arxiv.org/abs/1711.10561
- CROWN/DeepPoly context: Zhang et al., 2018 (CROWN) https://arxiv.org/abs/1811.00866
- Lipschitz enclosure idea for `sin`/`cos`:
  $\left|\frac{d}{dx}\sin x\right|\le 1$ and
  $\left|\frac{d}{dx}\cos x\right|\le 1$, so a ball or interval can be mapped using a
  Lipschitz radius (as implemented in `Runtime/Ops`).
-/

@[expose] public section


namespace NN.MLTheory.CROWN.Operators.Trigonometric

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN

variable {α : Type} [Context α]

/-- Extra trigonometric functions outside the base `Spec.MathFunctions` / `Context` interface. -/
class TanAtan (α : Type) where
  /-- Tangent. -/
  tan : α → α
  /-- Arctangent. -/
  atan : α → α

/-- IBP for `sin`.

This optional operator delegates to the conservative 1-Lipschitz enclosure used by the runtime graph
verifier. We use that rule instead of endpoint-only periodic reasoning, since endpoint checks alone
can miss internal extrema without exact period tracking.
-/
def ibpSin {n : Nat} (xB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  NN.MLTheory.CROWN.Runtime.Ops.IBP.sin xB

/-- IBP for `cos`.

This optional operator delegates to the same conservative 1-Lipschitz enclosure as the runtime graph
verifier.
-/
def ibpCos {n : Nat} (xB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  NN.MLTheory.CROWN.Runtime.Ops.IBP.cos xB

/-- IBP for `tan` under the caller-side monotonic-branch precondition.

This rule is intended for intervals contained in a single branch of `tan` and away from poles. If a
workflow may cross a pole, it should reject the certificate or provide a different enclosure rule.
-/
def ibpTanAssumingMonotoneBranch {n : Nat} [TanAtan α]
    (xB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match xB.lo, xB.hi with
  | .dim lo, .dim hi =>
    let outLo := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar l, .scalar _u =>
        -- In monotonic region, tan(l) is the lower bound
        Tensor.scalar (TanAtan.tan l))
    let outHi := Tensor.dim (fun i =>
      match lo i, hi i with
      | .scalar _l, .scalar u =>
        -- In monotonic region, tan(u) is the upper bound
        Tensor.scalar (TanAtan.tan u))
    { lo := outLo, hi := outHi }

/--
Endpoint interval propagation for `atan`, under the caller-side assumption that the supplied
`TanAtan.atan` implementation is monotone on every input interval.
-/
def ibpAtanAssumingMonotone {n : Nat} [TanAtan α]
    (xB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match xB.lo, xB.hi with
  | .dim lo, .dim hi =>
    let outLo := Tensor.dim (fun i =>
      match lo i with
      | .scalar l => Tensor.scalar (TanAtan.atan l))
    let outHi := Tensor.dim (fun i =>
      match hi i with
      | .scalar u => Tensor.scalar (TanAtan.atan u))
    { lo := outLo, hi := outHi }

/-- Minimum of two scalar endpoints using the executable order from `Context`. -/
def scalarMin (x y : α) : α :=
  if x < y then x else y

/-- Maximum of two scalar endpoints using the executable order from `Context`. -/
def scalarMax (x y : α) : α :=
  if x > y then x else y

/-- Multiply two scalar intervals using all four endpoint products. -/
def scalarIntervalMul (aLo aHi bLo bHi : α) : α × α :=
  let p1 := aLo * bLo
  let p2 := aLo * bHi
  let p3 := aHi * bLo
  let p4 := aHi * bHi
  (scalarMin (scalarMin p1 p2) (scalarMin p3 p4),
    scalarMax (scalarMax p1 p2) (scalarMax p3 p4))

/-- Square a scalar interval. -/
def scalarIntervalSquare (lo hi : α) : α × α :=
  let lo2 := lo * lo
  let hi2 := hi * hi
  if lo < Numbers.zero then
    if hi > Numbers.zero then
      (Numbers.zero, scalarMax lo2 hi2)
    else
      (hi2, lo2)
  else
    (lo2, hi2)

/--
Derivative propagation for `sin` using the global enclosure `cos(x) ∈ [-1, 1]`.

This deliberately avoids endpoint-only trigonometric bounds, which are unsound whenever an interval
contains an interior extremum.
-/
def derivSin {n : Nat} (_xB : Box α (.dim n .scalar))
    (dB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match dB.lo, dB.hi with
  | .dim dlo, .dim dhi =>
    let outLo := Tensor.dim (fun i =>
      match dlo i, dhi i with
      | .scalar dl, .scalar dh =>
        Tensor.scalar (scalarIntervalMul (-Numbers.one) Numbers.one dl dh).1)
    let outHi := Tensor.dim (fun i =>
      match dlo i, dhi i with
      | .scalar dl, .scalar dh =>
        Tensor.scalar (scalarIntervalMul (-Numbers.one) Numbers.one dl dh).2)
    { lo := outLo, hi := outHi }

/-- Derivative propagation for `cos` using the global enclosure `-sin(x) ∈ [-1, 1]`. -/
def derivCos {n : Nat} (_xB : Box α (.dim n .scalar))
    (dB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match dB.lo, dB.hi with
  | .dim dlo, .dim dhi =>
    let outLo := Tensor.dim (fun i =>
      match dlo i, dhi i with
      | .scalar dl, .scalar dh =>
        Tensor.scalar (scalarIntervalMul (-Numbers.one) Numbers.one dl dh).1)
    let outHi := Tensor.dim (fun i =>
      match dlo i, dhi i with
      | .scalar dl, .scalar dh =>
        Tensor.scalar (scalarIntervalMul (-Numbers.one) Numbers.one dl dh).2)
    { lo := outLo, hi := outHi }

/--
Derivative bounds for arctangent under the standard ordered-field, absolute-value, and arctangent
laws. The `TanAtan` class supplies executable functions but does not itself prove these laws, so the
assumption is kept in the declaration name.
-/
def derivAtanAssumingStandardLaws {n : Nat} (xB : Box α (.dim n .scalar))
    (dB : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  match xB.lo, xB.hi, dB.lo, dB.hi with
  | .dim xlo, .dim xhi, .dim dlo, .dim dhi =>
    let outLo := Tensor.dim (fun i =>
      match xlo i, xhi i, dlo i, dhi i with
      | .scalar xl, .scalar xu, .scalar dl, .scalar dh =>
        -- d(atan)/dx = 1/(1+x²), decreasing in |x|
        -- For interval [xl, xu], max derivative at x closest to 0
        let x_abs_max := if MathFunctions.abs xl > MathFunctions.abs xu
                         then MathFunctions.abs xl
                         else MathFunctions.abs xu
        let x_abs_min := if xl < Numbers.zero then
                           if xu > Numbers.zero then Numbers.zero
                           else MathFunctions.abs xu
                         else xl
        let d_lo := Numbers.one / (Numbers.one + x_abs_max * x_abs_max)
        let d_hi := Numbers.one / (Numbers.one + x_abs_min * x_abs_min)
        -- Multiply by input derivatives
        let p1 := d_lo * dl
        let p2 := d_lo * dh
        let p3 := d_hi * dl
        let p4 := d_hi * dh
        let m1 := if p1 < p2 then p1 else p2
        let m2 := if p3 < p4 then p3 else p4
        Tensor.scalar (if m1 < m2 then m1 else m2))
    let outHi := Tensor.dim (fun i =>
      match xlo i, xhi i, dlo i, dhi i with
      | .scalar xl, .scalar xu, .scalar dl, .scalar dh =>
        let x_abs_max := if MathFunctions.abs xl > MathFunctions.abs xu
                         then MathFunctions.abs xl
                         else MathFunctions.abs xu
        let x_abs_min := if xl < Numbers.zero then
                           if xu > Numbers.zero then Numbers.zero
                           else MathFunctions.abs xu
                         else xl
        let d_lo := Numbers.one / (Numbers.one + x_abs_max * x_abs_max)
        let d_hi := Numbers.one / (Numbers.one + x_abs_min * x_abs_min)
        let p1 := d_lo * dl
        let p2 := d_lo * dh
        let p3 := d_hi * dl
        let p4 := d_hi * dh
        let M1 := if p1 > p2 then p1 else p2
        let M2 := if p3 > p4 then p3 else p4
        Tensor.scalar (if M1 > M2 then M1 else M2))
    { lo := outLo, hi := outHi }

/--
Second-derivative propagation for `sin` using global `[-1,1]` enclosures for both `cos` and
`-sin`, together with full interval multiplication.
-/
def secondDerivSin {n : Nat} (_xB : Box α (.dim n .scalar))
    (dB d2B : Box α (.dim n .scalar)) : Box α (.dim n .scalar) :=
  -- y'' = f''(x) * (dx)² + f'(x) * d²x
  -- where f'(x) = cos(x) and f''(x) = -sin(x)
  match dB.lo, dB.hi, d2B.lo, d2B.hi with
  | .dim dlo, .dim dhi, .dim d2lo, .dim d2hi =>
    let outLo := Tensor.dim (fun i =>
      match dlo i, dhi i, d2lo i, d2hi i with
      | .scalar dl, .scalar dh, .scalar d2l, .scalar d2h =>
        let dxSq := scalarIntervalSquare dl dh
        let first := scalarIntervalMul (-Numbers.one) Numbers.one dxSq.1 dxSq.2
        let second := scalarIntervalMul (-Numbers.one) Numbers.one d2l d2h
        Tensor.scalar (first.1 + second.1))
    let outHi := Tensor.dim (fun i =>
      match dlo i, dhi i, d2lo i, d2hi i with
      | .scalar dl, .scalar dh, .scalar d2l, .scalar d2h =>
        let dxSq := scalarIntervalSquare dl dh
        let first := scalarIntervalMul (-Numbers.one) Numbers.one dxSq.1 dxSq.2
        let second := scalarIntervalMul (-Numbers.one) Numbers.one d2l d2h
        Tensor.scalar (first.2 + second.2))
    { lo := outLo, hi := outHi }

end NN.MLTheory.CROWN.Operators.Trigonometric
