/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.Runtime.Autograd.Engine.Core.Indexing

/-!
Elementwise eager-engine operations.

This file contains scalar-lifted tensor nodes and their runtime/autograd implementation, including
arithmetic, comparisons, activations, and loss-adjacent pointwise operations.
-/

@[expose] public section

namespace Runtime
namespace Autograd

open Spec
open Tensor

namespace Tape

/-- Elementwise addition. PyTorch: `torch.add` / `+`. -/
def add {α : Type} [Add α] [DecidableEq Shape] {s : Shape}
  (t : Tape α) (aId bId : Nat) : Result (Tape α × Nat) := do
  let a ← requireValue (α:=α) (t:=t) (s:=s) aId
  let b ← requireValue (α:=α) (t:=t) (s:=s) bId
  let y := addSpec a b
  let node : Node α :=
    { name := some "add"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        pure [(aId, AnyTensor.mk dLdy), (bId, AnyTensor.mk dLdy)]
    }
  pure (t.addNode node)

/-- Elementwise subtraction. PyTorch: `torch.sub` / `-`. -/
def sub {α : Type} [Sub α] [Zero α] [DecidableEq Shape] {s : Shape}
  (t : Tape α) (aId bId : Nat) : Result (Tape α × Nat) := do
  let a ← requireValue (α:=α) (t:=t) (s:=s) aId
  let b ← requireValue (α:=α) (t:=t) (s:=s) bId
  let y := subSpec a b
  let node : Node α :=
    { name := some "sub"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let neg_dLdy : Tensor α s := subSpec (fill (0 : α) s) dLdy
        pure [(aId, AnyTensor.mk dLdy), (bId, AnyTensor.mk neg_dLdy)]
    }
  pure (t.addNode node)

/-- Elementwise multiplication. PyTorch: `torch.mul` / `*`. -/
def mul {α : Type} [Mul α] [DecidableEq Shape] {s : Shape}
  (t : Tape α) (aId bId : Nat) : Result (Tape α × Nat) := do
  let a ← requireValue (α:=α) (t:=t) (s:=s) aId
  let b ← requireValue (α:=α) (t:=t) (s:=s) bId
  let y := mulSpec a b
  let node : Node α :=
    { name := some "mul"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let da : Tensor α s := mulSpec dLdy b
        let db : Tensor α s := mulSpec dLdy a
        pure [(aId, AnyTensor.mk da), (bId, AnyTensor.mk db)]
    }
  pure (t.addNode node)

/-- Elementwise division. PyTorch: `torch.div` / `/`. Backward is the ordinary quotient
rule, valid for nonzero denominators: `∂(a/b)/∂a = 1/b`, `∂(a/b)/∂b = −a/b²` (mirrors the
CUDA `div` node; negation reuses `subSpec (fill 0)` as `sub` does, so no `Neg α` is required).

Domain: real calculus does not define the derivative of `a/b` at `b = 0`, so this backward is
the genuine quotient rule only where `b ≠ 0`. The carrier's `/` (and hence `divSpec`) may
totalize or be backend-dependent at `b = 0`, but no real-valued gradient is implied there.

Requires `[Context α]` like the sibling `abs`/`sqrt`/`exp` nodes (its `divSpec` forward rides
the carrier's `/`). -/
def div {α : Type} [Context α] [DecidableEq Shape] {s : Shape}
  (t : Tape α) (aId bId : Nat) : Result (Tape α × Nat) := do
  let a ← requireValue (α:=α) (t:=t) (s:=s) aId
  let b ← requireValue (α:=α) (t:=t) (s:=s) bId
  let y := divSpec a b
  let node : Node α :=
    { name := some "div"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let da : Tensor α s := divSpec dLdy b
        let dLdyA : Tensor α s := mulSpec dLdy (divSpec a (mulSpec b b))
        let db : Tensor α s := subSpec (fill (0 : α) s) dLdyA
        pure [(aId, AnyTensor.mk da), (bId, AnyTensor.mk db)]
    }
  pure (t.addNode node)

/-- Multiply a tensor by a scalar constant. PyTorch: `x * c` for Python scalar `c`. -/
def scale {α : Type} [Mul α] [DecidableEq Shape] {s : Shape}
  (t : Tape α) (xId : Nat) (c : α) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y := scaleSpec x c
  let node : Node α :=
    { name := some "scale"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        pure [(xId, AnyTensor.mk (scaleSpec dLdy c))]
    }
  pure (t.addNode node)

/--
Elementwise absolute value.

Backward uses the sign function (`sign_spec`) as a subgradient at `0`.
PyTorch comparison: `torch.abs`.
-/
def abs {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := s)
    "abs" xId
    (forward := fun x => absSpec (α := α) (s := s) x)
    (backward := fun x dLdy =>
      let dabs : Tensor α s := signSpec (α := α) (s := s) x
      mulSpec dabs dLdy)

/--
Elementwise square root.

Backward uses `1 / (2 * sqrt(x))` for `x > 0` and `0` otherwise (totalized).
PyTorch comparison: `torch.sqrt`.
-/
def sqrt {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := s)
    "sqrt" xId
    (forward := fun x => sqrtSpec (α := α) (s := s) x)
    (backward := fun x dLdy =>
      let dsqrt : Tensor α s :=
        mapSpec (α := α) (s := s) (fun v =>
          if v > 0 then
            (1 : α) / (((2 : Nat) : α) * MathFunctions.sqrt v)
          else
            (0 : α)) x
      mulSpec dsqrt dLdy)

/--
Elementwise clamp to `[minVal, maxVal]`.

Backward multiplies by an indicator of the open interval `(minVal, maxVal)` (zero at boundaries).
PyTorch comparison: `torch.clamp`.
-/
def clamp {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (xId : Nat) (minVal maxVal : α) : Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := s)
    "clamp" xId
    (forward := fun x => clampSpec (α := α) (s := s) x minVal maxVal)
    (backward := fun x dLdy =>
      let dclamp : Tensor α s :=
        mapSpec (α := α) (s := s) (fun v =>
          if v > minVal ∧ maxVal > v then (1 : α) else (0 : α)) x
      mulSpec dclamp dLdy)

/--
Elementwise maximum.

Tie-breaking: when `a = b`, the upstream gradient is split evenly (`0.5`) between both inputs.
PyTorch comparison: `torch.maximum`.
-/
def max {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (aId bId : Nat) : Result (Tape α × Nat) := do
  let a ← requireValue (α:=α) (t:=t) (s:=s) aId
  let b ← requireValue (α:=α) (t:=t) (s:=s) bId
  let y := maxSpec (α := α) (s := s) a b
  let node : Node α :=
    { name := some "max"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let half : α := (1 : α) / ((2 : Nat) : α)
        let maskA : Tensor α s :=
          map2Spec (α := α) (β := α) (γ := α) (s := s) (fun x y =>
            if x > y then (1 : α) else if y > x then (0 : α) else half) a b
        let maskB : Tensor α s :=
          map2Spec (α := α) (β := α) (γ := α) (s := s) (fun x y =>
            if y > x then (1 : α) else if x > y then (0 : α) else half) a b
        pure [
          (aId, AnyTensor.mk (mulSpec maskA dLdy)),
          (bId, AnyTensor.mk (mulSpec maskB dLdy))
        ]
    }
  pure (t.addNode node)

/--
Elementwise minimum.

Tie-breaking: when `a = b`, the upstream gradient is split evenly (`0.5`) between both inputs.
PyTorch comparison: `torch.minimum`.
-/
def min {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (aId bId : Nat) : Result (Tape α × Nat) := do
  let a ← requireValue (α:=α) (t:=t) (s:=s) aId
  let b ← requireValue (α:=α) (t:=t) (s:=s) bId
  let y := minSpec (α := α) (s := s) a b
  let node : Node α :=
    { name := some "min"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let half : α := (1 : α) / ((2 : Nat) : α)
        let maskA : Tensor α s :=
          map2Spec (α := α) (β := α) (γ := α) (s := s) (fun x y =>
            if y > x then (1 : α) else if x > y then (0 : α) else half) a b
        let maskB : Tensor α s :=
          map2Spec (α := α) (β := α) (γ := α) (s := s) (fun x y =>
            if x > y then (1 : α) else if y > x then (0 : α) else half) a b
        pure [
          (aId, AnyTensor.mk (mulSpec maskA dLdy)),
          (bId, AnyTensor.mk (mulSpec maskB dLdy))
        ]
    }
  pure (t.addNode node)

/--
Elementwise ReLU.

PyTorch comparison: `torch.relu(x)` / `torch.nn.functional.relu(x)`.
-/
def relu {α : Type}
  [Mul α] [Zero α] [Max α] [One α] [LT α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {s : Shape} (t : Tape α) (xId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=s) xId
  let y := Activation.reluSpec (α:=α) x
  let node : Node α :=
    { name := some "relu"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := s) dLdyAny
        let drelu := Activation.reluDerivSpec (α:=α) x
        pure [(xId, AnyTensor.mk (mulSpec drelu dLdy))]
    }
  pure (t.addNode node)
