/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.Runtime.Autograd.Engine.Core.Base

/-!
Shape-changing eager-engine operations.

This module implements reshape, transpose, broadcast, slice, gather/scatter, and related view-style
nodes while preserving the graph metadata needed by autograd.
-/

@[expose] public section

namespace Runtime
namespace Autograd

open Spec
open Tensor

namespace Tape

/--
Flatten a tensor `s` into a 1D vector of length `Spec.Shape.size s`.

PyTorch comparison: `torch.flatten(x)` with `start_dim=0`.
-/
def flatten {α : Type} [Inhabited α] [DecidableEq Shape] {s : Shape}
  (t : Tape α) (xId : Nat) : Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := .dim (Spec.Shape.size s) .scalar)
    "flatten" xId
    (forward := fun x => flattenSpec (α := α) x)
    (backward := fun _x dLdz => unflattenSpec (α := α) s dLdz)

/--
Reshape a tensor while preserving number of elements.

The proof argument `h` enforces `Spec.Shape.size s₁ = Spec.Shape.size s₂`.
PyTorch comparison: `x.reshape(new_shape)` / `x.view(new_shape)` (when valid).
-/
def reshape {α : Type} [Inhabited α] [DecidableEq Shape] {s₁ s₂ : Shape}
  (t : Tape α) (xId : Nat) (h : Spec.Shape.size s₁ = Spec.Shape.size s₂) : Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := s₁) (τ := s₂)
    "reshape" xId
    (forward := fun x => reshapeSpec (α := α) (s₁ := s₁) (s₂ := s₂) x h)
    (backward := fun _x dLdz => reshapeSpec (α := α) (s₁ := s₂) (s₂ := s₁) dLdz h.symm)

/-- Transpose a 2D matrix. PyTorch: `x.t()` / `x.transpose(0,1)`. -/
def transpose2d {α : Type} [DecidableEq Shape] {m n : Nat}
  (t : Tape α) (xId : Nat) : Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := .dim m (.dim n .scalar)) (τ := .dim n (.dim m .scalar))
    "transpose2d" xId
    (forward := fun x => matrixTransposeSpec (α := α) x)
    (backward := fun _x dLdz => matrixTransposeSpec (α := α) dLdz)

/-- Permute a 3D tensor `(a,b,c) → (b,c,a)`. PyTorch: `x.permute(1,2,0)`. -/
def transpose3dFirstToLast {α : Type} [DecidableEq Shape] {a b c : Nat}
  (t : Tape α) (xId : Nat) : Result (Tape α × Nat) :=
  unary (α := α) (t := t)
    (σ := .dim a (.dim b (.dim c .scalar)))
    (τ := .dim b (.dim c (.dim a .scalar)))
    "transpose3d_first_to_last" xId
    (forward := fun x => Spec.Tensor.transpose3DFirstToLastSpec (α := α) (a := a) (b := b) (c
      := c) x)
    (backward := fun _x dLdz =>
      Spec.Tensor.transpose3DLastToFirstSpec (α := α) (a := b) (b := c) (c := a) dLdz)

/-- Permute a 3D tensor `(a,b,c) → (c,a,b)`. PyTorch: `x.permute(2,0,1)`. -/
def transpose3dLastToFirst {α : Type} [DecidableEq Shape] {a b c : Nat}
  (t : Tape α) (xId : Nat) : Result (Tape α × Nat) :=
  unary (α := α) (t := t)
    (σ := .dim a (.dim b (.dim c .scalar)))
    (τ := .dim c (.dim a (.dim b .scalar)))
    "transpose3d_last_to_first" xId
    (forward := fun x => Spec.Tensor.transpose3DLastToFirstSpec (α := α) (a := a) (b := b) (c
      := c) x)
    (backward := fun _x dLdz =>
      Spec.Tensor.transpose3DFirstToLastSpec (α := α) (a := c) (b := a) (c := b) dLdz)

/-- Swap the last two axes of a 3D tensor `(a,b,c) → (a,c,b)`. PyTorch: `x.transpose(1,2)`. -/
def transpose3dLastTwo {α : Type} [DecidableEq Shape] {a b c : Nat}
  (t : Tape α) (xId : Nat) : Result (Tape α × Nat) :=
  unary (α := α) (t := t)
    (σ := .dim a (.dim b (.dim c .scalar)))
    (τ := .dim a (.dim c (.dim b .scalar)))
    "transpose3d_last_two" xId
    (forward := fun x => Spec.Tensor.transpose3DLastTwoSpec (α := α) (a := a) (b := b) (c := c)
      x)
    (backward := fun _x dLdz =>
      Spec.Tensor.transpose3DLastTwoSpec (α := α) (a := a) (b := c) (c := b) dLdz)

/--
Swap adjacent axes at a given depth inside a general `Shape`.

This is a more general analogue of `transpose` operations.
-/
def swapAdjacentAtDepth {α : Type} [DecidableEq Shape] {s : Shape}
  (t : Tape α) (depth : Nat) (xId : Nat) : Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := s.swapAdjacentAtDepth depth)
    "swapAdjacentAtDepth" xId
    (forward := fun x => Spec.Tensor.swapAdjacentAxes (tensor := x) depth)
    (backward := fun _x dLdz =>
      let dx' := Spec.Tensor.swapAdjacentAxes (tensor := dLdz) depth
      Tensor.castShape dx' (by simpa using (Spec.Shape.swapAdjacentAtDepth_involutive s depth)))

/--
Broadcast `x : s₁` to `s₂` using a proof `Shape.CanBroadcastTo s₁ s₂`.

PyTorch comparison: implicit broadcasting / `x.expand(...)`.
-/
def broadcastTo {α : Type} [Inhabited α] [Add α] [Zero α] [DecidableEq Shape]
  {s₁ s₂ : Shape} (cb : Shape.CanBroadcastTo s₁ s₂) (t : Tape α) (xId : Nat) :
  Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := s₁) (τ := s₂)
    "broadcastTo" xId
    (forward := fun x => Spec.Tensor.broadcastTo (α := α) cb x)
    (backward := fun _x dLdz => Spec.Tensor.reduceFromBroadcastTo (α := α) cb dLdz)

/--
Sum-reduce along `axis`.

PyTorch comparison: `torch.sum(x, dim=axis)`.
-/
def reduceSum {α : Type} [Add α] [Zero α] [Inhabited α] [DecidableEq Shape]
  {s : Shape} (axis : Nat) [_valid : Shape.HasNonemptyAxis axis s]
  [_wf : Shape.WellFormed s]
  (t : Tape α) (xId : Nat) : Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := shapeAfterSum s axis)
    s!"reduce_sum(axis={axis})" xId
    (forward := fun x => Spec.Tensor.reduceSum (α := α) (s := s) axis x _valid.proof)
    (backward := fun _x dLdz => Spec.Tensor.broadcastAfterSum s axis dLdz)

/--
Mean-reduce along `axis`.

Backward rule: broadcast the upstream cotangent back to `s` and divide by the reduced dimension.
PyTorch comparison: `torch.mean(x, dim=axis)`.
-/
def reduceMean {α : Type} [Context α] [DecidableEq Shape]
  {s : Shape} (axis : Nat) [valid : Shape.HasNonemptyAxis axis s] [_wf : Shape.WellFormed s]
  (t : Tape α) (xId : Nat) : Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := shapeAfterSum s axis)
    s!"reduce_mean(axis={axis})" xId
    (forward := fun x =>
      let h := valid.proof
      Spec.Tensor.reduceMean (α := α) (s := s) axis x h)
    (backward := fun _x dLdz =>
      let dLdx := Spec.Tensor.broadcastAfterSum s axis dLdz
      letI : Shape.AxisInBounds axis s := valid.proof.toAxisInBounds
      let denomNat := Shape.axisSize s axis
      Spec.Tensor.scaleSpec (α := α) (s := s) dLdx (1 / (denomNat : α)))
