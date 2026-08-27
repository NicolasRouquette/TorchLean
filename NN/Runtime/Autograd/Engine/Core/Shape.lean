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

/--
Swap adjacent axes at a given depth inside a general `Shape`.

General permutations and arbitrary-axis transpose are lowered to this operation.
-/
def swapAdjacentAtDepth {α : Type} [DecidableEq Shape] {s : Shape}
  (t : Tape α) (depth : Nat) (xId : Nat) : Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := s) (τ := s.swapAdjacentAtDepth depth)
    "swapAdjacentAtDepth" xId
    (forward := fun x => Spec.Tensor.swapAdjacentAxes (tensor := x) depth)
    (backward := fun _x dLdz =>
      let dx' := Spec.Tensor.swapAdjacentAxes (tensor := dLdz) depth
      Tensor.castShape dx' (by simp))

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
