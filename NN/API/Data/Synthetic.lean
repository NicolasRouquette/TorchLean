/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Tensor

import Mathlib.Algebra.Order.Algebra

/-!
# Synthetic Data

This module provides deterministic tabular grids used by examples and tests. Domain-specific
datasets and sample packing belong in their respective data modules.
-/

@[expose] public section


namespace TorchLean.Data
namespace Synthetic

open Spec

/-! ## Tabular grids -/

/--
Cartesian product of two vectors (batched tensor of points).

`cartesianGrid xs ys` produces a tensor `X : (m*n, 2)` containing all pairs `(x, y)` with:
- `x` taken from `xs : (m,)`
- `y` taken from `ys : (n,)`

Ordering is row-major: for each `x` in `xs` (outer loop), we sweep all `y` in `ys` (inner loop).

PyTorch analogue: `torch.cartesian_prod(xs, ys)` (up to shape).
-/
def cartesianGrid {α : Type} [Zero α] {m n : Nat}
    (xs : Spec.Tensor α (.dim m .scalar)) (ys : Spec.Tensor α (.dim n .scalar)) :
    Spec.Tensor α (.dim (m * n) (.dim 2 .scalar)) :=
  Spec.Tensor.dim (fun ij =>
    let i : Fin m := ij.divNat (m := m) (n := n)
    let j : Fin n := ij.modNat (m := m) (n := n)
    let x : α := _root_.Spec.Tensor.item (_root_.Spec.get xs i)
    let y : α := _root_.Spec.Tensor.item (_root_.Spec.get ys j)
    Spec.Tensor.dim (fun k =>
      Spec.Tensor.scalar <|
        match k.val with
        | 0 => x
        | 1 => y
        | _ => 0))

/--
Linearly spaced points including endpoints.

`linspace lo hi count` returns a vector tensor of shape `(count,)`:
- empty if `count = 0`
- `[lo]` if `count = 1`
- otherwise `count` points from `lo` to `hi` (inclusive).

PyTorch analogue: `torch.linspace`.
-/
def linspace {α : Type} [Context α] (lo hi : α) (count : Nat) :
    Spec.Tensor α (.dim count .scalar) :=
  match count with
  | 0 => Spec.Tensor.dim (fun i => nomatch i)
  | 1 => Spec.Tensor.dim (fun _ => Spec.Tensor.scalar lo)
  | n + 2 =>
      let denom : α := (n + 1 : Nat)
      Spec.Tensor.dim (fun i =>
        let t : α := (i.1 : Nat) / denom
        Spec.Tensor.scalar (lo + t * (hi - lo)))

/-- Rectangular grid over `[xLo, xHi] x [yLo, yHi]`. -/
def rectangularGrid {α : Type} [Context α] (xLo xHi yLo yHi : α) (xCount yCount : Nat) :
    Spec.Tensor α (.dim (xCount * yCount) (.dim 2 .scalar)) :=
  cartesianGrid (linspace xLo xHi xCount) (linspace yLo yHi yCount)

/-- Square grid over `[lo, hi] x [lo, hi]`. -/
def squareGrid {α : Type} [Context α] (lo hi : α) (count : Nat) :
    Spec.Tensor α (.dim (count * count) (.dim 2 .scalar)) :=
  rectangularGrid lo hi lo hi count count

end Synthetic
end TorchLean.Data
