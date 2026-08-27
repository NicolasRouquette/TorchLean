/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor

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
    (xs : Tensor α [m]) (ys : Tensor α [n]) :
    Tensor α [m * n, 2] :=
  Tensor.stack 0 (fun ij =>
    let i : Fin m := ij.divNat (m := m) (n := n)
    let j : Fin n := ij.modNat (m := m) (n := n)
    let x : α := Tensor.item (Tensor.get xs i)
    let y : α := Tensor.item (Tensor.get ys j)
    Tensor.stack 0 (fun k =>
      Tensor.full [] <|
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
    Tensor α [count] :=
  match count with
  | 0 => by
      simpa [Shape.insertAxis] using
        Tensor.stack (α := α) (count := 0) (shape := Spec.Shape.scalar) 0 Fin.elim0
  | 1 => Tensor.repeatAxis 0 1 (Tensor.full [] lo)
  | n + 2 =>
      let denom : α := (n + 1 : Nat)
      Tensor.stack 0 (fun i =>
        let t : α := (i.1 : Nat) / denom
        Tensor.full [] (lo + t * (hi - lo)))

/-- Square grid over `[lo, hi] x [lo, hi]`. -/
def squareGrid {α : Type} [Context α] (lo hi : α) (count : Nat) :
    Tensor α [count * count, 2] :=
  let axis := linspace lo hi count
  cartesianGrid axis axis

end Synthetic
end TorchLean.Data
