/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Macros
public import NN.API.Tensor
public import NN.API.TensorPack
public import NN.API.Runtime

import Mathlib.Algebra.Order.Algebra

/-!
# Synthetic Data

This module provides deterministic data generators used by examples and tests:
- 2D tabular grids (`cartesianGrid`, `linspace`)
- simple regression/classification sample builders

These helpers are in-memory. They keep example code focused on models and verification rather than
data-loading infrastructure.

Domain-specific datasets belong in their own modules. The band-classification example, for
instance, lives in `TorchLean.Data.Bands`; this file contains only shape-independent generators.
-/

@[expose] public section


namespace TorchLean.Data
namespace Synthetic

open Spec

/-! ## Tabular 2D -/

/-- Affine function `w₁x₁ + w₂x₂ + b` for synthetic regression data. -/
def affinePlane (w₁ w₂ b x₁ x₂ : Float) : Float :=
  w₁ * x₁ + w₂ * x₂ + b

/-- A length-1 float vector tensor. -/
def singletonVectorFloat (y : Float) : Spec.Tensor Float (.dim 1 .scalar) :=
  Spec.Tensor.dim (fun _ => Spec.Tensor.scalar y)

/-- A length-2 float vector tensor. -/
def pointVectorFloat (x y : Float) : Spec.Tensor Float (.dim 2 .scalar) :=
  Spec.Tensor.dim (fun i =>
    Spec.Tensor.scalar <|
      match i.val with
      | 0 => x
      | 1 => y
      | _ => 0.0)

/--
Cartesian product of two float vectors (batched tensor of points).

`cartesianGrid xs ys` produces a tensor `X : (m*n, 2)` containing all pairs `(x, y)` with:
- `x` taken from `xs : (m,)`
- `y` taken from `ys : (n,)`

Ordering is row-major: for each `x` in `xs` (outer loop), we sweep all `y` in `ys` (inner loop).

PyTorch analogue: `torch.cartesian_prod(xs, ys)` (up to shape).
-/
def cartesianGrid {m n : Nat}
    (xs : Spec.Tensor Float (.dim m .scalar)) (ys : Spec.Tensor Float (.dim n .scalar)) :
    Spec.Tensor Float (.dim (m * n) (.dim 2 .scalar)) :=
  Spec.Tensor.dim (fun ij =>
    let i : Fin m := ij.divNat (m := m) (n := n)
    let j : Fin n := ij.modNat (m := m) (n := n)
    let x : Float := _root_.Spec.Tensor.toScalar (_root_.Spec.get xs i)
    let y : Float := _root_.Spec.Tensor.toScalar (_root_.Spec.get ys j)
    pointVectorFloat x y)

/--
Linearly spaced points including endpoints.

`linspace lo hi count` returns a vector tensor of shape `(count,)`:
- empty if `count = 0`
- `[lo]` if `count = 1`
- otherwise `count` points from `lo` to `hi` (inclusive).

PyTorch analogue: `torch.linspace`.
-/
def linspace (lo hi : Float) (count : Nat) : Spec.Tensor Float (.dim count .scalar) :=
  match count with
  | 0 => Spec.Tensor.dim (fun i => nomatch i)
  | 1 => Spec.Tensor.dim (fun _ => Spec.Tensor.scalar lo)
  | n + 2 =>
      let denom := Float.ofNat (n + 1)
      Spec.Tensor.dim (fun i =>
        let t := (Float.ofNat i.1) / denom
        Spec.Tensor.scalar (lo + t * (hi - lo)))

/-- Rectangular grid over `[xLo, xHi] x [yLo, yHi]`. -/
def rectangularGrid (xLo xHi yLo yHi : Float) (xCount yCount : Nat) :
    Spec.Tensor Float (.dim (xCount * yCount) (.dim 2 .scalar)) :=
  cartesianGrid (linspace xLo xHi xCount) (linspace yLo yHi yCount)

/-- Square grid over `[lo, hi] x [lo, hi]`. -/
def squareGrid (lo hi : Float) (count : Nat) :
    Spec.Tensor Float (.dim (count * count) (.dim 2 .scalar)) :=
  rectangularGrid lo hi lo hi count count

/--
Compute 2D→1D regression targets for a batched grid.

Input `X` has shape `(n,2)` and the output `Y` has shape `(n,1)`.
-/
def regressionTargetsFloat {n : Nat} (X : Spec.Tensor Float (.dim n (.dim 2 .scalar)))
    (f : Float → Float → Float) : Spec.Tensor Float (.dim n (.dim 1 .scalar)) :=
  Spec.Tensor.dim (fun i =>
    let x : Spec.Tensor Float (.dim 2 .scalar) := Spec.get X i
    let firstCoord : Float := _root_.Spec.Tensor.toScalar (_root_.Spec.get x ⟨0, by decide⟩)
    let secondCoord : Float := _root_.Spec.Tensor.toScalar (_root_.Spec.get x ⟨1, by decide⟩)
    singletonVectorFloat (f firstCoord secondCoord))

/-- Casted version of `singletonVectorFloat` under an arbitrary scalar semantics `α`. -/
def singletonVector {α : Type} [Context α] (cast : Float → α) (y : Float) : Spec.Tensor α (.dim 1 .scalar) :=
  TorchLean.Tensor.castFloat cast (singletonVectorFloat y)

/-- Casted version of `pointVectorFloat` under an arbitrary scalar semantics `α`. -/
def pointVector {α : Type} [Context α] (cast : Float → α) (x y : Float) : Spec.Tensor α (.dim 2 .scalar)
  :=
  TorchLean.Tensor.castFloat cast (pointVectorFloat x y)

/-! ## Labels and Packing -/

/-- One-hot encode a label as a float vector of shape `Vec classes`. -/
def oneHotFloat (classes label : Nat) : Spec.Tensor Float (.dim classes .scalar) :=
  NN.Tensor.oneHotNat (α := Float) classes label

/-- Casted version of `oneHotFloat`. -/
def oneHot {α : Type} [Context α] (cast : Float → α) (classes label : Nat) :
    Spec.Tensor α (.dim classes .scalar) :=
  TorchLean.Tensor.castFloat cast (oneHotFloat classes label)

/--
Convert `(x, label)` pairs into `(x, oneHot(label))` pairs.

This is a pure preprocessing step that keeps the data in-memory.
-/
def classification {α : Type} [Context α] {σ : Spec.Shape}
    (cast : Float → α) (classes : Nat) (xs : List (Spec.Tensor Float σ × Nat)) :
    List (Spec.Tensor α σ × Spec.Tensor α (.dim classes .scalar)) :=
  xs.map (fun (xF, label) =>
    (TorchLean.Tensor.castFloat cast xF, oneHot cast classes label))

/--
Pack `(x, y)` tensor pairs into TorchLean supervised tensor-pack samples.

This is the common sample representation used by the training helpers.
-/
def supervised {α : Type} [Context α] {σ τ : Spec.Shape}
    (cast : Float → α) (xs : List (Spec.Tensor Float σ × Spec.Tensor Float τ)) :
    List (_root_.TorchLean.TensorPack α [σ, τ]) :=
  xs.map (fun (xF, yF) =>
    tensorpack! (TorchLean.Tensor.castFloat cast xF), (TorchLean.Tensor.castFloat cast yF))

/-- Convert `(x, label)` pairs into TorchLean tensor-pack samples with one-hot targets. -/
def labeled {α : Type} [Context α] {σ : Spec.Shape}
    (cast : Float → α) (classes : Nat) (xs : List (Spec.Tensor Float σ × Nat)) :
    List (_root_.TorchLean.TensorPack α [σ, .dim classes .scalar]) :=
  (classification (α := α) (σ := σ) cast classes xs).map (fun (x, y) =>
    tensorpack! x, y)

end Synthetic
end TorchLean.Data
