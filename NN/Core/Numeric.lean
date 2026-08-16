/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.Order.Group.Unbundled.Abs
public import Mathlib.Analysis.Complex.Exponential
public import Mathlib.Analysis.Complex.Trigonometric
public import Mathlib.Analysis.Real.Sqrt
public import Mathlib.Analysis.SpecialFunctions.Log.Basic
public import Mathlib.Analysis.SpecialFunctions.Pow.Real
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
public import Mathlib.Data.Real.Basic

/-!
# Foundational Numeric Interfaces

This module contains the small scalar interfaces shared by TorchLean's floating-point library and
its tensor specifications. It deliberately knows nothing about tensors, models, runtimes, or
verification.

`MathFunctions` names the transcendental operations used by numerical code. `Numbers` collects the
few non-integral constants needed by scalar-polymorphic model definitions. The broader neural-model
interface, `Context`, lives in `NN.Spec.Core.Context` and extends these foundations.
-/

@[expose] public section

/-- Scalar transcendental functions shared by numerical and model code. -/
class MathFunctions (α : Type) where
  exp : α → α
  tanh : α → α
  cosh : α → α
  sqrt : α → α
  abs : α → α
  log : α → α
  pi : α
  cos : α → α
  sin : α → α
  sinh : α → α

/-- Common scalar constants used by scalar-polymorphic model definitions. -/
class Numbers (α : Type) where
  /-- Backend representation of `-0.5`. -/
  neg_point_five : α
  /-- Backend representation of `-1`. -/
  neg_one : α
  /-- Backend representation of `0.1`. -/
  pointone : α
  /-- Backend representation of `0.5`. -/
  pointfive : α
  one : α
  zero : α
  /-- Backend representation of `2`. -/
  two : α
  /-- Backend representation of `3`. -/
  three : α
  /-- Backend representation of `4`. -/
  four : α
  /-- Backend representation of `5`. -/
  five : α
  /-- Backend representation of `10`. -/
  ten : α
  /-- Backend representation of the natural logarithm of `10`. -/
  log10 : α
  /-- Backend representation of the natural logarithm of `10000`. -/
  log10000 : α
  /-- Backend-supplied tolerance for numerically guarded formulas. -/
  epsilon : α

/-- Host implementations of the scalar transcendental interface. -/
instance : MathFunctions Float where
  exp := Float.exp
  tanh := Float.tanh
  cosh := Float.cosh
  sqrt := Float.sqrt
  abs := Float.abs
  log := Float.log
  pi := 3.14159265358979323846
  cos := Float.cos
  sin := Float.sin
  sinh := Float.sinh

/-- Native binary32 implementations of the scalar transcendental interface. -/
instance : MathFunctions Float32 where
  exp := Float32.exp
  tanh := Float32.tanh
  cosh := Float32.cosh
  sqrt := Float32.sqrt
  abs := Float32.abs
  log := Float32.log
  pi := (3.14159265358979323846 : Float).toFloat32
  cos := Float32.cos
  sin := Float32.sin
  sinh := Float32.sinh

/-- Exact-real interpretations of the scalar transcendental interface. -/
noncomputable instance : MathFunctions ℝ where
  exp := Real.exp
  tanh := Real.tanh
  cosh := Real.cosh
  sinh := Real.sinh
  sqrt := Real.sqrt
  abs := fun x => |x|
  log := Real.log
  pi := Real.pi
  cos := Real.cos
  sin := Real.sin

/-- Constants for Lean's host `Float`. -/
instance : Numbers Float where
  neg_point_five := -0.5
  neg_one := -1
  pointone := 0.1
  pointfive := 0.5
  zero := 0
  one := 1
  two := 2
  three := 3
  four := 4
  five := 5
  ten := 10
  log10 := Float.log 10
  log10000 := Float.log 10000
  epsilon := 1e-6

/-- Constants rounded once to native binary32. -/
instance : Numbers Float32 where
  neg_point_five := (-0.5 : Float).toFloat32
  neg_one := (-1.0 : Float).toFloat32
  pointone := (0.1 : Float).toFloat32
  pointfive := (0.5 : Float).toFloat32
  zero := 0
  one := 1
  two := 2
  three := 3
  four := 4
  five := 5
  ten := 10
  log10 := Float32.log 10
  log10000 := Float32.log 10000
  epsilon := (1e-6 : Float).toFloat32

/-- Constants for exact-real specifications. -/
noncomputable instance : Numbers ℝ where
  neg_point_five := -0.5
  neg_one := -1
  pointone := 0.1
  pointfive := 0.5
  zero := 0
  one := 1
  two := 2
  three := 3
  four := 4
  five := 5
  ten := 10
  log10 := Real.log 10
  log10000 := Real.log 10000
  epsilon := 1e-6

/-- Coerce naturals into Lean's host `Float`. -/
instance : Coe Nat Float where
  coe := Float.ofNat

/-- Coerce naturals into native binary32. -/
instance : Coe Nat Float32 where
  coe n := (Float.ofNat n).toFloat32

/-- Coerce naturals into `ℝ`. -/
instance : Coe Nat ℝ where
  coe n := (n : ℕ)

/-- Coerce naturals into `ℚ`. -/
instance : Coe Nat ℚ where
  coe n := (n : Nat)
