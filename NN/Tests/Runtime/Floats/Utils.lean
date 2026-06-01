/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Lean.Data.Json
public import NN.Spec.Core.Tensor
public import NN.Entrypoint.Tensor
public import NN.Runtime.External.Process
public import Std

/-!
# Floats Utils

Shared helpers for the Float runtime checks.

These helpers keep the curated test files focused on the checked behavior instead of re-declaring
the same tensor accessors and approximate equality checks.
-/

@[expose] public section


open Spec
open Tensor
open Lean

namespace Tests
namespace Floats
namespace Utils

/-- Approximate equality for runtime checks over `Float`. -/
def assertApprox (msg : String) (x y : Float) (tol : Float := 1e-5) : IO Unit := do
  if Float.abs (x - y) > tol then
    throw <| IO.userError s!"{msg}: got {x}, expected {y} (tol={tol})"

/-- Reject `NaN` and infinities in a runtime check. -/
def assertFinite (msg : String) (x : Float) : IO Unit := do
  if x.isNaN || x.isInf then
    throw <| IO.userError s!"{msg}: expected finite, got {x}"

/-- Check that a value lies in `[0, 1]` up to a small tolerance. -/
def assertIn01 (msg : String) (x : Float) : IO Unit := do
  if x < -1e-6 || x > 1.0 + 1e-6 then
    throw <| IO.userError s!"{msg}: expected in [0,1], got {x}"

/-- Approximate equality for same-length float arrays. -/
def assertArrayApprox (label : String) (got expected : Array Float) (tol : Float := 2e-5) :
    IO Unit := do
  unless got.size = expected.size do
    throw (IO.userError s!"{label}: length mismatch {got.size} vs {expected.size}")
  for i in [0:got.size] do
    assertApprox s!"{label}[{i}]" got[i]! expected[i]! tol

/-- Parse a JSON field containing a flat array of floats. -/
def jsonFloatArrayField (j : Json) (key : String) : Except String (Array Float) := do
  let arr ←
    match ← j.getObjVal? key with
    | .arr xs => pure xs
    | other => throw s!"field `{key}` was not an array: {other}"
  arr.mapM fun
    | .num n => pure n.toFloat
    | other => throw s!"field `{key}` contained non-number: {other}"

/-- Whether the active Python environment can import PyTorch. -/
def pythonHasTorch : IO Bool := do
  Runtime.External.Process.pythonCanImport #["torch"]

/-- Read the scalar payload from a scalar tensor. -/
def scalarVal (t : Tensor Float Shape.scalar) : Float :=
  match t with
  | Tensor.scalar v => v

/-- Read one coordinate from a vector tensor. -/
def vecVal {n : Nat} (t : Tensor Float (.dim n .scalar)) (i : Fin n) : Float :=
  match t with
  | Tensor.dim f =>
      match f i with
      | Tensor.scalar v => v

/-- Read one coordinate from a matrix tensor. -/
def matVal {rows cols : Nat} (t : Tensor Float (.dim rows (.dim cols .scalar)))
    (i : Fin rows) (j : Fin cols) : Float :=
  match t with
  | Tensor.dim f =>
      match f i with
      | Tensor.dim g =>
          match g j with
          | Tensor.scalar v => v

/-- Read one coordinate from a `C × H × W` tensor. -/
def chwVal {c h w : Nat} (t : Tensor Float (.dim c (.dim h (.dim w .scalar))))
    (ci : Fin c) (hi : Fin h) (wi : Fin w) : Float :=
  match t with
  | Tensor.dim f =>
      match f ci with
      | Tensor.dim fh =>
          match fh hi with
          | Tensor.dim fw =>
              match fw wi with
              | Tensor.scalar v => v

/-- Read one coordinate from an `N × C × H × W` tensor. -/
def nchwVal {n c h w : Nat} (t : Tensor Float (.dim n (.dim c (.dim h (.dim w .scalar)))))
    (ni : Fin n) (ci : Fin c) (hi : Fin h) (wi : Fin w) : Float :=
  match t with
  | Tensor.dim fn =>
      match fn ni with
      | Tensor.dim fc =>
          match fc ci with
          | Tensor.dim fh =>
              match fh hi with
              | Tensor.dim fw =>
                  match fw wi with
                  | Tensor.scalar v => v

end Utils
end Floats
end Tests
