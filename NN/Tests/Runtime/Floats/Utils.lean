/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Lean.Data.Json
public import NN.Spec.Core.Tensor
public import NN.Tensor
public import NN.Core.ExternalProcess
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

/-- Check that a value lies in $[0,1]$ up to a small tolerance. -/
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
  TorchLean.External.Process.pythonCanImport #["torch"]

/-- Read the scalar payload from a scalar tensor. -/
def scalarVal (t : Tensor Float Shape.scalar) : Float :=
  match t with
  | Tensor.scalar v => v

/-- Read one coordinate from a vector tensor. -/
def vecVal {n : Nat} (t : Tensor Float [n]) (i : Fin n) : Float :=
  Tensor.getScalar t i

/-- Read one coordinate from a matrix tensor. -/
def matVal {rows cols : Nat} (t : Tensor Float [rows, cols])
    (i : Fin rows) (j : Fin cols) : Float :=
  Tensor.get2 t i j

/-- Read one scalar coordinate from a tensor of arbitrary rank.

This helper is intentionally rank-neutral: tests pass the coordinates they are checking rather
than introducing layout-specific accessors. An invalid coordinate is a test-authoring error and
therefore fails immediately.
-/
def tensorVal {s : Shape} (t : Tensor Float s) (indices : List Nat) : Float :=
  match Spec.getSpec t indices with
  | some value => value
  | none => panic! s!"tensor coordinate {indices} is invalid for shape {repr s}"

end Utils
end Floats
end Tests
