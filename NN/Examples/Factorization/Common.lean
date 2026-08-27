/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Factorizations
public import NN.Spec.Core.TensorReductionShape.LinearAlgebra

/-!
# Shared Factorization Helpers

Small `Float`-valued helpers used by the Cholesky and QR examples. Each example reconstructs the
original matrix from its factors and checks the numerical error during the build.
-/

@[expose] public section

namespace NN.Examples.Factorization

/-- Build an `m × n` matrix from row-major arrays. Missing entries are zero. -/
def mkMat {m n : Nat} (rows : Array (Array Float)) : Spec.Tensor Float [m, n] :=
  Spec.Tensor.matrix fun i j =>
    (rows[i.val]?.bind fun row => row[j.val]?).getD 0.0

/-- Maximum entrywise absolute difference between two matrices. -/
def maxMatErr {m n : Nat} (A B : Spec.Tensor Float [m, n]) : Float :=
  (List.finRange m).foldl (fun acc i =>
    (List.finRange n).foldl
      (fun a j => max a (Float.abs (Spec.get2 A i j - Spec.get2 B i j))) acc) 0.0

/-- Matrix product $AB$. -/
def matmul {m n p : Nat} (A : Spec.Tensor Float [m, n])
    (B : Spec.Tensor Float [n, p]) : Spec.Tensor Float [m, p] :=
  Spec.Tensor.matmulSpec (Spec.Shape.CanBroadcastTo.refl .scalar)
    (Spec.Shape.CanBroadcastTo.refl .scalar) A B

/-- Matrix transpose. -/
def tr {m n : Nat} (A : Spec.Tensor Float [m, n]) : Spec.Tensor Float [n, m] :=
  Spec.Tensor.swapAdjacentAxes A 0

/-- Materialize a vector for display. -/
def vectorToArray {n : Nat} (v : Spec.Tensor Float [n]) : Array Float :=
  Array.ofFn fun i => Spec.Tensor.item (Spec.get v i)

/-- Squared Frobenius distance between two matrices. -/
def frobSqErr {m n : Nat} (A B : Spec.Tensor Float [m, n]) : Float :=
  (List.finRange m).foldl (fun acc i =>
    (List.finRange n).foldl
      (fun a j => let d := Spec.get2 A i j - Spec.get2 B i j; a + d * d) acc) 0.0

/-- Shared tolerance for reconstruction-error assertions. -/
def tol : Float := 1e-6

/-- Fail unless `err` is below `tolerance`. -/
def assertLt (name : String) (err : Float) (tolerance : Float := tol) : IO Unit :=
  if err < tolerance then
    IO.println s!"{name}: OK (err = {err})"
  else
    throw (IO.userError s!"{name}: FAIL (err = {err} ≥ tol = {tolerance})")

/-- Fail unless `err` is at least `threshold`. -/
def assertGe (name : String) (err : Float) (threshold : Float := 0.5) : IO Unit :=
  if err ≥ threshold then
    IO.println s!"{name}: OK (correctly rejected, err = {err} ≥ {threshold})"
  else
    throw (IO.userError s!"{name}: FAIL (err = {err} < {threshold}; expected the property to fail)")

/-- Fail when a reconstruction unexpectedly succeeds. -/
def assertReconFails (name : String) (err : Float) (tolerance : Float := tol) : IO Unit :=
  if err < tolerance then
    throw (IO.userError s!"{name}: FAIL (unexpectedly reconstructed, err = {err} < {tolerance})")
  else
    IO.println s!"{name}: OK (correctly failed, err = {err})"

end NN.Examples.Factorization
