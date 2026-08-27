/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape

/-!
# Tensor Numerical Algorithms

Reference implementations shared by classical models, graph specifications, and runtime checks:

- matrix minors, determinants, inverses, and power iteration;
- distances between vectors;
- vector normalization.

## Intent / tradeoffs

These definitions prioritize:
- **mathematical clarity**, and
- **shape safety** (via `Spec.Tensor`),
over performance.

In particular, `determinantSpec` uses Laplace expansion, which is exponentially expensive and is
only meant for small matrices (for example, 2 x 2 or 3 x 3) and proof-oriented reference code. For
large-scale linear algebra, use the runtime layer with array-backed kernels.
-/

@[expose] public section


namespace Spec

open Tensor

variable {α : Type} [Context α]

-- Matrix operations used by classical model specifications.

/-- The index in `Fin n` corresponding to `i : Fin (n - 1)` after skipping one position. -/
lemma minorIndex_lt {n : ℕ} (skip : Fin n) (i : Fin (n - 1)) :
    (if i.val < skip.val then i.val else i.val + 1) < n := by
  by_cases h : i.val < skip.val <;> simp [h]
  · exact Nat.lt_of_lt_of_le i.isLt (Nat.pred_le n)
  · grind

/--
Matrix minor: delete `row` and `col` from an `n × n` matrix, producing an `(n-1) × (n-1)` matrix.

This is used by `determinantSpec` (Laplace expansion) and the adjugate-based inverse below.
-/
def matrixMinorSpec {α : Type} {n : Nat}
    (matrix : Tensor α [n, n])
    (row col : Fin n) :
    Tensor α [n - 1, n - 1] :=
  Tensor.dim (fun i =>
    Tensor.dim (fun j =>
      let actualI := if i.val < row.val then i.val else i.val + 1
      let actualJ := if j.val < col.val then j.val else j.val + 1
      Tensor.scalar (
        get2 matrix
          ⟨actualI, minorIndex_lt row i⟩
          ⟨actualJ, minorIndex_lt col j⟩
      )
    )
  )


/--
Determinant of an `n × n` matrix (spec-level reference implementation).

This uses Laplace expansion (cofactor expansion) along the first row, with special-cased base cases
for `n = 0, 1, 2`. It is mathematically clear but exponentially slow, so it is intended only for
very small `n` and/or proof-oriented reference code.
-/
def determinantSpec {α : Type} [Context α] :
  ∀ {n : Nat}, Tensor α [n, n] → Tensor α .scalar
| 0, _ => Tensor.scalar 1
| 1, A =>
  match A with
  | Tensor.dim rows =>
    match rows ⟨0, Nat.zero_lt_succ 0⟩ with
    | Tensor.dim cols =>
      match cols ⟨0, Nat.zero_lt_succ 0⟩ with
      | Tensor.scalar val => Tensor.scalar val
| 2, A =>
  match A with
  | Tensor.dim rows =>
    match rows ⟨0, Nat.zero_lt_succ 1⟩, rows ⟨1, Nat.succ_lt_succ (Nat.zero_lt_succ 0)⟩ with
    | Tensor.dim row0, Tensor.dim row1 =>
      match row0 ⟨0, Nat.zero_lt_succ 1⟩, row0 ⟨1, Nat.succ_lt_succ (Nat.zero_lt_succ 0)⟩,
            row1 ⟨0, Nat.zero_lt_succ 1⟩, row1 ⟨1, Nat.succ_lt_succ (Nat.zero_lt_succ 0)⟩ with
      | Tensor.scalar a, Tensor.scalar b, Tensor.scalar c, Tensor.scalar d =>
        Tensor.scalar (a * d - b * c)
| n+2, A =>
  let laplaceTerm (j : Fin (n+2)) :=
    let minor := matrixMinorSpec A ⟨0, Nat.zero_lt_succ (n + 1)⟩ j
    let cofactor := if j.val % 2 = 0 then 1 else Numbers.negOne
    let element := get2 A ⟨0, Nat.zero_lt_succ (n+1)⟩ j
    cofactor * element * Tensor.item (determinantSpec minor)
  let sum := (List.finRange (n+2)).foldl (fun acc j => acc + laplaceTerm j) 0
  Tensor.scalar sum


-- Matrix inverse (implemented using adjugate method)
/--
Matrix inverse via the adjugate formula (spec-level reference implementation).

The result is `none` when the determinant is zero. Returning an unrelated matrix for a singular
input would make downstream statistical formulas appear defined when they are not.

PyTorch analogue: `torch.linalg.inv`, with failure represented explicitly by `Option`.
-/
def inverseSpec? {n : Nat}
  (matrix : Tensor α [n, n]) :
  Option (Tensor α [n, n]) :=
  let det := Tensor.item (determinantSpec matrix)
  if det == 0 then
    none
  else
    -- Compute the cofactor matrix `C` and then transpose it to get the adjugate `adj(A) = Cᵀ`.
    --
    -- Note: the transpose matters. Without it you'd get the cofactor matrix, not the adjugate.
    let cofactors :=
      Tensor.dim (fun i =>
        Tensor.dim (fun j =>
          let cofactor := if (i.val + j.val) % 2 = 0 then 1 else -1
          let minor := matrixMinorSpec matrix i j
          let minorDet := Tensor.item (determinantSpec minor)
          Tensor.scalar (cofactor * minorDet)))
    let adjugate := swapAdjacentAxes cofactors 0
    -- Scale by 1/det
    some (scaleSpec adjugate (1 / det))

/--
Approximate the leading eigenpair by a caller-selected number of power-iteration steps.

The scalar is the final Rayleigh quotient and the tensor is the corresponding normalized iterate.
This definition does not claim to compute a full eigendecomposition. Convergence to a dominant
eigenvector requires the usual spectral assumptions on `matrix` and a suitable initial vector.
-/
def powerIterationLeadingEigenpairSpec {n : Nat}
    (matrix : Tensor α [n, n]) (iterations : Nat) :
    α × Tensor α [n] :=
  -- Power iteration: returns normalized eigenvector and its Rayleigh quotient
  let rec powerIteration (v : Tensor α [n]) (iter : Nat) :
    (Tensor α [n] × α) :=
    if iter = 0 then
      let Av := matVecMulSpec matrix v
      let eigenvalue := dotSpec v Av
      (v, eigenvalue)
    else
      let Av := matVecMulSpec matrix v
      let norm := MathFunctions.sqrt (sumSpec (squareSpec Av))
      let normalized := if norm > 0 then
        Tensor.dim (fun i =>
          match get Av i with
          | Tensor.scalar val => Tensor.scalar (val / norm)
        )
      else v
      powerIteration normalized (iter - 1)

  -- Start from a normalized all-ones vector.  Normalizing before the first multiplication matters
  -- for the zero matrix: every direction is then an eigenvector, and the fallback branch below
  -- should still return a unit vector rather than the raw all-ones vector.
  let initialRaw : Tensor α [n] := Tensor.dim (fun _ => Tensor.scalar 1)
  let initialNorm := MathFunctions.sqrt (sumSpec (squareSpec initialRaw))
  let initialTensor :=
    if initialNorm > 0 then scaleSpec initialRaw (1 / initialNorm) else initialRaw
  let (eigenvector, eigenvalue) := powerIteration initialTensor iterations
  (eigenvalue, eigenvector)


-- Distance functions used by nearest-neighbor, clustering, and metric-learning specs.

/--
Euclidean (L2) distance between two feature vectors.

PyTorch analogue: `torch.linalg.vector_norm(x - y)` or `torch.cdist` (batched).
-/
def euclideanDistanceSpec {nFeatures : Nat}
  (x y : Tensor α [nFeatures]) : α :=
  let diff := subSpec x y
  let squaredDiff := squareSpec diff
  let sumSquared := sumSpec squaredDiff
  MathFunctions.sqrt sumSquared

/-- Squared Euclidean distance (avoids the final square root). -/
def squaredEuclideanDistanceSpec {nFeatures : Nat}
  (x y : Tensor α [nFeatures]) : α :=
  let diff := subSpec x y
  let squaredDiff := squareSpec diff
  sumSpec squaredDiff

/-- Manhattan (L1) distance between two feature vectors. -/
def manhattanDistanceSpec {nFeatures : Nat}
  (x y : Tensor α [nFeatures]) : α :=
  let diff := subSpec x y
  let absDiff := mapSpec MathFunctions.abs diff
  sumSpec absDiff

/--
Cosine distance `1 - cos(theta)` between two feature vectors.

If either vector has zero norm, this returns `1`.
-/
def cosineDistanceSpec {nFeatures : Nat}
  (x y : Tensor α [nFeatures]) : α :=
  let dotProduct := dotSpec x y
  let normX := MathFunctions.sqrt (sumSpec (squareSpec x))
  let normY := MathFunctions.sqrt (sumSpec (squareSpec y))
  let denominator := normX * normY
  if denominator == 0 then 1 else 1 - (dotProduct / denominator)

/--
Minkowski distance of order `p` between two feature vectors.

This generalizes L1 (Manhattan) and L2 (Euclidean). The explicit positivity hypothesis rules out
the undefined order-zero and negative-order cases.
-/
def minkowskiDistanceSpec {nFeatures : Nat}
  (p : α) (_hp : p > 0) (x y : Tensor α [nFeatures]) : α :=
  let diff := subSpec x y
  let absDiff := mapSpec MathFunctions.abs diff
  let powered := mapSpec (fun a => a ^ p) absDiff
  let sumPowered := sumSpec powered
  sumPowered ^ (1 / p)

-- Normalization and utility functions shared by model specifications.

/--
Divide a vector by its sum when that sum is positive.

If the sum is not positive, this returns the uniform vector. When the input entries are
nonnegative, the positive-sum branch is a probability distribution.

PyTorch analogue: `probs / probs.sum()` (with an explicit zero-sum guard).
-/
def normalizeByPositiveSumSpec {n : Nat} (values : Tensor α [n]) :
  Tensor α [n] :=
  let total := sumSpec values
  if total > 0 then
    Tensor.dim (fun i =>
      match get values i with
      | Tensor.scalar p => Tensor.scalar (p / total)
    )
  else
    Tensor.dim (fun _ => Tensor.scalar (1 / n))

/--
L2-normalize a vector.

If the norm is `0`, this returns the input unchanged.
-/
def normalizeL2Spec {n : Nat} (vector : Tensor α [n]) :
  Tensor α [n] :=
  let norm := MathFunctions.sqrt (sumSpec (squareSpec vector))
  if norm > 0 then
    Tensor.dim (fun i =>
      match get vector i with
      | Tensor.scalar v => Tensor.scalar (v / norm)
    )
  else
    vector

/--
L2-normalize a vector with an additive regularizer under the square root.

The denominator is

`sqrt (sumᵢ vector[i] ^ 2 + regularizer)`.

Unlike `normalizeL2Spec`, this operation has no zero-norm branch. Callers are responsible for
choosing a regularizer that makes the denominator meaningful in their scalar context. This is the
normalization convention used by several attention and recurrent architectures, where the exact
regularizer is part of the model specification.
-/
def normalizeL2RegularizedSpec {n : Nat}
    (vector : Tensor α [n]) (regularizer : α) :
    Tensor α [n] :=
  let norm := MathFunctions.sqrt (sumSpec (squareSpec vector) + regularizer)
  Tensor.mapSpec (fun value => value / norm) vector

/--
Z-score normalization: subtract mean and divide by standard deviation.

If the standard deviation is `0`, this returns the mean-centered vector.
-/
def normalizeZscoreSpec {n : Nat} (vector : Tensor α [n]) :
  Tensor α [n] :=
  let mean := sumSpec vector / n
  let centered := Tensor.dim (fun i =>
    match get vector i with
    | Tensor.scalar v => Tensor.scalar (v - mean)
  )
  let variance := sumSpec (squareSpec centered) / n
  let std := MathFunctions.sqrt variance
  if std > 0 then
    Tensor.dim (fun i =>
      match get centered i with
      | Tensor.scalar v => Tensor.scalar (v / std)
    )
  else
    centered

end Spec
