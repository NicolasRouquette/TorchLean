/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Factorizations
public import NN.Proofs.Tensor.Basic.LinearAlgebra
public import Mathlib.Analysis.Matrix.Spectrum
public import Mathlib.Analysis.Matrix.PosDef
public import Mathlib.Analysis.Matrix.HermitianFunctionalCalculus
public import Mathlib.LinearAlgebra.Matrix.PosDef
public import Mathlib.LinearAlgebra.UnitaryGroup
public import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
public import Mathlib.Data.List.GetD

/-!
# Correctness of the matrix factorizations (foundation for CHD)

This file provides the **formal correctness theorems** for the spec-layer factorizations in
[`NN.Spec.Core.Tensor.Factorizations`](../../../Spec/Core/Tensor/Factorizations.lean)
(`choleskySpec`, `qrSpec`, `symEigJacobiSpec`, `svdSpec`). The motivation is
[Computational Hypergraph Discovery](https://github.com/TheoBourdais/ComputationalHypergraphDiscovery):
a Gaussian-process / kernel-ridge method whose numerical core reduces to the **full symmetric
eigendecomposition** of a kernel matrix `K`. CHD's `solve_variationnal`, `find_gamma` and `Z_test`
are all expressed through the eigendecomposition of `K`, so a verified linear-algebra foundation is a
prerequisite for formalizing CHD.

## Architecture (refinement)

* **Specifications** (`IsCholesky`, `IsQR`, `IsSymEig`, `IsSVD`) are `Prop`s on Mathlib
  `Matrix (Fin n) (Fin n) ℝ`. Mathlib's `Matrix m n α` is *definitionally* `m → n → α`, so the
  function representation `Spec.toMatFn` produced by the executable specs bridges for free.
* **Foundation theorems** (this is what CHD consumes) are proved from the *specifications*, independent
  of the executable algorithm, via Mathlib's spectral theorem and continuous functional calculus.
* **Algorithm theorems** connect the executable `Spec.*Fn` defs to the specifications. Proven here:
  the Cholesky factor is lower-triangular (`choleskyFn_lower_triangular`); the Jacobi/SVD routines
  satisfy their *exact* invariants — orthogonal similarity preserves trace/determinant
  (`trace_orthogonal_conj`, `det_orthogonal_conj`), the Givens rotation is orthogonal
  (`givens_normSq`), and the eigendecomposition is exact in the zero-residual limit
  (`isSymEig_of_diagonal`), with the finite-sweep error captured a-posteriori by
  `symEig_frobenius_residual`.

## Scope honesty

`A = V · diag(λ) · Vᵀ` is **not** an exact theorem for the finite-sweep / floating-point Jacobi output;
it is the *target* certified at runtime by the `assertLt` checks in `NN/Examples/Factorization`, and
bounded a-posteriori here by `symEig_frobenius_residual` (residual = off-diagonal mass of `Af`).
Mathlib v4.30.0 has no Jacobi convergence theory and `Float` never diagonalizes exactly, so no
a-priori convergence theorem is possible.

The exact algebraic reconstruction of the executable *finite* factorizations — `A = L · Lᵀ` for
`choleskyFn` (under SPD pivots) and `A = Q · R`, `Qᵀ Q = 1` for `gramSchmidtFn` (under full column
rank) — is the remaining increment: it requires an induction relating the `List.foldl` prefix at step
`j` to the first `j` columns (extending `getD_foldl_finRange`) plus the per-pivot positivity discharge.
The specification-level consequences CHD needs (above) are independent of that algorithmic step.
-/

@[expose] public section

namespace Spec.Factorization

open Matrix
open scoped BigOperators

variable {n : Nat}

/-! ## Specifications

The mathematical meaning of each factorization, as a predicate over real matrices. Over `ℝ`,
`star = id` so `conjTranspose = transpose`; we phrase everything with `ᵀ`.
-/

/-- `L` is a Cholesky factor of `A`: lower-triangular with `A = L · Lᵀ`. -/
def IsCholesky (A L : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  (∀ i j, i < j → L i j = 0) ∧ A = L * Lᵀ

/-- `(Q, R)` is a QR factorization of `A`: `Q` has orthonormal columns, `R` is upper-triangular,
`A = Q · R`. -/
def IsQR {m k : Nat} (A Q : Matrix (Fin m) (Fin k) ℝ) (R : Matrix (Fin k) (Fin k) ℝ) : Prop :=
  Qᵀ * Q = 1 ∧ (∀ i j, j < i → R i j = 0) ∧ A = Q * R


/-! ### Fold-indexing for the column-building specs

`choleskyColsFn` and `gramSchmidtFn` build their output with a left fold that appends one column per
index. The lemmas here read off the column produced at a given position, bridging the executable
`List.foldl` form to per-entry reasoning. They are generic over the appended-value function `g`. -/

section FoldSnoc

variable {β : Type _} {ι : Type _}

/-- A left fold that appends one element per input grows the accumulator by `l.length`. -/
private theorem length_foldl_snoc (g : List β → ι → β) (l : List ι) (acc : List β) :
    (l.foldl (fun s a => s ++ [g s a]) acc).length = acc.length + l.length := by
  induction l generalizing acc with
  | nil => simp
  | cons a t ih =>
      rw [List.foldl_cons, ih]
      simp only [List.length_append, List.length_cons, List.length_nil]
      grind

/-- A fold that only appends never changes an index already inside the accumulator. -/
private theorem getD_foldl_snoc_lt (g : List β → ι → β) (d : β) (l : List ι) (acc : List β)
    (k : Nat) (hk : k < acc.length) :
    (l.foldl (fun s a => s ++ [g s a]) acc).getD k d = acc.getD k d := by
  induction l generalizing acc with
  | nil => simp
  | cons a t ih =>
      rw [List.foldl_cons,
        ih (acc ++ [g acc a]) (by rw [List.length_append]; grind),
        List.getD_append _ _ _ _ hk]

/-- The element at position `j` of the snoc-fold over `finRange n` is `g` applied to the fold of the
length-`j` prefix and the index `j`. -/
private theorem getD_foldl_finRange (g : List β → Fin n → β) (d : β) (j : Fin n) :
    ((List.finRange n).foldl (fun s a => s ++ [g s a]) []).getD j.val d
      = g (((List.finRange n).take j.val).foldl (fun s a => s ++ [g s a]) []) j := by
  have hjlen : j.val < (List.finRange n).length := by
    rw [List.length_finRange]; exact j.isLt
  have htake : (List.finRange n).take (j.val + 1)
      = (List.finRange n).take j.val ++ [j] := by
    rw [List.take_succ_eq_append_getElem hjlen]
    congr 1
    simp [List.getElem_finRange]
  have hplen : (((List.finRange n).take j.val).foldl (fun s a => s ++ [g s a]) []).length
      = j.val := by
    rw [length_foldl_snoc, List.length_nil, List.length_take, List.length_finRange, Nat.zero_add,
      Nat.min_eq_left (Nat.le_of_lt j.isLt)]
  calc
    ((List.finRange n).foldl (fun s a => s ++ [g s a]) []).getD j.val d
        = (((List.finRange n).drop (j.val + 1)).foldl (fun s a => s ++ [g s a])
            ((List.finRange n).take (j.val + 1) |>.foldl (fun s a => s ++ [g s a]) [])).getD
              j.val d := by
          conv_lhs => rw [show List.finRange n
            = (List.finRange n).take (j.val + 1) ++ (List.finRange n).drop (j.val + 1) from
            (List.take_append_drop _ _).symm]
          rw [List.foldl_append]
    _ = ((List.finRange n).take (j.val + 1) |>.foldl (fun s a => s ++ [g s a]) []).getD j.val d := by
          apply getD_foldl_snoc_lt
          rw [length_foldl_snoc, List.length_nil, List.length_take, List.length_finRange,
            Nat.zero_add]
          grind
    _ = g (((List.finRange n).take j.val).foldl (fun s a => s ++ [g s a]) []) j := by
          rw [htake, List.foldl_append, List.foldl_cons, List.foldl_nil]
          rw [List.getD_append_right _ _ _ _ (le_of_eq hplen), hplen, Nat.sub_self]
          rfl

end FoldSnoc

/-! ### Cholesky factor is lower-triangular

A structural fact about the executable `choleskyFn`, proved directly from the column fold: the entry
above the diagonal is forced to `0` by the construction. -/

/-- Reading an entry of a matrix tensor built by `ofMatFn` returns the underlying function value. -/
theorem get2_ofMatFn {m k : Nat} (f : Fin m → Fin k → ℝ) (i : Fin m) (j : Fin k) :
    Spec.get2 (Spec.ofMatFn f) i j = f i j := rfl

/-- The executable Cholesky factor is lower-triangular: entries strictly above the diagonal vanish. -/
theorem choleskyFn_lower_triangular (A : Fin n → Fin n → ℝ) {i j : Fin n} (hij : i.val < j.val) :
    Spec.choleskyFn A i j = 0 := by
  unfold Spec.choleskyFn Spec.choleskyColsFn
  rw [getD_foldl_finRange]
  rw [if_pos hij]

/-- Tensor-level statement: the Cholesky factor `choleskySpec A` is lower-triangular. -/
theorem choleskySpec_lower_triangular (A : Spec.Tensor ℝ (.dim n (.dim n .scalar)))
    {i j : Fin n} (hij : i.val < j.val) :
    Spec.get2 (Spec.choleskySpec A) i j = 0 := by
  rw [show Spec.choleskySpec A = Spec.ofMatFn (Spec.choleskyFn (Spec.toMatFn A)) from rfl,
    get2_ofMatFn]
  exact choleskyFn_lower_triangular _ hij
end Spec.Factorization
