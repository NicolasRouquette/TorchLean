/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Factorization.Common
meta import NN.Examples.Factorization.Common

/-!
# Example: generalized symmetric eigenproblem via Cholesky whitening (CCA / dim reduction, S9)

`genSymEigCholeskySpec A B sweeps` solves the **generalized** symmetric eigenproblem
`A·v = λ·B·v` — `A` symmetric, `B` symmetric positive-definite — by *whitening* `B` and reducing to
the standard symmetric eigenproblem the landed `symEigJacobiSpec` already solves:
factor `B = L·Lᵀ`, eigendecompose the whitened `C = L⁻¹·A·L⁻ᵀ = W·diag(λ)·Wᵀ`, then unwhiten
`v = L⁻ᵀ·w`. This is the algebraic core of *canonical-correlation analysis* (the CHD
`dimension_reduction.jl` / `cca.jl` reduction).

The fixture is a genuine **CCA covariance pencil**: with two views' covariance blocks `Σxx`, `Σyy`
(each SPD) and cross-covariance `Σxy`,

* `A = [[0, Σxy], [Σyx, 0]]` (symmetric, bordered), and
* `B = blockdiag(Σxx, Σyy)` (SPD),

the generalized eigenvalues are the *canonical correlations* `±ρ` and the eigenvectors are the
canonical directions. The golden values (`scipy.linalg.eigh(A, B)` / Julia `eigen(A, B)`, LAPACK
`sygv`) are `λ ≈ {±0.39596, ±0.21265}`, so the canonical correlations are `{0.396, 0.213}`.

What the checks exhibit — the precise boundary the reduction is designed to make visible:

* **Positive.** The recovered pairs satisfy the *generalized* eigen-equation `A·V = B·V·diag(λ)` and
  the **whitening guarantee** `Vᵀ·B·V = I` (the eigenvectors are `B`-orthonormal, not
  ordinary-orthonormal); the eigenvalues match the LAPACK golden to `1e-7` despite a completely
  different (Jacobi vs LAPACK) eigensolver underneath; the whitened `C` is symmetric.
* **Negative.** `Vᵀ·V ≠ I` — whitening *trades* plain orthonormality for `B`-orthonormality, the
  whole point of the reduction; the generalized eigenvalues *differ* from the standard eigenvalues of
  `A` alone, so `B` genuinely participates; and an **indefinite** `B` breaks the Cholesky pivot
  (`√(negative)` ⟹ `NaN`), so the SPD hypothesis is necessary, not decorative.

All checks run over `Float`, sorry/admit/omega-free.
-/

@[expose] public section


namespace NN.Examples.Factorization.GenSymEig

/-- Cross-covariance and view covariances of the CCA pencil — bordered symmetric `A`. -/
def A : Spec.Tensor Float (.dim 4 (.dim 4 .scalar)) :=
  mkMat [[0.0, 0.0, 0.6, 0.1],
         [0.0, 0.0, 0.2, 0.5],
         [0.6, 0.2, 0.0, 0.0],
         [0.1, 0.5, 0.0, 0.0]]

/-- `B = blockdiag(Σxx, Σyy)`, SPD: each `2×2` block is a positive-definite covariance. -/
def B : Spec.Tensor Float (.dim 4 (.dim 4 .scalar)) :=
  mkMat [[2.0,  0.3,  0.0,  0.0],
         [0.3,  1.5,  0.0,  0.0],
         [0.0,  0.0,  1.8, -0.4],
         [0.0,  0.0, -0.4,  2.2]]

/-- The `4×4` identity (target for the orthonormality checks). -/
def I4 : Spec.Tensor Float (.dim 4 (.dim 4 .scalar)) := Spec.identityTensorSpec 4

/-- Generalized eigendecomposition `A·v = λ·B·v` via Cholesky whitening (30 sweeps on the whitened C). -/
def ge : Spec.Tensor Float (.dim 4 .scalar) × Spec.Tensor Float (.dim 4 (.dim 4 .scalar)) :=
  Spec.genSymEigCholeskySpec A B 30
/-- The generalized eigenvalues (canonical correlations `±ρ`). -/
def evals : Spec.Tensor Float (.dim 4 .scalar) := ge.1
/-- The generalized eigenvectors (columns), `B`-orthonormal. -/
def Vg : Spec.Tensor Float (.dim 4 (.dim 4 .scalar)) := ge.2

/-! ## The whitened matrix `C = L⁻¹·A·L⁻ᵀ` (the reduction made explicit) -/

/-- The Cholesky factor `L` of `B` (so `B = L·Lᵀ`). -/
def L : Spec.Tensor Float (.dim 4 (.dim 4 .scalar)) := Spec.choleskySpec B
/-- The whitened matrix `C = L⁻¹·A·L⁻ᵀ`, symmetric, with the SAME eigenvalues as the pencil `(A, B)`. -/
def C : Spec.Tensor Float (.dim 4 (.dim 4 .scalar)) :=
  Spec.ofMatFn (Spec.whitenFn (Spec.toMatFn L) (Spec.toMatFn A))

/-! ## Error metrics -/

/-- Generalized eigen-equation residual `‖A·V − B·V·diag(λ)‖_max`. -/
def genResid : Float := maxMatErr (mm A Vg) (mm (mm B Vg) (diagFromVec evals))
/-- Whitening guarantee residual `‖Vᵀ·B·V − I‖_max` (`B`-orthonormality). -/
def bOrthoErr : Float := maxMatErr (mm (mm (tr Vg) B) Vg) I4
/-- Symmetry residual `‖C − Cᵀ‖_max` of the whitened matrix. -/
def cSymErr : Float := maxMatErr C (tr C)
/-- Ordinary-orthonormality residual `‖Vᵀ·V − I‖_max` — LARGE: `V` is `B`-orthonormal, not plain. -/
def plainOrthoErr : Float := maxMatErr (mm (tr Vg) Vg) I4

/-- Sort a `Float` list ascending (via the `Context` order), for set-level eigenvalue comparison. -/
def sortF (xs : List Float) : List Float := xs.mergeSort Spec.leBool
/-- The recovered eigenvalues, sorted ascending (Jacobi returns them in no particular order). -/
def evalsSorted : List Float := sortF (vecToList evals)
/-- LAPACK golden generalized eigenvalues (`Julia eigen(A, B)`, ascending). -/
def goldenEvals : List Float :=
  [-0.39596464934201431, -0.21264898960921266, 0.21264898960921277, 0.39596464934201425]
/-- Max entrywise error between the sorted recovered eigenvalues and the golden. -/
def evalsErr : Float :=
  (List.zip evalsSorted goldenEvals).foldl (fun a (x, g) => max a (Float.abs (x - g))) 0.0

/-- Standard eigenvalues of `A` *alone* (ignoring `B`), sorted — for the negative control. -/
def evalsAonly : List Float := sortF (vecToList (Spec.symEigJacobiSpec A 30).1)
/-- Max gap between the generalized eigenvalues and the standard eigenvalues of `A` alone — LARGE. -/
def bMattersErr : Float :=
  (List.zip evalsSorted evalsAonly).foldl (fun a (x, g) => max a (Float.abs (x - g))) 0.0

/-- The largest generalized eigenvalue = the leading canonical correlation. -/
def canonicalCorr : Float := evalsSorted.foldl max (-1.0)

/-! ## An indefinite `B` (the SPD hypothesis is necessary) -/

/-- A non-SPD `B`: the `(0,0)` covariance entry flipped negative, so the first block is indefinite. -/
def Bbad : Spec.Tensor Float (.dim 4 (.dim 4 .scalar)) :=
  mkMat [[-2.0,  0.3,  0.0,  0.0],
         [ 0.3,  1.5,  0.0,  0.0],
         [ 0.0,  0.0,  1.8, -0.4],
         [ 0.0,  0.0, -0.4,  2.2]]
/-- The Cholesky pivot `L[0,0] = √(B[0,0])` for the indefinite `B`: `√(−2)` ⟹ `NaN`. -/
def badPivot : Float := Spec.get2 (Spec.choleskySpec Bbad) ⟨0, by decide⟩ ⟨0, by decide⟩

#eval IO.println s!"generalized eigenvalues (sorted) = {evalsSorted}"
#eval IO.println s!"canonical correlations = {canonicalCorr} and \
  {(sortF (vecToList evals)).getD 2 0.0}  (golden {0.39596464934201425}, {0.21264898960921277})"
#eval IO.println s!"genResid={genResid}  bOrtho={bOrthoErr}  plainOrtho={plainOrthoErr}  bMatters={bMattersErr}"

/-! ## Positive checks — the reduction recovers the generalized eigenpairs -/

-- The recovered pairs satisfy the GENERALIZED eigen-equation `A·V = B·V·diag(λ)`.
#eval assertLt "GenSymEig  A·V = B·V·diag(λ)  (generalized eigen-equation)" genResid
-- The whitening guarantee: the eigenvectors are `B`-orthonormal `Vᵀ·B·V = I`.
#eval assertLt "GenSymEig  Vᵀ·B·V = I  (B-orthonormality / whitening guarantee)" bOrthoErr
-- The whitened matrix `C = L⁻¹·A·L⁻ᵀ` is symmetric (so its standard eig is well-posed).
#eval assertLt "GenSymEig  C = L⁻¹·A·L⁻ᵀ is symmetric" cSymErr
-- The eigenvalues match the LAPACK golden — a different eigensolver, the same spectrum of the pencil.
#eval assertLt "GenSymEig  eigenvalues match LAPACK eigen(A,B) golden" evalsErr 1e-7
-- The leading canonical correlation is recovered (`scipy`/Julia: `0.39596…`).
#eval assertApproxEq "GenSymEig  leading canonical correlation" canonicalCorr 0.39596464934201425 1e-7

/-! ## Negative controls — the reduction is doing real work -/

-- `V` is NOT ordinary-orthonormal: whitening trades plain orthonormality for `B`-orthonormality.
-- This is the whole point of the reduction; the golden `‖VᵀV − I‖ ≈ 0.44`.
#eval assertGe "GenSymEig  Vᵀ·V ≠ I  (whitening trades plain for B-orthonormality)" plainOrthoErr 0.1
-- The generalized eigenvalues DIFFER from `A`'s standard eigenvalues — `B` genuinely participates
-- (ignoring `B`, i.e. solving `A·v = λ·v`, gives a different spectrum; golden gap `≈ 0.314`).
#eval assertGe "GenSymEig  generalized λ ≠ standard λ of A alone (B matters)" bMattersErr 0.1
-- An indefinite `B` breaks the Cholesky pivot (`√(negative)` ⟹ `NaN`): the SPD hypothesis is needed.
#eval assertReconFails "GenSymEig  indefinite B ⟹ NaN Cholesky pivot (SPD hypothesis necessary)"
  (if badPivot.isNaN then 1.0 else 0.0)

end NN.Examples.Factorization.GenSymEig
