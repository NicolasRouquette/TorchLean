/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Basic.Factorizations
public import Mathlib.Analysis.Matrix.Spectrum
public import Mathlib.Analysis.Matrix.PosDef
public import Mathlib.Analysis.Matrix.HermitianFunctionalCalculus
public import Mathlib.LinearAlgebra.Matrix.PosDef
public import Mathlib.LinearAlgebra.UnitaryGroup
public import Mathlib.LinearAlgebra.Matrix.NonsingularInverse

/-!
# Spectral consequences and the Jacobi residual certificate

The eigendecomposition / SVD half of the factorization correctness layer: the `IsSymEig` / `IsSVD`
predicates, the spectral consequences (regularized inverse, trace = Σλ, det = Πλ, SVD ⟹ Gram
eigendecomposition), the orthogonal-similarity invariants and Givens orthogonality, and the
a-posteriori residual certificate that captures cyclic-Jacobi convergence without an a-priori
convergence theory. Built on the exact Cholesky/QR layer in `NN.Proofs.Tensor.Basic.Factorizations`.
-/

@[expose] public section

namespace Spec.Factorization

open Matrix
open scoped BigOperators

variable {n : Nat}

/-- `(Λ, V)` is a symmetric eigendecomposition of `A`: `V` orthogonal, `A = V · diag(Λ) · Vᵀ`. -/
def IsSymEig (A : Matrix (Fin n) (Fin n) ℝ) (Λ : Fin n → ℝ) (V : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  Vᵀ * V = 1 ∧ A = V * Matrix.diagonal Λ * Vᵀ

/-- `(U, σ, V)` is a (thin) SVD of `A`: `U`, `V` have orthonormal columns, `σ ≥ 0`,
`A = U · diag(σ) · Vᵀ`. -/
def IsSVD {m k : Nat} (A U : Matrix (Fin m) (Fin k) ℝ) (σ : Fin k → ℝ)
    (V : Matrix (Fin k) (Fin k) ℝ) : Prop :=
  Uᵀ * U = 1 ∧ Vᵀ * V = 1 ∧ (∀ j, 0 ≤ σ j) ∧ A = U * Matrix.diagonal σ * Vᵀ

/-! ## Foundation theorems consumed by CHD

These follow from the *specification*, not from any particular algorithm. -/

/-- A symmetric eigendecomposition exhibits `A` as Hermitian (here: symmetric, over `ℝ`). -/
theorem IsSymEig.isHermitian {A : Matrix (Fin n) (Fin n) ℝ} {Λ V}
    (h : IsSymEig A Λ V) : A.IsHermitian := by
  obtain ⟨_, hA⟩ := h
  unfold Matrix.IsHermitian
  rw [hA]
  simp [Matrix.mul_assoc]

/-- From a symmetric eigendecomposition, an orthogonal matrix `V` satisfies `V · Vᵀ = 1` as well as
`Vᵀ · V = 1`. -/
theorem IsSymEig.mul_transpose_self {A : Matrix (Fin n) (Fin n) ℝ} {Λ V}
    (h : IsSymEig A Λ V) : V * Vᵀ = 1 :=
  mul_eq_one_comm.mp h.1

/-! ### The kernel-ridge / `solve_variationnal` identity

CHD repeatedly forms `(K + γ I)⁻¹ b`. Diagonalizing `K = V diag(λ) Vᵀ` turns this into a per-eigenvalue
rescaling `V diag(1/(λ+γ)) Vᵀ b`, which is the basis of `solve_variationnal`, `find_gamma` and the
`Z_test`. The identity below is proved purely from orthogonality of `V` (no appeal to Mathlib's own
spectral decomposition), so it holds for *any* eigendecomposition the algorithm returns. -/

/-- Conjugating a diagonal by an orthogonal `V` is inverted by conjugating the entrywise inverse:
`(V · diag(d) · Vᵀ) · (V · diag(d⁻¹) · Vᵀ) = 1` when every `d i ≠ 0`. -/
theorem orthogonal_conj_diagonal_mul_inv {V : Matrix (Fin n) (Fin n) ℝ} (hV : Vᵀ * V = 1)
    {d : Fin n → ℝ} (hd : ∀ i, d i ≠ 0) :
    (V * Matrix.diagonal d * Vᵀ) * (V * Matrix.diagonal (fun i => (d i)⁻¹) * Vᵀ) = 1 := by
  have hdd : (Matrix.diagonal d) * (Matrix.diagonal (fun i => (d i)⁻¹))
      = (1 : Matrix (Fin n) (Fin n) ℝ) := by
    rw [Matrix.diagonal_mul_diagonal]
    rw [show (fun i => d i * (d i)⁻¹) = (fun _ : Fin n => (1 : ℝ)) from
      funext fun i => mul_inv_cancel₀ (hd i)]
    exact Matrix.diagonal_one
  calc
    (V * Matrix.diagonal d * Vᵀ) * (V * Matrix.diagonal (fun i => (d i)⁻¹) * Vᵀ)
        = V * Matrix.diagonal d * (Vᵀ * V) * Matrix.diagonal (fun i => (d i)⁻¹) * Vᵀ := by
          simp [Matrix.mul_assoc]
    _ = V * (Matrix.diagonal d * Matrix.diagonal (fun i => (d i)⁻¹)) * Vᵀ := by
          rw [hV]; simp [Matrix.mul_assoc]
    _ = V * Vᵀ := by rw [hdd, Matrix.mul_one]
    _ = 1 := mul_eq_one_comm.mp hV

/-- `K + γ I` rewritten through the eigendecomposition: `V · diag(λ + γ) · Vᵀ`. -/
theorem IsSymEig.add_smul_eq {A : Matrix (Fin n) (Fin n) ℝ} {Λ V}
    (h : IsSymEig A Λ V) (γ : ℝ) :
    A + γ • (1 : Matrix (Fin n) (Fin n) ℝ)
      = V * Matrix.diagonal (fun i => Λ i + γ) * Vᵀ := by
  obtain ⟨hV, hA⟩ := h
  have hVV : V * Vᵀ = 1 := mul_eq_one_comm.mp hV
  have hsplit : Matrix.diagonal (fun i => Λ i + γ)
      = Matrix.diagonal Λ + γ • (1 : Matrix (Fin n) (Fin n) ℝ) := by
    ext i j
    by_cases hij : i = j <;>
      simp [Matrix.add_apply, Matrix.smul_apply, hij]
  rw [hsplit, hA]
  rw [Matrix.mul_add, Matrix.add_mul]
  congr 1
  rw [Matrix.mul_smul, Matrix.smul_mul, Matrix.mul_one, hVV]

/-- **Regularized inverse / `solve_variationnal`.** For `γ` avoiding `-λᵢ`, the regularized system
`K + γ I` is inverted by per-eigenvalue rescaling: `(K + γ I)⁻¹ = V · diag(1/(λ + γ)) · Vᵀ`. -/
theorem IsSymEig.add_smul_inv {A : Matrix (Fin n) (Fin n) ℝ} {Λ V}
    (h : IsSymEig A Λ V) (γ : ℝ) (hγ : ∀ i, Λ i + γ ≠ 0) :
    (A + γ • (1 : Matrix (Fin n) (Fin n) ℝ))⁻¹
      = V * Matrix.diagonal (fun i => (Λ i + γ)⁻¹) * Vᵀ := by
  apply Matrix.inv_eq_right_inv
  rw [h.add_smul_eq γ]
  exact orthogonal_conj_diagonal_mul_inv h.1 hγ

/-! ### Spectral trace and determinant (used by `find_gamma` / model-evidence terms) -/

/-- `trace K = Σ λᵢ`. -/
theorem IsSymEig.trace_eq {A : Matrix (Fin n) (Fin n) ℝ} {Λ V}
    (h : IsSymEig A Λ V) : A.trace = ∑ i, Λ i := by
  obtain ⟨hV, hA⟩ := h
  rw [hA, Matrix.trace_mul_comm, ← Matrix.mul_assoc, hV, Matrix.one_mul,
    Matrix.trace_diagonal]

/-- `det K = Π λᵢ`. -/
theorem IsSymEig.det_eq {A : Matrix (Fin n) (Fin n) ℝ} {Λ V}
    (h : IsSymEig A Λ V) : A.det = ∏ i, Λ i := by
  obtain ⟨hV, hA⟩ := h
  have hVV : V * Vᵀ = 1 := mul_eq_one_comm.mp hV
  rw [hA, Matrix.det_mul, Matrix.det_mul, Matrix.det_diagonal,
    mul_right_comm, ← Matrix.det_mul, hVV, Matrix.det_one, one_mul]

/-! ### SVD ⟹ eigendecomposition of the Gram matrix

CHD forms the kernel/Gram matrix `K = Aᵀ A` and eigendecomposes it. An SVD of `A` *is* such an
eigendecomposition, with eigenvalues `σᵢ²` and the same orthogonal `V`. -/

/-- The right singular vectors `V` of `A` diagonalize the Gram matrix `Aᵀ A`, with eigenvalues `σᵢ²`. -/
theorem IsSVD.gram_isSymEig {m k : Nat} {A U : Matrix (Fin m) (Fin k) ℝ}
    {σ : Fin k → ℝ} {V} (h : IsSVD A U σ V) :
    IsSymEig (Aᵀ * A) (fun i => σ i ^ 2) V := by
  obtain ⟨hU, hV, _, hA⟩ := h
  refine ⟨hV, ?_⟩
  have hσσ : Matrix.diagonal σ * Matrix.diagonal σ
      = Matrix.diagonal (fun i => σ i ^ 2) := by
    rw [Matrix.diagonal_mul_diagonal]; simp [pow_two]
  rw [hA, Matrix.transpose_mul, Matrix.transpose_mul, Matrix.transpose_transpose,
    Matrix.diagonal_transpose]
  -- V Dᵀ Uᵀ · U D Vᵀ  with Dᵀ = D
  calc
    V * (Matrix.diagonal σ * Uᵀ) * (U * Matrix.diagonal σ * Vᵀ)
        = V * Matrix.diagonal σ * (Uᵀ * U) * Matrix.diagonal σ * Vᵀ := by
          simp [Matrix.mul_assoc]
    _ = V * (Matrix.diagonal σ * Matrix.diagonal σ) * Vᵀ := by
          rw [hU]; simp [Matrix.mul_assoc]
    _ = V * Matrix.diagonal (fun i => σ i ^ 2) * Vᵀ := by rw [hσσ]

/-! ## Tier B — exact structural & invariant facts

These hold *exactly* (no convergence/rounding caveat). The orthogonal-similarity invariants below are
the precise sense in which the Jacobi iteration is faithful: every sweep is an orthogonal similarity
`A ← Jᵀ A J`, so trace, determinant and spectrum are preserved at every step, independent of how far
the off-diagonal has been driven down. -/

/-- Orthogonal similarity preserves the trace: `trace (V · M · Vᵀ) = trace M` when `Vᵀ · V = 1`. -/
theorem trace_orthogonal_conj {V M : Matrix (Fin n) (Fin n) ℝ} (hV : Vᵀ * V = 1) :
    (V * M * Vᵀ).trace = M.trace := by
  rw [Matrix.trace_mul_comm, ← Matrix.mul_assoc, hV, Matrix.one_mul]

/-- Orthogonal similarity preserves the determinant: `det (V · M · Vᵀ) = det M` when `Vᵀ · V = 1`. -/
theorem det_orthogonal_conj {V M : Matrix (Fin n) (Fin n) ℝ} (hV : Vᵀ * V = 1) :
    (V * M * Vᵀ).det = M.det := by
  have hVV : V * Vᵀ = 1 := mul_eq_one_comm.mp hV
  rw [Matrix.det_mul, Matrix.det_mul, mul_right_comm, ← Matrix.det_mul, hVV, Matrix.det_one,
    one_mul]

/-- **Givens rotation is orthogonal.** With `c = 1/√(1+t²)` and `s = t·c` (the parameters
`arrJacobiRotate` uses), the rotation satisfies `c² + s² = 1`, so every Jacobi step is an orthogonal
transformation. -/
theorem givens_normSq (t : ℝ) :
    (1 / Real.sqrt (1 + t ^ 2)) ^ 2 + (t * (1 / Real.sqrt (1 + t ^ 2))) ^ 2 = 1 := by
  have hpos : (0 : ℝ) < 1 + t ^ 2 := by positivity
  have hsqrt : Real.sqrt (1 + t ^ 2) ^ 2 = 1 + t ^ 2 := Real.sq_sqrt hpos.le
  have hne : (1 + t ^ 2) ≠ 0 := ne_of_gt hpos
  have hc2 : (1 / Real.sqrt (1 + t ^ 2)) ^ 2 = 1 / (1 + t ^ 2) := by
    rw [div_pow, one_pow, hsqrt]
  rw [mul_pow, hc2]
  field_simp
/-! ## Tier D — convergence as an a-posteriori residual certificate

The cyclic Jacobi iteration produces `(Λ, V)` from the rotated matrix `Af = Vᵀ A V` (an *exact*
orthogonal similarity — see `trace_orthogonal_conj`), with `Λ` the diagonal of `Af`. After finitely
many sweeps `Af` is only *approximately* diagonal, so `A = V·diag(Λ)·Vᵀ` does not hold exactly (and
never does in floating point). Mathlib v4.30.0 has no Jacobi convergence theory, so instead of an
*a-priori* convergence proof we give the *a-posteriori* certificate: the reconstruction residual is
exactly the orthogonal conjugation of the off-diagonal part of `Af`, hence its Frobenius mass equals
the off-diagonal mass — which the runtime `assertLt` checks in `NN/Examples/Factorization` bound on
concrete inputs. -/

/-- The off-diagonal part of a matrix (`0` iff the matrix is diagonal). -/
def offDiagonal (M : Matrix (Fin n) (Fin n) ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  M - Matrix.diagonal (fun i => M i i)

/-- **Exact residual identity.** Reconstructing with the diagonal of `Af` leaves exactly the orthogonal
conjugation of `Af`'s off-diagonal part: `A − V·diag(Af)·Vᵀ = V · offDiag(Af) · Vᵀ`. -/
theorem symEig_reconstruction_residual {A V Af : Matrix (Fin n) (Fin n) ℝ}
    (hA : A = V * Af * Vᵀ) :
    A - V * Matrix.diagonal (fun i => Af i i) * Vᵀ = V * offDiagonal Af * Vᵀ := by
  rw [hA, offDiagonal, Matrix.mul_sub, Matrix.sub_mul]

/-- **Frobenius residual certificate.** The squared Frobenius reconstruction error
`‖A − V·diag(Af)·Vᵀ‖²` equals the squared Frobenius off-diagonal mass `‖offDiag(Af)‖²` (expressed as
`trace(Rᵀ R)`), because orthogonal conjugation preserves the Frobenius norm. In particular it is `0`
iff `Af` is diagonal — the exact sense in which "more Jacobi sweeps ⟹ smaller residual". -/
theorem symEig_frobenius_residual {A V Af : Matrix (Fin n) (Fin n) ℝ} (hV : Vᵀ * V = 1)
    (hA : A = V * Af * Vᵀ) :
    ((A - V * Matrix.diagonal (fun i => Af i i) * Vᵀ)ᵀ
        * (A - V * Matrix.diagonal (fun i => Af i i) * Vᵀ)).trace
      = ((offDiagonal Af)ᵀ * offDiagonal Af).trace := by
  rw [symEig_reconstruction_residual hA]
  have hB : (V * offDiagonal Af * Vᵀ)ᵀ = V * (offDiagonal Af)ᵀ * Vᵀ := by
    rw [Matrix.transpose_mul, Matrix.transpose_mul, Matrix.transpose_transpose, Matrix.mul_assoc]
  have key : (V * offDiagonal Af * Vᵀ)ᵀ * (V * offDiagonal Af * Vᵀ)
      = V * ((offDiagonal Af)ᵀ * offDiagonal Af) * Vᵀ := by
    rw [hB]
    calc
      (V * (offDiagonal Af)ᵀ * Vᵀ) * (V * offDiagonal Af * Vᵀ)
          = V * (offDiagonal Af)ᵀ * (Vᵀ * V) * offDiagonal Af * Vᵀ := by simp [Matrix.mul_assoc]
      _ = V * ((offDiagonal Af)ᵀ * offDiagonal Af) * Vᵀ := by rw [hV]; simp [Matrix.mul_assoc]
  rw [key]
  exact trace_orthogonal_conj hV

/-- **Conditional correctness of Jacobi.** When the rotated matrix `Af = Vᵀ A V` is diagonal (zero
residual — the limit the sweeps drive toward), the Jacobi output `(diag Af, V)` is an *exact*
symmetric eigendecomposition `IsSymEig`. Together with `symEig_frobenius_residual` this is the precise
correctness statement: orthogonality and the orthogonal-similarity hold always; full diagonalization
holds exactly in the zero-residual limit. -/
theorem isSymEig_of_diagonal {A V Af : Matrix (Fin n) (Fin n) ℝ} (hV : Vᵀ * V = 1)
    (hA : A = V * Af * Vᵀ) (hdiag : Af = Matrix.diagonal (fun i => Af i i)) :
    IsSymEig A (fun i => Af i i) V :=
  ⟨hV, by rw [hA]; conv_lhs => rw [hdiag]⟩
end Spec.Factorization
