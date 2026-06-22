import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "Symmetric Eigendecomposition and SVD" =>
%%%
tag := "factorizations-eig-svd"
%%%

On the exact Cholesky/QR layer this chapter builds the two *iterative* factorizations: the full
symmetric eigendecomposition by the cyclic Jacobi algorithm, and the singular value decomposition on top
of it. Unlike Cholesky and QR, Jacobi converges only *asymptotically*, so its correctness splits into a
part that is exact at every sweep and a part captured by an a-posteriori certificate.

# Specifications

- `IsSymEig A Λ V` — `V` is orthogonal (`Vᵀ·V = 1`) and `A = V·diag(Λ)·Vᵀ`;
- `IsSVD A U σ V` — `U`, `V` have orthonormal columns, `σ ≥ 0`, and `A = U·diag(σ)·Vᵀ`.

`symEigJacobiSpec` and `svdSpec` are the executable specs; `svdSpec` is the symmetric
eigendecomposition of the Gram matrix `Aᵀ·A`, with `σ = √λ` and `U = A·V·diag(1/σ)`.

# What is exact at every sweep

Each Jacobi sweep is an *orthogonal similarity* `A ← Jᵀ·A·J` with `J` a Givens rotation. Two facts hold
*exactly*, at any sweep count, with no convergence hypothesis:

- *Givens orthogonality* (`givens_normSq`): with `c = 1/√(1+t²)` and `s = t·c` the rotation satisfies
  `c² + s² = 1`, so `Vᵀ·V = 1` exactly;
- *orthogonal-similarity invariance* (`trace_orthogonal_conj`, `det_orthogonal_conj`): conjugation by an
  orthogonal `V` preserves trace and determinant, hence the spectrum is preserved at every step
  regardless of how far the off-diagonal has been driven down.

The spectral consequences the downstream kernel methods consume are proved here from the predicate alone:
the regularized inverse `(A + γ·1)⁻¹ = V·diag(1/(λ+γ))·Vᵀ` for `γ > 0`, `trace = Σλ`, `det = Πλ`, and
that an SVD of `A` is a symmetric eigendecomposition of `Aᵀ·A` (`IsSVD.gram_isSymEig`).

# Convergence as an a-posteriori residual certificate

Mathlib v4.30.0 has no Jacobi convergence theory, so full diagonalization is *not* asserted a-priori (and
never holds exactly in floating point). Instead the residual is bounded exactly. Writing `Af = Vᵀ·A·V`
for the rotated matrix, the reconstruction residual is exactly the orthogonal conjugation of `Af`'s
off-diagonal part, so its Frobenius mass equals the off-diagonal mass:

$$`\bigl\|A - V\,\mathrm{diag}(Af)\,V^{\top}\bigr\|_F^2 \;=\; \bigl\|\mathrm{offDiag}(Af)\bigr\|_F^2`

(`symEig_frobenius_residual`). It is `0` *iff* `Af` is diagonal, in which case the Jacobi output is an
*exact* `IsSymEig` (`isSymEig_of_diagonal`). This is the precise sense of "more sweeps ⟹ smaller
residual": orthogonality and the similarity invariants are exact always; full diagonalization is the
zero-residual limit, and the runtime `assertLt` examples are concrete instances of the bound.

The *generalized* symmetric eigenproblem `A·v = λ·B·v`, reduced to this standard one by Cholesky
whitening, is developed in the companion chapter (*Generalized Eigenproblems by Cholesky Whitening*).

# Executable witnesses

`NN.Examples.Factorization.SymEig`, `…SVD`, `…JacobiDecrease`, and `…JacobiRate` exhibit the
eigendecomposition and its convergence behaviour on concrete matrices — positive reconstruction /
orthogonality checks alongside negative controls (e.g. a non-largest Givens pivot failing the classical
contraction rate), every check a `#eval` over `Float`, with no unproved goals, omega-free.
