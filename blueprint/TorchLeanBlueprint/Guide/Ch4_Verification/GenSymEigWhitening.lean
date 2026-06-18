import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "Generalized Eigenproblems by Cholesky Whitening: CCA and Dimension Reduction (S9)" =>
%%%
tag := "gen-sym-eig-whitening"
%%%

The landed eigensolver `symEigJacobiSpec` solves the *standard* symmetric eigenproblem
`A·v = λ·v`. The CHD dimension-reduction layer (`dimension_reduction.jl` / `cca.jl`) needs the
*generalized* one,

$$`A\,v = \lambda\, B\, v, \qquad A = A^{\top},\ \ B = B^{\top} \succ 0,`

whose solutions are, for the canonical-correlation pencil, the *canonical correlations* and the
canonical directions. S9 is the *whitening reduction*: it turns this generalized problem into a
standard one the verified primitives already solve, so *no new eigensolver is needed* — only a
composition of `choleskyFn`, the triangular solves, and `symEigJacobiSpec`.

# The reduction: whiten `B`, solve, unwhiten

Because `B` is symmetric positive-definite it has a Cholesky factor `B = L·Lᵀ` (`choleskyFn`, whose
positive pivots are exactly the SPD keystone proved for the kernel-ridge solve). Substituting
`w = Lᵀ·v` turns the pencil into a standard eigenproblem on the *whitened* matrix:

$$`A\,v = \lambda\,B\,v \iff \underbrace{L^{-1} A\, L^{-\top}}_{C}\,\underbrace{(L^{\top} v)}_{w} = \lambda\,(L^{\top} v) \iff C\,w = \lambda\,w.`

So the four steps are: factor `B = L·Lᵀ`; form `C = L⁻¹·A·L⁻ᵀ` (`whitenFn`, by triangular solves
only — never an explicit inverse); standard-eigendecompose `C = W·diag(λ)·Wᵀ` (`symEigJacobiSpec`);
and *unwhiten* the eigenvectors by back-substitution `v = L⁻ᵀ·w`. The matrix `C` is symmetric (it is
a congruence of the symmetric `A`), so `symEigJacobiSpec` applies *unchanged*; the eigenvalues of `C`
*are* the generalized eigenvalues of the pencil, exactly, as pure algebra — no new approximation is
introduced by the reduction itself. This is `genSymEigCholeskySpec`.

# What the reduction guarantees: `B`-orthonormality, not plain orthonormality

The point of whitening — and the reason CCA wants it — is the inner-product the eigenvectors are
orthonormal in. The standard solver returns `Wᵀ·W = I`; unwhitening `v = L⁻ᵀ·w` carries this to

$$`V^{\top} B\, V = W^{\top} L^{-1}\,(L\,L^{\top})\,L^{-\top} W = W^{\top} W = I,`

so the recovered `V` is `B`-*orthonormal*: `Vᵀ·B·V = I`. That is the whitening guarantee. It is
*not* ordinary orthonormality — generically `Vᵀ·V ≠ I` — and the difference is the substance of the
reduction, not an artifact. On the fixture the harness reads `‖Vᵀ·B·V − I‖ ≈ 0` against
`‖Vᵀ·V − I‖ ≈ 0.44`: the eigenvectors are exactly `B`-orthonormal and visibly *not* plain
orthonormal, so the negative control has teeth.

# The CCA pencil, and parity with LAPACK despite a different eigensolver

The fixture is a genuine *canonical-correlation pencil*. From two views' covariance blocks `Σxx`,
`Σyy` (each SPD) and their cross-covariance `Σxy`, set

$$`A = \begin{pmatrix} 0 & \Sigma_{xy} \\ \Sigma_{yx} & 0 \end{pmatrix}, \qquad B = \begin{pmatrix} \Sigma_{xx} & 0 \\ 0 & \Sigma_{yy} \end{pmatrix}.`

The generalized eigenvalues are then the canonical correlations `±ρ`. Solving the pencil through the
whitening reduction recovers `λ ≈ {±0.396,\ ±0.213}` — the two canonical correlations `0.396` and
`0.213`, matching the LAPACK reference `eigen(A, B)` (`sygv`) to `1e-7`. This match is worth stating
plainly: the verified *Jacobi* eigensolver and the reference *LAPACK* eigensolver produce different
eigenvectors, yet they agree on every generalized *eigenvalue* — because the eigenvalues of the pencil
are basis-independent, a property of `(A, B)` alone, exactly as the CHD `noise` thresholds were in S8.

# The honest scope, and where the SPD hypothesis bites

The reduction's *algebra* — the eigenvalue-equivalence `A·v = λ·B·v ⟺ C·w = λ·w`, the
`B`-orthonormality of the unwhitened vectors — is exact. The one inexactness inherited is the
*standard* eigendecomposition of `C`, which rides the same cyclic-Jacobi solver characterized in the
SymEig chapter: orthogonality of `W` is a-priori exact at any sweep count, full diagonalization is
asymptotic and captured by the residual certificate. S9 adds no new approximation of its own; it is an
*examples-level* validation deliverable (the same posture as the S8 parity harness), exhibiting that
the composed reduction reproduces the LAPACK pencil to machine precision on the CCA fixture.

The *negative controls* keep the scope honest. The recovered `V` is shown *not* ordinary-orthonormal
(`‖Vᵀ·V − I‖ ≈ 0.44`), so the `B`-orthonormality is a real, distinct guarantee. The generalized
spectrum is shown to *differ* from the standard spectrum of `A` alone (gap `≈ 0.31`), so `B` genuinely
participates — the reduction is not silently ignoring it. And an *indefinite* `B` breaks the very first
step: `√(\text{negative})` in the Cholesky pivot yields `NaN`, which propagates through the whole
reduction, so the `B ≻ 0` hypothesis is necessary, not decorative.

# Executable witnesses

`NN.Examples.Factorization.GenSymEig` is the harness: `genSymEigCholeskySpec` on the CCA covariance
pencil, with the generalized eigen-equation `A·V = B·V·diag(λ)`, the whitening guarantee `Vᵀ·B·V = I`,
the symmetry of the whitened `C`, the LAPACK eigenvalue golden, and the leading canonical correlation
all checked positive, alongside the three negative controls (not plain-orthonormal, `B` matters,
indefinite `B` breaks Cholesky). Every check is a `#eval` over `Float`, with no unproved goals, omega-free, green
on `lake build NN.Examples.Factorization`.
