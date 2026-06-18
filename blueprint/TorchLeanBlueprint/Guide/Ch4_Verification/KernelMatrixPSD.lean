import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "The KernelFlows Build is SPD: Cholesky Fires (S2)" =>
%%%
tag := "kernelflows-kernel-matrix-spd"
%%%

S1 built the KernelFlows unary kernel matrix `K(logθ)` and proved it *symmetric*. The verified
Cholesky / `find_gamma` / ridge-solve development consumes `K` under a positive-(semi)definiteness
hypothesis; this step (S2) supplies the *other* half of symmetry — positive-definiteness — so that
standing hypothesis is discharged and the landed positive-pivot keystone
`choleskyFn_diag_pos_of_posDef` fires.

# The structural keystone: the nugget turns PSD into SPD

The build is `K = R + δ·I + scale·Φ·Φᵀ` with `R` the radial block, `δ = nuggetFn logθ₄ > 0` the
nugget, and `scale = e^{logθ₃} ≥ 0` the linear weight. The linear term `Φ·Φᵀ` is a Gram matrix, hence
positive-semidefinite; so the moment the radial block `R` is PSD the *base* `R + scale·Φ·Φᵀ` is PSD,
and the strictly positive nugget `δ·I` lifts it to positive-*definite*:

$$`x^\top K x = \underbrace{x^\top(R + \mathrm{scale}\cdot\Phi\Phi^\top)x}_{\ge 0} + \delta\,\lVert x\rVert^2 \;>\; 0 \quad (x \ne 0).`

This is `unaryKernelBuild_posDef`: for *any* PSD radial matrix `R`, nonnegative `scale`, and strictly
positive `δ`, the assembled `K[i,j] = R[i,j] + δ·[i=j] + scale·⟨Φᵢ, Φⱼ⟩` is `PosDef`. It is exactly
*why* KernelFlows adds a nugget — the `δ·I` is what makes a merely-PSD kernel strictly SPD, so `K = L·Lᵀ`
succeeds with strictly positive pivots and the ridge solve is exact. The keystone reduces the whole
question to one fact: *is the radial block PSD?*

# The RBF radial block: discharged in full

For the squared-exponential (RBF) radial kernel — KernelFlows' `spherical_sqexp`,
`k(d; a, b) = a·e^{-d^2/2b}` — the answer is *yes*, with no Bochner/Schoenberg machinery. The RBF
depends on the *squared* distance, so over the Euclidean distance `d = ‖Xᵢ − Xⱼ‖` it factors:

$$`a\,e^{-\lVert X_i - X_j\rVert^2/2b} \;=\; a\prod_k e^{-(X_{ik}-X_{jk})^2/2b},`

a Hadamard product over features of one-dimensional Gaussians. Each factor is PSD
(`posSemidef_gaussianCol`), the product is PSD by the *Schur product theorem*
(`posSemidef_prod_hadamard`), and scaling by `a ≥ 0` preserves it — this is the new reusable lemma
`posSemidef_gaussianRadial`. Feeding it through the keystone gives `kernelMatrixSqexpFn_posDef`: *the
RBF unary build is `PosDef` for every `logθ`, with no side hypotheses.* Then
`kernelMatrixSqexpFn_cholesky_pos` gives strictly positive Cholesky pivots and
`kernelMatrixSqexpFn_solveRidge_exact` the exact regularized solve — the pipeline closes end-to-end.

# The honest gap: Matérn is *not* elementary

The README's S2 plan expected Matérn-3/2, Matérn-5/2, and `inverse_quadratic` to fall to the same
"elementary route." Formalization corrected that. Those kernels — and KernelFlows' `spherical_exp` —
carry the *bare* Euclidean distance `d = ‖Xᵢ − Xⱼ‖`, *not* `d²`. The Gaussian trick fails: there is
no finite Hadamard/Schur certificate, because `(1 + h)e^{-h}` with `h = \sqrt3\,d/b` does not split into
a diagonal congruence of an entrywise-exponential of a Gram the way `e^{-d^2/2b}` does.

These kernels *are* positive-definite — but proving it rests on a *Gaussian scale-mixture* (Bochner /
Schoenberg) representation

$$`\varphi(d) \;=\; \int_0^\infty e^{-t\,d^2}\,d\mu(t), \qquad \mu \ge 0,`

whose defining theorems (Bernstein's completely-monotone characterization, Schoenberg's theorem) are
absent from Mathlib v4.30.0. So rather than fake a proof, S2 states the reduction with the radial PSD
as a clean hypothesis — `kernelMatrixMatern32Fn_posDef_of_radial` (and the Matérn-5/2 sibling):
`R.PosSemidef → K.PosDef`, isolating *exactly* the open analytic fact. Discharging it via the
scale-mixture integral representation is scoped as a Year-2 analytic deliverable; the RBF result shows
the keystone is real and the pipeline closes whenever the radial block is PSD.

This is itself an unintended benefit of formalization: it pinned down *which* of the five KernelFlows
radial kernels is elementary (the one on `d²`) and *why* the other four are genuinely
representation-theoretic, rather than letting an over-optimistic "all elementary" claim stand.

# Executable witnesses

The example `NN.Examples.Factorization.KernelMatrixPSD` checks the keystone numerically: the RBF build
`K(logθ)` reconstructs from its Cholesky factor (`A = L·Lᵀ` to machine precision — only possible if
every pivot `L[j,j] > 0`), with pivots `[1.19, 0.92, 0.66, 0.80]` all positive. The *negative control*
takes an indefinite symmetric matrix (eigenvalues `{3, −1}`): Cholesky takes `\sqrt{\text{negative}}`
and the reconstruction is `NaN` — no factor exists. Adding a nugget `δ·I` (here `δ = 2`) lifts it to SPD
and the factor reappears — the very `+ δ·I` lift `unaryKernelBuild_posDef` performs.
