import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "Predictive-Variance Models Are Correct on the SPD Build (S4)" =>
%%%
tag := "kernelflows-condvar-correct"
%%%

S2 made the kernel matrix `Ω = K(logθ)` SPD; S3 showed every KernelFlows loss rests on the inverse
quadratic form `yᵀ Ω⁻¹ y`. This step (S4) ports KernelFlows' uncertainty quantification
(`conditional_variance.jl`) — the *full-rank* and *low-rank* conditional-variance models — and proves
them correct on an SPD `Ω`. The predictive variance at a test point `x*` with cross-covariance
`m = k(x*, X)` is the inverse quadratic form `mᵀ Ω⁻¹ m`; the two models compute it two ways.

# Full-rank: the conditional variance is the S3 quadratic form

`ScalarFullRankUncertaintyModel.predict_variance` factors `Ω = L·Lᵀ` (Cholesky, S2) and returns the
forward-solve sum of squares `Σ (L⁻¹ m)²`:

$$`v_{\mathrm{full}}(m) \;=\; \texttt{condVarFullFn}\,\Omega\,m \;=\; \lVert L^{-1} m\rVert^2.`

This is *exactly* S3's `q(m) = yᵀΩ⁻¹y` with `y = m`: `condVarFullFn_eq_quadInv` is a one-line corollary of
the `‖L⁻¹y‖² = yᵀΩ⁻¹y` identity (`rhoMLE_data_eq_quadInv`). So the full-rank model inherits S3's facts
verbatim — `condVarFullFn_eq_inv_quadForm` (`= m ⬝ᵥ (Ω⁻¹ *ᵥ m)`), `condVarFullFn_nonneg`, and
`condVarFullFn_pos` (strictly positive for `m ≠ 0`). The UQ layer is *another* reuse of the one verified
quadratic form, with no new linear algebra.

*The posterior variance is nonnegative.* The model subtracts the explained variance from the prior:
`condVarPostFn Ω m k_{**} = k(x*,x*) − mᵀΩ⁻¹m`. Whenever the bordered kernel

$$`\begin{pmatrix} \Omega & m \\ m^\top & k(x^*,x^*) \end{pmatrix}`

is positive-semidefinite — automatic for a genuine kernel Gram matrix over the training points together
with the test point — this is nonnegative (`condVarPostFn_nonneg`). The proof is a Schur complement:
`Matrix.PosDef.fromBlocks₁₁` turns the bordered-PSD hypothesis into PSD-ness of the `1×1` Schur
complement `k(x*,x*) − mᵀΩ⁻¹m`, whose entry is therefore `≥ 0`. The model never explains away more
variance than the prior holds.

# Low-rank: the diagonal correction matches marginals exactly

`ScalarLowRankUncertaintyModel` does not store the full precision. It eigendecomposes `Ω = V·diag(λ)·Vᵀ`
(`eigh`, S2), writes the precision spectrally as `Ω⁻¹ = Σₖ (1/λₖ) vₖ vₖᵀ`, keeps a leading set of
eigenpairs as a low-rank `PᵀP`, and *adds back the dropped eigenpairs on the diagonal* as `DᵀD` so the
marginal variances stay exact:

$$`(P^\top P)[a,b] = \sum_{k\in\text{keep}} \tfrac1{\lambda_k} V_{ak} V_{bk}, \qquad
   D[a,a]^2 = \sum_{k\notin\text{keep}} \tfrac1{\lambda_k} V_{ak}^2.`

The headline `precision_resid_diag_eq_inv` is that, for *any* kept set,

$$`(P^\top P)[a,a] + D[a,a]^2 \;=\; (\Omega^{-1})[a,a]`

*exactly*. The proof is a pure sum split — kept plus dropped is all of `Σₖ (1/λₖ) V_{ak}²`, which is the
`(a,a)` entry of the spectral inverse `IsSymEig.inv_apply` — so it is independent of the eigenvalue
ordering and holds for whatever `eigh` returns. This is precisely the model's stated design goal, "can
still match the marginal uncertainties exactly." Two more facts make the model well-formed:

* `residDiagLowRankFn_nonneg` — `D[a,a]² ≥ 0`, so `D = √(dfull − d)` is *real* (each dropped term
  `(1/λₖ) V_{ak}²` is nonnegative because `λₖ > 0` on an SPD `Ω`);
* `condVarLowRankFn_nonneg` — the predicted variance `mᵀ(PᵀP + DᵀD)m ≥ 0`, since (closed form
  `bilinFn_precisionLowRank_eq`) the kept term is a nonnegative combination of squared projections
  `Σ_{k∈keep} (1/λₖ) ⟨vₖ, m⟩²` and the residual term is a nonnegative-weighted sum of squares.

The spectral inverse `Ω⁻¹ = V·diag(1/λ)·Vᵀ` (`IsSymEig.inv`) is the `γ = 0` specialization of the CHD
regularized inverse `IsSymEig.add_smul_inv` proved earlier — so the low-rank UQ and CHD's
`solve_variationnal` rest on the *same* spectral identity.

# What S4 surfaced: UQ is the same quadratic form, plus an exact marginal split

S4 has no new analytic obstruction. The full-rank model *is* the S3 quadratic form, so its correctness is
a corollary; the only genuinely new content is the low-rank model's diagonal correction, and the fact
that it matches marginals exactly is a *sum split over the spectral inverse* — finite linear algebra over
the landed `eigh` development, no asymptotics. The low-rank approximation is real (it drops the
off-diagonal precision, so the rank-`r` value differs from the full value), but the *marginals* it
reports are exact regardless of rank. This is why S4, like S3, is a Year-1 direct-reuse step.

# Executable witnesses

The example `NN.Examples.Factorization.CondVar` checks both models on a concrete SPD kernel `K`,
against the `conditional_variance.jl` *golden values* (computed in Julia). The full-rank
`Σ(L⁻¹m)²` reproduces `mᵀK⁻¹m = [0.2832, 0.6312]` to machine precision (Cholesky is exact); the rank-`2`
low-rank value reproduces `[0.2467, 0.5523]` (Jacobi `eigh` accuracy); and the low-rank diagonal
`PᵀP + DᵀD` reproduces `diag(K⁻¹) = [0.28, 0.48, 0.88, 0.72]` to machine precision for the kept set — a
witness that the marginal match is *exact*, not approximate. The posterior variance `1 − mᵀK⁻¹m` is
positive. The *negative controls* take an indefinite matrix — the Cholesky takes `√(\text{negative})`,
so the full-rank value is `NaN` and SPD-ness is necessary — and confirm that the rank-`2` value genuinely
differs from the full value (the approximation has teeth) even though its marginals match exactly.
