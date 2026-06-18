import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "KernelFlows Losses Are Well-Posed on the SPD Build (S3)" =>
%%%
tag := "kernelflows-loss-well-posed"
%%%

S2 made the KernelFlows kernel matrix `Ω = K(logθ)` symmetric positive-definite (SPD). KernelFlows then
*learns* the kernel by minimizing a cross-validation loss `ρ(logθ)` over the hyperparameters
(`loss_functions.jl`). This step (S3) ports the four production losses — `ρ_KF`, `ρ_LOI`, `ρ_LOO`,
`ρ_MLE` — and proves they are *well-posed* on an SPD `Ω`: the shared denominator never vanishes, the
bounded ones really are bounded, and `ρ_MLE`'s data term is exactly the Gaussian-process quadratic form.

# One primitive under all four: the regularized quadratic form

Every KernelFlows loss is `inv(Symmetric(Ω))`-quadratic — built on `yᵀ Ω⁻¹ y`. The executable spec
computes it through the *landed* verified Cholesky solve rather than by forming an inverse:

$$`q(y) \;=\; \texttt{quadInvFn}\,\Omega\,y \;=\; y^\top\bigl(\texttt{cholSolveFn}\,(\texttt{choleskyFn}\,\Omega)\,y\bigr).`

On an SPD `Ω` this *is* the inverse quadratic form. The positive-pivot keystone makes the executable
Cholesky a genuine factor `Ω = L·Lᵀ`, the two-pass substitution solves `Ω·x = y`, and invertibility
identifies `x` with `Ω⁻¹·y` — this is `quadInvFn_eq_dotProduct_inv`: `q(y) = y ⬝ᵥ (Ω⁻¹ *ᵥ y)`. Two
consequences flow from the *regularized inverse* `Ω⁻¹` being itself positive-(semi)definite:

* `quadInvFn_nonneg` — `q(y) ≥ 0`;
* `quadInvFn_pos` — `q(y) > 0` whenever `y ≠ 0`.

The strict bound is the load-bearing one: it is exactly why no KernelFlows loss divides by zero. The S2
nugget `+δI` is what put `Ω` in the SPD cone where `q(y) > 0` holds, so S3 is where the nugget *pays off*
for the loss layer.

# The four losses

The losses share `q(y)` as denominator and differ in the numerator (`loss_functions.jl`):

$$`\rho_{\mathrm{LOI}} = 1 - \frac{y^\top y / \Omega_{00} / n}{q(y)}, \qquad
   \rho_{\mathrm{KF}} = 1 - \frac{q_c(y_c)}{q(y)}, \qquad
   \rho_{\mathrm{LOO}} = N - \frac{y^\top M y}{q(y)}, \qquad
   \rho_{\mathrm{MLE}} = \tfrac12\lVert L^{-1}y\rVert^2 - \textstyle\sum_i \log\tfrac1{L_{ii}}.`

*`ρ_LOI` and `ρ_KF` are bounded above by `1`.* Each is `1 − (\text{nonnegative}) / (\text{positive})`.
For `ρ_LOI` the numerator is nonnegative because `Ω₀₀ > 0` on an SPD matrix (`rhoLOIFn_le_one`). For
`ρ_KF` the center-block form `q_c(y_c) = y_c^\top Ω_c^{-1} y_c` is nonnegative because *a principal
submatrix of an SPD matrix is SPD* — `Matrix.PosDef.submatrix` applied to the injective center embedding
`e : Fin nc ↪ Fin n` — so `rhoKFFn_le_one` holds for *any* choice of center indices, not just the
leading half KernelFlows uses.

*`ρ_MLE` is the GP negative log marginal likelihood.* With `L` the Cholesky factor and `z = L^{-1}y`
(forward substitution), the data term is `½‖z‖²`. The KernelFlows source *asserts* — in a commented-out
check, `should be 0: ret1 - ret2` — that this equals `½ yᵀΩ⁻¹y`. Here it is *proved*
(`rhoMLE_data_eq_quadInv`): from the verified substitutions `L·z = y` and `Lᵀ·x = z`,

$$`\lVert z\rVert^2 = (Lz)^\top x = y^\top x = y^\top \Omega^{-1} y,`

a purely algebraic identity needing no asymptotics. With the log-determinant term
`−Σ log(1/L_{ii}) = Σ log L_{ii} = ½ \log\det\Omega`, this exhibits `ρ_MLE = ½ yᵀΩ⁻¹y + ½ \log\det\Omega`
— the textbook GP NLL (minus the constant `½ n \log 2π` KernelFlows omits). The data term is nonnegative
(`rhoMLE_data_nonneg`).

`ρ_LOO`'s denominator is the same `q(y)`, so `quadInvFn_pos` discharges its well-posedness too; its
leave-one-out operator `M` is assembled from `Ω⁻¹` columns by the same Cholesky solve, with no explicit
inverse ever formed.

# What S3 surfaced: the losses inherit the nugget's SPD-ness, exactly

The honest content of S3 is that *all four losses are one quadratic form away from S2*. There is no
separate analytic obstruction here — once `Ω` is SPD, the loss layer is finite linear algebra over the
landed Cholesky/solve development. The single mathematical fact that does the work is that the
*regularized inverse of an SPD matrix is positive-definite*, which the strict denominator bound
`q(y) > 0` packages. This is why S3 is a Year-1 *direct-reuse* step: the verified `solve_variationnal`
substrate is the loss substrate.

# Executable witnesses

The example `NN.Examples.Factorization.KernelLoss` checks the losses numerically on the SPD RBF build
from S2. On the SPD kernel: the quadratic form `q = yᵀΩ⁻¹y` is strictly positive; the `ρ_MLE` data term
`½‖L⁻¹y‖²` matches `½ q` to machine precision (witnessing the `a1 = a2` identity); and `ρ_KF`, `ρ_LOI`
are `≤ 1`, with all four losses finite. The *negative control* takes an indefinite symmetric matrix
(eigenvalues `{3, −1}`): its Cholesky takes `√(\text{negative})`, the quadratic form is `NaN`, and the
losses are `NaN` — so SPD-ness is necessary. Adding a nugget `δ·I` (here `δ = 2`) lifts it back to SPD
and the quadratic form is finite and positive again — the very `+δI` lift S2 performs.
