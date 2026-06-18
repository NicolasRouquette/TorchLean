import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "Jacobi's Formula Is Exact Polynomial Arithmetic (S7)" =>
%%%
tag := "jacobi-formula-logdet"
%%%

The KernelFlows maximum-likelihood loss `ρ_MLE = ½ yᵀΩ⁻¹y + ½ log det Ω` (the Gaussian-process negative
log marginal likelihood, S3) carries one term the three cross-validation losses do not — the
*log-determinant*. Step S6 differentiated everything built on the inverse quadratic form `yᵀΩ⁻¹y`; S7
differentiates the one remaining piece. Its gradient is *Jacobi's formula*

$$`\frac{\partial}{\partial t}\, \log \det \Omega = \operatorname{tr}\!\big(\Omega^{-1}\,\partial_t\Omega\big),`

the *single genuinely new differentiation lemma* in the entire KernelFlows gradient. Like S6, no
iteration-count *convergence* claim is made; S7 certifies the gradient arithmetic only.

# The data half is free; only the log-determinant is new

`∂ρ_MLE = ½ ∂(yᵀΩ⁻¹y) + ½ ∂(log det Ω)`. The data term `∂(yᵀΩ⁻¹y)` is *exactly* S6's inverse-form
primitive `quadInvGradFn` — `−(Ω⁻¹y)ᵀ H (Ω⁻¹y)`, reused verbatim. So the whole content of S7 is the
complexity term `∂(log det Ω)`, and the spec `rhoMLEGradFn` is literally
`½ quadInvGradFn + ½ logDetGradFn`. The theorem `rhoMLEGradFn_eq` certifies that decomposition on an SPD
`Ω`, with both halves in closed form.

# Jacobi's formula is the coefficient of `t` in a determinant

The headline `derivative_det_eq_det_mul_trace` is *not* an analytic statement. It is an exact identity in
the polynomial ring `ℝ[X]`: for invertible `Ω`,

$$`\partial_t\, \det(\Omega + t H)\big|_{t=0} = \det\Omega \cdot \operatorname{tr}\!\big(\Omega^{-1}H\big),`

where the derivative is the formal `Polynomial.derivative`, the coefficient of `t`. The proof is two
moves. First, factor over `ℝ[X]`:

$$`\det(\Omega + t H) = \det\Omega \cdot \det\!\big(1 + t\,\Omega^{-1}H\big),`

which is just `det_mul` after pulling `Ω` out (`Ω·(1 + t·Ω⁻¹H) = Ω + t·H` because `Ω·Ω⁻¹ = 1`). Second,
read off the linear coefficient of `det(1 + t·M)` — that is Mathlib's `derivative_det_one_add_X_smul`,
the exact statement `det(1 + X•M) = 1 + (\operatorname{tr} M)\,X + O(X^2)`. Composing the two gives the
formula. There is *no* `HasDerivAt`, no normed algebra, no limit — Jacobi's formula is finite linear
algebra over a polynomial ring, exactly the *citation-only, exact-arithmetic* posture S6 established for
the inverse derivative.

# From `∂ det` to `∂ log det`

`∂ log det Ω = (∂ det Ω)/\det\Omega = \operatorname{tr}(\Omega^{-1}H)` — dividing the determinant
derivative by `det Ω` is the cited chain rule for `log`, the only step that is not pure polynomial
algebra (and `det Ω > 0` for SPD `Ω`, so the division is well-posed). The closed form is therefore the
trace `tr(Ω⁻¹H)`, with *no determinant and no inverse ever materialized*: the theorem
`logDetGradFn_eq_trace` proves the spec `logDetGradFn` — a sum of `n` landed Cholesky solves, the `c`-th
recovering the `c`-th diagonal entry of `Ω⁻¹H` — equals exactly this Mathlib trace.

# What S7 surfaced: the determinant's derivative needs no calculus

The textbook derivation of Jacobi's formula goes through limits and the adjugate. S7 shows that for the
purposes of a *verified* gradient, none of that analysis is required: the directional derivative of `det`
is a *single polynomial coefficient*, and Mathlib already proves the one fact needed
(`det(1 + X•M)` has linear coefficient `tr M`). The KernelFlows log-marginal-likelihood gradient — the
last unported loss gradient — reduces to one trace of `Ω⁻¹H`, certified exactly, with the data half
inherited unchanged from S6. KernelFlows' `ρ_MLE` training gradient now rests on a verified core.

# Executable witnesses

The example `NN.Examples.Factorization.KernelMLEGrad` differentiates `ρ_MLE` over the same concrete `4×2`
problem as S6 and closes the loop two independent ways. First, the closed-form Jacobi gradient
`tr(Ω⁻¹ ∂Ω/∂logθ_c)` matches a *central finite difference of the Lean log-determinant*
`log det Ω = 2 Σ log L[i,i]` (`L` the landed Cholesky factor) to `< 10⁻⁴`, and the full `∂ρ_MLE` matches
a finite difference of `ρ_MLE` itself — precisely "the closed form is the derivative of the spec."
Second, `ρ_MLE`, all four `∂ρ_MLE/∂logθ_c`, and all four Jacobi traces match *golden values* from the
independent Julia program. The *negative controls* confirm the test has teeth: the closed form against
the *wrong* component's finite difference, and the *sign-flipped* `∂Ω/∂logθ` in Jacobi's formula
(`tr(Ω⁻¹(−H)) = −tr(Ω⁻¹H)`, off by `≈ 2|tr Ω⁻¹H|`), both fail by a wide margin, and a real Jacobi
component is shown genuinely nonzero.
