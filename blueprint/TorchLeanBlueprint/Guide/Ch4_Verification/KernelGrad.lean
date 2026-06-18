import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "Differentiating the KernelFlows Losses Is Exact Arithmetic (S6)" =>
%%%
tag := "kernelflows-losses-differentiable"
%%%

KernelFlows trains the kernel hyperparameters `logθ` by gradient descent on a cross-validation loss `ρ`
(S3 ported the evaluation of `ρ_KF`, `ρ_LOI`, `ρ_LOO`). Step S6 certifies the *gradient* of those losses.
The whole point is structural: every KernelFlows loss is a rational function of one object — the
regularized inverse quadratic form `yᵀ Ω⁻¹ y` — so its gradient is, by the chain rule, the composition of
two *cited* derivatives and *no new differentiation lemma*. Like the cyclic-Jacobi rate, no
iteration-count *convergence* claim is made; S6 certifies the gradient arithmetic only.

# The gradient factors through two cited pieces

Write `Ω = Ω(logθ)` for the SPD kernel matrix. The two pieces are:

* the *derivative of a matrix inverse*, `∂(Ω⁻¹)/∂t = −Ω⁻¹ (∂Ω/∂t) Ω⁻¹` — Mathlib's Fréchet derivative of
  inversion in a normed algebra (`hasFDerivAt_ring_inverse`), whose *exact finite form* is the resolvent
  identity `A⁻¹ − B⁻¹ = A⁻¹(B − A)B⁻¹` (`Matrix.inv_sub_inv`); and
* the *closed-form kernel derivative* `∂Ω/∂logθ`, ported from KernelFlows' analytic `Matern32_αgrad!`.

Neither is invented here. What S6 proves is the *exact algebra* that sits under the inverse-derivative,
over `ℝ`, with the loss gradients themselves checked against finite-difference / Zygote golden.

# The resolvent identity is the exact `∂(inverse)`

The single primitive is `quadInvGradFn`, the gradient of `yᵀΩ⁻¹y` in a matrix direction `H = ∂Ω/∂logθ`:

$$`\frac{\partial}{\partial t}\, y^\top \Omega^{-1} y = -\,(\Omega^{-1}y)^\top H\,(\Omega^{-1}y) = -\,y^\top \Omega^{-1} H \Omega^{-1} y.`

The theorem `quadInvGradFn_eq_neg_quadForm` proves the closed form equals exactly this standard
inverse-form derivative `−(Ω⁻¹y)ᵀ H (Ω⁻¹y)` — and it materializes *no* inverse: `Ω⁻¹y` is the landed
Cholesky solve of S1. The exact finite difference of the loss is the headline
`quadInvFn_secant_eq`: for SPD `Ω` and `Ω'`,

$$`y^\top \Omega'^{-1} y - y^\top \Omega^{-1} y = -\, y^\top\, \Omega'^{-1}\,(\Omega' - \Omega)\,\Omega^{-1}\, y.`

This is *not* asymptotic — it holds for *every* pair of SPD matrices, a one-line consequence of Mathlib's
`Matrix.inv_sub_inv`. Letting `Ω' → Ω` (so `Ω'⁻¹ → Ω⁻¹`) turns its right-hand side into the closed-form
gradient `quadInvGradFn`. That limit is the only analytic step, and it is the *cited* Mathlib derivative,
not a new lemma — exactly the "citation-only" posture S6 promised.

# The ported `∂Ω/∂logθ` (analytic `Matern32_αgrad!`)

For the S2 build `Ω[i,j] = Matérn32(d; a, b) + δ·[i=j] + L·⟨Φᵢ,Φⱼ⟩`, with `a = e^{logθ₀}`, `b = e^{logθ₁}`,
`L = e^{logθ₂}`, `δ = e^{−12} + e^{logθ₃}`, distance `d = ‖Xᵢ−Xⱼ‖`, and `h = √3·d/b`, differentiating
through `θ = e^{logθ}` (so each derivative carries a factor of its own parameter) gives the four
matrices `kernelMatrixMatern32GradFn` returns:

$$`\frac{\partial \Omega}{\partial \log\theta_0} = a(1+h)e^{-h},\quad
   \frac{\partial \Omega}{\partial \log\theta_1} = a\,h^2 e^{-h},\quad
   \frac{\partial \Omega}{\partial \log\theta_2} = L\,\langle\Phi_i,\Phi_j\rangle,\quad
   \frac{\partial \Omega}{\partial \log\theta_3} = e^{\log\theta_3}[i=j].`

The length-scale derivative `a·h²·e^{−h}` is the one a casual port gets wrong; the `#eval`s pin it (and
the other three) to a finite difference of the actual Lean kernel build to machine precision.

# The loss gradients are quotient-rule assemblies

Each loss gradient is the quotient rule on top of `quadInvGradFn`. `ρ_KF = 1 − N/D` with
`N = y_cᵀΩ_c⁻¹y_c` over the center block and `D = yᵀΩ⁻¹y`, so `∂ρ_KF = (N·∂D − ∂N·D)/D²`; `ρ_LOI` adds
the scalar `Ω₀₀` term; `ρ_LOO` propagates the inverse-derivative `∂(Ω⁻¹) = −Ω⁻¹HΩ⁻¹` (`dInvFn`) through
its full leave-one-out operator `M` (`dLooMFn`). These assemblies are *specs*, not theorems — their
correctness is the `#eval` against the autodiff truth.

# What S6 surfaced: the loss is differentiable without an autodiff engine

KernelFlows ships *two* gradient paths: a Zygote autodiff path and a hand-written analytic path
(`Matern32_αgrad!` + the `ρ_RMSE` chain rule). S6 shows the analytic path is not a separate algorithm to
be trusted on faith — it is the composition of the resolvent identity (proved exactly here) with a
closed-form `∂Ω/∂logθ` (checked exactly here). The inverse-form loss is differentiable *in closed form*,
no autodiff engine in the trusted core; the one transcendental step, the inverse-derivative limit, is a
cited Mathlib theorem.

# Executable witnesses

The example `NN.Examples.Factorization.KernelGrad` differentiates all three losses over a concrete `4×2`
problem and closes the loop two independent ways. First, every closed-form gradient — the four
`∂Ω/∂logθ_c` matrices and all twelve `∂ρ/∂logθ_c` — matches a *central finite difference of the same Lean
loss* to `< 10⁻⁴` (in fact to machine precision), which is precisely the statement "the closed form is the
derivative of the spec." Second, the loss values and all twelve gradients match *golden values* from an
independent Julia program. The *negative controls* confirm the finite-difference test has teeth: the
closed form matched against the *wrong* component's finite difference, and the sign-flipped `∂Ω/∂logθ`,
both fail by a wide margin, and a real gradient component is shown genuinely nonzero so the fixed-point
`quadInvGradFn_zero` is not vacuous.
