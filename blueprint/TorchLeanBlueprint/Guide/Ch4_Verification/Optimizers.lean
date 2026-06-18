import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "The KernelFlows Optimizer Steps Are Arithmetically Correct (S5)" =>
%%%
tag := "kernelflows-optimizers-correct"
%%%

KernelFlows trains its kernel hyperparameters `logθ` by gradient descent on the flow loss `ρ`, using one
of two optimizers (`optimizers.jl`). This step (S5) ports both as exact specs and certifies their
*update arithmetic*. It is independent of the kernel/PSD/UQ chain (S1–S4) — it only needs gradient
vectors — so it lands as a standalone Year-1 increment. As with the cyclic-Jacobi rate, non-convex
*convergence* is recorded as a certificate, never claimed as a theorem.

# AMSGrad keeps the second moment as a *scalar*

KernelFlows' `AMSGrad` is the AMSGrad variant of Adam, with one detail a casual port would miss: the
second moment is a *scalar*, not a per-coordinate vector. `iterate!` reads

$$`m \leftarrow \beta_1 m + (1-\beta_1)g,\quad v \leftarrow \beta_2 v + (1-\beta_2)\lVert g\rVert^2,\quad
   \hat v \leftarrow \max(\hat v, v),\quad x \leftarrow x - \frac{\epsilon\, m}{\sqrt{\hat v} + \delta}.`

So the adaptive denominator `√v̂ + δ` is a single *shared* scale (a normalized-gradient AMSGrad), not
the elementwise `√v̂ᵢ + δ` of textbook Adam. This is exactly the bit that distinguishes it from the
elementwise `Optim.Adam`/`Optim.AdamW` family already in the runtime; `condVarFullFn`-style, S5 ports the
*KernelFlows* rule faithfully so the executable `#eval` matches `optimizers.jl` to machine precision.

The defining property of AMSGrad over Adam is the running max `v̂ ← max(v̂, v)`: the second-moment scale
never shrinks, which is what restores convergence where plain Adam can diverge (Reddi–Kale–Kumar 2018).
We prove that invariant — `amsGradStep_vhat_mono` (`v̂` non-decreasing across a step) and
`amsGradStep_vhat_ge_v` (`v ≤ v̂`) — both one-line consequences of `le_max_left`/`le_max_right`. The
bit-faithful parameter step is reproduced verbatim (`amsGradStep_x_apply`, using the *updated* moment and
`v̂`), and a zero gradient with zeroed momentum is shown to be a fixed point
(`amsGradStep_eq_of_grad_moment_zero`).

# Fixed-step SGD is a trust region

`SGD` is plain gradient descent with an optional `fixed`-step mode. With `fixed = false` it is the
ordinary `x ← x − ϵ g` (`sgdStep_nonfixed_apply`). The interesting mode is `fixed = true`: the gradient is
normalized by `√(‖g‖² + δ)`, so the displacement is

$$`\Delta x = \frac{\epsilon}{\sqrt{\lVert g\rVert^2 + \delta}}\, g, \qquad
   \lVert \Delta x\rVert^2 = \frac{\epsilon^2\,\lVert g\rVert^2}{\lVert g\rVert^2 + \delta} \le \epsilon^2.`

That bound is `sgdStep_fixed_displacement_normSq_le`: the fixed step is a *trust region* of radius `ϵ` —
the move length approaches `ϵ` as `‖g‖ → ∞` and `0` as `‖g‖ → 0`, never exceeding `ϵ`. The proof pulls
the square out of the self dot product (`dotFn_smul_self`), uses `√(·)² = ·` on the nonnegative argument,
and finishes with `‖g‖² ≤ ‖g‖² + δ`. A zero gradient is a fixed point (`sgdStep_eq_of_grad_zero`).

# Convergence is a certificate, never a theorem

KernelFlows' flow loss `ρ` is non-convex, and Mathlib v4.30.0 has no first-order non-convex convergence
theory. Following the same honest posture S2–S4 took toward the cyclic-Jacobi rate, S5 makes *no* global
convergence claim. The single convergence statement is the a-posteriori certificate `IsApproxStationary`
(`‖∇ρ(x)‖ ≤ ε`); we record that a zero gradient certifies `0`-stationarity
(`isApproxStationary_of_grad_zero`) — the same condition under which both `sgdStep` and `amsGradStep`
leave the parameters fixed. That is the precise, provable statement. Iteration-count rates are out of
scope and would be false in general for non-convex `ρ`.

# What S5 surfaced: the second moment is a scalar, and "fixed" means a trust region

Two facts came out of forcing the `#eval` to match `optimizers.jl` bit-for-bit. First, KernelFlows'
AMSGrad is *not* the elementwise Adam family already in the runtime — it normalizes by a single scalar
`√v̂`, so the whole gradient is rescaled by one number per step. Second, the `fixed = true` SGD default is
not "a fixed learning rate" but a *fixed step length*: the `√(‖g‖² + 10⁻⁹)` normalizer turns every step
into a trust-region move of radius `≈ ϵ`. The `10⁻⁹` floor is the SGD analogue of S1's `5·eps` distance
floor — a stabilizer that keeps the `√` argument positive when `g → 0` (and the step → 0), not part of
the mathematical rule.

# Executable witnesses

The example `NN.Examples.Factorization.Optimizers` replays both optimizers over a shared 3-step gradient
trajectory and checks the resulting state against *golden values* from running the verbatim `iterate!`
rules in Julia. The AMSGrad trajectory reproduces `x`, the first moment `m`, and the scalar `v = v̂` to
machine precision (the step is pure IEEE arithmetic); `v̂` is confirmed non-decreasing; the fixed-step
length sits at `≈ ϵ` (`0.0999999996 ≤ ϵ = 0.1`, the trust-region bound with equality in the limit); and
the plain-SGD step reproduces `x − ϵ g` exactly. The *negative controls* confirm that fixed-step
normalization genuinely changes the trajectory (it is not a no-op versus plain SGD) and that a nonzero
gradient genuinely moves the parameters (so the fixed-point property is not vacuous).
