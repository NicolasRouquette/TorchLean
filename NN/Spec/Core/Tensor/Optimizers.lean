/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Factorizations

/-!
# The KernelFlows optimizers — AMSGrad and SGD (S5)

KernelFlows trains its kernel hyperparameters `logθ` by gradient descent on the flow loss `ρ`. The two
optimizers it ships ([`optimizers.jl`](../../../../../KernelFlows.jl/src/optimizers.jl)) both follow the
`iterate!(O, g)` convention — update internal state from the gradient `g`, return the new parameters:

* **`AMSGrad`** — the AMSGrad variant of Adam (Reddi–Kale–Kumar, 2018), with one twist worth recording:
  KernelFlows keeps the **second moment as a scalar**, not a per-coordinate vector. `iterate!` reads

  ```julia
  O.m    = β1 * O.m + (1 - β1) * g        # first moment   (vector)
  O.v    = β2 * O.v + (1 - β2) * dot(g,g)  # second moment  (SCALAR ‖g‖²)
  O.vhat = max(O.vhat, O.v)               # running max    (the AMSGrad fix)
  O.x  .-= ϵ .* O.m / (sqrt(O.vhat) + δ)  # parameter step
  ```

  So the adaptive denominator `√v̂ + δ` is a single shared scale (a normalized-gradient AMSGrad), not the
  elementwise `√v̂ᵢ + δ` of textbook Adam/AMSGrad. This is the bit that distinguishes it from the
  elementwise `Optim.Adam`/`Optim.AdamW` family in [`NN/Runtime/Optim/Optimizers.lean`](../../../Runtime/Optim/Optimizers.lean);
  we port it faithfully here so the `#eval` matches `optimizers.jl` to machine precision.

* **`SGD`** — plain gradient descent with an optional `fixed`-step mode. With `fixed = true` (the
  default) the gradient is normalized by `√(‖g‖² + 10⁻⁹)`, so every step has length `≈ ϵ` regardless of
  the gradient magnitude (a trust-region step); with `fixed = false` it is the ordinary `x ← x − ϵ g`.

*Scope (S5).* This is the **exact update spec** plus its arithmetic correctness facts, proved over `ℝ` in
[`NN.Proofs.Tensor.Basic.FactorizationsOptimizers`](../../../Proofs/Tensor/Basic/FactorizationsOptimizers.lean):
the AMSGrad running-max invariant `v̂` is monotone non-decreasing (the property that fixes Adam's
non-convergence), the documented parameter step is reproduced verbatim, and the fixed-step SGD
displacement is bounded by `ϵ` (the trust region). Non-convex *convergence* is **not** claimed as a
theorem — only the a-posteriori certificate that the iterations' fixed points are exactly the stationary
points (`IsApproxStationary`), the same honest posture S2–S4 took toward the cyclic-Jacobi rate. Numeric
`#eval`s check both optimizers against `optimizers.jl` golden trajectories.
-/

@[expose] public section

namespace Spec

variable {α : Type} [Context α]
variable {n : Nat}

/-! ## AMSGrad (scalar second moment) -/

/-- **KernelFlows `AMSGrad` state** (`optimizers.jl`). The first moment `m` is a length-`n` vector; the
second moment `v` and its running max `vhat` are **scalars** (`v = β₂v + (1−β₂)‖g‖²`). `lr` is the
learning rate `ϵ`, `reg` the denominator regularizer `δ`. -/
structure AMSGradState (α : Type) (n : Nat) where
  /-- Parameter vector `x`. -/
  x : Fin n → α
  /-- First-moment EMA `m` (vector). -/
  m : Fin n → α
  /-- Second-moment EMA `v` (scalar `‖g‖²` average). -/
  v : α
  /-- Running max of `v` (the AMSGrad correction). -/
  vhat : α
  /-- Learning rate `ϵ`. -/
  lr : α
  /-- First-moment decay `β₁`. -/
  beta1 : α
  /-- Second-moment decay `β₂`. -/
  beta2 : α
  /-- Denominator regularizer `δ`. -/
  reg : α

/-- Standard AMSGrad initializer: `m = 0`, `v = 0`, `vhat = 0` (`AMSGrad(x_start; …)`). -/
def amsGradInit (x : Fin n → α) (lr beta1 beta2 reg : α) : AMSGradState α n :=
  { x := x, m := fun _ => 0, v := 0, vhat := 0, lr := lr, beta1 := beta1, beta2 := beta2, reg := reg }

/-- **One `AMSGrad` step** (`iterate!(O::AMSGrad, g)`), bit-faithful to `optimizers.jl`:
`m ← β₁m + (1−β₁)g`, `v ← β₂v + (1−β₂)‖g‖²`, `v̂ ← max(v̂, v)`, `x ← x − ϵ·m / (√v̂ + δ)`. -/
def amsGradStep (S : AMSGradState α n) (g : Fin n → α) : AMSGradState α n :=
  let m' : Fin n → α := fun i => S.beta1 * S.m i + (1 - S.beta1) * g i
  let v' : α := S.beta2 * S.v + (1 - S.beta2) * dotFn g g
  let vhat' : α := max S.vhat v'
  let x' : Fin n → α := fun i => S.x i - S.lr * m' i / (MathFunctions.sqrt vhat' + S.reg)
  { S with x := x', m := m', v := v', vhat := vhat' }

/-! ## SGD (with optional fixed-step normalization) -/

/-- **KernelFlows `SGD` state** (`optimizers.jl`). `lr` is the step size `ϵ`; `fixed` selects the
normalized trust-region step; `stab` is the `√` stabilizer (Julia hardcodes `T(1e-9)`). -/
structure SGDState (α : Type) (n : Nat) where
  /-- Parameter vector `x`. -/
  x : Fin n → α
  /-- Learning rate `ϵ`. -/
  lr : α
  /-- If `true`, normalize the gradient so each step has length `≈ ϵ`. -/
  fixed : Bool
  /-- `√`-argument stabilizer (Julia uses `1e-9`). -/
  stab : α

/-- SGD initializer (Julia defaults: `ϵ = 1e-3`, `fixed = true`, stabilizer `1e-9`). -/
def sgdInit (x : Fin n → α) (lr : α) (fixed : Bool) (stab : α) : SGDState α n :=
  { x := x, lr := lr, fixed := fixed, stab := stab }

/-- The SGD step-size normalizer `α`: `√(‖g‖² + stab)` in fixed mode, `1` otherwise. -/
def sgdNorm (S : SGDState α n) (g : Fin n → α) : α :=
  if S.fixed then MathFunctions.sqrt (dotFn g g + S.stab) else 1

/-- The SGD parameter displacement `Δx = (ϵ / α) · g` (so `x ← x − Δx`). -/
def sgdDisplacement (S : SGDState α n) (g : Fin n → α) : Fin n → α :=
  fun i => S.lr / sgdNorm S g * g i

/-- **One `SGD` step** (`iterate!(O::SGD, g)`): `x ← x − (ϵ / α) · g`, with `α = sgdNorm`. -/
def sgdStep (S : SGDState α n) (g : Fin n → α) : SGDState α n :=
  { S with x := fun i => S.x i - sgdDisplacement S g i }

end Spec
