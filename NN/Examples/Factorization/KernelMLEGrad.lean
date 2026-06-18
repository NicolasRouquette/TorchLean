/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Factorization.Common
public import NN.Examples.Factorization.KernelGrad
public import NN.Spec.Core.Tensor.KernelGrad
meta import NN.Examples.Factorization.Common
meta import NN.Examples.Factorization.KernelGrad
meta import NN.Spec.Core.Tensor.KernelGrad

/-!
# Differentiating `ρ_MLE` — Jacobi's formula (S7 examples)

S7 differentiates the maximum-likelihood loss `ρ_MLE = ½ yᵀΩ⁻¹y + ½ log det Ω`. The data term reuses
S6's inverse-form gradient `quadInvGradFn`; the complexity term needs the one genuinely new lemma,
**Jacobi's formula** `∂(log det Ω) = tr(Ω⁻¹ ∂Ω)` (`logDetGradFn`), proved exactly in
[`FactorizationsKernelMLEGrad`](../../../Proofs/Tensor/Basic/FactorizationsKernelMLEGrad.lean) via the
polynomial identity `∂_t det(Ω+tH)|₀ = det Ω · tr(Ω⁻¹H)`.

These `#eval`s close the loop *numerically* on the same concrete `4×2` problem as S6 (reusing its `X`,
`wlin`, `y`, `logθ`, `Ω`, `Hc`, `ε`, `perturb`, `Kof`, `fdLoss`):

* **Jacobi check** — `logDetGradFn Ω (∂Ω/∂logθ_c)` matches a central finite difference of the *Lean*
  log-determinant `log det Ω = 2 Σ log L[i,i]` (`L` the landed Cholesky factor) to `< 10⁻⁴`. This is
  exactly "the closed-form `tr(Ω⁻¹H)` is the derivative of `log det`."
* **`∂ρ_MLE` check** — every `∂ρ_MLE/∂logθ_c` matches the finite difference of `ρ_MLE` itself.
* **Golden cross-check** — `ρ_MLE`, all four `∂ρ_MLE/∂logθ_c`, and all four Jacobi `tr(Ω⁻¹H)` match
  golden values from the independent Julia program.

The **negative controls** confirm teeth: the closed form against the *wrong* component's finite
difference, and the *sign-flipped* `∂Ω/∂logθ` in Jacobi's formula, both fail by a wide margin.
-/

@[expose] public section

namespace NN.Examples.Factorization.KernelMLEGrad

open NN.Examples.Factorization
open NN.Examples.Factorization.KernelGrad

/-! ### `ρ_MLE`, its gradient, and the log-determinant -/

/-- `ρ_MLE` at a given log-parameter vector. -/
def rhoMLE (lt : Fin 4 → Float) : Float := Spec.rhoMLEFn (Kof lt) yv

/-- The closed-form `∂ρ_MLE/∂logθ_c = ½ quadInvGradFn + ½ logDetGradFn`. -/
def mleGrad (c : Fin 4) : Float := Spec.rhoMLEGradFn Ω (Hc c) yv

/-- The log-determinant of `Ω` from the landed Cholesky factor: `log det Ω = 2 Σ_i log L[i,i]`. -/
def logDetOf (lt : Fin 4 → Float) : Float :=
  let L := Spec.choleskyFn (Kof lt)
  2.0 * (List.finRange 4).foldl (fun s i => s + Float.log (L i i)) 0.0

/-- The closed-form Jacobi log-determinant gradient `tr(Ω⁻¹ ∂Ω/∂logθ_c)`. -/
def jacGrad (c : Fin 4) : Float := Spec.logDetGradFn Ω (Hc c)

#eval IO.println s!"ρ_MLE = {rhoMLE logθ}"
#eval IO.println s!"∂ρ_MLE/∂logθ = {(List.finRange 4).map mleGrad}"
#eval IO.println s!"∂(log det Ω)/∂logθ = {(List.finRange 4).map jacGrad}"

/-! ### Positive — closed forms vs central finite differences (the spec's own derivatives) -/

-- Jacobi's formula: `tr(Ω⁻¹ ∂Ω/∂logθ_c)` is the derivative of the Lean log-determinant, all components.
#eval assertLt "∂(log det Ω)/∂logθ (Jacobi tr Ω⁻¹H) vs finite difference"
  (gradErr jacGrad logDetOf) 1e-4

-- The full `∂ρ_MLE` matches the finite difference of `ρ_MLE`.
#eval assertLt "∂ρ_MLE/∂logθ (closed form) vs finite difference" (gradErr mleGrad rhoMLE) 1e-4

/-! ### Positive — values and gradients vs the independent Julia golden -/

#eval assertApproxEq "ρ_MLE value vs Julia golden" (rhoMLE logθ) 1.4979403138414304 1e-9

-- ∂ρ_MLE/∂logθ_c (the "#eval vs Zygote golden" deliverable):
#eval assertApproxEq "∂ρ_MLE/∂logθ₀ vs golden" (mleGrad 0)  0.16775225202021193 1e-8
#eval assertApproxEq "∂ρ_MLE/∂logθ₁ vs golden" (mleGrad 1)  0.19756207643561197 1e-8
#eval assertApproxEq "∂ρ_MLE/∂logθ₂ vs golden" (mleGrad 2)  0.24616224756033592 1e-8
#eval assertApproxEq "∂ρ_MLE/∂logθ₃ vs golden" (mleGrad 3) (-0.08192102319386413) 1e-8

-- Jacobi tr(Ω⁻¹H) components vs golden:
#eval assertApproxEq "∂(log det Ω)/∂logθ₀ vs golden" (jacGrad 0)  2.3660308286138316  1e-8
#eval assertApproxEq "∂(log det Ω)/∂logθ₁ vs golden" (jacGrad 1) (-1.606813947001336) 1e-8
#eval assertApproxEq "∂(log det Ω)/∂logθ₂ vs golden" (jacGrad 2)  0.828785581353821   1e-8
#eval assertApproxEq "∂(log det Ω)/∂logθ₃ vs golden" (jacGrad 3)  0.8051470364134461  1e-8

/-! ### Negative controls — the finite-difference test is not vacuous -/

-- The closed form against the WRONG component's finite difference fails by a wide margin
-- (`∂ρ_MLE/∂logθ₀ ≈ 0.168` vs the `logθ₂` finite difference `≈ 0.246`).
#eval assertGe "∂ρ_MLE/∂logθ₀ ≠ finite difference in wrong direction (logθ₂)"
  (Float.abs (mleGrad 0 - fdLoss rhoMLE 2)) 0.05

-- The sign-flipped `∂Ω/∂logθ₀` is NOT the log-det derivative (Jacobi's formula sign matters):
-- `tr(Ω⁻¹(−H)) = −tr(Ω⁻¹H)`, off the finite difference by `≈ 2|tr Ω⁻¹H| ≈ 4.7`.
#eval assertGe "−∂Ω/∂logθ₀ is not the log-det derivative (Jacobi sign matters)"
  (Float.abs (Spec.logDetGradFn Ω (fun i j => -(Hc 0) i j) - fdLoss logDetOf 0)) 0.1

-- A real Jacobi component is genuinely nonzero — the closed form is not trivially zero.
#eval assertGe "∂(log det Ω)/∂logθ₀ is genuinely nonzero (Jacobi not vacuous)"
  (Float.abs (jacGrad 0)) 0.5

end NN.Examples.Factorization.KernelMLEGrad
