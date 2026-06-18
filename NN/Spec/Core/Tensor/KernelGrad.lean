/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.KernelMatrix
public import NN.Spec.Core.Tensor.KernelLoss

/-!
# Differentiating the KernelFlows losses — inverse-form gradients + `ρ_MLE` (S6/S7)

KernelFlows trains the kernel hyperparameters `logθ` by gradient descent on a cross-validation loss `ρ`
(S3: `ρ_KF`, `ρ_LOI`, `ρ_LOO`, `ρ_MLE`). The three cross-validation losses are rational functions of the
**regularized inverse quadratic form** `yᵀ Ω⁻¹ y` (`quadInvFn`), so their gradient is, by the chain rule,
the composition of two pieces — and **neither is a new differentiation lemma**:

* the **derivative of a matrix inverse**, `∂(Ω⁻¹)/∂t = −Ω⁻¹ (∂Ω/∂t) Ω⁻¹` (Mathlib's Fréchet derivative
  of inversion in a normed algebra, `hasFDerivAt_ring_inverse`; its exact finite form is the *resolvent
  identity* `A⁻¹ − B⁻¹ = A⁻¹(B − A)B⁻¹`, `Matrix.inv_sub_inv`); and
* the **closed-form kernel derivative** `∂Ω/∂logθ`, ported here from KernelFlows' analytic
  `Matern32_αgrad!` (`src/kernel_functions_analytic.jl`).

This file is the **evaluation spec** for those gradients. The single primitive is `quadInvGradFn`, the
gradient of `yᵀ Ω⁻¹ y` in a matrix direction `H = ∂Ω/∂logθ`:
`∂(yᵀΩ⁻¹y) = −(Ω⁻¹y)ᵀ H (Ω⁻¹y) = −yᵀ Ω⁻¹ H Ω⁻¹ y`. The three loss gradients are quotient-rule
assemblies on top of it. The companion proofs
([`NN.Proofs.Tensor.Basic.FactorizationsKernelGrad`](../../../Proofs/Tensor/Basic/FactorizationsKernelGrad.lean))
prove the exact algebraic core — the resolvent secant identity and that `quadInvGradFn` is the
`−Ω⁻¹HΩ⁻¹` quadratic form — over `ℝ`; the full loss gradients are checked against finite-difference /
Zygote golden by `#eval` (`NN.Examples.Factorization.KernelGrad`). Following the S2–S5 posture, no
iteration-count *convergence* claim is made; this step certifies the *gradient arithmetic* only.

## `ρ_MLE` and Jacobi's formula (S7)

`ρ_MLE = ½ yᵀΩ⁻¹y + ½ log det Ω` carries one term the cross-validation losses do not — the
log-determinant. Its gradient `∂(log det Ω) = tr(Ω⁻¹ ∂Ω)` (`logDetGradFn`) is **Jacobi's formula**, the
single genuinely new differentiation lemma in the whole KernelFlows gradient: the companion proof derives
it from the *exact* polynomial identity `∂_t det(Ω + tH)|₀ = det Ω · tr(Ω⁻¹H)` (Mathlib's
`derivative_det_one_add_X_smul`, the coefficient-of-`t` form — no analysis), then `∂ log = ∂det/det`. The
data half of `∂ρ_MLE` reuses S6's `quadInvGradFn` verbatim, so `rhoMLEGradFn = ½ quadInvGradFn + ½
logDetGradFn`.

## The ported `∂Ω/∂logθ` (analytic `Matern32_αgrad!`)

For the S2 build `Ω[i,j] = Matérn32(d; a, b) + δ·[i=j] + L·⟨Φᵢ,Φⱼ⟩` with `a = e^{logθ₀}`, `b = e^{logθ₁}`,
`L = e^{logθ₂}`, `δ = e^{−12} + e^{logθ₃}`, `d = ‖Xᵢ−Xⱼ‖`, `h = √3·d/b`, the log-parameter derivatives are
(differentiating through `θ = e^{logθ}`, so each picks up a factor of the parameter itself):

* `∂Ω/∂logθ₀ = a(1+h)e^{−h} = Matérn32(d; a, b)` — the radial term (KernelFlows `Kgrads[a]`);
* `∂Ω/∂logθ₁ = a·h²·e^{−h}` — length-scale derivative (`∂_b · b`);
* `∂Ω/∂logθ₂ = L·⟨Φᵢ,Φⱼ⟩` — the linear term;
* `∂Ω/∂logθ₃ = e^{logθ₃}·[i=j]` — the nugget (diagonal only).
-/

@[expose] public section

namespace Spec

variable {α : Type} [Context α]
variable {n d : Nat}

/-! ## The closed-form kernel derivative `∂Ω/∂logθ` (ported `Matern32_αgrad!`) -/

/-- `∂Ω/∂logθ₀` — derivative w.r.t. the log Matérn **weight** `logθ₀` (`a = e^{logθ₀}`). Since `Ω` is
linear in `a`, this is exactly the radial term `a(1+h)e^{−h} = Matérn32(d; a, b)` (KernelFlows
`Kgrads[nλ+1] = Dbuf`). -/
def dKMatern32_dLogWeight (X : Fin n → Fin d → α) (logθ : Fin 4 → α) : Fin n → Fin n → α :=
  fun i j => matern32Fn (pairwiseEuclideanFn X i j) (MathFunctions.exp (logθ 0)) (MathFunctions.exp (logθ 1))

/-- `∂Ω/∂logθ₁` — derivative w.r.t. the log **length scale** `logθ₁` (`b = e^{logθ₁}`). With `h = √3·d/b`,
`∂Ω/∂b = a·h²·e^{−h}/b`, so `∂Ω/∂logθ₁ = b·∂Ω/∂b = a·h²·e^{−h}`. -/
def dKMatern32_dLogScale (X : Fin n → Fin d → α) (logθ : Fin 4 → α) : Fin n → Fin n → α :=
  fun i j =>
    let b := MathFunctions.exp (logθ 1)
    let h := MathFunctions.sqrt Numbers.three * pairwiseEuclideanFn X i j / b
    MathFunctions.exp (logθ 0) * (h * h) * MathFunctions.exp (-h)

/-- `∂Ω/∂logθ₂` — derivative w.r.t. the log **linear weight** `logθ₂` (`L = e^{logθ₂}`). `Ω` is linear in
`L`, so this is the linear term `L·⟨Φᵢ,Φⱼ⟩`. -/
def dKMatern32_dLogLinear (X : Fin n → Fin d → α) (wlin : Fin d → α) (logθ : Fin 4 → α) :
    Fin n → Fin n → α :=
  fun i j => MathFunctions.exp (logθ 2) * dotFn (maskColsFn X wlin i) (maskColsFn X wlin j)

/-- `∂Ω/∂logθ₃` — derivative w.r.t. the log **nugget** `logθ₃` (`δ = e^{−12} + e^{logθ₃}`). Only the
learned ridge `e^{logθ₃}` depends on `logθ₃`, and only on the diagonal. -/
def dKMatern32_dLogNugget (logθ : Fin 4 → α) : Fin n → Fin n → α :=
  fun i j => if i = j then MathFunctions.exp (logθ 3) else 0

/-- Dispatcher for the four log-hyperparameter derivative matrices `∂Ω/∂logθ_c`, `c ∈ {0,1,2,3}`
(the ported `Matern32_αgrad!` returning the `c`-th `Kgrads`). -/
def kernelMatrixMatern32GradFn (X : Fin n → Fin d → α) (wlin : Fin d → α) (logθ : Fin 4 → α)
    (c : Fin 4) : Fin n → Fin n → α :=
  if c = 0 then dKMatern32_dLogWeight X logθ
  else if c = 1 then dKMatern32_dLogScale X logθ
  else if c = 2 then dKMatern32_dLogLinear X wlin logθ
  else dKMatern32_dLogNugget logθ

/-! ## The inverse-form gradient primitive `∂(yᵀΩ⁻¹y)` -/

/-- **The inverse-form gradient.** The gradient of the regularized quadratic form `yᵀ Ω⁻¹ y`
(`quadInvFn`) in the matrix direction `H = ∂Ω/∂t` is `−(Ω⁻¹y)ᵀ H (Ω⁻¹y)`: differentiate `yᵀΩ⁻¹y` through
`∂(Ω⁻¹) = −Ω⁻¹HΩ⁻¹`, and `Ω⁻¹y = cholSolveFn (choleskyFn Ω) y` is the landed Cholesky solve. So no
inverse is ever formed — one solve `s = Ω⁻¹y`, then `−sᵀ H s`. -/
def quadInvGradFn (Ω : Fin n → Fin n → α) (H : Fin n → Fin n → α) (y : Fin n → α) : α :=
  let s := cholSolveFn (choleskyFn Ω) y;
  -bilinFn H s

/-- The directional derivative of the inverse, `∂(Ω⁻¹) = −Ω⁻¹ H Ω⁻¹`, evaluated entrywise from the
column-wise inverse `W = Ω⁻¹` (`invCholFn`): `(−Ω⁻¹HΩ⁻¹)[a,b] = −Σ_{p,q} W[a,p] H[p,q] W[q,b]`. Only
`ρ_LOO`, whose operator `M` depends on the full inverse, needs the inverse-derivative as a *matrix*. -/
def dInvFn (Ω : Fin n → Fin n → α) (H : Fin n → Fin n → α) : Fin n → Fin n → α :=
  let W := invCholFn Ω;
  fun a b => -dotFn (fun p => W a p) (fun p => dotFn (H p) (fun q => W q b))

/-! ## Loss gradients (chain rule on `quadInvGradFn`) -/

/-- **`∂ρ_LOI/∂logθ`.** `ρ_LOI = 1 − c/D`, `c = (yᵀy/Ω₀₀)/n`, `D = yᵀΩ⁻¹y`. With `∂c = −(yᵀy/n)·H₀₀/Ω₀₀²`
and `∂D = quadInvGradFn`, the quotient rule gives `∂ρ_LOI = (c·∂D − ∂c·D)/D²`. -/
def rhoLOIGradFn {m : Nat} (Ω : Fin (m + 1) → Fin (m + 1) → α) (H : Fin (m + 1) → Fin (m + 1) → α)
    (y : Fin (m + 1) → α) : α :=
  let D := quadInvFn Ω y;
  let dD := quadInvGradFn Ω H y;
  let c := dotFn y y / Ω 0 0 / ((m + 1 : Nat) : α);
  let dc := -(dotFn y y / ((m + 1 : Nat) : α)) / (Ω 0 0 * Ω 0 0) * H 0 0;
  (c * dD - dc * D) / (D * D)

/-- **`∂ρ_KF/∂logθ`.** `ρ_KF = 1 − N/D`, `N = y_cᵀΩ_c⁻¹y_c` (the center block via `e`), `D = yᵀΩ⁻¹y`. The
quotient rule on the two inverse-form gradients gives `∂ρ_KF = (N·∂D − ∂N·D)/D²`. -/
def rhoKFGradFn {n nc : Nat} (Ω : Fin n → Fin n → α) (H : Fin n → Fin n → α) (y : Fin n → α)
    (e : Fin nc → Fin n) : α :=
  let Ωc := fun i j => Ω (e i) (e j);
  let Hc := fun i j => H (e i) (e j);
  let yc := fun i => y (e i);
  let N := quadInvFn Ωc yc;
  let dN := quadInvGradFn Ωc Hc yc;
  let D := quadInvFn Ω y;
  let dD := quadInvGradFn Ω H y;
  (N * dD - dN * D) / (D * D)

/-- The derivative of the leave-one-out operator `M = N·W − Σ_l (W e_l)(W e_l)ᵀ/W[l,l]` (`looMFn`) given
`W = Ω⁻¹` and its directional derivative `dW = −Ω⁻¹HΩ⁻¹` (`dInvFn`): the product/quotient rule applied
entrywise to each term of `M`. -/
def dLooMFn (W : Fin n → Fin n → α) (dW : Fin n → Fin n → α) : Fin n → Fin n → α :=
  fun a b => ((n : Nat) : α) * dW a b
    - (List.finRange n).foldl
        (fun s l =>
          s + ((dW a l * W b l + W a l * dW b l) / W l l
                - W a l * W b l * dW l l / (W l l * W l l))) 0

/-- **`∂ρ_LOO/∂logθ`.** `ρ_LOO = N − B/D`, `B = yᵀMy`, `M = looMFn(Ω⁻¹)`, `D = yᵀΩ⁻¹y`. Propagating
`∂(Ω⁻¹) = dInvFn` through `M` (`dLooMFn`) and the shared denominator (`quadInvGradFn`), the quotient rule
gives `∂ρ_LOO = −(∂B·D − B·∂D)/D²`. -/
def rhoLOOGradFn (Ω : Fin n → Fin n → α) (H : Fin n → Fin n → α) (y : Fin n → α) : α :=
  let W := invCholFn Ω;
  let dW := dInvFn Ω H;
  let M := looMFn W;
  let dM := dLooMFn W dW;
  let B := bilinFn M y;
  let dB := bilinFn dM y;
  let D := quadInvFn Ω y;
  let dD := quadInvGradFn Ω H y;
  (B * dD - dB * D) / (D * D)

/-! ## `ρ_MLE` gradient — Jacobi's formula (S7) -/

/-- **The log-determinant gradient (Jacobi's formula).** `∂(log det Ω)/∂t = tr(Ω⁻¹ ∂Ω/∂t)`. Evaluated
through the landed Cholesky solve without forming `Ω⁻¹`: the `c`-th diagonal entry of `Ω⁻¹H` is
`(Ω⁻¹ (H·e_c))[c]`, the `c`-th component of the solve of `Ω·x = (H·e_c)` (column `c` of `H`), so the
trace is `Σ_c (cholSolveFn (choleskyFn Ω) (fun k => H k c)) c`. This is the one genuinely new
differentiation piece in KernelFlows — every other loss gradient is the inverse-form `quadInvGradFn`. -/
def logDetGradFn (Ω : Fin n → Fin n → α) (H : Fin n → Fin n → α) : α :=
  let L := choleskyFn Ω;
  (List.finRange n).foldl (fun s c => s + cholSolveFn L (fun k => H k c) c) 0

/-- **`∂ρ_MLE/∂logθ`.** `ρ_MLE = ½ yᵀΩ⁻¹y + ½ log det Ω` (the GP negative log marginal likelihood, S3),
so its gradient is `½` the inverse-form gradient `quadInvGradFn` (the data term) plus `½` Jacobi's
log-determinant gradient `logDetGradFn` (the complexity term). The *only* loss whose gradient needs the
new `∂ log det` lemma; the data half reuses S6's primitive verbatim. -/
def rhoMLEGradFn (Ω : Fin n → Fin n → α) (H : Fin n → Fin n → α) (y : Fin n → α) : α :=
  Numbers.pointfive * quadInvGradFn Ω H y + Numbers.pointfive * logDetGradFn Ω H

/-! ## Tensor-level wrappers -/

/-- Tensor-level inverse-form gradient `∂(yᵀΩ⁻¹y)` in direction `H`. -/
def quadInvGradSpec (Ω : Tensor α (.dim n (.dim n .scalar))) (H : Tensor α (.dim n (.dim n .scalar)))
    (y : Tensor α (.dim n .scalar)) : α :=
  quadInvGradFn (toMatFn Ω) (toMatFn H) (toVecFn y)

/-- Tensor-level `∂Ω/∂logθ_c` (the ported `Matern32_αgrad!` `c`-th gradient matrix). -/
def kernelMatrixMatern32GradSpec (X : Tensor α (.dim n (.dim d .scalar)))
    (wlin : Tensor α (.dim d .scalar)) (logθ : Tensor α (.dim 4 .scalar)) (c : Fin 4) :
    Tensor α (.dim n (.dim n .scalar)) :=
  ofMatFn (kernelMatrixMatern32GradFn (toMatFn X) (toVecFn wlin) (toVecFn logθ) c)

/-- Tensor-level `ρ_MLE` gradient in direction `H = ∂Ω/∂logθ`. -/
def rhoMLEGradSpec (Ω : Tensor α (.dim n (.dim n .scalar))) (H : Tensor α (.dim n (.dim n .scalar)))
    (y : Tensor α (.dim n .scalar)) : α :=
  rhoMLEGradFn (toMatFn Ω) (toMatFn H) (toVecFn y)

end Spec
