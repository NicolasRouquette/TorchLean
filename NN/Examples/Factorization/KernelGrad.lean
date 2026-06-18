/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Factorization.Common
public import NN.Spec.Core.Tensor.KernelGrad
meta import NN.Examples.Factorization.Common
meta import NN.Spec.Core.Tensor.KernelGrad

/-!
# Differentiating the KernelFlows losses (S6 examples)

S6 ports KernelFlows' closed-form kernel derivative `∂Ω/∂logθ` (analytic `Matern32_αgrad!`) and assembles
the gradients of the inverse-form losses `ρ_KF`, `ρ_LOI`, `ρ_LOO` from the single primitive
`quadInvGradFn` (`∂(yᵀΩ⁻¹y) = −(Ω⁻¹y)ᵀ H (Ω⁻¹y)`). The proofs
([`FactorizationsKernelGrad`](../../../Proofs/Tensor/Basic/FactorizationsKernelGrad.lean)) certify the
exact algebra — the resolvent secant identity and the closed-form inverse-form gradient.

These `#eval`s close the loop *numerically* in two independent ways over a concrete `4×2` example
(`X`, `wlin`, `y`, `logθ`; the `ρ_KF` center block is the leading `nc = 2`):

* **Self-contained gradient check** — every closed-form gradient is compared against a **central finite
  difference** of the *same Lean loss* (`∂Ω/∂logθ_c` vs `(K(logθ+εe_c) − K(logθ−εe_c))/2ε`, and each
  `∂ρ/∂logθ_c` vs the finite difference of `ρ`). Agreeing to `< 10⁻⁴` is exactly "the closed form is the
  derivative of the spec" — the Zygote-equivalent autodiff ground truth.
* **Golden cross-check** — the loss values and all twelve `∂ρ/∂logθ_c` match **golden values** produced
  by an independent Julia program replicating the same math (finite-difference / Zygote ground truth).

The **negative controls** confirm the finite-difference test has teeth: the closed form matched against
the *wrong* component's finite difference, and the sign-flipped `∂Ω/∂logθ`, both fail by a wide margin.
-/

@[expose] public section

namespace NN.Examples.Factorization.KernelGrad

open NN.Examples.Factorization

/-! ### Concrete `4 × 2` example -/

/-- Sample matrix `X` (4 points in ℝ²). -/
def X : Fin 4 → Fin 2 → Float :=
  fun i k => ([[0.0, 0.0], [1.0, 0.5], [0.5, 1.0], [1.5, 1.2]].getD i.val []).getD k.val 0.0

/-- Linear-term column mask (all columns active). -/
def wlin : Fin 2 → Float := fun _ => 1.0

/-- Labels `y`. -/
def yv : Fin 4 → Float := fun i => [1.0, -0.5, 0.8, 0.2].getD i.val 0.0

/-- Log-hyperparameters `logθ = [log a, log b, log L, log nugget]`. -/
def logθ : Fin 4 → Float := fun i => [-0.3, 0.1, -1.0, -2.0].getD i.val 0.0

/-- The `ρ_KF` center embedding — the leading `nc = 2` indices into the `n = 4` samples. -/
def e : Fin 2 → Fin 4 := Fin.castLE (by decide)

/-- Perturb `logθ` in component `c` by `δ` (for finite differences). -/
def perturb (c : Fin 4) (δ : Float) : Fin 4 → Float := fun j => logθ j + (if j = c then δ else 0.0)

/-- The kernel matrix at a given log-parameter vector. -/
def Kof (lt : Fin 4 → Float) : Fin 4 → Fin 4 → Float := Spec.kernelMatrixMatern32Fn X wlin lt

/-- `Ω = K(logθ)`. -/
def Ω : Fin 4 → Fin 4 → Float := Kof logθ

/-- The closed-form `∂Ω/∂logθ_c` (ported `Matern32_αgrad!`). -/
def Hc (c : Fin 4) : Fin 4 → Fin 4 → Float := Spec.kernelMatrixMatern32GradFn X wlin logθ c

/-- Step for central finite differences. -/
def ε : Float := 1e-6

/-- Central finite difference of a scalar loss `f` in log-component `c`. -/
def fdLoss (f : (Fin 4 → Float) → Float) (c : Fin 4) : Float :=
  (f (perturb c ε) - f (perturb c (-ε))) / (2.0 * ε)

/-- Max over the four components of `|closed-form gradient − finite difference|`. -/
def gradErr (gf : Fin 4 → Float) (f : (Fin 4 → Float) → Float) : Float :=
  (List.finRange 4).foldl (fun acc c => max acc (Float.abs (gf c - fdLoss f c))) 0.0

/-- Max entrywise distance between two `4×4` matrices given as functions. -/
def maxMatErrFn (A B : Fin 4 → Fin 4 → Float) : Float :=
  (List.finRange 4).foldl (fun acc i =>
    (List.finRange 4).foldl (fun a j => max a (Float.abs (A i j - B i j))) acc) 0.0

/-- Central finite difference of the kernel matrix in component `c` (a matrix). -/
def fdK (c : Fin 4) : Fin 4 → Fin 4 → Float :=
  fun i j => (Kof (perturb c ε) i j - Kof (perturb c (-ε)) i j) / (2.0 * ε)

/-! ### Losses and their closed-form gradients -/

def rhoKF (lt : Fin 4 → Float) : Float := Spec.rhoKFFn (Kof lt) yv e
def rhoLOI (lt : Fin 4 → Float) : Float := Spec.rhoLOIFn (Kof lt) yv
def rhoLOO (lt : Fin 4 → Float) : Float := Spec.rhoLOOFn (Kof lt) yv

def kfGrad (c : Fin 4) : Float := Spec.rhoKFGradFn Ω (Hc c) yv e
def loiGrad (c : Fin 4) : Float := Spec.rhoLOIGradFn Ω (Hc c) yv
def looGrad (c : Fin 4) : Float := Spec.rhoLOOGradFn Ω (Hc c) yv

#eval IO.println s!"ρ_KF = {rhoKF logθ}  ρ_LOI = {rhoLOI logθ}  ρ_LOO = {rhoLOO logθ}"
#eval IO.println s!"∂ρ_KF/∂logθ = {(List.finRange 4).map kfGrad}"

/-! ### Positive — closed-form gradients vs central finite differences (the spec's own derivative) -/

-- `∂Ω/∂logθ_c` (ported `Matern32_αgrad!`) is the derivative of the Lean kernel build, all four components.
#eval assertLt "∂Ω/∂logθ₀ (Matérn weight) vs finite difference"
  (maxMatErrFn (Hc 0) (fdK 0)) 1e-4
#eval assertLt "∂Ω/∂logθ₁ (length scale, a·h²·e⁻ʰ) vs finite difference"
  (maxMatErrFn (Hc 1) (fdK 1)) 1e-4
#eval assertLt "∂Ω/∂logθ₂ (linear weight) vs finite difference"
  (maxMatErrFn (Hc 2) (fdK 2)) 1e-4
#eval assertLt "∂Ω/∂logθ₃ (nugget, diagonal) vs finite difference"
  (maxMatErrFn (Hc 3) (fdK 3)) 1e-4

-- The inverse-form loss gradients match the finite difference of the loss, every component.
#eval assertLt "∂ρ_KF/∂logθ  (closed form) vs finite difference" (gradErr kfGrad rhoKF) 1e-4
#eval assertLt "∂ρ_LOI/∂logθ (closed form) vs finite difference" (gradErr loiGrad rhoLOI) 1e-4
#eval assertLt "∂ρ_LOO/∂logθ (closed form) vs finite difference" (gradErr looGrad rhoLOO) 1e-4

/-! ### Positive — values and gradients vs the independent Julia golden -/

#eval assertApproxEq "ρ_KF  value vs Julia golden"  (rhoKF logθ)  0.45282874339470025 1e-9
#eval assertApproxEq "ρ_LOI value vs Julia golden"  (rhoLOI logθ) 0.8349234586270599  1e-9
#eval assertApproxEq "ρ_LOO value vs Julia golden"  (rhoLOO logθ) 1.3126306859751642  1e-9

-- ∂ρ_KF/∂logθ_c (the "#eval vs Zygote golden" deliverable):
#eval assertApproxEq "∂ρ_KF/∂logθ₀ vs golden"  (kfGrad 0)  0.026935632414271885 1e-9
#eval assertApproxEq "∂ρ_KF/∂logθ₁ vs golden"  (kfGrad 1)  0.06936946945092468  1e-9
#eval assertApproxEq "∂ρ_KF/∂logθ₂ vs golden"  (kfGrad 2)  0.02382513170789217  1e-9
#eval assertApproxEq "∂ρ_KF/∂logθ₃ vs golden"  (kfGrad 3) (-0.050758459691658986) 1e-9
-- ∂ρ_LOI/∂logθ_c:
#eval assertApproxEq "∂ρ_LOI/∂logθ₀ vs golden" (loiGrad 0)  0.03910029083224005  1e-9
#eval assertApproxEq "∂ρ_LOI/∂logθ₁ vs golden" (loiGrad 1)  0.09906204682849203  1e-9
#eval assertApproxEq "∂ρ_LOI/∂logθ₂ vs golden" (loiGrad 2) (-0.01664912810366878) 1e-9
#eval assertApproxEq "∂ρ_LOI/∂logθ₃ vs golden" (loiGrad 3) (-0.02245014349363348) 1e-9
-- ∂ρ_LOO/∂logθ_c:
#eval assertApproxEq "∂ρ_LOO/∂logθ₀ vs golden" (looGrad 0)  0.08458533699299974  1e-8
#eval assertApproxEq "∂ρ_LOO/∂logθ₁ vs golden" (looGrad 1)  0.20602390367886425  1e-8
#eval assertApproxEq "∂ρ_LOO/∂logθ₂ vs golden" (looGrad 2) (-0.005413358313286423) 1e-8
#eval assertApproxEq "∂ρ_LOO/∂logθ₃ vs golden" (looGrad 3) (-0.0791683844406198) 1e-8

/-! ### Negative controls — the finite-difference test is not vacuous -/

-- The closed form matched against the WRONG component's finite difference fails by a wide margin
-- (`∂ρ_KF/∂logθ₀ = 0.027` vs the `logθ₁` finite difference `≈ 0.069`).
#eval assertGe "∂ρ_KF/∂logθ₀ ≠ finite difference in wrong direction (logθ₁)"
  (Float.abs (kfGrad 0 - fdLoss rhoKF 1)) 0.01

-- The sign-flipped `∂Ω/∂logθ₀` is NOT the kernel derivative (it is ≈ −fd, so off by ≈ 2|∂Ω|).
#eval assertGe "−∂Ω/∂logθ₀ is not the kernel derivative (sign matters)"
  (maxMatErrFn (fun i j => -(Hc 0) i j) (fdK 0)) 0.1

-- A zero perturbation direction yields a zero inverse-form gradient (`quadInvGradFn_zero`), so a real
-- gradient component is genuinely nonzero — the closed form is not trivially zero.
#eval assertGe "∂ρ_LOO/∂logθ₁ is genuinely nonzero (gradient not vacuous)"
  (Float.abs (looGrad 1)) 0.05

end NN.Examples.Factorization.KernelGrad
