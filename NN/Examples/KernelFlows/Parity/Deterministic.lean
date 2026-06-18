/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.KernelFlows.Parity.Common
meta import NN.Examples.KernelFlows.Parity.Common

/-!
# KernelFlows parity harness (S8) — the deterministic float-path

The deterministic half of the parity harness: the shared fixture
([`Common`](Common.lean)) is driven through the whole KernelFlows pipeline and **every stage** is
checked against the golden tile to the float-path tolerance `1e-8`:

1. **kernel build** `Ω = K(logθ)` vs the verbatim KernelFlows.jl `kernel_matrix` tile (`~10⁻¹⁵`);
2. **losses** `ρ_KF`, `ρ_LOI`, `ρ_LOO`, `ρ_MLE` (S3) vs `loss_functions.jl`;
3. **kernel derivative tiles** `∂Ω/∂logθ_c` (S6) vs `Matern32_αgrad!`;
4. **loss gradients** `∂ρ_KF`, `∂ρ_LOI`, `∂ρ_LOO`, `∂ρ_MLE`, and the Jacobi `∂logdet` (S6/S7);
5. **optimizer step** — one AMSGrad and one fixed-step SGD update of `logθ` driven by the *actual*
   `∇ρ_MLE` (S5 ∘ S7) vs `optimizers.jl` (bit-faithful);
6. **conditional variance** `mᵀΩ⁻¹m` (S4) vs `conditional_variance.jl`.

The composition is the point: stage 5 consumes stage 4's gradient, which consumes stage 1's kernel —
the harness verifies the *pipeline*, not isolated pieces. Negative controls confirm the parity metric
has teeth. All checks are `#eval`'d over `Float`.
-/

@[expose] public section

namespace NN.Examples.KernelFlows.Parity.Deterministic

open NN.Examples.KernelFlows.Parity
open NN.Examples.Factorization (assertLt assertGe)

/-! ## Stage 1 — kernel build vs the verbatim KernelFlows.jl tile -/

/-- The golden kernel tile from KernelFlows.jl `kernel_matrix(::UnaryKernel{Matern32}, …)`. -/
def Kgold : Fin 4 → Fin 4 → Float := tileOf
  [[1.4176726537516589, 0.5626260453760491, 1.0850527096273428, 1.2983849277189337],
   [0.5626260453760491, 1.0497932125802165, 0.71717326845590046, 0.37933628856076407],
   [1.0850527096273428, 0.71717326845590046, 1.4176726537516589, 1.4529321507987851],
   [1.2983849277189337, 0.37933628856076407, 1.4529321507987851, 2.5213109772659861]]

#eval IO.println s!"Ω[0] = {(List.finRange 4).map (Ω 0)}"

-- The Lean exact-distance build reproduces the KernelFlows.jl floored-distance tile to `~10⁻¹⁵`
-- (the `5·eps` floor only moves `K` through the Matérn flat top — S1), well inside the parity band.
#eval assertParity "Ω = K(logθ) tile" (tileErr Ω Kgold) 0.0 1e-8

/-! ## Stage 2 — losses vs `loss_functions.jl` -/

def rhoKF  : Float := Spec.rhoKFFn Ω yv e
def rhoLOI : Float := Spec.rhoLOIFn Ω yv
def rhoLOO : Float := Spec.rhoLOOFn Ω yv
def rhoMLE : Float := Spec.rhoMLEFn Ω yv

#eval IO.println s!"ρ_KF={rhoKF} ρ_LOI={rhoLOI} ρ_LOO={rhoLOO} ρ_MLE={rhoMLE}"

#eval assertParity "ρ_KF"  rhoKF  0.56141965623425893
#eval assertParity "ρ_LOI" rhoLOI 0.91107495995177767
#eval assertParity "ρ_LOO" rhoLOO 1.7830377244070612
#eval assertParity "ρ_MLE" rhoMLE 1.512575717526133

/-! ## Stage 3 — kernel derivative tiles vs `Matern32_αgrad!` -/

def dKgold0 : Fin 4 → Fin 4 → Float := tileOf
  [[1.0, 0.5626260453760491, 0.71717326845590046, 0.5626260453760491],
   [0.5626260453760491, 1.0, 0.71717326845590046, 0.37933628856076407],
   [0.71717326845590046, 0.71717326845590046, 1.0, 0.71717326845590046],
   [0.5626260453760491, 0.37933628856076407, 0.71717326845590046, 1.0]]

def dKgold3 : Fin 4 → Fin 4 → Float := tileOf
  [[0.049787068367863944, 0.0, 0.0, 0.0],
   [0.0, 0.049787068367863944, 0.0, 0.0],
   [0.0, 0.0, 0.049787068367863944, 0.0],
   [0.0, 0.0, 0.0, 0.049787068367863944]]

#eval assertParity "∂Ω/∂logθ₀ (Matérn weight) tile" (tileErr (Hc 0) dKgold0) 0.0 1e-8
#eval assertParity "∂Ω/∂logθ₃ (nugget) tile"        (tileErr (Hc 3) dKgold3) 0.0 1e-8

/-! ## Stage 4 — loss gradients vs `Matern32_αgrad!` ∘ quotient rule (S6/S7) -/

def kfGrad  (c : Fin 4) : Float := Spec.rhoKFGradFn  Ω (Hc c) yv e
def loiGrad (c : Fin 4) : Float := Spec.rhoLOIGradFn Ω (Hc c) yv
def looGrad (c : Fin 4) : Float := Spec.rhoLOOGradFn Ω (Hc c) yv
def mleGrad (c : Fin 4) : Float := Spec.rhoMLEGradFn Ω (Hc c) yv
def jacGrad (c : Fin 4) : Float := Spec.logDetGradFn Ω (Hc c)

#eval IO.println s!"∇ρ_MLE = {(List.finRange 4).map mleGrad}"

#eval assertParity "∂ρ_KF/∂logθ"  (vecErr kfGrad
  [-0.069622875967251002, 0.28971589443873613, 0.11681747395771418, -0.047188774433055178]) 0.0 1e-8
#eval assertParity "∂ρ_LOI/∂logθ" (vecErr loiGrad
  [-0.0086535451752164129, 0.12399612861046813, 0.021609603790844669, -0.012954459908268939]) 0.0 1e-8
#eval assertParity "∂ρ_LOO/∂logθ" (vecErr looGrad
  [-0.0869018997528066, 0.77790447607740865, 0.18214584812613963, -0.095232195786708024]) 0.0 1e-8
#eval assertParity "∂ρ_MLE/∂logθ" (vecErr mleGrad
  [-0.054783030477864481, 1.3111280815493438, 0.2793256725921609, -0.13819665294170599]) 0.0 1e-8
#eval assertParity "∂logdet/∂logθ (Jacobi tr Ω⁻¹H)" (vecErr jacGrad
  [2.9626195050268898, -2.7145489357959001, 0.62174938602490415, 0.41557982232375207]) 0.0 1e-8

/-! ## Stage 5 — optimizer step driven by the actual `∇ρ_MLE` (S5 ∘ S7) -/

/-- The gradient `∇ρ_MLE` fed to the optimizer (the genuine end-to-end coupling). -/
def gMLE : Fin 4 → Float := mleGrad

/-- One AMSGrad step of `logθ` (ϵ=0.1, β₁=0.9, β₂=0.999, δ=1e-8), the verbatim `optimizers.jl` rule. -/
def amsNext : Fin 4 → Float := (Spec.amsGradStep (Spec.amsGradInit logθ 0.1 0.9 0.999 1e-8) gMLE).x

/-- One fixed-step SGD step of `logθ` (ϵ=0.1, stab=1e-9). -/
def sgdNext : Fin 4 → Float := (Spec.sgdStep (Spec.sgdInit logθ 0.1 true 1e-9) gMLE).x

#eval IO.println s!"AMSGrad logθ' = {(List.finRange 4).map amsNext}"

#eval assertParity "AMSGrad step on ∇ρ_MLE" (vecErr amsNext
  [0.012844234126101216, 0.19259758173600477, -1.0654897019188097, -2.9675989051656417]) 0.0 1e-8
#eval assertParity "SGD (fixed) step on ∇ρ_MLE" (vecErr sgdNext
  [0.0040617044150706109, 0.40279079723959765, -1.0207096669847164, -2.9897538717645764]) 0.0 1e-8

/-! ## Stage 6 — conditional variance vs `conditional_variance.jl` -/

def cvFull : Float := Spec.condVarFullFn Ω mtest

#eval assertParity "conditional variance mᵀΩ⁻¹m" cvFull 1.555396862329099

/-! ## Negative controls — the parity metric is not vacuous -/

-- The kernel tile at the WRONG hyperparameters drifts far from the golden (parity would catch a
-- mis-specified build): a perturbed `logθ` moves `Ω` by `≫ 1e-8`.
#eval assertGe "Ω at wrong logθ drifts from the golden tile"
  (tileErr (Spec.kernelMatrixMatern32Fn X wlin (fun i => logθ i + (if i.val == 1 then 0.3 else 0.0))) Kgold)
  0.05

-- A loss value compared against the WRONG golden (ρ_LOO's golden) is correctly rejected — the parity
-- check is specific to each stage, not a blanket pass.
#eval assertGe "ρ_KF ≠ ρ_LOO golden (parity is stage-specific)"
  (Float.abs (rhoKF - 1.7830377244070612)) 0.5

-- The sign-flipped MLE gradient is NOT the gradient the optimizer should follow: one AMSGrad step in
-- the wrong direction lands far from the golden next-iterate.
#eval assertGe "AMSGrad on −∇ρ_MLE diverges from the golden iterate"
  (vecErr (Spec.amsGradStep (Spec.amsGradInit logθ 0.1 0.9 0.999 1e-8) (fun c => -gMLE c)).x
    [0.012844234126101216, 0.19259758173600477, -1.0654897019188097, -2.9675989051656417]) 0.05

end NN.Examples.KernelFlows.Parity.Deterministic
