/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Optimizers
public import NN.Proofs.Tensor.Basic.FactorizationsOrthonormal

/-!
# The KernelFlows optimizer steps are arithmetically correct (S5)

This file certifies the *update arithmetic* of the two KernelFlows optimizers
([`optimizers.jl`](../../../../../KernelFlows.jl/src/optimizers.jl)), ported in
[`NN.Spec.Core.Tensor.Optimizers`](../../../Spec/Core/Tensor/Optimizers.lean). Everything is over `ℝ`.

## AMSGrad

The defining property of AMSGrad over Adam is the **running max** `v̂ ← max(v̂, v)`: the second-moment
scale never shrinks, which is exactly what restores convergence where plain Adam can diverge
(Reddi–Kale–Kumar 2018). We prove that invariant (`amsGradStep_vhat_mono`, `amsGradStep_vhat_ge_v`),
reproduce the bit-faithful parameter step (`amsGradStep_x_apply`, using the *updated* moment and `v̂`),
and show a zero gradient with zeroed momentum is a fixed point (`amsGradStep_eq_of_grad_moment_zero`).

## SGD

`sgdStep_nonfixed_apply` is the ordinary `x ← x − ϵ g`. The interesting mode is `fixed = true`: the
gradient is normalized by `√(‖g‖² + δ)`, giving a **trust-region** step whose length is bounded by `ϵ`
(`sgdStep_fixed_displacement_normSq_le`: `‖Δx‖² ≤ ϵ²`) — the `#eval` golden shows the step length sitting
at `≈ ϵ` for non-tiny gradients. A zero gradient is a fixed point (`sgdStep_eq_of_grad_zero`).

## Convergence is a certificate, never a theorem

KernelFlows' flow loss `ρ` is non-convex, and Mathlib v4.30.0 has no first-order non-convex convergence
theory; following the same honest posture S2–S4 took toward the cyclic-Jacobi rate, we make **no** global
convergence claim. `IsApproxStationary` is the a-posteriori certificate (`‖∇ρ(x)‖ ≤ ε`), and we record
that a zero gradient certifies `0`-stationarity (`isApproxStationary_of_grad_zero`) — the same condition
under which both optimizers stop moving. That is the precise, provable statement; iteration-count
convergence rates are out of scope (and would be false in general for non-convex `ρ`).

Sorry/admit/omega-free; the arithmetic facts are exact over `ℝ`.
-/

@[expose] public section

namespace Spec.Factorization

open scoped BigOperators
open Spec.Factorization.Reconstruction

variable {n : Nat}

/-- The `Context ℝ` square root is `Real.sqrt` (a definitional bridge for the `Real.sqrt` lemmas). -/
theorem mathSqrt_eq_realSqrt (x : ℝ) : (MathFunctions.sqrt x) = Real.sqrt x := rfl

/-! ## AMSGrad -/

/-- AMSGrad's scalar second moment is the EMA of `‖g‖²` (`v ← β₂v + (1−β₂)‖g‖²`). -/
theorem amsGradStep_v (S : AMSGradState ℝ n) (g : Fin n → ℝ) :
    (Spec.amsGradStep S g).v = S.beta2 * S.v + (1 - S.beta2) * Spec.dotFn g g := rfl

/-- AMSGrad's first moment is the per-coordinate EMA (`m ← β₁m + (1−β₁)g`). -/
theorem amsGradStep_m_apply (S : AMSGradState ℝ n) (g : Fin n → ℝ) (i : Fin n) :
    (Spec.amsGradStep S g).m i = S.beta1 * S.m i + (1 - S.beta1) * g i := rfl

/-- The bit-faithful AMSGrad parameter step: `x ← x − ϵ·m̂ / (√v̂ + δ)`, using the **updated** moment
`m̂` and running max `v̂` (exactly `O.x .-= O.ϵ .* O.m / (sqrt(O.vhat) + O.δ)`). -/
theorem amsGradStep_x_apply (S : AMSGradState ℝ n) (g : Fin n → ℝ) (i : Fin n) :
    (Spec.amsGradStep S g).x i
      = S.x i - S.lr * (Spec.amsGradStep S g).m i
          / (MathFunctions.sqrt (Spec.amsGradStep S g).vhat + S.reg) := rfl

/-- **The AMSGrad invariant.** The running max `v̂` is monotone non-decreasing across a step — the
property that distinguishes AMSGrad from Adam and restores convergence. -/
theorem amsGradStep_vhat_mono (S : AMSGradState ℝ n) (g : Fin n → ℝ) :
    S.vhat ≤ (Spec.amsGradStep S g).vhat := by
  show S.vhat ≤ max S.vhat (S.beta2 * S.v + (1 - S.beta2) * Spec.dotFn g g)
  exact le_max_left _ _

/-- The running max dominates the current second moment (`v ≤ v̂`). -/
theorem amsGradStep_vhat_ge_v (S : AMSGradState ℝ n) (g : Fin n → ℝ) :
    (Spec.amsGradStep S g).v ≤ (Spec.amsGradStep S g).vhat := by
  show (S.beta2 * S.v + (1 - S.beta2) * Spec.dotFn g g)
      ≤ max S.vhat (S.beta2 * S.v + (1 - S.beta2) * Spec.dotFn g g)
  exact le_max_right _ _

/-- A zero gradient with zeroed momentum is a fixed point of AMSGrad: the parameters do not move. -/
theorem amsGradStep_eq_of_grad_moment_zero (S : AMSGradState ℝ n) (g : Fin n → ℝ)
    (hg : ∀ i, g i = 0) (hm : ∀ i, S.m i = 0) (i : Fin n) :
    (Spec.amsGradStep S g).x i = S.x i := by
  rw [amsGradStep_x_apply]
  have hmi : (Spec.amsGradStep S g).m i = 0 := by
    rw [amsGradStep_m_apply, hm i, hg i]; ring
  rw [hmi]; ring

/-! ## SGD -/

/-- The SGD step subtracts the displacement `Δx = (ϵ / α) · g`. -/
theorem sgdStep_x_apply (S : SGDState ℝ n) (g : Fin n → ℝ) (i : Fin n) :
    (Spec.sgdStep S g).x i = S.x i - Spec.sgdDisplacement S g i := rfl

/-- In non-fixed mode the normalizer is `1`. -/
theorem sgdNorm_nonfixed (S : SGDState ℝ n) (g : Fin n → ℝ) (h : S.fixed = false) :
    Spec.sgdNorm S g = 1 := by
  simp [Spec.sgdNorm, h]

/-- In non-fixed mode SGD is ordinary gradient descent: `x ← x − ϵ g`. -/
theorem sgdStep_nonfixed_apply (S : SGDState ℝ n) (g : Fin n → ℝ) (h : S.fixed = false) (i : Fin n) :
    (Spec.sgdStep S g).x i = S.x i - S.lr * g i := by
  rw [sgdStep_x_apply]
  unfold Spec.sgdDisplacement
  rw [sgdNorm_nonfixed S g h, div_one]

/-- A zero gradient is a fixed point of SGD (a stationary point is not moved). -/
theorem sgdStep_eq_of_grad_zero (S : SGDState ℝ n) (g : Fin n → ℝ) (hg : ∀ i, g i = 0) (i : Fin n) :
    (Spec.sgdStep S g).x i = S.x i := by
  rw [sgdStep_x_apply]
  unfold Spec.sgdDisplacement
  rw [hg i, mul_zero, sub_zero]

/-! ## Reusable numeric facts about `dotFn` over `ℝ` -/

/-- `dotFn g g = Σ gᵢ²` is nonnegative. -/
theorem dotFn_self_nonneg (g : Fin n → ℝ) : 0 ≤ Spec.dotFn g g := by
  rw [dotFn_eq_sum]
  exact Finset.sum_nonneg (fun i _ => mul_self_nonneg (g i))

/-- Scaling pulls a square out of the self dot product: `⟨cg, cg⟩ = c²⟨g, g⟩`. -/
theorem dotFn_smul_self (c : ℝ) (g : Fin n → ℝ) :
    Spec.dotFn (fun i => c * g i) (fun i => c * g i) = c ^ 2 * Spec.dotFn g g := by
  rw [dotFn_eq_sum, dotFn_eq_sum, Finset.mul_sum]
  exact Finset.sum_congr rfl (fun i _ => by ring)

/-- **The fixed-step SGD trust region.** With `fixed = true` and a positive stabilizer, the squared
displacement length is bounded by `ϵ²`: `‖Δx‖² = ϵ²‖g‖²/(‖g‖² + δ) ≤ ϵ²`. So fixed-step SGD never moves
more than `ϵ`, approaching `ϵ` as `‖g‖ → ∞` and `0` as `‖g‖ → 0`. -/
theorem sgdStep_fixed_displacement_normSq_le (S : SGDState ℝ n) (g : Fin n → ℝ)
    (hfix : S.fixed = true) (hstab : 0 < S.stab) :
    Spec.dotFn (Spec.sgdDisplacement S g) (Spec.sgdDisplacement S g) ≤ S.lr ^ 2 := by
  have hGnn : 0 ≤ Spec.dotFn g g := dotFn_self_nonneg g
  have hnorm : Spec.sgdNorm S g = Real.sqrt (Spec.dotFn g g + S.stab) := by
    simp [Spec.sgdNorm, hfix, mathSqrt_eq_realSqrt]
  have hdisp : Spec.sgdDisplacement S g
      = fun i => (S.lr / Real.sqrt (Spec.dotFn g g + S.stab)) * g i := by
    funext i; unfold Spec.sgdDisplacement; rw [hnorm]
  rw [hdisp, dotFn_smul_self]
  have hαsq : Real.sqrt (Spec.dotFn g g + S.stab) ^ 2 = Spec.dotFn g g + S.stab :=
    Real.sq_sqrt (by linarith)
  rw [div_pow, hαsq, div_mul_eq_mul_div, div_le_iff₀ (by linarith)]
  exact mul_le_mul_of_nonneg_left (by linarith) (sq_nonneg _)

/-! ## Convergence as an a-posteriori certificate (never a theorem) -/

/-- `x` is an `ε`-approximate stationary point of the loss whose gradient field is `grad` when
`‖grad x‖ ≤ ε`. KernelFlows' flow loss is non-convex; this is the *only* convergence statement we make —
an a-posteriori certificate, not an iteration-count guarantee. -/
def IsApproxStationary (grad : (Fin n → ℝ) → (Fin n → ℝ)) (x : Fin n → ℝ) (ε : ℝ) : Prop :=
  Spec.normFn (grad x) ≤ ε

/-- A point with zero gradient is certified `0`-stationary — the same condition under which both
`sgdStep` and `amsGradStep` leave the parameters fixed. -/
theorem isApproxStationary_of_grad_zero (grad : (Fin n → ℝ) → (Fin n → ℝ)) (x : Fin n → ℝ)
    (h : grad x = fun _ => 0) : IsApproxStationary grad x 0 := by
  unfold IsApproxStationary Spec.normFn
  rw [h]
  have hz : Spec.dotFn (fun _ : Fin n => (0 : ℝ)) (fun _ => 0) = 0 := by
    rw [dotFn_eq_sum]; simp
  rw [hz, mathSqrt_eq_realSqrt, Real.sqrt_zero]

end Spec.Factorization
