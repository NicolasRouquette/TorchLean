/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.KernelGrad
public import NN.Proofs.Tensor.Basic.FactorizationsKernelLoss

/-!
# Differentiating the inverse-form losses is exact arithmetic (S6)

This file certifies the *gradient arithmetic* of the KernelFlows cross-validation losses
([`loss_functions.jl`](../../../../../KernelFlows.jl/src/loss_functions.jl)). Every loss `ρ` is a
rational function of the regularized inverse quadratic form `yᵀ Ω⁻¹ y` (`quadInvFn`, S3), so its gradient
factors — by the chain rule — into two *cited* pieces: the derivative of a matrix inverse
`∂(Ω⁻¹) = −Ω⁻¹ (∂Ω) Ω⁻¹` and the closed-form kernel derivative `∂Ω/∂logθ` (ported `Matern32_αgrad!`).
**No new differentiation lemma is introduced.** What we prove here is the *exact algebra* underneath the
inverse-derivative, over `ℝ`:

* **`quadInvFn_secant_eq`** — the **resolvent (secant) identity**: for SPD `Ω, Ω'`,
  `quadInvFn Ω' y − quadInvFn Ω y = −yᵀ Ω'⁻¹ (Ω' − Ω) Ω⁻¹ y`. This is the *exact* finite difference of the
  inverse-form loss (Mathlib's `Matrix.inv_sub_inv` — the finite form of the Fréchet derivative of
  inversion, `hasFDerivAt_ring_inverse`). No asymptotics: it holds for every perturbation `Ω' − Ω`.

* **`quadInvGradFn_eq_neg_quadForm`** — the closed-form gradient `quadInvGradFn` is exactly the standard
  inverse-form derivative `−(Ω⁻¹y)ᵀ H (Ω⁻¹y)`. Letting `Ω' → Ω` in the secant identity (`Ω'⁻¹ → Ω⁻¹`)
  turns its right-hand side into this gradient — the cited derivative is the `H → 0` limit of the exact
  secant. That limit is the only analytic step, and it is the cited Mathlib derivative, not a new lemma.

The full loss gradients (`rhoKFGradFn`, `rhoLOIGradFn`, `rhoLOOGradFn`) are quotient-rule assemblies on
`quadInvGradFn`; they are checked against finite-difference / Zygote golden by `#eval` in
[`NN.Examples.Factorization.KernelGrad`](../../../Examples/Factorization/KernelGrad.lean). Following the
S2–S5 posture, **no iteration-count convergence claim is made** — this is the gradient arithmetic, exact
over `ℝ`, `sorry`/`admit`/`omega`-free.
-/

@[expose] public section

namespace Spec.Factorization

open Matrix
open scoped BigOperators
open Spec.Factorization.Reconstruction

variable {n : Nat}

/-! ## The bilinear form as a matrix quadratic form -/

/-- `yᵀ M y = y ⬝ᵥ (M *ᵥ y)`: the spec `bilinFn` is the Mathlib matrix quadratic form. -/
theorem bilinFn_eq (M : Fin n → Fin n → ℝ) (y : Fin n → ℝ) :
    Spec.bilinFn M y = y ⬝ᵥ ((Matrix.of M) *ᵥ y) := by
  simp only [Spec.bilinFn, dotFn_eq_sum, dotProduct, Matrix.mulVec, Matrix.of_apply]

/-! ## The Cholesky solve is the inverse times the vector -/

/-- For an SPD `Ω`, the Cholesky solve recovers `Ω⁻¹ *ᵥ y` (extracted from the `quadInvFn` proof so the
gradient specs can reuse it). -/
theorem cholSolveFn_eq_inv_mulVec (Ω : Fin n → Fin n → ℝ) (hpd : (Matrix.of Ω).PosDef)
    (y : Fin n → ℝ) :
    Spec.cholSolveFn (Spec.choleskyFn Ω) y = (Matrix.of Ω)⁻¹ *ᵥ y := by
  have hsolve : Matrix.of Ω *ᵥ (Spec.cholSolveFn (Spec.choleskyFn Ω) y) = y :=
    cholSolveFn_mulVec_of_posDef Ω hpd y
  have hdet : IsUnit (Matrix.of Ω).det := (Matrix.isUnit_iff_isUnit_det _).mp hpd.isUnit
  calc Spec.cholSolveFn (Spec.choleskyFn Ω) y
      = ((Matrix.of Ω)⁻¹ * Matrix.of Ω) *ᵥ (Spec.cholSolveFn (Spec.choleskyFn Ω) y) := by
        rw [Matrix.nonsing_inv_mul _ hdet, Matrix.one_mulVec]
    _ = (Matrix.of Ω)⁻¹ *ᵥ (Matrix.of Ω *ᵥ (Spec.cholSolveFn (Spec.choleskyFn Ω) y)) := by
        rw [Matrix.mulVec_mulVec]
    _ = (Matrix.of Ω)⁻¹ *ᵥ y := by rw [hsolve]

/-! ## The inverse-form gradient is the standard `−(Ω⁻¹y)ᵀ H (Ω⁻¹y)` -/

/-- **The closed-form inverse-form gradient.** For SPD `Ω`, `quadInvGradFn Ω H y = −(Ω⁻¹y)ᵀ H (Ω⁻¹y)` —
exactly the derivative of `yᵀΩ⁻¹y` obtained by composing `∂(Ω⁻¹) = −Ω⁻¹HΩ⁻¹` with the bilinear form. No
inverse is materialized: `Ω⁻¹y` is the landed Cholesky solve. -/
theorem quadInvGradFn_eq_neg_quadForm (Ω H : Fin n → Fin n → ℝ) (hpd : (Matrix.of Ω).PosDef)
    (y : Fin n → ℝ) :
    Spec.quadInvGradFn Ω H y
      = - (((Matrix.of Ω)⁻¹ *ᵥ y) ⬝ᵥ (Matrix.of H *ᵥ ((Matrix.of Ω)⁻¹ *ᵥ y))) := by
  show - Spec.bilinFn H (Spec.cholSolveFn (Spec.choleskyFn Ω) y) = _
  rw [cholSolveFn_eq_inv_mulVec Ω hpd y, bilinFn_eq]

/-- A zero perturbation gives a zero gradient (a parameter `Ω` does not depend on does not move `ρ`). -/
theorem quadInvGradFn_zero (Ω : Fin n → Fin n → ℝ) (y : Fin n → ℝ) :
    Spec.quadInvGradFn Ω (fun _ _ => 0) y = 0 := by
  show - Spec.bilinFn (fun _ _ => 0) (Spec.cholSolveFn (Spec.choleskyFn Ω) y) = 0
  simp [Spec.bilinFn, dotFn_eq_sum]

/-! ## The resolvent (secant) identity — the exact `∂(inverse)` core -/

/-- **The resolvent secant identity.** For SPD `Ω` and `Ω'`, the *exact* finite difference of the
inverse-form loss is `quadInvFn Ω' y − quadInvFn Ω y = −yᵀ Ω'⁻¹ (Ω' − Ω) Ω⁻¹ y`. This is the finite form
of the matrix-inverse derivative (Mathlib `Matrix.inv_sub_inv`; the Fréchet `hasFDerivAt_ring_inverse`):
substituting the perturbation `H = Ω' − Ω`, the right-hand side is `−yᵀ Ω'⁻¹ H Ω⁻¹ y`, whose `Ω' → Ω`
limit is the closed-form gradient `quadInvGradFn`. Nothing here is asymptotic — it is an exact identity
for every pair of SPD matrices. -/
theorem quadInvFn_secant_eq (Ω Ω' : Fin n → Fin n → ℝ) (hpd : (Matrix.of Ω).PosDef)
    (hpd' : (Matrix.of Ω').PosDef) (y : Fin n → ℝ) :
    Spec.quadInvFn Ω' y - Spec.quadInvFn Ω y
      = - (y ⬝ᵥ
          (((Matrix.of Ω')⁻¹ * (Matrix.of Ω' - Matrix.of Ω) * (Matrix.of Ω)⁻¹) *ᵥ y)) := by
  rw [quadInvFn_eq_dotProduct_inv Ω' hpd' y, quadInvFn_eq_dotProduct_inv Ω hpd y,
    ← dotProduct_sub, ← Matrix.sub_mulVec]
  have hiff : IsUnit (Matrix.of Ω') ↔ IsUnit (Matrix.of Ω) := iff_of_true hpd'.isUnit hpd.isUnit
  rw [Matrix.inv_sub_inv hiff]
  rw [show (Matrix.of Ω - Matrix.of Ω') = -(Matrix.of Ω' - Matrix.of Ω) from (neg_sub _ _).symm]
  rw [Matrix.mul_neg, Matrix.neg_mul, Matrix.neg_mulVec, dotProduct_neg]

end Spec.Factorization
