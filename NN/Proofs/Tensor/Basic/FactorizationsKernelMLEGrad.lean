/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.KernelGrad
public import NN.Proofs.Tensor.Basic.FactorizationsKernelGrad
public import Mathlib.LinearAlgebra.Matrix.Charpoly.Coeff

/-!
# Differentiating `ρ_MLE`: Jacobi's formula is exact polynomial arithmetic (S7)

`ρ_MLE = ½ yᵀΩ⁻¹y + ½ log det Ω` is the Gaussian-process negative log marginal likelihood (S3,
`loss_functions.jl`). Its data term `½ yᵀΩ⁻¹y` is differentiated by S6's inverse-form primitive
`quadInvGradFn` verbatim. The complexity term `½ log det Ω` needs the *one genuinely new differentiation
lemma* in the whole KernelFlows gradient — **Jacobi's formula** `∂(log det Ω) = tr(Ω⁻¹ ∂Ω)`.

What we prove here, exactly and without analysis:

* **`derivative_det_eq_det_mul_trace`** — Jacobi's formula for the determinant, as an *exact polynomial
  identity*: for invertible `Ω`, the coefficient of `t` in `det(Ω + tH)` is `det Ω · tr(Ω⁻¹H)`, i.e.
  `∂_t det(Ω + tH)|₀ = det Ω · tr(Ω⁻¹H)`. This is the finite, algebraic core — it rides on Mathlib's
  `derivative_det_one_add_X_smul` (the coefficient-of-`X` form of `det(1 + X•M) = 1 + tr(M)·X + O(X²)`),
  composed with the factorization `det(Ω + tH) = det Ω · det(1 + t·Ω⁻¹H)`. No `HasDerivAt`, no normed
  algebra — the derivative is `Polynomial.derivative`, exact over `ℝ[X]`.

* **`logDetGradFn_eq_trace`** — the spec `logDetGradFn` (assembled from the landed Cholesky solve, no
  inverse materialized) equals the Mathlib trace `tr(Ω⁻¹ H)` on an SPD `Ω`.

Together: `∂(log det Ω) = (∂ det Ω)/det Ω = tr(Ω⁻¹H) = logDetGradFn Ω H`, the dividing step `∂ log = ∂/·`
being the cited chain rule. The full `rhoMLEGradFn = ½ quadInvGradFn + ½ logDetGradFn` is checked against
finite-difference / Zygote golden by `#eval` ([`NN.Examples.Factorization.KernelMLEGrad`]). Following the
S2–S6 posture, no iteration-count convergence claim is made; this certifies the gradient arithmetic only,
`sorry`/`admit`/`omega`-free.
-/

@[expose] public section

namespace Spec.Factorization

open Matrix Polynomial
open scoped BigOperators
open Spec.Factorization.Reconstruction

variable {n : Nat}

/-! ## Jacobi's formula for the determinant (exact polynomial identity) -/

/-- **Jacobi's formula.** For an invertible `Ω`, the coefficient of `t` in the polynomial `det(Ω + tH)`
is `det Ω · tr(Ω⁻¹ H)` — equivalently `∂_t det(Ω + tH)|₀ = det Ω · tr(Ω⁻¹ H)`. The proof factors
`det(Ω + tH) = det Ω · det(1 + t·Ω⁻¹H)` over `ℝ[X]` and reads off the linear coefficient with Mathlib's
`derivative_det_one_add_X_smul`. This is the exact algebraic heart of `∂ log det = tr(Ω⁻¹ ∂Ω)`: no
analysis, the "derivative" is the formal `Polynomial.derivative`. -/
theorem derivative_det_eq_det_mul_trace (Ω H : Matrix (Fin n) (Fin n) ℝ) (hΩ : IsUnit Ω.det) :
    (Polynomial.derivative
        (Matrix.det (Ω.map (C : ℝ →+* ℝ[X]) + (X : ℝ[X]) • H.map (C : ℝ →+* ℝ[X])))).eval 0
      = Ω.det * Matrix.trace (Ω⁻¹ * H) := by
  have hfac :
      Ω.map (C : ℝ →+* ℝ[X]) + (X : ℝ[X]) • H.map (C : ℝ →+* ℝ[X])
        = Ω.map (C : ℝ →+* ℝ[X]) * (1 + (X : ℝ[X]) • (Ω⁻¹ * H).map (C : ℝ →+* ℝ[X])) := by
    rw [Matrix.mul_add, Matrix.mul_one, Matrix.mul_smul]
    congr 2
    rw [← Matrix.map_mul, ← Matrix.mul_assoc, Matrix.mul_nonsing_inv Ω hΩ, Matrix.one_mul]
  have hdetmap : (Ω.map (C : ℝ →+* ℝ[X])).det = C Ω.det := by
    rw [← RingHom.mapMatrix_apply, ← RingHom.map_det]
  rw [hfac, Matrix.det_mul, hdetmap, Polynomial.derivative_C_mul, Polynomial.eval_mul,
    Polynomial.eval_C, derivative_det_one_add_X_smul]

/-! ## The spec `logDetGradFn` is the trace `tr(Ω⁻¹ H)` -/

/-- The spec `logDetGradFn` (a sum of `n` Cholesky solves, no inverse materialized) equals the Mathlib
trace `tr(Ω⁻¹ H)` on an SPD `Ω`: the `c`-th solve recovers the `c`-th diagonal entry of `Ω⁻¹H`. Combined
with `derivative_det_eq_det_mul_trace`, `logDetGradFn Ω H = tr(Ω⁻¹H) = (∂ det Ω)/det Ω = ∂ log det Ω`. -/
theorem logDetGradFn_eq_trace (Ω H : Fin n → Fin n → ℝ) (hpd : (Matrix.of Ω).PosDef) :
    Spec.logDetGradFn Ω H = Matrix.trace ((Matrix.of Ω)⁻¹ * Matrix.of H) := by
  show (List.finRange n).foldl
      (fun s c => s + Spec.cholSolveFn (Spec.choleskyFn Ω) (fun k => H k c) c) 0 = _
  rw [finRange_foldl_add_eq_finset_sum
    (f := fun c => Spec.cholSolveFn (Spec.choleskyFn Ω) (fun k => H k c) c)]
  rw [Matrix.trace]
  refine Finset.sum_congr rfl (fun c _ => ?_)
  rw [cholSolveFn_eq_inv_mulVec Ω hpd (fun k => H k c)]
  simp [Matrix.diag_apply, Matrix.mulVec, Matrix.mul_apply, dotProduct, Matrix.of_apply]

/-! ## `∂ρ_MLE` decomposes into the two certified pieces -/

/-- **`∂ρ_MLE` = ½ (inverse-form gradient) + ½ (Jacobi trace).** On an SPD `Ω`, the spec `rhoMLEGradFn`
equals `½ ∂(yᵀΩ⁻¹y) + ½ ∂(log det Ω)` with both halves in closed form: the data term is S6's
`−(Ω⁻¹y)ᵀ H (Ω⁻¹y)` (`quadInvGradFn_eq_neg_quadForm`) and the complexity term is Jacobi's `tr(Ω⁻¹H)`
(`logDetGradFn_eq_trace`, certified `= (∂ det Ω)/det Ω` by `derivative_det_eq_det_mul_trace`). -/
theorem rhoMLEGradFn_eq (Ω H : Fin n → Fin n → ℝ) (hpd : (Matrix.of Ω).PosDef) (y : Fin n → ℝ) :
    Spec.rhoMLEGradFn Ω H y
      = Numbers.pointfive * (- (((Matrix.of Ω)⁻¹ *ᵥ y) ⬝ᵥ (Matrix.of H *ᵥ ((Matrix.of Ω)⁻¹ *ᵥ y))))
        + Numbers.pointfive * Matrix.trace ((Matrix.of Ω)⁻¹ * Matrix.of H) := by
  rw [show Spec.rhoMLEGradFn Ω H y
        = Numbers.pointfive * Spec.quadInvGradFn Ω H y + Numbers.pointfive * Spec.logDetGradFn Ω H
      from rfl,
    quadInvGradFn_eq_neg_quadForm Ω H hpd y, logDetGradFn_eq_trace Ω H hpd]
