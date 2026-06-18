/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.CondVar
public import NN.Proofs.Tensor.Basic.FactorizationsKernelLoss
public import NN.Proofs.Tensor.Basic.Factorizations

/-!
# The KernelFlows predictive-variance models are correct on an SPD kernel (S4)

S2 built the SPD kernel matrix `Ω`; S3 showed every KernelFlows loss rests on the inverse quadratic
form `mᵀ Ω⁻¹ m` (`quadInvFn`). This file uses both to certify the two uncertainty-quantification models
of [`conditional_variance.jl`](../../../../../KernelFlows.jl/src/conditional_variance.jl).

## Full-rank model — `Σ (L⁻¹ m)²` *is* the inverse quadratic form

`ScalarFullRankUncertaintyModel.predict_variance` returns `Σ (L⁻¹ m)²` with `L` the Cholesky factor of
`Ω`. `condVarFullFn_eq_quadInv` identifies this with S3's `quadInvFn Ω m`, hence (`…eq_inv_quadForm`)
with the genuine inverse quadratic form `mᵀ Ω⁻¹ m`; it is nonnegative (`…nonneg`) and strictly positive
for `m ≠ 0` (`…pos`). The GP **posterior** variance `k(x*,x*) − mᵀΩ⁻¹m` is nonnegative whenever the
bordered kernel `[[Ω, m], [mᵀ, k(x*,x*)]]` is positive-semidefinite (`condVarPostFn_nonneg`, a Schur
complement), so the model never explains away more variance than the prior holds.

## Low-rank model — the diagonal correction matches marginals exactly

`ScalarLowRankUncertaintyModel` keeps a set of eigenpairs (`keep`) of `Ω = V·diag(λ)·Vᵀ`, approximating
the precision `Ω⁻¹ = Σₖ (1/λₖ) vₖ vₖᵀ` by `PᵀP` (the kept terms) plus a diagonal `DᵀD` chosen so the
marginal precision is exact. The headline:

* `precision_resid_diag_eq_inv` — `(PᵀP)[a,a] + D[a,a]² = (Ω⁻¹)[a,a]` *exactly*, for any kept set. This
  is the model's design goal ("can still match the marginal uncertainties exactly"): the dropped
  eigenpairs are returned to the diagonal verbatim. A pure sum split, independent of the eigenvalue
  ordering — proved from `IsSymEig`, so it holds for whatever `eigh` returns.
* `residDiagLowRankFn_nonneg` — `D[a,a]² ≥ 0`, so `D = √(dfull − d)` is real (`λₖ > 0` on an SPD `Ω`).
* `condVarLowRankFn_nonneg` — the predicted low-rank variance `mᵀ(PᵀP + DᵀD)m` is nonnegative; via the
  closed form `bilinFn_precisionLowRank_eq` it is a nonnegative combination of squared projections.

Scope honesty: everything is exact over `ℝ`, proved from the specifications `IsSymEig` / `PosDef` (so it
holds for any eigendecomposition the solver returns), reusing S3's verified Cholesky solve — no
asymptotics, no `sorry`.
-/

@[expose] public section

namespace Spec.Factorization

open Matrix
open scoped BigOperators
open Spec.Factorization.Reconstruction

variable {n : Nat}

/-! ## Full-rank model: `Σ (L⁻¹ m)² = mᵀ Ω⁻¹ m` -/

/-- **The full-rank conditional variance is the inverse quadratic form.** `Σ (L⁻¹ m)²` (the Cholesky
forward-solve sum of squares of `ScalarFullRankUncertaintyModel`) equals S3's `quadInvFn Ω m`. Direct
corollary of `rhoMLE_data_eq_quadInv` (the `‖L⁻¹y‖² = yᵀΩ⁻¹y` identity), since `condVarFullFn` *is* that
sum of squares. -/
theorem condVarFullFn_eq_quadInv (Ω : Fin n → Fin n → ℝ) (hpd : (Matrix.of Ω).PosDef)
    (m : Fin n → ℝ) : Spec.condVarFullFn Ω m = Spec.quadInvFn Ω m := by
  show Spec.dotFn (Spec.triSolveLowerFn (Spec.choleskyFn Ω) m)
      (Spec.triSolveLowerFn (Spec.choleskyFn Ω) m) = Spec.quadInvFn Ω m
  exact rhoMLE_data_eq_quadInv Ω hpd m

/-- The full-rank conditional variance is the inverse quadratic form `m ⬝ᵥ (Ω⁻¹ *ᵥ m) = mᵀ Ω⁻¹ m`. -/
theorem condVarFullFn_eq_inv_quadForm (Ω : Fin n → Fin n → ℝ) (hpd : (Matrix.of Ω).PosDef)
    (m : Fin n → ℝ) : Spec.condVarFullFn Ω m = m ⬝ᵥ ((Matrix.of Ω)⁻¹ *ᵥ m) := by
  rw [condVarFullFn_eq_quadInv Ω hpd m, quadInvFn_eq_dotProduct_inv Ω hpd m]

/-- The full-rank conditional variance is nonnegative on an SPD `Ω`. -/
theorem condVarFullFn_nonneg (Ω : Fin n → Fin n → ℝ) (hpd : (Matrix.of Ω).PosDef) (m : Fin n → ℝ) :
    0 ≤ Spec.condVarFullFn Ω m := by
  rw [condVarFullFn_eq_quadInv Ω hpd m]; exact quadInvFn_nonneg Ω hpd m

/-- The full-rank conditional variance is strictly positive for a nonzero cross-covariance `m ≠ 0`. -/
theorem condVarFullFn_pos (Ω : Fin n → Fin n → ℝ) (hpd : (Matrix.of Ω).PosDef) {m : Fin n → ℝ}
    (hm : m ≠ 0) : 0 < Spec.condVarFullFn Ω m := by
  rw [condVarFullFn_eq_quadInv Ω hpd m]; exact quadInvFn_pos Ω hpd hm

/-- **The GP posterior variance is nonnegative.** If the bordered kernel
`[[Ω, m], [mᵀ, k(x*,x*)]]` is positive-semidefinite — automatic for a genuine kernel Gram matrix over the
training points together with the test point — then `condVarPostFn Ω m kss = k(x*,x*) − mᵀΩ⁻¹m ≥ 0`: the
full-rank model never explains away more variance than the prior holds. A Schur-complement consequence of
`Matrix.PosDef.fromBlocks₁₁`. -/
theorem condVarPostFn_nonneg (Ω : Fin n → Fin n → ℝ) (hpd : (Matrix.of Ω).PosDef)
    (m : Fin n → ℝ) (kss : ℝ)
    (hB : (Matrix.fromBlocks (Matrix.of Ω) (Matrix.replicateCol Unit m)
            (Matrix.replicateCol Unit m)ᴴ (Matrix.of (fun _ _ => kss))).PosSemidef) :
    0 ≤ Spec.condVarPostFn Ω m kss := by
  letI : Invertible (Matrix.of Ω) :=
    Matrix.invertibleOfIsUnitDet _ ((Matrix.isUnit_iff_isUnit_det _).mp hpd.isUnit)
  have hschur := (hpd.fromBlocks₁₁ (Matrix.replicateCol Unit m)
      (Matrix.of (fun _ _ => kss))).mp hB
  have hquad : ((Matrix.replicateCol Unit m)ᴴ * (Matrix.of Ω)⁻¹ * Matrix.replicateCol Unit m) ()
        () = m ⬝ᵥ ((Matrix.of Ω)⁻¹ *ᵥ m) := by
    rw [Matrix.mul_assoc, ← Matrix.replicateCol_mulVec, Matrix.conjTranspose_replicateCol,
      Matrix.replicateRow_mul_replicateCol_apply, star_trivial]
  have hentry := hschur.diag_nonneg (i := (() : Unit))
  rw [Matrix.sub_apply, hquad, Matrix.of_apply] at hentry
  rw [Spec.condVarPostFn, condVarFullFn_eq_inv_quadForm Ω hpd m]
  exact hentry

/-! ## Low-rank model: spectral precision and the marginal-matching diagonal -/

/-- The executable low-rank precision entry as a `Finset` sum over the eigenpairs. -/
theorem precisionLowRankFn_eq_sum (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (keep : Fin n → Bool)
    (a b : Fin n) :
    Spec.precisionLowRankFn Λ V keep a b
      = ∑ k, (if keep k then 1 / Λ k else 0) * V a k * V b k := by
  unfold Spec.precisionLowRankFn
  rw [foldl_addf_eq_sum (fun k => (if keep k then 1 / Λ k else 0) * V a k * V b k)
      (List.finRange n) 0, zero_add, ← finsum_eq_finRange_sum]

/-- The executable residual-diagonal entry as a `Finset` sum over the *dropped* eigenpairs. -/
theorem residDiagLowRankFn_eq_sum (Λ : Fin n → ℝ) (V : Fin n → Fin n → ℝ) (keep : Fin n → Bool)
    (a : Fin n) :
    Spec.residDiagLowRankFn Λ V keep a
      = ∑ k, (if keep k then 0 else 1 / Λ k) * V a k * V a k := by
  unfold Spec.residDiagLowRankFn
  rw [foldl_addf_eq_sum (fun k => (if keep k then 0 else 1 / Λ k) * V a k * V a k)
      (List.finRange n) 0, zero_add, ← finsum_eq_finRange_sum]

/-- `A⁻¹ = V · diag(1/λ) · Vᵀ` for a symmetric eigendecomposition with nonzero eigenvalues
(the `γ = 0` specialization of `IsSymEig.add_smul_inv`). -/
theorem IsSymEig.inv {A V : Matrix (Fin n) (Fin n) ℝ} {Λ : Fin n → ℝ}
    (h : IsSymEig A Λ V) (hΛ : ∀ i, Λ i ≠ 0) :
    A⁻¹ = V * Matrix.diagonal (fun i => (Λ i)⁻¹) * Vᵀ := by
  have hγ : ∀ i, Λ i + 0 ≠ 0 := by simpa using hΛ
  have hh := h.add_smul_inv 0 hγ
  simpa only [zero_smul, add_zero] using hh

/-- Entry `(a, b)` of an orthogonal conjugation of a diagonal: `(V · diag(d) · Vᵀ)[a,b] =
Σₖ V[a,k] · d k · V[b,k]`. -/
theorem conj_diagonal_apply (V : Matrix (Fin n) (Fin n) ℝ) (d : Fin n → ℝ) (a b : Fin n) :
    (V * Matrix.diagonal d * Vᵀ) a b = ∑ k, V a k * d k * V b k := by
  rw [Matrix.mul_apply]
  refine Finset.sum_congr rfl (fun k _ => ?_)
  rw [Matrix.mul_diagonal, Matrix.transpose_apply]

/-- Entry `(a, b)` of the inverse from a symmetric eigendecomposition:
`(A⁻¹)[a,b] = Σₖ V[a,k] · (1/λₖ) · V[b,k]` — the spectral form of the precision. -/
theorem IsSymEig.inv_apply {A V : Matrix (Fin n) (Fin n) ℝ} {Λ : Fin n → ℝ}
    (h : IsSymEig A Λ V) (hΛ : ∀ i, Λ i ≠ 0) (a b : Fin n) :
    A⁻¹ a b = ∑ k, V a k * (Λ k)⁻¹ * V b k := by
  rw [h.inv hΛ, conj_diagonal_apply]

/-- **The low-rank diagonal correction matches the full marginal precision exactly.** For any kept set
`keep`, `(PᵀP)[a,a] + D[a,a]² = (Ω⁻¹)[a,a]`: splitting `Σₖ (1/λₖ) V[a,k]²` into kept + dropped recovers
the full marginal precision. This is the design goal of `ScalarLowRankUncertaintyModel` ("match the
marginal uncertainties exactly"), proved as a pure sum split. -/
theorem precision_resid_diag_eq_inv {Ω : Fin n → Fin n → ℝ} {Λ : Fin n → ℝ}
    {V : Matrix (Fin n) (Fin n) ℝ} (keep : Fin n → Bool)
    (h : IsSymEig (Matrix.of Ω) Λ V) (hΛ : ∀ i, 0 < Λ i) (a : Fin n) :
    Spec.precisionLowRankFn Λ V keep a a + Spec.residDiagLowRankFn Λ V keep a
      = (Matrix.of Ω)⁻¹ a a := by
  rw [precisionLowRankFn_eq_sum, residDiagLowRankFn_eq_sum, ← Finset.sum_add_distrib,
    h.inv_apply (fun i => (hΛ i).ne') a a]
  refine Finset.sum_congr rfl (fun k _ => ?_)
  by_cases hk : keep k = true
  · rw [if_pos hk, if_pos hk, one_div]; ring
  · rw [if_neg hk, if_neg hk, one_div]; ring

/-- **The residual variances are nonnegative**, so `D = √(dfull − d)` is real: `D[a,a]² ≥ 0` because each
dropped term `(1/λₖ) V[a,k]²` is nonnegative (`λₖ > 0` on an SPD `Ω`). -/
theorem residDiagLowRankFn_nonneg {Λ : Fin n → ℝ} {V : Matrix (Fin n) (Fin n) ℝ}
    (keep : Fin n → Bool) (hΛ : ∀ i, 0 < Λ i) (a : Fin n) :
    0 ≤ Spec.residDiagLowRankFn Λ V keep a := by
  rw [residDiagLowRankFn_eq_sum]
  refine Finset.sum_nonneg (fun k _ => ?_)
  split_ifs with hk
  · simp
  · rw [show (1 / Λ k) * V a k * V a k = (1 / Λ k) * (V a k * V a k) from by ring]
    exact mul_nonneg (div_nonneg zero_le_one (hΛ k).le) (mul_self_nonneg _)

/-- The low-rank quadratic form `mᵀ PᵀP m` in closed form: a nonnegative combination of squared
projections `Σ_{k ∈ keep} (1/λₖ) ⟨vₖ, m⟩²`. -/
theorem bilinFn_precisionLowRank_eq {Λ : Fin n → ℝ} {V : Matrix (Fin n) (Fin n) ℝ}
    (keep : Fin n → Bool) (m : Fin n → ℝ) :
    Spec.bilinFn (Spec.precisionLowRankFn Λ V keep) m
      = ∑ k, (if keep k then 1 / Λ k else 0) * (∑ a, m a * V a k) ^ 2 := by
  rw [Spec.bilinFn, dotFn_eq_sum]
  -- expand to a triple sum ∑ a ∑ b ∑ k
  have hexp : ∀ a, m a * Spec.dotFn (Spec.precisionLowRankFn Λ V keep a) m
      = ∑ b, ∑ k, (if keep k then 1 / Λ k else 0) * (m a * V a k) * (m b * V b k) := by
    intro a
    rw [dotFn_eq_sum, Finset.mul_sum]
    refine Finset.sum_congr rfl (fun b _ => ?_)
    rw [precisionLowRankFn_eq_sum, Finset.sum_mul, Finset.mul_sum]
    exact Finset.sum_congr rfl (fun k _ => by ring)
  rw [Finset.sum_congr rfl (fun a _ => hexp a)]
  -- reorder ∑ a ∑ b ∑ k → ∑ k ∑ a ∑ b
  rw [show (∑ a, ∑ b, ∑ k, (if keep k then 1 / Λ k else 0) * (m a * V a k) * (m b * V b k))
        = ∑ a, ∑ k, ∑ b, (if keep k then 1 / Λ k else 0) * (m a * V a k) * (m b * V b k) from
      Finset.sum_congr rfl (fun a _ => Finset.sum_comm)]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl (fun k _ => ?_)
  -- per kept eigenpair: factor the scalar and collapse the outer product to a square
  have hscal : (∑ a, ∑ b, (if keep k then 1 / Λ k else 0) * (m a * V a k) * (m b * V b k))
      = (if keep k then 1 / Λ k else 0) * ∑ a, ∑ b, (m a * V a k) * (m b * V b k) := by
    rw [Finset.mul_sum]
    refine Finset.sum_congr rfl (fun a _ => ?_)
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl (fun b _ => by ring)
  rw [hscal, ← Finset.sum_mul_sum, sq]

/-- **The low-rank predicted variance is nonnegative.** `mᵀ(PᵀP + DᵀD)m ≥ 0`: the kept term is a
nonnegative combination of squared projections (`bilinFn_precisionLowRank_eq`) and the residual term is
a nonnegative-weighted sum of squares (`residDiagLowRankFn_nonneg`). -/
theorem condVarLowRankFn_nonneg {Λ : Fin n → ℝ} {V : Matrix (Fin n) (Fin n) ℝ}
    (keep : Fin n → Bool) (hΛ : ∀ i, 0 < Λ i) (m : Fin n → ℝ) :
    0 ≤ Spec.condVarLowRankFn Λ V keep m := by
  rw [Spec.condVarLowRankFn]
  apply add_nonneg
  · rw [bilinFn_precisionLowRank_eq]
    refine Finset.sum_nonneg (fun k _ => ?_)
    split_ifs with hk
    · exact mul_nonneg (div_nonneg zero_le_one (hΛ k).le) (sq_nonneg _)
    · simp
  · rw [foldl_addf_eq_sum (fun a => Spec.residDiagLowRankFn Λ V keep a * m a * m a)
        (List.finRange n) 0, zero_add, ← finsum_eq_finRange_sum]
    refine Finset.sum_nonneg (fun a _ => ?_)
    rw [show Spec.residDiagLowRankFn Λ V keep a * m a * m a
        = Spec.residDiagLowRankFn Λ V keep a * (m a * m a) from by ring]
    exact mul_nonneg (residDiagLowRankFn_nonneg keep hΛ a) (mul_self_nonneg _)

/-! ## Tensor-level corollaries -/

/-- Tensor-level: the full-rank conditional variance is strictly positive for a nonzero cross-covariance
on an SPD kernel. -/
theorem condVarFullSpec_pos (Ω : Spec.Tensor ℝ (.dim n (.dim n .scalar)))
    (m : Spec.Tensor ℝ (.dim n .scalar)) (hpd : (Matrix.of (Spec.toMatFn Ω)).PosDef)
    (hm : Spec.toVecFn m ≠ 0) : 0 < Spec.condVarFullSpec Ω m :=
  condVarFullFn_pos _ hpd hm

end Spec.Factorization
