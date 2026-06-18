/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.KernelLoss

/-!
# Predictive (conditional) variance — the KernelFlows UQ models (S4)

KernelFlows' uncertainty quantification ([`conditional_variance.jl`](../../../../../KernelFlows.jl/src/conditional_variance.jl))
predicts the GP posterior variance reduction at a test point `x*` from its cross-covariance vector
`m = k(x*, X)` against the SPD kernel matrix `Ω = K(X)`. Two models compute the **same** quadratic form
`mᵀ Ω⁻¹ m` two different ways:

* **`ScalarFullRankUncertaintyModel`** — the *full-rank* model factors `Ω = L·Lᵀ` (Cholesky, S2) and
  returns `Σ (L⁻¹ m)²` (`predict_variance`: `A = LI*M_cross; sum(A.^2)`). Since `‖L⁻¹m‖² = mᵀΩ⁻¹m`, this
  is exactly S3's `quadInvFn` — one inverse quadratic form per test column.
* **`ScalarLowRankUncertaintyModel`** — the *low-rank* model eigendecomposes `Ω = V·diag(λ)·Vᵀ`
  (`eigh`, S2) and approximates the precision `Ω⁻¹ = Σₖ (1/λₖ) vₖ vₖᵀ` by keeping a leading set of
  eigenpairs (`P`, the `r` columns with largest `1/λ`) plus a diagonal residual `D` chosen so the
  **marginal** variances match the full model exactly (`d = sum(P.^2); D = Diagonal(sqrt(dfull - d))`).
  `predict_variance` returns `mᵀ(PᵀP + DᵀD)m`.

This file ports both as `Context`-polymorphic specs from the landed verified pieces (`choleskyFn`,
`triSolveLowerFn`, and the eigendecomposition `symEigJacobiSpec`). The kept-eigenpair set is a general
`Bool` predicate `keep` (leading-`r` is `keep k = (k.val < r)`); stating it this way makes the
marginal-match proof a pure sum split, independent of the eigenvalue ordering.

*Scope (S4).* This is the **evaluation spec** plus its correctness facts, proved over `ℝ` in
[`NN.Proofs.Tensor.Basic.FactorizationsCondVar`](../../../Proofs/Tensor/Basic/FactorizationsCondVar.lean):
the full-rank value *is* the inverse quadratic form `mᵀΩ⁻¹m` (nonnegative, and the posterior variance
`k(x*,x*) − mᵀΩ⁻¹m ≥ 0` from a positive-semidefinite bordered kernel), and the low-rank model matches the
full model's diagonal **exactly** with a nonnegative residual (so `D = √(dfull − d)` is real). Numeric
`#eval`s check both against the `conditional_variance.jl` golden values.
-/

@[expose] public section

namespace Spec

variable {α : Type} [Context α]
variable {n : Nat}

/-! ## Full-rank model: `Σ (L⁻¹ m)²` via the landed Cholesky factor -/

/-- **KernelFlows full-rank conditional variance** (`conditional_variance.jl`,
`ScalarFullRankUncertaintyModel`): for a test point with cross-covariance vector `m = k(x*, X)`, the
explained variance is `‖L⁻¹ m‖² = Σᵢ (L⁻¹ m)ᵢ²`, with `L` the Cholesky factor of the SPD `Ω`. Equal to
the inverse quadratic form `mᵀ Ω⁻¹ m` (S3's `quadInvFn`), computed without forming `Ω⁻¹`. -/
def condVarFullFn (Ω : Fin n → Fin n → α) (m : Fin n → α) : α :=
  let z := triSolveLowerFn (choleskyFn Ω) m
  dotFn z z

/-- The GP **posterior** variance at a test point: prior variance `k(x*, x*)` minus the explained
variance `mᵀ Ω⁻¹ m`. On a valid (positive-semidefinite) kernel this is nonnegative — the model never
"explains away" more than the prior. -/
def condVarPostFn (Ω : Fin n → Fin n → α) (m : Fin n → α) (kss : α) : α :=
  kss - condVarFullFn Ω m

/-- The full-rank conditional variance over a family of test points `M` (row `j` is the
cross-covariance vector of test point `j`): the per-test vector `j ↦ mⱼᵀ Ω⁻¹ mⱼ`. Mirrors
`predict_variance(...; diagonals = true)` returning one variance per test column. -/
def condVarFullDiagFn {nte : Nat} (Ω : Fin n → Fin n → α) (M : Fin nte → Fin n → α) : Fin nte → α :=
  fun j => condVarFullFn Ω (M j)

/-! ## Low-rank model: leading-`r` `eigh` precision plus a marginal-matching diagonal -/

/-- The leading-eigenpair part of the precision `Ω⁻¹ = Σₖ (1/λₖ) vₖ vₖᵀ`, restricted to the kept
eigenvectors `keep`: `PᵀP[a,b] = Σ_{k ∈ keep} (1/λₖ) · V[a,k] · V[b,k]`. With `keep k = (k.val < r)`
and the eigenpairs in `eigh` order this is the rank-`r` matrix `P` of `ScalarLowRankUncertaintyModel`. -/
def precisionLowRankFn (Λ : Fin n → α) (V : Fin n → Fin n → α) (keep : Fin n → Bool) :
    Fin n → Fin n → α :=
  fun a b => (List.finRange n).foldl
    (fun s k => s + (if keep k then 1 / Λ k else 0) * V a k * V b k) 0

/-- The residual diagonal `D² = dfull − d` of `ScalarLowRankUncertaintyModel`, evaluated entrywise:
`D[a,a]² = Σ_{k ∉ keep} (1/λₖ) · V[a,k]²`, the marginal precision carried by the *dropped* eigenpairs.
Added to `PᵀP`'s diagonal it restores the full marginal precision `(Ω⁻¹)[a,a]` exactly. -/
def residDiagLowRankFn (Λ : Fin n → α) (V : Fin n → Fin n → α) (keep : Fin n → Bool) : Fin n → α :=
  fun a => (List.finRange n).foldl
    (fun s k => s + (if keep k then 0 else 1 / Λ k) * V a k * V a k) 0

/-- **KernelFlows low-rank conditional variance** (`conditional_variance.jl`,
`ScalarLowRankUncertaintyModel`, `predict_variance`): `mᵀ(PᵀP + DᵀD)m`, the leading-eigenpair quadratic
form `mᵀ PᵀP m` plus the marginal-matching diagonal correction `Σₐ D[a,a]² · mₐ²` (`ret1 + ret2`). -/
def condVarLowRankFn (Λ : Fin n → α) (V : Fin n → Fin n → α) (keep : Fin n → Bool) (m : Fin n → α) :
    α :=
  bilinFn (precisionLowRankFn Λ V keep) m
    + (List.finRange n).foldl (fun s a => s + residDiagLowRankFn Λ V keep a * m a * m a) 0

/-! ## Tensor-level wrappers -/

/-- Tensor-level full-rank conditional variance `mᵀ Ω⁻¹ m`. -/
def condVarFullSpec (Ω : Tensor α (.dim n (.dim n .scalar))) (m : Tensor α (.dim n .scalar)) : α :=
  condVarFullFn (toMatFn Ω) (toVecFn m)

/-- Tensor-level full-rank conditional variance over a test family `M` (row `j` = test `j`'s
cross-covariance), returning one variance per test point. -/
def condVarFullDiagSpec {nte : Nat} (Ω : Tensor α (.dim n (.dim n .scalar)))
    (M : Tensor α (.dim nte (.dim n .scalar))) : Tensor α (.dim nte .scalar) :=
  ofVecFn (condVarFullDiagFn (toMatFn Ω) (toMatFn M))

/-- Tensor-level low-rank conditional variance `mᵀ(PᵀP + DᵀD)m`, from eigenpairs `(Λ, V)` and a kept
set `keep`. -/
def condVarLowRankSpec (Λ : Tensor α (.dim n .scalar)) (V : Tensor α (.dim n (.dim n .scalar)))
    (keep : Fin n → Bool) (m : Tensor α (.dim n .scalar)) : α :=
  condVarLowRankFn (toVecFn Λ) (toMatFn V) keep (toVecFn m)

end Spec
