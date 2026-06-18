/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Factorization.Common
public import NN.Spec.Core.Tensor.CondVar
meta import NN.Examples.Factorization.Common
meta import NN.Spec.Core.Tensor.CondVar

/-!
# KernelFlows predictive-variance models (S4 examples)

S4 ports the two uncertainty-quantification models of
[`conditional_variance.jl`](../../../../../KernelFlows.jl/src/conditional_variance.jl) and proves them
correct on an SPD kernel `Ω`: the full-rank model returns `Σ(L⁻¹m)² = mᵀΩ⁻¹m` (`condVarFullFn_eq_quadInv`),
the low-rank model's diagonal correction matches the full marginal precision exactly
(`precision_resid_diag_eq_inv`) with a nonnegative residual (`residDiagLowRankFn_nonneg`), and the GP
posterior variance is nonnegative (`condVarPostFn_nonneg`).

These `#eval`s witness all of that on a concrete SPD kernel `K` and cross-covariance vectors, checked
against the **golden values** from `conditional_variance.jl` (computed in Julia):

```
full-rank  mᵀK⁻¹m              = [0.2832, 0.6312]
low-rank   mᵀ(PᵀP+DᵀD)m (r=2)  = [0.2467, 0.5523]
diag(K⁻¹)  marginal precision  = [0.28, 0.48, 0.88, 0.72]   (matched exactly by PᵀP+DᵀD)
```

* **Positive** — the full-rank value equals the Cholesky `Σ(L⁻¹m)²` and the inverse quadratic form
  `mᵀK⁻¹m`; the low-rank diagonal `PᵀP + DᵀD` reproduces `diag(K⁻¹)` to machine precision for *any* kept
  set; the posterior variance `k(x*,x*) − mᵀK⁻¹m` is positive.
* **Negative** — on an *indefinite* matrix the Cholesky takes `√(negative)` and the full-rank value is
  `NaN` (SPD is necessary); and the rank-`2` low-rank value genuinely differs from the full value (the
  approximation has teeth) even though its marginals still match exactly.
-/

@[expose] public section

namespace NN.Examples.Factorization.CondVar

open NN.Examples.Factorization

/-- Length-`n` `Float` vector tensor from a list (missing entries `0`). -/
def mkVec {n : Nat} (xs : List Float) : Spec.Tensor Float (.dim n .scalar) :=
  Spec.ofVecFn (fun i => xs.getD i.val 0.0)

/-- A concrete SPD kernel matrix (tridiagonal, diagonally dominant ⟹ SPD), standing in for an
`spherical_sqexp` build `Ω = K(logθ)` (S2). -/
def K : Spec.Tensor Float (.dim 4 (.dim 4 .scalar)) :=
  mkMat [[4, 1, 0, 0],
         [1, 3, 1, 0],
         [0, 1, 2, 1],
         [0, 0, 1, 2]]

/-- Test-point cross-covariance vectors `m = k(x*, X)` (one per test point). -/
def m1 : Spec.Tensor Float (.dim 4 .scalar) := mkVec [1.0, 0.5, 0.2, 0.0]
def m2 : Spec.Tensor Float (.dim 4 .scalar) := mkVec [0.0, 0.1, 1.0, 0.3]

/-- The test family as a `2 × 4` matrix (row `j` = test point `j`'s cross-covariance). -/
def Mcross : Spec.Tensor Float (.dim 2 (.dim 4 .scalar)) :=
  mkMat [[1.0, 0.5, 0.2, 0.0],
         [0.0, 0.1, 1.0, 0.3]]

/-! ### Full-rank model: `Σ(L⁻¹m)² = mᵀ K⁻¹ m` -/

/-- Full-rank conditional variance per test point (the `predict_variance` diagonal). -/
def vfull : Spec.Tensor Float (.dim 2 .scalar) := Spec.condVarFullDiagSpec K Mcross

#eval IO.println s!"full-rank mᵀK⁻¹m  = {vecToList vfull}   (Julia golden [0.2832, 0.6312])"

-- Full-rank value matches the Julia `conditional_variance.jl` golden (Cholesky solve is exact).
#eval assertApproxEq "full-rank vfull[0] = 0.2832 (Julia golden)" (Spec.condVarFullSpec K m1) 0.2832 1e-4
#eval assertApproxEq "full-rank vfull[1] = 0.6312 (Julia golden)" (Spec.condVarFullSpec K m2) 0.6312 1e-4

-- The full-rank value is strictly positive (`condVarFullFn_pos`): a genuine explained variance.
#eval assertLt "full-rank mᵀK⁻¹m > 0 (explained variance positive)"
  (if Spec.condVarFullSpec K m1 > 0.0 && Spec.condVarFullSpec K m2 > 0.0 then 0.0 else 1.0)

-- Posterior variance `k(x*,x*) − mᵀK⁻¹m ≥ 0` (`condVarPostFn_nonneg`): with prior `k(x*,x*) = 1`.
#eval assertLt "posterior variance 1 − mᵀK⁻¹m ≥ 0 (never explains away more than the prior)"
  (if Spec.condVarPostFn (Spec.toMatFn K) (Spec.toVecFn m1) 1.0 ≥ 0.0
      && Spec.condVarPostFn (Spec.toMatFn K) (Spec.toVecFn m2) 1.0 ≥ 0.0 then 0.0 else 1.0)

/-! ### Low-rank model: leading-`r` `eigh` precision + marginal-matching diagonal -/

/-- The `eigh(K)` eigenvalues (diagonal) and eigenvectors (columns of `V`). -/
def eig : Spec.Tensor Float (.dim 4 .scalar) × Spec.Tensor Float (.dim 4 (.dim 4 .scalar)) :=
  Spec.symEigJacobiSpec K

def Λ : Fin 4 → Float := Spec.toVecFn eig.1
def V : Fin 4 → Fin 4 → Float := Spec.toMatFn eig.2

/-- The stable rank of eigenvalue `k` (how many eigenvalues are strictly smaller, ties broken by
index): selecting the `r` *smallest* eigenvalues of `K` = the `r` *largest* of `K⁻¹`, exactly as
`ScalarLowRankUncertaintyModel` keeps `λ[1:r]` after `eigen`'s ascending sort. -/
def rankOf (k : Fin 4) : Nat :=
  (List.finRange 4).foldl (fun c j =>
    if Λ j < Λ k || (Λ j == Λ k && j.val < k.val) then c + 1 else c) 0

/-- Keep the `r` smallest-eigenvalue eigenpairs. -/
def keepR (r : Nat) : Fin 4 → Bool := fun k => rankOf k < r

/-- Low-rank conditional variance with rank `r = 2`. -/
def vlow (m : Spec.Tensor Float (.dim 4 .scalar)) : Float :=
  Spec.condVarLowRankFn Λ V (keepR 2) (Spec.toVecFn m)

#eval IO.println s!"eigenvalues (eigh) = {vecToList eig.1}"
#eval IO.println s!"low-rank r=2 value = [{vlow m1}, {vlow m2}]   (Julia golden [0.2467, 0.5523])"

-- Low-rank value matches the Julia golden (depends on the Jacobi `eigh`, looser tolerance).
#eval assertApproxEq "low-rank vlow[0] = 0.2467 (Julia golden, r=2)" (vlow m1) 0.24669709 1e-3
#eval assertApproxEq "low-rank vlow[1] = 0.5523 (Julia golden, r=2)" (vlow m2) 0.55226740 1e-3

/-! ### The diagonal correction matches the full marginal precision exactly -/

/-- Full marginal precision `(K⁻¹)[a,a] = Σₖ (1/λₖ) V[a,k]²` (the `keep = all` precision diagonal). -/
def fullInvDiag (a : Fin 4) : Float := Spec.precisionLowRankFn Λ V (fun _ => true) a a

/-- The rank-`2` low-rank marginal `(PᵀP)[a,a] + D[a,a]²`. -/
def lowMarginal (a : Fin 4) : Float :=
  Spec.precisionLowRankFn Λ V (keepR 2) a a + Spec.residDiagLowRankFn Λ V (keepR 2) a

/-- Max over `a` of the marginal mismatch `|(PᵀP+DᵀD)[a,a] − (K⁻¹)[a,a]|`. -/
def maxMarginalErr : Float :=
  (List.finRange 4).foldl (fun acc a => max acc (Float.abs (lowMarginal a - fullInvDiag a))) 0.0

#eval IO.println s!"diag(K⁻¹)         = {(List.finRange 4).map fullInvDiag}   (Julia golden [0.28, 0.48, 0.88, 0.72])"
#eval IO.println s!"low-rank diag     = {(List.finRange 4).map lowMarginal}"

-- `precision_resid_diag_eq_inv`: the diagonal correction restores the full marginal precision EXACTLY,
-- regardless of which 2 eigenpairs are kept (a pure sum split — machine precision, not Jacobi accuracy).
#eval assertLt "low-rank diag = diag(K⁻¹) exactly (marginal uncertainties matched)" maxMarginalErr 1e-9

-- The reconstructed `diag(K⁻¹)` matches the Julia golden (Jacobi `eigh` accuracy).
#eval assertApproxEq "diag(K⁻¹)[2] = 0.88 (Julia golden)" (fullInvDiag 2) 0.88 1e-3

/-! ### Negative controls: SPD necessity, and the low-rank approximation has teeth -/

/-- A symmetric but **indefinite** matrix (eigenvalues `{3, −1}`): the Cholesky takes `√(negative)`. -/
def Kbad : Spec.Tensor Float (.dim 2 (.dim 2 .scalar)) :=
  mkMat [[1, 2],
         [2, 1]]

def mBad : Spec.Tensor Float (.dim 2 .scalar) := mkVec [1.0, 1.0]

#eval IO.println s!"indefinite Kbad: condVarFull = {Spec.condVarFullSpec Kbad mBad}  (NaN ⟹ SPD required)"

-- Negative — the full-rank model on a non-SPD matrix is `NaN`: SPD-ness is necessary.
#eval assertReconFails "indefinite matrix: full-rank conditional variance is NaN (SPD necessary)"
  (Float.abs (Spec.condVarFullSpec Kbad mBad))

-- Negative — the rank-`2` low-rank value genuinely differs from the full value: it is a real
-- approximation (the off-diagonal precision is dropped) even though the marginals match exactly.
#eval assertGe "low-rank r=2 ≠ full-rank (the approximation has teeth)"
  (Float.abs (vlow m1 - Spec.condVarFullSpec K m1)) 0.01

end NN.Examples.Factorization.CondVar
