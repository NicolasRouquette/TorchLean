/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.KernelLoss
public import NN.Proofs.Tensor.Basic.FactorizationsSolve

/-!
# The KernelFlows cross-validation losses are well-posed on an SPD kernel (S3)

S2 proved the KernelFlows kernel-matrix build `Ω = K(logθ)` is symmetric positive-definite (SPD). This
file uses that to show the four production losses of `loss_functions.jl` — `ρ_KF`, `ρ_LOI`, `ρ_LOO`,
`ρ_MLE` — are *well-posed* on an SPD `Ω`: their shared denominator never vanishes, the bounded ones are
genuinely bounded, and `ρ_MLE`'s data term is exactly the GP quadratic form.

## The shared primitive: the regularized quadratic form `yᵀ Ω⁻¹ y`

Every KernelFlows loss is built on `quadInvFn Ω y = dotFn y (cholSolveFn (choleskyFn Ω) y)`, the
`yᵀ Ω⁻¹ y` quadratic form computed through the landed Cholesky solve. On an SPD `Ω`:

* `quadInvFn_eq_dotProduct_inv` — it *is* the inverse quadratic form `y ⬝ᵥ (Ω⁻¹ *ᵥ y)` (the Cholesky
  solve equals `Ω⁻¹ *ᵥ y` because `Ω` is invertible);
* `quadInvFn_nonneg` / `quadInvFn_pos` — `Ω⁻¹` is itself positive-(semi)definite, so the form is `≥ 0`,
  and *strictly* `> 0` for `y ≠ 0`. This is what makes every loss denominator nonzero.

## The losses

* `rhoLOIFn_le_one` / `rhoKFFn_le_one` — `ρ_LOI`, `ρ_KF` are `1 − (nonneg)/(positive) ≤ 1`. For `ρ_KF`
  the center-block quadratic form is nonnegative because a *principal submatrix of an SPD matrix is SPD*
  (`Matrix.PosDef.submatrix`), so the loss is bounded for any choice of center indices.
* `rhoMLE_data_eq_quadInv` — the `ρ_MLE` data term `½‖L⁻¹y‖²` equals `½ yᵀΩ⁻¹y` (the KernelFlows source
  asserts this equality `a1 == a2` in a comment; here it is proved), and `rhoMLE_data_nonneg` gives
  `0 ≤ ½‖L⁻¹y‖²`. Hence `ρ_MLE` is the Gaussian-process negative-log-marginal-likelihood quadratic form.

`ρ_LOO`'s denominator is the same `quadInvFn Ω y`, so `quadInvFn_pos` discharges its well-posedness too.

Scope honesty: everything is exact over `ℝ`, proved from S2's SPD-ness through the landed verified
Cholesky/triangular-solve correctness — no asymptotics, no `sorry`. Differentiation of the losses (the
KernelFlows training gradient) is S6/S7.
-/

@[expose] public section

namespace Spec.Factorization

open Matrix
open scoped BigOperators
open Spec.Factorization.Reconstruction

variable {n : Nat}

/-! ## The Cholesky solve of an SPD matrix is the inverse -/

/-- For an SPD `Ω`, the Cholesky solve `cholSolveFn (choleskyFn Ω) b` solves `Ω·x = b` exactly. The
positive-pivot keystone makes the executable Cholesky a genuine factor `Ω = L·Lᵀ`, and the verified
two-pass substitution then solves `(L·Lᵀ)·x = b`. -/
theorem cholSolveFn_mulVec_of_posDef (Ω : Fin n → Fin n → ℝ) (hpd : (Matrix.of Ω).PosDef)
    (b : Fin n → ℝ) :
    Matrix.of Ω *ᵥ (Spec.cholSolveFn (Spec.choleskyFn Ω) b) = b := by
  obtain ⟨⟨hlowM, hrecon⟩, hpos⟩ := cholesky_posDef Ω hpd
  have hlow : ∀ i j, i < j → Spec.choleskyFn Ω i j = 0 := fun i j hij => by simpa using hlowM i j hij
  have hdiag : ∀ i, Spec.choleskyFn Ω i i ≠ 0 := fun i => ne_of_gt (hpos i)
  rw [hrecon]
  exact cholSolveFn_mulVec (Spec.choleskyFn Ω) hlow hdiag b

/-- **The KernelFlows quadratic form is the inverse quadratic form.** For an SPD `Ω`,
`quadInvFn Ω y = y ⬝ᵥ (Ω⁻¹ *ᵥ y) = yᵀ Ω⁻¹ y`: the Cholesky solve under it equals `Ω⁻¹ *ᵥ y`, because
`Ω` is invertible. This is the `inv(Symmetric(Ω))` quadratic form every KernelFlows loss is built on. -/
theorem quadInvFn_eq_dotProduct_inv (Ω : Fin n → Fin n → ℝ) (hpd : (Matrix.of Ω).PosDef)
    (y : Fin n → ℝ) :
    Spec.quadInvFn Ω y = y ⬝ᵥ ((Matrix.of Ω)⁻¹ *ᵥ y) := by
  have hsolve : Matrix.of Ω *ᵥ (Spec.cholSolveFn (Spec.choleskyFn Ω) y) = y :=
    cholSolveFn_mulVec_of_posDef Ω hpd y
  have hdet : IsUnit (Matrix.of Ω).det := (Matrix.isUnit_iff_isUnit_det _).mp hpd.isUnit
  have hinv : Spec.cholSolveFn (Spec.choleskyFn Ω) y = (Matrix.of Ω)⁻¹ *ᵥ y := by
    calc Spec.cholSolveFn (Spec.choleskyFn Ω) y
        = ((Matrix.of Ω)⁻¹ * Matrix.of Ω) *ᵥ (Spec.cholSolveFn (Spec.choleskyFn Ω) y) := by
          rw [Matrix.nonsing_inv_mul _ hdet, Matrix.one_mulVec]
      _ = (Matrix.of Ω)⁻¹ *ᵥ (Matrix.of Ω *ᵥ (Spec.cholSolveFn (Spec.choleskyFn Ω) y)) := by
          rw [Matrix.mulVec_mulVec]
      _ = (Matrix.of Ω)⁻¹ *ᵥ y := by rw [hsolve]
  rw [Spec.quadInvFn, hinv, dotFn_eq_sum]
  rfl

/-- **The KernelFlows quadratic form is nonnegative** on an SPD `Ω` (the regularized inverse `Ω⁻¹` is
positive-semidefinite). -/
theorem quadInvFn_nonneg (Ω : Fin n → Fin n → ℝ) (hpd : (Matrix.of Ω).PosDef) (y : Fin n → ℝ) :
    0 ≤ Spec.quadInvFn Ω y := by
  rw [quadInvFn_eq_dotProduct_inv Ω hpd y]
  simpa using ((hpd.inv).posSemidef).dotProduct_mulVec_nonneg y

/-- **The KernelFlows quadratic form is strictly positive for `y ≠ 0`** on an SPD `Ω` (the regularized
inverse `Ω⁻¹` is positive-definite). This is what makes every loss denominator `yᵀ Ω⁻¹ y` nonzero, so the
losses are well-defined. -/
theorem quadInvFn_pos (Ω : Fin n → Fin n → ℝ) (hpd : (Matrix.of Ω).PosDef) {y : Fin n → ℝ}
    (hy : y ≠ 0) : 0 < Spec.quadInvFn Ω y := by
  rw [quadInvFn_eq_dotProduct_inv Ω hpd y]
  simpa using (hpd.inv).dotProduct_mulVec_pos hy

/-! ## `ρ_LOI` and `ρ_KF` are bounded above by `1` -/

/-- **`ρ_LOI ≤ 1`.** The numerator `yᵀy / Ω₀₀ / n` is nonnegative (`Ω₀₀ > 0` since `Ω` is SPD) and the
denominator `yᵀΩ⁻¹y > 0` for `y ≠ 0`, so `ρ_LOI = 1 − (nonneg)/(pos) ≤ 1`. -/
theorem rhoLOIFn_le_one {m : Nat} (Ω : Fin (m + 1) → Fin (m + 1) → ℝ)
    (hpd : (Matrix.of Ω).PosDef) {y : Fin (m + 1) → ℝ} (hy : y ≠ 0) :
    Spec.rhoLOIFn Ω y ≤ 1 := by
  have hq : 0 < Spec.quadInvFn Ω y := quadInvFn_pos Ω hpd hy
  have hΩ00 : 0 < Ω 0 0 := hpd.diag_pos
  have hyy : 0 ≤ Spec.dotFn y y := by
    rw [dotFn_eq_sum]; exact Finset.sum_nonneg (fun i _ => mul_self_nonneg _)
  have hnum : 0 ≤ Spec.dotFn y y / Ω 0 0 / ((m + 1 : Nat) : ℝ) :=
    div_nonneg (div_nonneg hyy hΩ00.le) (by positivity)
  have hfrac : 0 ≤ (Spec.dotFn y y / Ω 0 0 / ((m + 1 : Nat) : ℝ)) / Spec.quadInvFn Ω y :=
    div_nonneg hnum hq.le
  rw [Spec.rhoLOIFn]
  -- normalize the `Context`-coercion `Coe.coe (m+1)` to the standard `Nat.cast` (defeq) so it matches
  change 1 - (Spec.dotFn y y / Ω 0 0 / ((m + 1 : Nat) : ℝ)) / Spec.quadInvFn Ω y ≤ 1
  linarith

/-- **`ρ_KF ≤ 1`** for any choice of center indices `e`. The center-block quadratic form
`y_cᵀ Ω_c⁻¹ y_c` is nonnegative because a principal submatrix of an SPD matrix is SPD
(`Matrix.PosDef.submatrix`, `e` injective), and the full denominator `yᵀΩ⁻¹y > 0` for `y ≠ 0`. -/
theorem rhoKFFn_le_one {nc : Nat} (Ω : Fin n → Fin n → ℝ) (hpd : (Matrix.of Ω).PosDef)
    {y : Fin n → ℝ} (hy : y ≠ 0) (e : Fin nc → Fin n) (he : Function.Injective e) :
    Spec.rhoKFFn Ω y e ≤ 1 := by
  have hq : 0 < Spec.quadInvFn Ω y := quadInvFn_pos Ω hpd hy
  have hpdc : (Matrix.of (fun i j => Ω (e i) (e j))).PosDef := hpd.submatrix he
  have hqc : 0 ≤ Spec.quadInvFn (fun i j => Ω (e i) (e j)) (fun i => y (e i)) :=
    quadInvFn_nonneg _ hpdc _
  have hfrac : 0 ≤ Spec.quadInvFn (fun i j => Ω (e i) (e j)) (fun i => y (e i)) / Spec.quadInvFn Ω y :=
    div_nonneg hqc hq.le
  rw [Spec.rhoKFFn]; linarith

/-! ## `ρ_MLE`: the data term is the GP quadratic form -/

/-- **`ρ_MLE`'s data term is the inverse quadratic form.** With `L` the Cholesky factor of `Ω` and
`z = L⁻¹y` (forward substitution), `‖z‖² = yᵀΩ⁻¹y`. KernelFlows' `ρ_MLE` asserts this (`a1 == a2`) in a
comment; here it is proved from the verified forward/back substitutions: `L·z = y` and `Lᵀ·x = z` give
`zᵀz = (L z)ᵀ x = yᵀ x = yᵀΩ⁻¹y` with `x = cholSolveFn L y`. -/
theorem rhoMLE_data_eq_quadInv (Ω : Fin n → Fin n → ℝ) (hpd : (Matrix.of Ω).PosDef) (y : Fin n → ℝ) :
    Spec.dotFn (Spec.triSolveLowerFn (Spec.choleskyFn Ω) y)
        (Spec.triSolveLowerFn (Spec.choleskyFn Ω) y)
      = Spec.quadInvFn Ω y := by
  obtain ⟨⟨hlowM, _⟩, hpos⟩ := cholesky_posDef Ω hpd
  have hlow : ∀ i j, i < j → Spec.choleskyFn Ω i j = 0 := fun i j hij => by simpa using hlowM i j hij
  have hdiag : ∀ i, Spec.choleskyFn Ω i i ≠ 0 := fun i => ne_of_gt (hpos i)
  set L := Spec.choleskyFn Ω with hLdef
  set z := Spec.triSolveLowerFn L y with hzdef
  set x := Spec.cholSolveFn L y with hxdef
  -- `L·z = y` (row `i`).
  have hLz : ∀ i, (∑ k, L i k * z k) = y i := fun i => triSolveLowerFn_mulVec L hlow hdiag y i
  -- `Lᵀ·x = z` (row `i`): `∑ k, L k i · x k = z i`.
  have hup : ∀ i j, j < i → (fun a b => L b a) i j = 0 := fun i j hji => hlow j i hji
  have hUdiag : ∀ i, (fun a b => L b a) i i ≠ 0 := fun i => hdiag i
  have hUx : ∀ i, (∑ k, L k i * x k) = z i := fun i =>
    triSolveUpperFn_mulVec (fun a b => L b a) hup hUdiag z i
  show Spec.dotFn z z = Spec.dotFn y x
  rw [dotFn_eq_sum, dotFn_eq_sum]
  calc ∑ i, z i * z i
      = ∑ i, z i * (∑ k, L k i * x k) := by
        refine Finset.sum_congr rfl (fun i _ => ?_); rw [hUx i]
    _ = ∑ i, ∑ k, z i * (L k i * x k) := by
        refine Finset.sum_congr rfl (fun i _ => ?_); rw [Finset.mul_sum]
    _ = ∑ k, ∑ i, z i * (L k i * x k) := Finset.sum_comm
    _ = ∑ k, x k * (∑ i, L k i * z i) := by
        refine Finset.sum_congr rfl (fun k _ => ?_)
        rw [Finset.mul_sum]; exact Finset.sum_congr rfl (fun i _ => by ring)
    _ = ∑ k, x k * y k := by
        refine Finset.sum_congr rfl (fun k _ => ?_); rw [hLz k]
    _ = ∑ i, y i * x i := Finset.sum_congr rfl (fun k _ => by ring)

/-- **`ρ_MLE`'s data term is nonnegative**: `0 ≤ ½‖L⁻¹y‖²` (a half of a sum of squares). -/
theorem rhoMLE_data_nonneg (Ω : Fin n → Fin n → ℝ) (y : Fin n → ℝ) :
    0 ≤ (Numbers.pointfive : ℝ)
      * Spec.dotFn (Spec.triSolveLowerFn (Spec.choleskyFn Ω) y)
          (Spec.triSolveLowerFn (Spec.choleskyFn Ω) y) := by
  have hpf : (Numbers.pointfive : ℝ) = 1 / 2 := by norm_num [Numbers.pointfive]
  have hzz : 0 ≤ Spec.dotFn (Spec.triSolveLowerFn (Spec.choleskyFn Ω) y)
      (Spec.triSolveLowerFn (Spec.choleskyFn Ω) y) := by
    rw [dotFn_eq_sum]; exact Finset.sum_nonneg (fun i _ => mul_self_nonneg _)
  rw [hpf]; positivity

/-! ## Tensor-level corollaries -/

/-- Tensor-level: the KernelFlows quadratic form `yᵀΩ⁻¹y` is strictly positive for `y ≠ 0` on an SPD
kernel — so the loss denominators are nonzero and every `ρ` is well-defined. -/
theorem quadInvSpec_pos (Ω : Spec.Tensor ℝ (.dim n (.dim n .scalar)))
    (y : Spec.Tensor ℝ (.dim n .scalar)) (hpd : (Matrix.of (Spec.toMatFn Ω)).PosDef)
    (hy : Spec.toVecFn y ≠ 0) : 0 < Spec.quadInvSpec Ω y :=
  quadInvFn_pos _ hpd hy

/-- Tensor-level: the `ρ_MLE` data term equals the GP quadratic form `½‖L⁻¹y‖² = ½ yᵀΩ⁻¹y`. -/
theorem rhoMLESpec_data_eq_quadInv (Ω : Spec.Tensor ℝ (.dim n (.dim n .scalar)))
    (y : Spec.Tensor ℝ (.dim n .scalar)) (hpd : (Matrix.of (Spec.toMatFn Ω)).PosDef) :
    Spec.dotFn (Spec.triSolveLowerFn (Spec.choleskyFn (Spec.toMatFn Ω)) (Spec.toVecFn y))
        (Spec.triSolveLowerFn (Spec.choleskyFn (Spec.toMatFn Ω)) (Spec.toVecFn y))
      = Spec.quadInvSpec Ω y :=
  rhoMLE_data_eq_quadInv _ hpd _

end Spec.Factorization
