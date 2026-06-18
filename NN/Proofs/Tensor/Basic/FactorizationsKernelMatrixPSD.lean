/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Tensor.Basic.FactorizationsKernelMatrix
public import NN.Proofs.Tensor.Basic.FactorizationsKernels
public import NN.Proofs.Tensor.Basic.FactorizationsSolve

/-!
# The KernelFlows kernel-matrix build is SPD ⟹ Cholesky / ridge-solve fires (S2)

S1 built the KernelFlows unary kernel matrix `K(logθ)` and proved it symmetric. The verified
Cholesky / `find_gamma` / ridge-solve development consumes `K` under a `PosDef` (or `PosSemidef`)
hypothesis; this file discharges *positive-definiteness* — the second half of S1's symmetry — so that
standing hypothesis is met and the landed positive-pivot keystone `choleskyFn_diag_pos_of_posDef`
fires.

## The structural keystone: the nugget turns PSD into SPD

The build is `K = R + δ·I + scale·Φ·Φᵀ` with `R` the radial block, `δ = nuggetFn logθ₄ > 0` the nugget,
and `scale = exp logθ₃ ≥ 0` the linear weight. `Φ·Φᵀ` is a Gram (PSD), so the moment the radial block
`R` is PSD the *base* `R + scale·Φ·Φᵀ` is PSD, and the strictly-positive nugget `δ·I` lifts it to
positive-definite:

* `unaryKernelBuild_posDef` — for **any** PSD radial matrix `R`, `scale ≥ 0`, `δ > 0`, the assembled
  `K[i,j] = R[i,j] + δ·[i=j] + scale·⟨Φᵢ, Φⱼ⟩` is `PosDef`. This is exactly why KernelFlows adds a
  nugget: it is what makes a merely-PSD kernel SPD, so `K = L·Lᵀ` succeeds with strictly positive
  pivots and the ridge solve is exact.

## Where the radial PSD comes from

The build keystone reduces `K.PosDef` to the single fact `R.PosSemidef`. For the **RBF /
`spherical_sqexp`** radial kernel this is *discharged in full*: the RBF is `exp` of the **squared**
Euclidean distance, so `posSemidef_gaussianRadial` (the entrywise-exponential / Schur route) applies and

* `kernelMatrixSqexpFn_posDef` / `kernelMatrixSqexpSpec_posDef` — the RBF unary build is `PosDef` for
  **every** `logθ`, with no side hypotheses. `kernelMatrixSqexpFn_cholesky_pos` then gives strictly
  positive Cholesky pivots, and `kernelMatrixSqexpFn_solveRidge_exact` the exact regularized solve.

For the **Matérn-3/2 and Matérn-5/2** radial kernels (and KernelFlows' `spherical_exp` and
`inverse_quadratic`) the radial block carries the *bare* Euclidean distance `d = ‖Xᵢ − Xⱼ‖`, not `d²`.
These kernels are genuinely positive-definite — but their PSD-ness is *not* elementary: it rests on a
Gaussian scale-mixture (Bochner / Schoenberg) representation `φ(d) = ∫₀^∞ e^{−t d²} dμ(t)` whose
defining theorems are absent from Mathlib v4.30.0, so there is no finite Hadamard/Schur certificate.
Rather than fake it, we state the reduction with the radial-PSD as a clean hypothesis:

* `kernelMatrixMatern32Fn_posDef_of_radial` / `…Spec…` / `…Matern52…` — `R.PosSemidef → K.PosDef`,
  isolating exactly the open analytic fact. Discharging it (the scale-mixture integral representation)
  is scoped as a Year-2 analytic deliverable; the RBF result above shows the keystone is real and the
  pipeline closes end-to-end whenever the radial block is PSD.
-/

@[expose] public section

namespace Spec.Factorization

open Matrix
open scoped BigOperators
open Spec.Factorization.Reconstruction

variable {n d : Nat}

/-! ## The build keystone: PSD radial block + positive nugget ⟹ PosDef -/

/-- **The unary kernel-matrix build is positive-definite once its radial block is PSD.** Given any
positive-semidefinite radial matrix `R`, a nonnegative linear weight `scale`, and a strictly positive
nugget `δ`, the assembled `K[i,j] = R[i,j] + δ·[i=j] + scale·⟨Φᵢ, Φⱼ⟩` (with `Φ = maskColsFn X wlin`) is
`PosDef`: the radial block and the PSD linear Gram `scale·Φ·Φᵀ` sum to a PSD base, and the nugget `δ·I`
(`δ > 0`) lifts it to strictly positive-definite. This is the structural reason KernelFlows adds a
nugget — it is exactly what turns a merely-PSD kernel into an SPD one, so the Cholesky/ridge-solve
keystone fires. -/
theorem unaryKernelBuild_posDef (R : Fin n → Fin n → ℝ) (hR : (Matrix.of R).PosSemidef)
    (X : Fin n → Fin d → ℝ) (wlin : Fin d → ℝ) {scale δ : ℝ} (hscale : 0 ≤ scale) (hδ : 0 < δ) :
    (Matrix.of (fun i j => R i j + (if i = j then δ else 0)
      + scale * Spec.dotFn (Spec.maskColsFn X wlin i) (Spec.maskColsFn X wlin j))).PosDef := by
  set Φ : Matrix (Fin n) (Fin d) ℝ := Matrix.of (Spec.maskColsFn X wlin) with hΦ
  -- the linear-term Gram `Φ·Φᵀ` is PSD
  have hGram : (Φ * Φᵀ).PosSemidef := by
    have h := Matrix.posSemidef_self_mul_conjTranspose Φ
    have he : (Φᴴ : Matrix (Fin d) (Fin n) ℝ) = Φᵀ := by
      ext a b; simp [Matrix.conjTranspose_apply, Matrix.transpose_apply]
    rwa [he] at h
  -- base `R + scale•(Φ·Φᵀ)` is PSD
  have hbase : (Matrix.of R + scale • (Φ * Φᵀ)).PosSemidef := hR.add (hGram.smul hscale)
  -- the nugget `δ•I` is PosDef
  have hnug : ((δ : ℝ) • (1 : Matrix (Fin n) (Fin n) ℝ)).PosDef := Matrix.PosDef.one.smul hδ
  -- the assembled matrix is `base + δ•I`
  have heq : (Matrix.of (fun i j => R i j + (if i = j then δ else 0)
        + scale * Spec.dotFn (Spec.maskColsFn X wlin i) (Spec.maskColsFn X wlin j)))
      = (Matrix.of R + scale • (Φ * Φᵀ)) + δ • (1 : Matrix (Fin n) (Fin n) ℝ) := by
    ext i j
    by_cases h : i = j
    · simp only [Matrix.add_apply, Matrix.smul_apply, Matrix.of_apply, smul_eq_mul, Matrix.mul_apply,
        Matrix.transpose_apply, Matrix.one_apply, if_pos h, hΦ, Spec.maskColsFn, dotFn_eq_sum]
      ring
    · simp only [Matrix.add_apply, Matrix.smul_apply, Matrix.of_apply, smul_eq_mul, Matrix.mul_apply,
        Matrix.transpose_apply, Matrix.one_apply, if_neg h, hΦ, Spec.maskColsFn, dotFn_eq_sum]
      ring
  rw [heq]
  exact Matrix.PosDef.posSemidef_add hbase hnug

/-! ## RBF (`spherical_sqexp`) radial block: PSD discharged in full -/

/-- **The RBF (`spherical_sqexp`) radial block is positive-semidefinite.** For `a ≥ 0`, `b > 0`,
`a·exp(−½·‖Xᵢ − Xⱼ‖²/b)` is PSD: squaring the Euclidean distance turns it into `a` times the
multivariate Gaussian `exp(−c·∑ₖ(X[i,k] − X[j,k])²)` with `c = ½/b ≥ 0`, which is PSD by
`posSemidef_gaussianRadial`. -/
theorem posSemidef_sphericalSqexpRadial (X : Fin n → Fin d → ℝ) {a b : ℝ} (ha : 0 ≤ a) (hb : 0 < b) :
    (Matrix.of (fun i j => Spec.sphericalSqexpFn (Spec.pairwiseEuclideanFn X i j) a b)).PosSemidef := by
  set c : ℝ := (1 / 2) / b with hc
  have hcnn : 0 ≤ c := by rw [hc]; positivity
  have hnpf : (Numbers.neg_point_five : ℝ) = -(1 / 2 : ℝ) := by
    show (-0.5 : ℝ) = -(1 / 2); norm_num
  have heq : (Matrix.of (fun i j => Spec.sphericalSqexpFn (Spec.pairwiseEuclideanFn X i j) a b))
      = a • Matrix.of (fun i j => Real.exp (-(c * ∑ k, (X i k - X j k) * (X i k - X j k)))) := by
    ext i j
    simp only [Matrix.smul_apply, Matrix.of_apply, smul_eq_mul, Spec.sphericalSqexpFn]
    -- the squared Euclidean distance is the sum of squared coordinate gaps
    have hnn : (0 : ℝ) ≤ Spec.dotFn (fun k => X i k - X j k) (fun k => X i k - X j k) := by
      rw [dotFn_eq_sum]; exact Finset.sum_nonneg (fun k _ => mul_self_nonneg _)
    have hsq : Spec.pairwiseEuclideanFn X i j * Spec.pairwiseEuclideanFn X i j
        = ∑ k, (X i k - X j k) * (X i k - X j k) := by
      show Real.sqrt (Spec.dotFn (fun k => X i k - X j k) (fun k => X i k - X j k))
          * Real.sqrt (Spec.dotFn (fun k => X i k - X j k) (fun k => X i k - X j k))
          = ∑ k, (X i k - X j k) * (X i k - X j k)
      rw [Real.mul_self_sqrt hnn, dotFn_eq_sum]
    congr 1
    rw [hsq]
    congr 1
    rw [hnpf, hc]; ring
  rw [heq]
  exact (posSemidef_gaussianRadial X hcnn).smul ha

/-- **The RBF unary kernel-matrix build is positive-definite — for every `logθ`.** The amplitude
`exp logθ₁ ≥ 0` and length scale `exp logθ₂ > 0` make the radial block PSD
(`posSemidef_sphericalSqexpRadial`), the linear weight `exp logθ₃ ≥ 0`, and the nugget
`δ = nuggetFn logθ₄ = exp(−12) + exp logθ₄ > 0` — so `unaryKernelBuild_posDef` gives `PosDef`. -/
theorem kernelMatrixSqexpFn_posDef (X : Fin n → Fin d → ℝ) (wlin : Fin d → ℝ) (logθ : Fin 4 → ℝ) :
    (Matrix.of (Spec.kernelMatrixSqexpFn X wlin logθ)).PosDef := by
  have hR := posSemidef_sphericalSqexpRadial X (a := Real.exp (logθ 0)) (b := Real.exp (logθ 1))
    (Real.exp_pos _).le (Real.exp_pos _)
  have hscale : (0 : ℝ) ≤ Real.exp (logθ 2) := (Real.exp_pos _).le
  have hδ : (0 : ℝ) < Spec.nuggetFn (logθ 3) := add_pos (Real.exp_pos _) (Real.exp_pos _)
  exact unaryKernelBuild_posDef _ hR X wlin hscale hδ

/-- Tensor-level: the RBF unary build is `PosDef` for every `logθ`. -/
theorem kernelMatrixSqexpSpec_posDef (X : Spec.Tensor ℝ (.dim n (.dim d .scalar)))
    (wlin : Spec.Tensor ℝ (.dim d .scalar)) (logθ : Spec.Tensor ℝ (.dim 4 .scalar)) :
    (Matrix.of (Spec.toMatFn (Spec.kernelMatrixSqexpSpec X wlin logθ))).PosDef := by
  have hround : Spec.toMatFn (Spec.kernelMatrixSqexpSpec X wlin logθ)
      = Spec.kernelMatrixSqexpFn (Spec.toMatFn X) (Spec.toVecFn wlin) (Spec.toVecFn logθ) := by
    funext i j; rfl
  rw [hround]; exact kernelMatrixSqexpFn_posDef _ _ _

/-- **The RBF build's Cholesky pivots are strictly positive** — the positive-pivot keystone fires, so
`K = L·Lᵀ` succeeds with `0 < L[m,m]` for every `m`. -/
theorem kernelMatrixSqexpFn_cholesky_pos (X : Fin n → Fin d → ℝ) (wlin : Fin d → ℝ) (logθ : Fin 4 → ℝ)
    (m : Fin n) : 0 < Spec.choleskyFn (Spec.kernelMatrixSqexpFn X wlin logθ) m m :=
  choleskyFn_diag_pos_of_posDef _ (kernelMatrixSqexpFn_posDef X wlin logθ) m

/-- **The kernel-ridge solve on the RBF build is exact for any `γ > 0`** — the build is `PosDef`, hence
`PosSemidef`, so `(K + γ·I)·x = b` is solved exactly by `solveRidgeFn`. -/
theorem kernelMatrixSqexpFn_solveRidge_exact (X : Fin n → Fin d → ℝ) (wlin : Fin d → ℝ)
    (logθ : Fin 4 → ℝ) {γ : ℝ} (hγ : 0 < γ) (b : Fin n → ℝ) :
    (Matrix.of (Spec.addScaledIdFn (Spec.kernelMatrixSqexpFn X wlin logθ) γ)) *ᵥ
      (Spec.solveRidgeFn (Spec.kernelMatrixSqexpFn X wlin logθ) γ b) = b :=
  solveRidgeFn_mulVec_of_posSemidef _ γ b (kernelMatrixSqexpFn_posDef X wlin logθ).posSemidef hγ

/-! ## Matérn radial blocks: the reduction to the (scale-mixture-deferred) radial PSD -/

/-- **The Matérn-3/2 unary build is `PosDef` once its radial block is PSD.** `unaryKernelBuild_posDef`
specialized to the Matérn-3/2 radial kernel: the linear weight `exp logθ₃ ≥ 0` and nugget
`nuggetFn logθ₄ > 0` are automatic, so the *only* remaining hypothesis is positive-semidefiniteness of
the radial block `Matérn32(‖Xᵢ − Xⱼ‖; exp logθ₁, exp logθ₂)` — the classical Matérn covariance-validity
fact, which over ℝ rests on a Gaussian scale-mixture representation outside Mathlib v4.30.0. -/
theorem kernelMatrixMatern32Fn_posDef_of_radial (X : Fin n → Fin d → ℝ) (wlin : Fin d → ℝ)
    (logθ : Fin 4 → ℝ)
    (hR : (Matrix.of (fun i j => Spec.matern32Fn (Spec.pairwiseEuclideanFn X i j)
      (Real.exp (logθ 0)) (Real.exp (logθ 1)))).PosSemidef) :
    (Matrix.of (Spec.kernelMatrixMatern32Fn X wlin logθ)).PosDef := by
  have hscale : (0 : ℝ) ≤ Real.exp (logθ 2) := (Real.exp_pos _).le
  have hδ : (0 : ℝ) < Spec.nuggetFn (logθ 3) := add_pos (Real.exp_pos _) (Real.exp_pos _)
  exact unaryKernelBuild_posDef _ hR X wlin hscale hδ

/-- Tensor-level Matérn-3/2 build `PosDef` from its radial PSD. -/
theorem kernelMatrixMatern32Spec_posDef_of_radial (X : Spec.Tensor ℝ (.dim n (.dim d .scalar)))
    (wlin : Spec.Tensor ℝ (.dim d .scalar)) (logθ : Spec.Tensor ℝ (.dim 4 .scalar))
    (hR : (Matrix.of (fun i j => Spec.matern32Fn (Spec.pairwiseEuclideanFn (Spec.toMatFn X) i j)
      (Real.exp (Spec.toVecFn logθ 0)) (Real.exp (Spec.toVecFn logθ 1)))).PosSemidef) :
    (Matrix.of (Spec.toMatFn (Spec.kernelMatrixMatern32Spec X wlin logθ))).PosDef := by
  have hround : Spec.toMatFn (Spec.kernelMatrixMatern32Spec X wlin logθ)
      = Spec.kernelMatrixMatern32Fn (Spec.toMatFn X) (Spec.toVecFn wlin) (Spec.toVecFn logθ) := by
    funext i j; rfl
  rw [hround]; exact kernelMatrixMatern32Fn_posDef_of_radial _ _ _ hR

/-- **The Matérn-3/2 build's Cholesky pivots are strictly positive once its radial block is PSD.** -/
theorem kernelMatrixMatern32Fn_cholesky_pos_of_radial (X : Fin n → Fin d → ℝ) (wlin : Fin d → ℝ)
    (logθ : Fin 4 → ℝ)
    (hR : (Matrix.of (fun i j => Spec.matern32Fn (Spec.pairwiseEuclideanFn X i j)
      (Real.exp (logθ 0)) (Real.exp (logθ 1)))).PosSemidef) (m : Fin n) :
    0 < Spec.choleskyFn (Spec.kernelMatrixMatern32Fn X wlin logθ) m m :=
  choleskyFn_diag_pos_of_posDef _ (kernelMatrixMatern32Fn_posDef_of_radial X wlin logθ hR) m

/-- **The Matérn-5/2 unary build is `PosDef` once its radial block is PSD** (same reduction, Matérn-5/2
radial kernel). -/
theorem kernelMatrixMatern52Fn_posDef_of_radial (X : Fin n → Fin d → ℝ) (wlin : Fin d → ℝ)
    (logθ : Fin 4 → ℝ)
    (hR : (Matrix.of (fun i j => Spec.matern52Fn (Spec.pairwiseEuclideanFn X i j)
      (Real.exp (logθ 0)) (Real.exp (logθ 1)))).PosSemidef) :
    (Matrix.of (fun i j =>
        Spec.matern52Fn (Spec.pairwiseEuclideanFn X i j) (Real.exp (logθ 0)) (Real.exp (logθ 1))
          + (if i = j then Spec.nuggetFn (logθ 3) else 0)
          + Real.exp (logθ 2) * Spec.dotFn (Spec.maskColsFn X wlin i) (Spec.maskColsFn X wlin j))).PosDef := by
  have hscale : (0 : ℝ) ≤ Real.exp (logθ 2) := (Real.exp_pos _).le
  have hδ : (0 : ℝ) < Spec.nuggetFn (logθ 3) := add_pos (Real.exp_pos _) (Real.exp_pos _)
  exact unaryKernelBuild_posDef _ hR X wlin hscale hδ

end Spec.Factorization
