/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.KernelMatrix
public import NN.Proofs.Tensor.Basic.FactorizationsOrthonormal

/-!
# Exact facts about the KernelFlows kernel-matrix build (S1)

The cheap, algorithm-independent guarantees that the KernelFlows unary kernel-matrix build
([`NN.Spec.Core.Tensor.KernelMatrix`](../../../Spec/Core/Tensor/KernelMatrix.lean)) rests on, proved
over `ℝ`:

* `pairwiseEuclideanFn_symm` / `pairwiseEuclideanFn_self` — the distance matrix is symmetric and its
  diagonal vanishes (`‖Xᵢ − Xᵢ‖ = 0`).
* `matern32Fn_zero` — the Matérn-3/2 kernel has a flat top: `Matérn32(0; a, b) = a`. So on the diagonal
  the radial term contributes exactly the amplitude `a`, and it is the *nugget* `δ` that lifts the
  diagonal off that plateau — the structural reason the nugget restores strict positive-definiteness
  (the SPD keystone S2 will discharge).
* `kernelMatrixMatern32Fn_symm` / `kernelMatrixMatern32Spec_symm` — the assembled kernel matrix is
  symmetric (each of the three summands is), at both the function and tensor level. Symmetry is the
  `IsHermitian` half of the `PosSemidef` hypothesis every downstream solve / `find_gamma` theorem
  assumes; S2 supplies positive-semidefiniteness.

These do **not** need any algebra laws beyond `ℝ`, so they land independently of the (harder) PSD proof.
-/

@[expose] public section

namespace Spec.Factorization

open Spec.Factorization.Reconstruction

variable {n d : Nat}

/-- The pairwise Euclidean distance is symmetric: `‖Xᵢ − Xⱼ‖ = ‖Xⱼ − Xᵢ‖`. -/
theorem pairwiseEuclideanFn_symm (X : Fin n → Fin d → ℝ) (i j : Fin n) :
    Spec.pairwiseEuclideanFn X i j = Spec.pairwiseEuclideanFn X j i := by
  unfold Spec.pairwiseEuclideanFn Spec.normFn
  congr 1
  rw [dotFn_eq_sum, dotFn_eq_sum]
  exact Finset.sum_congr rfl (fun k _ => by ring)

/-- The distance diagonal vanishes: `‖Xᵢ − Xᵢ‖ = 0`. -/
theorem pairwiseEuclideanFn_self (X : Fin n → Fin d → ℝ) (i : Fin n) :
    Spec.pairwiseEuclideanFn X i i = 0 := by
  unfold Spec.pairwiseEuclideanFn Spec.normFn
  have h : Spec.dotFn (fun k => X i k - X i k) (fun k => X i k - X i k) = (0 : ℝ) := by
    rw [dotFn_eq_sum]
    exact Finset.sum_eq_zero (fun k _ => by ring)
  rw [h]; exact Real.sqrt_zero

/-- **The Matérn-3/2 kernel has a flat top:** `Matérn32(0; a, b) = a`. The `√3·d/b` exponent vanishes at
`d = 0`, leaving `a·(1 + 0)·exp(0) = a`. -/
theorem matern32Fn_zero (a b : ℝ) : Spec.matern32Fn 0 a b = a := by
  unfold Spec.matern32Fn
  rw [show (MathFunctions.sqrt (Numbers.three : ℝ) * 0 / b) = 0 by rw [mul_zero, zero_div]]
  show a * (1 + 0) * Real.exp (-0) = a
  rw [neg_zero, Real.exp_zero, add_zero, mul_one, mul_one]

/-- **The KernelFlows kernel matrix is symmetric:** `K[i,j] = K[j,i]`. Each summand is symmetric — the
radial term through `pairwiseEuclideanFn_symm`, the nugget through `i = j ↔ j = i`, and the linear Gram
through symmetry of the dot product. -/
theorem kernelMatrixMatern32Fn_symm (X : Fin n → Fin d → ℝ) (wlin : Fin d → ℝ) (logθ : Fin 4 → ℝ)
    (i j : Fin n) :
    Spec.kernelMatrixMatern32Fn X wlin logθ i j = Spec.kernelMatrixMatern32Fn X wlin logθ j i := by
  unfold Spec.kernelMatrixMatern32Fn
  rw [pairwiseEuclideanFn_symm X i j]
  congr 1
  · congr 1
    by_cases h : i = j
    · rw [if_pos h, if_pos h.symm]
    · rw [if_neg h, if_neg (fun hji => h hji.symm)]
  · congr 1
    rw [dotFn_eq_sum, dotFn_eq_sum]
    exact Finset.sum_congr rfl (fun k _ => by simp only [Spec.maskColsFn]; ring)

/-- Tensor-level: the KernelFlows kernel matrix is symmetric (the `IsHermitian` half of the standing
`PosSemidef` hypothesis the solve consumes). -/
theorem kernelMatrixMatern32Spec_symm (X : Spec.Tensor ℝ (.dim n (.dim d .scalar)))
    (wlin : Spec.Tensor ℝ (.dim d .scalar)) (logθ : Spec.Tensor ℝ (.dim 4 .scalar)) (i j : Fin n) :
    Spec.get2 (Spec.kernelMatrixMatern32Spec X wlin logθ) i j
      = Spec.get2 (Spec.kernelMatrixMatern32Spec X wlin logθ) j i := by
  show Spec.kernelMatrixMatern32Fn (Spec.toMatFn X) (Spec.toVecFn wlin) (Spec.toVecFn logθ) i j
      = Spec.kernelMatrixMatern32Fn (Spec.toMatFn X) (Spec.toVecFn wlin) (Spec.toVecFn logθ) j i
  exact kernelMatrixMatern32Fn_symm _ _ _ i j

end Spec.Factorization
