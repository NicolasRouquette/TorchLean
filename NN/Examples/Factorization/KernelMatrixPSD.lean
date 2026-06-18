/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Factorization.Common
public import NN.Spec.Core.Tensor.KernelMatrix
meta import NN.Examples.Factorization.Common

/-!
# The KernelFlows build is SPD ⟹ Cholesky fires (S2 examples)

S2 proves the KernelFlows RBF (`spherical_sqexp`) unary kernel-matrix build is positive-definite for
every `logθ` (`kernelMatrixSqexpFn_posDef`), so the positive-pivot keystone fires and `K = L·Lᵀ`
succeeds with strictly positive pivots. These executable checks witness that numerically and show the
keystone's mechanism — the nugget is exactly what turns a merely-PSD (or indefinite) matrix into an
SPD one.

* **Positive** — the RBF build `K(logθ)` reconstructs from its Cholesky factor (`A = L·Lᵀ` to machine
  precision, which is only possible if every pivot `L[j,j] > 0`), and the pivots are printed.
* **Negative** — an indefinite symmetric matrix takes `√(negative)` and the reconstruction is `NaN`
  (no factor exists). Adding a large-enough nugget `δ·I` lifts it to SPD and the factor reappears —
  the same `δ·I` lift `unaryKernelBuild_posDef` uses.
-/

@[expose] public section

namespace NN.Examples.Factorization.KernelMatrixPSD

open NN.Examples.Factorization

/-- Length-`n` `Float` vector tensor from a list (missing entries `0`). -/
def mkVec {n : Nat} (xs : List Float) : Spec.Tensor Float (.dim n .scalar) :=
  Spec.ofVecFn (fun i => xs.getD i.val 0.0)

/-- A 4 × 2 data matrix (4 samples, 2 features). -/
def X : Spec.Tensor Float (.dim 4 (.dim 2 .scalar)) :=
  mkMat [[1, 0], [0, 1], [1, 1], [2, 1]]

/-- Linear-term column mask `wlin = [1, 0]` (KernelFlows `nXlinear = 1`). -/
def wlin : Spec.Tensor Float (.dim 2 .scalar) := mkVec [1, 0]

/-- Log-hyperparameters `logθ = (0, 0.5, −1, −3)`: amplitude `1`, length scale² `e^{0.5}`, linear
weight `e^{−1}`, ridge `e^{−3}`. -/
def logθ : Spec.Tensor Float (.dim 4 .scalar) := mkVec [0.0, 0.5, -1.0, -3.0]

/-- The executable KernelFlows **RBF** kernel matrix `K(logθ)` (4×4) — SPD for every `logθ`. -/
def Ksq : Spec.Tensor Float (.dim 4 (.dim 4 .scalar)) := Spec.kernelMatrixSqexpSpec X wlin logθ

/-- Its Cholesky factor `L` (lower-triangular). The factorization only succeeds (no `√(negative)`)
because `Ksq` is SPD — the content of `kernelMatrixSqexpFn_posDef`. -/
def Lsq : Spec.Tensor Float (.dim 4 (.dim 4 .scalar)) := Spec.choleskySpec Ksq

/-- Reconstruction error `‖K − L·Lᵀ‖_max`. -/
def reconErr : Float := maxMatErr Ksq (mm Lsq (tr Lsq))

/-- Smallest Cholesky pivot `min_j L[j,j]`. Strictly positive ⟺ the keystone fired. -/
def minPivot : Float := (List.finRange 4).foldl (fun acc i => min acc (Spec.get2 Lsq i i)) 1e9

#eval IO.println s!"RBF build K(logθ) =\n{(List.finRange 4).map (fun i =>
  (List.finRange 4).map (fun j => Spec.get2 Ksq i j))}"
#eval IO.println s!"Cholesky pivots L[j,j] = {vecToList (diagOf Lsq)}  (all > 0 ⟹ SPD)"

/-! ### Positive checks -/

#eval assertLt "RBF build is SPD: K = L·Lᵀ reconstructs (so every pivot L[j,j] > 0)" reconErr
-- `minPivot` reported via `assertReconFails`-style negation would be confusing; instead state the
-- positive fact directly: `tol − minPivot < tol` holds iff `minPivot > 0`.
#eval assertLt "RBF build has strictly positive Cholesky pivots (keystone fires)"
  (if minPivot > 0.0 then 0.0 else 1.0)

/-! ### Negative control + nugget rescue -/

/-- A symmetric but **indefinite** matrix (eigenvalues `{3, −1}`): outside the SPD cone, so Cholesky
takes `√(negative)`. This is the situation the nugget exists to fix. -/
def Mbad : Spec.Tensor Float (.dim 2 (.dim 2 .scalar)) :=
  mkMat [[1, 2],
         [2, 1]]

def Lbad : Spec.Tensor Float (.dim 2 (.dim 2 .scalar)) := Spec.choleskySpec Mbad
-- Summed Frobenius error (IEEE `max` ignores `NaN`; the sum propagates the `√(negative)` `NaN`).
def reconErrBad : Float := frobSqErr Mbad (mm Lbad (tr Lbad))

#eval assertReconFails "indefinite matrix: no Cholesky factor (the nugget's reason to exist)" reconErrBad

/-- The **same** matrix lifted by a nugget `δ·I` with `δ = 2` — now `[[3,2],[2,3]]`, eigenvalues
`{5, 1}`, SPD. This is exactly the `+ δ·I` lift in `unaryKernelBuild_posDef`; with it the factor
reappears. -/
def Mrescued : Spec.Tensor Float (.dim 2 (.dim 2 .scalar)) :=
  Spec.ofMatFn (Spec.addScaledIdFn (Spec.toMatFn Mbad) 2.0)

def Lrescued : Spec.Tensor Float (.dim 2 (.dim 2 .scalar)) := Spec.choleskySpec Mrescued
def reconErrRescued : Float := maxMatErr Mrescued (mm Lrescued (tr Lrescued))

#eval assertLt "nugget rescue: indefinite + δ·I is SPD, so K = L·Lᵀ reconstructs again" reconErrRescued

end NN.Examples.Factorization.KernelMatrixPSD
