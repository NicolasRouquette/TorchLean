/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Factorization.Common
public import NN.Spec.Core.Tensor.KernelMatrix
public import NN.Spec.Core.Tensor.KernelLoss
meta import NN.Examples.Factorization.Common
meta import NN.Spec.Core.Tensor.KernelLoss

/-!
# KernelFlows losses are well-posed on the SPD build (S3 examples)

S3 proves the four KernelFlows cross-validation losses (`ρ_KF`, `ρ_LOI`, `ρ_LOO`, `ρ_MLE`) are
well-posed on an SPD kernel `Ω`: their shared denominator `yᵀ Ω⁻¹ y` is strictly positive
(`quadInvFn_pos`), `ρ_KF`/`ρ_LOI ≤ 1` (`rhoKFFn_le_one`/`rhoLOIFn_le_one`), and `ρ_MLE`'s data term is
exactly `½ yᵀΩ⁻¹y` (`rhoMLE_data_eq_quadInv`). These executable checks witness all of that on the
verified RBF (`spherical_sqexp`) build from S2 — SPD for every `logθ`.

* **Positive** — on the SPD RBF kernel: the quadratic form `yᵀΩ⁻¹y > 0`; the `ρ_MLE` data term
  `½‖L⁻¹y‖²` matches `½ yᵀΩ⁻¹y` (the source's `a1 == a2`); each loss is a finite number with
  `ρ_KF, ρ_LOI ≤ 1`.
* **Negative** — on an *indefinite* symmetric matrix the Cholesky takes `√(negative)`, the quadratic
  form is `NaN`, and the losses are `NaN`: SPD-ness (the S2 nugget) is exactly what the losses need.
-/

@[expose] public section

namespace NN.Examples.Factorization.KernelLoss

open NN.Examples.Factorization

/-- Length-`n` `Float` vector tensor from a list (missing entries `0`). -/
def mkVec {n : Nat} (xs : List Float) : Spec.Tensor Float (.dim n .scalar) :=
  Spec.ofVecFn (fun i => xs.getD i.val 0.0)

/-- A 4 × 2 data matrix (4 samples, 2 features). -/
def X : Spec.Tensor Float (.dim 4 (.dim 2 .scalar)) :=
  mkMat [[1, 0], [0, 1], [1, 1], [2, 1]]

/-- Linear-term column mask `wlin = [1, 0]`. -/
def wlin : Spec.Tensor Float (.dim 2 .scalar) := mkVec [1, 0]

/-- Log-hyperparameters `logθ = (0, 0.5, −1, −3)`. -/
def logθ : Spec.Tensor Float (.dim 4 .scalar) := mkVec [0.0, 0.5, -1.0, -3.0]

/-- The SPD KernelFlows **RBF** kernel matrix `Ω = K(logθ)` (4×4) — SPD for every `logθ` (S2). -/
def Ω : Spec.Tensor Float (.dim 4 (.dim 4 .scalar)) := Spec.kernelMatrixSqexpSpec X wlin logθ

/-- A nonzero response vector `y`. -/
def y : Spec.Tensor Float (.dim 4 .scalar) := mkVec [1.0, -0.5, 2.0, 0.3]

/-! ### The shared quadratic form and the four losses -/

/-- The shared regularized quadratic form `q = yᵀ Ω⁻¹ y` (every loss denominator). -/
def q : Float := Spec.quadInvSpec Ω y

/-- `ρ_MLE` data term `½‖L⁻¹y‖²` (forward-substitution `z = L⁻¹y`). -/
def mleData : Float :=
  let L := Spec.choleskyFn (Spec.toMatFn Ω)
  let z := Spec.triSolveLowerFn L (Spec.toVecFn y)
  0.5 * Spec.dotFn z z

/-- `Σ log(1/L[i,i])` — the `ρ_MLE` log-determinant term (`l1` in `loss_functions.jl`). -/
def mleLogTerm : Float :=
  let L := Spec.choleskyFn (Spec.toMatFn Ω)
  (List.finRange 4).foldl (fun s i => s + Float.log (1 / L i i)) 0.0

/-- The leading-half center embedding `Fin 2 ↪ Fin 4` (`Nc = n ÷ 2 = 2`), KernelFlows' `ρ_KF` block. -/
def eCenter : Fin 2 → Fin 4 := Fin.castLE (by grind)

def rhoMLE : Float := Spec.rhoMLESpec Ω y
def rhoLOI : Float := Spec.rhoLOISpec Ω y
def rhoLOO : Float := Spec.rhoLOOSpec Ω y
def rhoKF  : Float := Spec.rhoKFFn (Spec.toMatFn Ω) (Spec.toVecFn y) eCenter

#eval IO.println s!"Ω = RBF SPD build; y = {vecToList y}"
#eval IO.println s!"quadratic form  q = yᵀΩ⁻¹y = {q}"
#eval IO.println s!"ρ_MLE = {rhoMLE}   ρ_LOI = {rhoLOI}   ρ_KF = {rhoKF}   ρ_LOO = {rhoLOO}"

/-! ### Positive checks -/

-- The shared denominator is strictly positive ⟹ every loss is well-defined (`quadInvFn_pos`).
#eval assertLt "shared quadratic form yᵀΩ⁻¹y > 0 (every ρ denominator nonzero)"
  (if q > 0.0 then 0.0 else 1.0)

-- `ρ_MLE`'s data term equals `½ yᵀΩ⁻¹y` — the source's `a1 == a2` (`rhoMLE_data_eq_quadInv`).
#eval assertApproxEq "ρ_MLE data term ½‖L⁻¹y‖² = ½ yᵀΩ⁻¹y (a1 = a2)" mleData (0.5 * q)

-- `ρ_MLE = ½‖L⁻¹y‖² − Σ log(1/L[i,i])` reassembles from the data identity and the log-det term.
#eval assertApproxEq "ρ_MLE = ½ yᵀΩ⁻¹y − Σ log(1/Lᵢᵢ)" rhoMLE (0.5 * q - mleLogTerm)

-- `ρ_LOI ≤ 1` (`rhoLOIFn_le_one`): numerator nonneg, denominator positive.
#eval assertLt "ρ_LOI ≤ 1 (1 − nonneg/positive)" (if rhoLOI > 1.0 then 1.0 else 0.0)

-- `ρ_KF ≤ 1` (`rhoKFFn_le_one`): center-block form nonneg (principal SPD submatrix), denominator pos.
#eval assertLt "ρ_KF ≤ 1 (center block SPD ⟹ nonneg numerator)" (if rhoKF > 1.0 then 1.0 else 0.0)

-- Every loss is a finite number on the SPD build (no NaN/∞).
#eval assertLt "all four ρ are finite on the SPD build"
  (if rhoMLE.isFinite && rhoLOI.isFinite && rhoKF.isFinite && rhoLOO.isFinite then 0.0 else 1.0)

/-! ### Negative control: SPD-ness is necessary

On an *indefinite* symmetric matrix the Cholesky hits `√(negative)`, so the quadratic form `yᵀΩ⁻¹y` is
`NaN` and every loss built on it is `NaN`. The S2 nugget `+δI` is exactly what lifts the kernel into the
SPD cone where the losses are well-defined. -/

/-- A symmetric but **indefinite** matrix (eigenvalues `{3, −1}`): outside the SPD cone. -/
def Mbad : Spec.Tensor Float (.dim 2 (.dim 2 .scalar)) :=
  mkMat [[1, 2],
         [2, 1]]

def yBad : Spec.Tensor Float (.dim 2 .scalar) := mkVec [1.0, 1.0]

/-- The quadratic form on the indefinite matrix — `NaN` (Cholesky took `√(negative)`). -/
def qBad : Float := Spec.quadInvSpec Mbad yBad

#eval IO.println s!"indefinite Mbad: yᵀMbad⁻¹y = {qBad}  (NaN ⟹ no SPD factor); ρ_MLE = {Spec.rhoMLESpec Mbad yBad}"

-- Negative — the quadratic form on a non-SPD matrix is not a usable number (NaN); SPD is required.
#eval assertReconFails "indefinite matrix: yᵀΩ⁻¹y is NaN (SPD necessary for the losses)"
  (Float.abs qBad)

/-- The **same** matrix lifted by a nugget `δ·I` (`δ = 2`): now `[[3,2],[2,3]]`, eigenvalues `{5,1}`,
SPD — the `+δI` lift S2 performs. The quadratic form (and hence every loss) becomes finite again. -/
def Mrescued : Spec.Tensor Float (.dim 2 (.dim 2 .scalar)) :=
  Spec.ofMatFn (Spec.addScaledIdFn (Spec.toMatFn Mbad) 2.0)

def qRescued : Float := Spec.quadInvSpec Mrescued yBad

#eval IO.println s!"nugget rescue Mbad + 2·I (SPD): yᵀΩ⁻¹y = {qRescued}"

-- Positive — the nugget lift restores SPD-ness, so the quadratic form is finite and positive again.
#eval assertLt "nugget rescue: indefinite + δ·I is SPD ⟹ yᵀΩ⁻¹y > 0"
  (if qRescued > 0.0 && qRescued.isFinite then 0.0 else 1.0)

end NN.Examples.Factorization.KernelLoss
