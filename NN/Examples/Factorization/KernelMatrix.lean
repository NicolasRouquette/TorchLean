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
# KernelFlows kernel-matrix build examples (S1)

Executable checks for the KernelFlows unary kernel-matrix build
(`NN.Spec.Core.Tensor.KernelMatrix`): the pairwise Euclidean distance matrix and the Matérn-3/2 kernel
`K(logθ) = Matérn32(‖Xᵢ − Xⱼ‖) + δ·[i=j] + exp(logθ₃)·⟨Φᵢ,Φⱼ⟩`.

The **golden tile** `Kgold` / `Dgold` was produced by running the *verbatim* KernelFlows.jl source
formulas (`pairwise_Euclidean`, `Matern32`, `kernel_matrix(::UnaryKernel, …)` from
`src/kernel_matrices.jl` + `src/kernel_functions_unary.jl`) on the data `X`, mask `wlin = [1,0]`
(i.e. `nXlinear = 1`), and `logθ = [0, 0.5, −1, −3]`. The Lean spec reproduces it to machine precision
(`tol = 1e-6`), the cross-language anchor for the formalization. (Julia adds a `5·eps`
Zygote-stabilization shift under the distance `√`, so its diagonal reads `≈ 3.3·10⁻⁸` instead of `0`;
the Matérn flat top at `d = 0` makes the kernel-matrix effect `~10⁻¹⁵` — both stay far under `tol`.
Bit-for-bit runtime parity is the Year-2 deliverable S8.)

Both **positive** checks (the spec matches the golden tile / is symmetric / has the predicted diagonal)
and **negative controls** (wrong hyperparameters drift far from the tile; a scrambled matrix is caught
as non-symmetric; dropping the nugget shifts the diagonal) are included, so the checks are not vacuous.
-/

@[expose] public section

namespace NN.Examples.Factorization.KernelMatrix

open NN.Examples.Factorization

/-- Length-`n` `Float` vector tensor from a list (missing entries `0`). -/
def mkVec {n : Nat} (xs : List Float) : Spec.Tensor Float (.dim n .scalar) :=
  Spec.ofVecFn (fun i => xs.getD i.val 0.0)

/-- A 4 × 2 data matrix (4 samples, 2 features). -/
def X : Spec.Tensor Float (.dim 4 (.dim 2 .scalar)) :=
  mkMat [[1, 0], [0, 1], [1, 1], [2, 1]]

/-- Linear-term column mask `wlin = [1,0]` — selects the first feature only (KernelFlows `nXlinear=1`). -/
def wlin : Spec.Tensor Float (.dim 2 .scalar) := mkVec [1, 0]

/-- Log-hyperparameters `logθ = (logθ₁, logθ₂, logθ₃, logθ₄) = (0, 0.5, −1, −3)`:
amplitude `a = 1`, length scale `b = e^{0.5}`, linear weight `e^{−1}`, ridge `e^{−3}`. -/
def logθ : Spec.Tensor Float (.dim 4 .scalar) := mkVec [0.0, 0.5, -1.0, -3.0]

/-- The executable KernelFlows Matérn-3/2 kernel matrix `K(logθ)` (4×4). -/
def K : Spec.Tensor Float (.dim 4 (.dim 4 .scalar)) := Spec.kernelMatrixMatern32Spec X wlin logθ

/-- The pairwise Euclidean distance matrix `D[i,j] = ‖Xᵢ − Xⱼ‖` (4×4). -/
def D : Spec.Tensor Float (.dim 4 (.dim 4 .scalar)) := Spec.pairwiseEuclideanSpec X

/-- **Golden distance tile** from KernelFlows.jl `pairwise_Euclidean` (diagonal `≈ 3.3·10⁻⁸` is the
`5·eps` stabilization shift; the clean spec reads `0` there — difference well under `tol`). -/
def Dgold : Spec.Tensor Float (.dim 4 (.dim 4 .scalar)) :=
  mkMat [[3.3320009373125282e-08, 1.4142135623730954, 1.0000000000000004, 1.4142135623730954],
         [1.4142135623730954, 3.3320009373125282e-08, 1.0000000000000004, 2.0],
         [1.0000000000000004, 1.0000000000000004, 3.3320009373125282e-08, 1.0000000000000004],
         [1.4142135623730954, 2.0, 1.0000000000000004, 3.3320009373125282e-08]]

/-- **Golden kernel tile** from KernelFlows.jl `kernel_matrix(::UnaryKernel{Matern32}, logθ, X)`. -/
def Kgold : Spec.Tensor Float (.dim 4 (.dim 4 .scalar)) :=
  mkMat [[1.4176726537516589, 0.5626260453760491, 1.0850527096273428, 1.2983849277189337],
         [0.5626260453760491, 1.0497932125802165, 0.71717326845590046, 0.37933628856076407],
         [1.0850527096273428, 0.71717326845590046, 1.4176726537516589, 1.4529321507987851],
         [1.2983849277189337, 0.37933628856076407, 1.4529321507987851, 2.5213109772659861]]

#eval IO.println s!"pairwise Euclidean D =\n{(List.finRange 4).map (fun i =>
  (List.finRange 4).map (fun j => Spec.get2 D i j))}"
#eval IO.println s!"KernelFlows kernel K(logθ) =\n{(List.finRange 4).map (fun i =>
  (List.finRange 4).map (fun j => Spec.get2 K i j))}"

/-! ### Positive checks -/

#eval assertLt "pairwise Euclidean matches KernelFlows golden tile" (maxMatErr D Dgold)
#eval assertLt "kernel matrix K(logθ) matches KernelFlows golden tile" (maxMatErr K Kgold)
#eval assertLt "kernel matrix is symmetric: K = Kᵀ" (maxMatErr K (tr K))
#eval assertLt "distance matrix is symmetric: D = Dᵀ" (maxMatErr D (tr D))

/-- The Matérn flat top + nugget identity: on the diagonal, `K[i,i] = a + δ + exp(logθ₃)·⟨Φᵢ,Φᵢ⟩`
(distance `0`, `Matérn32(0) = a`). Here `a = 1`, `δ = exp(−12) + exp(−3)`. -/
def Kdiag_ref : Spec.Tensor Float (.dim 4 .scalar) :=
  Spec.ofVecFn (fun i =>
    1.0 + (Float.exp (-12.0) + Float.exp (-3.0))
      + Float.exp (-1.0) * (Spec.get2 X i 0 * Spec.get2 X i 0))
#eval assertLt "diagonal equals a + nugget + linear‖Φᵢ‖² (Matérn flat top)"
  ((List.finRange 4).foldl (fun acc i => max acc (Float.abs (Spec.get2 K i i -
    Spec.Tensor.toScalar (Spec.get Kdiag_ref i)))) 0.0)

/-! ### Negative controls -/

/-- Same build at a **different length scale** `logθ₂ = 1.5` — should drift far from the golden tile. -/
def Kwrong : Spec.Tensor Float (.dim 4 (.dim 4 .scalar)) :=
  Spec.kernelMatrixMatern32Spec X wlin (mkVec [0.0, 1.5, -1.0, -3.0])
#eval assertGe "wrong length scale drifts from golden tile" (maxMatErr Kwrong Kgold) 0.1

/-- A **scrambled** matrix (golden tile with one off-diagonal entry overwritten) is *not* symmetric, so
the symmetry metric has teeth. -/
def Kscram : Spec.Tensor Float (.dim 4 (.dim 4 .scalar)) :=
  Spec.ofMatFn (fun i j => if i.val == 0 ∧ j.val == 1 then 9.0 else Spec.get2 Kgold i j)
#eval assertGe "scrambled kernel is correctly caught as non-symmetric" (maxMatErr Kscram (tr Kscram)) 1.0

/-- Dropping the nugget shifts the diagonal by `δ = exp(−12) + exp(−3) ≈ 0.0498`, detectable on `K`. -/
def Knonug : Spec.Tensor Float (.dim 4 (.dim 4 .scalar)) :=
  Spec.ofMatFn (fun i j => if i.val == j.val
    then Spec.get2 Kgold i j - (Float.exp (-12.0) + Float.exp (-3.0)) else Spec.get2 Kgold i j)
#eval assertGe "removing the nugget shifts the diagonal off the golden tile" (maxMatErr Knonug Kgold) 0.01

end NN.Examples.Factorization.KernelMatrix
