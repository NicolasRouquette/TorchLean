/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Factorizations

/-!
# KernelFlows kernel-matrix build (`pairwise_Euclidean` + Matérn + nugget + linear)

The CHD development built the kernel `K` from *feature inner products* (linear / quadratic / Gaussian
modes, see `Factorizations` and `FactorizationsKernels`). **KernelFlows** builds `K` differently: a
*radial* kernel evaluated on the **pairwise Euclidean distance matrix**, plus a diagonal *nugget* and a
*linear* Gram term. This file ports that build — KernelFlows.jl `kernel_matrix(k::UnaryKernel, logθ, X)`
(`src/kernel_matrices.jl`) — as `Context`-polymorphic specs composed from the landed primitives
(`dotFn`, `normFn`, `maskColsFn`, `ofMatFn`), so the *same* definition runs over `Float` (executable
golden-tile check) and over `ℝ` (proof).

The KernelFlows unary kernel matrix is, for log-hyperparameters `logθ = (logθ₁, logθ₂, logθ₃, logθ₄)`:

`K[i,j] = k(‖Xᵢ − Xⱼ‖; a, b) + δ·[i=j] + exp(logθ₃)·⟨Φᵢ, Φⱼ⟩`,

with amplitude `a = exp(logθ₁)`, length scale `b = exp(logθ₂)`, nugget `δ = exp(−12) + exp(logθ₄)`,
`Φ = maskColsFn X wlin` the (column-selected) data for the linear term, and `k` the Matérn-3/2 radial
kernel `k(d; a, b) = a·(1 + h)·exp(−h)`, `h = √3·d/b` (KernelFlows.jl `Matern32`,
`src/kernel_functions_unary.jl`). The Matérn-5/2 sibling `k(d; a, b) = a·(1 + h + h²/3)·exp(−h)`,
`h = √5·d/b`, is provided too.

*Scope (this step, S1).* This is the **build spec**: the executable `K(logθ)` and the cheap exact facts
it rests on — `K` is symmetric, the distance diagonal vanishes, and `Matérn(0) = a` (so the nugget is
exactly what lifts the diagonal off the radial plateau). The companion proofs are in
[`NN.Proofs.Tensor.Basic.FactorizationsKernelMatrix`](../../../Proofs/Tensor/Basic/FactorizationsKernelMatrix.lean).
Positive-definiteness of the Matérn build (the SPD keystone the ridge solve consumes) is the next step
(S2), and bit-for-bit parity against the KernelFlows.jl runtime is the Year-2 deliverable (S8); the
executable example here checks against a golden tile produced by the verbatim KernelFlows.jl formulas.
-/

@[expose] public section

namespace Spec

variable {α : Type} [Context α]
variable {n d : Nat}

/-! ## Pairwise Euclidean distances -/

/-- The pairwise Euclidean distance `‖Xᵢ − Xⱼ‖ = √(Σ_k (X[i,k] − X[j,k])²)` between samples `i` and `j`.
KernelFlows `pairwise_Euclidean` (`src/kernel_matrices.jl`); we take the mathematically exact distance
(the Julia version adds a `5·eps` Zygote-stabilization shift under the `√`, a ~3·10⁻⁸ perturbation that
the radial kernel's flat top at `d = 0` makes negligible — see the golden-tile example). -/
def pairwiseEuclideanFn (X : Fin n → Fin d → α) : Fin n → Fin n → α :=
  fun i j => normFn (fun k => X i k - X j k)

/-- Tensor-level pairwise Euclidean distance matrix `D[i,j] = ‖Xᵢ − Xⱼ‖`. -/
def pairwiseEuclideanSpec (X : Tensor α (.dim n (.dim d .scalar))) :
    Tensor α (.dim n (.dim n .scalar)) :=
  ofMatFn (pairwiseEuclideanFn (toMatFn X))

/-! ## Radial (Matérn) kernels on a distance -/

/-- Matérn-3/2 radial kernel `k(d; a, b) = a·(1 + h)·exp(−h)`, `h = √3·d/b` (KernelFlows.jl `Matern32`,
`src/kernel_functions_unary.jl`): amplitude `a`, length scale `b`. -/
def matern32Fn (dist a b : α) : α :=
  a * (1 + MathFunctions.sqrt Numbers.three * dist / b)
    * MathFunctions.exp (-(MathFunctions.sqrt Numbers.three * dist / b))

/-- Matérn-5/2 radial kernel `k(d; a, b) = a·(1 + h + h²/3)·exp(−h)`, `h = √5·d/b` (KernelFlows.jl
`Matern52`). -/
def matern52Fn (dist a b : α) : α :=
  let h := MathFunctions.sqrt Numbers.five * dist / b
  a * (1 + h + h * h / Numbers.three) * MathFunctions.exp (-h)

/-- Squared-exponential (RBF) radial kernel `k(d; a, b) = a·exp(−½·d²/b)` (KernelFlows.jl
`spherical_sqexp`, `src/kernel_functions_unary.jl`): amplitude `a`, length scale² `b`. Unlike Matérn it
depends on the **squared** distance `d²`, so — over the Euclidean distance `d = ‖Xᵢ − Xⱼ‖` — it is
`a·exp(−½‖Xᵢ − Xⱼ‖²/b) = a·∏_k exp(−½(X[i,k] − X[j,k])²/b)`, a Hadamard product of one-dimensional
Gaussians. That is exactly what makes its kernel matrix provably positive-semidefinite by the landed
entrywise-exponential / Schur machinery (`posSemidef_gaussianRadial`), *without* the Gaussian
scale-mixture (Bochner/Schoenberg) representation the bare-distance kernels need. -/
def sphericalSqexpFn (dist a b : α) : α :=
  a * MathFunctions.exp (Numbers.neg_point_five * (dist * dist) / b)

/-! ## Nugget and the unary kernel-matrix build -/

/-- The KernelFlows nugget `δ = exp(−12) + exp(logθ₄)` (`src/kernel_matrices.jl`): a fixed floor
`exp(−12)` plus the learned ridge `exp(logθ₄)`, added on the diagonal to make the kernel SPD. -/
def nuggetFn (logθ₄ : α) : α :=
  MathFunctions.exp (-(Numbers.three * Numbers.four)) + MathFunctions.exp logθ₄

/-- **KernelFlows unary kernel-matrix build** `K(logθ)` for the Matérn-3/2 radial kernel
(KernelFlows.jl `kernel_matrix(k::UnaryKernel, logθ, X)`, `src/kernel_matrices.jl`):

`K[i,j] = Matérn32(‖Xᵢ − Xⱼ‖; exp logθ₁, exp logθ₂) + (if i = j then δ else 0) + exp(logθ₃)·⟨Φᵢ, Φⱼ⟩`,

with `δ = nuggetFn logθ₄` and `Φ = maskColsFn X wlin` (the linear term sees the columns selected by
`wlin`, mirroring KernelFlows' `X[:, 1:nXlinear]` slice — a 0/1 `wlin` is exactly that slice). The four
log-hyperparameters are read positionally from `logθ : Fin 4 → α`, matching the Julia `logθ[1..4]`. -/
def kernelMatrixMatern32Fn (X : Fin n → Fin d → α) (wlin : Fin d → α) (logθ : Fin 4 → α) :
    Fin n → Fin n → α :=
  fun i j =>
    matern32Fn (pairwiseEuclideanFn X i j) (MathFunctions.exp (logθ 0)) (MathFunctions.exp (logθ 1))
      + (if i = j then nuggetFn (logθ 3) else 0)
      + MathFunctions.exp (logθ 2) * dotFn (maskColsFn X wlin i) (maskColsFn X wlin j)

/-- Tensor-level KernelFlows unary Matérn-3/2 kernel-matrix build. The form S2's PSD proof and the
ridge solve consume: `kernelMatrixMatern32Spec X wlin logθ` is the executable `K(logθ)`. -/
def kernelMatrixMatern32Spec (X : Tensor α (.dim n (.dim d .scalar)))
    (wlin : Tensor α (.dim d .scalar)) (logθ : Tensor α (.dim 4 .scalar)) :
    Tensor α (.dim n (.dim n .scalar)) :=
  ofMatFn (kernelMatrixMatern32Fn (toMatFn X) (toVecFn wlin) (toVecFn logθ))

/-- **KernelFlows unary kernel-matrix build** `K(logθ)` for the squared-exponential (RBF) radial kernel
(KernelFlows.jl `kernel_matrix(k::UnaryKernel{spherical_sqexp}, logθ, X)`):

`K[i,j] = sqexp(‖Xᵢ − Xⱼ‖; exp logθ₁, exp logθ₂) + (if i = j then δ else 0) + exp(logθ₃)·⟨Φᵢ, Φⱼ⟩`,

the same assembly as the Matérn build but with the RBF radial kernel. Because the RBF is `exp` of the
*squared* distance, this build is provably **symmetric positive-definite** for every `logθ` (the nugget
`δ = nuggetFn logθ₄ > 0` lifts the PSD base) — S2's fully-discharged keystone instance. -/
def kernelMatrixSqexpFn (X : Fin n → Fin d → α) (wlin : Fin d → α) (logθ : Fin 4 → α) :
    Fin n → Fin n → α :=
  fun i j =>
    sphericalSqexpFn (pairwiseEuclideanFn X i j) (MathFunctions.exp (logθ 0)) (MathFunctions.exp (logθ 1))
      + (if i = j then nuggetFn (logθ 3) else 0)
      + MathFunctions.exp (logθ 2) * dotFn (maskColsFn X wlin i) (maskColsFn X wlin j)

/-- Tensor-level KernelFlows unary RBF kernel-matrix build `K(logθ)`. Positive-definite for all `logθ`
(`kernelMatrixSqexpSpec_posDef`), so the Cholesky/ridge-solve development fires unconditionally. -/
def kernelMatrixSqexpSpec (X : Tensor α (.dim n (.dim d .scalar)))
    (wlin : Tensor α (.dim d .scalar)) (logθ : Tensor α (.dim 4 .scalar)) :
    Tensor α (.dim n (.dim n .scalar)) :=
  ofMatFn (kernelMatrixSqexpFn (toMatFn X) (toVecFn wlin) (toVecFn logθ))

end Spec
