import VersoManual
import VersoBlueprint

open Verso.Genre Manual

#doc (Manual) "The KernelFlows Kernel-Matrix Build" =>
%%%
tag := "kernelflows-kernel-matrix"
%%%

The factorization chapter built the verified linear-algebra core — symmetric eigendecomposition,
Cholesky, the SPD ridge solve — and discharged the *positive-semidefinite* hypothesis for the CHD mode
kernels, which assemble `K` from *feature inner products*.
[KernelFlows](https://github.com/SandiaGPMethods/KernelFlows.jl) assembles `K` a different way: a
*radial* kernel evaluated on the **pairwise Euclidean distance matrix**, plus a diagonal *nugget* and a
*linear* Gram term. This chapter ports that build, the first step (S1) of the KernelFlows foundation.

The target is KernelFlows.jl `kernel_matrix(k::UnaryKernel, logθ, X)`
(`src/kernel_matrices.jl`). For log-hyperparameters `logθ = (logθ₁, logθ₂, logθ₃, logθ₄)` and data `X`,

$$`K[i,j] = k\!\left(\lVert X_i - X_j\rVert;\, a, b\right) \;+\; \delta\,[i=j] \;+\; e^{\log\theta_3}\,\langle \Phi_i, \Phi_j\rangle,`

with amplitude `a = e^{logθ₁}`, length scale `b = e^{logθ₂}`, nugget `δ = e^{−12} + e^{logθ₄}`, and `Φ`
the column-selected data of the linear term. The radial kernel `k` is Matérn-3/2,
`k(d; a, b) = a·(1 + h)·e^{−h}` with `h = √3·d/b` (KernelFlows.jl `Matern32`); the Matérn-5/2 sibling is
provided too. The whole build is a `Context`-polymorphic spec composed from the landed primitives
(`dotFn`, `normFn`, `maskColsFn`, `ofMatFn`) in
[`NN.Spec.Core.Tensor.KernelMatrix`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/Core/Tensor/KernelMatrix.lean),
so the *same* definition runs over `Float` (the executable golden-tile check) and over `ℝ` (the proofs).

# What this step proves, and what it defers

S1 is the *build spec*, not the positive-definiteness proof. The cheap, algorithm-independent facts the
build rests on are proved over `ℝ` in
[`NN.Proofs.Tensor.Basic.FactorizationsKernelMatrix`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Tensor/Basic/FactorizationsKernelMatrix.lean):

- *The distance matrix is symmetric with a vanishing diagonal* — `pairwiseEuclideanFn_symm` and
  `pairwiseEuclideanFn_self` (`‖Xᵢ − Xᵢ‖ = 0`).
- *The Matérn kernel has a flat top* — `matern32Fn_zero` gives `Matérn32(0; a, b) = a`: the `√3·d/b`
  exponent vanishes at `d = 0`. So on the diagonal the radial term contributes *exactly* the amplitude
  `a`, and it is the nugget `δ` that lifts the diagonal off that plateau. This is the structural reason
  the nugget restores strict positive-definiteness — the keystone the next step (S2) discharges.
- *The assembled kernel is symmetric* — `kernelMatrixMatern32Fn_symm` (and the tensor-level
  `kernelMatrixMatern32Spec_symm`): each of the three summands is symmetric, so `K = Kᵀ`. Symmetry is
  the `IsHermitian` half of the `PosSemidef` hypothesis every downstream solve and `find_gamma` theorem
  assumes; S2 supplies positive-semidefiniteness, and bit-for-bit runtime parity is the Year-2
  deliverable S8.

# A golden tile from the reference implementation

The executable example
[`NN.Examples.Factorization.KernelMatrix`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Factorization/KernelMatrix.lean)
checks the spec against a *golden tile* produced by the **verbatim** KernelFlows.jl source formulas
(`pairwise_Euclidean`, `Matern32`, `kernel_matrix(::UnaryKernel, …)`) on a fixed `4 × 2` dataset, mask
`wlin = [1,0]` (the `nXlinear = 1` slice), and `logθ = [0, 0.5, −1, −3]`. The Lean spec reproduces the
tile to machine precision (`tol = 10⁻⁶`).

One honest gap is visible at the distance diagonal. The Julia `pairwise_Euclidean` adds a `5·eps`
Zygote-stabilization shift under the `√`, so its diagonal reads `≈ 3.3·10⁻⁸` instead of `0`; the clean
spec reads `0`. Because the Matérn kernel is flat at `d = 0` (`matern32Fn_zero`, with vanishing first
derivative), this `~10⁻⁸` perturbation moves the kernel matrix by `~10⁻¹⁵` — far under tolerance. The
example records this explicitly rather than hiding it, and scopes bit-for-bit runtime parity to S8.

The example pairs each positive check with a *negative control*: a wrong length scale drifts the build
far from the tile, a scrambled matrix is caught as non-symmetric, and removing the nugget shifts the
diagonal by `δ ≈ 0.05` — so the metrics that report `≈ 0` on the correct build report a large error when
a piece of the formula is wrong, and the checks are not vacuous.
