/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Factorizations

/-!
# KernelFlows cross-validation losses (`ρ_KF`, `ρ_LOI`, `ρ_LOO`, `ρ_MLE`) — evaluation spec

KernelFlows learns a kernel by minimizing a cross-validation loss `ρ(logθ)` by gradient descent
([`loss_functions.jl`](../../../../../KernelFlows.jl/src/loss_functions.jl)). All four production losses
are built on the kernel matrix `Ω = kernel_matrix(k, logθ, X)` (the SPD build verified in S1/S2) and one
arithmetic primitive — the **regularized quadratic form** `yᵀ Ω⁻¹ y` — plus, for `ρ_MLE`, a Cholesky
log-determinant. This file ports the *evaluation* of those losses as `Context`-polymorphic specs built
from the landed verified pieces (`choleskyFn`, `triSolveLowerFn`, `cholSolveFn`), so the same definition
runs over `Float` (executable) and `ℝ` (proof).

The Julia losses (verbatim, `loss_functions.jl`):

* `ρ_LOI = 1 − (yᵀy / Ω₁₁ / n) / (yᵀ Ω⁻¹ y)` — leave-one-in, averaged over singletons;
* `ρ_KF  = 1 − (y_cᵀ Ω_c⁻¹ y_c) / (yᵀ Ω⁻¹ y)`, `Ω_c` the leading `Nc = n÷2` block;
* `ρ_LOO = N − (yᵀ M y) / (yᵀ Ω⁻¹ y)`, `M = N·Ω⁻¹ − Σ_l (Ω⁻¹ e_l)(Ω⁻¹ e_l)ᵀ / Ω⁻¹[l,l]`;
* `ρ_MLE = ½‖L⁻¹y‖² − Σ log diag(L⁻¹)`, `L` the Cholesky factor (`= ½ yᵀΩ⁻¹y + ½ log det Ω`).

*Scope (this step, S3).* This is the **evaluation spec** and its well-posedness. The shared denominator
`yᵀ Ω⁻¹ y` is `quadInvFn`; on an SPD `Ω` it is the genuine inverse quadratic form, nonnegative, and
*strictly positive* for `y ≠ 0` — so every `ρ` is well-defined (no division by zero), `ρ_KF`/`ρ_LOI` are
bounded above by `1`, and `ρ_MLE`'s data term is the GP negative-log-marginal-likelihood quadratic form.
The companion proofs are in
[`NN.Proofs.Tensor.Basic.FactorizationsKernelLoss`](../../../Proofs/Tensor/Basic/FactorizationsKernelLoss.lean).
Differentiation of `ρ` (the KernelFlows training gradient) is the next step (S6/S7); bit-for-bit parity
against the KernelFlows.jl runtime is the Year-2 deliverable (S8).
-/

@[expose] public section

namespace Spec

variable {α : Type} [Context α]
variable {n : Nat}

/-! ## The shared regularized quadratic form `yᵀ Ω⁻¹ y` -/

/-- The regularized quadratic form `yᵀ Ω⁻¹ y`, computed via the landed Cholesky solve of the SPD `Ω`
(`cholSolveFn (choleskyFn Ω) y` solves `Ω·x = y`, so `dotFn y x = yᵀ Ω⁻¹ y`). This is the single
numerical primitive under every KernelFlows loss — `inv(Symmetric(Ω))` quadratic forms. -/
def quadInvFn (Ω : Fin n → Fin n → α) (y : Fin n → α) : α :=
  dotFn y (cholSolveFn (choleskyFn Ω) y)

/-- Entry `[i,j]` of `Ω⁻¹`, recovered column-by-column by the Cholesky solve: column `j` is
`Ω⁻¹ e_j = cholSolveFn (choleskyFn Ω) e_j`. Used to assemble the `ρ_LOO` leave-one-out operator without
ever forming an explicit inverse. -/
def invCholFn (Ω : Fin n → Fin n → α) : Fin n → Fin n → α :=
  fun i j => cholSolveFn (choleskyFn Ω) (fun k => if k = j then 1 else 0) i

/-- The bilinear form `yᵀ M y = Σ_{a,b} y_a · M[a,b] · y_b`. -/
def bilinFn (M : Fin n → Fin n → α) (y : Fin n → α) : α :=
  dotFn y (fun a => dotFn (M a) y)

/-! ## `ρ_MLE` — maximum-likelihood loss -/

/-- **KernelFlows `ρ_MLE`** (`loss_functions.jl`): with `L` the Cholesky factor of `Ω` and
`z = L⁻¹ y` (forward substitution `triSolveLowerFn`), `ρ_MLE = ½‖z‖² − Σ_i log(1/L[i,i])`. The data
term `½‖z‖²` equals `½ yᵀΩ⁻¹y`, and `−Σ log(1/L[i,i]) = Σ log L[i,i] = ½ log det Ω`, so this is the
Gaussian-process negative log marginal likelihood (up to the constant `½ n log 2π` KernelFlows drops). -/
def rhoMLEFn (Ω : Fin n → Fin n → α) (y : Fin n → α) : α :=
  let L := choleskyFn Ω
  let z := triSolveLowerFn L y
  Numbers.pointfive * dotFn z z
    - (List.finRange n).foldl (fun s i => s + MathFunctions.log (1 / L i i)) 0

/-! ## `ρ_LOI` — leave-one-in loss -/

/-- **KernelFlows `ρ_LOI`** (`loss_functions.jl`): `1 − (yᵀy / Ω₀₀ / n) / (yᵀ Ω⁻¹ y)`, the
leave-one-in CV loss averaged over all singleton centers (the `n = m+1` samples give index `0` a
meaning, so `Ω 0 0` is the first diagonal entry). -/
def rhoLOIFn {m : Nat} (Ω : Fin (m + 1) → Fin (m + 1) → α) (y : Fin (m + 1) → α) : α :=
  1 - (dotFn y y / Ω 0 0 / ((m + 1 : Nat) : α)) / quadInvFn Ω y

/-! ## `ρ_KF` — Kernel-Flows leave-half-out loss -/

/-- **KernelFlows `ρ_KF`** (`loss_functions.jl`): `1 − (y_cᵀ Ω_c⁻¹ y_c) / (yᵀ Ω⁻¹ y)`, where the
center block `Ω_c`, `y_c` is selected by an embedding `e : Fin nc → Fin n` of the center indices into
the sample indices. KernelFlows uses the leading half `e = Fin.castLE` with `nc = n ÷ 2`; stating it for
a general injective `e` makes the well-posedness proof block-agnostic (a principal submatrix of an SPD
matrix is SPD). -/
def rhoKFFn {n nc : Nat} (Ω : Fin n → Fin n → α) (y : Fin n → α) (e : Fin nc → Fin n) : α :=
  1 - quadInvFn (fun i j => Ω (e i) (e j)) (fun i => y (e i)) / quadInvFn Ω y

/-! ## `ρ_LOO` — full leave-one-out loss -/

/-- The KernelFlows leave-one-out operator `M = N·W − Σ_l (W e_l)(W e_l)ᵀ / W[l,l]` evaluated entrywise,
given `W = Ω⁻¹`: `M[a,b] = N·W[a,b] − Σ_l W[a,l]·W[b,l] / W[l,l]` (`loss_functions.jl`, the `M` update
loop). -/
def looMFn (W : Fin n → Fin n → α) : Fin n → Fin n → α :=
  fun a b => ((n : Nat) : α) * W a b
    - (List.finRange n).foldl (fun s l => s + W a l * W b l / W l l) 0

/-- **KernelFlows `ρ_LOO`** (`loss_functions.jl`): `N − (yᵀ M y) / (yᵀ Ω⁻¹ y)`, the full leave-one-out
CV loss, with `M` the leave-one-out operator (`looMFn`) of `Ω⁻¹` (`invCholFn`). -/
def rhoLOOFn (Ω : Fin n → Fin n → α) (y : Fin n → α) : α :=
  ((n : Nat) : α) - bilinFn (looMFn (invCholFn Ω)) y / quadInvFn Ω y

/-! ## Tensor-level wrappers -/

/-- Tensor-level `yᵀ Ω⁻¹ y`. -/
def quadInvSpec (Ω : Tensor α (.dim n (.dim n .scalar))) (y : Tensor α (.dim n .scalar)) : α :=
  quadInvFn (toMatFn Ω) (toVecFn y)

/-- Tensor-level KernelFlows `ρ_MLE`. -/
def rhoMLESpec (Ω : Tensor α (.dim n (.dim n .scalar))) (y : Tensor α (.dim n .scalar)) : α :=
  rhoMLEFn (toMatFn Ω) (toVecFn y)

/-- Tensor-level KernelFlows `ρ_LOI`. -/
def rhoLOISpec {m : Nat} (Ω : Tensor α (.dim (m + 1) (.dim (m + 1) .scalar)))
    (y : Tensor α (.dim (m + 1) .scalar)) : α :=
  rhoLOIFn (toMatFn Ω) (toVecFn y)

/-- Tensor-level KernelFlows `ρ_LOO`. -/
def rhoLOOSpec (Ω : Tensor α (.dim n (.dim n .scalar))) (y : Tensor α (.dim n .scalar)) : α :=
  rhoLOOFn (toMatFn Ω) (toVecFn y)

end Spec
