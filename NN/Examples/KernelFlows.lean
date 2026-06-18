/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.KernelFlows.Parity.Common
public import NN.Examples.KernelFlows.Parity.Deterministic
public import NN.Examples.KernelFlows.Parity.Stochastic

/-!
# KernelFlows / CHD parity harness (S8)

The Year-2 parity deliverable: the verified S1–S7 KernelFlows / Computational-Hypergraph-Discovery
specs, exercised **end-to-end on one shared fixture** against golden tiles from the reference
implementations (KernelFlows.jl `kernel_matrices.jl` / `loss_functions.jl` /
`kernel_functions_analytic.jl` / `optimizers.jl` / `conditional_variance.jl`, and the CHD `Z_test`).

* [`Parity.Common`](KernelFlows/Parity/Common.lean) — the shared fixture (the verbatim KernelFlows.jl
  S1 kernel tile, a label, a test vector) and the parity assertion helpers.
* [`Parity.Deterministic`](KernelFlows/Parity/Deterministic.lean) — the deterministic float-path:
  `kernel → losses → gradients → optimizer step → conditional variance`, every stage matched to the
  golden to `1e-8`, with the optimizer step driven by the *actual* `∇ρ_MLE` so the harness verifies the
  pipeline's *composition*.
* [`Parity.Stochastic`](KernelFlows/Parity/Stochastic.lean) — the CHD `Z_test`, matched at the
  **decision** level: the recovered edge is RNG-invariant across two independent null tiles, with the
  deterministic thresholds also matching the golden to `1e-7`.

Each file lands positive parity checks and negative controls; all are `#eval`'d over `Float`,
sorry/admit/omega-free.
-/
