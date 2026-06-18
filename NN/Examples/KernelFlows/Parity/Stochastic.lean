/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.KernelFlows.Parity.Common
meta import NN.Examples.KernelFlows.Parity.Common

/-!
# KernelFlows parity harness (S8) — the stochastic Z-test, at the decision level

The CHD edge test (`Z_test`, `interpolatory.py`) is **stochastic**: it judges the observed `noise`
against the null distribution of the *same* statistic under `N` random draws (`jax.random.normal`),
keeping the edge when `noise < Z_low` (the 5th percentile of the null). Bit-for-bit parity against a
particular RNG stream is neither possible nor meaningful — what CHD consumes, and what must match the
reference, is the **decision**: the recovered kernel/edge set.

This file checks parity at exactly that level, on the shared fixture's kernel `Ω` ([`Common`](Common.lean)):

* **deterministic backbone** — the eigendecomposition, the observed signal `noise`, and the `Z_low`/
  `Z_high` thresholds of each fixed null tile match the golden to `1e-7` (the Jacobi eig vs the
  reference's LAPACK eig agree because `noise` is a *spectral function of `Ω`*, basis-independent);
* **RNG-invariance** — the recovered edge survives under **two independent** deterministic null tiles
  (different generators standing in for two `jax.random` keys): the verdict is identical, because the
  signal `noise` clears each null's lower tail by a wide **separation margin**;
* **negative controls** — a borderline `noise` *inside* the null band is correctly rejected, and a
  borderline signal between the two tiles' thresholds is shown to be genuinely *not* RNG-invariant —
  so the invariance above is a property of well-separated edges, not an artifact of the test.

All checks run over `Float`. The `noise ∈ [0,1]` bound and the `Z_low ≤ Z_high` ordering these rest on
are proved over `ℝ` in `NN.Proofs.Tensor.Basic.FactorizationsDecision` (S4-era); S8 exercises their
*composition* into a stable edge decision.
-/

@[expose] public section

namespace NN.Examples.KernelFlows.Parity.Stochastic

open NN.Examples.KernelFlows.Parity
open NN.Examples.Factorization (assertGe)

/-! ## Eigendecomposition of the shared kernel and the observed signal -/

/-- Eigenvalues of `Ω` from the verified Jacobi solver (ascending). -/
def evals : Fin 4 → Float := Spec.toVecFn (Spec.symEigJacobiSpec Ωt 12).1
/-- Eigenvectors of `Ω` (columns). -/
def V : Fin 4 → Fin 4 → Float := Spec.toMatFn (Spec.symEigJacobiSpec Ωt 12).2

/-- The Z-test regularization. -/
def γZ : Float := 0.1

/-- The dominant eigen-direction (largest eigenvalue), via the verified `argMaxFn`. -/
def domIdx : Fin 4 := Spec.argMaxFn evals
/-- A "real signal": data aligned with the dominant eigenvector — the kind of edge CHD keeps. -/
def signalGa : Fin 4 → Float := fun i => V i domIdx
/-- The observed `noise` of the signal-aligned data: `γ/(λ_dom+γ)`, the smallest shrinkage. -/
def obsSignal : Float := Spec.varNoiseFn evals γZ (Spec.projFn V signalGa)

/-! ## Two independent null tiles (two `jax.random` keys) -/

/-- Null tile A — `N = 20` deterministic draws in ℝ⁴ (matches `Discovery.lean`'s generator). -/
def tile1 : Fin 20 → Fin 4 → Float :=
  fun j i => (Float.ofNat ((j.val * 31 + i.val * 17 + 7) % 23) - 11.0) / 7.0
/-- Null tile B — a *different* generator (a second `jax.random` key). -/
def tile2 : Fin 20 → Fin 4 → Float :=
  fun j i => (Float.ofNat ((j.val * 13 + i.val * 29 + 3) % 19) - 9.0) / 5.0

def zLow1  : Float := Spec.zLowFn  evals V γZ tile1
def zHigh1 : Float := Spec.zHighFn evals V γZ tile1
def zLow2  : Float := Spec.zLowFn  evals V γZ tile2
def zHigh2 : Float := Spec.zHighFn evals V γZ tile2

/-- The recovered edge verdict under a null tile's lower threshold. -/
def sig1 : Bool := Spec.zSignificantFn obsSignal zLow1
def sig2 : Bool := Spec.zSignificantFn obsSignal zLow2

#eval IO.println s!"obsSignal={obsSignal}  tile1: Zlow={zLow1} Zhigh={zHigh1} sig={sig1}  \
  tile2: Zlow={zLow2} Zhigh={zHigh2} sig={sig2}"

/-! ## Deterministic backbone — the eig-derived statistics match the golden (1e-7) -/

#eval assertParity "observed signal noise γ/(λ_dom+γ)" obsSignal 0.020884213493943281 1e-7
#eval assertParity "Z_low (null tile A)"  zLow1  0.15989024093221382 1e-7
#eval assertParity "Z_high (null tile A)" zHigh1 0.30190820241621869 1e-7
#eval assertParity "Z_low (null tile B)"  zLow2  0.25323164966953415 1e-7
#eval assertParity "Z_high (null tile B)" zHigh2 0.29242630161115857 1e-7

/-! ## Decision-level parity — the recovered edge is RNG-invariant -/

-- Positive — the signal-aligned edge is recovered under null tile A (`noise < Z_low`).
#eval assertDecision "edge recovered under null tile A" sig1 true
-- Positive — the *same* edge is recovered under an independent null tile B.
#eval assertDecision "edge recovered under independent null tile B (RNG re-draw)" sig2 true
-- Positive — the recovered decision is identical across the two RNG streams: decision-level parity.
#eval assertDecision "Z-test edge decision is RNG-invariant (same recovered edge across tiles)"
  (sig1 == sig2) true

-- Positive — the recovered *kernel set* matches too: `MinNoiseKernelChooser` admits the single kernel
-- under both tiles (`some 0`), so the recovered structure is identical.
#eval assertDecision "recovered kernel set is identical across tiles (chooser admits the edge under both)"
  ((Spec.kernelChooserFn (fun _ : Fin 1 => obsSignal) (fun _ : Fin 1 => zLow1)).isSome
    == (Spec.kernelChooserFn (fun _ : Fin 1 => obsSignal) (fun _ : Fin 1 => zLow2)).isSome) true

-- Positive — *why* the decision is RNG-invariant: the signal noise clears each null's lower tail by a
-- wide separation margin (`Z_low − obsSignal`), so a different RNG draw cannot flip the verdict.
#eval assertGe "separation margin under null tile A (Z_low − obsSignal)" (zLow1 - obsSignal) 0.1
#eval assertGe "separation margin under null tile B (Z_low − obsSignal)" (zLow2 - obsSignal) 0.1

/-! ## Negative controls — the decision-level test is not vacuous -/

/-- A borderline `noise` sitting *inside* tile A's null band (`Z_low < midband < Z_high`). -/
def midbandA : Float := (zLow1 + zHigh1) / 2.0

-- Negative — a noise inside the null band is correctly *not* a recovered edge under tile A.
#eval assertDecision "borderline noise inside the null band is not a recovered edge"
  (Spec.zSignificantFn midbandA zLow1) false

/-- A borderline signal between the two tiles' lower thresholds (`Z_low,A < 0.2 < Z_low,B`). -/
def borderline : Float := 0.2

-- Negative — for a borderline signal the verdict genuinely DIFFERS across the two tiles (significant
-- under B, not under A): RNG-invariance is a property of *well-separated* edges, not automatic. This
-- gives the positive RNG-invariance check above its teeth.
#eval assertDecision "borderline signal is NOT RNG-invariant (verdict differs across tiles)"
  (Spec.zSignificantFn borderline zLow1 == Spec.zSignificantFn borderline zLow2) false

-- Negative — the observed signal noise is a genuine fraction in `[0,1]`, far from the wrong golden
-- (the null upper tail `Z_high`): the parity check is specific, not a blanket pass.
#eval assertGe "obsSignal ≠ Z_high golden (parity is statistic-specific)"
  (Float.abs (obsSignal - zHigh1)) 0.1

end NN.Examples.KernelFlows.Parity.Stochastic
