/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded
public import NN.Runtime.Autograd.TorchLean.Fno

/-!
# Fourier Neural Operators

The public FNO model is polymorphic in spatial rank. Its portable implementation uses a dense
multidimensional DFT with separate real and imaginary tensors. Accelerated implementations may use
backend capsules such as the specialized cuFFT path.

Reference: Zongyi Li et al., *Fourier Neural Operator for Parametric Partial Differential
Equations*, ICLR 2021.
-/

@[expose] public section

namespace TorchLean.nn.models

/-- Configuration for a scalar-field FNO over `d` spatial axes. -/
structure FnoConfig (d : Nat) where
  /-- Extent of each spatial axis. -/
  spatial : Tensor Nat [d]
  /-- Number of low and high Fourier modes retained along each axis. -/
  modes : Tensor Nat [d]
  /-- Every spatial axis is nonempty. -/
  spatialNonzero : ∀ axis : Fin d, spatial.getScalar axis ≠ 0
  /-- Low and high retained bands do not overlap along any axis. -/
  modesFit : ∀ axis : Fin d, 2 * modes.getScalar axis ≤ spatial.getScalar axis
  /-- Width of the latent channel representation. -/
  width : Nat
  /-- The latent channel representation is nonempty. -/
  widthNonzero : width ≠ 0
  /-- Number of spectral residual blocks. -/
  blocks : Nat

/-- Input shape of the scalar field, with any independently mapped axes prepended. -/
abbrev FnoConfig.inputShape {d : Nat} (cfg : FnoConfig d)
    (leading : List Nat := []) : List Nat :=
  leading ++ cfg.spatial.toList

/-- Output shape of the scalar field, with any independently mapped axes prepended. -/
abbrev FnoConfig.outputShape {d : Nat} (cfg : FnoConfig d)
    (leading : List Nat := []) : List Nat :=
  leading ++ cfg.spatial.toList

/--
Build the portable multidimensional FNO model.

The shape and mode contracts are independent of the selected device and provider and are retained
when a fused kernel is chosen.
-/
def fno {d : Nat} (cfg : FnoConfig d) (leading : List Nat := []) :
    nn.Builder (nn.Sequential (cfg.inputShape leading) (cfg.outputShape leading)) :=
  nn.withSeed fun seed =>
    nn.mapEach leading <|
      _root_.Runtime.Autograd.TorchLean.NN.FNO.model
        cfg.spatial cfg.modes cfg.width cfg.blocks (seed := seed)

end TorchLean.nn.models
