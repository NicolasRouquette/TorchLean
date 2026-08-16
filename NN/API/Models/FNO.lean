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
structure FNOConfig (d : Nat) where
  /-- Extent of each spatial axis. -/
  spatial : Vector Nat d
  /-- Number of low and high Fourier modes retained along each axis. -/
  modes : Vector Nat d
  /-- Every spatial axis is nonempty. -/
  spatialNonzero : ∀ axis : Fin d, spatial.get axis ≠ 0
  /-- Low and high retained bands do not overlap along any axis. -/
  modesFit : ∀ axis : Fin d, 2 * modes.get axis ≤ spatial.get axis
  /-- Width of the latent channel representation. -/
  width : Nat
  /-- The latent channel representation is nonempty. -/
  widthNonzero : width ≠ 0
  /-- Number of spectral residual blocks. -/
  blocks : Nat
  /-- Base seed for parameter initialization. -/
  seed : Nat := 0

/-- Input shape of the scalar field sampled on `cfg.spatial`. -/
abbrev fnoInShape {d : Nat} (cfg : FNOConfig d) : Spec.Shape :=
  Spec.Shape.ofList cfg.spatial.toList

/-- Output shape of the scalar field sampled on `cfg.spatial`. -/
abbrev fnoOutShape {d : Nat} (cfg : FNOConfig d) : Spec.Shape :=
  Spec.Shape.ofList cfg.spatial.toList

/--
Build the portable multidimensional FNO model.

The shape and mode contracts are independent of the selected device and provider and are retained
when a fused kernel is chosen.
-/
def fno {d : Nat} (cfg : FNOConfig d) :
    nn.Builder (nn.Sequential (fnoInShape cfg) (fnoOutShape cfg)) :=
  pure <| _root_.Runtime.Autograd.TorchLean.NN.FNO.model
    cfg.spatial cfg.modes cfg.width cfg.blocks (seed := cfg.seed)

end TorchLean.nn.models
