/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.Pca
public import NN.Spec.Module.Core

/-!
# PCA as an `Spec.Module`

The PCA spec model defines a projection `y = (x - mean) · componentsᵀ`.
This file provides the `Spec.Module` wrapper used for composition and export.
-/

@[expose] public section


namespace Spec.Module

open Tensor

variable {α : Type} [Context α]

/-- PCA module specification following `Spec.Module`. -/
def pca {inDim outDim : Nat} (m : PCASpec α inDim outDim) :
  Spec.Module α (.dim inDim .scalar) (.dim outDim .scalar) :=
  { forward := pcaForwardSpec m
    kind := "PCA"
    -- Centering contributes the affine bias `-components * mean`.
    pythonExpr := s!"nn.Linear({inDim}, {outDim}, bias=True)" }

end Spec.Module
