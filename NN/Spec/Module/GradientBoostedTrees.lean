/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.GradientBoostedTrees
public import NN.Spec.Module.Core

/-!
# Gradient boosted trees as an `Spec.Module`

The model spec defines the ensemble prediction function. This file adds the `Spec.Module` wrapper
for composition and export.
-/

@[expose] public section


namespace Spec.Module

open Tensor

variable {α : Type} [Context α]

/-- Gradient boosted trees as an `Spec.Module`. -/
def gradientBoostedTrees {nTrees maxDepth nFeatures : Nat}
  (model : GradientBoostedTreesSpec α nFeatures nTrees maxDepth) :
  Spec.Module α (.dim nFeatures .scalar) .scalar :=
{
  forward := fun x => gradientBoostedTreesForwardSpec model x,
  kind := "GradientBoostedTrees",
  pythonExpr := "UnsupportedLayer(\"GradientBoostedTrees\", "
        ++ "\"sklearn.ensemble.GradientBoostingRegressor\")"
}

end Spec.Module
