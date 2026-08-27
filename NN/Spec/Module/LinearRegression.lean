/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.LinearRegression
public import NN.Spec.Module.Core

/-!
# Linear regression as an `Spec.Module`

The spec model file (`NN/Spec/Models/LinearRegression.lean`) defines the math: forward and
backward/VJP pieces.

This file provides the small `Spec.Module` wrapper so linear regression can be composed via
`Spec.Module.Chain` and recognized by export tooling (PyTorch string renderings, dimension metadata, etc.).
-/

@[expose] public section


namespace Spec.Module

open Tensor

variable {α : Type} [Context α]

/-- Package a fixed-parameter linear regression as a `Spec.Module`.

PyTorch analogy: `nn.Linear(inDim, 1)`.
-/
def linearRegression {inDim : Nat}
  (model : LinearRegressionSpec α inDim) :
  Spec.Module α ([inDim]) .scalar :=
{
  forward := fun x => linearRegressionForwardSpec model x,
  kind := "LinearRegression",
  pythonExpr := s!"nn.Linear({inDim}, 1)"
}

end Spec.Module
