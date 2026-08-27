/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Linear
public import NN.Spec.Models.LogisticRegression
public import NN.Spec.Module.Core

/-!
# Logistic regression as an `Spec.Module`

The model spec provides a standalone gradient-descent baseline and prediction helpers.
This file adds the `Spec.Module` wrapper (Linear + Sigmoid) for composition and export.
-/

@[expose] public section


namespace Spec.Module

open Tensor
open Activation

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-- Logistic regression wrapped as `Spec.Module` (linear + sigmoid). -/
def logisticRegression {p : ℕ} (model : LogisticRegression p 0 α) :
  Spec.Module α ([p]) ([1]) :=
  let weightMatrix : Tensor α [1, p] :=
    Tensor.dim (fun _ => model.weights)
  let biasVec : Tensor α [1] :=
    Tensor.dim (fun _ => Tensor.scalar model.intercept)
  let linearSpec : Spec.LinearSpec α p 1 :=
    { weights := weightMatrix, bias := biasVec }
  {
    forward := fun x =>
      Activation.sigmoidSpec (Spec.linearSpec (α := α) linearSpec x),
    kind := "LogisticRegression",
    pythonExpr := s!"nn.Sequential(nn.Linear({p}, 1), nn.Sigmoid())"
  }

end Spec.Module
