/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.Svm
public import NN.Spec.Module.Linear

/-!
# Linear SVM as an `Spec.Module`

The SVM spec model includes a gradient-descent baseline and prediction helpers.
This file adds the `Spec.Module` wrapper so it can be composed/exported in the module system.

References:
- For the actual SVM objective/gradients and classic SVM citations (Cortes-Vapnik, Vapnik),
  see `NN.Spec.Models.Svm`:
  `NN/Spec/Models/Svm.lean`.
- PyTorch analogy for the forward map: a linear score `X @ w + b` is the same shape-level role as
  `torch.nn.Linear(p, 1)` (no activation).
-/

@[expose] public section


namespace Spec.Module

open Tensor

variable {α : Type} [Context α]

/-- A linear SVM represented as a single-output linear module. -/
def linearSvm {p : ℕ} (model : LinearSVM p α) :
  Spec.Module α (.dim p .scalar) (.dim 1 .scalar) :=
  let weightMatrix : Tensor α (.dim 1 (.dim p .scalar)) :=
    Tensor.dim (fun _ => model.w)
  let biasVec : Tensor α (.dim 1 .scalar) :=
    Tensor.dim (fun _ => Tensor.scalar model.b)
  let linearSpec : Spec.LinearSpec α p 1 :=
    { weights := weightMatrix, bias := biasVec }
  Spec.Module.linear (α := α) linearSpec

end Spec.Module
