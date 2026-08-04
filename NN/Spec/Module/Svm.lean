/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.Svm
public import NN.Spec.Module.Linear

/-!
# Linear SVM as an `NNModuleSpec`

The SVM spec model includes a gradient-descent baseline and prediction helpers.
This file adds the `NNModuleSpec` wrapper so it can be composed/exported in the module system.

References:
- For the actual SVM objective/gradients and classic SVM citations (Cortes-Vapnik, Vapnik),
  see `NN.Spec.Models.Svm`:
  `NN/Spec/Models/Svm.lean`.
- PyTorch analogy for the forward map: a linear score `X @ w + b` is the same shape-level role as
  `torch.nn.Linear(p, 1)` (no activation).
-/

@[expose] public section


namespace Spec

open Tensor
open ModSpec

variable {α : Type} [Context α]

/-- A linear SVM represented as a single-output linear module. -/
def linearSVMModule {p : ℕ} (model : LinearSVM p α) :
  NNModuleSpec α (.dim p .scalar) (.dim 1 .scalar) :=
  let weightMatrix : Tensor α (.dim 1 (.dim p .scalar)) :=
    Tensor.dim (fun _ => model.w)
  let biasVec : Tensor α (.dim 1 .scalar) :=
    Tensor.dim (fun _ => Tensor.scalar model.b)
  let lspec : Spec.LinearSpec α p 1 :=
    { weights := weightMatrix, bias := biasVec }
  Spec.LinearModuleSpec (α := α) lspec

/-- A one-node specification chain containing a linear SVM. -/
def linearSVMChain {p : ℕ} (model : LinearSVM p α) :
  SpecChain α (.dim p .scalar) (.dim 1 .scalar) :=
  SpecChain.single (linearSVMModule (α := α) model)

end Spec
