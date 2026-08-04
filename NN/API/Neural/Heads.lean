/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.API.Neural.Transformer

/-!
Task heads for public neural-network models.

The definitions here package classifier, regression, and language-model heads that sit on top of
the reusable block and Transformer APIs.
-/

@[expose] public section

namespace TorchLean
namespace nn
namespace Internal
namespace heads


/--
Classification head: `Flatten -> Linear`.

Named head constructor built from `nn.flattenLinear`.
-/
def classifier {s : Spec.Shape} (classes : Nat) (seedW seedB : Nat := 0) :
    Sequential s (.dim classes .scalar) :=
  flattenLinear (s := s) classes seedW seedB

/-- Regression head: `Flatten -> Linear` with `outDim` outputs. -/
def regressor {s : Spec.Shape} (outDim : Nat := 1) (seedW seedB : Nat := 0) :
    Sequential s (.dim outDim .scalar) :=
  flattenLinear (s := s) outDim seedW seedB

/--
`Flatten(start_dim=1) -> Linear` head for batched tensors.

Input:  `N × σ`
Output: `Mat N classes`
-/
def classifierBatch {n : Nat} {s : Spec.Shape} (classes : Nat) (seedW seedB : Nat := 0) :
    Sequential (.dim n s) (.dim n (.dim classes .scalar)) :=
  seq!
    flattenBatch (n := n) (s := s),
    linear (Spec.Shape.size s) classes (seedW := seedW) (seedB := seedB) (pfx :=
      .dim n .scalar)

/-- Batched regression head: `Flatten(start_dim=1) -> Linear(_, outDim)` producing `Mat N outDim`.
  -/
def regressorBatch {n : Nat} {s : Spec.Shape} (outDim : Nat := 1) (seedW seedB : Nat := 0) :
    Sequential (.dim n s) (.dim n (.dim outDim .scalar)) :=
  seq!
    flattenBatch (n := n) (s := s),
    linear (Spec.Shape.size s) outDim (seedW := seedW) (seedB := seedB) (pfx :=
      .dim n .scalar)

end heads

end Internal

end nn

end TorchLean
