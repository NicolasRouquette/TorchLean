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


/-- Classification head that preserves any supplied leading dimensions. -/
def classifier (leading : Spec.Shape := .scalar) {s : Spec.Shape}
    (classes : Nat) (seedW seedB : Nat := 0) :
    Sequential (leading.concat s) (leading.appendDim classes) :=
  seq!
    flattenLeading leading (s := s),
    linear (Spec.Shape.size s) classes (seedW := seedW) (seedB := seedB) (pfx :=
      leading)

/-- Regression head that preserves any supplied leading dimensions. -/
def regressor (leading : Spec.Shape := .scalar) {s : Spec.Shape}
    (outDim : Nat := 1) (seedW seedB : Nat := 0) :
    Sequential (leading.concat s) (leading.appendDim outDim) :=
  seq!
    flattenLeading leading (s := s),
    linear (Spec.Shape.size s) outDim (seedW := seedW) (seedB := seedB) (pfx :=
      leading)

end heads

end Internal

end nn

end TorchLean
