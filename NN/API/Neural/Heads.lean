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
def classifier (leading : List Nat := []) {shape : List Nat}
    (classes : Nat) (seedW seedB : Nat := 0) :
    Sequential (leading ++ shape) (leading ++ [classes]) :=
  seq!
    flattenAfter leading (shape := shape),
    linear shape.prod classes (seedW := seedW) (seedB := seedB) (leading := leading)

/-- Regression head that preserves any supplied leading dimensions. -/
def regressor (leading : List Nat := []) {shape : List Nat}
    (outDim : Nat := 1) (seedW seedB : Nat := 0) :
    Sequential (leading ++ shape) (leading ++ [outDim]) :=
  seq!
    flattenAfter leading (shape := shape),
    linear shape.prod outDim (seedW := seedW) (seedB := seedB) (leading := leading)

end heads

end Internal

end nn

end TorchLean
