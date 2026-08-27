/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Constructors
public import NN.Spec.Core.Tensor.Core
public import NN.Spec.Core.Tensor.Linalg

/-!
# Tensor

Umbrella module for the core tensor API.

Most downstream modules import this umbrella. The concrete tensor definitions live in focused files:
- `NN.Spec.Core.Tensor.Core`          (datatype + accessors)
- `NN.Spec.Core.Tensor.Constructors`  (total builders)
- `NN.Spec.Core.Tensor.Linalg`        (matrix and vector operations)

Elementwise ops and reductions remain in:
- `NN.Spec.Core.TensorOps`
- `NN.Spec.Core.TensorReductionShape`
-/

@[expose] public section


namespace Spec
namespace Tensor

-- Convenience re-exports: make accessors/constructors from `Spec` available as `Spec.Tensor.*`,
-- so `open Tensor` brings them into scope.
  export Spec (shapeOf getSpec get get2 getAtOrZero finZero getHead getTail
  tensorCast replicate
  sliceRangeSpec
  fill singleton padLeft
  identityTensorSpec matMulSpec matVecMulSpec vecMatMulSpec outerProductSpec)

end Tensor
end Spec
