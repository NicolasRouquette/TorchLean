/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape.LinearAlgebra

/-!
# Shape-Tagged Tensors

A small existential package for storing tensors whose shapes are known only at runtime. The package
and its shape-changing adjacent-swap operation are shared by executable tensor interpreters.
-/

@[expose] public section

namespace Spec

/-- A tensor paired with the shape that indexes its type. -/
abbrev PackedTensor (alpha : Type) [Context alpha] : Type :=
  Sigma fun shape : Shape => Tensor alpha shape

namespace PackedTensor

/-- The runtime shape carried by a packed tensor. -/
@[simp] def shape {alpha : Type} [Context alpha] (value : PackedTensor alpha) : Shape := value.1

/-- The tensor stored in a packed tensor, indexed by its recovered shape. -/
@[simp] def tensor {alpha : Type} [Context alpha] (value : PackedTensor alpha) :
    Tensor alpha value.shape := value.2

/-- Package a statically shaped tensor for shape-erased storage. -/
@[simp] def mk {alpha : Type} [Context alpha] (shape : Shape) (tensor : Tensor alpha shape) :
    PackedTensor alpha :=
  ⟨shape, tensor⟩

/-- Swap two adjacent axes at `depth`, retaining the resulting shape in the package. -/
def swapAdjacentAtDepth {alpha : Type} [Context alpha]
    (value : PackedTensor alpha) (depth : Nat) : PackedTensor alpha :=
  match value with
  | ⟨shape, tensor⟩ =>
      ⟨shape.swapAdjacentAtDepth depth, Tensor.swapAtDepthHelper (tensor := tensor) depth⟩

end PackedTensor
end Spec
