/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape.LinearAlgebra

/-!
# Shape-Tagged Tensors

`PackedTensor` stores a tensor together with the shape that indexes its type. It is the canonical
representation used whenever a collection contains tensors of different shapes, including runtime
contexts, autograd tapes, graph interpreters, and certificate checkers.
-/

@[expose] public section

namespace Spec

/-- A tensor paired with the shape that indexes its type. -/
structure PackedTensor (α : Type) where
  /-- The runtime shape of the tensor. -/
  shape : Shape
  /-- The tensor value, indexed by its stored shape. -/
  tensor : Tensor α shape

namespace PackedTensor

/-- Package a statically shaped tensor for shape-erased storage. -/
@[simp] def ofTensor {α : Type} {shape : Shape} (tensor : Tensor α shape) : PackedTensor α :=
  ⟨shape, tensor⟩

@[simp] theorem shape_ofTensor {α : Type} {shape : Shape} (tensor : Tensor α shape) :
    (ofTensor tensor).shape = shape :=
  rfl

@[simp] theorem tensor_ofTensor {α : Type} {shape : Shape} (tensor : Tensor α shape) :
    (ofTensor tensor).tensor = tensor :=
  rfl

/-- Cast the stored tensor after checking its runtime shape. -/
def cast {α : Type} {shape : Shape} (value : PackedTensor α) (h : value.shape = shape) :
    Tensor α shape :=
  Tensor.castShape value.tensor h

@[simp] theorem cast_self {α : Type} (value : PackedTensor α) (h : value.shape = value.shape) :
    value.cast h = value.tensor := by
  rw [Subsingleton.elim h rfl]
  rfl

/-- Materialize the tensor payload without changing its denotation or shape. -/
def materialize {α : Type} (value : PackedTensor α) : PackedTensor α :=
  ⟨value.shape, Tensor.materialize value.tensor⟩

/-- Materialization preserves a packed tensor extensionally. -/
@[simp] theorem materialize_eq {α : Type} (value : PackedTensor α) : materialize value = value := by
  cases value
  simp [materialize]

/-- Swap two adjacent axes at `depth`, retaining the resulting shape in the package. -/
def swapAdjacentAtDepth {α : Type}
    (value : PackedTensor α) (depth : Nat) : PackedTensor α :=
  match value with
  | ⟨shape, tensor⟩ =>
      ⟨shape.swapAdjacentAtDepth depth, Tensor.swapAdjacentAxes (tensor := tensor) depth⟩

end PackedTensor
end Spec
