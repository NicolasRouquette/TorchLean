/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape.LinearAlgebra

/-!
# Shape-Erased Tensors

`SomeTensor` stores a `Spec.Tensor` together with the shape that indexes its type. It is the sole
general shape-erasure boundary for tensors in TorchLean. Runtime collections use it when they must
contain tensors of different shapes, including autograd tapes, graph interpreters, and certificate
checkers.

Backend-specific resources are not alternative tensor wrappers. For example, the CUDA tape keeps
an opaque device buffer together with runtime shape and allocation metadata; operation-polymorphic
programs similarly package references owned by a monad rather than tensor values.
-/

@[expose] public section

namespace Spec

/-- A tensor paired with the shape that indexes its type. -/
structure SomeTensor (α : Type) where
  /-- The runtime shape of the tensor. -/
  shape : Shape
  /-- The tensor value, indexed by its stored shape. -/
  tensor : Tensor α shape

namespace SomeTensor

/-- Package a statically shaped tensor for shape-erased storage. -/
@[simp] def ofTensor {α : Type} {shape : Shape} (tensor : Tensor α shape) : SomeTensor α :=
  ⟨shape, tensor⟩

@[simp] theorem shape_ofTensor {α : Type} {shape : Shape} (tensor : Tensor α shape) :
    (ofTensor tensor).shape = shape :=
  rfl

@[simp] theorem tensor_ofTensor {α : Type} {shape : Shape} (tensor : Tensor α shape) :
    (ofTensor tensor).tensor = tensor :=
  rfl

/-- Cast the stored tensor after checking its runtime shape. -/
def cast {α : Type} {shape : Shape} (value : SomeTensor α) (h : value.shape = shape) :
    Tensor α shape :=
  Tensor.castShape value.tensor h

@[simp] theorem cast_self {α : Type} (value : SomeTensor α) (h : value.shape = value.shape) :
    value.cast h = value.tensor := by
  rw [Subsingleton.elim h rfl]
  rfl

/-- Repacking a tensor after a successful shape cast recovers the original value. -/
@[simp] theorem ofTensor_cast {α : Type} (value : SomeTensor α) {shape : Shape}
    (h : value.shape = shape) : ofTensor (value.cast h) = value := by
  subst shape
  rw [cast_self]
  cases value
  rfl

/-- Erasing a tensor's shape after transport recovers the original shape-erased value. -/
@[simp] theorem ofTensor_castShape {α : Type} {shape shape' : Shape}
    (tensor : Tensor α shape) (h : shape = shape') :
    ofTensor (Tensor.castShape tensor h) = ofTensor tensor := by
  cases h
  rfl

/-- Materialize the tensor payload without changing its denotation or shape. -/
def materialize {α : Type} (value : SomeTensor α) : SomeTensor α :=
  ⟨value.shape, Tensor.materialize value.tensor⟩

/-- Materialization preserves a shape-erased tensor extensionally. -/
@[simp] theorem materialize_eq {α : Type} (value : SomeTensor α) : materialize value = value := by
  cases value
  simp [materialize]

/-- Swap two adjacent axes at `depth`, retaining the resulting shape in the package. -/
def swapAdjacentAtDepth {α : Type}
    (value : SomeTensor α) (depth : Nat) : SomeTensor α :=
  match value with
  | ⟨shape, tensor⟩ =>
      ⟨shape.swapAdjacentAtDepth depth, Tensor.swapAdjacentAxes (tensor := tensor) depth⟩

end SomeTensor
end Spec
