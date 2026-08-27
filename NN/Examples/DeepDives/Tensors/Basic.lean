/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor
public import NN.Widgets

/-!
# Tensor Basics

TorchLean uses one shape-indexed tensor type for model inputs, parameters, intermediate values,
and mathematical specifications. The shape is part of the Lean type, so indexing removes one axis
and operations can state their input and output shapes directly.

Flat arrays still appear at file, FFI, and kernel boundaries. They are storage, not another tensor
API: a size check converts incoming storage to `Tensor`, and `Tensor.toArray` materializes outgoing
row-major data.
-/

@[expose] public section

namespace NN.Examples.DeepDives.Tensors.Basic

open TorchLean

/-! ## Construction and indexing -/

/-- A matrix literal. Its element type and both dimensions are visible in the type. -/
def matrix : Tensor Float [2, 3] :=
  tensor! [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

/-- Selecting one coordinate on the leading axis returns a tensor of the remaining shape. -/
def firstRow : Tensor Float [3] :=
  Tensor.get matrix ⟨0, by decide⟩

/--
The same matrix constructed from an array arriving at a dynamic boundary.

The proof discharges the boundary check; the result is the same canonical `Tensor` used above.
-/
def matrixFromArray : Tensor Float [2, 3] :=
  Tensor.ofFlatArrayExact [2, 3]
    #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    (by decide)

/-- Row-major storage for serialization or a native kernel call. -/
def matrixStorage : Array Float :=
  matrix.toArray

#tensor_view matrix
#tensor_view firstRow
#tensor_stats_view matrix

/-! ## Linear algebra -/

/-- A `4 × 3` parameter tensor. -/
def weights : Tensor Float [4, 3] :=
  Tensor.full [4, 3] 0.1

/-- A three-component input vector. -/
def input : Tensor Float [3] :=
  tensor! [1.0, 2.0, 3.0]

/-- Matrix multiplication preserves the output dimension in the result type. -/
def linearOutput : Tensor Float [4] :=
  Spec.Tensor.matVecMulSpec weights input

#tensor_view linearOutput
#tensor_stats_view linearOutput

/-! ## Batches and reshaping -/

/-- Two `2 × 3` samples stacked along a leading batch axis. -/
def batch : Tensor Float [2, 2, 3] :=
  tensor! [
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
  ]

def firstSample : Tensor Float [2, 3] :=
  Tensor.get batch ⟨0, by decide⟩

def secondSample : Tensor Float [2, 3] :=
  Tensor.get batch ⟨1, by decide⟩

#tensor_view batch
#tensor_view firstSample
#tensor_view secondSample

/-- Reshape a vector without changing row-major element order. -/
def reshapedMatrix : Tensor Float [2, 3] :=
  Spec.Tensor.reshapeSpec (tensor! [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] : Tensor Float [6])
    (by decide)

#tensor_view reshapedMatrix

/-! ## Shape-preserving transformations -/

def tensorA : Tensor Float [2, 2] :=
  Tensor.full [2, 2] 2.0

def tensorB : Tensor Float [2, 2] :=
  Tensor.full [2, 2] 3.0

def elementwiseSum : Tensor Float [2, 2] :=
  Spec.Tensor.addSpec tensorA tensorB

/-- A pointwise transformation has the same shape discipline as a gradient buffer. -/
def doubledOutput : Tensor Float [4] :=
  Tensor.map (2.0 * ·) linearOutput

#tensor_view elementwiseSum
#tensor_view doubledOutput

end NN.Examples.DeepDives.Tensors.Basic
