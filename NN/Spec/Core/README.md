# `NN.Spec.Core`

This directory is the spec-level core of TorchLean: shapes, scalar contexts, and shape-indexed
tensors. It is the vocabulary used by layers, losses, graph semantics, runtime approximation
theorems, and many verification statements.

The key idea is that a tensor carries more than a blob of numbers. Its shape, scalar
interpretation, and allowed operations are part of the object being specified.

## Files

- `Context.lean`: the `Context` typeclass for scalar backends: algebraic operations, comparisons,
  casts, and the small amount of structure needed by specs.
- `Scalar.lean`: scalar helpers and instances used across the spec layer.
- `Shape.lean`: the `Shape` datatype, runtime axis validation (`validAxis?`), axis-permutation
  planning (`swapAdjacentAxes`), well-formedness, and broadcasting evidence such as
  `CanBroadcastTo`.
- `Tensor/`: the core tensor datatype plus constructors, rank-one operations, linear algebra, and
  factorizations.
- `Tensor/SomeTensor.lean`: internal runtime shape erasure for evaluators that must store tensors of
  different shapes together. The stored shape is checked before recovering a typed tensor.
- `Tensor.lean`: umbrella import for the core tensor API.
- `TensorOps.lean`: elementwise maps and pointwise tensor operations.
- `TensorReductionShape.lean` and `TensorReductionShape/`: reductions, reshape/flatten/unflatten,
  concat/slice, broadcasting, and shape-changing helpers.
- `Sequence.lean`: helpers for common time and sequence-axis patterns.
- `Tensor/Constructors.lean`: total builders, including checked flat-list and flat-array boundaries.
- `TensorGrad.lean`: gradient-related specs, including clipping helpers.
- `Complex.lean`: small complex-number support used by FFT/FNO-style specifications.
- `Utils.lean`: glue utilities: casting maps, `*_like` constructors, list/tensor helpers, and
  pretty-printing.

Model and layer APIs live under `NN/Spec/Layers` and `NN/Spec/Models`. They should be built from
these primitives rather than inventing a second tensor language.

## Boundary

This folder defines meanings, not fast kernels. Runtime code may execute the same operations over
`Float`, `IEEE32Exec`, CUDA buffers, or external providers, but a theorem should say which spec
object that runtime path is meant to approximate or preserve.
