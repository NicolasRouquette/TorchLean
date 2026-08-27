# `NN.Tensor`

This folder is the small, user-facing tensor API for TorchLean. It is the layer you reach for when
you want to write a tensor literal, build a compact example, or pass shaped data into a model without
opening the lower-level spec and runtime internals.

The design is intentionally modest. TorchLean tensors should feel close enough to ordinary ML code
that examples are readable, but they should still carry the information Lean needs: the scalar type,
the shape, and enough construction evidence to avoid silent shape mistakes.

For public use, prefer the curated library import or the tensor entrypoint:

```lean
import NN
-- or, if you only want this subsystem:
import NN.Tensor
```

`NN.Tensor` is the canonical import for this layer.

Use `import NN` for ordinary model code. Use `NN.Tensor` only when the file is explicitly
about tensor construction, indexing, shape operations, printing, or literal syntax.

## What This Layer Owns

The key invariant is that semantics and proofs stay in the spec layer (`NN/Spec/*`), while this
layer stays focused on ergonomics:

- DTypes are Lean types: you write `Tensor Float s`, `Tensor ℚ s`, and other scalar backends used
  by the project.
- Shapes live in the type when the program asks for a static tensor: `tensor!` and
  `Spec.Tensor.ofFn` construct vectors and higher-rank values without introducing parallel
  container types.
- If you see `Tensor Float _`, the `_` asks Lean to infer the shape from the right hand side.
- When dimensions arrive at runtime, use `ofArray dims.toList values`. Lean may use the runtime
  dimension value in the result type, so this still returns an ordinary `Tensor`.
- For constants, `tensorOfArray!` and `tensorF!` trade a bit of macro expansion for cleaner literal code.
- `tensor!` accepts nested bracket syntax and flattens in row-major order, which is handy for
  handwritten examples.

TorchLean does not expose a second fixed-shape vector or array-backed tensor type. Vectors
are tensors such as `Tensor Float [128]`. Runtime numerical storage uses `Array`; a checked boundary
converts it with `Tensor.ofArray` or `Tensor.ofFlatArrayExact`. List syntax remains available for
shapes and tensor literals, but lists are not a public numerical storage format. Use
`Tensor.toArray` when storage must leave the typed tensor layer again.

A tensor literal can appear in a training example, an executable regression check, or a theorem
statement. The mathematical meanings of matrix multiplication, convolution, softmax, and reductions
are defined in `NN.Spec`; this directory provides their convenient tensor syntax.

## Static And Dynamic Shapes

Prefer statically shaped tensors when the shape is part of the claim you are making. For example, a
small MLP theorem should expose the input and output dimensions in the type so the layer composition
is checked by Lean before any runtime code is involved.

Runtime dimensions occur in file-backed batches, loaded NumPy arrays, and exported artifacts. They
do not require another public tensor type: `ofArray dims values` returns `Tensor α dims` after
checking the flat payload length. Users still receive an ordinary `Tensor`; shape erasure begins
only inside runtime components that must place differently shaped values in one collection.

## Runtime Relationship

There is one user-facing tensor type. The two dependent containers below solve narrower internal
problems; neither replaces `Tensor` in model code:

- `Tensor α shape` is the shape-indexed value used by specifications, proofs, and the CPU reference
  runtime. It is executable whenever `α` is executable.
- `TensorPack α shapes` is a statically heterogeneous tuple. Its type records the shape of every
  tensor, so model state can contain a weight matrix, bias vector, and normalization parameters
  without shape erasure.
- `SomeTensor α` is internal runtime shape erasure. It pairs one tensor with its runtime shape so
  tapes and graph interpreters can keep differently shaped values in one array.
- Runtime value references identify values owned by an eager session or typed graph. They are
  handles, not another tensor representation; the public generic spelling is `Runtime.ValueRef`.
- CUDA execution stores flat device memory in `AnyBuffer`, together with the runtime shape needed to
  validate uploads, downloads, and kernel results.

Thus the CPU path does not translate a proof tensor into a second host tensor class: it evaluates the
same `Tensor` definition. CUDA necessarily uses native device storage, and the upload/download and
kernel-contract layers state where that representation leaves the logical model. The graph and
runtime correctness modules prove correspondence for supported operations; they do not silently
assert that arbitrary external machine code is correct.

## Files

- `../Tensor.lean`: import-only umbrella for the complete public tensor surface.
- `Constructors.lean`: the `TorchLean.Tensor` type and shape exports, static constructors, checked
  array conversion, one-hot encoding, and scalar conversion helpers.
- `Operations.lean`: coordinate lookup, stacking, arbitrary-axis `take`, prefix mapping,
  `flattenAfter`, and `flattenThenTake`.
- `Syntax.lean`: checked tensor literal and bounded-index macros.
- `Printing.lean`: dtype labels and printing, including explicit refusal for proof-level scalar
  backends such as `ℝ`.
- `Pack.lean`: the statically heterogeneous tuple used for model state and typed contexts.
- `ShapeErasure.lean`: checked conversion between typed packs and runtime shape-erased arrays.
