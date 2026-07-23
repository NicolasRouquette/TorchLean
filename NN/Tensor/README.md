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

`NN.Tensor.API` is the implementation leaf behind the entrypoint.

Use `import NN` for ordinary model code. Use `NN.Tensor` only when the file is explicitly
about tensor construction, printing, or low-level tensor API behavior.

## What This Layer Owns

The key invariant is that semantics and proofs stay in the spec layer (`NN/Spec/*`), while this
layer stays focused on ergonomics:

- DTypes are Lean types: you write `Tensor Float s`, `Tensor ℚ s`, and other scalar backends used by the project.
- Shapes live in the type when the program asks for a static tensor: constructors like `vector`
  remember `xs.length` in the result type.
- If you see `Tensor Float _`, the `_` asks Lean to infer the shape from the right hand side.
- When you need dynamic shapes, use `ofList` (runtime dims + runtime length check) or
  `DynTensor` (store the shape as data instead of in the type).
- For constants, `tensorOfList!` and `tensorF!` trade a bit of macro expansion for cleaner literal code.
- `tensor!` accepts nested bracket syntax and flattens in row-major order, which is handy for
  handwritten examples.

This separation matters for the proof story. A tensor literal can appear in a training example, an
executable regression check, or a theorem statement, but the mathematical meaning of operations such
as matrix multiplication, convolution, softmax, and reductions is defined elsewhere. The tensor API
is the front door; it is not the proof boundary.

## Static And Dynamic Shapes

Prefer statically shaped tensors when the shape is part of the claim you are making. For example, a
small MLP theorem should expose the input and output dimensions in the type so the layer composition
is checked by Lean before any runtime code is involved.

Use dynamic tensors when the shape really is data: file-backed batches, loaded NumPy arrays, exported
runtime artifacts, or tools that inspect tensors produced outside Lean. Dynamic shape checks are still
explicit; they are just values rather than type indices.

## Runtime Relationship

`NN.Tensor` does not decide whether execution happens through TorchLean-native kernels, CUDA kernels,
ATen/libtorch, or a proof-oriented scalar model. Runtime modules consume these tensors or translate
from them, then make their own backend choice. Keeping this layer backend-neutral is what lets the
same example be read as user code, lowered into an IR, or connected to a verification checker.

## Files

- `../Entrypoint/Tensor.lean`: the stable tensor subsystem import.
- `API.lean`: the implementation leaf. Includes shape aliases, `vector`/`matrix`, dynamic N-D
  constructors, padding friendly 2D constructors for ragged data, and a small printing API that
  refuses to print proof level scalar backends like `ℝ`.
