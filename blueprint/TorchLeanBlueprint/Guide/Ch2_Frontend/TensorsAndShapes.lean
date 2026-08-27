import VersoManual

open Verso.Genre Manual

#doc (Manual) "Tensors That Remember Their Shapes" =>
%%%
tag := "tensors-shapes"
%%%

Most tensor libraries carry a shape beside a buffer and check compatibility when an operation
runs. TorchLean also carries the shape in the Lean type. A function that accepts a length-four
vector cannot accidentally receive a $`2\times 2` matrix, even though both contain four scalar values.

That choice is the foundation for the rest of the library. Layers become checked maps between
shapes, parameter packs remember the layout expected by a model, and theorem statements do not
need a side condition saying that every intermediate tensor happened to have the right dimensions.

# Run The First Tensor Program

From the repository root, run:

```
lake exe torchlean quickstart_tensors
```

The output is:

```
== Quickstart: tensor basics ==
[Float] [0.100000, 0.200000, 0.300000, 0.400000]
[ℚ] [1/10, 1/5, 3/10, 2/5]
[Int] [1, 2, 3, 4]
[Float32] [0.100000, 0.200000, 0.300000, 0.400000]
[IEEE32Exec] [0.100000, 0.200000, 0.300000, 0.400000]
[Float] [[[1.000000, 2.000000], [3.000000, 4.000000]],
         [[5.000000, 6.000000], [7.000000, 8.000000]]]
Expected failure printing Tensor ℝ:
  Refusing to print `Tensor ℝ` (proof-level);
  cast to `Float`/`Float32`/`IEEE32Exec`/`ℚ` to display.
```

The first five tensors have the same shape but different scalar meanings. `ℚ` displays exact
fractions. `Float32` uses Lean's native binary32 operations, while `IEEE32Exec` stores and executes
explicit binary32 bit patterns. `Float` is Lean's host binary64 type. The final attempted tensor
over `ℝ` is a mathematical object; arbitrary real numbers are not executable data, so printing it
is rejected rather than pretending to approximate it.

The complete program is in
[`TensorBasics.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/TensorBasics.lean),
and the following definitions develop the same ideas one operation at a time.

# One Tensor Type

The canonical specification type is:

$$`\operatorname{Spec.Tensor}\;\alpha\;s`.

The application-facing spelling for concrete dimensions is `Tensor α [dims...]`. The first
parameter is the scalar type; the second is the dimension list:

```
Tensor Float      [4, 2]  -- Lean `Float` values
Tensor ℝ          [4, 2]  -- mathematical real numbers
Tensor IEEE32Exec [4, 2]  -- explicit binary32 bit-level execution
```

Shape-polymorphic code may use a variable for the complete shape, as in `Tensor α s`. This is the
same tensor type, with `s` standing for the dimension list rather than writing its entries out.

Public tensor annotations use list syntax. Shape operations such as concatenation, axis lookup, and
size calculation preserve that representation in application code.

`[4, 2]` says only that the tensor has four rows and two columns. It does not carry or infer a
dtype. Lean obtains the element type from `Tensor`'s first argument, or infers it from the value on
the right-hand side when the annotation is omitted.

The trainer's scalar mode chooses the instantiated tensor element type and arithmetic semantics.
For example, `scalar := .float32` instantiates the model at Lean 4.33's native `Float32`; it does not
reinterpret an already constructed `Tensor Float [4, 2]`. Device storage is a further backend
contract: TorchLean's CUDA path stores native `Float32` model values as contiguous binary32 data.
Keeping shape, scalar semantics, and device representation distinct prevents a shape annotation
from silently deciding numerical behavior.

Shape literals are written from the outermost dimension to the innermost one:

```
import NN.API
open TorchLean

def vectorShape : Shape := [4]
def matrixShape : Shape := [3, 2]
def rankFourShape : Shape := [8, 3, 32, 32]
```

At the type level, `rankFourShape` is an ordinary rank-four tensor. Axis meanings come from the
operation or model: `[batch, channel, height, width]` and `[batch, height, width, channel]` are two
layout conventions over the same tensor type. The same tensor core can represent language tokens,
PDE grids, volumetric data, batched matrices, or an unusual scientific coordinate system.

# The Logical Representation

The specification tensor mirrors the recursive shape internally: a scalar stores one value, and
each tensor dimension is a total function from an in-bounds `Fin` index to the remaining tensor.
Ordinary model code does not construct that representation directly; list-shaped constructors and
tensor literals do it. The recursive definition matters in proofs because it gives us three facts
from the type:

- every axis has the length written in the shape;
- every legal index carries its own bounds proof;
- definitions and proofs can recurse over the shape and tensor together.

For example, `Tensor.map` changes the scalar type while preserving every dimension:

```
#check Tensor.map

-- Tensor.map : (α → β) → Tensor α s → Tensor β s
```

The result shape is fixed before any values are evaluated. The definition recurses through
`scalar` and `dim`, applies the supplied function only at scalar leaves, and therefore works at
every rank.

# Literals Prove Their Own Shape

The `tensor!` macro reads a rectangular nested literal and infers its shape:

```
def x : Tensor Float [2, 2] :=
  tensor! [[1.0, 2.0], [3.0, 4.0]]

def y : Tensor Float [2, 2] :=
  tensor! [[0.2, -0.1], [0.0, 0.3]]

def z : Tensor Float [2, 2] :=
  x + y
```

The literal is flattened in row-major order: the last index changes fastest. In this example the
flat order is `1, 2, 3, 4`.

Try either of these deliberate mistakes in a scratch Lean file:

```
def ragged :=
  tensor! [[1.0, 2.0], [3.0]]

def wrongAnnotation : Tensor Float [4] :=
  tensor! [[1.0, 2.0], [3.0, 4.0]]
```

The first literal is not rectangular. The second has four values but the wrong structure. Both are
rejected while Lean elaborates the file. The total element count alone is not enough to identify a
shape.

When the scalar type is ambiguous, make it explicit:

```
def q : Tensor Rat [2, 2] :=
  tensor! (ty := Rat) [[1, 2], [3, 4]]
```

One-dimensional literals use the same notation:

```
def v : Tensor Float [4] :=
  tensor! [0.0, 1.0, 2.0, 3.0]
```

When values arrive as a computed flat array rather than a literal, `Tensor.ofArray` checks that its
size agrees with the supplied dimensions before returning a typed tensor.

In scalar-polymorphic runtime code, `tensorF! cast dims values` authors constants as `Float` and
maps a supplied `Float → α` conversion over them. The conversion remains visible because scalar
semantics affect more than storage metadata.

# Indexing Is Total

Specification-level indices use `Fin`, so every index includes a proof that it lies inside its
axis. Indexing therefore does not return `Option` or throw an out-of-range exception. The result
type also records which axis was removed. Runtime APIs that receive an unchecked natural-number
index validate it at the boundary before constructing the corresponding typed operation.

This is useful in proofs: a theorem about a tensor with one or more dimensions introduces an
arbitrary valid index and applies its induction hypothesis to the smaller tensor selected there.

# Runtime Data Must Earn A Shape

A file or network payload arrives as bytes and runtime dimensions. Lean cannot know its shape
before reading it. The correct boundary is therefore a checked constructor:

```
def loadTensor4 (xs : Array Float) :
    Except String (Tensor Float [4]) :=
  Tensor.ofArray [4] xs
```

`Tensor.ofArray` checks that the array size equals the product of the dimensions. Only the success
branch returns the typed tensor.

For dimensions that are themselves known only at runtime:

```
def loadDynamic (dims : Array Nat) (xs : Array Float) :=
  Tensor.ofArray dims.toList xs
```

the result is still an ordinary `Tensor Float dims.toList`. Lean permits a result type to depend on
the runtime `dims` value. A caller may inspect those dimensions, establish that they equal the shape
required by a model, and then cast the tensor using that equality.

This is a recurring TorchLean pattern:

```
untyped external payload
  -> parser
  -> runtime validation
  -> typed Lean object
  -> theorem or model API
```

The check proves something about the accepted payload. It does not prove that every future file is
valid, and it does not certify the code that produced the file.

# Scalar Types Carry Semantics

These tensors have the same shape and different semantics:

:::table +header
*
  * Tensor element
  * Meaning
*
  * `Float`
  * executable host floating point
*
  * `Rat` or `ℚ`
  * executable exact rationals
*
  * `Real` or `ℝ`
  * proof-level exact reals
*
  * `TorchLean.Floats.IEEE32Exec`
  * executable bit-level binary32
*
  * `TorchLean.Floats.FP32`
  * rounded-real binary32-precision proof model; no upper exponent bound or IEEE special values
*
  * `TorchLean.Complex TorchLean.Floats.IEEE32Exec`
  * executable complex scalar with binary32 real and imaginary components
:::

The trainer's `scalar` field selects executable arithmetic for the run. Proofs instantiate tensors
over `ℝ` or `.fp32` directly; those noncomputable types do not appear as command-line runtime
choices. The high-level trainer currently rejects complex prediction because it has no public
host-`Float` readback path. The executable binary32 constructor is:

```
def x32 :
    Tensor TorchLean.Floats.IEEE32Exec [3] :=
  tensor32! [0.1, 0.2, 0.3]
```

The decimal source literal `0.1` is converted to a binary32 bit pattern. The floating-point chapter
explains why the printed decimal and the stored mathematical value are not identical.

# Scalar Types

`Tensor α s` is homogeneous: it cannot place an `Int`, FP16 value, and FP32 value in different
entries. Current model programs and `NN.IR.Semantics` likewise select one numeric `α` for learned
parameters, activations, and outputs. Scalar polymorphism means the same definition can be
interpreted again at another `α`. Integer indices and boolean masks use dedicated operation
interfaces so they are not silently treated as differentiable numeric tensors.

# Tensors And Arrays

TorchLean uses two ordinary containers for numerical data. A statically shaped value is a
`Tensor`; a runtime-sized homogeneous collection is an `Array`. In particular, a length-$`n`
value has type `Tensor α [n]`. It is not wrapped in Lean's `Vector` type, because the tensor shape
already records the length.

Lists still appear in types such as `Tensor α [batch, width]`: there they are shape syntax, not
storage. They also index dependent parameter packs whose tensors have different shapes. Runtime
samples, token buffers, coordinates, trainability flags, and collections of predictions use
arrays. This leaves one numerical representation at the specification boundary and one compact
container at dynamic boundaries.

# What The Runtime Stores

The pure tensor is also the CPU reference representation. Users still write and receive
`Tensor α shape`. A CPU eager node internally stores `SomeTensor α`, which pairs one runtime shape
with a `Tensor α shape`. This shape erasure lets one autograd tape contain activations and gradients
of different shapes in a single array. Recovering a typed tensor requires checking the stored
shape. The theorem `Spec.SomeTensor.materialize_eq` records that rebuilding the internal tensor
closure into its compact materialized form preserves the tensor exactly.

`TensorPack α shapes` solves a different problem. It is a statically heterogeneous tuple: the full
list of member shapes remains in its type. Model parameters, gradients, and typed graph contexts use
it when a fixed collection contains tensors of different shapes. It does not erase shapes, and it
is not an alternative input or output tensor type.

A model program does not pass those values around directly while it records a computation. It uses
`Runtime.ValueRef m α shape`, a shape-indexed handle owned by one session. Reading the handle
returns the corresponding tensor value. It has no meaning in another session and is not a second
tensor datatype. Session-backed runtime handles carry an owner token and recording generation;
operations reject handles from another session and handles retained across `resetTape`.

CUDA execution is the one place where the physical representation must differ. An `AnyBuffer`
contains a runtime shape and a native device buffer in contiguous row-major order. Upload and
download functions connect it to `Tensor`; CUDA tape operations validate the stored shape and buffer
size before dispatch. Proofs about graph evaluation and explicit kernel contracts cover the
operations for which TorchLean has established a bridge. The native CUDA compiler, driver, and
hardware remain named external boundaries rather than being treated as consequences of the tensor
definition.

# Linear Layers Preserve Prefix Dimensions

PyTorch's `Linear(in_features, out_features)` acts on the last axis. TorchLean follows that useful
convention while checking the complete map. The same layer:

```
nn.linear 2 8
```

can occur in:

```
[2]              -> [8]
[batch, 2]       -> [batch, 8]
[batch, time, 2] -> [batch, time, 8]
```

Any leading shape can serve as the prefix; it need not mean “batch” or “time.” One linear
definition therefore works for single examples, minibatches, sequences, and higher-rank
collections.

Try changing the second linear layer in the quickstart MLP from `nn.linear 8 1` to
`nn.linear 7 1` without changing the preceding layer. The sequential model no longer composes:
one layer produces a last dimension of eight while the next requires seven. The error appears when
the model is defined, before initialization, data loading, or training.

# Reshape Changes Structure, Not Data

A reshape from `[2,3]` to `[6]` is permitted because both shapes contain six values. It still needs
an explicit operation because the indexing interpretation changes. Conversely, reshaping `[2,3]`
to `[2,4]` is impossible because the element counts differ.

The layout convention matters at the representation boundary. In row-major order:

$$`\operatorname{flatIndex}(i,j)=3i+j`

for a $`2\times 3` matrix. A column-major native library would use a different equation. Shape equality
does not prove layout agreement, so backend capsules record layout requirements separately.

# Specification Tensors And Runtime Buffers

`Spec.Tensor` is a nested total function, a representation chosen for definitions and proofs.
Runtime CPU and CUDA code uses arrays, native storage, or device buffers. These are not competing
tensor systems; they are two representations with an explicit bridge.

There is one runtime detail worth knowing even when writing pure Lean code. Repeated functional
updates can build chains of closures. A long optimizer run that repeatedly asks for the newest
value may then spend more time walking old closures than doing arithmetic. `Tensor.materialize`
rebuilds the same mathematical tensor into an array-backed normal form at each dimension:

```
#check Spec.Tensor.materialize
#check Spec.Tensor.materialize_eq
```

The second declaration proves

$$`\operatorname{materialize}(t)=t.`

So materialization is not an approximation and does not change the tensor seen by a theorem. It is
a representation change used to keep repeated updates from accumulating runtime indirection. It
visits every scalar once, so its work is linear in `Shape.size s`; callers should place it at a
deliberate boundary rather than inside every small tensor operation.

The checked boundary is small. An array arriving from a file or native library becomes a tensor
only after its size is shown to equal the shape's element count:

```
def matrix : Tensor Float [2, 3] :=
  Tensor.ofFlatArrayExact [2, 3]
    #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    (by decide)
```

From that point onward, model and specification code use `Tensor`. Calling `matrix.toArray`
materializes row-major storage when serialization or a kernel boundary needs it. The array does not
carry a second numerical semantics.

# Cost And Representation Notes

The shape indices remove ambiguity, but they do not erase the cost of an operation. The useful
cost model is:

:::table +header
*
  * Operation
  * Specification view
  * Array or native view
*
  * index a rank-$`r` tensor
  * apply one finite function per axis
  * compute a row-major offset, then read the buffer
*
  * map
  * visit every scalar and preserve the shape
  * one pass over contiguous storage
*
  * reshape
  * prove equal element counts and reinterpret index structure
  * retain or rebuild storage according to the runtime path
*
  * materialize
  * extensionally the identity
  * one traversal that removes accumulated closure chains
*
  * matrix multiplication
  * the mathematical sum declared by the spec
  * a loop nest, CUDA kernel, cuBLAS call, or another accepted capsule
:::

Big-O notation alone cannot settle provider agreement. Two matrix multiplications may both take
$`O(mnk)` arithmetic operations while accumulating in different orders and returning different
Float32 bits. The graph and backend chapters keep the operation, provider, and numerical contract
separate for this reason.

# Common Tensor Declarations

These are the names I reach for most often when reading or writing a small example:

:::table +header
*
  * Declaration
  * Use
*
  * `[d₀, ..., dₙ]`
  * build a shape known while Lean elaborates the file
*
  * `tensor!`
  * construct a rectangular nested literal and infer its shape
*
  * `tensorOfArray!`
  * construct a statically shaped tensor from a flat literal
*
  * `Tensor.ofArray`
  * check runtime data against a requested static shape
*
  * `Tensor.ofArray dims xs`
  * construct the same `Tensor` type when `dims` is known only at runtime
*
  * `Spec.Tensor.castShape`
  * transport a tensor along a proved equality of shapes
*
  * `Spec.Tensor.materialize`
  * normalize the pure representation without changing its value
*
  * `Tensor.ofFlatArrayExact`
  * check flat storage against a shape and construct the canonical tensor
*
  * `Tensor.toArray`
  * materialize row-major storage for serialization or native execution
:::

The generated API reference gives the complete signatures. This table is the smaller working set
used by the examples in this guide.

# Inspect Tensors In The Lean Infoview

Open
[`NN/Examples/DeepDives/Tensors/Basic.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/DeepDives/Tensors/Basic.lean)
in VS Code with the Lean extension. Place the cursor on:

```
#tensor_view matrix
#tensor_stats_view matrix
```

The first widget renders a tensor; the second summarizes its values. Move the cursor to
`#tensor_view firstRow` to see the shape change after indexing.

# What Shape Safety Proves

If a model accepts `Tensor α inputShape`, Lean checks that every statically represented layer
composes and that the final result has the declared output shape. It can also check that a parsed
runtime payload has the length promised by its dimensions.

Shape safety does not, by itself, prove:

- that external memory uses the expected row-major layout;
- that two axes have the intended domain meaning;
- that a CUDA kernel wrote within bounds;
- that an arithmetic operation is numerically correct;
- that training converges.

Those are separate contracts. Keeping them separate is stronger than calling a tensor “safe”
without saying which property was established.

# Continue With Models

The next chapter turns shape maps into layers and model architectures. The most useful sources to
keep nearby are:

- [NN/Tensor.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/Tensor.lean) for the public
  tensor surface and its focused constructor, syntax, printing, and operation modules;
- [NN/Spec/Core/Tensor.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/Core/Tensor.lean)
  for the recursive specification representation;
- [NN/Proofs/Tensor/Basic.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Tensor/Basic.lean)
  for flattening, unflattening, and algebraic laws.
