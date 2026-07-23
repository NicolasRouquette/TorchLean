import VersoManual

open Verso.Genre Manual

#doc (Manual) "Tensors That Remember Their Shapes" =>
%%%
tag := "tensors-shapes"
%%%

Most tensor libraries carry a shape beside a buffer and check compatibility when an operation
runs. TorchLean also carries the shape in the Lean type. A function that accepts a length-four
vector cannot accidentally receive a `2 × 2` matrix, even though both contain four scalar values.

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
[IEEE32Exec] [0.100000, 0.200000, 0.300000, 0.400000]
[Float] [[[1.000000, 2.000000], [3.000000, 4.000000]],
         [[5.000000, 6.000000], [7.000000, 8.000000]]]
Expected failure printing Tensor ℝ:
  Refusing to print `Tensor ℝ` (proof-level);
  cast to `Float`/`IEEE32Exec`/`ℚ` to display.
```

The first four lines have the same shape but different scalar meanings. `ℚ` displays exact
fractions. `IEEE32Exec` stores and executes explicit binary32 bit patterns. `Float` uses Lean's host
runtime. The final attempted tensor over `ℝ` is a mathematical object; arbitrary real numbers are
not executable data, so printing it is rejected rather than pretending to approximate it.

The source is
[`TensorBasics.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/TensorBasics.lean).
Keep it open while reading this chapter: every definition below is a small variation of that file.

# One Tensor Type

The canonical specification type is:

$$`\operatorname{Spec.Tensor}\;\alpha\;s`.

The application-facing spelling is `Tensor.T α s`. The first parameter is the scalar type and the second is the
shape. A shape is recursively built from:

```
inductive Shape
  | scalar
  | dim (size : Nat) (rest : Shape)
```

Thus a matrix of shape `[3,2]` is represented by:

$$`
\operatorname{dim}(3,\operatorname{dim}(2,\operatorname{scalar})).
`

The `shape!` macro gives the familiar notation:

```
import NN.API
open TorchLean

def scalarShape : Shape := .scalar
def vectorShape : Shape := shape![4]
def matrixShape : Shape := shape![3, 2]
def rankFourShape : Shape := shape![8, 3, 32, 32]
```

At the type level, `rankFourShape` is an ordinary rank-four tensor. NCHW, NHWC, sequence, and
feature conventions come from operations and models rather than separate tensor datatypes. The
same tensor core can therefore represent language tokens, PDE grids, volumetric data, batched
matrices, or an unusual scientific coordinate system.

# Literals Prove Their Own Shape

The `tensor!` macro reads a rectangular nested literal and infers its shape:

```
def x : Tensor.T Float (shape![2, 2]) :=
  tensor! [[1.0, 2.0], [3.0, 4.0]]

def y : Tensor.T Float (shape![2, 2]) :=
  tensor! [[0.2, -0.1], [0.0, 0.3]]

def z : Tensor.T Float (shape![2, 2]) :=
  x + y
```

The literal is flattened in row-major order: the last index changes fastest. In this example the
flat order is `1, 2, 3, 4`.

Try either of these deliberate mistakes in a scratch Lean file:

```
def ragged :=
  tensor! [[1.0, 2.0], [3.0]]

def wrongAnnotation : Tensor.T Float (shape![4]) :=
  tensor! [[1.0, 2.0], [3.0, 4.0]]
```

The first literal is not rectangular. The second has four values but the wrong structure. Both are
rejected while Lean elaborates the file. The total element count alone is not enough to identify a
shape.

When the scalar type is ambiguous, make it explicit:

```
def q : Tensor.T Rat (shape![2, 2]) :=
  tensor! (ty := Rat) [[1, 2], [3, 4]]
```

For a flat literal, `tensorOfList!` proves the length obligation:

```
def v : Tensor.T Float (shape![4]) :=
  tensorOfList! [4] [0.0, 1.0, 2.0, 3.0]
```

In scalar-polymorphic runtime code, `tensorF! cast dims values` authors constants as `Float` and
maps a supplied `Float → α` conversion over them. The conversion remains visible because scalar
semantics affect more than storage metadata.

# Indexing Is Total

The recursive representation of a tensor mirrors its shape:

```
Tensor α (.dim n rest)
```

contains a function from `Fin n` to `Tensor α rest`. An index has both a natural number and a proof
that it is smaller than the dimension. Consequently, indexing does not return `Option α` and does
not throw an out-of-range exception: an invalid index cannot be constructed without supplying a
false proof.

For the matrix above, the first row can be written:

```
def firstRow : Tensor.T Float (shape![2]) :=
  match x with
  | .dim rows => rows ⟨0, by decide⟩
```

The output shape is visible in the function type. Indexing once removed the outer dimension; it did
not flatten or reinterpret the remaining data.

This representation is particularly pleasant in proofs. A theorem about a rank-`n+1` tensor can
introduce an arbitrary `i : Fin size` and apply the induction hypothesis to the smaller tensor at
that index.

# Runtime Data Must Earn A Shape

A file or network payload arrives as bytes and runtime dimensions. Lean cannot know its shape
before reading it. The correct boundary is therefore a checked constructor:

```
def loadVector4 (xs : List Float) :
    Except String (Tensor.T Float (shape![4])) :=
  Tensor.ofList [4] xs
```

`Tensor.ofList` checks that the list length equals the product of the dimensions. Only the success
branch returns the typed tensor.

For dimensions that are themselves known only at runtime:

```
def loadDynamic (dims : List Nat) (xs : List Float) :=
  NN.Tensor.dynamicOfList dims xs
```

the result packages an existential shape together with the corresponding tensor. A caller may
inspect the dimensions, establish that they equal the shape required by a model, and then cross
into the statically typed API.

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

# Scalar Type Is More Than A DType Label

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
  * `TorchLean.Floats.F32 .ieee754Exec`
  * executable bit-level binary32
*
  * `TorchLean.Floats.F32 .fp32`
  * rounded-real binary32-precision proof model; no upper exponent bound or IEEE special values
*
  * `TorchLean.Complex (TorchLean.Floats.F32 .ieee754Exec)`
  * executable complex scalar with binary32 real and imaginary components
:::

The trainer has a `dtype` option, but it is selecting a scalar interpretation, not adding a
decorative field to an untyped buffer. Executable trainer paths reject `.real` and the
noncomputable `.fp32` proof mode; the high-level trainer also currently rejects complex execution
because prediction has no host-`Float` readback path. The executable binary32 constructor is:

```
def x32 :
    Tensor.T (TorchLean.Floats.F32 .ieee754Exec) (shape![3]) :=
  tensor32! [0.1, 0.2, 0.3]
```

The decimal source literal `0.1` is converted to a binary32 bit pattern. The floating-point chapter
explains why the printed decimal and the stored mathematical value are not identical.

# One Scalar Type Per Tensor And Run

`Tensor.T α s` is homogeneous: it cannot place an `Int`, FP16 value, and FP32 value in different
entries. Current model programs and `NN.IR.Semantics` likewise select one numeric `α` for learned
parameters, activations, and outputs. Scalar polymorphism means the same definition can be run
again at another `α`; it does not provide per-node promotion or autocast.

The executable complex choice is also one scalar type, not a mixed pair of tensor dtypes. Integer
indices and boolean masks have dedicated operation interfaces, but a PyTorch-style graph mixing
FP8/BF16/FP16/FP32 numeric tensors will need an explicitly dtype-indexed graph and cast rules that
the current API does not yet claim to implement.

# Linear Layers Preserve Prefix Dimensions

PyTorch's `Linear(in_features, out_features)` acts on the last axis. TorchLean follows that useful
convention while checking the complete map. The same layer:

```
nn.linear 2 8
```

can occur in:

```
shape![2]              -> shape![8]
shape![batch, 2]       -> shape![batch, 8]
shape![batch, time, 2] -> shape![batch, time, 8]
```

Any leading shape can serve as the prefix; it need not mean “batch” or “time.” One linear
definition therefore works for single examples, minibatches, sequences, and higher-rank
collections.

Now try changing the second linear layer in the quickstart MLP from `nn.linear 8 1` to
`nn.linear 7 1` without changing the preceding layer. The sequential model no longer composes:
one layer produces a last dimension of eight while the next requires seven. The error appears when
the model is defined, before initialization, data loading, or training.

# Reshape Changes Structure, Not Data

A reshape from `[2,3]` to `[6]` is permitted because both shapes contain six values. It still needs
an explicit operation because the indexing interpretation changes. Conversely, reshaping `[2,3]`
to `[2,4]` is impossible because the element counts differ.

The layout convention matters at the representation boundary. In row-major order:

$$`\operatorname{flatIndex}(i,j)=3i+j`

for a `2 × 3` matrix. A column-major native library would use a different equation. Shape equality
does not prove layout agreement, so backend capsules record layout requirements separately.

# Specification Tensors And Runtime Buffers

`Spec.Tensor` is a nested total function, a representation chosen for definitions and proofs.
Runtime CPU and CUDA code uses arrays, native storage, or device buffers. These are not competing
tensor systems; they are two representations with an explicit bridge.

The deep-dive file
[`Tensors/Basic.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/DeepDives/Tensors/Basic.lean)
shows both:

```
def matrixArray : TensorArray.Tensor Float [2, 3] :=
  TensorArray.ofArray
    #[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    [2, 3]
    (by simp)

def matrixSpec : Spec.Tensor Float (listToShape [2, 3]) :=
  toTensor matrixArray
```

`TensorArray` makes row-major storage explicit. `Spec.Tensor` makes shape recursion explicit.
Conversion theorems and runtime checks connect them.

# Inspect Tensors In The Lean Infoview

Open
[`NN/Examples/DeepDives/Tensors/Basic.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/DeepDives/Tensors/Basic.lean)
in VS Code with the Lean extension. Place the cursor on:

```
#tensor_view matrixSpecView
#tensor_stats_view matrixSpecView
```

The first widget renders the matrix; the second summarizes its values. Move the cursor to
`firstRowSpecView` to see the shape change after indexing. The widget declarations are `meta`
because the editor evaluates them for display. Ordinary model code continues to use plain `def`.

# What Shape Safety Proves

If a model accepts `Tensor.T α inputShape`, Lean checks that every statically represented layer
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

- [NN/Tensor/API.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/Tensor/API.lean) for
  constructors and operations;
- [NN/Spec/Core/Tensor.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/Core/Tensor.lean)
  for the recursive specification representation;
- [NN/Proofs/Tensor/Basic.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Tensor/Basic.lean)
  for flattening, unflattening, and algebraic laws.
