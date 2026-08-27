import VersoManual

open Verso.Genre Manual

#doc (Manual) "The Mathematical Specification" =>
%%%
tag := "spec-layer"
%%%

The model we trained earlier can be written on one line:

$$`F_\theta(x)=W_2\,\operatorname{ReLU}(W_1x+b_1)+b_2.`

That formula is the mathematical center of the example. It says nothing about mutable buffers,
CUDA launches, PyTorch modules, reverse-mode tapes, or JSON files. It does say exactly how the four
parameter tensors and the input determine the output. TorchLean's `NN.Spec` library is where such
formulas live as Lean definitions.

The distinction is important because executable ML code changes form constantly. A linear layer
may be evaluated by a nested Lean function, a CPU loop, cuBLAS, or an ATen kernel. The specification
gives all of those implementations one statement to satisfy.

# The Tensor Behind The Formula

A specification tensor is defined recursively from its shape:

- a scalar shape stores one value;
- a dimension of length `n` stores a function from `Fin n` to a smaller tensor.

Thus a vector of length two is, mathematically, two scalar values indexed by `Fin 2`; a matrix of
shape `[3, 2]` is three such vectors. The index type prevents an out-of-range lookup. It also makes
shape induction natural: a proof about an arbitrary tensor can follow the same scalar-or-dimension
recursion as the datatype.

This is not a claim that CUDA stores a matrix as nested Lean functions. Native runtimes use flat,
contiguous buffers. The specification chooses the representation that makes mathematical reasoning
clear; a layout contract is needed when an implementation flattens that value into memory.

Model code uses ordinary shape lists, so the recursive representation stays out of tensor types:

```
import NN.Spec.Layers.Linear
import NN.Spec.Layers.Activation

open Spec
open Spec.Tensor

#check Tensor ℝ [2]
#check Tensor ℝ [3, 2]
#check LinearSpec ℝ 2 3
```

For a linear map from two inputs to three outputs,
`LinearSpec ℝ 2 3` contains

$$`W\in\mathbb{R}^{3\times 2},
\qquad b\in\mathbb{R}^{3}.`

The order is the same convention used by PyTorch's `nn.Linear`: output features index the rows of
the weight matrix.

# Algebra Comes From The Scalar Context

The shape fixes where values live. The scalar context fixes which formulas are available. A linear
layer needs zero, addition, and multiplication. Softmax also needs exponentiation and division.
Square root, comparison, trigonometric operations, and Boolean decisions require their own pieces
of structure.

TorchLean packages the common operations in `Context α`, so the same tensor definition can be
interpreted over reals, rounded formats, interval endpoints, or executable binary32. This
polymorphism is useful only when the hypotheses remain honest. A proof of a softmax identity over
`ℝ` may use properties of the real exponential that an arbitrary `Context α` does not provide.

The usual progression is:

:::table +header
*
  * Level
  * Typical statement
*
  * generic scalar context
  * the operation is well shaped and computes the declared recursion
*
  * algebraic structure
  * addition, multiplication, linearity, or order laws
*
  * real analysis
  * derivatives, continuity, convexity, and analytic bounds
*
  * rounded arithmetic
  * representability, rounding error, and finite-path conditions
*
  * native execution
  * agreement with one selected provider and reduction policy
:::

Moving down this table adds hypotheses. A theorem should not inherit a stronger numerical meaning
because the same operation name appears at every level.

# One Layer, Forward And Backward

The forward definition is deliberately short:

```
def linearSpec {α : Type} [Add α] [Mul α] [Zero α]
    {inDim outDim : Nat}
    (layer : LinearSpec α inDim outDim)
    (x : Tensor α [inDim]) :
    Tensor α [outDim] :=
  addSpec (matVecMulSpec layer.weights x) layer.bias
```

At coordinate $`i`, this means

$$`y_i=b_i+\sum_{j=0}^{\mathrm{inDim}-1}W_{ij}x_j.`

The backward specification receives an upstream cotangent
$`g=\partial L/\partial y` and returns

$$`
\frac{\partial L}{\partial W}=g\,x^\mathsf{T},\qquad
\frac{\partial L}{\partial b}=g,\qquad
\frac{\partial L}{\partial x}=W^\mathsf{T}g.
`

In Lean these three tensors have shapes `[outDim, inDim]`, `[outDim]`, and `[inDim]`. Returning a
transposed weight gradient by accident is therefore a type error rather than a numerically plausible
array.

The source definitions are
[`linearSpec`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Spec/Layers/Linear.lean)
and `linearBackwardSpec` in the same file. Later, the autograd proofs compare executable VJP rules
with these definitions. The existence of the definitions alone does not prove the comparison.

# The Complete MLP Meaning

The running model is the composition of two linear specifications and a pointwise activation:

```
def twoLayerMlp
    {α : Type} [Context α]
    {inDim hidden outDim : Nat}
    (first : LinearSpec α inDim hidden)
    (second : LinearSpec α hidden outDim)
    (x : Tensor α [inDim]) :
    Tensor α [outDim] :=
  linearSpec second
    (Activation.reluSpec (linearSpec first x))
```

Expanding this definition gives the formula at the start of the chapter. No graph traversal or
runtime state is hidden inside it.

The ReLU definition commits to two choices:

$$`\operatorname{ReLU}(z)=\max(z,0),`

and for the selected derivative,

$$`\operatorname{ReLU}'(z)=
  \begin{cases}
    1 & z>0,\\
    0 & z\le 0.
  \end{cases}`

The value at the kink matters. Other subgradients are mathematically defensible, but a forward and
backward correctness theorem needs one concrete rule. TorchLean chooses zero at the kink, matching
the rule used by the current runtime path.

# Run The Same Formula

The quickest executable using this architecture is:

```
lake exe torchlean quickstart_mlp --device cpu --steps 200 --seed 2026
```

On the current example dataset it reports:

```
dataset size = 25
mean_loss(before) = 0.761530
mean_loss(after) = 0.003234
heldout x=(0.25,-0.75), target=0.2, prediction(after)=[0.210239]
```

The command executes runtime tensors and an autograd tape; it does not evaluate `twoLayerMlp` by
reducing the pure Lean definition above. The connection is made operation by operation:

1. the `nn.linear` builder fixes the same parameter shapes;
2. its forward program emits the runtime linear operation;
3. the graph interpreter assigns `.linear` the `linearSpec` denotation;
4. the VJP proof identifies the selected backward rule with `linearBackwardSpec`;
5. a backend capsule records which native provider, if any, executed the operation.

This chain is why a specification is useful. It gives each bridge a stable target.

# Change One Value

The formula can be inspected without training. Take

$$`
W_1=\begin{bmatrix}1&1\\-1&1\end{bmatrix},
\quad b_1=0,\quad
W_2=\begin{bmatrix}0.8&-0.4\end{bmatrix},
\quad b_2=0.2.
`

For $`x=(0.25,-0.75)`,

$$`
W_1x=(-0.5,-1),\qquad
\operatorname{ReLU}(W_1x)=(0,0),
`

so the exact-real output is $`0.2`. Change only the first bias to $`0.6`. The first hidden
preactivation becomes $`0.1`, and the output becomes

$$`0.8(0.1)+0.2=0.28.`

This tiny calculation is the same kind of reasoning used in interval propagation: replace one
point by a set of possible inputs, then bound every intermediate tensor.

# Scalar Polymorphism Is A Real Choice

The type parameter $`\alpha` determines what the symbols `+`, `*`, `max`, `exp`, and division mean.
TorchLean reuses the tensor structure at several scalar interpretations:

:::table +header
*
  * Scalar
  * Meaning
*
  * `ℝ`
  * exact real arithmetic used for mathematical statements
*
  * `FP32`
  * binary32-precision, gradual-underflow rounded-real values with no upper exponent cutoff
*
  * `IEEE32Exec`
  * executable binary32 bit patterns, including signed zero, infinity, and NaN
*
  * interval contexts
  * sets of possible values, with outward enclosure operations
*
  * runtime `Float`
  * Lean's native executable floating-point value
:::

Writing one polymorphic definition is not a proof that these interpretations agree. For example,

$$`\operatorname{softmax}(x)_i=
\frac{\exp(x_i-m)}{\sum_j\exp(x_j-m)},\qquad m=\max_jx_j`

is a clean real-valued formula. A binary32 implementation introduces rounding in `max`,
subtraction, exponential approximation, summation, and division. A CUDA reduction may also choose
a different summation tree. The later floating-point and runtime-approximation chapters state the
conditions under which one interpretation encloses or approximates another.

# The Implemented Specifications

The spec layer is broader than the running MLP. `NN.Spec.Layers` contains typed meanings for
linear algebra, rank-polymorphic convolution, transposed convolution, and pooling,
activations and losses, normalization, dropout, embeddings, recurrent cells, selective scan, and
scaled dot-product attention. Model definitions under `NN.Spec.Models` compose these operations
into families such as CNNs, transformers, recurrent networks, and state-space models.

That inventory is a semantics inventory, not a runtime support matrix. An operation can have a pure
definition before it has an eager tape rule, typed graph lowering, a CUDA kernel, or an end-to-end
correctness theorem. The relevant runtime and lowering chapters name those smaller supported
fragments explicitly.

# Shape Indices And Scalar Types

The shape index $`s` and scalar parameter $`\alpha` answer different questions. `Tensor α s` fixes both for
one value, and `Graph.denote` follows the homogeneous-scalar contract from *Tensors And Shapes*.
The later IR chapter uses “heterogeneous” only for a table of differently shaped values, not for a
mixed-dtype graph.

# Conventions That Must Be In The Definition

Shapes are only one source of ambiguity. The spec layer also fixes choices that a model name does
not determine.

## Loss reductions

`mseSpec` takes a global mean over all entries. Cross-entropy over logits applies log-softmax along
the selected class dimension and averages over the remaining slices. Changing `mean` to `sum`
changes both the loss and every gradient by a scale factor.

## Attention masks

A boolean attention mask is a hard support constraint:

- `true` allows the key;
- `false` gives the key exactly zero softmax numerator;
- if every key in a row is blocked, the output row is zero.

This is the finite formulation of a negative-infinity mask. It is not an additive score bias of
$`-1000` or any other finite sentinel. For sufficiently large logits a finite sentinel can leak
nonzero probability into a blocked position; the hard-mask definition cannot.

## Dropout

Randomness is explicit. A masked dropout specification receives the mask as an argument.
Runtime training code may generate that mask from a seed and tape state, but the semantic function
does not consult hidden global randomness.

## Invalid windows

Some mathematical operations are total where the runtime is partial. A convolution output extent
is normally

$$`\left\lfloor\frac{n+2p-k}{s}\right\rfloor+1.`

The shape helper defines an answer even around invalid kernel or stride configurations, while the
runtime validator rejects unsupported calls before reaching native code. A total denotation and an
admission check answer different questions.

# A Proof Checkpoint

GraphSpec's checked MLP uses the same four parameter tensors:

```
[W₁ : [hidden, input],
 b₁ : [hidden],
 W₂ : [output, hidden],
 b₂ : [output]]
```

The theorem
`NN.GraphSpec.Models.mlp_interp_eq_spec_mlp_forward` proves that interpreting that GraphSpec model
is equal to the hand-written two-layer specification. Its conclusion is about two pure meanings.
It does not mention the eager tape, CUDA, or an imported checkpoint, so it should not be reported as
a proof of those objects.

That theorem supplies the first exact link in a longer chain and prevents later engineering layers
from silently changing what “the MLP” means.

The next chapter turns these formulas into typed architectures with explicit parameter layouts.
