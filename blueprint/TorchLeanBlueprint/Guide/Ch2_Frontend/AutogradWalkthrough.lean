import VersoManual

open Verso.Genre Manual

#doc (Manual) "Differentiation By Example" =>
%%%
tag := "autograd-walkthrough"
%%%

Reverse-mode automatic differentiation is usually introduced as “call backward and read the
gradients.” TorchLean makes the derivative object explicit: a function, input, and output
cotangent determine a vector-Jacobian product. Parameter gradients are returned in a pack whose
shapes match the model's parameters.

We will begin with a two-coordinate function, then differentiate a linear model, stop a gradient
with `detach`, and finish with Jacobian and Hessian products.

# Run The Complete Tour

```
lake exe torchlean quickstart_autograd
```

The run prints several derivative views. The final portion is:

```
jacfwdInput(square) cols = 2
  col[0] = [1.000000, -0.000000]
  col[1] = [0.000000, -2.400000]

hessianInput(mean(x^2)) cols = 2
  H*e[0] = [1.000000, 0.000000]
  H*e[1] = [0.000000, 1.000000]

vjp(square, seed=ones) = [1.000000, -2.400000]
grad1(mean(x^2)) = [0.500000, -1.200000]
valueAndGradScalar(mean(x^2))
  value = 0.845000
  grad = [0.500000, -1.200000]
```

We will derive each value below. The source is
[`AutogradBasics.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/AutogradBasics.lean).

# A Scalar Function

Define:

$$`
f(x_0,x_1)=\frac{x_0^2+x_1^2}{2}.
`

In the `autograd.func` API:

```
import NN.API
open TorchLean

def sumsq :
    autograd.func.TensorFunction [2] ([] : List Nat) :=
  fun x => do
    let y ← nn.functional.square x
    nn.functional.mean y
```

The type states that `sumsq` maps a two-vector to a scalar. The `do` block builds a checked tensor
program; it is not using a special untyped scalar expression language.

The mathematical gradient is:

$$`
\nabla f(x_0,x_1)=(x_0,x_1).
`

At $`x=(0.5,-1.2)`:

$$`
f(x)=\frac{0.25+1.44}{2}=0.845,
\qquad
\nabla f(x)=(0.5,-1.2).
`

The executable call is:

```
def example : IO Unit := do
  let x : Tensor Float [2] :=
    tensor! [0.5, -1.2]
  let (value, grad) ←
    autograd.func.valueAndGradScalar
      (alpha := Float) sumsq x
  IO.println s!"value={value}, grad={Tensor.pretty grad}"
```

The observed values match the hand calculation. That is a useful executable check. A theorem that
the autograd transform is correct for every input requires the semantic proof layer described
later; one agreeing sample is not that theorem.

At a nonsmooth point, “the derivative” also needs a convention. ReLU at zero, max ties, and similar
operations have runtime VJP rules even where a classical Fréchet derivative does not exist. A
correctness theorem must either exclude those points or state the selected generalized derivative;
shape-correct gradient output alone does not settle that question.

# Why A VJP Is The Primitive Reverse Operation

Let:

$$`f:\mathbb R^n\to\mathbb R^m`

with Jacobian:

$$`J_f(x)\in\mathbb R^{m\times n}`.

Reverse mode accepts an output cotangent $`\bar y\in\mathbb R^m` and returns:

$$`\bar x=J_f(x)^\mathsf{T}\bar y`.

TorchLean writes:

```
let dx ←
  autograd.func.vjp
    (alpha := Float)
    f x seedOut
```

The types enforce:

```
x       : Tensor α inputShape
seedOut : Tensor α outputShape
dx      : Tensor α inputShape
```

For a scalar output, choosing seed one gives the ordinary gradient. For a vector output, the seed
selects a linear combination of output rows without materializing the whole Jacobian.

# Elementwise Square

Let:

$$`g(x_0,x_1)=(x_0^2,x_1^2)`.

Its Jacobian is diagonal:

$$`
J_g(x)=
\begin{bmatrix}
2x_0&0\\
0&2x_1
\end{bmatrix}.
`

The TorchLean definition and VJP are:

```
def square :
    autograd.func.TensorFunction [2] [2] :=
  fun x => nn.functional.square x

def vjpExample : IO Unit := do
  let x : Tensor Float [2] :=
    tensor! [0.5, -1.2]
  let seed : Tensor Float [2] :=
    tensor! [1.0, 1.0]
  let dx ←
    autograd.func.vjp
      (alpha := Float) square x seed
  IO.println s!"dx={Tensor.pretty dx}"
```

The result is:

$$`
J_g(x)^\mathsf T(1,1)=(1,-2.4).
`

Change the seed to $`(1,10)`. The expected result becomes $`(1,-24)`. This is a clean way to see that
a VJP is not “the gradient” of a vector-valued function until an output cotangent is chosen.

# Full Jacobians

TorchLean provides:

- `jacfwd`, one output-shaped column per input coordinate;
- `jacrev`, one input-shaped row per output coordinate.

For `square` at $`(0.5,-1.2)`, the quickstart prints:

```
col[0] = [1.000000, -0.000000]
col[1] = [0.000000, -2.400000]
```

These are the columns of the diagonal Jacobian. Reverse mode prints the same entries organized by
rows.

The result is an array of shaped tensors rather than one anonymous flat matrix. The order follows
the row-major flattened coordinate order of the relevant shape. This preserves enough structure to
relate a derivative coordinate back to a tensor axis.

Forward mode is attractive when the input dimension is small; reverse mode is attractive when the
output dimension is small. Building the full Jacobian requires repeating one of these products for
each basis direction.

# Parameter VJPs

Consider a linear map with three outputs and two inputs:

$$`y=Wx+b`,

where:

$$`W\in\mathbb R^{3\times2},\qquad b\in\mathbb R^3`.

For an output cotangent $`\bar y`, the derivatives are:

$$`
\bar W=\bar y\,x^\mathsf T,
\qquad
\bar b=\bar y,
\qquad
\bar x=W^\mathsf T\bar y.
`

With $`x=(0.5,-1.2)` and $`\bar y=(1,1,1)`, each row of `dW` is $`x` and each bias derivative is
one. The quickstart prints:

```
vjpOutParams (seed=ones) dW =
  [[0.500000, -1.200000],
   [0.500000, -1.200000],
   [0.500000, -1.200000]]
vjpOutParams (seed=ones) db =
  [1.000000, 1.000000, 1.000000]
```

The model API returns a dependent parameter pack:

```
let dState ←
  autograd.model.vjpState
    (alpha := Float)
    model params x seedOut
```

`dState` has exactly the shapes and order of `params`. There is no mutable `.grad` field that might
contain stale values from a previous reverse pass.

# A Loss Turns Output Derivatives Into Parameter Gradients

For a model output and target of the same shape:

```
def loss :
    autograd.model.OutputLoss outputShape outputShape :=
  autograd.model.OutputLoss.mse

let (lossValue, dState) ←
  autograd.model.valueAndGradStateScalar
    (alpha := Float)
    model loss params x target
```

Conceptually:

$$`
\nabla_\theta L
=
J_{F_\theta}^{\theta}(x)^\mathsf T
\nabla_{\widehat y}L(\widehat y,y).
`

The quickstart's concrete result is:

```
loss(mse) = 0.165133
gradState (mse) gW =
  [[-0.156667, 0.376000],
   [-0.160000, 0.384000],
   [0.070000, -0.168000]]
gradState (mse) gb =
  [-0.313333, -0.320000, 0.140000]
```

An optimizer consumes this pack together with parameters and its own state.

# `detach` Preserves Values And Cuts Derivatives

`autograd.model.OutputLoss.detach` leaves the forward value unchanged while replacing the backward
map by zero.

The quickstart checks both effects:

```
loss(mse) = 0.165133
loss(mse ∘ detach) = 0.165133

gradState (mse ∘ detach) gW =
  [[0.000000, 0.000000],
   [0.000000, 0.000000],
   [0.000000, 0.000000]]
gradState (mse ∘ detach) gb =
  [0.000000, 0.000000, 0.000000]
```

This is a good diagnostic experiment: if the loss value changes, the forward implementation of
detach is wrong; if a gradient remains nonzero, the backward edge was not cut.

Detach is useful for target networks, stop-gradient estimators, contrastive objectives, and
statistics that should not receive gradients. It also changes the mathematical objective seen by
the optimizer, so it should never be inserted merely to make an error disappear.

# JVPs And Hessian-Vector Products

Forward mode computes:

$$`J_f(x)v`.

TorchLean exposes JVPs for functions and model parameters. Combining forward and reverse
differentiation gives a Hessian-vector product:

$$`H_f(x)v`

without constructing the full Hessian.

For:

$$`f(x)=\frac{x_0^2+x_1^2}{2}`,

the Hessian is the identity matrix:

$$`H_f(x)=I_2`.

The quickstart therefore prints:

```
H*e[0] = [1.000000, 0.000000]
H*e[1] = [0.000000, 1.000000]
```

Change the function from `mean(square x)` to `sum(square x)`. The Hessian should become `2I`.
This variation checks both the reduction convention and second derivative.

# Local Rules And The Global Reverse Pass

At runtime, each primitive operation contributes:

1. a forward value;
2. parent references;
3. a local VJP rule.

Reverse traversal starts from an output seed, applies each local rule in reverse topological order,
and adds cotangents when several paths meet. For:

$$`z=x^2+x^2`,

the two paths each contribute $`2x`, so accumulation returns $`4x`. A tape that overwrote one parent
contribution instead of adding would be wrong even if the local square rule were correct.

This local-to-global split also structures the proofs. Primitive derivative theorems establish the
local rules; tape soundness establishes that reverse accumulation composes them according to the
graph.

# Runtime Derivatives And Mathematical Derivatives

There are three claims to distinguish:

1. the calculus derivative of the ideal real operation;
2. the derivative program generated by TorchLean's graph/autograd transform;
3. the numeric values produced by CPU, CUDA, or an external VJP provider.

The first two can be related by Lean theorems for supported primitives and well-formed graphs. The
third additionally needs a runtime refinement or an explicit backend boundary. A CUDA kernel may
use floating-point rounding and a different reduction tree even when it implements the same formal
VJP equation.

The `autograd.func` helpers execute the lowered derivative graph directly. Trainer
eager/typed-graph selection and native capsule selection are separate runtime choices. Primals,
cotangents, parameters, and returned derivatives follow the scalar contract described in
*Tensors And Shapes*.

# Inspect The Tape In VS Code

Open the widget deep dive and place the cursor on:

```
#tape_view ...
#tape_grads_view ...
#tape_trace_view ...
```

The views show nodes, parent edges, accumulated gradients, and reverse traversal. Use them to answer:

- which operations are on the path to the scalar output?
- where do cotangents merge?
- which leaf corresponds to each parameter?
- did a detached branch disappear from reverse traversal?

Widgets visualize an executable artifact. They do not add a theorem to it.

# API Summary

The derivative APIs differ in what they seed and what they return:

:::table +header
*
  * Query
  * Seed
  * Result
  * Best use
*
  * `grad`
  * scalar output cotangent `1`
  * one gradient with the input shape
  * scalar objectives
*
  * `valueAndGradScalar`
  * scalar output cotangent `1`
  * primal value and gradient
  * losses that should be evaluated only once
*
  * `vjp`
  * caller-supplied output cotangent
  * pullback into the input shape
  * vector-valued functions
*
  * `jacfwd`
  * input basis tangents
  * full Jacobian
  * small input dimension
*
  * `jacrev`
  * output basis cotangents
  * full Jacobian
  * small output dimension
*
  * `hessian`
  * nested forward/reverse queries
  * full second-derivative array
  * small scalar problems
*
  * `hvpState`
  * one parameter-shaped direction
  * Hessian-vector product
  * curvature information without a full Hessian
:::

A full Jacobian or Hessian can be much larger than the model output. The API makes those requests
explicit rather than computing them as a side effect of ordinary training. For a scalar loss with
millions of parameters, reverse mode gives all parameter gradients in one traversal; constructing
the full Hessian would be a very different computation.

For one-input tensor functions:

```
autograd.func.grad
autograd.func.valueAndGradScalar
autograd.func.vjp
autograd.func.jacfwd
autograd.func.jacrev
autograd.func.hessian
```

For checked models:

```
autograd.model.gradState
autograd.model.gradInputs
autograd.model.valueAndGrads
autograd.model.vjpState
autograd.model.vjpInput
autograd.model.jacrevState
autograd.model.jvpState
autograd.model.hvpState
```

The next page applies the same machinery to a scientific equation where a wrong derivative sign is
easy to see. The runtime chapter after it then explains which graph or tape each call constructs
and why the canonical verification IR is a different artifact.

References:

- [NN/API/Autograd.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Autograd.lean);
- Baydin et al.,
  [Automatic Differentiation in Machine Learning](https://arxiv.org/abs/1502.05767);
- [PyTorch autograd reference](https://docs.pytorch.org/docs/stable/autograd.html).
