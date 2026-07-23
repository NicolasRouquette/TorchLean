import VersoManual

open Verso.Genre Manual

#doc (Manual) "A Running Example" =>
%%%
tag := "running-example"
%%%

Our running example is the checked-in `quickstart_mlp` program. It is deliberately small, but it is
not pseudocode: it builds a dataset, initializes a model, runs autograd and Adam, prints predictions,
and can execute through the same runtime interfaces used by larger examples.

The task is to learn the piecewise-linear function

$$`y(x_1,x_2)
  =0.8\,\operatorname{ReLU}(x_1+x_2)
   -0.4\,\operatorname{ReLU}(x_2-x_1)+0.2`

on a grid in `[-1,1]^2`. This target is a useful first case for three reasons. Its nonlinearity rules
out a single affine layer, while its ReLU form lets a small ReLU network represent it
without approximation-theory distractions. Finally, its two-dimensional input lets us inspect every
shape and parameter without pages of indices.

# The Source Program

The model is a two-layer MLP:

```
import NN.API

open TorchLean

def inDim : Nat := 2
def outDim : Nat := 1

def model :
    nn.M (nn.Sequential (.dim inDim .scalar) (.dim outDim .scalar)) :=
  nn.Sequential![
    nn.linear inDim 8,
    nn.relu,
    nn.linear 8 outDim
  ]
```

Read the type before the body. `nn.M` says that model construction consumes a deterministic seed
stream. `nn.Sequential (.dim 2 .scalar) (.dim 1 .scalar)` says that the initialized model will map a
length-two tensor to a length-one tensor. The hidden width `8` is checked through composition: the
first linear layer produces length eight, ReLU preserves the shape, and the second linear layer
expects length eight.

The model returns a one-element tensor rather than a scalar because the shape-indexed layer API treats the
output feature dimension uniformly. A dataset target must therefore have shape `[1]`, not shape `[]`
and not shape `[batch]`.

# Why The Builder Is Seeded

Linear layers need initial weights and biases. Hiding randomness in a global generator would make a
model definition depend on ambient state. TorchLean instead represents initialization as a pure
state computation:

```
def initialized :=
  nn.run 2026 model
```

Running the same builder with the same seed produces the same initialization stream. The resulting
`nn.Sequential` contains four parameter tensors in layer order:

1. first-layer weights of shape `[8, 2]`;
2. first-layer bias of shape `[8]`;
3. second-layer weights of shape `[1, 8]`;
4. second-layer bias of shape `[1]`.

We can ask Lean for that layout:

```
#check nn.paramShapes initialized
#check nn.initParams initialized
```

The order is part of the forward-program interface. It is not a PyTorch-style dictionary of names.
Checkpoint adapters may use names at an external boundary, but they must eventually construct this
typed ordered payload.

# The Dataset

The target in the source example is:

```
def target (x1 x2 : Float) : Float :=
  let relu (x : Float) := if x < 0.0 then 0.0 else x
  (0.8 * relu (x1 + x2)) -
    (0.4 * relu (x2 - x1)) + 0.2
```

`Data.regressionGrid (-1.0) 1.0 5 target` samples five positions on each input axis, giving 25
examples. Each input has shape `[2]` and each target has shape `[1]`:

```
def dataset : Trainer.Dataset (.dim 2 .scalar) (.dim 1 .scalar) :=
  Data.regressionGrid (-1.0) 1.0 5 target
```

The tensor types settle the dimensional question: every row fits the model. Whether 25 points are
enough to learn the target is a different question, and this small run gives us a concrete place to
start asking it.

# Training It

The trainer combines the model, task, optimizer, seed, and runtime choices:

```
def trainer :=
  Trainer.new model
    { task := .regression
      optimizer := optim.adam { lr := 0.03 }
      seed := 2026 }
```

Run the complete checked-in command from the repository root:

```
lake exe torchlean quickstart_mlp \
  --device cpu \
  --steps 200 \
  --seed 2026
```

On the current implementation, this deterministic run reports:

```
dataset size = 25
mean_loss(before) = 0.761530
mean_loss(after)  = 0.003234

heldout: x=(0.250000,-0.750000)
target=0.200000
prediction(after)=[0.210239]
```

The exact log also includes intermediate losses and the prediction before training. Whenever we
return to `quickstart_mlp`, we keep this seed and configuration fixed; later chapters also introduce
smaller purpose-built variants.

# What Happens During One Step

For one sampled pair `(x,y)`, the runtime performs:

1. read the current four parameter tensors;
2. compute `Linear -> ReLU -> Linear`;
3. compute the regression loss;
4. seed the loss cotangent with one;
5. traverse the autograd tape backward to obtain parameter gradients;
6. update Adam's first and second moments;
7. replace each parameter with its updated value;
8. release or retain runtime buffers according to ownership.

The mathematical forward map is

$$`f_\theta(x)
  =W_2\,\operatorname{ReLU}(W_1x+b_1)+b_2`.

The loss for a single example is a scalar function of `θ`, even though the prediction has shape
`[1]`. Reverse mode computes vector-Jacobian products from that scalar back to all four tensors.
Adam then uses those gradients and its optimizer state to construct the next payload.

The repository contains four views of this step:

- eager runtime code that actually allocates tensors and records tape nodes;
- ideal VJP definitions used in autograd proofs;
- rounded VJP error transformers used by runtime-approximation proofs;
- optimizer contracts for SGD, momentum, and AdamW-style updates.

That gives us a running program, an ideal derivative, and a way to discuss the gap between them.

Running the model once with host `Float` and once with `IEEE32Exec` compares the two homogeneous
interpretations described in the tensor chapter. This matters when reading a checkpoint or backend
report: the scalar choice belongs to the run.

# Changing The Device

The source model does not mention CPU or CUDA. Device selection belongs to the runtime
configuration:

```
lake exe torchlean quickstart_mlp --device cpu --steps 200
```

or, in a CUDA-enabled build:

```
lake -R -K cuda=true exe torchlean \
  quickstart_mlp --device cuda --steps 200
```

The model type and parameter layout stay put. The selected backend profile plans the operations
using the capsules available in that build, and `--show-backend` prints the resulting plan. We will
look inside that plan in the backend chapter; there is no need to understand it before running this
example.

# Lowering The Forward Map

Verification starts from an initialized model and a concrete parameter payload:

```
import NN

open TorchLean

#check Verification.compileForward
  initialized
  (nn.initParams initialized)
```

The result is a `CompiledIR Float` containing:

- an `NN.IR.Graph`;
- the payload store used by parameterized operations;
- the distinguished input node;
- the distinguished output node.

`CompiledIR Float` makes the scalar choice visible. The underlying operation graph records shapes
and operation tags, while this compiled artifact and its payload use the scalar type selected for
the run.

For this MLP, the graph has an input, two linear operations, a ReLU, and an output path determined by
the compiler's lowering. Each node records its parents and output shape. The parameter store carries
the matrices and biases.

Using `nn.initParams initialized` analyzes the initial model. To analyze the trained model, the
compiler must receive the trained runtime parameters through the lower-level manual interface or a
saved exact-bits payload. Reusing the initial payload after training would verify another function.

# An Input Region Instead Of One Point

A forward prediction answers "what did the model return at `x`?" Verification usually asks a
quantified question. Around the held-out point

$$`c=(0.25,-0.75)`,

an `L∞` ball of radius `ε` is the box

$$`B_\varepsilon(c)
 =\{x\mid |x_i-c_i|\leq\varepsilon\text{ for }i=1,2\}`.

`Verification.seedLInfBall` places this box at the graph's input node. IBP propagates one interval
per coordinate through the graph. CROWN propagates affine lower and upper forms and can retain
dependencies that plain intervals lose.

For this one-output regression model, a useful property might be

$$`\forall x\in B_\varepsilon(c),\qquad
  |f_\theta(x)-0.2|\leq\delta`.

A computed output interval `[l,u]` supports this claim when

$$`0.2-\delta\leq l
  \quad\text{and}\quad
  u\leq 0.2+\delta`,

provided a soundness theorem covers the graph operations, payload, input bounds, and scalar
semantics used by the pass.

# Where We Go From Here

We now have one object worth following: a trained two-layer network with a known seed, parameter
layout, dataset, and held-out prediction. The frontend chapters explain how its tensors and tape are
built. The graph chapters lower the same forward map. The floating-point chapter compares ideal and
rounded evaluations. Finally, the verification chapters replace the single held-out point with an
entire input region.
