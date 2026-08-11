import VersoManual

open Verso.Genre Manual

#doc (Manual) "The Problem TorchLean Solves" =>
%%%
tag := "overview"
%%%

Suppose we train a small network, save its parameters, run it on a GPU, and later claim that its
output stays in a safe range for every input near a test point. During that one workflow, the phrase
"the model" can refer to at least six different things:

1. the source-level architecture;
2. the initialized parameter tensors;
3. the mutable parameters after training;
4. an exported operation graph;
5. the CPU or accelerator program that actually runs;
6. the graph and parameter payload analyzed by a verifier.

Usually these objects agree. When they do not, the resulting bug can be remarkably quiet. A
checkpoint loader may transpose a weight matrix. A verification script may forget that the model
expects normalized inputs. A compiler may replace separate multiplication and addition with an FMA.
The program still runs, but computes a different function from the one we had in mind.

TorchLean was designed around that problem. It is a neural-network library in which the
architecture, parameter payload, graph, arithmetic, and property can be named in the same language.
Numerical kernels may still run through optimized native providers; their inputs, outputs, and
assumptions remain attached to the computation being studied.

# A First Program

Application code uses the focused `NN.API` import:

```
import NN.API

open TorchLean

def model :
    nn.M (nn.Sequential (.dim 2 .scalar) (.dim 1 .scalar)) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]
```

The model accepts a length-two tensor and returns a length-one prediction. The dimensions occur in the model's
type, so the output of a layer with width seven cannot be fed to a layer expecting width eight.
`nn.M` means that this is a seeded model builder. It describes initialization but has not yet chosen
the random seed or produced concrete parameter tensors.

We can initialize it directly:

```
def initialized :=
  nn.run 2026 model
```

or ask the trainer to initialize and execute it:

```
def trainer :=
  Trainer.new model
    { task := .regression
      optimizer := optim.adam { lr := 0.001 }
      seed := 2026 }
```

The notation is intentionally familiar to a PyTorch user, but Lean learns more from the declaration.
It checks the input and output dimensions while elaborating the file. It distinguishes the seeded
builder from the initialized model. Once initialized, the model exposes the exact order and shape of
its parameter tensors. That information is available later to the trainer, graph compiler, and
verifier without rediscovering it from runtime metadata.

# Opening The Model Up

The same initialized model can participate in several parts of TorchLean.

## The model description

An `nn.Sequential` stores layer definitions, parameter shapes, initialization values, gradient
flags, and a forward program. This is the object used by the high-level trainer. Training creates an
effectful runtime runner whose parameters change over time; it does not rewrite the source
definition.

## The equations

The specification layer defines tensors as shape-indexed objects and gives operations such as
matrix multiplication, softmax, normalization, and loss functions explicit mathematical meanings.
Proofs use these definitions rather than reverse-engineering an opaque native buffer.

## The running program

The eager runtime owns autograd tape nodes, parameter and optimizer state, and optional accelerator
buffers. A backend profile decides which device and provider supply an operation. The checked CPU
profile, native CUDA profile, and LibTorch-forward profile are execution choices. Selecting one does
not, by itself, prove its kernels satisfy the mathematical specification.

## The operation graph

`NN.IR.Graph` is a directed acyclic graph of operation-tagged nodes. Each node has parent ids and a
declared output shape. Constants, weights, convolution parameters, and similar data live in a
separate payload store. This separation lets the graph structure be inspected without pretending
that two different payloads are the same trained model.

The graph structure itself does not choose a scalar type. Evaluation does: `NN.IR.Payload α` and
the graph denotation use one `α` for the numeric input, parameters, intermediates, and output. The
tensor chapter explains this homogeneous-scalar boundary; the PyTorch comparison explains what
would have to change for mixed precision.

## The object checked by a verifier

The verification compiler lowers supported forward programs and a concrete parameter payload
to a `CompiledIR`. IBP and CROWN operate on that graph. Numerical certificates can replay
bit-level ranges and backend policies over it. Other checkers consume external artifacts such as
alpha-beta-CROWN leaves or PINN residual certificates.

These are not rival implementations. They are views of the same computation made for different
jobs. The model description is pleasant to write. The runtime is built to execute. The graph is
easy to inspect. The equations are suitable for proofs. TorchLean's architecture is largely the
collection of translations that lets those views meet.

# Shapes Catch The Bug Where It Starts

A TorchLean tensor has a scalar type and a shape:

$$`\operatorname{Tensor.T}\;\alpha\;s`.

For example,

```
def predictions : Tensor.T Float (shape![32, 1]) :=
  tensorOfList! [32, 1] (List.replicate 32 0.0)

def labels : Tensor.T Float (shape![32]) :=
  tensorOfList! [32] (List.replicate 32 0.0)
```

These are different types. A loss that expects equal shapes cannot silently broadcast the label
vector across the second axis. Lean asks us to choose: reshape the labels, squeeze the predictions,
or use another loss. That is exactly where the ambiguity belongs.

Shape typing is deliberately modest. It catches structural mistakes while they are still local.
Questions about units, label quality, normalization, finiteness, and generalization appear later as
their own definitions and hypotheses instead of being smuggled into the word "tensor."

The scalar parameter is equally explicit. `Tensor.T α s` is homogeneous: every element of that
tensor has type `α`. The tensor chapter develops that point once, alongside the distinction between
scalar polymorphism and mixed precision.

# From One Output To A Statement

Suppose the trained model has parameters $`\theta`, its graph is $`g`, and the two input features
vary in a box $`B`. A range statement might be

$$`\forall x\in B,\qquad
  |\operatorname{denote}(g,\theta,x)_0-y^\star|\leq\delta`.

This line records the details that are easy to lose in prose:

- $`g` identifies the operation graph;
- $`\theta` identifies the concrete trained parameters;
- $`B` identifies the quantified input convention;
- `denote` identifies the scalar and operator semantics;
- index $`0`, target $`y^\star`, and tolerance $`\delta` identify the output property.

An interval pass can compute lower and upper output bounds. A Boolean check can then confirm that the
whole interval lies in $`[y^\star-\delta,y^\star+\delta]`. The last ingredient is a soundness theorem
saying that the computed interval really encloses `denote` for this graph.

The guide uses the following vocabulary throughout:

:::table +header
*
  * Evidence
  * What it establishes
*
  * a successful run
  * one execution produced a value
*
  * a parser or shape check
  * an artifact satisfies a structural predicate
*
  * a certificate replay
  * an artifact satisfies the checker's acceptance predicate
*
  * a soundness theorem
  * the accepted predicate implies a semantic proposition
*
  * a backend assumption
  * an external implementation is being trusted to meet a stated contract
:::

# Fast Kernels Still Belong

PyTorch has far broader operator coverage, distributed training, mature compilers, pretrained
models, and years of production optimization. Reimplementing all of that inside Lean would be a
poor use of both systems.

TorchLean can therefore call native CUDA or LibTorch for expensive operations. The source model,
parameter layout, and graph remain TorchLean objects; a backend profile chooses the implementation.
For example, LibTorch may compute the forward value of scaled dot-product attention while
TorchLean records the tape node and applies its own backward rule. Another profile may use native
TorchLean CUDA for both directions. Later we will inspect the capsule that records this choice.

This gives the project a practical division of labor: use established kernels where scale matters,
and use Lean to make the surrounding mathematical claim precise.

# Imports

Use the focused TorchLean API for application code:

```
import NN.API
open TorchLean
```

Use the complete library import when one file combines application code with proofs, verification,
floating-point semantics, backends, or widgets:

```
import NN
open TorchLean
```

Focused subsystem imports such as `NN.IR`, `NN.Floats`, `NN.Proofs`, and `NN.Verification` avoid
loading the whole project when only one layer is needed.

# References

- Leonardo de Moura and Sebastian Ullrich,
  [“The Lean 4 Theorem Prover and Programming
  Language”](https://lean-lang.org/papers/lean4.pdf), CADE 2021.
- Adam Paszke et al.,
  [“PyTorch: An Imperative Style, High-Performance Deep Learning
  Library”](https://arxiv.org/abs/1912.01703), NeurIPS 2019.
- Shiqi Wang et al.,
  [“Beta-CROWN: Efficient Bound Propagation with Per-neuron Split Constraints for Complete and
  Incomplete Neural Network Robustness Verification”](https://arxiv.org/abs/2103.06624),
  NeurIPS 2021.
