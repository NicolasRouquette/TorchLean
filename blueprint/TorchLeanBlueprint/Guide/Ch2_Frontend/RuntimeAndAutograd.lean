import VersoManual

open Verso.Genre Manual

#doc (Manual) "What Actually Runs" =>
%%%
tag := "runtime-autograd"
%%%

The earlier autograd walkthrough used one high-level call:

```
autograd.model.valueAndGradParamsScalar ...
```

Behind that call, TorchLean may construct a tape, replay a compiled derivative graph, invoke native
CUDA kernels, and return a dependent gradient pack. Other parts of the repository also use explicit
IR graphs and backend execution plans. These objects are related, but they are not interchangeable.

This chapter identifies each artifact, its lifetime, and the claim it can support.

# Four Objects Commonly Called “The Graph”

:::table +header
*
  * Object
  * Purpose
  * Contains
*
  * eager tape
  * reverse-mode execution
  * values, parents, local VJPs
*
  * compiled derivative graph
  * repeated runtime execution
  * forward, JVP, VJP closures
*
  * `NN.IR.Graph`
  * inspection and verification
  * explicit operation tags and payload references
*
  * backend execution plan
  * provider selection
  * accepted capsules and audit metadata
:::

A fifth object, a CUDA Graph capture, is a device launch-replay mechanism. Selecting TorchLean's
`.compiled` backend does not mean CUDA Graph capture.

Confusing these artifacts leads to bad guarantees. For example, accepting a backend plan does not
prove that the compiled trainer executed it, and proving an IR semantics theorem does not certify a
native tape node whose provider was never related to that IR operation.

# Eager Execution

Run a short eager training job:

```
lake exe torchlean quickstart_mlp \
  --device cpu --backend eager --steps 2 --seed 2026
```

An eager session is created for the chosen scalar and profile. As the model runs, each operation:

1. reads one or more parent values;
2. computes and stores its output;
3. records parent identifiers;
4. records a local reverse rule.

For:

$$`y=\operatorname{ReLU}(Wx+b)`,

the tape has operations corresponding to matrix multiplication, bias addition, and ReLU. The loss
adds subtraction, squaring, and reduction nodes. Reverse traversal begins from cotangent one at the
scalar loss.

CPU eager values and VJPs live on the ordinary tape. CUDA eager values are device buffers and their
reverse actions live on the CUDA tape. Both obey the same high-level reverse traversal idea, but
their storage and primitive providers differ.

# Gradient Accumulation Is Part Of The Tape Semantics

Consider:

$$`z=x^2+x^2`.

The graph contains two paths from $`x` to $`z`. Each square contributes $`2x`; the addition sends the
output seed to both parents. The final cotangent is:

$$`\bar x=2x+2x=4x`.

The runtime must add contributions associated with the same parent identifier. A correct local VJP
for square is insufficient if the tape traversal overwrites one contribution.

This is why TorchLean's autograd proofs have two layers:

- primitive derivative facts;
- global tape/traversal soundness.

The theorem covers the composition of the primitive derivative facts with the tape traversal.

# Inspect An Eager Tape

Open the widgets deep dive in VS Code and place the cursor on:

```
#tape_view ...
#tape_grads_view ...
#tape_trace_view ...
```

The first view shows operation nodes and parent edges. The second evaluates a scalar-output reverse
pass and annotates gradients. The third exposes traversal order.

A useful experiment is to duplicate one branch of a scalar expression, inspect the two incoming
paths, and verify that the leaf gradient doubles. Then wrap one branch with `detach`; its forward
node remains visible while its reverse contribution becomes zero.

The widget reads a runtime artifact. It does not alter execution or prove the tape correct.

# Compiled Execution

Now run:

```
lake exe torchlean quickstart_mlp \
  --device cpu --backend compiled --steps 2 --seed 2026
```

Compiled execution records the model's scalar loss once in a typed graph-building monad. Nodes carry
forward behavior and derivative behavior used for JVPs and VJPs. Each training step supplies current
parameters and inputs and replays the graph.

This avoids reconstructing the same high-level program on every step. The trainer method remains
`train`; the backend choice changes execution without introducing a second model API.

The current compiled trainer is CPU-only. Asking for a non-CPU compiled run is rejected. That
failure is preferable to printing “compiled” while silently using another path.

Compiled execution is distinct from `NN.IR.Graph` for an important reason: compiled nodes may carry
Lean functions implementing behavior, whereas a verification/import/export IR needs explicit,
inspectable operation tags and serializable payload references.

Both paths follow the homogeneous-scalar contract from *Tensors And Shapes*: a compiled program is
parameterized by one `α`, while an IR denotation receives one `Payload α` and produces values over
that same `α`.

# The Canonical IR

`NN.IR.Graph` represents operations such as linear, ReLU, reduction, normalization, and shape
transforms as data. Each node records:

- operation tag;
- input node identifiers;
- input/output shapes;
- operation-specific payload reference where needed.

Because the operation is data, an importer can validate it, a verifier can interpret it, and a code
generator can reject unsupported cases without executing an arbitrary closure.

The graph chapter later constructs:

$$`
x
\to\operatorname{Linear}_1
\to\operatorname{ReLU}
\to\operatorname{Linear}_2
\to\operatorname{sum}
\to\tanh.
`

Its evaluator interprets the same graph over real, interval, and IEEE scalar contexts. An eager tape
produced while evaluating this model is still a separate trace of one execution.

# Backend Planning

Before an eager operation executes, the session asks its backend profile for a capsule. The capsule
declares:

- semantic operation;
- target and provider;
- forward and VJP ownership;
- shape and layout requirements;
- numerical policy;
- evidence level.

The planner filters by operation and target, ranks candidates according to the profile, and runs an
acceptance gate. The accepted capsule is cached in the session. Before execution, the runtime binds
it to a typed `KernelHandler` with the same operation, provider, and device; a missing or mismatched
handler is an error. The capsule can be printed on first use.

Run:

```
lake exe torchlean quickstart_mlp \
  --device cpu --steps 1 --seed 2026 --show-backend
```

On a CUDA build:

```
lake -R -K cuda=true exe torchlean quickstart_mlp \
  --device cuda --steps 1 --seed 2026 --show-backend
```

The report answers “which provider was selected for this semantic operation?” The handler binding
also prevents a native CUDA selection from quietly entering the reference CPU closure. Neither fact
proves that the provider computes the declared formula. That depends on the capsule's evidence and
the executor's guards.

# Planning Is Not Execution

Suppose the planner accepts a native CUDA matmul capsule. The executor must still:

1. bind the capsule to a matmul handler for native CUDA on the CUDA device;
2. find the linked native symbol;
3. verify device availability;
4. check concrete dimensions and layout;
5. allocate or reuse buffers;
6. launch the operation;
7. turn native failures into Lean errors;
8. register the output and backward action.

An accepted plan describes admissible execution. It is not a receipt showing that the launch
completed.

Native wrappers still validate concrete storage before trusting a launch result. Maintained CUDA
convolution and pooling paths check dimension conversion, zero strides, element-count overflow,
input/output buffer lengths, and operation domains before or across the FFI. These checks turn bad
runtime inputs into errors and protect memory; they do not prove that a valid kernel's numeric value
refines the real specification.

Conversely, a native launch can succeed while violating an undeclared numerical assumption. This is
why capsule metadata includes contraction, reduction, subnormal, and rounding policies rather than
only a function pointer.

# Runtime Parameter Storage

A model parameter pack contains differently shaped tensors that share one scalar type `α`:

```
[weight1 : Tensor α [8,2],
 bias1   : Tensor α [8],
 weight2 : Tensor α [1,8],
 bias2   : Tensor α [1]]
```

The parameter-pack type preserves this dependent list. Runtime registries sometimes need to iterate over
parameters by name or identifier, so they package each tensor existentially with its shape.

This is not erasing shape information. It moves the shape from a compile-time index of the whole
collection into a value stored beside each registry entry. A lookup must recover and validate the
expected shape before returning to the typed API.

Parameter names, integer token inputs, RNG state, optimizer memory, and mutable model buffers belong
to the instantiated runner. Train/eval mode also belongs there.

# Train Mode And Eval Mode

Some operations depend on mode:

- dropout samples a mask during training and becomes identity during evaluation;
- batch normalization may update running statistics during training and use stored statistics at
  evaluation;
- other stochastic or stateful layers can follow the same pattern.

Calling `trained.predict` uses the retained runner in evaluation mode. `Trainer.Manual` exposes
mode changes directly for custom loops.

Mode is not a backend. The same eager CUDA runner can switch between training and evaluation while
remaining on the same device and provider profile.

# Randomness Is Runtime State

Dropout and stochastic model components need explicit generator state. A replayable run must know
which random state was used at each operation. The tape records the realized forward values needed
by the backward rule; it should not resample a different mask during reverse traversal.

This distinction becomes important for checkpointing. Saving parameters without RNG, optimizer, and
loader state can reproduce inference but not necessarily the next training update.

# Autograd Interfaces

Function-level calls:

```
autograd.func.grad
autograd.func.valueAndGradScalar
autograd.func.vjp
autograd.func.jacfwd
autograd.func.jacrev
autograd.func.hessian
```

compile a backend-generic tensor program and execute the requested derivative.

Model-level calls accept:

```
model
parameter pack
input
target or output cotangent
loss when needed
```

and return parameter-structured or input-structured derivatives. The high-level trainer uses this
runtime machinery inside its optimizer loop.

An instantiated `TorchLean.Module` also exposes three paired operations:

```
TorchLean.Module.lossAndBackward
TorchLean.Module.stepWithLoss
TorchLean.Module.stepWithOptimizerAndLoss
```

`lossAndBackward` returns a scalar loss and parameter gradients from one forward tape.
`stepWithLoss` applies a plain SGD update from that tape, while `stepWithOptimizerAndLoss` does the
same for an explicit optimizer state. Both step functions return the loss before the update. This
keeps the reported value tied to the random choices and saved intermediates that produced the
gradient.

# From Runtime To Proof

The proof path is:

```
calculus rule for each primitive
  -> semantic local VJP
  -> well-formed graph/tape composition
  -> global reverse result
```

Runtime refinement adds:

```
executable primitive
  -> declared semantic primitive
```

for every provider used by the run.

The first path can be entirely internal to Lean for a supported abstract graph. The second may be a
theorem, a sound checked guard, a numerical certificate, or an explicit trusted boundary. Native
CUDA, LibTorch, compiler, driver, and hardware behavior do not become proved merely because the
abstract derivative theorem exists.

Relevant proof sources:

- [`Autograd/Overview.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Overview.lean);
- [`Tape/Algebra/Soundness.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Tape/Algebra/Soundness.lean);
- [`Runtime/Link.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Autograd/Runtime/Link.lean).

# A Debugging Checklist

When a gradient is surprising:

1. Recompute a tiny case by hand.
2. Print the scalar semantics, device, backend, and selected capsules.
3. Inspect the forward tape and ensure the expected branch is reachable.
4. Check output cotangent shape and values.
5. Look for detach, train/eval mode, or stochastic state.
6. Compare eager and compiled CPU on the same explicit parameters.
7. Compare a native provider with `IEEE32Exec` or a reference path on a small finite case.
8. Distinguish a numerical discrepancy from a wrong derivative graph.

This workflow is more informative than asking whether “autograd” is correct as one indivisible
component. It identifies which artifact and which boundary must explain the mismatch.

# What To Carry Into The Graph Chapter

The eager tape explains one execution. The compiled graph accelerates repeated differentiation. The
canonical IR makes operation structure inspectable. The backend plan records provider choices.

TorchLean keeps all four because they solve different problems. Before defining the canonical IR,
we will cross one more concrete boundary: a named parameter payload moving between PyTorch and
Lean. The graph chapters then give that exchanged computation an inspectable semantics and relate
runtime approximation and verification claims to it.

References:

- Baydin et al.,
  [Automatic Differentiation in Machine Learning](https://arxiv.org/abs/1502.05767);
- [PyTorch autograd](https://pytorch.org/docs/stable/autograd.html);
- [PyTorch FX](https://pytorch.org/docs/stable/fx.html).
