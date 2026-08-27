import VersoManual

open Verso.Genre Manual

#doc (Manual) "TorchLean and PyTorch" =>
%%%
tag := "torchlean_vs_pytorch"
%%%

TorchLean looks familiar on purpose. It has tensors, modules, parameters, autograd, optimizers,
devices, and checkpoints because those ideas already work well in modern ML. But TorchLean is not
PyTorch rewritten in Lean, and it is not trying to catch PyTorch by accumulating the same number of
operators.

PyTorch's center of gravity is execution. A Python program can assemble a large model dynamically,
train it with highly tuned kernels, distribute the work over many devices, and deploy the result.
TorchLean's center of gravity is the connection between a running model and a mathematical
statement. It can execute models too, but it keeps asking questions that PyTorch normally leaves to
the surrounding project: What is the exact shape contract? Which parameter payload did the verifier
read? What equation does this graph node denote? Which arithmetic appears in the theorem?

That difference is easier to see in code than in slogans, so we will build the same MLP in both
systems and follow it through initialization, autograd, lowering, and execution.

# The Same MLP In Both Systems

A PyTorch model commonly owns its parameters through `nn.Module`:

```
# Python / PyTorch
model = torch.nn.Sequential(
    torch.nn.Linear(4, 8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 2),
)

logits = model(x)
```

The corresponding TorchLean builder is:

```
import NN.API

open TorchLean

def model :
    nn.Builder (nn.Sequential [4] [2]) :=
  nn.Sequential![
    nn.linear 4 8,
    nn.relu,
    nn.linear 8 2
  ]
```

Both programs describe an affine map, ReLU, and another affine map. Their surrounding contracts are
different.

In PyTorch, an eager tensor carries its shape as runtime metadata. Calling a layer inspects those
dimensions while the program executes. PyTorch's export and compilation systems can add symbolic
shape constraints later, but an ordinary annotation such as `torch.Tensor` does not distinguish a
vector of length four from a matrix with four columns.

In TorchLean, the input and output shapes index the `nn.Sequential` type. Layer composition is
checked while Lean elaborates the definition. `nn.Builder` also records that `model` is a deterministic
seed-state computation waiting to initialize its parameters.

Dynamic shapes are convenient during exploration and for data-dependent programs. Shape-indexed
types require more information up front, but they make layer composition and later theorem
statements much cleaner. TorchLean still accepts runtime-loaded data; it checks the dimensions once
at the boundary and then works with the resulting typed tensor.

Operator names match PyTorch only when the semantics match. Use `matmul` for matrices, batches of
matrices, and higher-rank collections; it broadcasts compatible leading dimensions.

There is also no special batch tensor type. Operations that preserve outer axes accept a leading
shape explicitly. Thus `nn.flattenAfter [batch, time]` keeps the batch and time axes and
flattens everything after them. Classification and regression heads use the same argument, so one
definition works for single examples, batches, sequences of examples, and higher-rank collections.
The resulting input and output shapes are still checked while the model is built.

# Parameters: Object Fields Versus Explicit Payloads

A PyTorch `nn.Module` registers parameter objects. Calling `model(x)` reads the module's current
fields. An optimizer mutates those parameters, usually through gradient fields populated by
autograd.

TorchLean's model description contains parameter shapes, initialization tensors, gradient
flags, and a forward program. The forward program receives the *live parameter payload* explicitly.
Initialization and execution are therefore related but distinguishable:

```
def initialized :=
  nn.build 2026 model

#check nn.stateShapes initialized
#check nn.initState initialized
#check nn.forwardProgram (model := initialized)
```

The initial tensors are part of the initialized description. Execution instantiates a mutable
`nn.Module` from them:

```
let module ← nn.Module.instantiate initialized { device := .cpu }
let before ← module.state
let output ← module.run input
module.eval
```

The module owns the live parameters, persistent buffers, and train/eval mode. The immutable
`nn.Sequential` remains the value used by graph lowering and proofs. A theorem about
`nn.initState initialized` is therefore not a theorem about the parameters after 10,000 optimizer
steps; a claim about trained weights must name the saved payload it analyzes.

This explicit payload pays off when the model becomes a graph. Lowering receives the graph and
the exact tensors being analyzed rather than reading whatever happens to be stored in mutable fields
at that moment. A PyTorch checkpoint importer therefore has a concrete job: map names, shapes,
order, and layout into this payload.

Integer-indexed embeddings follow the same rule without encoding token ids as floating-point
one-hot vectors. `nn.embedding vocab embedDim` constructs a trainable table,
`nn.Embedding.ofWeight weight` preserves an existing table exactly, and `freeze := true` keeps that
table in module state without requesting a parameter gradient. If the index tensor has dimensions
`dims`, its output has dimensions `dims ++ [embedDim]`; repeated ids accumulate into the same
gradient row.

# Autograd: A Tape In Two Different Roles

PyTorch dynamically constructs an autograd graph while tensor operations run. During backward,
saved tensors and derivative rules propagate vector-Jacobian products to leaves. The implementation
is mature, broad, and deeply integrated with the dispatcher.

TorchLean's eager runtime also records a tape. The difference is that the codebase keeps three
layers available side by side:

1. the runtime node and saved values used to compute a gradient;
2. the ideal derivative or VJP definition used in a theorem;
3. the proof that the runtime rule implements that mathematical derivative.

For a scalar loss $`L(\theta)`, reverse mode does not need to materialize the full Jacobian. Starting
with the cotangent $`1`, each node applies a local VJP:

$$`\bar{x}=J_f(x)^\mathsf{T}\bar{y}`.

The runtime rule is the fast implementation. The ideal VJP is the equation we want it to implement.
The theorem, when available for that operation, connects the two.

External kernels make the three layers particularly important because forward and backward may
cross different boundaries. The backend chapter records that ownership per operation instead of
letting it hide inside the training loop.

# Training Loops

A conventional PyTorch loop is explicit Python mutation:

```
for x, y in loader:
    optimizer.zero_grad()
    prediction = model(x)
    loss = loss_fn(prediction, y)
    loss.backward()
    optimizer.step()
```

TorchLean's trainer packages the same lifecycle:

```
def trainer :=
  Trainer.new model
    { task := .oneHotCrossEntropy 1
      optimizer := optim.adam { lr := 0.001 }
      seed := 2026 }

-- let result ← trainer.train dataset
--   { steps := 200, batchSize := 16, logEvery := 25 }
```

Internally, the runtime owns mutable parameters, gradients, optimizer moments, tape state, and
possibly device buffers. TorchLean does not force a large GPU training loop to allocate a new pure
tensor tree at every step. The semantic interfaces remain explicit while the execution engine uses
mutation and ownership where performance requires it.

The `train` call returns a trained result whose prediction closures refer to the trained runner.
Lower-level manual APIs expose parameter tensors and individual forward, backward, and optimizer
steps when verification or research code needs them.

This is still an ordinary training loop: it produces updated parameters, logs, and predictions. The
later proof chapters start from those concrete objects rather than trying to read a theorem out of
the loss curve.

# Dispatch And Backend Selection

PyTorch routes an operation through its dispatcher. Device, dtype, layout, compilation state, and
available libraries determine the eventual implementation. A CUDA matrix multiplication may use
cuBLAS, convolution may use cuDNN, and attention may select one of several fused kernels.

TorchLean records the corresponding choice as a device, operation, provider, and accepted kernel
capsule. Reports name the implementation selected for each operation, and unavailable requests
fail instead of becoming an unreported CPU run. This keeps the model, parameter layout, graph, and
tape stable while selected operations cross a native boundary.

[Inside The Backend Planner](Runtime___-Autograd___-and-Interop/Inside-The-Backend-Planner/)
explains capsules, provider preference, LibTorch forward ownership, and assurance policies in full.
[From A Tensor Operation To A GPU Kernel](Floating-Point-and-Native-Boundaries/From-A-Tensor-Operation-To-A-GPU-Kernel/)
then follows the native CUDA boundary.

# Graphs, Lowering, And Compilation

PyTorch offers FX, `torch.export`, AOTAutograd, and compiler stacks such as `torch.compile`.
Their graphs support transformation and deployment inside the PyTorch ecosystem.

TorchLean has two graph-facing layers with different purposes:

- `GraphSpec` describes structured architectures and lowers them to TorchLean programs;
- `NN.IR.Graph` is the canonical operation DAG used by lower-level evaluation and verification.

An IR node contains an operation tag, parent ids, and an output shape. Parameter and constant values
live in payload stores. `NN.IR.Semantics` defines how supported nodes are interpreted over a scalar
domain.

The verification lowering pass can lower supported initialized models and parameter payloads to
this IR. A separate first-order source language under `NN.Verification.TorchLean.Proved` has an
end-to-end lowering-correctness theorem. They share an IR target, but the theorem applies to the
proved source fragment, not automatically to every model accepted by the broader lowering pass.

TorchLean calls the conversion to `NN.IR.Graph` *verification lowering*. Typed graph lowering is a
separate path: it records a shape-indexed SSA graph whose nodes store forward and differentiation
functions. Derivative correctness is established by separate theorems over proof-carrying nodes and
graphs. Neither lowering path performs optimization, fusion, scheduling, or native code generation,
so selecting typed graph execution does not imply a `torch.compile`-style compiler.

# Checkpoints And Graph Import

PyTorch checkpoints are Python-oriented zip/pickle artifacts. TorchLean does not duplicate that
loader inside Lean. Its adapter generates Python that asks PyTorch to load a `state_dict` and emit
named tensor data as plain JSON. Lean then parses each requested tensor into a statically known
shape.

Graph import is separate. A generated adapter uses `torch.export` or FX to emit the
`torchlean.ir.v1` format, which Lean parses into `NN.IR.Graph`. The ONNX adapter lowers supported
static nodes to the same format.

These steps establish progressively stronger but still limited facts:

1. JSON parsing establishes that the bytes match the expected schema.
2. Tensor parsing establishes that a requested payload has the expected shape.
3. Graph checks establish node references and shape contracts.
4. Operator-by-operator semantic refinement would establish equality with the source graph.
5. A runtime bridge would connect that graph semantics to the deployed kernels.

The first three steps are already useful: they catch malformed exports and shape/layout mistakes.
The remaining two are where a claim about the imported program becomes a claim about its meaning.

# Floating Point

PyTorch's numerical behavior depends on dtype, device, library versions, compiler transformations,
and reduction algorithms. Its numerical-accuracy documentation explicitly warns that mathematically
identical computations are not guaranteed to be bitwise identical across batched, sliced, device,
or backend paths.

TorchLean names several numerical meanings:

- real-valued specifications for ideal mathematics;
- `NF`, a configurable rounded-real arithmetic;
- `FP32`, a rounded-real specialization with binary32 precision and gradual underflow, but without
  an upper exponent bound or IEEE special values;
- `IEEE32Exec`, an executable bit-level binary32 model;
- runtime CPU, CUDA, and LibTorch representations.

The generic layer was influenced by Flocq's separation of formats from rounding operators.
`IEEE32Exec` is TorchLean's executable Lean reference for binary32. The floating-point chapters
derive these layers from examples and show how they reconnect to a runtime.

Training can proceed without this extra structure. Claims that compare an exact equation with an
executable result need it.

# What TorchLean Adds

For the small MLP, PyTorch gives an excellent route from Python source to fast training. TorchLean
adds a route from a typed model to several inspectable semantic objects:

```
shape-typed model
  -> initialized parameter layout
  -> runtime tape and trained payload
  -> canonical operation graph
  -> exact or rounded scalar semantics
  -> bound or certificate
  -> theorem with named assumptions
```

Each arrow is an interface that can be tested, checked, or proved independently as coverage grows.

# When To Use Which

PyTorch is the natural choice when the main requirement is broad model coverage, pretrained
ecosystems, distributed training, compilation, and production deployment. TorchLean becomes useful
when the project also needs:

- dimensions checked in model and tensor types;
- mathematical operations available as Lean definitions;
- an inspectable graph and parameter payload;
- executable verification and certificate checking;
- formal statements about graph, autograd, optimizer, or numerical behavior;
- an explicit ledger of native and external assumptions.

Many projects should use both. Training can remain in PyTorch while TorchLean checks an exported
artifact, or training can run in TorchLean while selected bottlenecks use PyTorch kernels. They solve
different parts of the same problem.

# References

- [PyTorch `nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html).
- [PyTorch `nn.Embedding`](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html).
- [PyTorch `torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html).
- Adam Paszke et al.,
  [“PyTorch: An Imperative Style, High-Performance Deep Learning
  Library”](https://arxiv.org/abs/1912.01703), NeurIPS 2019.
- [PyTorch autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html).
- [PyTorch `torch.export`](https://docs.pytorch.org/docs/stable/export.html).
- [PyTorch numerical accuracy notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html).
- Sylvie Boldo and Guillaume Melquiond,
  [“Flocq: A Unified Library for Proving Floating-Point Algorithms in
  Coq”](https://doi.org/10.1109/ARITH.2011.40), IEEE ARITH 2011.
