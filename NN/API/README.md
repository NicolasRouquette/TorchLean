# NN/API

These modules provide TorchLean's application API. Code that needs models, data, and training can
use the focused import:

```lean
import NN.API
open TorchLean
```

Use the complete library when the same file also needs specification, proof, or backend internals:

```lean
import NN
open TorchLean
```

The optimizer API includes `optim.sgd`, `optim.momentumSgd`, `optim.adagrad`,
`optim.rmsprop`, `optim.adam`, `optim.adamW`, and `optim.adadelta`. Optimizers are ordinary
runtime objects, but the proof layer files do not treat them as opaque callbacks. They package each
optimizer as a shape-polymorphic `TensorOptimizer`, whose `runSteps_append` theorem applies to every
update rule. An optimizer can additionally supply an independent `StepSpec`; TorchLean does not
duplicate runtime equations merely to manufacture an agreement theorem.

Related APIs are organized by object:

* **Muon** is an optimizer with an explicit orthogonalization backend. Runtime code uses
  `optim.muon.optimizer`. Proof code uses `Optim.TensorOptimizer.muon` and the generic
  `TensorOptimizer` interface. The Muon proofs separate three facts:
  the momentum buffer recurrence, the backend output used as the update direction, and the
  parameter update equation. Checked backends can provide either an exact certificate
  $Q^\mathsf{T}Q=I$ or an approximate certificate bounding $Q^\mathsf{T}Q-I$ entrywise. QR gives an exact path
  under positive-pivot hypotheses; Newton-Schulz gives a residual-checked approximate path and a
  fixed-point exact path. The detailed theorems are in
  `NN.MLTheory.Optimization.Muon` and `NN.MLTheory.Optimization.OptimizerLaws`.
* **GaLore** is gradient-projection machinery. Its runtime name is
  `optim.galore.sgd`, because the projection and the optimizer applied afterward are both
  part of the update statement.
* **LoRA** is adapter/parameterization structure. It lives under `TorchLean.Adapters.LoRA`, so
  examples keep adapter weights and optimizer state separate.

For application code, import `NN.API` and use the `TorchLean` namespace. Import
`NN.API.Verification` to lower models and call IBP or CROWN directly. The specialized
`Trainer.trainVerified` workflow is available only from `NN.API.Verification.Trainer`; its
`VerifiedTrainResult` retains the `verifyRobustLInf` operation. A file that also needs proofs,
graph specifications, or backend internals can import `NN`. Focused imports remain available when
only one subsystem is needed.

Trainable neural constructors live under `TorchLean.nn.models`. KNN, random-forest, Naive Bayes,
SVM, GMM, PCA, regression, gradient-boosted-tree, HMM, and Hopfield definitions retain their
original names from `NN.Spec.Models`. `NN.API` imports those definitions directly instead of
maintaining a second namespace of aliases.

## Application API

Execution settings stay on one model instead of creating a separate method for every runtime path:

```lean
import NN.API
open TorchLean

def model :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]

def task :=
  Trainer.new model
    { task := .regression
      optimizer := optim.adam { lr := 0.001 }
      execution := .typedGraph
      device := .cpu
      scalar := .ieee32Exec }
```

Training, prediction, repeated prediction, verification hooks, logs, and training results
belong to that trainer/result lifecycle. Execution mode and device selection are settings on the
same model, rather than different names for the forward pass.

The closest PyTorch terms are similar, but not identical:

| TorchLean | Meaning |
| --- | --- |
| `nn.Sequential` | Immutable, shape-checked model definition used by execution and proofs. |
| `nn.Module` | Live parameter-and-buffer state plus train/eval mode; this is closest to a PyTorch `nn.Module`. |
| `nn.Layer` | Immutable definition of one shape-checked layer; it is not a mutable module instance. |
| `Module.Objective` | Low-level mutable model state paired with a scalar objective; it is not merely a PyTorch loss function. |
| `nn.build seed builder` | Evaluate deterministic model initialization; this constructs state and does not run a forward pass. |
| `nn.forward` | Differentiable model program. |
| `module.run` | Concrete execution without gradient recording, in the current train/eval mode. |
| `predict` | Evaluation-mode, no-gradient execution that restores the previous mode. |
| `nn.softmax (axis := k)` | Softmax over any valid tensor dimension. |
| `Runtime.matmul` | Matrix multiplication with broadcasting over compatible leading dimensions. The same operation handles matrices, batches, and higher-rank collections. |
| `Tensor.oneHot n k` | One-hot encoding with `k : Fin n`; use `Tensor.checkIndex` or `Tensor.checkIndices` at raw-data boundaries. |
| `module.state` | Ordered, shape-indexed parameter and buffer tensors, not a string-keyed Python `state_dict`. |
| `scalar` | Arithmetic semantics for a run, not merely a storage `dtype`. |
| `execution := .typedGraph` | Record and reuse a typed forward/JVP/VJP graph; this is not `torch.compile`. |
| `device` | Storage and kernel target, independent of execution mode. |

The loss names also record their target convention. `task := .oneHotCrossEntropy axis` expects a
target tensor with the same shape as the logits and applies the loss along `axis`. Indexed targets
use the corresponding integer-label loss rather than sharing an ambiguous `crossEntropy` name.
Dimensions are zero-based and need not be final: use axis `0` for `[classes, batch]`, axis `1` for
`[batch, classes]`, and axis `2` for `[batch, time, vocabulary]`.

The main entry points are:

- `trainer.predict`,
- `trainer.train`,
- `trained.predict`,
- `trained.predictMany`,
- `trainer.trainVerified` and `trained.verifyRobustLInf` after importing
  `NN.API.Verification.Trainer`,
- `execution := .eager` or `execution := .typedGraph`,
- CUDA flags on the command line when native execution is selected.

## Numerical Containers

TorchLean writes a concrete statically shaped value as `Tensor α [dims...]`: the scalar type comes
first, followed by a list of dimensions. For example, `Tensor Float [batch, width]` contains
`Float` values and has two axes. Shape-polymorphic definitions may replace the dimension list with
a variable, as in `Tensor α s`.

TorchLean uses `Array α` for homogeneous collections whose length is known only while the program
runs. A rank-one tensor is already the fixed-length type: write `Tensor Float [width]`, not a
separate vector wrapper. Dataset storage, token buffers, sampled batches, and serialized payloads
use arrays until their shape is checked and they enter the tensor API.

Application code does not choose among several tensor representations. It constructs and receives
values such as `Tensor α [batch, width]`. Internal typed contexts use `TensorPack α shapes` only
when a fixed tuple contains tensors of different shapes. Runtime evaluators use
`Spec.SomeTensor α` only after they must erase a shape to store differently shaped intermediate
values in one array.

Lists have a narrower structural role. Dimension syntax such as `[batch, width]`, graph parent
identifiers, recursive architecture descriptions, and shape-indexed parameter packs use lists
because Lean computes types or performs structural recursion over them. They are not an alternative
container for tensor entries. The autograd proof layer also uses mathlib's finite Euclidean spaces
internally when applying Fréchet-derivative and adjoint APIs; model and runtime interfaces remain
tensor-valued.

Typed graph execution records a model once as a shape-indexed SSA graph and reuses its
forward/JVP/VJP programs. It is graph lowering, not native-code compilation; the guide's execution
chapter explains the distinction and the separate proof obligations.

The comparison follows PyTorch's documentation for
[`nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html),
[`nn.Embedding`](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html), and
[`torch.compile`](https://pytorch.org/docs/stable/generated/torch.compile.html).

Code that needs the reusable graph directly can call `nn.lowerToTypedGraph model`. The result is an
`nn.TypedGraphModel`, the public type view of the runtime `TypedGraph`. Its `forward`, `jvp`, and
`vjpWithSeed` methods keep the model's state tensors separate from its one input tensor.

Manual runtime code can instantiate a live module without changing the checked model definition:

```lean
nn.withModel model fun checked => do
  let module ← nn.Module.instantiate checked { device := .cpu }
  let trainingOutput ← module.run input
  module.eval
  let evaluationOutput ← module.run input
  let savedState ← module.state
  module.loadState savedState
```

`nn.Sequential` is immutable and remains the object used by lowering and proofs. `nn.Module`
owns live parameters, persistent buffers, and a train/eval flag. It starts in training mode, as a
PyTorch module does. `module.run` respects the current layer mode and does not construct a
backward tape; `predict` temporarily selects eval mode and restores the previous mode afterward.
The ordinary `instantiate` method uses native `Float32`. Use `instantiateAs` only when a manual
runtime deliberately needs another scalar interpretation.

Embedding models take token ID tensors rather than one-hot vectors:

```lean
let table := nn.build 2026 <| nn.embedding vocab embedDim
let tokenShape := [batch, seqLen]
let tokenIds ← Tensor.checkIndices vocab rawTokenIds
let module ← nn.IndexedModule.instantiate (table.model tokenShape) { device := .cpu }
let vectors ← module.predict tokenIds
```

Here `rawTokenIds` is a `Tensor Nat [batch, seqLen]` from an untrusted boundary. After
`Tensor.checkIndices`, `tokenIds` is a bounded `Tensor (Fin vocab) [batch, seqLen]`. The result has
shape `[batch, seqLen, embedDim]`. Token IDs are not
differentiable, and repeated IDs accumulate gradients into the same table row. Fresh tables use
standard-normal weights, matching `torch.nn.Embedding.reset_parameters`; model constructors such
as GPT-2 supply their own initialization scale.

An existing tensor can be used without reinitializing it:

```lean
let table := nn.Embedding.ofWeight weight
let frozenTable := nn.Embedding.ofWeight weight (freeze := true)
```

Both dimensions come from the type of `weight`. `ofWeight` records the same row-major payload for
semantic initialization and runtime storage, while `freeze := true` excludes the table from
parameter gradients. TorchLean's current indexed embedding implements dense row lookup and
scatter-add gradients. PyTorch options whose behavior changes the backward pass or mutates the
weight during forward, including `padding_idx`, `max_norm`, `scale_grad_by_freq`, and sparse
gradients, are not silently approximated by this constructor.

The public spatial layers accept a rank-one tensor of spatial extents. Convolution, transposed
convolution, normalization, and pooling use one rank-polymorphic API from their specification down
to the native CPU and CUDA boundary. Layout names remain only where an external format, such as an
imported PyTorch program, actually carries that convention.

Flattening and task heads use the same shape vocabulary. `nn.flattenAfter leading` preserves
every axis in `leading` and flattens only the remaining shape. For example, a classifier with
`leading := [batch, time]` maps `[batch, time, height, width]` to
`[batch, time, classes]`. TorchLean therefore has no separate batch-only classifier or
regressor API. `Tensor.flattenThenTake` preserves the same leading shape while retaining a prefix of
the flattened suffix. Random tensors use that same shape directly. For dimensions stored in a list,
write `rand.uniform key (s := dims)`.

## Graph Vocabulary

The relevant types appear in this order:

- `Tensor α [dims...]` is the public concrete syntax for a tensor whose dimensions are part of
  its type. Generic definitions use `Tensor α s` when the whole shape is a variable.
- `TensorPack α shapes` is the statically heterogeneous tuple used by model state and typed
  contexts. Every member shape remains in the type; ordinary model inputs and outputs stay
  `Tensor` values.
- `Runtime.Autograd.TorchLean.Program` is model code abstract over an operation interpreter. It is
  a polymorphic function, not a stored graph.
- `GraphSpec.Chain` and `GraphSpec.DAG.Model` are shape-indexed architecture syntax.
- `nn.TypedGraphModel` is the model-facing spelling of the same persistent `TypedGraph`; it factors
  the leaf context as model state followed by one input. Its inputs are tensor values, not
  session handles. The stored output is a typed reference to an input or recorded node; it is not
  required to be the last node in the graph.
- `Runtime.ValueRef` is a temporary handle owned by an eager tape or low-level graph-recording
  session. It names a runtime value but does not contain tensor elements.
- `NN.IR.Graph` is the op-tagged representation used for inspection, exchange, and verification.
- `Runtime.Autograd.IRExec.ForwardGraph` executes the supported forward fragment of `NN.IR.Graph`.
  It deliberately has no public differentiation operation.
- `NN.Backend.IR.GraphKernelPlan` records selected kernel capsules; it is metadata, not an
  executable graph.

The transitions are named explicitly: `GraphSpec.Chain.toProgram` builds an executable
TorchLean program, `nn.lowerToTypedGraph` records a model as a differentiable typed graph,
`Verification.lowerForwardToIR` runs a supported program through the broad IR-building interpreter,
and `Runtime.Autograd.IRExec.lowerToForwardGraph` lowers supported canonical IR nodes to the
forward-only IR executor. The separate
`NN.Verification.TorchLean.Proved.lowerForwardProgramToIR` path starts from a smaller first-order
language and has an end-to-end source-evaluation theorem.

## Typical Workflow

A typical application follows this lifecycle:

1. Define a model with `TorchLean.nn` constructors.
2. Create a `Trainer` with the task, optimizer, scalar semantics, seed, and execution mode.
3. Use `trainer.predict` for a before-training probe if the example needs one.
4. Run `trainer.train` on typed samples, batches, or a stream.
5. Use the training result for prediction, logging, export, or verification hooks.

That lifecycle keeps parameters and optimizer state together. The execution mode and device are
selected through `Options` or the trainer configuration.

When an example needs lower-level control, use the runtime API deliberately:

- `TorchLean.Module.*` for manually instantiated modules and custom losses.
- `TorchLean.Runtime.*` for scalar semantics, devices, kernel profiles, and runtime plumbing.
- `TorchLean.Data.*` for shape-checked loaders and batch streams.
- `TorchLean.Verification.*` for lowering verification graphs and reading check results.

## Related References

* Lean language reference, for the module/import and namespace mechanisms that make this API
  structure possible: <https://lean-lang.org/doc/reference/latest/>
* PyTorch documentation, for the familiar model/optimizer/dataloader vocabulary that the
  training API follows where it helps readability:
  <https://pytorch.org/docs/stable/index.html>
