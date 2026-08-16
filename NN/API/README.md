# NN/API

These modules provide TorchLean's application API. Code that needs models, data, and training can
use the focused import:

```lean
import NN.API
open TorchLean
```

Use the complete library when the same file also needs specifications, proofs, verification, or
backend internals:

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

For application code, import `NN.API` and use the `TorchLean` namespace. A file that also needs
proofs, verification, graph specifications, or backend internals can import `NN`. Focused imports
remain available when only one subsystem is needed.

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
| `nn.softmaxLast` | Softmax layer over the final axis; lower-level functional APIs accept an explicit axis. |
| `mm` | Two-dimensional matrix multiplication, matching `torch.mm`; use `bmm` for rank-three batches. TorchLean does not currently expose PyTorch's rank-polymorphic, broadcasting `torch.matmul`. |
| `Tensor.oneHot n k` | One-hot encoding with `k : Fin n`; raw natural-number conversion is explicitly named `oneHotNatOrZero`. |
| `module.state` | Ordered, shape-indexed parameter and buffer tensors, not a string-keyed Python `state_dict`. |
| `scalar` | Arithmetic semantics for a run, not merely a storage `dtype`. |
| `execution := .typedGraph` | Record and reuse a typed forward/JVP/VJP graph; this is not `torch.compile`. |
| `device` | Storage and kernel target, independent of execution mode. |

The loss names also record their target convention. `task := .oneHotCrossEntropy` expects a target
tensor with the same shape as the logits. Indexed language-model APIs instead accept `Tensor Nat`
targets and use `Loss.crossEntropyRowsNat`; TorchLean does not call both conventions simply
`crossEntropy`.

The main entry points are:

- `trainer.predict`,
- `trainer.train`,
- `trained.predict`,
- `trained.predictMany`,
- `trained.verifyRobustLInf`,
- `execution := .eager` or `execution := .typedGraph`,
- CUDA flags on the command line when native execution is selected.

Typed graph execution is not TorchLean's spelling of `torch.compile`. It records the model once as
a shape-indexed SSA graph and reuses its forward/JVP/VJP programs. This checks graph construction
and tensor shapes, but does not automatically prove every stored derivative rule. TorchLean
reserves *compilation* for optimization, fusion, scheduling, and native code generation.

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

Embedding models use integer tensors rather than one-hot vectors:

```lean
let table := nn.build 2026 <| nn.embedding vocab embedDim
let module ← nn.IndexedModule.instantiate (table.model tokenShape) { device := .cpu }
let vectors ← module.predict tokenIds
```

The result has shape `tokenShape.appendDim embedDim`. Token IDs are not differentiable, repeated
IDs accumulate gradients into the same table row, and the module rejects an ID outside
`0, ..., vocab - 1` before invoking the runtime gather. Fresh tables use standard-normal weights,
matching `torch.nn.Embedding.reset_parameters`; model constructors such as GPT-2 supply their own
initialization scale.

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

The public spatial layers accept an arbitrary vector of spatial extents. Names such as `NCHW` or
`Conv2d` are confined to layout-specific runtime kernels, where the suffix records an actual
precondition rather than a user-facing tensor category.

Flattening and task heads use the same shape vocabulary. `nn.flattenLeading leading` preserves
every axis in `leading` and flattens only the remaining shape. For example, a classifier with
`leading := shape![batch, time]` maps `shape![batch, time, height, width]` to
`shape![batch, time, classes]`. TorchLean therefore has no separate batch-only classifier or
regressor API. `Tensor.flattenPrefix` preserves the same leading shape while retaining a prefix of
the flattened suffix. Random tensors follow the same convention: `rand.uniform` is used when the
result shape is known statically, while `rand.uniformDims` is the boundary constructor for a
runtime list of dimensions.

## Graph Vocabulary

The relevant types appear in this order:

- `Tensor α s` is a tensor value whose shape `s` is part of its type.
- `Runtime.Autograd.TorchLean.Program` is model code abstract over an operation interpreter. It is
  a polymorphic function, not a stored graph.
- `GraphSpec.Chain` and `GraphSpec.DAG.Model` are shape-indexed architecture syntax.
- `nn.TypedGraphModel` is the model-facing spelling of the same persistent `TypedGraph`; it factors
  the leaf context as model state followed by one input. Its inputs are tensor values, not
  session handles. The stored output is a typed reference to an input or recorded node; it is not
  required to be the last node in the graph.
- `TensorRef` is a temporary handle owned by an eager tape or low-level graph-recording session.
- `NN.IR.Graph` is the op-tagged representation used for inspection, exchange, and verification.
- `Runtime.Autograd.IRExec.ForwardGraph` executes the supported forward fragment of `NN.IR.Graph`.
  It deliberately has no public differentiation operation.
- `NN.Backend.GraphKernelPlan` records selected kernel capsules; it is metadata, not an executable
  graph.

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
