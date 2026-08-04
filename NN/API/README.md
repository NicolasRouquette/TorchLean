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

The optimizer API includes `optim.sgd`, `optim.momentumSGD`, `optim.adagrad`,
`optim.rmsprop`, `optim.adam`, `optim.adamw`, and `optim.adadelta`. Optimizers are ordinary
runtime objects, but the proof layer files do not treat them as opaque callbacks. They package each
optimizer as a shape-polymorphic `TensorOptimizer`, whose `runSteps_append` theorem applies to every
update rule. An optimizer can additionally supply an independent `StepSpec`; TorchLean does not
duplicate runtime equations merely to manufacture an agreement theorem.

Related APIs are organized by object:

* **Muon** is an optimizer with an explicit orthogonalization backend. Runtime code uses
  `optim.runtimeMuon`. Proof code uses `Optim.TensorOptimizer.muon` and the generic
  `TensorOptimizer` interface. The Muon proofs separate three facts:
  the momentum buffer recurrence, the backend output used as the update direction, and the
  parameter update equation. Checked backends can provide either an exact certificate
  $Q^\mathsf{T}Q=I$ or an approximate certificate bounding $Q^\mathsf{T}Q-I$ entrywise. QR gives an exact path
  under positive-pivot hypotheses; Newton-Schulz gives a residual-checked approximate path and a
  fixed-point exact path. The detailed theorem handles live in
  `NN.MLTheory.Optimization.Muon` and `NN.MLTheory.Optimization.OptimizerLaws`.
* **GaLore** is gradient-projection machinery. Its runtime name is
  `optim.galore.projectedSGD`, because the projection and the optimizer applied afterward are both
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

Backend selection stays on one model instead of creating a separate method for every backend:

```lean
import NN.API
open TorchLean

def model :=
  nn.Sequential![
    nn.Linear 2 8,
    nn.ReLU,
    nn.Linear 8 1
  ]

def task :=
  Trainer.new model
    { task := .regression
      optimizer := optim.adam { lr := 0.001 }
      backend := .compiled
      dtype := .float32 }
```

Training, prediction, batched prediction, verification hooks, logs, and trained-result handles
belong to that trainer/result lifecycle. Backend selection is an option on the same model, rather
than a different name for the forward pass.

The main entry points are:

- `trainer.predict`,
- `trainer.train`,
- `trained.predict`,
- `trained.predictBatch`,
- `trained.verify`,
- `backend := .eager` or `backend := .compiled`,
- CUDA flags on the command line when native execution is selected.

PyTorch's `torch.compile` is the closest analogy. Compilation is a property of the model or trainer
path rather than a second mathematical forward function. TorchLean also keeps the graph and proof
objects explicit.

## Typical Workflow

A typical application follows this lifecycle:

1. Define a model with `TorchLean.nn` constructors.
2. Create a `Trainer` with the task, optimizer, dtype, seed, and backend.
3. Use `trainer.predict` for a before-training probe if the example needs one.
4. Run `trainer.train` on typed samples, batches, or a stream.
5. Use the trained handle for prediction, logging, export, or verification hooks.

That lifecycle keeps parameters and optimizer state together. The backend is selected through
`Options` or the trainer configuration.

When an example needs lower-level control, use the runtime API deliberately:

- `TorchLean.Module.*` for manually instantiated modules and custom losses.
- `TorchLean.Runtime.*` for scalar/backend options and runtime plumbing.
- `TorchLean.Data.*` for shape-checked loaders and batch streams.
- `TorchLean.Verification.*` for compiling checks and reading their results.

## Related References

* Lean language reference, for the module/import and namespace mechanisms that make this API
  structure possible: <https://lean-lang.org/doc/reference/latest/>
* PyTorch documentation, for the familiar model/optimizer/dataloader vocabulary that the
  training API follows where it helps readability:
  <https://pytorch.org/docs/stable/index.html>
