# NN/API

These modules back the public TorchLean API. Application code that only needs models, data, and
training can use the focused import:

```lean
import NN.API
open TorchLean
```

User-facing declarations should live in their canonical `TorchLean.*` namespace when practical.
For example, the lightweight module `NN.API.CLI` now defines its parsers directly under
`TorchLean.CLI`; the application API adds only the distinct IO conveniences. Lean's `export` command helps
unqualified lookup after a namespace is opened, but it does not create a qualified declaration such
as `TorchLean.Shape.scalar`. An `abbrev` is therefore still required where the stable qualified API
bridges a declaration owned by an internal namespace. The API ownership check reports those
bridges explicitly so subsystems can migrate without silently breaking public names.

Use the complete library when the same file also needs specifications, proofs, verification, or
backend internals:

```lean
import NN
open TorchLean
```

Use the `NN.API.*` imports when you are extending TorchLean itself or working inside a subsystem:

* `NN.API.Public` backs the `TorchLean.nn`, `TorchLean.optim`, `TorchLean.Trainer`,
  `TorchLean.classical`, `TorchLean.Data`, `TorchLean.Loss`, and `TorchLean.Metrics` namespaces.
* `NN.API.Runtime` exposes the executable runtime API for code that works directly with tensor
  operations, module execution, autograd, supervised training, and session-level tools.
* Focused files under `NN/API` give subsystem code a smaller import target than the full `NN`
  umbrella.

The public optimizer API includes `optim.sgd`, `optim.momentumSGD`, `optim.adagrad`,
`optim.rmsprop`, `optim.adam`, `optim.adamw`, and `optim.adadelta`. Optimizers are ordinary
runtime objects, but the proof layer files do not treat them as opaque callbacks. They package each
optimizer as a shape-polymorphic `TensorOptimizer`, whose `runSteps_append` theorem applies to every
update rule. An optimizer can additionally supply an independent `StepSpec`; TorchLean does not
duplicate runtime equations merely to manufacture an agreement theorem.

Optimizer-adjacent APIs use names that expose the object they actually own:

* **Muon** is an optimizer with an explicit orthogonalization backend. Runtime code uses
  `optim.runtimeMuon`. Proof code uses `Optim.TensorOptimizer.muon` and the generic
  `TensorOptimizer` interface. The Muon proofs separate three facts:
  the momentum buffer recurrence, the backend output used as the update direction, and the
  parameter update equation. Checked backends can provide either an exact certificate
  `QᵀQ = I` or an approximate certificate bounding `QᵀQ - I` entrywise. QR gives an exact path
  under positive-pivot hypotheses; Newton-Schulz gives a residual-checked approximate path and a
  fixed-point exact path. The detailed theorem handles live in
  `NN.MLTheory.Optimization.Muon` and `NN.MLTheory.Optimization.OptimizerLaws`.
* **GaLore** is gradient-projection machinery. Its runtime name is
  `optim.galore.projectedSGD`, because the projection and the optimizer applied afterward are both
  part of the update statement.
* **LoRA** is adapter/parameterization structure. It lives under `TorchLean.Adapters.LoRA`, so
  examples keep adapter weights and optimizer state separate.

Application code should import `NN.API` and use the `TorchLean` namespace. Files that also use
proofs, verification, graph specifications, or backend internals can import the complete `NN`
umbrella. TorchLean's implementation uses narrower modules to keep subsystem dependencies explicit.

Neural architectures and classical models have different interfaces because they are different
kinds of object. Trainable neural constructors live under `TorchLean.nn.models`. Existing KNN,
random-forest, Naive Bayes, SVM, GMM, PCA, regression, gradient-boosted-tree, HMM, and Hopfield
definitions are available under `TorchLean.classical`. These are aliases to the mathematical model
definitions in `NN.Spec.Models`, not copied implementations.

## Public API Shape

The public API should feel like one model with selectable execution modes, not like a separate
method for every backend. The intended pattern is:

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
should hang off that trainer/result story. Backend selection is an option on the same model, not a
different public name for the forward pass.

That is why public code should prefer names like:

- `trainer.predict`,
- `trainer.train`,
- `trained.predict`,
- `trained.predictBatch`,
- `trained.verify`,
- `backend := .eager` or `backend := .compiled`,
- CUDA flags on the command line when native execution is selected.

Do not introduce public names that encode arity or implementation details. If an implementation
layer needs an internal helper, keep it internal and give the application API a model-level name.

The comparison to PyTorch's `torch.compile` is the right public analogy: compiling should be a
property of the model or trainer path, not a second mathematical forward function. TorchLean differs
because the graph and proof objects are explicit, but the user-facing shape should still stay simple.

## Public Workflow

A public example should follow this lifecycle:

1. Define a model with `TorchLean.nn` constructors.
2. Create a `Trainer` with the task, optimizer, dtype, seed, and backend.
3. Use `trainer.predict` for a before-training probe if the example needs one.
4. Run `trainer.train` on typed samples, batches, or a stream.
5. Use the trained handle for prediction, logging, export, or verification hooks.

That lifecycle keeps parameters and optimizer state inside one owner. Eager inference, compiled
inference, and no-gradient inference should not be taught as separate public entrypoints. The
backend is a choice in `Options`/trainer configuration; it is not a new public mathematical
operation.

When an example needs lower-level control, use the runtime API deliberately:

- `TorchLean.Module.*` for manually instantiated modules and custom losses.
- `TorchLean.Runtime.*` for scalar/backend options and runtime plumbing.
- `TorchLean.Data.*` for shape-checked loaders and batch streams.
- `TorchLean.Verification.*` for certificate/checker entrypoints.

If a feature needs to appear in ordinary tutorials, add it through this application API instead of
teaching users an implementation path under `NN/Runtime`.

## Folder Rule

Top-level `NN/API/*.lean` files are import entrypoints. Definitions belong in matching subfolders:

* `NN.API.Core` imports `NN.API.Core.Basic`.
* `NN.API.Data` imports `NN.API.Data.Core`; transforms live in `NN.API.Data.Transforms`.
* `NN.API.Text` imports `NN.API.Text.Core`; BPE and Unicode tables live in `NN.API.Text.*`.
* `NN.API.Common` imports `NN.API.Common.Core`.
* `NN.API.CLI` imports `NN.API.CLI.Core`.
* `NN.API.Init`, `NN.API.Json`, `NN.API.Macros`, `NN.API.Rand`, and `NN.API.Samples` import their
  matching `*.Core` modules.
* `NN.API.Public.Facade.Base`, `NN.API.Public.Facade.NN`,
  `NN.API.Public.Facade.Runtime`, and `NN.API.Public.Facade.Data` are import-only entrypoints for
  their matching `*.Core` definition files.
* `NN.API.Public.Facade.Base.Core` is itself an import-only aggregator. The root public API
  definitions are split into `Base.Root`, `Base.Verification`, `Base.CLI`, `Base.Runtime`, and
  `Base.Tensor`.
* `NN.API.Public.Facade.Classical` gives the classical model definitions one discoverable public
  namespace without duplicating their semantics.
* `NN.API.Public.Facade.Runtime.Core` is itself an import-only aggregator. The runtime API
  definitions are split into `Runtime.Autograd`, `Runtime.TensorPack`,
  `Runtime.ObjectiveAdapters`, `Runtime.RL`, `Runtime.Module`, `Runtime.Supervised`,
  `Runtime.Optim`, `Runtime.LossMetrics`, `Runtime.Text`, and `Runtime.Adapters`.
* `NN.API.Public.Facade.NN.Core` is itself an import-only aggregator. The `TorchLean.nn` API is
  split into `NN.Basic`, `NN.Summary`, `NN.Params`, `NN.Runtime`, `NN.Layers`, and `NN.Models`.
* `NN.API.Public.Facade.Data.Core` is itself an import-only aggregator. The public data API is
  split into `Data.Sample`, `Data.Datasets`, `Data.Text`, `Data.Builtin`, `Data.Checkpoint`, and
  `Data.DotInfo`.
* `NN.API.Public.Facade.Trainer` imports focused trainer modules for summaries, core types,
  construction, trained-result handles, runtime options, verification APIs, training
  definitions, and trained-result methods.
* `NN.API.Public.Facade.Trainer.Train` is itself an import-only aggregator. Training
  definitions are split into `Train.Regression`, `Train.CrossEntropy`, `Train.Custom`, and
  `Train.Streams`.
* `NN.Examples.ModelZoo` holds flags, logging, paths, and banners used by repository examples;
  these command adapters are not part of the application API. `Trainer.Command` in the example
  tree owns runnable command orchestration.
* `NN.API.Public.Autograd`, `NN.API.Public.Seeded`, and `NN.API.Public.TensorPack` are import-only
  entrypoints for their matching `*.Core` definition files.
* `NN.API.Public.Training` imports `NN.API.Public.Training.Core`; this is the lower-level
  callback/runner layer, not the path ordinary examples should teach first.

Umbrella files preserve short import names. Implementation files live in the matching
subdirectories.

The public API exposes scalar evidence, shape-indexed parameter packs, graph semantics, and checker
boundaries when those objects are part of the example or proof. The import story is:

* `import NN.API` for application-facing model, data, and training code.
* `import NN` when the same file also needs verification, proofs, or backend infrastructure.
* `NN.Runtime`, `NN.Spec`, `NN.Proofs`, and other focused imports for subsystem files.

## Related References

* Lean language reference, for the module/import and namespace mechanisms that make this API
  structure possible: <https://lean-lang.org/doc/reference/latest/>
* PyTorch documentation, for the familiar model/optimizer/dataloader vocabulary that the public
  training API follows where it helps readability:
  <https://pytorch.org/docs/stable/index.html>
