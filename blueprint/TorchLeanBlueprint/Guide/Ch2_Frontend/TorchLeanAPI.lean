import VersoManual

open Verso.Genre Manual

#doc (Manual) "The TorchLean API" =>
%%%
tag := "torchlean-api"
%%%

Most user programs need two lines:

```
import NN.API
open TorchLean
```

`NN.API` is the maintained import for application code. It brings in tensors, models, data, training,
optimizers, prediction, and explicit differentiation. It does not pull every proof, verifier,
floating-point implementation, widget, and backend-internal module into a small training file.

This chapter builds one program using only that API, then explains when a narrower or
lower import is appropriate.

# The Namespaces

You can read the import as a small workbench rather than a module-tree lesson. `Tensor` gives us
values, `nn` turns them into models, `Data` supplies examples, and `Trainer` carries the model
through optimization and prediction. The remaining names become useful when we ask a more specific
question:

:::table +header
*
  * Namespace
  * Responsibility
*
  * `Tensor`, `Shape`
  * shape-indexed values and constructors
*
  * `nn`
  * layers, blocks, model families, functional operations
*
  * `Data`
  * datasets, loaders, batching, text and checkpoint helpers
*
  * `Trainer`
  * configuration, training, reports, prediction, manual loops
*
  * `optim`
  * optimizer configuration
*
  * `autograd`
  * function and model derivatives
*
  * `Runtime`
  * dtype, execution mode, device, and backend-contract selection
*
  * `Verification`
  * model-to-IR compilation and IBP/CROWN helpers
*
  * `classical`
  * classical and statistical model APIs
:::

The lowercase `nn.linear`, `nn.relu`, and `optim.adam` names are the canonical spellings.
Internal implementation namespaces may be longer because they distinguish specification, runtime,
and proof layers. Application code should not depend on those names unless it genuinely needs the
lower layer.

# Make A Scratch Program

Create `Scratch.lean` at the repository root:

```
import NN.API

open TorchLean

def model :
    nn.M (nn.Sequential (shape![2]) (shape![1])) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]

def xs : Tensor.T Float (shape![4, 2]) :=
  tensor! [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
  ]

def ys : Tensor.T Float (shape![4, 1]) :=
  tensor! [[0.0], [1.0], [1.0], [0.0]]

def data : Trainer.Dataset (shape![2]) (shape![1]) :=
  Data.tensorDataset xs ys

def trainer :=
  Trainer.new model
    { task := .regression
      optimizer := optim.adam { lr := 0.03 }
      dtype := .float
      backend := .eager
      seed := 2026 }

def main : IO Unit := do
  let trained ← trainer.train data
    { steps := 20
      batchSize := 4
      logEvery := 5 }
  trained.printSummary

  let heldout : Tensor.T Float (shape![2]) :=
    tensor! [0.25, -0.75]
  let yhat ← trained.predict heldout
  IO.println s!"prediction={Tensor.pretty yhat}"
```

Check the definitions without starting the training run:

```
lake env lean Scratch.lean
```

Lean normally stays silent here apart from diagnostics. To execute the `main` already present in
the file, run:

```
lake env lean --run Scratch.lean
```

The current deterministic run ends with:

```
dataset size = 4
mean_loss(before) = 0.299386
step 0: loss=0.265440
step 5: loss=0.117008
step 10: loss=0.052596
step 15: loss=0.013994
mean_loss(after) = 0.001102
steps=20 loss0=0.299386 loss1=0.001102
prediction=[0.683571]
```

The loss says that this small network fit the four supplied rows well; the final line is one
prediction at a fifth point, not a generalization claim. The `--run` path is convenient for a
one-file experiment. A maintained command should still receive a Lake executable target so it can
be built and invoked by name.

# Read The Types In VS Code

Place the cursor on `model`. The infoview should show:

```
nn.M (nn.Sequential (shape![2]) (shape![1]))
```

Then change the final layer from `nn.linear 8 1` to `nn.linear 7 1`. The error is attached to model
construction rather than the training call.

Place the cursor on `trained.predict`. Its input and output are trainer-facing Float tensors with the model's
checked shapes. The retained runner handles conversion to the scalar selected by the training
configuration.

# Builder, Trainer, And Trained Handle

These three values have different lifetimes:

## Model builder

```
model : nn.M (nn.Sequential inputShape outputShape)
```

describes architecture and seeded initialization.

## Trainer

```
Trainer.new model config
```

materializes the builder at the selected seed and attaches task, optimizer, and runtime choices. It
has not consumed data.

## Trained handle

```
trainer.train data options
```

executes updates and retains final parameters and prediction closures.

Keeping these objects separate permits the same architecture to be initialized with several seeds,
trained on several datasets, or interpreted by another runtime without redefining its layers.

# Persistent And Per-Call Configuration

Persistent choices can be expressed as `Trainer.RunConfig`:

```
def eagerCpu : Trainer.RunConfig :=
  { optimizer := optim.adam { lr := 0.03 }
    dtype := .float
    backend := .eager }

def compiledCpu : Trainer.RunConfig :=
  eagerCpu.compiled.cpu

def configuredTrainer :=
  Trainer.new model
    (Trainer.Config.fromRunConfig
      compiledCpu .regression
      (seed := 2026))
```

`RunConfig` contains the optimizer, scalar implementation, execution mode, and complete backend
profile. The profile keeps device, providers, evidence policy, and VJP ownership consistent.

Per-call `TrainOptions` controls step count, sample grouping, logging cadence, and artifact fields.
Not every task dispatch consumes every optional field: exact-bits `loadParams?` and `saveParams?`
are currently wired through classification/cross-entropy training, while regression and custom-loss
code should use the direct checkpoint helpers described in the training chapter. `trainWithRun`
applies a temporary runtime configuration for one call.

# DType Means Scalar Semantics

For an ordinary executable trainer, `.float` selects Lean's host `Float`, while `.float32` selects
the bit-level `IEEE32Exec` implementation by default. These are not two labels on the same
untyped buffer: the choice changes the implementation of `+`, `*`, `exp`, reductions, and the other
scalar operations used by the run.

The dtype language also names proof-oriented interpretations. `.real` denotes mathematical reals,
and `.float32 { mode := .fp32 }` denotes the rounded-real binary32-precision model with gradual
underflow but without an upper exponent bound or IEEE special values. They can appear in common
configuration and theorem-facing code, but an IO trainer rejects them because they are
noncomputable. The generic dispatcher can construct executable complex binary32 as well; the
high-level trainer currently rejects complex training and prediction because it has no host-Float
readback path.

One run follows the scalar contract from *Tensors And Shapes*. The comparison with PyTorch explains
the present mixed-precision boundary.

For theorem work, instantiate specification tensors directly over `ℝ` or `FP32`. For a runtime run,
choose an executable scalar and record the provider boundary.

# Data Is Runtime-Polymorphic

`Trainer.Dataset σ τ` knows how to materialize samples after the trainer selects a scalar:

```
Data.tensorDataset
Data.regressionGrid
Data.supervisedNpyDataset
Data.tabularCsvDataset
Data.batchDataset
```

The model and dataset must agree on `σ` and `τ`. A file loader checks runtime dimensions before
constructing the typed dataset.

A true tensor minibatch changes shapes to `[batch,...]`. `TrainOptions.batchSize` on an unbatched
model is a different scheduling choice, as explained in the data chapter.

# Explicit Differentiation

For a tensor function:

```
autograd.func.grad
autograd.func.valueAndGradScalar
autograd.func.vjp
autograd.func.jacfwd
autograd.func.jacrev
autograd.func.hessian
```

For a checked model:

```
autograd.model.gradParams
autograd.model.gradInputs
autograd.model.valueAndGradParamsScalar
autograd.model.vjpParams
autograd.model.jvpParams
autograd.model.hvpParams
```

Derivatives are returned as values. Parameter derivatives have the same dependent tensor-pack
structure as the parameters; there is no mutable `.grad` field on `Tensor.T` values.

# Functional Tensor Operations

Use `nn.functional` when constructing a differentiable tensor program:

```
def energy :
    autograd.func.Fn (shape![4]) Shape.scalar :=
  fun x => do
    let x2 ← nn.functional.square x
    nn.functional.mean x2
```

This program can be differentiated by `autograd.func`. An arbitrary Lean function over
`Spec.Tensor` is useful for specifications but does not automatically carry runtime graph and
derivative behavior.

The distinction is analogous to an embedded differentiable language: operations must register the
semantics needed by execution and AD.

# Classical Models Use The Same Tensor Foundation

The `classical` namespace covers statistical and classical ML models that do not need a neural
layer stack. They still use general tensors, explicit shapes, and declared numerical semantics.

Keeping these models in the library does not require pretending they are neural networks. What
they share with the neural code is the shape-indexed tensor foundation and explicit data, not one
forced architecture abstraction.

Here is a complete executable k-nearest-neighbor classifier. The model stores its labeled samples;
evaluation computes distances and applies a deterministic majority vote, including deterministic
tie-breaking by neighbor order.

```
import NN.API

open TorchLean

def point (x y : Float) : Tensor.T Float (shape![2]) :=
  tensor! [x, y]

def labels : classical.knn.Model Float String 2 :=
  classical.knn.fromData Float String 2 3 [
    (point 0.0 0.0, "blue"),
    (point 0.0 1.0, "blue"),
    (point 3.0 3.0, "orange"),
    (point 3.0 4.0, "orange")
  ]

#eval classical.knn.classify Float String 2 labels (point 0.2 0.1)
-- "blue"
```

The exported families deliberately expose different amounts of fitting and inference machinery:

:::table +header
*
  * Family
  * Available operations
  * Current boundary
*
  * kNN
  * nearest neighbors, classification, regression, confidence, and batch mapping
  * lazy stored-data model; no learned index or metric
*
  * random forest
  * symbolic-tree aggregation plus numeric regression fitting and Gini classification-tree fitting
  * deterministic reference fitting; rotated resamples replace randomized bootstrapping
*
  * naive Bayes
  * multinomial string-feature counting, log scores, prediction, and negative log likelihood
  * specialized to bags of `String` features and labels; fitting is counting, not gradient training
*
  * SVM
  * linear decisions, hinge objective, VJP, gradient-descent fit, prediction, and kernel functions
  * the fitter is a linear primal baseline; exported kernels do not constitute a kernel-SVM solver
*
  * GMM
  * component log densities, responsibilities, VJP, log likelihood, initialization, and EM
  * evaluation is optional and rejects invalid weights or non-positive-definite covariances
*
  * PCA
  * projection, inverse, VJP, reconstruction statistics, and a leading-component fit
  * fitting approximates one component with fixed power iteration; it is not a full SVD-based PCA fit
*
  * linear regression
  * scalar and batched forward/VJP, one gradient step, metrics, and regularized loss variants
  * exposes mathematical update primitives rather than a separate multi-step estimator
*
  * logistic regression
  * deterministic gradient-descent fit, probabilities, and thresholded predictions
  * unregularized binary baseline with explicit sigmoid evaluation, not an optimized solver
*
  * gradient-boosted trees
  * regression ensemble evaluation and boosting steps plus standalone Gini classification-tree fitting
  * bounded, deterministic tree routines; no exported boosted classification ensemble
*
  * HMM
  * scaled and unscaled forward passes, batching, likelihood, initialization, and Baum-Welch updates
  * finite discrete observations; probability normalization is an input invariant, not a type invariant
:::

These definitions execute directly when their scalar `Context` is executable, as `Float` is in the
example. They are pure tensor/reference algorithms rather than automatic `Trainer.new`, compiled
graph, LibTorch, or CUDA routes. Shape indices rule out dimensional mismatches, but they do not by
themselves prove statistical assumptions, optimizer convergence, or a family-wide correctness
claim; consult each declaration's hypotheses and result type, especially `Option`-returning GMM and
HMM likelihood operations.

# When To Import More

Use:

```
import NN
```

when a file genuinely needs several lower layers, such as model code plus proof declarations and
backend inspection.

Focused subsystem imports include:

```
import NN.Spec
import NN.Runtime
import NN.Floats
import NN.Verification
import NN.GraphSpec
```

Prefer the narrowest stable import that expresses the file's responsibility. A numerical theorem
should not import the entire executable model zoo merely for convenience, and a training script
should not depend on an internal tape constructor.

# API Boundaries Are Semantic

These objects may all refer to the same architecture:

:::table +header
*
  * Object
  * What it says
*
  * model declaration
  * layer structure and shapes
*
  * trainer run
  * one runtime configuration executed
*
  * `NN.IR.Graph`
  * explicit operation data
*
  * backend audit
  * provider and evidence choices
*
  * theorem
  * exactly one Lean proposition under hypotheses
*
  * certificate
  * accepted external claim plus checker theorem
:::

The TorchLean API makes the common path concise without collapsing these meanings.

# Find A Declaration

Use the generated API search:

```
/docs/search.html
```

The path works on the published site and on a local site preview. For source search:

```
rg -n "def valueAndGradParamsScalar|theorem .*sound" NN
```

The API reference answers “what is the exact declaration?” The surrounding chapters explain why
and when to use it. Source remains authoritative when a lower-level contract matters.

# Continue From Here

Run:

```
lake exe torchlean --help
lake exe torchlean quickstart_tensors
lake exe torchlean quickstart_autograd
lake exe torchlean quickstart_mlp --steps 20
```

These four commands cover the tensor, derivative, model, dataset, trainer, and prediction
interfaces without requiring backend or proof internals.

Lean's
[source-file and module reference](https://lean-lang.org/doc/reference/latest/Source-Files-and-Modules/)
explains how these imports determine the environment in which a file elaborates. With the common
path now in one place, the next chapter keeps this same model and changes only how it executes.
