import VersoManual

open Verso.Genre Manual

#doc (Manual) "The Public API" =>
%%%
tag := "torchlean-api"
%%%

Most user programs need two lines:

```
import NN.API
open TorchLean
```

`NN.API` is the maintained application surface. It imports tensors, models, data, training,
optimizers, prediction, and explicit differentiation. It does not pull every proof, verifier,
floating-point implementation, widget, and backend-internal module into a small training file.

This chapter builds one program using only that public surface, then explains when a narrower or
lower import is appropriate.

# The Namespaces

The main public namespaces are:

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
  * `classical`
  * classical and statistical model APIs
:::

The lowercase `nn.linear`, `nn.relu`, and `optim.adam` names are the canonical public spellings.
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

Check it:

```
lake env lean Scratch.lean
```

To run a standalone `main`, add a temporary executable target or adapt the maintained
`quickstart_mlp` example. Compiling the scratch file already checks all model, data, and API types.

# Read The Types In VS Code

Place the cursor on `model`. The infoview should show:

```
nn.M (nn.Sequential (shape![2]) (shape![1]))
```

Then change the final layer from `nn.linear 8 1` to `nn.linear 7 1`. The error is attached to model
construction rather than the training call.

Place the cursor on `trained.predict`. Its input and output are public Float tensors with the model's
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

Per-call `TrainOptions` controls step count, logging cadence, artifacts, and checkpoint choices.
`trainWithRun` applies a temporary run configuration for one call.

# DType Means Scalar Semantics

The public trainer choices include:

- `.float`: executable Lean host `Float`;
- `.float32`: executable bit-level `IEEE32Exec` by default;
- `.float32 { mode := .fp32 }`: noncomputable finite rounded-real proof model;
- `.real`: noncomputable mathematical reals.

Executable training rejects proof-only choices. The architecture shape is unchanged, but the
meaning of `+`, `*`, `exp`, and reductions changes with the scalar implementation.

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
structure as the parameters; there is no mutable `.grad` field on public tensors.

# Public Functional Operations

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

Keeping these models in the public library does not require pretending they are neural networks.
The common surface is data and tensor structure, not one forced architecture abstraction.

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

The public API makes the common path concise without collapsing these meanings.

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

These four commands cover the public tensor, derivative, model, dataset, trainer, and prediction
surfaces without requiring backend or proof internals.

Lean's
[source-file and module reference](https://lean-lang.org/doc/reference/latest/Source-Files-and-Modules/)
explains how these imports determine the environment in which a file elaborates.
