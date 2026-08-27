import VersoManual

open Verso.Genre Manual

#doc (Manual) "Training, One State Transition At A Time" =>
%%%
tag := "training-from-scratch"
%%%

Training repeatedly transforms parameters, optimizer memory, data order, and runtime state.
TorchLean's high-level trainer packages that transition, while the lower manual API exposes each
piece.

We will train the running $`2\to8\to1` MLP and then unpack what happened.

# The Smallest Complete Run

Execute:

```
lake exe torchlean quickstart_mlp \
  --device cpu --steps 200 --seed 2026
```

The current checkout reports:

```
dataset size = 25
mean_loss(before) = 0.761530
mean_loss(after) = 0.003234
heldout x=(0.25,-0.75), target=0.2,
prediction(after)=[0.210239]
```

The source is
[`SimpleMlpTrain.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleMlpTrain.lean).
Its program structure is:

```
model builder
  + trainer configuration
  + dataset
  + train options
  -> trained result
```

The loss values are measurements from this run. The model and optimizer definitions, by contrast,
are reusable objects that can appear in theorem statements or another runtime profile.

# Declare The Architecture

```
import NN.API
open TorchLean

def model :
    nn.Builder (nn.Sequential [2] [1]) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]
```

At this point we have:

- input and output shapes;
- layer structure;
- parameter shapes;
- seeded initialization actions.

We do not yet have a loss, optimizer, concrete parameter values, or device.

# Attach A Training Problem

```
def trainer (seed : Nat) :=
  Trainer.new model
    { task := .regression
      optimizer := optim.adam { lr := 0.03 }
      scalar := .float32
      execution := .eager
      seed := seed }
```

`Trainer.new` runs the seeded builder and stores persistent choices. It still does not consume data
or update a parameter.

For regression, the default objective is mean-squared error. If a prediction and target each have
$`n` entries:

$$`
L(\theta;x,y)
=
\frac1n\sum_{i=1}^{n}
\left(F_\theta(x)_i-y_i\right)^2.
`

Changing `.regression` to `.oneHotCrossEntropy axis` changes the objective and target convention
without changing the architecture. The zero-based `axis` may name any output dimension; Lean
rejects an axis outside the output shape. A custom task supplies a checked scalar loss program.

# Build The Dataset

The quickstart uses a deterministic grid. A four-point example can be written directly:

```
def xs : Tensor Float [4, 2] :=
  tensor! [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
  ]

def ys : Tensor Float [4, 1] :=
  tensor! [[0.0], [1.0], [1.0], [0.0]]

def data := Data.tensorDataset xs ys
```

The dataset type matches the model map `[2] → [1]`. A batched model would require a batched dataset
with a leading dimension in both item shapes.

# Call Train And Keep The Result

```
def run : IO Unit := do
  let trained ← (trainer 2026).train data
    { steps := 200
      batchSize := 4
      logEvery := 25 }

  trained.printSummary

  let heldout : Tensor Float [2] :=
    tensor! [0.25, -0.75]
  let yhat ← trained.predict heldout
  IO.println s!"prediction={Tensor.pretty yhat}"
```

The returned `TrainResult` retains:

- final parameters;
- runtime model state;
- before/after summary;
- prediction and verification closures.

Prediction accepts Float tensors and performs the runtime scalar conversion selected by the
trainer. It does not rebuild or reinitialize the model.

# What One Update Computes

Let $`\theta_t` be the parameter pack at update $`t`. A plain SGD update is:

$$`
g_t=\nabla_\theta L(\theta_t;x_t,y_t),
\qquad
\theta_{t+1}=\theta_t-\eta g_t.
`

Every symbol is structured:

- $`\theta_t` is a shape-indexed pack of differently shaped tensors sharing scalar type `α`;
- $`g_t` has exactly the same pack structure;
- $`L` is a scalar tensor program;
- the subtraction and scaling occur coordinatewise in the selected scalar semantics.

An optimizer is therefore not merely a function from a flat vector to a flat vector. It owns
shape-aligned state.

# The State Carried Between Updates

It is tempting to describe training as repeated calls to `backward`, but the value passed from one
update to the next is larger than a gradient. For a reproducible run we need to account for:

:::table +header
*
  * State
  * Why the next update needs it
*
  * parameters
  * they are the point at which the next loss and gradient are evaluated
*
  * optimizer memory
  * Adam moments, momentum buffers, and step counters change the update
*
  * scheduler state
  * the update index determines the learning rate
*
  * loader or stream position
  * it determines the next samples and final partial-batch behavior
*
  * random-generator state
  * dropout, augmentation, sampling, and some data sources consume it
*
  * model buffers and mode
  * normalization statistics and other persistent state may change in training mode
*
  * backend profile
  * it fixes the providers and backward ownership used by the executable step
:::

The high-level API keeps these pieces in its trainer and result values. The manual API exposes them
when an experiment needs a custom loop or a checkpoint must record more than parameter tensors.
Saving only weights is enough for inference, but it is not enough to resume Adam at the same
update.

# Adam's Hidden State Is Explicit

Adam maintains first and second moment estimates:

$$`
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
`

$$`
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
`

With bias correction:

$$`
\widehat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\widehat v_t=\frac{v_t}{1-\beta_2^t},
`

and the parameter update is:

$$`
\theta_{t+1}
=
\theta_t-\eta
\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
`

The two moment packs have the same dependent tensor shapes as $`\theta`. The step counter matters
because it changes the bias correction. Restoring only parameters from a checkpoint but not Adam
state is not the same continuation of training.

TorchLean also provides SGD, momentum SGD, AdamW, AdaGrad, RMSProp, Adadelta, and Muon-related
runtime configuration. Their constructors share one trainer interface; their state and laws
remain optimizer-specific.

# What Does `steps` Count?

For the unbatched model `[2] → [1]`, this configuration:

```
steps := 200
batchSize := 4
```

means 200 optimizer updates. Each update consumes four samples, differentiates them at the same
parameter point, and averages their gradient packs. Logging reports the mean pre-update loss from
those same forward tapes.
The trainer does not run an extra forward pass merely to print that loss. Before building those
tapes, it advances mutable model buffers once per item; trainable parameters stay fixed until the
mean gradient is ready. This matters for dropout, BatchNorm, and any operation whose tape retains
data needed by backward: the displayed scalar, saved state, and gradient come from the same step.

For a true vectorized minibatch, define:

```
def batchedModel :
    nn.Builder
      (nn.Sequential
        [2, 2]
        [2, 1]) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]

def batchedData :
    Trainer.Dataset [2, 2] [2, 1] :=
  Data.batchDataset 2 data
    (shuffle := true)
    (seed := 2026)
```

Each item in `batchedData` holds two samples in one tensor. With the default
`TrainOptions.batchSize := 1`, an update consumes one item and therefore runs one vectorized
forward/backward operation over two samples. The four-row example produces two full items; asking
`Data.batchDataset` for groups of five would produce none because typed batching drops incomplete
groups. Setting `TrainOptions.batchSize` above one would accumulate gradients across several of
these tensor minibatches. Vectorization can change reduction order and performance; it is not an
optimization flag applied to the first model.

Run the maintained version:

```
lake exe torchlean quickstart_minibatch_mlp \
  --device cpu --batch 5 --steps 5 --seed 2026
```

The printed model confirms:

```
Sequential: [5, 2] -> [5, 1], layers=3, params=33
```

The parameter count remains 33 because linear layers preserve the batch prefix; the batch axis does
not create separate weights per sample.

# Eager And Typed Graph Are Execution Choices

Compare:

```
lake exe torchlean quickstart_mlp \
  --device cpu --execution eager --steps 20 --seed 2026

lake exe torchlean quickstart_mlp \
  --device cpu --execution typed-graph --steps 20 --seed 2026
```

The trainer method remains `train`; `execution := .typedGraph` changes how that method runs without
changing the model's `forward` definition. The execution chapter develops the graph reuse and proof
boundaries in detail.

Typed graph trainer execution is currently CPU-only. A non-CPU typed graph request is rejected rather
than silently falling back to a different semantics.

# Device Selection Is A Profile

Programmatically:

```
def cudaRun : Trainer.RunConfig :=
  ({ optimizer := optim.adam { lr := 0.03 }
     scalar := .float32
     execution := .eager } :
    Trainer.RunConfig).cuda

def cudaTrainer :=
  Trainer.new model
    (Trainer.Config.fromRunConfig
      cudaRun .regression
      (seed := 2026))
```

A maintained CUDA run requires a CUDA-enabled build:

```
lake -R -K cuda=true exe torchlean quickstart_mlp \
  --device cuda --steps 20 --seed 2026 --show-backend
```

`--show-backend` prints the selected implementation for each operation. The
[backend chapter](Runtime___-Autograd___-and-Interop/Inside-The-Backend-Planner/) explains the
profile, provider, VJP, and evidence fields in that report.

Named future devices such as Metal, ROCm, TPU, or Trainium may be represented in configuration, but
`withDevice` fails if the current build has no maintained profile. A name in an enum is not an
implementation.

# Scalar Semantics Are Part Of The Run

The common executable selections are:

```
--scalar float32
--scalar ieee32-exec
```

Native `Float32` uses Lean's builtin binary32 operations. `IEEE32Exec` is TorchLean's independent
raw-bit binary32 reference and is much slower, but exposes precise finite and exceptional behavior.
Proof-level `Real` and rounded-real `FP32` are not executable trainer choices. `FP32` has binary32
precision and gradual-underflow parameters, but no upper exponent bound, NaN, infinity, or signed
zero; bridge theorems therefore require finite/no-overflow hypotheses when relating it to
`IEEE32Exec`.

A loss curve without its scalar semantics is incomplete. The same architecture and seed may round
differently in native binary32, the bit-level reference, a fused CUDA kernel, or an external
provider.

# A Checkpoint Is A Particular Slice Of State

Native `Float32` modules use an exact binary32 checkpoint on CPU and CUDA. The binary64 `Float`
path retains its exact-bit JSON format on CPU. Neither representation passes
through decimal text. The expected state shapes come from the model, and every tensor must have the
right shape and scalar count before the checkpoint is accepted:

```
def loadForThisModel (path : System.FilePath) :=
  Checkpoint.loadModelState (nn.build 2026 model) path
```

The result is an `IO` action returning tensors whose dependent shape list is exactly
`nn.stateShapes (nn.build 2026 model)`. The runtime checkpoint loader turns such a checked pack into
runtime state handles, while `Checkpoint.loadModule` and
`Checkpoint.saveModule` work with an already instantiated runtime module.

The loader requires an exact tensor manifest. Missing tensors, extra tensors, shape mismatches, and
malformed scalar payloads are rejected before any state is installed. This matters when two models
share an initial state prefix: a checkpoint for the larger model cannot be accepted as a checkpoint
for the smaller one merely because the first few shapes agree.

Classification and cross-entropy training also wire the `loadCheckpoint?` and `saveCheckpoint?`
fields of `Trainer.TrainOptions` into the checkpoint path selected by the runtime scalar. For
example:

```
def saveClassifier : Trainer.TrainOptions :=
  { steps := 200
    logEvery := 25
    saveCheckpoint? := some "artifacts/classifier.state.json" }
```

The current high-level regression and custom-loss training paths do not consume those two fields;
use the direct `Checkpoint`/`TorchLean.Checkpoint` helpers with a manual module for those tasks.
This limitation is worth spelling out because accepting a shared options record is not evidence
that every task dispatch implements every optional field.

Even on the classification path, this is a *model-state* checkpoint, not a complete training
snapshot. The eager CUDA runtime can save Adam or AdamW moments and step counters separately with
`Checkpoint.saveOptimizerState`, then restore them with
`Checkpoint.loadOptimizerState`. That binary file records the optimizer kind, the
moment-defining hyperparameters, every parameter shape, and the `requiresGrad` mask. Loading rejects
a different optimizer configuration or parameter schema instead of silently attaching moments to
the wrong model. Integer metadata and float32 payloads use explicit little-endian encodings, and a
save is written to a fresh sibling file before it replaces the destination.

That optimizer file is still not a complete training snapshot. Replaying the next batch also needs
the loader or stream position; stochastic layers need generator state; and interpreting the result
needs the model, preprocessing, scalar semantics, backend profile, and device. A parameter-only
checkpoint remains appropriate for inference or a fresh optimizer run. Pairing it with native
optimizer state resumes more of an Adam trajectory, but only the state explicitly present in those
two files.

# Manual Training

The high-level trainer is intended for common runs. `Trainer.Manual` exposes:

```
stepper
step
Runner.train
Runner.eval
callbacks
loader loops
prediction
```

Use it when the program needs a custom accumulation policy, custom scheduling, multiple losses,
generated batches, reinforcement-learning interaction, or detailed instrumentation.

`Trainer.Manual.StepBatchStream α shapes` supplies already collated tensors as a function of the
step. PINN collocation points and simulator batches naturally fit this interface.

The lower API does not change the model or autograd semantics. It exposes the runner state that the
high-level trainer normally manages.

## Learning-rate schedules

The manual configuration has constant, step-decay, exponential, and warmup-cosine schedules. Here
a rate of `0.1` is halved after every three step indices:

```
def decay :=
  _root_.TorchLean.Trainer.Scheduler.step 0.1 3 0.5

#eval (List.range 10).map
  (_root_.TorchLean.Trainer.Scheduler.lrAt decay)
```

Lean prints:

```
[0.100000, 0.100000, 0.100000, 0.050000, 0.050000, 0.050000, 0.025000, 0.025000, 0.025000, 0.012500]
```

Attach the same schedule to a manual step configuration with `Trainer.stepLr`:

```
def scheduledManualRun :=
  Trainer.stepLr
    (Trainer.steps 10 (optim.sgd { lr := 0.1 }))
    0.1 3 0.5
```

The counter is zero-indexed, which is why the first decay occurs at index three. A step schedule
with `stepSize := 0` deliberately stays at its base rate instead of dividing by zero. The compact
high-level `Trainer.TrainOptions` does not currently carry a scheduler; use this manual
configuration when schedule state must be part of the loop.

Transformer runs commonly warm up from a small rate and then decay toward a nonzero floor. This
configuration reaches `0.0006` after 2,000 updates and follows a cosine curve toward `0.00006`:

```
def pretrainingSchedule :=
  _root_.TorchLean.Trainer.Scheduler.warmupCosine 0.0006 0.00006 2000 162761

def scheduledPretraining :=
  Trainer.warmupCosineLR
    (Trainer.steps 162761 (optim.adamW { lr := 0.0006 }))
    0.0006 0.00006 2000 162761
```

The first update uses `peak / warmupSteps`; the last warm-up update reaches `peak`. At and after
`totalSteps`, `lrAt` returns the floor exactly. The scheduler changes optimizer state only. It does
not depend on a particular model, loss, dataset, or device. If the requested warm-up is longer than
the run, TorchLean clamps it to `totalSteps`.

# Four Useful Experiments

## Initialization only

```
lake exe torchlean quickstart_mlp \
  --device cpu --steps 0 --seed 2026
```

This isolates initialization and the initial loss.

## Seed sensitivity

Run 20 steps with seeds `2026`, `2027`, and `2028`. Keep the dataset order fixed if you want to
study only initialization.

## Optimizer sensitivity

In the quickstart source, replace Adam with SGD while keeping the model, seed, and steps fixed.
Compare both the initial and final losses; the initial values should agree when initialization and
data order agree.

## Backend report

Add `--show-backend`. On CPU, inspect the reference capsules. On a CUDA build, inspect which native
capsules are selected and whether any trusted external provider appears.

# What Training Establishes

A successful run establishes that one configured pipeline executed:

```
data -> forward -> loss -> reverse pass -> optimizer -> parameters
```

It can produce valuable evidence:

- loss and prediction traces;
- exact parameter artifacts;
- capsule audit rows;
- reproducible configuration;
- runtime errors or successful completion.

It does not automatically prove:

- convergence for all initializations;
- generalization to unseen data;
- robustness to an input region;
- equality of eager, typed graph, CUDA, and LibTorch paths;
- correctness of every native instruction.

TorchLean gives the run enough structure for a theorem, numerical bound, backend contract, or
verification certificate to refer
to the same model without erasing the boundary between them.
