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
  -> trained handle
```

The loss values are measurements from this run. The model and optimizer definitions, by contrast,
are reusable objects that can appear in theorem statements or another runtime profile.

# Declare The Architecture

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
      dtype := .float
      backend := .eager
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

Changing `.regression` to `.crossEntropy` changes the objective and target convention without
changing the architecture. A custom task supplies a checked scalar loss program.

# Build The Dataset

The quickstart uses a deterministic grid. A four-point example can be written directly:

```
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

  let heldout : Tensor.T Float (shape![2]) :=
    tensor! [0.25, -0.75]
  let yhat ← trained.predict heldout
  IO.println s!"prediction={Tensor.pretty yhat}"
```

The returned handle retains:

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
    nn.M
      (nn.Sequential
        (shape![2, 2])
        (shape![2, 1])) :=
  nn.Sequential![
    nn.linear 2 8,
    nn.relu,
    nn.linear 8 1
  ]

def batchedData :
    Trainer.Dataset (shape![2, 2]) (shape![2, 1]) :=
  Data.batchDataset 2 data
    (shuffle := true)
    (seed := 2026)
```

Each item in `batchedData` now holds two samples in one tensor. With the default
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

# Eager And Compiled Are Execution Choices

Compare:

```
lake exe torchlean quickstart_mlp \
  --device cpu --backend eager --steps 20 --seed 2026

lake exe torchlean quickstart_mlp \
  --device cpu --backend compiled --steps 20 --seed 2026
```

Eager mode records operations and local reverse rules as they execute. Compiled mode builds a typed
forward/derivative graph and replays it with current parameters and inputs.

The trainer method remains `train`; compilation is a property of the configured runner, analogous to
wrapping a PyTorch model with an execution transform rather than renaming the model's `forward`
method.

Compiled trainer execution is currently CPU-only. A non-CPU compiled request is rejected rather
than silently falling back to a different semantics.

# Device Selection Is A Profile

The execution profile records more than a device enum:

- target device and platform;
- provider preference;
- numerical and assurance policy;
- forward and VJP ownership;
- available kernel capsules.

Programmatically:

```
def cudaRun : Trainer.RunConfig :=
  ({ optimizer := optim.adam { lr := 0.03 }
     dtype := .float
     backend := .eager } :
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

`--show-backend` prints the selected capsules as operations first execute. This is how a run reports
which provider owned matmul, ReLU, loss, and their VJPs.

Named future devices such as Metal, ROCm, TPU, or Trainium may be represented in configuration, but
`withDevice` fails if the current build has no maintained profile. A name in an enum is not an
implementation.

# Scalar Semantics Are Part Of The Run

The common executable selections are:

```
--dtype float
--dtype ieee754exec
```

Host `Float` is fast and relies on the platform runtime. `IEEE32Exec` is TorchLean's bit-level
binary32 reference and is much slower, but exposes precise finite and exceptional behavior.
Proof-level `Real` and rounded-real `FP32` are not executable trainer choices. `FP32` has binary32
precision and gradual-underflow parameters, but no upper exponent bound, NaN, infinity, or signed
zero; bridge theorems therefore require finite/no-overflow hypotheses when relating it to
`IEEE32Exec`.

A loss curve without its scalar semantics is incomplete. The same architecture and seed may round
differently in binary32, binary64, a fused CUDA kernel, or an external provider.

# A Checkpoint Is A Particular Slice Of State

TorchLean's JSON parameter checkpoint records each host `Float` with `Float.toBits`, so saving and
loading does not pass through a decimal approximation. The expected parameter shapes come from the
model. Each expected tensor must be present with the right shape and scalar count before it becomes
a parameter pack:

```
def loadForThisModel (path : System.FilePath) :=
  Checkpoint.loadModelParamBits (nn.run 2026 model) path
```

The result is an `IO` action returning tensors whose dependent shape list is exactly
`nn.paramShapes (nn.run 2026 model)`. `Checkpoint.toRuntimeParams` turns such a checked pack into
runtime parameter handles, while `Checkpoint.loadModuleParamBits` and
`Checkpoint.saveModuleParamBits` work with an already instantiated host-Float module.

One boundary remains worth knowing: after all expected tensors have been decoded, the current
loader ignores extra trailing tensors in the JSON array. Missing tensors and malformed expected
tensors are rejected, but the file length alone is not yet a strict architecture identifier. If
that distinction matters, validate the artifact's tensor manifest before loading it.

Classification and cross-entropy training also wire the `loadParams?` and `saveParams?` fields of
`Trainer.TrainOptions` into this exact-bits format. For example, the following options ask that path
to save parameters after 200 updates:

```
def saveClassifierParams : Trainer.TrainOptions :=
  { steps := 200
    logEvery := 25
    saveParams? := some "artifacts/classifier.params.json" }
```

The current high-level regression and custom-loss training paths do not consume those two fields;
use the direct `Checkpoint`/`NN.API.TorchLean.ParamIO` helpers with a manual module for those tasks.
This limitation is worth spelling out because accepting a shared options record is not evidence
that every task dispatch implements every optional field.

Even on the classification path, this is a *parameter* checkpoint, not a complete training
snapshot. An exact continuation of Adam also needs its moment tensors and step number. Replaying the
next batch needs loader or stream position; stochastic layers need generator state; and interpreting
the result needs the model, preprocessing, scalar semantics, backend profile, and device. A
parameter-only checkpoint is excellent for inference or a fresh optimizer run, but it should not
be described as resuming the same optimizer trajectory.

# Manual Training

The high-level trainer is intended for common runs. `Trainer.Manual` exposes:

```
stepper
step
trainMode
evalMode
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

The manual configuration has constant, step-decay, and exponential schedules. Here a rate of
`0.1` is halved after every three step indices:

```
def decay :=
  NN.API.TorchLean.Schedulers.step 0.1 3 0.5

#eval (List.range 10).map
  (NN.API.TorchLean.Schedulers.lrAt decay)
```

Lean prints:

```
[0.100000, 0.100000, 0.100000, 0.050000, 0.050000, 0.050000, 0.025000, 0.025000, 0.025000, 0.012500]
```

Attach the same schedule to a manual step configuration with `Trainer.stepLR`:

```
def scheduledManualRun :=
  Trainer.stepLR
    (Trainer.steps 10 (optim.sgd { lr := 0.1 }))
    0.1 3 0.5
```

The counter is zero-indexed, which is why the first decay occurs at index three. A step schedule
with `stepSize := 0` deliberately stays at its base rate instead of dividing by zero. The compact
high-level `Trainer.TrainOptions` does not currently carry a scheduler; use this manual
configuration when schedule state must be part of the loop.

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
- equality of eager, compiled, CUDA, and LibTorch paths;
- correctness of every native instruction.

TorchLean gives the run enough structure for a theorem, numerical bound, backend contract, or
verification certificate to refer
to the same model without erasing the boundary between them.
