import VersoManual

open Verso.Genre Manual

#doc (Manual) "Choosing How A Model Runs" =>
%%%
tag := "execution-modes"
%%%

Start with the same MLP:

$$`F_\theta:[2]\to[1]`

Its type stays `[2] → [1]` whether it runs eagerly on the CPU, through a compiled graph, or with
native CUDA kernels. We can therefore keep the architecture, seed, and data fixed while changing
one runtime choice at a time.

There are four independent questions. Which scalar operations give meaning to the numbers? Is the
forward/loss program recorded once or rebuilt eagerly? Which device and providers carry out its
primitive operations? Is the runner training or evaluating? Treating these as separate questions
is useful: moving to CUDA should not silently turn dropout from training behavior into evaluation
behavior, and selecting binary32 should not secretly choose another model architecture.

The experiments below answer each question by asking the runner what it actually did.

# Ask The Runner

The example runner documents the flags it currently accepts:

```
lake exe torchlean --help
```

For one command:

```
lake exe torchlean quickstart_mlp --help
```

The quickstart's common flags are:

```
--dtype float|ieee754exec
--backend eager|compiled
--device auto|cpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external
--seed N
--show-backend
```

The canonical executable names are `float` and `ieee754exec`; the shared parser also accepts the
older `float32` and `ieee32` aliases. Individual commands may support only one of these choices.

The parser knows more device names than the current runtime implements. CPU and CUDA have maintained
profiles today; the other names reserve a clean place for future providers. Asking for one of them
gets an error rather than a suspiciously successful CPU run.

For an interactive device prompt:

```
lake exe torchlean --choose quickstart_mlp --steps 20
```

The prompt is opt-in so scripts and CI never block waiting for input.

# Experiment 1: CPU Eager

```
lake exe torchlean quickstart_mlp \
  --dtype float \
  --backend eager \
  --device cpu \
  --steps 20 \
  --seed 2026 \
  --show-backend
```

Eager execution creates a session and records operations as the model runs. Every operation asks the
profile for an admissible capsule, executes its provider, and appends a local VJP rule when gradients
are required.

On CPU, the maintained profile selects portable reference capsules. The report lets you verify that
the requested CPU path actually ran.

Eager mode is the natural starting point when operation structure depends on runtime values, when
you want to inspect the tape or provider choices, or when you are using the maintained CUDA
runtime. It also accepts more dynamic frontend programs than the fixed compiled recorder.

# Experiment 2: CPU Compiled

```
lake exe torchlean quickstart_mlp \
  --dtype float \
  --backend compiled \
  --device cpu \
  --steps 20 \
  --seed 2026
```

Compiled execution records the fixed scalar-loss program once, including forward, JVP, and VJP
behavior, and replays it with current parameters and data.

Compare the initial loss between eager and compiled using the same seed. It should agree for the
supported deterministic program. Then compare final parameters or predictions, not only
six-decimal loss summaries, because different execution orders can hide small discrepancies.

The current compiled trainer is CPU-only. It does not consume an accepted backend graph plan and it
does not mean CUDA Graph capture. A CUDA plus compiled request fails explicitly.

# Experiment 3: Native CUDA

Build and run:

```
lake -R -K cuda=true exe torchlean quickstart_mlp \
  --dtype float \
  --backend eager \
  --device cuda \
  --steps 20 \
  --seed 2026 \
  --show-backend
```

The CUDA profile selects native capsules for supported operations. The report names reshape,
permutation, matrix multiplication, broadcasting, addition, ReLU, and MSE providers as they are
first used.

CUDA currently requires host `Float` at the TorchLean module boundary. The native tensors use
device-side Float32 storage and operations according to their capsules. This is a runtime boundary,
not an identification of Lean `Float`, mathematical `FP32`, and CUDA `float`.

If the project is built without CUDA support, requesting CUDA fails. The stub archives permit the
repository to build on CPU-only systems; they do not pretend to execute GPU code.

# Experiment 4: Executable IEEE Binary32

```
lake exe torchlean quickstart_mlp \
  --dtype ieee754exec \
  --backend eager \
  --device cpu \
  --steps 2 \
  --seed 2026
```

This uses TorchLean's explicit bit-level `IEEE32Exec` scalar model. It is intentionally slower and
best used for small reference runs and numerical experiments.

The proof-oriented `FP32` model and exact `Real` live in theorem statements rather than the IO
trainer. `FP32` rounds on reals using binary32 precision and gradual-underflow parameters, but it
does not model overflow, NaN, infinity, or signed zero. The floating-point chapter shows how the
finite/no-overflow bridge to `IEEE32Exec` is stated.

The lower dtype dispatcher also recognizes executable complex binary32. The high-level trainer
currently rejects it because prediction has no Float readback path. All supported selections follow
the one-scalar-per-run contract from *Tensors And Shapes*.

# The Same Choices In Lean

```
def eagerCpu : Trainer.RunConfig :=
  { dtype := .float
    backend := .eager
    optimizer := optim.adam { lr := 0.03 } }

def compiledCpu : Trainer.RunConfig :=
  eagerCpu.compiled.cpu

def eagerCuda : Trainer.RunConfig :=
  eagerCpu.cuda
```

Attach a run configuration to a task and seed:

```
def trainerFromRun (run : Trainer.RunConfig) :=
  Trainer.new model
    (Trainer.Config.fromRunConfig
      run .regression
      (seed := 2026))
```

`trainWithRun` can apply a temporary per-call runtime override without rebuilding the model
declaration.

# Why Device Is Part Of A Profile

A device choice affects more than memory location. The associated profile says which providers the
planner should prefer, which evidence policy a capsule must satisfy, and whether TorchLean or the
provider owns each VJP. It also records the target operating system and architecture together with
the capsule modules available in this build.

Selecting only `.cuda` while retaining CPU provider assumptions would be an inconsistent
configuration. `RunConfig.withDevice` therefore installs a maintained profile as one value or
returns an error.

Custom and optional LibTorch paths use `withBackendProfile`, making the larger boundary explicit.

# Train And Evaluation Mode

Mode-sensitive layers include dropout and normalization:

```
Trainer.Manual.trainMode runner
Trainer.Manual.evalMode runner
Trainer.Manual.isTraining runner
```

Training mode may sample masks or update running statistics. Evaluation mode uses the corresponding
inference behavior. The high-level trainer enters training mode for updates and evaluation mode for
summary predictions and retained prediction handles.

Mode is independent of device and eager/compiled choice. A CUDA runner can switch mode without
changing model architecture or provider profile.

# A Small Dropout Thought Experiment

Suppose:

$$`y=\operatorname{Dropout}_{p}(x)`.

During training, a random mask is realized and retained for the backward rule. During evaluation,
the operation follows its deterministic inference semantics. Re-running the backward pass with a
newly sampled mask would not differentiate the forward value that was computed.

This is why RNG and mode belong to runtime state and to reproducible checkpoints.

# Dynamic Operations And Compilation

A fixed compiled graph needs operation structure and shapes known when recording. If a program
reads token values and changes the graph structure while constructing it, the current `GraphM`
compiler cannot represent that program as one fixed replay.

The correct response is not to coerce the values into a graph and hope. Keep genuinely dynamic
control flow in eager mode, represent the choice as a supported tensor operation, or compile
separate static branches and choose between them explicitly at runtime.

Unsupported compiled operations are rejected.

# Selecting A LibTorch Provider

LibTorch is an optional provider inside an eager backend profile. The maintained bridge currently
accelerates scaled-dot-product attention forward while TorchLean retains its tape and local
backward ownership.

Surrounding operations may still use native CUDA or reference capsules. Provider selection is
per semantic operation.

The next backend chapter opens the capsule and its evidence fields. At runtime, one rule matters
immediately: training cannot choose a forward-only capsule unless the profile also supplies an
admissible VJP path.

# Unsupported Means Failure

Try:

```
lake exe torchlean quickstart_mlp \
  --device metal --steps 1
```

on the current checkout. The target name is parsed, but profile selection rejects it. This confirms
that a future platform vocabulary is not reported as working implementation.

The same rule covers other unsupported combinations. CUDA in a CPU-only build, compiled mode with a
non-CPU profile, proof-only scalar semantics in `IO`, and an operation with no admissible capsule
all fail rather than changing the requested configuration behind the caller's back.

These failures protect benchmark provenance. “Requested GPU” must never become an unreported CPU
run.

# A Practical Selection Table

:::table +header
*
  * Goal
  * Scalar
  * Mode
  * Profile
*
  * inspect ordinary training
  * `Float`
  * eager
  * CPU
*
  * replay a supported fixed graph
  * `Float`
  * compiled
  * CPU
*
  * run native GPU training
  * `Float`
  * eager
  * CUDA
*
  * inspect binary32 reference behavior
  * `IEEE32Exec`
  * eager
  * CPU
*
  * use external attention forward
  * `Float`
  * eager
  * LibTorch-enabled CUDA
*
  * verify/export an operation graph
  * semantic context
  * IR evaluator
  * no trainer profile
:::

The final row is deliberately outside the trainer modes. Lowering a model to `NN.IR.Graph` creates
an inspectable semantic artifact, not another high-performance runtime switch.

# Record The Choice With Results

A useful run report includes:

```
model architecture and parameter count
dataset identity and preprocessing
seed and optimizer
scalar semantics
eager or compiled mode
device and provider capsules
train/eval mode
checkpoint and code revision
```

Without this information, two loss curves may be incomparable even when both are labeled
“TorchLean float32.”

Sources:

- [NN/API/Runtime/Module.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Runtime/Module.lean);
- [Core/Types.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Torch/Core/Types.lean);
- [Core/Trainer.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Torch/Core/Trainer.lean).
