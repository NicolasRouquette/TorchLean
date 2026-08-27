import VersoManual

open Verso.Genre Manual

#doc (Manual) "Choosing How A Model Runs" =>
%%%
tag := "execution-modes"
%%%

Start with the same MLP:

$$`F_\theta:[2]\to[1]`

Its type stays `[2] → [1]` whether it runs eagerly on the CPU, through a typed graph, or with
native CUDA kernels. We can therefore keep the architecture, seed, and data fixed while changing
one runtime choice at a time.

There are four independent choices: scalar semantics, execution mode, device and providers, and
training or evaluation mode. Moving to CUDA should not silently turn dropout from training behavior
into evaluation behavior, and selecting binary32 should not change the model architecture.

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
--scalar float32|ieee32-exec
--execution eager|typed-graph
--device auto|cpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external
--seed N
--show-backend
```

`float32` is the native binary32 runtime. `ieee32-exec` selects the independent bit-level reference
implementation. Individual commands may support only a subset of these choices.

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
  --scalar float32 \
  --execution eager \
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
runtime. It also accepts more dynamic frontend programs than the fixed typed graph recorder.

# Experiment 2: CPU Typed Graph

```
lake exe torchlean quickstart_mlp \
  --scalar float32 \
  --execution typed-graph \
  --device cpu \
  --steps 20 \
  --seed 2026
```

For this trainer path, typed graph execution records the fixed scalar-loss program once as a typed
SSA graph, including forward, JVP, and VJP behavior, then reuses it with current parameters and
data. The graph stores its output as a typed reference to an input or recorded node, rather than
assuming that the final node is the result. The shape-indexed builder rejects ill-shaped
connections and ill-shaped output references.

The graph stores executable derivative rules. Selecting this mode does not prove those rules
correct. A derivative theorem additionally needs the corresponding proof-carrying nodes from the
autograd proof layer. TorchLean proves that lowering `GraphData` to a runtime tape preserves its
stored backpropagation program; that implementation theorem is separate from mathematical
derivative correctness.

Compare the initial loss between eager and typed graph execution using the same seed. It should
agree for the supported deterministic program. Then compare final parameters or predictions, not
only six-decimal loss summaries, because different execution orders can hide small discrepancies.

The current typed graph trainer is CPU-only. It does not consume an `AcceptedGraphKernelPlan`,
perform native code generation, or mean CUDA Graph capture. A CUDA plus typed graph request fails
explicitly.

# Experiment 3: Native CUDA

Build and run:

```
lake -R -K cuda=true exe torchlean quickstart_mlp \
  --scalar float32 \
  --execution eager \
  --device cuda \
  --steps 20 \
  --seed 2026 \
  --show-backend
```

The CUDA profile selects native capsules for supported operations. The report names reshape,
permutation, matrix multiplication, broadcasting, addition, ReLU, and MSE providers as they are
first used.

The same native `Float32` scalar is used at the TorchLean module boundary and in CUDA tensor
storage. Kernel capsules still record the native boundary: selecting `Float32` does not by itself
prove a CUDA kernel, compiler, driver, or device correct.

If the project is built without CUDA support, requesting CUDA fails. The stub archives permit the
repository to build on CPU-only systems; they do not pretend to execute GPU code.

# Experiment 4: Executable IEEE Binary32

```
lake exe torchlean quickstart_mlp \
  --scalar ieee32-exec \
  --execution eager \
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

The lower scalar dispatcher also recognizes executable complex binary32. The high-level trainer
does not expose it as a training mode: its objectives are real supervised losses, while genuine
complex training needs a real-valued complex loss, conjugate-aware gradients, and a result type that
preserves complex predictions. All supported selections follow the one-scalar-per-run contract from
*Tensors And Shapes*.

# The Same Choices In Lean

```
def typedGraphOptions : Runtime.Autograd.Torch.Options :=
  { execution := .typedGraph
    device := .cpu }

def eagerCpu : Trainer.RunConfig :=
  { scalar := .float32
    execution := .eager
    device := .cpu
    optimizer := optim.adam { lr := 0.03 } }

def typedGraphCpu : Trainer.RunConfig :=
  { eagerCpu with execution := .typedGraph }

def eagerCuda : Trainer.RunConfig :=
  { eagerCpu with device := .cuda }
```

`ExecutionMode` has exactly two constructors: `.eager` and `.typedGraph`. The record syntax keeps
execution and device visibly separate. Convenience methods are available for interactive code, but
they do not introduce another execution mode.

Lowering is also available directly when a program needs to keep and reuse the graph itself:

```
let graph ← nn.lowerToTypedGraph model (α := Float)
let output := graph.forward params input
let directionalDerivative := graph.jvp params parameterTangents input inputTangent
let (parameterCotangents, inputCotangent) := graph.vjpWithSeed params input outputCotangent
```

`nn.lowerToTypedGraph` returns an `nn.TypedGraphModel`. This is a transparent model-facing view of
`Runtime.Autograd.Torch.TypedGraph`, not a second graph representation. Its type separates the
parameter layout from the model input, so forward and differentiation calls do not expose a raw
list of all graph leaves.

This lowering records a typed SSA graph. It does not optimize, fuse, schedule, or generate native
code; those are compiler responsibilities rather than properties of `.typedGraph` execution.

The imperative `Session` API has a different lifetime. Its typed-graph mode records one graph per
recording phase, and `resetTape` begins a fresh graph. Use `nn.lowerToTypedGraph` or the high-level
trainer when the graph itself must survive across calls.

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

# Device And Provider Profiles

Ordinary code selects a device directly. TorchLean resolves the maintained profile for that device
and rejects unavailable combinations. Advanced code can select a complete profile with
`withBackendProfile`; calling `withDevice` later clears that override.

The next chapter, [Inside The Backend Planner](Runtime___-Autograd___-and-Interop/Inside-The-Backend-Planner/), explains provider
preference, capsule evidence, VJP ownership, and report contents. Keeping those details there lets
the discussion here can stay focused on choosing and comparing execution modes.

# Train And Evaluation Mode

Mode-sensitive layers include dropout and normalization:

```
Trainer.Manual.Runner.train runner
Trainer.Manual.Runner.eval runner
Trainer.Manual.Runner.isTraining runner
```

Training mode may sample masks or update running statistics. Evaluation mode uses the corresponding
inference behavior. The high-level trainer enters training mode for updates and evaluation mode for
summary predictions and later calls to `TrainResult.predict`.

Mode is independent of device and execution choice. A CUDA runner can switch mode without
changing model architecture or provider profile.

# A Small Dropout Thought Experiment

Suppose:

$$`y=\operatorname{Dropout}_{p}(x)`.

During training, a random mask is realized and retained for the backward rule. During evaluation,
the operation follows its deterministic inference semantics. Re-running the backward pass with a
newly sampled mask would not differentiate the forward value that was computed.

This is why RNG and mode belong to runtime state and to reproducible checkpoints.

# Dynamic Operations And Typed Graphs

A fixed typed graph needs operation structure and shapes known when recording. If a program
reads token values and changes the graph structure while constructing it, the typed graph recorder
cannot represent that program as one fixed graph.

The correct response is not to coerce the values into a graph and hope. Keep genuinely dynamic
control flow in eager mode, represent the choice as a supported tensor operation, or record
separate static branches and choose between them explicitly at runtime.

Unsupported typed graph operations are rejected.

# Unsupported Means Failure

Try:

```
lake exe torchlean quickstart_mlp \
  --device metal --steps 1
```

on the current checkout. The target name is parsed, but profile selection rejects it. This confirms
that a future platform vocabulary is not reported as working implementation.

The same rule covers other unsupported combinations. CUDA in a CPU-only build, typed graph execution with a
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
  * native `Float32`
  * eager
  * CPU
*
  * replay a supported fixed graph
  * native `Float32`
  * typed graph
  * CPU
*
  * run native GPU training
  * native `Float32`
  * eager
  * CUDA
*
  * inspect binary32 reference behavior
  * `IEEE32Exec`
  * eager
  * CPU
*
  * use external attention forward
  * native `Float32`
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
eager or typed graph execution
device and provider capsules
train/eval mode
checkpoint and code revision
```

Without this information, two loss curves may be incomparable even when both are labeled
“TorchLean float32.”

Sources:

- [NN/API/Module/Execution.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Module/Execution.lean);
- [Core/Types.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Torch/Core/Types.lean);
- [Core/Trainer.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Torch/Core/Trainer.lean);
- [Trainer/Parameters.lean](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Torch/Core/Trainer/Parameters.lean).
