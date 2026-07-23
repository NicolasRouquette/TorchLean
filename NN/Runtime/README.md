# NN/Runtime

`NN/Runtime` is the part of TorchLean that actually runs a model.

The files here turn typed tensors and parameter lists into executable training loops, prediction
calls, autograd tapes, compiled graph executions, CUDA launches, PyTorch round trips, and
reinforcement-learning rollouts. The runtime is deliberately tied back to the spec and proof layers:
the same model should be runnable as ordinary Lean code, lowerable into the graph IR, and usable as
the object of a later verification statement.

Most downstream code should not import modules from this directory directly. Prefer:

* `import NN` for ordinary model and training code,
* `import NN.API.Runtime` when you are extending the runtime subsystem itself, or
* `import NN.Runtime` when you need the broad executable umbrella.

There is no top level `NN.Runtime` Lean file. User-facing code goes through the `TorchLean` API;
subsystem code uses focused `NN.API.*` or direct subsystem imports so implementation files do not
become a second public API.

## How A Run Moves

An ordinary training command follows this shape.

1. A model is built from typed tensors, layers, parameters, and a loss.
2. The public trainer chooses a scalar mode and backend, such as eager Float32, compiled Float32, or
   a CUDA-backed run.
3. The autograd engine records the operations that need gradients and stores enough local data for
   the backward pass.
4. The optimizer updates the parameter list using the same equations that appear in the optimizer
   theory files.
5. Optional exporters write logs, graphs, weights, predictions, or certificate inputs that other
   TorchLean modules can inspect.

That is why TorchLean has runtime tests and proof modules. The tests check that the executable path
still runs on real data and real devices. The proof modules state and prove mathematical facts about
the specifications, graph translations, interval bounds, floating-point envelopes, optimizer
updates, and verification checkers.

## Execution Paths

| Area | Role |
| --- | --- |
| `Autograd/Engine` | The small eager reverse-mode tape, closest to the local backward rules for primitive tensor operations. |
| `Autograd/Compiled` | Graph/IR execution that runs through the same runtime values instead of becoming a detached interpreter. |
| `Autograd/TorchLean` | The TorchLean-native runtime used by the trainer, layer functions, tensor packs, backend options, and scalar modes. |
| `Autograd/Torch` | Lower-level imperative sessions and linked-session machinery used for PyTorch interop and compiled sessions. |
| `Autograd/Train` | Deterministic datasets, step streams, loaders, losses, training loops, evaluation helpers, and optimizer integration. |
| `Optim` | Executable optimizer equations and scheduler utilities. Public optimizer names are re-exported through `TorchLean.optim`. |
| `PyTorch` | State-dict, Torch export, ONNX, and IR-to-PyTorch bridges used for round-trip checks and external model exchange. |
| `RL` | Gymnasium sessions, typed environments, PPO/DQN helpers, rollouts, and boundary checks for reinforcement learning examples. |
| `External` | Small process helpers for executable integrations that deliberately leave Lean's kernel. |
| `Training` | Training-log records shared by examples, plots, widgets, and command-line runs. |

## Backend Choices

The public API should read like one model with different execution choices, not like several
competing APIs. In ordinary code the user builds a trainer once and then selects the backend:

```lean
let trainer :=
  Trainer.new model
    { task := .regression
      backend := .compiled
      dtype := .float32
      optimizer := optim.adam { lr := 0.001 } }
let trained ← trainer.train data { steps := 200 }
let prediction ← trained.predict input
```

The names separate intent from implementation.

| Choice | Meaning |
| --- | --- |
| Eager TorchLean | Execute operations directly through the TorchLean-native tape; this is the most direct path to inspect. |
| Compiled TorchLean | Lower the model into the graph/IR path and execute that representation while keeping TorchLean as the owner of the model and graph. |
| CUDA TorchLean | Use native CUDA kernels for supported operations while keeping TorchLean's runtime and tape contract at the boundary. |
| ATen/libtorch provider | Use selected PyTorch/ATen kernels as fast numeric providers only when TorchLean still records the corresponding graph/tape node and owns the backward rule. |
| PyTorch/Julia/Gymnasium bridges | Exchange data with external systems when the example needs a comparison target, imported weights, a simulator, or a runtime environment. |

The compiled path runs the same TorchLean model through its graph/IR representation. Compiled/eager
equivalence checks and graph-correctness modules connect that execution path to the original object.

The ATen/libtorch direction follows the same rule. For no-grad inference, an external provider can
return a value under an explicit agreement assumption. For training, TorchLean cannot hand the whole
autograd story to libtorch without changing the proof boundary. A supported ATen-backed operation
should compute the forward value, attach that value to the normal TorchLean node/cache, and use the
normal TorchLean backward rule. If an operation cannot preserve that relation yet, the training path
should fall back to the TorchLean-native forward for that op.

## What Counts As Evidence

Runtime evidence and proof evidence are different, and both have a role.

| Evidence | Where it lives | What it says |
| --- | --- | --- |
| Executable examples | `NN/Examples`, `lake exe torchlean ...` | The command runs, uses the intended backend, and produces the expected artifact shape. |
| Runtime tests | `NN/Tests/Runtime` | The implementation agrees with closed forms, cross-backend checks, saved fixtures, or regression expectations. |
| Formal proofs | `NN/Proofs`, `NN/MLTheory`, `NN/Verification/TorchLean/Proved` | A Lean theorem establishes a mathematical property of the specification, translation, bound, or checker. |
| Certificate checks | `NN/Verification` | An external or generated artifact is parsed and checked against a Lean side condition. The checker can be proved sound even when the artifact producer is not trusted. |

## Trust Boundaries

The Lean runtime specifies and executes TorchLean's own tensor and tape semantics. Native CUDA,
ATen/libtorch, PyTorch exporters, Julia, and Gymnasium integrations are external trust boundaries:
Lean side modules record the contracts, shape checks, fixtures, regression tests, and theorem
interfaces for the objects those systems produce. Claims that depend on foreign C, CUDA, C++,
Python, or Julia code should cite the corresponding producer boundary or a separate checker/theorem
that discharges it.

For the proof story, look at `NN.Proofs.*`, `NN.MLTheory.*`, and `NN.Verification.*`. This directory
supplies the executable objects those theorems and checkers connect to.
