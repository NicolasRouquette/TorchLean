/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# `NN.Runtime.Autograd`: Runtime Autograd Overview

This directory is the runtime execution layer for TorchLean's automatic differentiation.
It contains two execution representations and an imperative API over them:

1. **Eager tape (dynamic DAG)**: record a runtime tape during the forward pass, then run a
   reverse-mode loop over that tape to accumulate gradients.
2. **Typed graph execution (typed SSA/DAG)**: record a well-typed graph (`GraphData`) carrying
   forward, JVP, and VJP rules, then lower it with concrete inputs to the runtime tape. Fixed
   programs can package this graph as a reusable artifact.
3. **Imperative sessions**: wrap either representation behind an API that feels closer to PyTorch
   (`TensorRef` objects, `backward`, `step`, etc.).

The key architectural idea is that we keep a small, explicit, pure core (the tape and its
reverse-mode loop), then build convenience layers on top (imperative sessions, training helpers,
optimizers). This makes it easier to:

- state precisely whether a result concerns executable `GraphData` or proof-carrying `Graph`,
- prove that lowering preserves the stored graph program,
- still offer familiar session-style ergonomics for executable training code.

## Where to look

- Eager tape engine (pure core):
  - `NN/Runtime/Autograd/Engine/Core.lean`
  - `NN/Runtime/Autograd/Engine/TapeM.lean` (StateT tape layer)
  - `NN/Runtime/Autograd/Engine/FastKernels.lean` (optional runtime-only speedups)
- Typed graph execution path:
  - `NN/Runtime/Autograd/TypedGraph.lean` (runtime umbrella)
  - `NN/Runtime/Autograd/TypedGraph/Core.lean`
  - `NN/Runtime/Autograd/TypedGraph/GraphM.lean` (authoring DSL for `GraphData`)
  - `NN/Runtime/Autograd/IRExec.lean` (shared `NN.IR.Graph` forward execution bridge)
- PyTorch-style imperative front-end:
  - `NN/Runtime/Autograd/Torch/Core.lean`
  - `NN/Runtime/Autograd/Torch/Utils.lean`
  - `NN/Runtime/Autograd/Torch/TypedGraphSession.lean` (records a typed graph, runs the lowered tape)
- Unified imperative runtime:
  - `NN/Runtime/Autograd/TorchLean/Program.lean` (execution-polymorphic tensor programs)
  - `NN/Runtime/Autograd/TorchLean.lean` (umbrella import plus re-exports)
  - `NN/Runtime/Autograd/TorchLean/Session.lean` (one API, eager or typed graph execution)
- Training helpers (datasets, logging, optimizers, trainer):
  - `NN/Runtime/Autograd/Train/*`
- Runtime utilities:
  - `NN/Runtime/Autograd/Utils.lean` (small umbrella for executable training scripts and tests)

## Connection to Proofs

The runtime files define executable behavior. Proof modules under `NN.Proofs.Autograd.*` state and
prove facts about the same tape/graph vocabulary. In particular, tape lowering preserves
`GraphData` backpropagation, while derivative soundness uses the stronger `Node` and `Graph` types
that carry local adjointness laws. CUDA and foreign-process bridges are checked by contracts and
tests at this layer, while their external implementations remain outside Lean's trusted kernel.

## References / citations

- PyTorch `torch.autograd` docs:
  https://pytorch.org/docs/stable/autograd.html
- PyTorch "Autograd mechanics" note on dynamic graph construction:
  https://pytorch.org/docs/stable/notes/autograd.html
- `torch.nn.functional.mse_loss` (mean reduction semantics):
  https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html
- `micrograd` (small reverse-mode AD engine, useful for intuition):
  https://github.com/karpathy/micrograd
-/

@[expose] public section
