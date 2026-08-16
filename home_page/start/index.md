---
title: Getting Started
layout: default
---

# Getting Started

Start with a model small enough that every part can be seen. The first run should feel like ordinary
ML: build the project, train a tiny model, and run a compact verifier. The classifier is
deliberately small, so the structure is visible: the same project contains the executable model, the
graph representation, and the checker that reasons about the graph.

```bash
lake build
lake exe torchlean quickstart_mlp --device cpu --steps 10 --scalar ieee32-exec --execution eager
lake exe verify -- torchlean-ibp
```

The first command checks the Lean project. The second command runs a compact supervised model through
the trainer: it initializes parameters, executes a Float32 forward pass, computes a loss, runs
reverse mode, and applies optimizer updates. The third command lowers a small TorchLean model into
the graph used by the verifier and runs interval bound propagation over an input box.

To see the command-line entry points:

```bash
lake exe torchlean --help
lake exe verify --help
```

The first lists runnable examples: quickstarts, supervised models, text models, diffusion, FNO
Burgers, reinforcement learning, data loaders, PyTorch interop, graph examples, and floating-point
checks. The second lists checker entry points: TorchLean IBP/CROWN paths, LiRPA-style fixtures,
PINN certificates, ODE enclosures, VNN-COMP-style MNIST queries, 3D projection certificates, spline
certificates, and two-stage Lyapunov experiments.

## Where To Go Next

1. [Installation]({{ '/installation/' | relative_url }}) covers Linux, macOS, Windows/WSL, CUDA,
   optional LibTorch integration, and backend capsules.
2. [Building Models]({{ '/blueprint/Building-Models/' | relative_url }}) introduces typed tensors,
   layers, parameter packs, datasets, losses, optimizers, and the trainer.
3. [Runtime and Interop]({{ '/blueprint/Runtime___-Autograd___-and-Interop/' | relative_url }})
   explains eager and typed graph execution, autograd, runtime artifacts, PyTorch interop boundaries,
   data streams, and backend selection.
4. [Semantics and Graphs]({{ '/blueprint/Semantics-and-Graphs/' | relative_url }}) explains the
   graph IR, graph denotation, shape discipline, named operations, and why verifiers reuse the same
   graph rather than inventing a second model language.
5. [Floating Point and Native Boundaries]({{ '/blueprint/Floating-Point-and-Native-Boundaries/' | relative_url }})
   separates real-valued specifications, executable Float32 models, CUDA/native execution, and
   external producer assumptions.
6. [Verification and Certificates]({{ '/blueprint/Verification-and-Certificates/' | relative_url }})
   covers IBP/CROWN bounds, imported artifacts, optimizer laws, autograd proof APIs, scientific
   ML certificates, and trust boundaries.
7. [Examples]({{ '/examples/' | relative_url }}) collects runnable model, scientific ML,
   verification, text, diffusion, geometry, and Bug Zoo workflows.

## Common Next Steps

- Train a model: `lake exe torchlean mlp --device cpu --steps 100 --scalar ieee32-exec`.
- Inspect a scientific ML run: [Scientific ML]({{ '/examples/scientific-ml/' | relative_url }}).
- Check a certificate or bound pass: [Verification Bounds]({{ '/examples/verification/' | relative_url }}).
- Start from the main import: `import NN; open TorchLean`.
- Follow declaration and proof dependencies: [Formalization graph]({{ '/blueprint/Dependency-Graph/' | relative_url }}).
- Explore module dependencies: [Import graphs]({{ '/graphs/' | relative_url }}).
- Understand CUDA assumptions: [GPU and CUDA Boundaries]({{ '/blueprint/Floating-Point-and-Native-Boundaries/From-A-Tensor-Operation-To-A-GPU-Kernel/' | relative_url }}).

When changing code, keep the edit loop small at first: run one example, inspect the file it writes,
then run the matching checker when the example has one.
