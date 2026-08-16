---
title: Guide
permalink: /manual/
layout: default
---

# Guide

TorchLean is easiest to learn by following one model carefully. At first it is just a small neural
network with tensors, parameters, and a loss. Then it is trained. Then it is lowered to a graph.
Later it may be run through a CUDA backend, exported across the PyTorch boundary, checked against a
certificate, or mentioned in a theorem.

The guide keeps that model recognizable while it changes form. A runtime result, a
graph denotation, a verifier certificate, and a Float32 approximation are different objects. They
can support different statements. TorchLean is built so those differences are written down instead
of left as background knowledge in a script.

The guide begins with executable code because TorchLean is an ML library with proof objects close
by. From there it adds the structure needed for serious claims: typed tensors, datasets,
optimizers, autograd, graph lowering, floating-point models, native runtime boundaries,
scientific ML artifacts, certificate formats, and Lean proof statements.

## Reading Route

1. [Introduction]({{ '/blueprint/Introduction/' | relative_url }}) explains why TorchLean keeps the
   runnable program and the mathematical object close together.
2. [Building Models]({{ '/blueprint/Building-Models/' | relative_url }}) starts from familiar ML:
   tensors, layers, datasets, losses, optimizers, and short training runs.
3. [Runtime and Interop]({{ '/blueprint/Runtime___-Autograd___-and-Interop/' | relative_url }})
   follows execution: eager runs, typed graph runs, autograd, backend selection, checkpoints, and
   PyTorch interop boundaries.
4. [Semantics and Graphs]({{ '/blueprint/Semantics-and-Graphs/' | relative_url }}) introduces the
   graph objects that runtime tools, exporters, widgets, and verification passes share.
5. [Verification and Certificates]({{ '/blueprint/Verification-and-Certificates/' | relative_url }})
   shows how bounds, imported artifacts, trusted producers, numerical bridges, and Lean theorems fit
   together.

[Read the guide]({{ '/blueprint/' | relative_url }})
