/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.DAG.Model

/-!
# Canonical GraphSpec DAG

This is the canonical import for GraphSpec's typed SSA/DAG representation. It exposes:

- shape-indexed DAG syntax, substitutions, and shared multi-result blocks;
- pure tensor semantics and preservation theorems;
- execution-polymorphic TorchLean lowering;
- single- and multi-output model wrappers.

The implementation is organized by concept in `DAG.Syntax`, `DAG.Semantics`, `DAG.Lowering`, and
`DAG.Model`. Most users should continue to import `NN.GraphSpec.DAG.Core`.
-/
