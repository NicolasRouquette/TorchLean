/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Chain.ToDAG.Model

/-!
# Conversion of sequential GraphSpec chains to DAGs

This is the canonical import for structural chain-to-DAG conversion and the associated DAG model
constructors. Term construction lives in `Chain.ToDAG.Core`; parameter initialization and model
packaging live in `Chain.ToDAG.Model`.
-/
