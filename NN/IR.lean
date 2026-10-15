/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Binary
public import NN.IR.BinaryProofs
public import NN.IR.Check
public import NN.IR.Graph
public import NN.IR.Infer
public import NN.IR.Operator
public import NN.IR.OpContracts
public import NN.IR.Pretty
public import NN.IR.Properties
public import NN.IR.Reference
public import NN.IR.Semantics
public import NN.IR.ShapeSoundness

/-!
# Intermediate Representation

Curated umbrella import for TorchLean's op-tagged intermediate representation.

Use this for graph syntax, shape and element-type checks, homogeneous and heterogeneous semantics,
FloatLib binary conversions and their correctness theorem, compositional proof rules,
shape-soundness results, and pretty-printers. Internal code can import the individual `NN.IR.*`
modules when it needs a smaller dependency footprint.
-/

@[expose] public section
