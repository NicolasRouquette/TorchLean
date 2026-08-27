/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor.Constructors
public import NN.Tensor.Operations
public import NN.Tensor.Printing
public import NN.Tensor.Syntax

/-!
# Tensors

Canonical public import for the `TorchLean.Tensor` surface. Declarations are grouped by concern in
`NN.Tensor.Constructors`, `NN.Tensor.Operations`, `NN.Tensor.Printing`, and `NN.Tensor.Syntax`.

Runtime tapes that need heterogeneous tensor packs import `NN.Tensor.ShapeErasure` explicitly.
That boundary is intentionally absent from this application-facing umbrella.
-/
