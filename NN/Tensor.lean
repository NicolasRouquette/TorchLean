/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor.API

/-!
# Tensors

Curated umbrella import for TorchLean's core tensor/shape API.

Use this import for tensor literals, shape syntax, dynamic tensors, and small executable tensor
helpers. `NN.Tensor.API` remains the implementation leaf; downstream users should use either `NN`
for the complete library, `NN.API` for application code, or `NN.Tensor` for this layer alone.
-/

@[expose] public section
