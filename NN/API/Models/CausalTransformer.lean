/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Models.CausalTransformer.Runtime

/-!
# Causal Transformer Models

Canonical public import for TorchLean's GPT-style causal Transformer support.

`NN.API.Models.CausalTransformer.Architecture` owns configuration, shape families, and graph
construction. `NN.API.Models.CausalTransformer.Runtime` owns indexed-token and tied-weight
execution programs and causal-language-model objectives.

Import this module for the complete API, including one-hot and bounded-token models with either an
independent vocabulary head or a projection tied to the token-embedding table. Tokenization and
checkpoint formats remain in their respective API modules.
-/
