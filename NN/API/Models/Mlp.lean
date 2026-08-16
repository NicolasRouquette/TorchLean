/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded

/-!
# Multilayer Perceptrons

Compact, reusable MLP-shaped model constructors for TorchLean examples.

These live in the API layer so runnable examples can stay focused on:
data loading, training loops, and CLI flags, rather than repeating the same
`linear → activation → linear` boilerplate.

Scope note: these are building blocks, not pretrained checkpoints.
-/

@[expose] public section

namespace TorchLean


open Spec Tensor

namespace nn
namespace models

/-- Configuration for a single-hidden-layer MLP over batched row vectors. -/
structure MLPConfig where
  batch : Nat
  inDim : Nat
  hidDim : Nat
  outDim : Nat
deriving Repr

/-- Input shape `(batch × inDim)` for an `MLPConfig`. -/
abbrev mlpInShape (cfg : MLPConfig) : Spec.Shape :=
  .dim cfg.batch (.dim cfg.inDim .scalar)

/-- Output shape `(batch × outDim)` for an `MLPConfig`. -/
abbrev mlpOutShape (cfg : MLPConfig) : Spec.Shape :=
  .dim cfg.batch (.dim cfg.outDim .scalar)

/--
Build a single-hidden-layer MLP with relu activation:

`linear(inDim → hidDim) → relu → linear(hidDim → outDim)`.
-/
def mlpRelu (cfg : MLPConfig) :
    nn.Builder (nn.Sequential (mlpInShape cfg) (mlpOutShape cfg)) :=
  nn.Sequential![
    linear cfg.inDim cfg.hidDim (pfx := .dim cfg.batch .scalar),
    relu,
    linear cfg.hidDim cfg.outDim (pfx := .dim cfg.batch .scalar)
  ]

end models
end nn

end TorchLean
