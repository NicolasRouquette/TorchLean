/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Normalization
public import NN.Spec.Module.Core

/-!
# Normalization module wrappers

This file wraps selected normalization specs as `Spec.Module`s for composition/export.

For `Spec.Module.layerNorm`, we require (defaulted) proofs that dimensions are positive. This matches
the spec-level intent: normalization divides by the number of features and uses variance/standard
 deviation, so degenerate "zero-width" cases are excluded when we want clean theorems.

PyTorch mental picture: `nn.LayerNorm(embedDim)` applied at each timestep, with `weight=gamma` and
`bias=beta`.
-/

@[expose] public section


namespace Spec.Module

open Tensor

variable {α : Type} [Context α]

/-- LayerNorm over the last dimension, wrapped as an `Spec.Module`. -/
def layerNorm (seqLen embedDim : Nat)
  (gamma : Tensor α (.dim embedDim .scalar))
  (beta : Tensor α (.dim embedDim .scalar))
  (hSeqPos : seqLen > 0)
  (hEmbedPos : embedDim > 0) :
  Spec.Module α (.dim seqLen (.dim embedDim .scalar)) (.dim seqLen (.dim embedDim .scalar)) :=
  { forward := fun x =>
      Spec.layerNorm (α := α) (seqLen := seqLen) (embedDim := embedDim)
        x gamma beta hSeqPos hEmbedPos
    kind := "LayerNorm"
    pythonExpr := s!"nn.LayerNorm({embedDim})" }

end Spec.Module
