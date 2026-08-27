/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Attention
public import NN.Spec.Module.Core

/-!
# Attention module wrappers

This file wraps a few attention blocks as `Spec.Module`s so we can:

- compose them with `Spec.Module.Chain` (shape-safe pipelines), and
- attach simple export/pretty-print metadata for examples.

The wrapper below builds a self-attention context with `Q=K=V=x` and no mask, which matches the
common "encoder block" usage. More specialized variants (cross-attention, causal masks, etc.) are
defined at the layer-spec level in `NN/Spec/Layers/Attention.lean`.

In PyTorch terms, the core computation is scaled dot-product self-attention:
`softmax(QK^T / sqrt(d)) V`, and newer PyTorch exposes it as
`torch.nn.functional.scaled_dot_product_attention`.

This wrapper stays focused: it is self-attention only (`Q=K=V=x`) with no causal mask.
-/

@[expose] public section


namespace Spec.Module

open Tensor
open Shape

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-- Self-attention block (`Q=K=V=x`, no mask) as an `Spec.Module`. -/
def scaledDotProductSelfAttention
  (n dModel : Nat) (h1 : n ≠ 0) :
  Spec.Module α ([n, dModel]) ([n, dModel]) :=
{ forward := fun x =>
    let ctx : AttentionContext α n n dModel h1 h1 :=
      { Q := x
        K := x
        V := x
        mask := none }
    scaledDotProductAttention (α := α) ctx,
  kind := "ScaledDotProductSelfAttention",
  pythonExpr := s!"ScaledDotProductSelfAttention(d_model={dModel})" }

end Spec.Module
