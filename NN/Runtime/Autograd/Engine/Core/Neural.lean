/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.Runtime.Autograd.Engine.Core.ConvPool

/-!
Neural-network operations for the eager engine.

This file implements runtime nodes such as dropout, normalization, attention, and recurrent/sequence
building blocks on top of the core tensor operation layer.
-/

@[expose] public section

namespace Runtime
namespace Autograd

open Spec
open Tensor

namespace Tape

/--
Layer normalization for `(seqLen, embedDim)` tensors.

This records a single node whose backward returns gradients for `x`, `gamma`, and `beta`.
PyTorch comparison: `torch.nn.LayerNorm(embedDim)` (applied per token) / `functional.layer_norm`.
-/
def layerNorm {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (t : Tape α) (xId gammaId betaId : Nat) : Result (Tape α × Nat) := do
  let x ← requireValue (α:=α) (t:=t) (s:=.dim seqLen (.dim embedDim .scalar)) xId
  let gamma ← requireValue (α:=α) (t:=t) (s:=.dim embedDim .scalar) gammaId
  let beta ← requireValue (α:=α) (t:=t) (s:=.dim embedDim .scalar) betaId
  let y := Spec.layerNorm (x := x) (gamma := gamma) (beta := beta) h_seq_pos h_embed_pos
  let node : Node α :=
    { name := some "layer_norm"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad := true
      parents := #[xId, gammaId, betaId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := .dim seqLen (.dim embedDim .scalar)) dLdyAny
        let (dx, dgamma, dbeta) :=
          Spec.layerNormBackward (h_seq_pos := h_seq_pos) (h_embed_pos := h_embed_pos)
            (x := x) (gamma := gamma) (_beta := beta) (grad_output := dLdy)
        pure #[
          (xId, Spec.SomeTensor.ofTensor dx),
          (gammaId, Spec.SomeTensor.ofTensor dgamma),
          (betaId, Spec.SomeTensor.ofTensor dbeta)
        ]
    }
  pure (t.addNode node)

/-- Batch normalization over every spatial axis of a channel-first tensor. -/
def batchNorm {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape] {channels : Nat} {sSpatial : Shape}
  (hWellFormed : (Shape.dim channels sSpatial).wellFormed)
  (t : Tape α) (xId gammaId betaId : Nat) : Result (Tape α × Nat) := do
  let _ : Shape.WellFormed (.dim channels sSpatial) := ⟨hWellFormed⟩
  let x ← requireValue (α:=α) (t:=t) (s:=.dim channels sSpatial) xId
  let gamma ← requireValue (α:=α) (t:=t) (s:=.dim channels .scalar) gammaId
  let beta ← requireValue (α:=α) (t:=t) (s:=.dim channels .scalar) betaId
  let y := Spec.batchNorm (x := x) (gamma := gamma) (beta := beta)
  let node : Node α :=
    { name := some "batch_norm"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad := true
      parents := #[xId, gammaId, betaId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := .dim channels sSpatial) dLdyAny
        let (dx, dgamma, dbeta) :=
          Spec.batchNormBackward (x := x) (gamma := gamma) (gradOutput := dLdy)
        pure #[
          (xId, Spec.SomeTensor.ofTensor dx),
          (gammaId, Spec.SomeTensor.ofTensor dgamma),
          (betaId, Spec.SomeTensor.ofTensor dbeta)
        ]
    }
  pure (t.addNode node)

/--
Multi-head self-attention.

This is a shape-specialized attention primitive used by transformer-style models. It depends on an
optional boolean `(n,n)` mask and returns the attended output of shape `(n,dModel)`.

PyTorch comparison: similar to `torch.nn.MultiheadAttention` / scaled dot-product attention.
-/
def multiHeadAttention {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq
  Shape]
  {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (t : Tape α) (wqId wkId wvId woId xId : Nat)
  (mask : Option (Tensor Bool [n, n]) := none) :
  Result (Tape α × Nat) := do
  let wq ← requireValue (α:=α) (t:=t)
    (s:=.dim dModel (.dim (numHeads * headDim) .scalar)) wqId
  let wk ← requireValue (α:=α) (t:=t)
    (s:=.dim dModel (.dim (numHeads * headDim) .scalar)) wkId
  let wv ← requireValue (α:=α) (t:=t)
    (s:=.dim dModel (.dim (numHeads * headDim) .scalar)) wvId
  let wo ← requireValue (α:=α) (t:=t)
    (s:=.dim (numHeads * headDim) (.dim dModel .scalar)) woId
  let x ← requireValue (α:=α) (t:=t) (s:=.dim n (.dim dModel .scalar)) xId
  let mha : Spec.MultiHeadAttention α numHeads dModel headDim :=
    { queryWeight := wq, keyWeight := wk, valueWeight := wv, outputWeight := wo }
  let y := Spec.MultiHeadAttention.forward (n := n) (h1 := h1) (mha := mha) (x := x) (mask := mask)
  let node : Node α :=
    { name := some "multi_head_attention"
      value := Spec.SomeTensor.ofTensor y
      requiresGrad := true
      parents := #[wqId, wkId, wvId, woId, xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := .dim n (.dim dModel .scalar)) dLdyAny
        let (dx, dWq, dWk, dWv, dWo) :=
          Spec.multiHeadAttentionBackward (h1 := h1) (mha := mha) (x := x) (mask := mask)
            (grad_output := dLdy)
        pure #[
          (xId, Spec.SomeTensor.ofTensor dx),
          (wqId, Spec.SomeTensor.ofTensor dWq),
          (wkId, Spec.SomeTensor.ofTensor dWk),
          (wvId, Spec.SomeTensor.ofTensor dWv),
          (woId, Spec.SomeTensor.ofTensor dWo)
        ]
    }
  pure (t.addNode node)
