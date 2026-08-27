/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Neural.Leading

/-!
# Attention

Multi-head self-attention configuration and layer construction.
-/

@[expose] public section

namespace TorchLean
namespace nn
namespace Internal

/--
Multi-head self-attention configuration.

PyTorch analogue: `torch.nn.MultiheadAttention` (conceptually).
See `https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html`.
-/
structure MultiHeadAttention where
  /-- Number of attention heads. -/
  numHeads : Nat
  /-- Per-head embedding dimension. -/
  headDim : Nat
  /-- Base seed for deterministic parameter initialization. -/
  seedW : Nat := 0
  /-- Projection-weight initialization. `none` retains Xavier-uniform initialization. -/
  weightInit? : Option _root_.Runtime.Autograd.Torch.Init.Scheme := none
  /--
  Optional initializer for the output projection.

  This is separate because deep residual stacks commonly scale the projection that writes back to
  the residual stream. When omitted, `weightInit?` is used.
  -/
  outputWeightInit? : Option _root_.Runtime.Autograd.Torch.Init.Scheme := none
  /-- Add a trainable bias after the output projection. -/
  outputBias : Bool := false

/--
Multi-head self-attention using `NeZero` to hide the nonzero sequence length proof.

If `mask` is provided, it is a boolean attention mask of shape `(n × n)` (e.g. causal masking).
-/
def multiHeadAttention (leading : List Nat := []) {n dModel : Nat}
    (cfg : MultiHeadAttention) [NeZero n]
    (mask : Option (Tensor Bool [n, n]) := none) :
    Sequential (leading ++ [n, dModel]) (leading ++ [n, dModel]) := by
  simpa only [Spec.Shape.ofList_append, Spec.Shape.concat_appendDim,
    Spec.Shape.appendDim] using
    (if cfg.outputBias then
      of <| adaptLeadingShape (Spec.Shape.ofList leading) <|
        _root_.Runtime.Autograd.TorchLean.NN.multiHeadAttentionOutputBias
        (Spec.Shape.size (Spec.Shape.ofList leading)) n dModel cfg.numHeads cfg.headDim
        (h1 := NeZero.ne (n := n)) (seedW := cfg.seedW) (weightInit? := cfg.weightInit?)
        (outputWeightInit? := cfg.outputWeightInit?) (mask := mask)
    else
      of <| adaptLeadingShape (Spec.Shape.ofList leading) <|
        _root_.Runtime.Autograd.TorchLean.NN.multiHeadAttention
        (Spec.Shape.size (Spec.Shape.ofList leading)) n dModel cfg.numHeads cfg.headDim
        (h1 := NeZero.ne (n := n)) (seedW := cfg.seedW) (weightInit? := cfg.weightInit?)
        (outputWeightInit? := cfg.outputWeightInit?) (mask := mask))
