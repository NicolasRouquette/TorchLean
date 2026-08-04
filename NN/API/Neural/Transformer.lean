/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.API.Neural.Blocks

/-!
# Transformer Blocks

This module exposes attention, feed-forward, and Transformer-stack constructors used by sequence
models and higher-level examples.
-/

@[expose] public section

namespace TorchLean
namespace nn
namespace Internal
namespace blocks

/--
Config record for `transformerEncoderBlock`.

Separating the config as a structure makes it easier to write readable examples and keep seed
management deterministic.
-/
structure TransformerEncoderBlock where
  /-- Number of attention heads. -/
  numHeads : Nat
  /-- Per-head embedding dimension. -/
  headDim : Nat
  /-- Hidden dimension of the feed-forward network. -/
  ffnHidden : Nat
  /-- Activation used in the feed-forward network. -/
  activation : Activation := .gelu
  /-- Optional dropout probability for examples; `none` means no dropout. -/
  dropout? : Option Float := none
  /-- Normalize before attention and feed-forward sublayers instead of after each residual. -/
  normFirst : Bool := false
  /-- Add a trainable bias after the attention output projection. -/
  attentionOutputBias : Bool := false
  /-- Attention and feed-forward weight initialization. `none` keeps each layer's default. -/
  weightInit? : Option _root_.Runtime.Autograd.Torch.Init.Scheme := none
  /--
  Initializer for the attention and feed-forward projections that write to residual streams.

  When omitted, `weightInit?` is used. The separate field supports depth-scaled residual
  initialization without imposing that convention on every Transformer.
  -/
  residualOutputInit? : Option _root_.Runtime.Autograd.Torch.Init.Scheme := none
  /-- Base seed used to derive deterministic per-layer seeds inside the block. -/
  seedBase : Nat := 0

/--
Transformer encoder block configuration.

This follows the familiar pattern:
`(residual MHA) -> LayerNorm -> (residual FFN) -> LayerNorm`.

PyTorch analogue:
- `torch.nn.TransformerEncoderLayer`
  (`https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html`)
-/
def transformerEncoderBlockWithMask {batch n dModel : Nat} [NeZero n] [NeZero dModel]
    (cfg : TransformerEncoderBlock)
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    Sequential (.dim batch (.dim n (.dim dModel .scalar)))
      (.dim batch (.dim n (.dim dModel .scalar))) :=
  let seedAttn := cfg.seedBase
  let seedNorm1Gamma := cfg.seedBase + 1
  let seedNorm1Beta := cfg.seedBase + 2
  let seedFfnW1 := cfg.seedBase + 3
  let seedFfnB1 := cfg.seedBase + 4
  let seedFfnW2 := cfg.seedBase + 5
  let seedFfnB2 := cfg.seedBase + 6
  let seedNorm2Gamma := cfg.seedBase + 7
  let seedNorm2Beta := cfg.seedBase + 8
  let seedDrop1 := cfg.seedBase + 9
  let seedDrop2 := cfg.seedBase + 10

  let attn : Sequential (.dim batch (.dim n (.dim dModel .scalar)))
      (.dim batch (.dim n (.dim dModel .scalar))) :=
    multiheadAttentionWith (batch := batch) (n := n) (dModel := dModel)
      { numHeads := cfg.numHeads, headDim := cfg.headDim, seedW := seedAttn,
        outputBias := cfg.attentionOutputBias, weightInit? := cfg.weightInit?,
        outputWeightInit? := cfg.residualOutputInit? }
      (hN := NeZero.ne (n := n))
      (mask := mask)
  let attnInner :=
    match cfg.dropout? with
    | none => attn
    | some p =>
        seq! attn, dropout (s := .dim batch (.dim n (.dim dModel .scalar))) p (seed := seedDrop1)
  let norm1 : Sequential (.dim batch (.dim n (.dim dModel .scalar)))
      (.dim batch (.dim n (.dim dModel .scalar))) :=
    layerNorm (batch := batch) (seqLen := n) (embedDim := dModel)
      { seedGamma := seedNorm1Gamma, seedBeta := seedNorm1Beta }

  let ffn : Sequential (.dim batch (.dim n (.dim dModel .scalar)))
      (.dim batch (.dim n (.dim dModel .scalar))) :=
    seq!
      linearWith dModel cfg.ffnHidden { weightInit? := cfg.weightInit? }
        seedFfnW1 seedFfnB1 (pfx := .dim batch (.dim n .scalar)),
      activation (s := .dim batch (.dim n (.dim cfg.ffnHidden .scalar))) cfg.activation,
      linearWith cfg.ffnHidden dModel
        { weightInit? := cfg.residualOutputInit?.orElse (fun _ => cfg.weightInit?) }
        seedFfnW2 seedFfnB2 (pfx := .dim batch (.dim n .scalar))
  let ffnInner :=
    match cfg.dropout? with
    | none => ffn
    | some p =>
        seq! ffn, dropout (s := .dim batch (.dim n (.dim dModel .scalar))) p (seed := seedDrop2)
  let norm2 : Sequential (.dim batch (.dim n (.dim dModel .scalar)))
      (.dim batch (.dim n (.dim dModel .scalar))) :=
    layerNorm (batch := batch) (seqLen := n) (embedDim := dModel)
      { seedGamma := seedNorm2Gamma, seedBeta := seedNorm2Beta }

  if cfg.normFirst then
    seq!
      residual (seq! norm1, attnInner),
      residual (seq! norm2, ffnInner)
  else
    seq!
      residual attnInner,
      norm1,
      residual ffnInner,
      norm2

/--
Transformer encoder block.

This is `transformerEncoderBlockWithMask`; pass `mask := ...` to enable causal masking (or other
attention masks).
-/
def transformerEncoderBlock {batch n dModel : Nat} [NeZero n] [NeZero dModel]
    (cfg : TransformerEncoderBlock)
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    Sequential (.dim batch (.dim n (.dim dModel .scalar)))
      (.dim batch (.dim n (.dim dModel .scalar))) :=
  transformerEncoderBlockWithMask (batch := batch) (n := n) (dModel := dModel) cfg (mask := mask)

/--
Config record for `transformerEncoderStack`.

This builds `layers` copies of `transformerEncoderBlock`, allocating seeds in a fixed stride.
-/
structure TransformerEncoderStack where
  /-- Layer stack. -/
  layers : Nat
  /-- Template config for each block (its `seedBase` is ignored; we allocate per-layer seeds). -/
  block : TransformerEncoderBlock
  /-- Base seed for the whole stack. -/
  seedBase : Nat := 0
  /-- Seed stride between consecutive blocks (must exceed the per-block seed footprint). -/
  seedStride : Nat := 100

/--
Internal recursion for `transformerEncoderStack`.

Builds `remaining` blocks starting at `layerIdx`, allocating each block's `seedBase` as
`seedBase + layerIdx * seedStride`.
-/
def transformerStackGoWithMask {batch n dModel : Nat} [NeZero n] [NeZero dModel]
    (template : TransformerEncoderBlock) (seedBase seedStride : Nat)
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    (layerIdx : Nat) → (remaining : Nat) →
      Sequential (.dim batch (.dim n (.dim dModel .scalar))) (.dim batch (.dim n (.dim dModel .scalar)))
  | _layerIdx, 0 =>
      _root_.Runtime.Autograd.TorchLean.NN.Seq.id (.dim batch (.dim n (.dim dModel .scalar)))
  | layerIdx, remaining + 1 =>
      let seed := seedBase + layerIdx * seedStride
      let blockCfg : TransformerEncoderBlock := { template with seedBase := seed }
      let here := transformerEncoderBlockWithMask (batch := batch) (n := n) (dModel := dModel) blockCfg
        (mask := mask)
      let rest :=
        transformerStackGoWithMask (batch := batch) (n := n) (dModel := dModel)
          template seedBase seedStride (mask := mask)
          (layerIdx + 1) remaining
      seq! here, rest

/--
Internal recursion for `transformerEncoderStack` (unmasked).

This is `transformerStackGoWithMask` with `mask := none`.
-/
def transformerStackGo {batch n dModel : Nat} [NeZero n] [NeZero dModel]
    (template : TransformerEncoderBlock) (seedBase seedStride : Nat) :
    (layerIdx : Nat) → (remaining : Nat) →
      Sequential (.dim batch (.dim n (.dim dModel .scalar))) (.dim batch (.dim n (.dim dModel .scalar))) :=
  transformerStackGoWithMask (batch := batch) (n := n) (dModel := dModel)
    template seedBase seedStride (mask := none)

/--
Stack `cfg.layers` copies of `blocks.transformerEncoderBlock`.

TorchLean analogue of composing `torch.nn.TransformerEncoderLayer` into a
`torch.nn.TransformerEncoder`, using `Seq` composition for the typed model.
-/
def transformerEncoderStackWithMask {batch n dModel : Nat} [NeZero n] [NeZero dModel]
    (cfg : TransformerEncoderStack)
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    Sequential (.dim batch (.dim n (.dim dModel .scalar))) (.dim batch (.dim n (.dim dModel .scalar))) :=
  transformerStackGoWithMask (batch := batch) (n := n) (dModel := dModel)
    cfg.block cfg.seedBase cfg.seedStride (mask := mask) 0 cfg.layers

/--
Stack `cfg.layers` copies of `blocks.transformerEncoderBlock`.

This is `transformerEncoderStackWithMask`; pass `mask := ...` to enable causal masking (or other
attention masks).
-/
def transformerEncoderStack {batch n dModel : Nat} [NeZero n] [NeZero dModel]
    (cfg : TransformerEncoderStack)
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    Sequential (.dim batch (.dim n (.dim dModel .scalar))) (.dim batch (.dim n (.dim dModel .scalar))) :=
  transformerEncoderStackWithMask (batch := batch) (n := n) (dModel := dModel) cfg (mask := mask)

/--
Transformer encoder followed by a flatten+linear classification head.

PyTorch analogue (approximately): `nn.TransformerEncoder(...)` + pooling/flattening + `nn.linear`.
-/
def transformerEncoderClassifier {batch n dModel : Nat} [NeZero n] [NeZero dModel]
    (classes : Nat) (cfg : TransformerEncoderStack) :
    Sequential (.dim batch (.dim n (.dim dModel .scalar))) (.dim batch (.dim classes .scalar)) :=
  let enc := transformerEncoderStack (batch := batch) (n := n) (dModel := dModel) cfg
  let seedHeadW := cfg.seedBase + cfg.layers * cfg.seedStride
  let seedHeadB := seedHeadW + 1
  let flat : Sequential (.dim batch (.dim n (.dim dModel .scalar)))
      (.dim batch (.dim (Spec.Shape.size (.dim n (.dim dModel .scalar))) .scalar)) :=
    flattenBatch (n := batch) (s := .dim n (.dim dModel .scalar))
  let head : Sequential (.dim batch (.dim (Spec.Shape.size (.dim n (.dim dModel .scalar))) .scalar))
      (.dim batch (.dim classes .scalar)) :=
    linear (Spec.Shape.size (.dim n (.dim dModel .scalar))) classes
      (seedW := seedHeadW) (seedB := seedHeadB)
      (pfx := .dim batch .scalar)
  seq! enc, flat, head

end blocks
