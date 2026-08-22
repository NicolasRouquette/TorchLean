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
  activation : _root_.Activation.Kind := .gelu
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
def transformerEncoderBlock (leading : Spec.Shape := .scalar) {n dModel : Nat}
    [NeZero n] [NeZero dModel]
    (cfg : TransformerEncoderBlock)
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    Sequential ((leading.concat (.dim n .scalar)).appendDim dModel)
      ((leading.concat (.dim n .scalar)).appendDim dModel) :=
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

  let tokenLeading := leading.concat (.dim n .scalar)
  let modelShape := tokenLeading.appendDim dModel
  let attn : Sequential modelShape modelShape :=
    multiHeadAttentionWith leading (n := n) (dModel := dModel)
      { numHeads := cfg.numHeads, headDim := cfg.headDim, seedW := seedAttn,
        outputBias := cfg.attentionOutputBias, weightInit? := cfg.weightInit?,
        outputWeightInit? := cfg.residualOutputInit? }
      (hN := NeZero.ne (n := n))
      (mask := mask)
  let attnInner :=
    match cfg.dropout? with
    | none => attn
    | some p =>
        seq! attn, dropout (s := modelShape) p (seed := seedDrop1)
  let norm1 : Sequential modelShape modelShape :=
    layerNorm tokenLeading (width := dModel)
      { seedGamma := seedNorm1Gamma, seedBeta := seedNorm1Beta }

  let ffn : Sequential modelShape modelShape :=
    seq!
      linearWith dModel cfg.ffnHidden { weightInit? := cfg.weightInit? }
        seedFfnW1 seedFfnB1 (leading := tokenLeading),
      activation (s := tokenLeading.appendDim cfg.ffnHidden) cfg.activation,
      linearWith cfg.ffnHidden dModel
        { weightInit? := cfg.residualOutputInit?.orElse (fun _ => cfg.weightInit?) }
        seedFfnW2 seedFfnB2 (leading := tokenLeading)
  let ffnInner :=
    match cfg.dropout? with
    | none => ffn
    | some p =>
        seq! ffn, dropout (s := modelShape) p (seed := seedDrop2)
  let norm2 : Sequential modelShape modelShape :=
    layerNorm tokenLeading (width := dModel)
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

/-- Build the remaining encoder blocks, assigning each block its configured seed interval. -/
def encoderStackLayers (leading : Spec.Shape := .scalar) {n dModel : Nat}
    [NeZero n] [NeZero dModel]
    (template : TransformerEncoderBlock) (seedBase seedStride : Nat)
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    (layerIdx : Nat) → (remaining : Nat) →
      Sequential ((leading.concat (.dim n .scalar)).appendDim dModel)
        ((leading.concat (.dim n .scalar)).appendDim dModel)
  | _layerIdx, 0 =>
      _root_.Runtime.Autograd.TorchLean.NN.Seq.id
        ((leading.concat (.dim n .scalar)).appendDim dModel)
  | layerIdx, remaining + 1 =>
      let seed := seedBase + layerIdx * seedStride
      let blockCfg : TransformerEncoderBlock := { template with seedBase := seed }
      let here := transformerEncoderBlock leading (n := n) (dModel := dModel) blockCfg
        (mask := mask)
      let rest :=
        encoderStackLayers leading (n := n) (dModel := dModel)
          template seedBase seedStride (mask := mask)
          (layerIdx + 1) remaining
      seq! here, rest

/--
Stack `cfg.layers` copies of `blocks.transformerEncoderBlock`.

TorchLean analogue of composing `torch.nn.TransformerEncoderLayer` into a
`torch.nn.TransformerEncoder`, using `Seq` composition for the typed model.
-/
def transformerEncoderStack (leading : Spec.Shape := .scalar) {n dModel : Nat}
    [NeZero n] [NeZero dModel]
    (cfg : TransformerEncoderStack)
    (mask : Option (Spec.Tensor Bool (.dim n (.dim n .scalar))) := none) :
    Sequential ((leading.concat (.dim n .scalar)).appendDim dModel)
      ((leading.concat (.dim n .scalar)).appendDim dModel) :=
  encoderStackLayers leading (n := n) (dModel := dModel)
    cfg.block cfg.seedBase cfg.seedStride (mask := mask) 0 cfg.layers

/--
Transformer encoder followed by a flatten+linear classification head.

PyTorch analogue (approximately): `nn.TransformerEncoder(...)` followed by pooling or flattening
and `nn.Linear`.
-/
def transformerEncoderClassifier (leading : Spec.Shape := .scalar) {n dModel : Nat}
    [NeZero n] [NeZero dModel]
    (classes : Nat) (cfg : TransformerEncoderStack) :
    Sequential ((leading.concat (.dim n .scalar)).appendDim dModel)
      (leading.appendDim classes) :=
  let enc := transformerEncoderStack leading (n := n) (dModel := dModel) cfg
  let seedHeadW := cfg.seedBase + cfg.layers * cfg.seedStride
  let seedHeadB := seedHeadW + 1
  let flat : Sequential ((leading.concat (.dim n .scalar)).appendDim dModel)
      (leading.appendDim (Spec.Shape.size (.dim n (.dim dModel .scalar)))) :=
    by
      simpa [Spec.Shape.appendDim] using
        flattenLeading leading (s := .dim n (.dim dModel .scalar))
  let head : Sequential (leading.appendDim (Spec.Shape.size (.dim n (.dim dModel .scalar))))
      (leading.appendDim classes) :=
    linear (Spec.Shape.size (.dim n (.dim dModel .scalar))) classes
      (seedW := seedHeadW) (seedB := seedHeadB)
      (leading := leading)
  seq! enc, flat, head

end blocks
