/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Models.Generative
public import NN.API.Models.Vit

/-!
# Self-Supervised Model Constructors

Most SSL machinery belongs in `TorchLean.ssl`: masks, tensor-to-training-sample transforms, and
objective-facing helpers should work with any compatible model.

This file keeps architecture-level conveniences. The compact MAE constructor below is useful for
examples, but the SSL idea itself is not tied to this model.
-/

@[expose] public section

namespace TorchLean


open Spec Tensor

namespace nn
namespace models

/-! ## ViT-MAE -/

/--
Configuration for a compact masked patch-transformer reconstructor.

The input/output contract is MAE-style:
- input: a masked tensor, `(batch, channels, spatial...)`;
- output: a flattened reconstruction vector, `N×reconDim`.

`reconDim` can be the full image size (`C*H*W`) or a prefix for faster experiments.
-/
structure VitMaeConfig (d : Nat) where
  /-- Patch-transformer encoder configuration. -/
  encoder : VitConfig d
  /-- Number of reconstructed output coordinates. -/
  reconDim : Nat

/-- Masked input shape. -/
def vitMaeInShape {d : Nat} (cfg : VitMaeConfig d) : Spec.Shape :=
  vitInShape cfg.encoder

/-- Reconstruction-vector output shape. -/
def vitMaeOutShape {d : Nat} (cfg : VitMaeConfig d) : Spec.Shape :=
  .dim cfg.encoder.batch (.dim cfg.reconDim .scalar)

/-- Number of patch tokens produced by the ViT-MAE patch embedding. -/
def VitMaeConfig.seqLen {d : Nat} (cfg : VitMaeConfig d) : Nat :=
  cfg.encoder.seqLen

/-- Flattened encoded-token representation size before the MAE decoder head. -/
def VitMaeConfig.flatDim {d : Nat} (cfg : VitMaeConfig d) : Nat :=
  cfg.encoder.flatDim

/--
Compact ViT-MAE image reconstructor.

This is a real image/patch transformer path:
1. patch embedding by strided convolution,
2. tokenization to `N×numPatches×dModel`,
3. one transformer encoder block,
4. a linear pixel decoder from encoded patch tokens to a reconstruction vector.

The masking objective is provided by `TorchLean.ssl.blockMaeSample`. Its axis policy is independent of
the model architecture and spatial rank, so this constructor uses the same checked operation as
signal, volume, and higher-dimensional masked-prediction models.
-/
def vitMaskedAutoencoder {d : Nat} (cfg : VitMaeConfig d)
    (h_inC : cfg.encoder.inChannels ≠ 0 := by decide)
    (h_seqLen : cfg.seqLen ≠ 0 := by decide)
    (h_dModel : cfg.encoder.patch.outChannels ≠ 0 := by decide) :
    nn.M (nn.Sequential (vitMaeInShape cfg) (vitMaeOutShape cfg)) :=
  let vitCfg := cfg.encoder
  letI : NeZero vitCfg.inChannels := ⟨h_inC⟩
  letI : NeZero vitCfg.seqLen := ⟨h_seqLen⟩
  letI : NeZero vitCfg.patch.outChannels := ⟨h_dModel⟩
  let patchEmbedding :=
    nn.conv (leading := .dim vitCfg.batch .scalar) vitCfg.spatial vitCfg.patch
  nn.Sequential![
    patchEmbedding,
    nn.lift (nn.of (spatialToTokens vitCfg)),
    nn.transformerEncoderBlock
      { numHeads := vitCfg.numHeads
        headDim := vitCfg.headDim
        ffnHidden := vitCfg.ffnHidden
        activation := .gelu
        dropout? := none },
    flattenBatch,
    linear cfg.flatDim cfg.reconDim (pfx := .dim vitCfg.batch .scalar)
  ]

/--
Compact vector masked autoencoder.

Architecturally this reuses the vector autoencoder body; the self-supervised part is in
`TorchLean.ssl.vectorMaeSample` or `TorchLean.ssl.tensorPrefixMaeSample`, which mask the input while
keeping the original tensor content as the target.
-/
def vectorMaskedAutoencoder (cfg : VectorGenerativeConfig) :
    nn.M (nn.Sequential (vectorDataShape cfg) (vectorDataShape cfg)) :=
  vectorAutoencoder cfg

end models
end nn

end TorchLean
