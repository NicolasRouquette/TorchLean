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

The input/output contract is MAE-style: a masked channel/spatial tensor is mapped to a flattened
reconstruction vector while every leading axis is preserved.

`reconDim` can be the full image size (`C*H*W`) or a prefix for faster experiments.
-/
structure VitMaeConfig (d : Nat) where
  /-- Patch-transformer encoder configuration. -/
  encoder : VitConfig d
  /-- Number of reconstructed output coordinates. -/
  reconDim : Nat

/-- Reconstruction-vector output shape after preserving every leading axis. -/
abbrev VitMaeConfig.outputShape {d : Nat} (cfg : VitMaeConfig d)
    (leading : List Nat := []) : List Nat :=
  leading ++ [cfg.reconDim]

/--
Compact ViT-MAE image reconstructor.

This is a real image/patch transformer path:
1. patch embedding by strided convolution,
2. tokenization to `N×numPatches×dModel`,
3. the configured Transformer encoder stack,
4. a linear pixel decoder from encoded patch tokens to a reconstruction vector.

The masking objective is provided by `TorchLean.ssl.BlockMAE.sample`. Its axis policy is
independent of the model architecture and spatial rank, so this constructor uses the same checked
operation as signal, volume, and higher-dimensional masked-prediction models.
-/
def vitMaskedAutoencoder {d : Nat} (cfg : VitMaeConfig d)
    (leading : List Nat := [])
    (h_inC : cfg.encoder.inChannels ≠ 0 := by decide)
    (h_seqLen : cfg.encoder.seqLen ≠ 0 := by decide)
    (h_dModel : cfg.encoder.patch.outChannels ≠ 0 := by decide) :
    nn.Builder (nn.Sequential (cfg.encoder.inputShape leading) (cfg.outputShape leading)) := do
  let encoder ← vitEncoder cfg.encoder leading h_inC h_seqLen h_dModel
  let flattenRaw ← flattenAfter leading
    (s := [cfg.encoder.encodedSeqLen, cfg.encoder.patch.outChannels])
  let flatten : Sequential
      (leading ++ [cfg.encoder.encodedSeqLen, cfg.encoder.patch.outChannels])
      (leading ++ [cfg.encoder.flatDim]) := by
    simpa [VitConfig.flatDim] using flattenRaw
  let decoder ← linear cfg.encoder.flatDim cfg.reconDim (leading := leading)
  pure <| encoder >>> flatten >>> decoder

end models
end nn

end TorchLean
