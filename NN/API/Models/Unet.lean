/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded

/-!
# U-Net

A typed encoder-decoder with one downsampling stage, one upsampling stage, and a channel-wise skip
connection. The spatial rank is a parameter: the same constructor applies to sequences, images,
volumes, and higher-dimensional grids.

The reusable operations remain visible in the type. Pooling determines the bottleneck extent,
transpose convolution returns it to the input extent, and branch concatenation changes the channel
count from `baseChannels` to `baseChannels + baseChannels`.

Reference: O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical
Image Segmentation," MICCAI 2015.
-/

@[expose] public section

namespace TorchLean
namespace nn
namespace models

/-- Configuration for a one-level U-Net over `d` spatial axes. -/
structure UnetConfig (d : Nat) where
  /-- Number of channels in each input sample. -/
  inChannels : Nat
  /-- Width of the full-resolution feature map. -/
  baseChannels : Nat
  /-- Number of channels produced by the output projection. -/
  outChannels : Nat
  /-- Extent of each input spatial axis. -/
  spatial : Tensor Nat [d]
  /-- Every input spatial axis is nonempty. -/
  spatialNonzero : ∀ i : Fin d, spatial.getScalar i ≠ 0
  /-- Downsampling operation. -/
  pool : Pool d
  /-- Every downsampled spatial axis is nonempty. -/
  pooledNonzero : ∀ i : Fin d,
    (Spec.poolOutSpatialPad spatial pool.kernel pool.stride pool.padding).getScalar i ≠ 0
  /-- Geometry used by the convolutional blocks at both resolutions. -/
  block : ConvGeometry d
  /-- The full-resolution convolutional blocks preserve the spatial extent. -/
  blockPreservesSpatial :
    Spec.convOutSpatial spatial block.kernel block.stride block.padding = spatial
  /-- The bottleneck convolutional blocks preserve the pooled spatial extent. -/
  blockPreservesPooledSpatial :
    Spec.convOutSpatial
      (Spec.poolOutSpatialPad spatial pool.kernel pool.stride pool.padding)
      block.kernel block.stride block.padding =
      Spec.poolOutSpatialPad spatial pool.kernel pool.stride pool.padding
  /-- Kernel extent of the transpose convolution. -/
  upKernel : Tensor Nat [d]
  /-- Stride of the transpose convolution. -/
  upStride : Tensor Nat [d]
  /-- Symmetric padding of the transpose convolution. -/
  upPadding : Tensor Nat [d] := Spec.fill 0 [d]
  /-- Every transpose-convolution kernel extent is positive. -/
  upKernelNonzero : ∀ i : Fin d, upKernel.getScalar i ≠ 0
  /-- Every transpose-convolution stride is positive. -/
  upStrideNonzero : ∀ i : Fin d, upStride.getScalar i ≠ 0
  /-- The upsampling stage restores the original spatial extent. -/
  upsampleShape :
    Spec.convTransposeOutSpatial
      (Spec.poolOutSpatialPad spatial pool.kernel pool.stride pool.padding)
      upKernel upStride upPadding = spatial

namespace UnetConfig

/-- Spatial extent after the downsampling stage. -/
def pooledSpatial {d : Nat} (cfg : UnetConfig d) : Tensor Nat [d] :=
  Spec.poolOutSpatialPad cfg.spatial cfg.pool.kernel cfg.pool.stride cfg.pool.padding

/-- Per-sample input shape. -/
def sampleInputShape {d : Nat} (cfg : UnetConfig d) : List Nat :=
  cfg.inChannels :: cfg.spatial.toList

/-- Per-sample output shape. -/
def sampleOutputShape {d : Nat} (cfg : UnetConfig d) : List Nat :=
  cfg.outChannels :: cfg.spatial.toList

/-- Input shape with arbitrary leading dimensions. -/
def inputShape {d : Nat} (cfg : UnetConfig d)
    (leading : List Nat := []) : List Nat :=
  leading ++ cfg.sampleInputShape

/-- Output shape with the same leading dimensions as the input. -/
def outputShape {d : Nat} (cfg : UnetConfig d)
    (leading : List Nat := []) : List Nat :=
  leading ++ cfg.sampleOutputShape

end UnetConfig

/--
Build a one-level U-Net over an arbitrary spatial rank.

The block geometry is part of `UnetConfig`, so spatially mixing kernels are available at every
rank. The shape equalities in the configuration ensure that both convolutional blocks preserve
their respective resolution before the skip branches are joined.
-/
def unet {d : Nat} (cfg : UnetConfig d) (leading : List Nat := [])
    (hInChannels : cfg.inChannels ≠ 0 := by decide)
    (hBaseChannels : cfg.baseChannels ≠ 0 := by decide) :
    Builder (Sequential (cfg.inputShape leading) (cfg.outputShape leading)) :=
  letI : NeZero cfg.inChannels := ⟨hInChannels⟩
  letI : NeZero cfg.baseChannels := ⟨hBaseChannels⟩
  letI : NeZero (cfg.baseChannels + cfg.baseChannels) :=
    ⟨fun h => hBaseChannels (Nat.eq_zero_of_add_eq_zero h).1⟩
  do
  let stemConv1Raw ← conv [] (inChannels := cfg.inChannels) cfg.spatial
    (cfg.block.toConv cfg.baseChannels)
  let stemConv1 : Sequential cfg.sampleInputShape
      (cfg.baseChannels :: cfg.spatial.toList) := by
    simpa [UnetConfig.sampleInputShape, ConvGeometry.toConv,
      ConvGeometry.outSpatial, cfg.blockPreservesSpatial] using stemConv1Raw
  let stemConv2Raw ← conv [] (inChannels := cfg.baseChannels) cfg.spatial
    (cfg.block.toConv cfg.baseChannels)
  let stemConv2 : Sequential
      (cfg.baseChannels :: cfg.spatial.toList)
      (cfg.baseChannels :: cfg.spatial.toList) := by
    simpa [ConvGeometry.toConv, ConvGeometry.outSpatial,
      cfg.blockPreservesSpatial] using stemConv2Raw
  let stem := seq! stemConv1, nn.Internal.relu, stemConv2, nn.Internal.relu

  let downsample ← maxPool [] (channels := cfg.baseChannels) cfg.spatial cfg.pool
  let bottleneckConv1Raw ← conv [] (inChannels := cfg.baseChannels) cfg.pooledSpatial
    (cfg.block.toConv (cfg.baseChannels + cfg.baseChannels))
  let bottleneckConv1 : Sequential
      (cfg.baseChannels :: cfg.pooledSpatial.toList)
      ((cfg.baseChannels + cfg.baseChannels) :: cfg.pooledSpatial.toList) := by
    simpa [UnetConfig.pooledSpatial, ConvGeometry.toConv,
      ConvGeometry.outSpatial, cfg.blockPreservesPooledSpatial] using bottleneckConv1Raw
  let bottleneckConv2Raw ← conv []
    (inChannels := cfg.baseChannels + cfg.baseChannels) cfg.pooledSpatial
    (cfg.block.toConv (cfg.baseChannels + cfg.baseChannels))
  let bottleneckConv2 : Sequential
      ((cfg.baseChannels + cfg.baseChannels) :: cfg.pooledSpatial.toList)
      ((cfg.baseChannels + cfg.baseChannels) :: cfg.pooledSpatial.toList) := by
    simpa [UnetConfig.pooledSpatial, ConvGeometry.toConv,
      ConvGeometry.outSpatial, cfg.blockPreservesPooledSpatial] using bottleneckConv2Raw
  let bottleneck :=
    seq! bottleneckConv1, nn.Internal.relu, bottleneckConv2, nn.Internal.relu
  let upsampleRaw ← convTranspose []
    (inChannels := cfg.baseChannels + cfg.baseChannels) cfg.pooledSpatial
    { outChannels := cfg.baseChannels
      kernel := cfg.upKernel
      stride := cfg.upStride
      padding := cfg.upPadding
      kernelNonzero := cfg.upKernelNonzero
      strideNonzero := cfg.upStrideNonzero }
  let upsample : Sequential
      ((cfg.baseChannels + cfg.baseChannels) :: cfg.pooledSpatial.toList)
      (cfg.baseChannels :: cfg.spatial.toList) := by
    simpa [UnetConfig.pooledSpatial, cfg.upsampleShape] using upsampleRaw

  let deep := seq! downsample, bottleneck, upsample
  let skip := _root_.Runtime.Autograd.TorchLean.NN.Seq.id
    (Shape.ofList (cfg.baseChannels :: cfg.spatial.toList))
  let merge := blocks.concatBranches skip deep

  let decoderConv1Raw ← conv []
    (inChannels := cfg.baseChannels + cfg.baseChannels) cfg.spatial
    (cfg.block.toConv cfg.baseChannels)
  let decoderConv1 : Sequential
      ((cfg.baseChannels + cfg.baseChannels) :: cfg.spatial.toList)
      (cfg.baseChannels :: cfg.spatial.toList) := by
    simpa [ConvGeometry.toConv, ConvGeometry.outSpatial,
      cfg.blockPreservesSpatial] using decoderConv1Raw
  let decoderConv2Raw ← conv [] (inChannels := cfg.baseChannels) cfg.spatial
    (cfg.block.toConv cfg.baseChannels)
  let decoderConv2 : Sequential
      (cfg.baseChannels :: cfg.spatial.toList)
      (cfg.baseChannels :: cfg.spatial.toList) := by
    simpa [ConvGeometry.toConv, ConvGeometry.outSpatial,
      cfg.blockPreservesSpatial] using decoderConv2Raw
  let head ← pointwiseConv [] cfg.spatial cfg.spatialNonzero cfg.outChannels
  let decoder :=
    seq! decoderConv1, nn.Internal.relu, decoderConv2, nn.Internal.relu, head

  let core : Sequential cfg.sampleInputShape cfg.sampleOutputShape :=
    seq! stem, merge, decoder
  mapEach leading core

end models
end nn
end TorchLean
