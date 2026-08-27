/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Module.Activation
public import NN.Spec.Module.Conv
public import NN.Spec.Module.Flatten
public import NN.Spec.Module.Linear
public import NN.Spec.Module.Pooling

/-!
# Convolutional Network Specifications

This module defines a two-block convolutional network over an arbitrary number of spatial axes.
Its spatial parameters are vectors, so the same model definition applies to sequence,
image, volume, and higher-rank data. Both the compositional module description and the explicit
reverse-mode specification use the generic convolution and pooling operations.
-/

@[expose] public section

namespace Models

open Spec.Module
open Spec
open Tensor
open Activation

namespace Cnn

/-- Spatial shape after one convolution followed by one pooling operation. -/
def blockOutSpatial {d : Nat} (spatial kernel convStride convPadding poolKernel poolStride
    poolPadding : Spec.Tensor Nat [d]) : Spec.Tensor Nat [d] :=
  poolOutSpatialPad (convOutSpatial spatial kernel convStride convPadding)
    poolKernel poolStride poolPadding

/-- Spatial shape after two convolution-pooling blocks. -/
def outputSpatial {d : Nat} (spatial kernel convStride₁ convPadding₁ convStride₂ convPadding₂
    poolKernel poolStride₁ poolPadding₁ poolStride₂ poolPadding₂ : Spec.Tensor Nat [d]) : Spec.Tensor Nat [d] :=
  blockOutSpatial
    (blockOutSpatial spatial kernel convStride₁ convPadding₁ poolKernel poolStride₁ poolPadding₁)
    kernel convStride₂ convPadding₂ poolKernel poolStride₂ poolPadding₂

/-- Feature-map shape after the second pooling operation. -/
def featureShape {d : Nat} (channels : Nat)
    (spatial kernel convStride₁ convPadding₁ convStride₂ convPadding₂ poolKernel poolStride₁
      poolPadding₁ poolStride₂ poolPadding₂ : Spec.Tensor Nat [d]) : Shape :=
  Shape.ofList (channels ::
    (outputSpatial spatial kernel convStride₁ convPadding₁ convStride₂ convPadding₂ poolKernel
      poolStride₁ poolPadding₁ poolStride₂ poolPadding₂).toList)

/-- Number of scalar features presented to the linear head. -/
def featureSize {d : Nat} (channels : Nat)
    (spatial kernel convStride₁ convPadding₁ convStride₂ convPadding₂ poolKernel poolStride₁
      poolPadding₁ poolStride₂ poolPadding₂ : Spec.Tensor Nat [d]) : Nat :=
  Shape.size (featureShape channels spatial kernel convStride₁ convPadding₁ convStride₂
    convPadding₂ poolKernel poolStride₁ poolPadding₁ poolStride₂ poolPadding₂)

/-- Two convolution-pooling blocks followed by a linear head. -/
def spec {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    {d inChannels hiddenChannels outputSize : Nat}
    {spatial kernel convStride₁ convPadding₁ convStride₂ convPadding₂ poolKernel poolStride₁
      poolPadding₁ poolStride₂ poolPadding₂ : Spec.Tensor Nat [d]}
    {hPoolKernel : ∀ i : Fin d, poolKernel.getScalar i ≠ 0}
    {hPoolStride₁ : ∀ i : Fin d, poolStride₁.getScalar i ≠ 0}
    {hPoolStride₂ : ∀ i : Fin d, poolStride₂.getScalar i ≠ 0}
    (conv₁ : ConvSpec d inChannels hiddenChannels kernel convStride₁ convPadding₁ α)
    (conv₂ : ConvSpec d hiddenChannels hiddenChannels kernel convStride₂ convPadding₂ α)
    (pool₁ : MaxPoolSpec d poolKernel poolStride₁ poolPadding₁ hPoolKernel hPoolStride₁)
    (pool₂ : MaxPoolSpec d poolKernel poolStride₂ poolPadding₂ hPoolKernel hPoolStride₂)
    (head : LinearSpec α
      (featureSize hiddenChannels spatial kernel convStride₁ convPadding₁ convStride₂
        convPadding₂ poolKernel poolStride₁ poolPadding₁ poolStride₂ poolPadding₂)
      outputSize) :
    Spec.Module.Chain α (Shape.ofList (inChannels :: spatial.toList))
      (.dim outputSize .scalar) :=
  let convSpatial₁ := convOutSpatial spatial kernel convStride₁ convPadding₁
  let pooledSpatial₁ := poolOutSpatialPad convSpatial₁ poolKernel poolStride₁ poolPadding₁
  let convSpatial₂ := convOutSpatial pooledSpatial₁ kernel convStride₂ convPadding₂
  let pooledSpatial₂ := poolOutSpatialPad convSpatial₂ poolKernel poolStride₂ poolPadding₂
  let convModule₁ : Spec.Module α (Shape.ofList (inChannels :: spatial.toList))
      (Shape.ofList (hiddenChannels :: convSpatial₁.toList)) := Spec.Module.conv conv₁
  let poolModule₁ : Spec.Module α (Shape.ofList (hiddenChannels :: convSpatial₁.toList))
      (Shape.ofList (hiddenChannels :: pooledSpatial₁.toList)) := Spec.Module.maxPool pool₁
  let convModule₂ : Spec.Module α (Shape.ofList (hiddenChannels :: pooledSpatial₁.toList))
      (Shape.ofList (hiddenChannels :: convSpatial₂.toList)) := Spec.Module.conv conv₂
  let poolModule₂ : Spec.Module α (Shape.ofList (hiddenChannels :: convSpatial₂.toList))
      (Shape.ofList (hiddenChannels :: pooledSpatial₂.toList)) := Spec.Module.maxPool pool₂
  let flattenModule := Spec.Module.flatten α (Shape.ofList (hiddenChannels :: pooledSpatial₂.toList))
  let headModule := Spec.Module.linear head
  Spec.Module.Chain.single convModule₁
    |>.append poolModule₁
    |>.append convModule₂
    |>.append poolModule₂
    |>.append flattenModule
    |>.append headModule

/-- The same network with ReLU after each convolution. -/
def withReluSpec {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    {d inChannels hiddenChannels outputSize : Nat}
    {spatial kernel convStride₁ convPadding₁ convStride₂ convPadding₂ poolKernel poolStride₁
      poolPadding₁ poolStride₂ poolPadding₂ : Spec.Tensor Nat [d]}
    {hPoolKernel : ∀ i : Fin d, poolKernel.getScalar i ≠ 0}
    {hPoolStride₁ : ∀ i : Fin d, poolStride₁.getScalar i ≠ 0}
    {hPoolStride₂ : ∀ i : Fin d, poolStride₂.getScalar i ≠ 0}
    (conv₁ : ConvSpec d inChannels hiddenChannels kernel convStride₁ convPadding₁ α)
    (conv₂ : ConvSpec d hiddenChannels hiddenChannels kernel convStride₂ convPadding₂ α)
    (pool₁ : MaxPoolSpec d poolKernel poolStride₁ poolPadding₁ hPoolKernel hPoolStride₁)
    (pool₂ : MaxPoolSpec d poolKernel poolStride₂ poolPadding₂ hPoolKernel hPoolStride₂)
    (head : LinearSpec α
      (featureSize hiddenChannels spatial kernel convStride₁ convPadding₁ convStride₂
        convPadding₂ poolKernel poolStride₁ poolPadding₁ poolStride₂ poolPadding₂)
      outputSize) :
    Spec.Module.Chain α (Shape.ofList (inChannels :: spatial.toList))
      (.dim outputSize .scalar) :=
  let convSpatial₁ := convOutSpatial spatial kernel convStride₁ convPadding₁
  let pooledSpatial₁ := poolOutSpatialPad convSpatial₁ poolKernel poolStride₁ poolPadding₁
  let convSpatial₂ := convOutSpatial pooledSpatial₁ kernel convStride₂ convPadding₂
  let pooledSpatial₂ := poolOutSpatialPad convSpatial₂ poolKernel poolStride₂ poolPadding₂
  let convModule₁ : Spec.Module α (Shape.ofList (inChannels :: spatial.toList))
      (Shape.ofList (hiddenChannels :: convSpatial₁.toList)) := Spec.Module.conv conv₁
  let reluModule₁ := Spec.Module.relu (α := α)
    (Shape.ofList (hiddenChannels :: convSpatial₁.toList))
  let poolModule₁ : Spec.Module α (Shape.ofList (hiddenChannels :: convSpatial₁.toList))
      (Shape.ofList (hiddenChannels :: pooledSpatial₁.toList)) := Spec.Module.maxPool pool₁
  let convModule₂ : Spec.Module α (Shape.ofList (hiddenChannels :: pooledSpatial₁.toList))
      (Shape.ofList (hiddenChannels :: convSpatial₂.toList)) := Spec.Module.conv conv₂
  let reluModule₂ := Spec.Module.relu (α := α)
    (Shape.ofList (hiddenChannels :: convSpatial₂.toList))
  let poolModule₂ : Spec.Module α (Shape.ofList (hiddenChannels :: convSpatial₂.toList))
      (Shape.ofList (hiddenChannels :: pooledSpatial₂.toList)) := Spec.Module.maxPool pool₂
  let flattenModule := Spec.Module.flatten α (Shape.ofList (hiddenChannels :: pooledSpatial₂.toList))
  let headModule := Spec.Module.linear head
  Spec.Module.Chain.single convModule₁
    |>.append reluModule₁
    |>.append poolModule₁
    |>.append convModule₂
    |>.append reluModule₂
    |>.append poolModule₂
    |>.append flattenModule
    |>.append headModule

/-- Evaluate a convolutional chain on one input tensor. -/
def forward {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
    {d inChannels hiddenChannels outputSize : Nat}
    {spatial kernel convStride₁ convPadding₁ convStride₂ convPadding₂ poolKernel poolStride₁
      poolPadding₁ poolStride₂ poolPadding₂ : Spec.Tensor Nat [d]}
    {hPoolKernel : ∀ i : Fin d, poolKernel.getScalar i ≠ 0}
    {hPoolStride₁ : ∀ i : Fin d, poolStride₁.getScalar i ≠ 0}
    {hPoolStride₂ : ∀ i : Fin d, poolStride₂.getScalar i ≠ 0}
    (conv₁ : ConvSpec d inChannels hiddenChannels kernel convStride₁ convPadding₁ α)
    (conv₂ : ConvSpec d hiddenChannels hiddenChannels kernel convStride₂ convPadding₂ α)
    (pool₁ : MaxPoolSpec d poolKernel poolStride₁ poolPadding₁ hPoolKernel hPoolStride₁)
    (pool₂ : MaxPoolSpec d poolKernel poolStride₂ poolPadding₂ hPoolKernel hPoolStride₂)
    (head : LinearSpec α
      (featureSize hiddenChannels spatial kernel convStride₁ convPadding₁ convStride₂
        convPadding₂ poolKernel poolStride₁ poolPadding₁ poolStride₂ poolPadding₂)
      outputSize)
    (x : Tensor α (Shape.ofList (inChannels :: spatial.toList))) :
    Tensor α [outputSize] :=
  (spec conv₁ conv₂ pool₁ pool₂ head).forward x

end Cnn

namespace TwoBlockCnn

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-- Hyperparameters for a two-block convolutional network of spatial rank `d`. -/
structure Config (d : Nat) where
  conv1Channels : Nat := 32
  conv2Channels : Nat := 64
  outputSize : Nat := 10
  kernel : Spec.Tensor Nat [d] := Spec.fill 3 [d]
  conv1Stride : Spec.Tensor Nat [d] := Spec.fill 1 [d]
  conv1Padding : Spec.Tensor Nat [d] := Spec.fill 1 [d]
  conv2Stride : Spec.Tensor Nat [d] := Spec.fill 1 [d]
  conv2Padding : Spec.Tensor Nat [d] := Spec.fill 1 [d]
  poolKernel : Spec.Tensor Nat [d] := Spec.fill 2 [d]
  poolStride1 : Spec.Tensor Nat [d] := Spec.fill 2 [d]
  poolPadding1 : Spec.Tensor Nat [d] := Spec.fill 0 [d]
  poolStride2 : Spec.Tensor Nat [d] := Spec.fill 2 [d]
  poolPadding2 : Spec.Tensor Nat [d] := Spec.fill 0 [d]

/-- Conditions needed by convolutional and pooling implementations. -/
structure Config.WF {d : Nat} (cfg : Config d) : Prop where
  conv1Channels_ne_zero : cfg.conv1Channels ≠ 0
  conv2Channels_ne_zero : cfg.conv2Channels ≠ 0
  outputSize_ne_zero : cfg.outputSize ≠ 0
  kernel_ne_zero : ∀ i : Fin d, cfg.kernel.getScalar i ≠ 0
  conv1Stride_ne_zero : ∀ i : Fin d, cfg.conv1Stride.getScalar i ≠ 0
  conv2Stride_ne_zero : ∀ i : Fin d, cfg.conv2Stride.getScalar i ≠ 0
  poolKernel_ne_zero : ∀ i : Fin d, cfg.poolKernel.getScalar i ≠ 0
  poolStride1_ne_zero : ∀ i : Fin d, cfg.poolStride1.getScalar i ≠ 0
  poolStride2_ne_zero : ∀ i : Fin d, cfg.poolStride2.getScalar i ≠ 0

/-- The default configuration at any spatial rank. -/
def defaultConfig (d : Nat) : Config d := {}

/-- The default configuration is well formed. -/
theorem defaultConfig_wf (d : Nat) : (defaultConfig d).WF := by
  constructor
  · simp [defaultConfig]
  · simp [defaultConfig]
  · simp [defaultConfig]
  · intro i
    simp [defaultConfig]
  · intro i
    simp [defaultConfig]
  · intro i
    simp [defaultConfig]
  · intro i
    simp [defaultConfig]
  · intro i
    simp [defaultConfig]
  · intro i
    simp [defaultConfig]

/-- A generic two-block convolutional network with an explicit linear head. -/
structure Model {d : Nat} (cfg : Config d) (inChannels : Nat) (spatial : Spec.Tensor Nat [d])
    (α : Type) (hCfg : cfg.WF) where
  conv1 : ConvSpec d inChannels cfg.conv1Channels cfg.kernel cfg.conv1Stride cfg.conv1Padding α
  conv2 : ConvSpec d cfg.conv1Channels cfg.conv2Channels cfg.kernel cfg.conv2Stride
    cfg.conv2Padding α
  pool1 : MaxPoolSpec d cfg.poolKernel cfg.poolStride1 cfg.poolPadding1
    hCfg.poolKernel_ne_zero hCfg.poolStride1_ne_zero
  pool2 : MaxPoolSpec d cfg.poolKernel cfg.poolStride2 cfg.poolPadding2
    hCfg.poolKernel_ne_zero hCfg.poolStride2_ne_zero
  head : LinearSpec α
    (Cnn.featureSize cfg.conv2Channels spatial cfg.kernel cfg.conv1Stride cfg.conv1Padding
      cfg.conv2Stride cfg.conv2Padding cfg.poolKernel cfg.poolStride1 cfg.poolPadding1
      cfg.poolStride2 cfg.poolPadding2)
    cfg.outputSize

/-- Parameter gradients for `Model`. -/
structure Grads {d : Nat} (cfg : Config d) (inChannels : Nat) (spatial : Spec.Tensor Nat [d])
    (α : Type) where
  conv1Kernel : Tensor α (Shape.ofList (cfg.conv1Channels :: inChannels :: cfg.kernel.toList))
  conv1Bias : Tensor α [cfg.conv1Channels]
  conv2Kernel : Tensor α
    (Shape.ofList (cfg.conv2Channels :: cfg.conv1Channels :: cfg.kernel.toList))
  conv2Bias : Tensor α [cfg.conv2Channels]
  headWeight : Tensor α [cfg.outputSize, Cnn.featureSize cfg.conv2Channels spatial cfg.kernel cfg.conv1Stride cfg.conv1Padding
      cfg.conv2Stride cfg.conv2Padding cfg.poolKernel cfg.poolStride1 cfg.poolPadding1
      cfg.poolStride2 cfg.poolPadding2]
  headBias : Tensor α [cfg.outputSize]

/-- Forward pass for `Model`. -/
def Model.forward {d : Nat} {cfg : Config d} {inChannels : Nat} {spatial : Spec.Tensor Nat [d]}
    {hCfg : cfg.WF} (m : Model cfg inChannels spatial α hCfg)
    (x : Tensor α (Shape.ofList (inChannels :: spatial.toList))) :
    Tensor α [cfg.outputSize] :=
  let y₁ := convSpec m.conv1 x
  let r₁ := reluSpec y₁
  let p₁ := maxPoolSpec m.pool1 r₁
  let y₂ := convSpec m.conv2 p₁
  let r₂ := reluSpec y₂
  let p₂ := maxPoolSpec m.pool2 r₂
  linearSpec m.head (Tensor.flattenSpec p₂)

/-- Reverse-mode parameter and input derivatives for `Model`. -/
def Model.backward {d : Nat} {cfg : Config d} {inChannels : Nat} {spatial : Spec.Tensor Nat [d]}
    {hCfg : cfg.WF} (m : Model cfg inChannels spatial α hCfg)
    (x : Tensor α (Shape.ofList (inChannels :: spatial.toList)))
    (gradOutput : Tensor α [cfg.outputSize]) :
    Grads cfg inChannels spatial α × Tensor α (Shape.ofList (inChannels :: spatial.toList)) :=
  let convSpatial₁ := convOutSpatial spatial cfg.kernel cfg.conv1Stride cfg.conv1Padding
  let pooledSpatial₁ := poolOutSpatialPad convSpatial₁ cfg.poolKernel cfg.poolStride1 cfg.poolPadding1
  let convSpatial₂ := convOutSpatial pooledSpatial₁ cfg.kernel cfg.conv2Stride cfg.conv2Padding
  let pooledSpatial₂ := poolOutSpatialPad convSpatial₂ cfg.poolKernel cfg.poolStride2 cfg.poolPadding2
  let y₁ := convSpec m.conv1 x
  let r₁ := reluSpec y₁
  let p₁ := maxPoolSpec m.pool1 r₁
  let y₂ := convSpec m.conv2 p₁
  let r₂ := reluSpec y₂
  let p₂ := maxPoolSpec m.pool2 r₂
  let flat := Tensor.flattenSpec p₂
  let (headWeight, headBias, dFlat) := linearBackwardSpec m.head flat gradOutput
  let featureShape := Shape.ofList (cfg.conv2Channels :: pooledSpatial₂.toList)
  let dP₂ : Tensor α featureShape := Tensor.unflattenSpec featureShape dFlat
  let dR₂ := maxPoolBackwardSpec m.pool2 r₂ dP₂
  let dY₂ := mulSpec dR₂ (reluDerivSpec y₂)
  let (conv2Kernel, conv2Bias, dP₁) := convBackwardSpec m.conv2 p₁ dY₂
  let dR₁ := maxPoolBackwardSpec m.pool1 r₁ dP₁
  let dY₁ := mulSpec dR₁ (reluDerivSpec y₁)
  let (conv1Kernel, conv1Bias, dX) := convBackwardSpec m.conv1 x dY₁
  ({ conv1Kernel := conv1Kernel
     conv1Bias := conv1Bias
     conv2Kernel := conv2Kernel
     conv2Bias := conv2Bias
     headWeight := headWeight
     headBias := headBias }, dX)

end TwoBlockCnn

end Models
