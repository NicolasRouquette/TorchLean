/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Primitives.Spatial

/-!
# GraphSpec Convolutional Classifier

A graph-authored classifier with two convolution-pooling blocks and a linear head. Every spatial
quantity is a `Spec.Tensor Nat [d]`, so the definition applies unchanged to signals, images,
volumes, and higher-dimensional grids.
-/

@[expose] public section

namespace NN
namespace GraphSpec
namespace Models

open _root_.Spec

/-- Spatial extent after the two convolution-pooling blocks. -/
def twoConvOutputSpatial {d : Nat}
    (spatial kernel convStride₁ convPadding₁ convStride₂ convPadding₂
      poolKernel poolStride₁ poolPadding₁ poolStride₂ poolPadding₂ : Spec.Tensor Nat [d]) :
    Spec.Tensor Nat [d] :=
  let convSpatial₁ := Spec.convOutSpatial spatial kernel convStride₁ convPadding₁
  let pooledSpatial₁ :=
    Spec.poolOutSpatialPad convSpatial₁ poolKernel poolStride₁ poolPadding₁
  let convSpatial₂ := Spec.convOutSpatial pooledSpatial₁ kernel convStride₂ convPadding₂
  Spec.poolOutSpatialPad convSpatial₂ poolKernel poolStride₂ poolPadding₂

/-- Shape of the final convolutional feature map. -/
def twoConvFeatureShape {d channels : Nat}
    (spatial kernel convStride₁ convPadding₁ convStride₂ convPadding₂
      poolKernel poolStride₁ poolPadding₁ poolStride₂ poolPadding₂ : Spec.Tensor Nat [d]) : Shape :=
  Shape.ofList (channels ::
    (twoConvOutputSpatial spatial kernel convStride₁ convPadding₁ convStride₂ convPadding₂
      poolKernel poolStride₁ poolPadding₁ poolStride₂ poolPadding₂).toList)

/-- Number of scalar features consumed by the linear head. -/
def twoConvFeatureSize {d channels : Nat}
    (spatial kernel convStride₁ convPadding₁ convStride₂ convPadding₂
      poolKernel poolStride₁ poolPadding₁ poolStride₂ poolPadding₂ : Spec.Tensor Nat [d]) : Nat :=
  Shape.size <| twoConvFeatureShape (channels := channels) spatial kernel
    convStride₁ convPadding₁ convStride₂ convPadding₂ poolKernel
    poolStride₁ poolPadding₁ poolStride₂ poolPadding₂

/--
Two convolution-pooling blocks followed by a linear classifier.

The parameter list records both convolution kernels and biases followed by the linear head. The
input has shape `(inChannels, spatial...)`; no batch axis is built into the architecture.
-/
def twoConvCnn {d inChannels firstChannels secondChannels outputSize : Nat}
    (spatial kernel convStride₁ convPadding₁ convStride₂ convPadding₂
      poolKernel poolStride₁ poolPadding₁ poolStride₂ poolPadding₂ : Spec.Tensor Nat [d])
    {hInChannels : inChannels ≠ 0}
    {hFirstChannels : firstChannels ≠ 0}
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    {hConvStride₁ : ∀ i : Fin d, convStride₁.getScalar i ≠ 0}
    {hConvStride₂ : ∀ i : Fin d, convStride₂.getScalar i ≠ 0}
    {hPoolKernel : ∀ i : Fin d, poolKernel.getScalar i ≠ 0}
    {hPoolStride₁ : ∀ i : Fin d, poolStride₁.getScalar i ≠ 0}
    {hPoolStride₂ : ∀ i : Fin d, poolStride₂.getScalar i ≠ 0} :
    Chain
      [ Shape.ofList (firstChannels :: inChannels :: kernel.toList)
      , [firstChannels]
      , Shape.ofList (secondChannels :: firstChannels :: kernel.toList)
      , [secondChannels]
      , [outputSize,
          (twoConvFeatureSize (channels := secondChannels) spatial kernel
            convStride₁ convPadding₁ convStride₂ convPadding₂ poolKernel
            poolStride₁ poolPadding₁ poolStride₂ poolPadding₂)]
      , [outputSize] ]
      (Shape.ofList (inChannels :: spatial.toList)) [outputSize] :=
  let convSpatial₁ := Spec.convOutSpatial spatial kernel convStride₁ convPadding₁
  let pooledSpatial₁ :=
    Spec.poolOutSpatialPad convSpatial₁ poolKernel poolStride₁ poolPadding₁
  let convSpatial₂ := Spec.convOutSpatial pooledSpatial₁ kernel convStride₂ convPadding₂
  let pooledSpatial₂ :=
    Spec.poolOutSpatialPad convSpatial₂ poolKernel poolStride₂ poolPadding₂
  let featureShape := Shape.ofList (secondChannels :: pooledSpatial₂.toList)
  Chain.conv inChannels firstChannels kernel convStride₁ convPadding₁ spatial
      (hInC := hInChannels) (hKernel := hKernel) (hStride := hConvStride₁) >>>
    Chain.relu (Shape.ofList (firstChannels :: convSpatial₁.toList)) >>>
    Chain.maxPool firstChannels poolKernel poolStride₁ poolPadding₁ convSpatial₁
      (hKernel := hPoolKernel) (hStride := hPoolStride₁) >>>
    Chain.conv firstChannels secondChannels kernel convStride₂ convPadding₂ pooledSpatial₁
      (hInC := hFirstChannels) (hKernel := hKernel) (hStride := hConvStride₂) >>>
    Chain.relu (Shape.ofList (secondChannels :: convSpatial₂.toList)) >>>
    Chain.maxPool secondChannels poolKernel poolStride₂ poolPadding₂ convSpatial₂
      (hKernel := hPoolKernel) (hStride := hPoolStride₂) >>>
    Chain.flatten featureShape >>>
    Chain.linear
      (inDim := twoConvFeatureSize (channels := secondChannels) spatial kernel
        convStride₁ convPadding₁ convStride₂ convPadding₂ poolKernel
        poolStride₁ poolPadding₁ poolStride₂ poolPadding₂)
      (outDim := outputSize)

/-- Lower `twoConvCnn` structurally to the DAG representation with zero-initialized parameters. -/
def twoConvCnnDAGModelZeroInit {d inChannels firstChannels secondChannels outputSize : Nat}
    (spatial kernel convStride₁ convPadding₁ convStride₂ convPadding₂
      poolKernel poolStride₁ poolPadding₁ poolStride₂ poolPadding₂ : Spec.Tensor Nat [d])
    {hInChannels : inChannels ≠ 0}
    {hFirstChannels : firstChannels ≠ 0}
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    {hConvStride₁ : ∀ i : Fin d, convStride₁.getScalar i ≠ 0}
    {hConvStride₂ : ∀ i : Fin d, convStride₂.getScalar i ≠ 0}
    {hPoolKernel : ∀ i : Fin d, poolKernel.getScalar i ≠ 0}
    {hPoolStride₁ : ∀ i : Fin d, poolStride₁.getScalar i ≠ 0}
    {hPoolStride₂ : ∀ i : Fin d, poolStride₂.getScalar i ≠ 0} :
    DAG.Model
      [ Shape.ofList (firstChannels :: inChannels :: kernel.toList)
      , [firstChannels]
      , Shape.ofList (secondChannels :: firstChannels :: kernel.toList)
      , [secondChannels]
      , [outputSize,
          (twoConvFeatureSize (channels := secondChannels) spatial kernel
            convStride₁ convPadding₁ convStride₂ convPadding₂ poolKernel
            poolStride₁ poolPadding₁ poolStride₂ poolPadding₂)]
      , [outputSize] ]
      [Shape.ofList (inChannels :: spatial.toList)] [outputSize] :=
  LowerToDAG.Chain.toDAGModelZeroInit <|
    twoConvCnn (inChannels := inChannels) (firstChannels := firstChannels)
      (secondChannels := secondChannels) (outputSize := outputSize)
      spatial kernel convStride₁ convPadding₁ convStride₂ convPadding₂
      poolKernel poolStride₁ poolPadding₁ poolStride₂ poolPadding₂
      (hInChannels := hInChannels) (hFirstChannels := hFirstChannels)
      (hKernel := hKernel) (hConvStride₁ := hConvStride₁) (hConvStride₂ := hConvStride₂)
      (hPoolKernel := hPoolKernel) (hPoolStride₁ := hPoolStride₁)
      (hPoolStride₂ := hPoolStride₂)

end Models
end GraphSpec
end NN
