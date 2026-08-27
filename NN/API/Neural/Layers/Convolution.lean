/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Neural.Leading

/-!
# Convolution

Arbitrary-rank convolution geometry, configuration, and layer constructors.
-/

@[expose] public section

namespace TorchLean
namespace nn
namespace Internal

/-- Kernel, stride, and padding shared by convolutional layers with different channel widths. -/
structure ConvGeometry (d : Nat) where
  /-- Kernel extent along each spatial axis. -/
  kernel : Tensor Nat [d]
  /-- Step along each spatial axis. -/
  stride : Tensor Nat [d] := Spec.fill 1 [d]
  /-- Symmetric zero-padding along each spatial axis. -/
  padding : Tensor Nat [d] := Spec.fill 0 [d]
  /-- Every kernel extent is positive. -/
  kernelNonzero : ∀ i : Fin d, kernel.getScalar i ≠ 0
  /-- Every stride is positive. -/
  strideNonzero : ∀ i : Fin d, stride.getScalar i ≠ 0

namespace ConvGeometry

/-- Spatial extent produced by this geometry. -/
def outSpatial {d : Nat} (geometry : ConvGeometry d) (input : Tensor Nat [d]) : Tensor Nat [d] :=
  Spec.convOutSpatial input geometry.kernel geometry.stride geometry.padding

/--
Unit-stride geometry with an odd kernel along each axis and padding equal to the kernel radius.
-/
def samePadding {d : Nat} (radius : Tensor Nat [d]) : ConvGeometry d :=
  { kernel := radius.map fun p => 2 * p + 1
    stride := Spec.fill 1 [d]
    padding := radius
    kernelNonzero := by intro i; simp
    strideNonzero := by intro i; simp }

/-- Same-padding geometry preserves every positive spatial extent. -/
theorem outSpatial_samePadding {d : Nat} (spatial radius : Tensor Nat [d])
    (hSpatial : ∀ i : Fin d, spatial.getScalar i ≠ 0) :
    (samePadding radius).outSpatial spatial = spatial := by
  simpa [outSpatial, samePadding] using Spec.convOutSpatial_same spatial radius hSpatial

end ConvGeometry

/-- Configuration shared by arbitrary-dimensional convolution layers. -/
structure Conv (d : Nat) where
  /-- Number of output channels. -/
  outChannels : Nat
  /-- Kernel extent along each spatial axis. -/
  kernel : Tensor Nat [d]
  /-- Step along each spatial axis. -/
  stride : Tensor Nat [d] := Spec.fill 1 [d]
  /-- Symmetric zero-padding along each spatial axis. -/
  padding : Tensor Nat [d] := Spec.fill 0 [d]
  /-- Every kernel extent is positive. -/
  kernelNonzero : ∀ i : Fin d, kernel.getScalar i ≠ 0
  /-- Every stride is positive. -/
  strideNonzero : ∀ i : Fin d, stride.getScalar i ≠ 0
  /-- Initialization scheme for the kernel weights. -/
  kernelInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .uniform (-0.1) 0.1

/-- Build a convolution configuration by adding an output-channel width to shared geometry. -/
def ConvGeometry.toConv {d : Nat} (geometry : ConvGeometry d) (outChannels : Nat) : Conv d :=
  { outChannels
    kernel := geometry.kernel
    stride := geometry.stride
    padding := geometry.padding
    kernelNonzero := geometry.kernelNonzero
    strideNonzero := geometry.strideNonzero }

/--
Apply an arbitrary-dimensional convolution to the channel and spatial suffix of a tensor.

The input suffix is `(inChannels, spatial...)`. Any axes in `leading` are preserved; internally
they are flattened into one runtime batch and restored after the convolution.
-/
def conv (leading : List Nat := []) {d inChannels : Nat} (spatial : Tensor Nat [d])
    (cfg : Conv d) (seedKernel seedBias : Nat := 0) [NeZero inChannels] :
    Sequential
      (leading ++ inChannels :: spatial.toList)
      (leading ++ cfg.outChannels ::
        (Spec.convOutSpatial spatial cfg.kernel cfg.stride cfg.padding).toList) := by
  simpa only [Spec.Shape.ofList_append] using
    (nn.of <| adaptLeadingShape (Spec.Shape.ofList leading) <|
      _root_.Runtime.Autograd.TorchLean.NN.conv
        (Spec.Shape.size (Spec.Shape.ofList leading)) d inChannels cfg.outChannels
        cfg.kernel cfg.stride cfg.padding spatial
        (hInC := NeZero.ne _) (hKernel := cfg.kernelNonzero) (hStride := cfg.strideNonzero)
        seedKernel seedBias cfg.kernelInit)

/-- Configuration shared by arbitrary-dimensional transpose-convolution layers. -/
structure ConvTranspose (d : Nat) where
  /-- Number of output channels. -/
  outChannels : Nat
  /-- Kernel extent along each spatial axis. -/
  kernel : Tensor Nat [d]
  /-- Step along each spatial axis. -/
  stride : Tensor Nat [d] := Spec.fill 1 [d]
  /-- Symmetric zero-padding along each spatial axis. -/
  padding : Tensor Nat [d] := Spec.fill 0 [d]
  /-- Every kernel extent is positive. -/
  kernelNonzero : ∀ i : Fin d, kernel.getScalar i ≠ 0
  /-- Every stride is positive. -/
  strideNonzero : ∀ i : Fin d, stride.getScalar i ≠ 0
  /-- Initialization scheme for the kernel weights. -/
  kernelInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .uniform (-0.1) 0.1

/--
Apply an arbitrary-dimensional transpose convolution to the channel and spatial suffix.

The input suffix is `(inChannels, spatial...)`. Any axes in `leading` are mapped independently and
restored after the operation.
-/
def convTranspose (leading : List Nat := []) {d inChannels : Nat}
    (spatial : Tensor Nat [d]) (cfg : ConvTranspose d)
    (seedKernel seedBias : Nat := 0) [NeZero inChannels] :
    Sequential
      (leading ++ inChannels :: spatial.toList)
      (leading ++ cfg.outChannels ::
        (Spec.convTransposeOutSpatial spatial cfg.kernel cfg.stride cfg.padding).toList) := by
  simpa only [Spec.Shape.ofList_append] using
    (nn.of <| adaptLeadingShape (Spec.Shape.ofList leading) <|
      _root_.Runtime.Autograd.TorchLean.NN.convTranspose
        (Spec.Shape.size (Spec.Shape.ofList leading)) d inChannels cfg.outChannels
        cfg.kernel cfg.stride cfg.padding spatial
        (hInC := NeZero.ne _) (hKernel := cfg.kernelNonzero)
        seedKernel seedBias cfg.kernelInit)
