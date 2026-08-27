/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.NN.Normalization

/-!
# TorchLean NN: Convolution and Pooling Layers
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor
open Proofs.Autograd.Algebra

namespace NN

/--
N-D convolution layer for a channels-first tensor `(inC, spatial...)` (no batch axis).

Parameters:
- kernel `K : (outC × inC × kernel[0] × ... × kernel[d-1])`,
- bias `b : (outC)`.

The output spatial shape is computed from `(stride, padding, kernel)`.

PyTorch analogy: `torch.nn.Conv{d}d` / `torch.nn.functional.conv{d}d` specialized to a single
sample (no batch axis), with `groups=1` and `dilation=1`.
-/
def conv
    (batch d inC outC : Nat)
    (kernel stride padding : Spec.Tensor Nat [d])
    (inSpatial : Spec.Tensor Nat [d])
    {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    {hStride : ∀ i : Fin d, stride.getScalar i ≠ 0}
    (seedK seedB : Nat := 0)
    (kInit : Torch.Init.Scheme := .uniform (-0.1) 0.1) :
    Layer (.dim batch (Shape.ofList (inC :: inSpatial.toList)))
      (.dim batch (Shape.ofList (outC :: (Spec.convOutSpatial inSpatial kernel stride padding).toList))) :=
  let KShape : Shape := Shape.ofList (outC :: inC :: kernel.toList)
  let bShape : Shape := .dim outC .scalar
  let k0 : Tensor Float KShape := Torch.Init.tensor (s := KShape) (sch := kInit) (seed := seedK)
  let b0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros) (seed := seedB)
  { kind := s!"Conv(rank={d}, in={inC}, out={outC})"
    stateShapes := [KShape, bShape]
    initState := .cons k0 (.cons b0 .nil)
    runtimeInit := some (.cons (TorchLean.Module.RuntimeInit.FloatInit.ofScheme kInit seedK)
      (.cons .zeros .nil))
    requiresGrad := #[true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun k b x =>
          TorchLean.conv (m := m) (α := α)
            (leadingShape := [batch]) (d := d) (inC := inC) (outC := outC)
            (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
            (hInC := hInC) (hKernel := hKernel) (_hStride := hStride)
            k b x
  }

/--
N-D transpose convolution layer for a channels-first tensor `(inC, spatial...)` (no batch axis).

Parameters:
- kernel `K : (inC × outC × kernel[0] × ... × kernel[d-1])` (PyTorch layout),
- bias `b : (outC)`.

The output spatial shape uses:
`out[a] = (in[a] - 1) * stride[a] - 2*padding[a] + kernel[a]` (with `output_padding = 0`).

PyTorch analogy: `torch.nn.ConvTranspose{d}d` / `torch.nn.functional.conv_transpose{d}d`
specialized to a single sample (no batch axis), with `groups=1`, `dilation=1`, `output_padding=0`.
-/
def convTranspose
    (batch d inC outC : Nat)
    (kernel stride padding : Spec.Tensor Nat [d])
    (inSpatial : Spec.Tensor Nat [d])
    {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    (seedK seedB : Nat := 0)
    (kInit : Torch.Init.Scheme := .uniform (-0.1) 0.1) :
    Layer (.dim batch (Shape.ofList (inC :: inSpatial.toList)))
      (.dim batch (Shape.ofList (outC :: (Spec.convTransposeOutSpatial inSpatial kernel stride padding).toList)))
      :=
  let KShape : Shape := Shape.ofList (inC :: outC :: kernel.toList)
  let bShape : Shape := .dim outC .scalar
  let k0 : Tensor Float KShape := Torch.Init.tensor (s := KShape) (sch := kInit) (seed := seedK)
  let b0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros) (seed := seedB)
  { kind := s!"ConvTranspose(rank={d}, in={inC}, out={outC})"
    stateShapes := [KShape, bShape]
    initState := .cons k0 (.cons b0 .nil)
    runtimeInit := some (.cons (TorchLean.Module.RuntimeInit.FloatInit.ofScheme kInit seedK)
      (.cons .zeros .nil))
    requiresGrad := #[true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun k b x =>
          TorchLean.convTranspose (m := m) (α := α)
            (leadingShape := [batch]) (d := d) (inC := inC) (outC := outC)
            (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
            (hInC := hInC) (hKernel := hKernel)
            k b x
  }

/--
N-D max pooling layer for a channels-first tensor `(batch, C, spatial...)` (no parameters).

Output spatial dims follow `Spec.pool_out_spatial_pad`.

PyTorch analogy: `torch.nn.functional.max_pool{d}d` on an `N×C×...` tensor.
-/
def maxPool
    (batch d C : Nat)
    (kernel stride padding : Spec.Tensor Nat [d])
    (inSpatial : Spec.Tensor Nat [d])
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    {hStride : ∀ i : Fin d, stride.getScalar i ≠ 0} :
    Layer (.dim batch (Shape.ofList (C :: inSpatial.toList)))
      (.dim batch (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList)))
      :=
  { kind := s!"MaxPool(rank={d})"
    stateShapes := []
    initState := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x =>
          TorchLean.maxPool (m := m) (α := α)
            (leadingShape := [batch]) (d := d) (channels := C)
            (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
            (hKernel := hKernel) (_hStride := hStride)
            x
  }

/--
N-D average pooling layer for a channels-first tensor `(batch, C, spatial...)` (no parameters).

PyTorch analogy: `torch.nn.functional.avg_pool{d}d` on an `N×C×...` tensor.
-/
def avgPool
    (batch d C : Nat)
    (kernel stride padding : Spec.Tensor Nat [d])
    (inSpatial : Spec.Tensor Nat [d])
    (hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0)
    (hStride : ∀ i : Fin d, stride.getScalar i ≠ 0) :
    Layer (.dim batch (Shape.ofList (C :: inSpatial.toList)))
      (.dim batch (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList)))
      :=
  { kind := s!"AvgPool(rank={d})"
    stateShapes := []
    initState := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x =>
          TorchLean.avgPool (m := m) (α := α)
            (leadingShape := [batch]) (d := d) (channels := C)
            (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
            (hKernel := hKernel) (_hStride := hStride)
            x
  }

end NN

end TorchLean
end Autograd
end Runtime
