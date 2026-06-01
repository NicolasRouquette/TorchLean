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
    (kernel stride padding : Vector Nat d)
    (inSpatial : Vector Nat d)
    {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    (seedK seedB : Nat := 0)
    (kInit : Torch.Init.Scheme := .uniform (-0.1) 0.1) :
    LayerDef (.dim batch (Shape.ofList (inC :: inSpatial.toList)))
      (.dim batch (Shape.ofList (outC :: (Spec.convOutSpatial inSpatial kernel stride padding).toList))) :=
  let KShape : Shape := Shape.ofList (outC :: inC :: kernel.toList)
  let bShape : Shape := NN.Tensor.Shape.Vec outC
  let k0 : Tensor Float KShape := Torch.Init.tensor (s := KShape) (sch := kInit) (seed := seedK)
  let b0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros) (seed := seedB)
  { paramShapes := [KShape, bShape]
    initParams := Torch.tlist2 k0 b0
    paramRequiresGrad := [true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun k b x =>
          TorchLean.conv (m := m) (α := α)
            (batch := batch) (d := d) (inC := inC) (outC := outC)
            (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
            (hInC := hInC) (hKernel := hKernel)
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
    (kernel stride padding : Vector Nat d)
    (inSpatial : Vector Nat d)
    {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    (seedK seedB : Nat := 0)
    (kInit : Torch.Init.Scheme := .uniform (-0.1) 0.1) :
    LayerDef (.dim batch (Shape.ofList (inC :: inSpatial.toList)))
      (.dim batch (Shape.ofList (outC :: (Spec.convTransposeOutSpatial inSpatial kernel stride padding).toList)))
      :=
  let KShape : Shape := Shape.ofList (inC :: outC :: kernel.toList)
  let bShape : Shape := NN.Tensor.Shape.Vec outC
  let k0 : Tensor Float KShape := Torch.Init.tensor (s := KShape) (sch := kInit) (seed := seedK)
  let b0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros) (seed := seedB)
  { paramShapes := [KShape, bShape]
    initParams := Torch.tlist2 k0 b0
    paramRequiresGrad := [true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun k b x =>
          TorchLean.convTranspose (m := m) (α := α)
            (batch := batch) (d := d) (inC := inC) (outC := outC)
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
    (kernel stride padding : Vector Nat d)
    (inSpatial : Vector Nat d)
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0} :
    LayerDef (.dim batch (Shape.ofList (C :: inSpatial.toList)))
      (.dim batch (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList)))
      :=
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x =>
          TorchLean.maxPool (m := m) (α := α)
            (batch := batch) (d := d) (C := C)
            (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
            (hKernel := hKernel)
            x
  }

/--
N-D average pooling layer for a channels-first tensor `(batch, C, spatial...)` (no parameters).

PyTorch analogy: `torch.nn.functional.avg_pool{d}d` on an `N×C×...` tensor.
-/
def avgPool
    (batch d C : Nat)
    (kernel stride padding : Vector Nat d)
    (inSpatial : Vector Nat d)
    (hKernel : ∀ i : Fin d, kernel.get i ≠ 0) :
    LayerDef (.dim batch (Shape.ofList (C :: inSpatial.toList)))
      (.dim batch (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList)))
      :=
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x =>
          TorchLean.avgPool (m := m) (α := α)
            (batch := batch) (d := d) (C := C)
            (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
            (hKernel := hKernel)
            x
  }

/--
2D convolution layer for a `C×H×W` (channel-first) input.

Parameters:
- kernel `K : (outC × inC × kH × kW)` (OIHW layout),
- bias `b : (outC)`.

The output spatial shape is computed from `(stride, padding, kH, kW)`.

PyTorch analogy: `torch.nn.Conv2d(inC, outC, (kH, kW), stride=stride, padding=padding)`.
-/
def conv2d
    (inC outC kH kW stride padding inH inW : Nat)
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (seedK seedB : Nat := 0)
    (kInit : Torch.Init.Scheme := .uniform (-0.1) 0.1) :
    LayerDef (NN.Tensor.Shape.CHW inC inH inW)
      (NN.Tensor.Shape.CHW outC ((inH + 2 * padding - kH) / stride + 1) ((inW + 2 * padding - kW) /
        stride + 1)) :=
  let _outH : Nat := (inH + 2 * padding - kH) / stride + 1
  let _outW : Nat := (inW + 2 * padding - kW) / stride + 1
  let KShape : Shape := NN.Tensor.Shape.OIHW outC inC kH kW
  let bShape : Shape := NN.Tensor.Shape.Vec outC
  let k0 : Tensor Float KShape := Torch.Init.tensor (s := KShape) (sch := kInit) (seed := seedK)
  let b0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros) (seed := seedB)
  { paramShapes := [KShape, bShape]
    initParams := Torch.tlist2 k0 b0
    paramRequiresGrad := [true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun k b x =>
          TorchLean.conv2d (m := m) (α := α)
            (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding :=
              padding)
            (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 := h3)
            k b x
  }

/--
2D transpose convolution layer for a `C×H×W` (channel-first) input.

Parameters:
- kernel `K : (inC × outC × kH × kW)` (PyTorch layout),
- bias `b : (outC)`.

PyTorch analogy: `torch.nn.ConvTranspose2d(inC, outC, (kH, kW), stride=stride, padding=padding)`
(single-sample CHW specialization).
-/
def convTranspose2d
    (inC outC kH kW stride padding inH inW : Nat)
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
    (seedK seedB : Nat := 0)
    (kInit : Torch.Init.Scheme := .uniform (-0.1) 0.1) :
    LayerDef (NN.Tensor.Shape.CHW inC inH inW)
      (NN.Tensor.Shape.CHW outC
        ((inH - 1) * stride - 2 * padding + kH)
        ((inW - 1) * stride - 2 * padding + kW)) :=
  let KShape : Shape := .dim inC (.dim outC (.dim kH (.dim kW .scalar)))
  let bShape : Shape := NN.Tensor.Shape.Vec outC
  let k0 : Tensor Float KShape := Torch.Init.tensor (s := KShape) (sch := kInit) (seed := seedK)
  let b0 : Tensor Float bShape := Torch.Init.tensor (s := bShape) (sch := .zeros) (seed := seedB)
  { paramShapes := [KShape, bShape]
    initParams := Torch.tlist2 k0 b0
    paramRequiresGrad := [true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun k b x =>
          TorchLean.convTranspose2d (m := m) (α := α)
            (inC := inC) (outC := outC) (kH := kH) (kW := kW)
            (stride := stride) (padding := padding)
            (inH := inH) (inW := inW)
            (h1 := h1) (h2 := h2) (h3 := h3)
            k b x
  }

/--
2D max pooling on a `C×H×W` tensor.

PyTorch analogy: `torch.nn.functional.max_pool2d` (channel-first layout).
-/
def maxPool2d
    (kH kW inH inW inC stride : Nat)
    {h1 : kH ≠ 0} {h2 : kW ≠ 0} :
    LayerDef (NN.Tensor.Shape.CHW inC inH inW)
      (NN.Tensor.Shape.CHW inC ((inH - kH) / stride + 1) ((inW - kW) / stride + 1)) :=
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x =>
          TorchLean.maxPool2d (m := m) (α := α)
            (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
            (h1 := h1) (h2 := h2) x
  }

/--
2D max pooling with padding on a `C×H×W` tensor.

PyTorch analogy: `torch.nn.functional.max_pool2d(..., padding=padding)`.
-/
def maxPool2dPad
    (kH kW inH inW inC stride padding : Nat)
    {h1 : kH ≠ 0} {h2 : kW ≠ 0} :
    LayerDef (NN.Tensor.Shape.CHW inC inH inW)
      (NN.Tensor.Shape.CHW inC ((inH + 2 * padding - kH) / stride + 1) ((inW + 2 * padding - kW) /
        stride + 1)) :=
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x =>
          TorchLean.maxPool2dPad (m := m) (α := α)
            (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC)
            (stride := stride) (padding := padding) (h1 := h1) (h2 := h2) x
  }

/--
2D average pooling on a `C×H×W` tensor.

PyTorch analogy: `torch.nn.functional.avg_pool2d` (channel-first layout).
-/
def avgPool2d
    (kH kW inH inW inC stride : Nat)
    (h1 : kH ≠ 0) (h2 : kW ≠ 0) :
    LayerDef (NN.Tensor.Shape.CHW inC inH inW)
      (NN.Tensor.Shape.CHW inC ((inH - kH) / stride + 1) ((inW - kW) / stride + 1)) :=
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x =>
          TorchLean.avgPool2d (m := m) (α := α)
            (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
            h1 h2 x
  }

/--
2D average pooling with padding on a `C×H×W` tensor.

PyTorch analogy: `torch.nn.functional.avg_pool2d(..., padding=padding)`.
-/
def avgPool2dPad
    (kH kW inH inW inC stride padding : Nat)
    (h1 : kH ≠ 0) (h2 : kW ≠ 0) :
    LayerDef (NN.Tensor.Shape.CHW inC inH inW)
      (NN.Tensor.Shape.CHW inC ((inH + 2 * padding - kH) / stride + 1) ((inW + 2 * padding - kW) /
        stride + 1)) :=
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x =>
          TorchLean.avgPool2dPad (m := m) (α := α)
            (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC)
            (stride := stride) (padding := padding) h1 h2 x
  }

/-- Alias for `max_pool2d_pad` (PyTorch-style shorthand). -/
abbrev maxPoolPad := maxPool2dPad

/-- Alias for `avg_pool2d_pad` (PyTorch-style shorthand). -/
abbrev avgPoolPad := avgPool2dPad

/--
Global average pooling over spatial axes of a `C×H×W` tensor, producing a `C` vector.

PyTorch analogy: `torch.nn.functional.adaptive_avg_pool2d(x, output_size=1)` followed by flattening.
-/
def globalAvgPool2dChw
    (c h w : Nat)
    {h_c_pos : c > 0} {h_h_pos : h > 0} {h_w_pos : w > 0} :
    LayerDef (NN.Tensor.Shape.CHW c h w) (NN.Tensor.Shape.Vec c) :=
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x =>
          _root_.Runtime.Autograd.Torch.globalAvgPool2dChw (m := m) (α := α) (c := c) (h := h) (w
            := w)
            h_c_pos h_h_pos h_w_pos x
  }

/--
Global average pooling over spatial axes of an `N×C×H×W` tensor, producing an `N×C` tensor.

PyTorch analogy: `torch.nn.functional.adaptive_avg_pool2d(x, output_size=1)` then reshape to `(N,
  C)`.
-/
def globalAvgPool2dNchw
    (n c h w : Nat)
    {h_n_pos : n > 0} {h_c_pos : c > 0} {h_h_pos : h > 0} {h_w_pos : w > 0} :
    LayerDef (NN.Tensor.Shape.NCHW n c h w) (.dim n (.dim c .scalar)) :=
  { paramShapes := []
    initParams := .nil
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun x =>
          _root_.Runtime.Autograd.Torch.globalAvgPool2dNchw (m := m) (α := α) (n := n) (c := c)
            (h := h) (w := w)
            h_n_pos h_c_pos h_h_pos h_w_pos x
  }
end NN

end TorchLean
end Autograd
end Runtime
