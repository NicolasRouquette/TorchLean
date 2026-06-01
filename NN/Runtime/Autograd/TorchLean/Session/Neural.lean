/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.Session.ShapeIndex

/-!
# Session Neural-Network Operations

This file contains higher-level neural-network session calls such as linear layers, normalization,
attention, and convolutional blocks. The operations share the same session dispatch discipline as
the elementary ops while preserving PyTorch-style call sites.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor

namespace Session

/--
Fully-connected (affine) layer on vectors: `y = w·x + b`.

PyTorch analogue: `torch.nn.functional.linear` (weight shape `(outDim, inDim)`).
-/
def linear {α : Type} (s : Session α) [Inhabited α] [Add α] [Mul α] [Zero α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.FastKernels.FastMatmul α]
  {inDim outDim : Nat}
  (w : _root_.Runtime.Autograd.Torch.TensorRef α (.dim outDim (.dim inDim .scalar)))
  (b : _root_.Runtime.Autograd.Torch.TensorRef α (.dim outDim .scalar))
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim inDim .scalar)) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim outDim .scalar)) := do
  match s.impl with
  | .eager sess =>
      EagerSession.linear (α := α) sess (inDim := inDim) (outDim := outDim) w b x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.linear (α := α) sess
        (inDim := inDim) (outDim := outDim) w b x

/--
Mean squared error loss returning a scalar.

PyTorch analogue: `torch.nn.functional.mse_loss(..., reduction='mean')`.
-/
def mseLoss {α : Type} (s : Session α)
  [Inhabited α] [Add α] [Sub α] [Mul α] [Div α] [Zero α] [One α] [Coe Nat α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  {sh : Shape}
  (yhat target : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) := do
  match s.impl with
  | .eager sess => EagerSession.mseLoss (α := α) sess (sh := sh) yhat target
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.mseLoss (α := α) sess (sh := sh) yhat target

/--
LayerNorm over a `seqLen × embedDim` tensor.

PyTorch analogue: `torch.nn.LayerNorm(embedDim)` applied per token.
-/
def layerNorm {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim seqLen (.dim embedDim .scalar)))
  (gamma : _root_.Runtime.Autograd.Torch.TensorRef α (.dim embedDim .scalar))
  (beta : _root_.Runtime.Autograd.Torch.TensorRef α (.dim embedDim .scalar)) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim seqLen (.dim embedDim .scalar))) := do
  match s.impl with
  | .eager sess =>
      EagerSession.layerNorm (α := α) sess
        (seqLen := seqLen) (embedDim := embedDim) (h_seq_pos := h_seq_pos) (h_embed_pos :=
          h_embed_pos)
        x gamma beta
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.layerNorm (α := α) sess
        (seqLen := seqLen) (embedDim := embedDim) (h_seq_pos := h_seq_pos) (h_embed_pos :=
          h_embed_pos)
        x gamma beta

/--
BatchNorm over a CHW tensor (channel-first).

PyTorch analogue: `torch.nn.BatchNorm2d` (in channel-first layout).
-/
def batchnormChannelFirst {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {channels height width : Nat} (h_c : channels > 0) (h_h : height > 0) (h_w : width > 0)
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim channels (.dim height (.dim width .scalar))))
  (gamma : _root_.Runtime.Autograd.Torch.TensorRef α (.dim channels .scalar))
  (beta : _root_.Runtime.Autograd.Torch.TensorRef α (.dim channels .scalar)) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim channels (.dim height (.dim width .scalar))))
    := do
  match s.impl with
  | .eager sess =>
      EagerSession.batchnormChannelFirst (α := α) sess
        (channels := channels) (height := height) (width := width) (h_c := h_c) (h_h := h_h) (h_w :=
          h_w)
        x gamma beta
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.batchnormChannelFirst (α := α) sess
        (channels := channels) (height := height) (width := width) (h_c := h_c) (h_h := h_h) (h_w :=
          h_w)
        x gamma beta

/--
N-D convolution over a channels-first tensor `(inC, spatial...)`.

This is the generic counterpart to `conv2d`.

PyTorch analogue: `torch.nn.functional.conv{d}d` specialized to a single sample.
-/
def conv {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  (hInC : inC ≠ 0) (hKernel : ∀ i : Fin d, kernel.get i ≠ 0)
  (w : _root_.Runtime.Autograd.Torch.TensorRef α (Shape.ofList (outC :: inC :: kernel.toList)))
  (b : _root_.Runtime.Autograd.Torch.TensorRef α (.dim outC .scalar))
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (Shape.ofList (inC :: inSpatial.toList))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α
    (Shape.ofList (outC :: (Spec.convOutSpatial inSpatial kernel stride padding).toList))) := do
  match s.impl with
  | .eager sess =>
      EagerSession.conv (α := α) sess
        (d := d) (inC := inC) (outC := outC)
        (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
        (hInC := hInC) (hKernel := hKernel)
        w b x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.conv (α := α) sess
        (d := d) (inC := inC) (outC := outC)
        (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
        (hInC := hInC) (hKernel := hKernel)
        w b x

/--
N-D transpose convolution over a channels-first tensor `(inC, spatial...)`.

PyTorch analogue: `torch.nn.functional.conv_transpose{d}d` specialized to a single sample.
-/
def convTranspose {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  (hInC : inC ≠ 0) (hKernel : ∀ i : Fin d, kernel.get i ≠ 0)
  (w : _root_.Runtime.Autograd.Torch.TensorRef α (Shape.ofList (inC :: outC :: kernel.toList)))
  (b : _root_.Runtime.Autograd.Torch.TensorRef α (.dim outC .scalar))
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (Shape.ofList (inC :: inSpatial.toList))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α
    (Shape.ofList (outC :: (Spec.convTransposeOutSpatial inSpatial kernel stride padding).toList)))
    := do
  match s.impl with
  | .eager sess =>
      EagerSession.convTranspose (α := α) sess
        (d := d) (inC := inC) (outC := outC)
        (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
        (hInC := hInC) (hKernel := hKernel)
        w b x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.convTranspose (α := α) sess
        (d := d) (inC := inC) (outC := outC)
        (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
        (hInC := hInC) (hKernel := hKernel)
        w b x

/--
2D convolution over a CHW tensor.

PyTorch analogue: `torch.nn.functional.conv2d` (channel-first layout).
-/
def conv2d {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {inC outC kH kW stride padding inH inW : Nat}
  (h1 : inC ≠ 0) (h2 : kH ≠ 0) (h3 : kW ≠ 0)
  (kernel : _root_.Runtime.Autograd.Torch.TensorRef α (.dim outC (.dim inC (.dim kH (.dim kW
    .scalar)))))
  (bias : _root_.Runtime.Autograd.Torch.TensorRef α (.dim outC .scalar))
  (input : _root_.Runtime.Autograd.Torch.TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α
    (.dim outC (.dim ((inH + 2 * padding - kH) / stride + 1)
      (.dim ((inW + 2 * padding - kW) / stride + 1) .scalar)))) := do
  match s.impl with
  | .eager sess =>
      EagerSession.conv2d (α := α) sess
        (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
        (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 := h3)
        kernel bias input
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.conv2d (α := α) sess
        (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
        (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 := h3)
        kernel bias input

/--
2D transpose convolution over a CHW tensor.

PyTorch analogue: `torch.nn.functional.conv_transpose2d` (channel-first layout).
-/
def convTranspose2d {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {inC outC kH kW stride padding inH inW : Nat}
  (h1 : inC ≠ 0) (h2 : kH ≠ 0) (h3 : kW ≠ 0)
  (kernel : _root_.Runtime.Autograd.Torch.TensorRef α (.dim inC (.dim outC (.dim kH (.dim kW
    .scalar)))))
  (bias : _root_.Runtime.Autograd.Torch.TensorRef α (.dim outC .scalar))
  (input : _root_.Runtime.Autograd.Torch.TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α
    (.dim outC (.dim ((inH - 1) * stride - 2 * padding + kH)
      (.dim ((inW - 1) * stride - 2 * padding + kW) .scalar)))) := do
  match s.impl with
  | .eager sess =>
      EagerSession.convTranspose2d (α := α) sess
        (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
        (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 := h3)
        kernel bias input
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.convTranspose2d (α := α) sess
        (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
        (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 := h3)
        kernel bias input

/--
Multi-head self-attention (single sequence, single batch).

This is a convenience op used by the transformer examples; it corresponds approximately to the forward
pass of `torch.nn.MultiheadAttention` in "self-attention" mode.
-/
def multiHeadAttention {α : Type} (s : Session α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (wq : _root_.Runtime.Autograd.Torch.TensorRef α (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wk : _root_.Runtime.Autograd.Torch.TensorRef α (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wv : _root_.Runtime.Autograd.Torch.TensorRef α (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wo : _root_.Runtime.Autograd.Torch.TensorRef α (.dim (numHeads * headDim) (.dim dModel .scalar)))
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n (.dim dModel .scalar)))
  (mask : Option (Tensor Bool (.dim n (.dim n .scalar))) := none) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim n (.dim dModel .scalar))) := do
  match s.impl with
  | .eager sess =>
      EagerSession.multiHeadAttention (α := α) sess
        (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim) (h1 := h1)
        wq wk wv wo x (mask := mask)
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.multiHeadAttention (α := α) sess
        (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim) (h1 := h1)
        wq wk wv wo x (mask := mask)


end Session

end TorchLean
end Autograd
end Runtime
