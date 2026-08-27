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
Fully-connected (affine) layer on vectors: $y=w\mathbin{\cdot}x+b$.

PyTorch analogue: `torch.nn.functional.linear` (weight shape `(outDim, inDim)`).
-/
def linear {α : Type} (s : Session α) [Inhabited α] [Add α] [Mul α] [Zero α] [DecidableEq Shape]
  {inDim outDim : Nat}
  (w : _root_.Runtime.Autograd.Torch.TensorRef α [outDim, inDim])
  (b : _root_.Runtime.Autograd.Torch.TensorRef α [outDim])
  (x : _root_.Runtime.Autograd.Torch.TensorRef α [inDim]) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α [outDim]) := do
  match s.state with
  | .eager sess =>
      EagerSession.linear (α := α) sess (inDim := inDim) (outDim := outDim) w b x
  | .typedGraph sess =>
      _root_.Runtime.Autograd.Torch.Internal.TypedGraphSession.linear (α := α) sess
        (inDim := inDim) (outDim := outDim) w b x

/--
Mean squared error loss returning a scalar.

PyTorch analogue: `torch.nn.functional.mse_loss(..., reduction='mean')`.
-/
def mseLoss {α : Type} (s : Session α)
  [Inhabited α] [Add α] [Sub α] [Mul α] [Div α] [Zero α] [One α] [Coe Nat α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.TensorTransfer α]
  {sh : Shape}
  (yhat target : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) := do
  match s.state with
  | .eager sess => EagerSession.mseLoss (α := α) sess (sh := sh) yhat target
  | .typedGraph sess =>
      _root_.Runtime.Autograd.Torch.Internal.TypedGraphSession.mseLoss (α := α) sess (sh := sh) yhat target

/--
LayerNorm over a `seqLen × embedDim` tensor.

PyTorch analogue: `torch.nn.LayerNorm(embedDim)` applied per token.
-/
def layerNorm {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (x : _root_.Runtime.Autograd.Torch.TensorRef α [seqLen, embedDim])
  (gamma : _root_.Runtime.Autograd.Torch.TensorRef α [embedDim])
  (beta : _root_.Runtime.Autograd.Torch.TensorRef α [embedDim]) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α [seqLen, embedDim]) := do
  match s.state with
  | .eager sess =>
      EagerSession.layerNorm (α := α) sess
        (seqLen := seqLen) (embedDim := embedDim) (h_seq_pos := h_seq_pos) (h_embed_pos :=
          h_embed_pos)
        x gamma beta
  | .typedGraph sess =>
      _root_.Runtime.Autograd.Torch.Internal.TypedGraphSession.layerNorm (α := α) sess
        (seqLen := seqLen) (embedDim := embedDim) (h_seq_pos := h_seq_pos) (h_embed_pos :=
          h_embed_pos)
        x gamma beta

/-- Batch normalization over every spatial axis of a channel-first tensor. -/
def batchNorm {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
    {channels : Nat} {sSpatial : Shape}
    (hWellFormed : (Shape.dim channels sSpatial).wellFormed)
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim channels sSpatial))
  (gamma : _root_.Runtime.Autograd.Torch.TensorRef α [channels])
  (beta : _root_.Runtime.Autograd.Torch.TensorRef α [channels]) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim channels sSpatial)) := do
  match s.state with
  | .eager sess =>
      EagerSession.batchNorm (α := α) sess
        (channels := channels) (sSpatial := sSpatial) hWellFormed x gamma beta
  | .typedGraph sess =>
      _root_.Runtime.Autograd.Torch.Internal.TypedGraphSession.batchNorm (α := α) sess
        (channels := channels) (sSpatial := sSpatial) hWellFormed x gamma beta

/--
N-D convolution over a channels-first tensor `(inC, spatial...)`.

PyTorch analogue: `torch.nn.functional.conv{d}d` specialized to a single sample.
-/
def conv {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {d inC outC : Nat}
  {kernel stride padding : Spec.Tensor Nat [d]}
  {inSpatial : Spec.Tensor Nat [d]}
  (hInC : inC ≠ 0) (hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0)
  (w : _root_.Runtime.Autograd.Torch.TensorRef α (Shape.ofList (outC :: inC :: kernel.toList)))
  (b : _root_.Runtime.Autograd.Torch.TensorRef α [outC])
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (Shape.ofList (inC :: inSpatial.toList))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α
    (Shape.ofList (outC :: (Spec.convOutSpatial inSpatial kernel stride padding).toList))) := do
  match s.state with
  | .eager sess =>
      EagerSession.conv (α := α) sess
        (d := d) (inC := inC) (outC := outC)
        (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
        (hInC := hInC) (hKernel := hKernel)
        w b x
  | .typedGraph sess =>
      _root_.Runtime.Autograd.Torch.Internal.TypedGraphSession.conv (α := α) sess
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
  {kernel stride padding : Spec.Tensor Nat [d]}
  {inSpatial : Spec.Tensor Nat [d]}
  (hInC : inC ≠ 0) (hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0)
  (w : _root_.Runtime.Autograd.Torch.TensorRef α (Shape.ofList (inC :: outC :: kernel.toList)))
  (b : _root_.Runtime.Autograd.Torch.TensorRef α [outC])
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (Shape.ofList (inC :: inSpatial.toList))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α
    (Shape.ofList (outC :: (Spec.convTransposeOutSpatial inSpatial kernel stride padding).toList)))
    := do
  match s.state with
  | .eager sess =>
      EagerSession.convTranspose (α := α) sess
        (d := d) (inC := inC) (outC := outC)
        (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
        (hInC := hInC) (hKernel := hKernel)
        w b x
  | .typedGraph sess =>
      _root_.Runtime.Autograd.Torch.Internal.TypedGraphSession.convTranspose (α := α) sess
        (d := d) (inC := inC) (outC := outC)
        (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
        (hInC := hInC) (hKernel := hKernel)
        w b x

/--
Multi-head self-attention (single sequence, single batch).

This is a convenience op used by the transformer examples; it corresponds approximately to the forward
pass of `torch.nn.MultiheadAttention` in "self-attention" mode.
-/
def multiHeadAttention {α : Type} (s : Session α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (wq : _root_.Runtime.Autograd.Torch.TensorRef α [dModel, numHeads * headDim])
  (wk : _root_.Runtime.Autograd.Torch.TensorRef α [dModel, numHeads * headDim])
  (wv : _root_.Runtime.Autograd.Torch.TensorRef α [dModel, numHeads * headDim])
  (wo : _root_.Runtime.Autograd.Torch.TensorRef α [numHeads * headDim, dModel])
  (x : _root_.Runtime.Autograd.Torch.TensorRef α [n, dModel])
  (mask : Option (Tensor Bool [n, n]) := none) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α [n, dModel]) := do
  match s.state with
  | .eager sess =>
      EagerSession.multiHeadAttention (α := α) sess
        (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim) (h1 := h1)
        wq wk wv wo x (mask := mask)
  | .typedGraph sess =>
      _root_.Runtime.Autograd.Torch.Internal.TypedGraphSession.multiHeadAttention (α := α) sess
        (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim) (h1 := h1)
        wq wk wv wo x (mask := mask)


end Session

end TorchLean
end Autograd
end Runtime
