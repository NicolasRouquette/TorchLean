/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Ops.Dispatch

/-!
# Eager Tensor Operations

PyTorch-style tensor operations backed by the eager CPU/CUDA tapes. These wrappers record runtime
nodes, dispatch CUDA kernels when requested, and preserve the typed `TensorRef` surface.
-/


@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec
open Tensor
open Proofs.Autograd.Algebra

namespace Internal

namespace EagerSession

/-! ## Neural-network layers -/

/-- Fully-connected linear layer `y = w x + b`. PyTorch: `torch.nn.functional.linear`. -/
def linear {α : Type} (s : EagerSession α) [Inhabited α] [Add α] [Mul α] [Zero α] [DecidableEq
  Shape]
  {inDim outDim : Nat}
  (w : TensorRef α (.dim outDim (.dim inDim .scalar)))
  (b : TensorRef α (.dim outDim .scalar))
  (x : TensorRef α (.dim inDim .scalar)) : IO (TensorRef α (.dim outDim .scalar)) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.linear (t := t0)
      (inDim := inDim) (outDim := outDim) w.id b.id x.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.linear (t := t0) (outDim := outDim) (inDim := inDim) w.id b.id x.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .linear cpu cuda

/-- Mean-squared-error loss returning a scalar. PyTorch: `torch.nn.functional.mse_loss`. -/
def mseLoss {α : Type} [CudaBridge.TensorConv α] (s : EagerSession α)
  [Inhabited α] [Add α] [Sub α] [Mul α] [Div α] [Zero α] [One α] [Coe Nat α] [DecidableEq Shape]
  {sh : Shape} (yhat target : TensorRef α sh) : IO (TensorRef α Shape.scalar) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.mseLoss (t := t0) (s := sh) yhat.id target.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow <|
      Runtime.Autograd.Cuda.Tape.mseLoss (t := t0) (s := sh) yhat.id target.id
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .mseLoss cpu cuda

/-- Layer normalization over embedding dimension. PyTorch: `nn.LayerNorm` / `functional.layer_norm`.
  -/
def layerNorm {α : Type} (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (x : TensorRef α (.dim seqLen (.dim embedDim .scalar)))
  (gamma : TensorRef α (.dim embedDim .scalar))
  (beta : TensorRef α (.dim embedDim .scalar)) : IO (TensorRef α (.dim seqLen (.dim embedDim
    .scalar))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.layerNorm (t := t0)
      (seqLen := seqLen) (embedDim := embedDim) (h_seq_pos := h_seq_pos) (h_embed_pos := h_embed_pos)
      x.id gamma.id beta.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.layerNorm (t := t0)
      (seqLen := seqLen) (embedDim := embedDim) (h_seq_pos := h_seq_pos) (h_embed_pos := h_embed_pos)
      x.id gamma.id beta.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .layerNorm cpu cuda

/-- BatchNorm for channel-first images `(C,H,W)` (no batch axis). PyTorch: `nn.BatchNorm2d`
  (conceptually). -/
def batchNormChannelFirst {α : Type} (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {channels height width : Nat} (h_c : channels > 0) (h_h : height > 0) (h_w : width > 0)
  (x : TensorRef α (.dim channels (.dim height (.dim width .scalar))))
  (gamma : TensorRef α (.dim channels .scalar))
  (beta : TensorRef α (.dim channels .scalar)) : IO (TensorRef α (.dim channels (.dim height (.dim
    width .scalar)))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.batchNormChannelFirst (t := t0)
      (channels := channels) (height := height) (width := width) (h_c := h_c) (h_h := h_h) (h_w :=
        h_w)
      x.id gamma.id beta.id)
    s.tape.set t1
    pure { id := id }
  let cuda := do
    let t0 ← s.cudaTape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Cuda.Tape.batchNormChannelFirst (t := t0)
      (channels := channels) (height := height) (width := width) (h_c := h_c) (h_h := h_h)
      (h_w := h_w)
      x.id gamma.id beta.id)
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaOpt (α := α) s .batchNorm cpu cuda

/-- Multi-head self-attention (typed, proof-friendly). PyTorch: `nn.MultiheadAttention`
  (conceptually). -/
def multiHeadAttention {α : Type} (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (wq : TensorRef α (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wk : TensorRef α (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wv : TensorRef α (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wo : TensorRef α (.dim (numHeads * headDim) (.dim dModel .scalar)))
  (x : TensorRef α (.dim n (.dim dModel .scalar)))
  (mask : Option (Tensor Bool (.dim n (.dim n .scalar))) := none) :
  IO (TensorRef α (.dim n (.dim dModel .scalar))) := do
  let cpu := do
    let t0 ← s.tape.get
    let (t1, id) ← okOrThrow (Runtime.Autograd.Tape.multiHeadAttention (t := t0)
      (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim) (h1 := h1)
      wq.id wk.id wv.id wo.id x.id mask)
    s.tape.set t1
    pure { id := id }
  let cuda := fun attentionCapsule => do
    let t0 ← s.cudaTape.get
    let result ← Runtime.Autograd.Cuda.Tape.multiHeadAttention (t := t0)
      (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim) (h1 := h1)
      wq.id wk.id wv.id wo.id x.id (mask := mask) (attentionCapsule := attentionCapsule)
    let (t1, id) ← okOrThrow result
    s.cudaTape.set t1
    pure (some { id := id })
  dispatchCudaCapsuleOpt (α := α) s .scaledDotProductAttention
    [.nativeCuda, .torchLean, .libTorch] cpu cuda

end EagerSession

end Internal
end Torch
end Autograd
end Runtime
