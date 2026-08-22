/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.TypedGraphSession.ShapeIndex

/-!
# Typed Graph Session: Neural-Network Operations
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec
open Tensor

namespace Internal

namespace TypedGraphSession

/--
Record elementwise logistic sigmoid.

PyTorch comparison: `torch.sigmoid(x)`.
-/
def sigmoid {α : Type} (s : TypedGraphSession α) [Context α] [DecidableEq Shape] {sh : Shape}
  (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.sigmoid (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise hyperbolic tangent.

PyTorch comparison: `torch.tanh(x)`.
-/
def tanh {α : Type} (s : TypedGraphSession α) [Context α] [DecidableEq Shape] {sh : Shape}
  (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.tanh (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/-- Record softmax along an explicitly selected tensor dimension. -/
def softmax {α : Type} (s : TypedGraphSession α) [Context α] [DecidableEq Shape]
    {sh : Shape} (axis : Nat) [Shape.AxisInBounds axis sh]
    (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.softmax
        (α := α) (Γ := Γ) (s := sh) axis { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/-- Record stable log-softmax along an explicitly selected tensor dimension. -/
def logSoftmax {α : Type} (s : TypedGraphSession α) [Context α] [DecidableEq Shape]
    {sh : Shape} (axis : Nat) [Shape.AxisInBounds axis sh]
    (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.logSoftmax
        (α := α) (Γ := Γ) (s := sh) axis { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise softplus.

PyTorch comparison: `torch.nn.functional.softplus(x)`.
-/
def softplus {α : Type} (s : TypedGraphSession α) [Context α] [DecidableEq Shape] {sh : Shape}
  (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.softplus (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise exponential.

PyTorch comparison: `torch.exp(x)`.
-/
def exp {α : Type} (s : TypedGraphSession α) [Context α] [DecidableEq Shape] {sh : Shape}
  (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.exp (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise natural logarithm.

PyTorch comparison: `torch.log(x)`.
-/
def log {α : Type} (s : TypedGraphSession α) [Context α] [DecidableEq Shape] {sh : Shape}
  (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.log (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record elementwise log with epsilon guard.

This is intended for numerically stable losses; it corresponds approximately to `log(max(x, ε))`.
PyTorch comparison: `torch.log(torch.clamp(x, min=ε))`.
-/
def safeLog {α : Type} (s : TypedGraphSession α) [Context α] [DecidableEq Shape] {sh : Shape}
  (x : TensorRef α sh) (ε : α := Numbers.epsilon) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.safeLog (α := α) (Γ := Γ) (s := sh) { id := x.id } (ε :=
        ε))
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Sum-reduce all elements to a scalar.

PyTorch comparison: `x.sum()`.
-/
def sum {α : Type} (s : TypedGraphSession α) [Context α] [DecidableEq Shape] {sh : Shape}
  (x : TensorRef α sh) : IO (TensorRef α Shape.scalar) :=
  commitGraphM (α := α) s (β := TensorRef α Shape.scalar) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.sum (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Record a fully-connected linear layer: `y = w • x + b`.

Type-level shapes enforce `w : (outDim, inDim)`, `b : (outDim,)`, and `x : (inDim,)`.
PyTorch comparison: `torch.nn.functional.linear(x, weight=w, bias=b)` (with the same weight layout).
-/
def linear {α : Type} (s : TypedGraphSession α) [Add α] [Mul α] [Zero α] [DecidableEq Shape]
  {inDim outDim : Nat}
  (w : TensorRef α (.dim outDim (.dim inDim .scalar)))
  (b : TensorRef α (.dim outDim .scalar))
  (x : TensorRef α (.dim inDim .scalar)) : IO (TensorRef α (.dim outDim .scalar)) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim outDim .scalar)) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.linear (α := α) (Γ := Γ)
        (inDim := inDim) (outDim := outDim) { id := w.id } { id := b.id } { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Mean-squared-error loss returning a scalar.

PyTorch comparison: `torch.nn.functional.mse_loss(yhat, target, reduction="mean")`.
-/
def mseLoss {α : Type} (s : TypedGraphSession α)
  [Add α] [Sub α] [Mul α] [Div α] [Zero α] [One α] [Coe Nat α] [DecidableEq Shape]
  {sh : Shape} (yhat target : TensorRef α sh) : IO (TensorRef α Shape.scalar) :=
  commitGraphM (α := α) s (β := TensorRef α Shape.scalar) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.mseLoss (α := α) (Γ := Γ) (s := sh) { id := yhat.id } { id
        := target.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Layer normalization over the trailing embedding dimension.

This variant is specialized to 2D tensors of shape `(seqLen, embedDim)` and expects positive
dimensions for numerical stability and well-formedness.
PyTorch comparison: `torch.nn.LayerNorm(embedDim)` (applied per token), or
`torch.nn.functional.layer_norm`.
-/
def layerNorm {α : Type} (s : TypedGraphSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (x : TensorRef α (.dim seqLen (.dim embedDim .scalar)))
  (gamma : TensorRef α (.dim embedDim .scalar))
  (beta : TensorRef α (.dim embedDim .scalar)) : IO (TensorRef α (.dim seqLen (.dim embedDim
    .scalar))) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim seqLen (.dim embedDim .scalar))) (fun {Γ} {ss} xv
    nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.layerNorm (α := α) (Γ := Γ)
        (seqLen := seqLen) (embedDim := embedDim) (h_seq_pos := h_seq_pos) (h_embed_pos :=
          h_embed_pos)
        { id := x.id } { id := gamma.id } { id := beta.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Batch normalization for a channel-first image `(C,H,W)` (no batch axis).

`gamma` and `beta` are per-channel scale/shift parameters.
PyTorch comparison: `torch.nn.BatchNorm2d(C)` (conceptually), or `torch.nn.functional.batch_norm`
specialized to a single "batch element" with NCHW layout.
-/
def batchNormChannelFirst {α : Type} (s : TypedGraphSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {channels height width : Nat} (h_c : channels > 0) (h_h : height > 0) (h_w : width > 0)
  (x : TensorRef α (.dim channels (.dim height (.dim width .scalar))))
  (gamma : TensorRef α (.dim channels .scalar))
  (beta : TensorRef α (.dim channels .scalar)) :
  IO (TensorRef α (.dim channels (.dim height (.dim width .scalar)))) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim channels (.dim height (.dim width .scalar)))) (fun
    {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.batchNormChannelFirst (α := α) (Γ := Γ)
        (channels := channels) (height := height) (width := width) (h_c := h_c) (h_h := h_h) (h_w :=
          h_w)
        { id := x.id } { id := gamma.id } { id := beta.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))
end TypedGraphSession

end Internal

end Torch
end Autograd
end Runtime
