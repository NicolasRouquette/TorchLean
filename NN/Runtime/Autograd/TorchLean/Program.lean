/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core
public import NN.Tensor.API

import Mathlib.Algebra.Order.Algebra

/-!
# Operation-Polymorphic Tensor Programs

`Program` is model code abstract over a tensor-operation interpreter. It is not stored graph data.
Supplying the eager interpreter executes the operations immediately and records a dynamic tape;
supplying the typed-graph interpreter records reusable, shape-indexed SSA data.

Users normally don’t import this directly; import `NN.Runtime.Autograd.TorchLean`,
`NN.API`, or `NN` instead.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor

export _root_.Runtime.Autograd.Torch (ExecutionMode Options TList Ops Ref RefList CurriedRef)

namespace Curried
export _root_.Runtime.Autograd.Torch.Curried (Fn curry uncurry)
end Curried

namespace CurriedRef
export _root_.Runtime.Autograd.Torch.CurriedRef (uncurry applyVarList)
end CurriedRef

namespace RefList
export _root_.Runtime.Autograd.Torch.RefList (append)
end RefList

/-! The execution-polymorphic surface shared by eager and typed graph execution. -/
export _root_.Runtime.Autograd.Torch
  (const add sub mul scale abs sqrt clamp max min
   broadcastTo reshape transpose2d swapAdjacentAtDepth
   reduceSum reduceMean
   gatherScalar gatherRow gatherScalarNatOrZero gatherVecNatOrZero gatherRowsNatOrZero scatterAddVec
     scatterAddRow
   mm bmm concatLeadingAxis sliceLeadingAxisRange
   maxPool avgPool smoothMaxPool
   maxPool2d maxPool2dPad smoothMaxPool2d avgPool2d avgPool2dPad
   relu silu gelu sigmoid tanh softmaxLast softplus exp log inv detach safeLog logSoftmaxLast
   sum flatten
   linear mseLoss layerNorm batchNormChannelFirst multiHeadAttention batchedMultiHeadAttention
   conv convTranspose conv2d convTranspose2d
   randUniform bernoulliMask)


/-! ## Operation-reference notation -/

/-- An operation reference to a tensor of shape `s`.

This is just `Runtime.Autograd.Torch.Ops.Ref`, but named so call sites can avoid repeating the
`Context`/`Ops` constraints.
-/
abbrev RefTy (m : Type → Type) (α : Type)
    [Context α] [DecidableEq Shape] [Ops (m := m) (α := α)]
    (s : Shape) : Type :=
  _root_.Runtime.Autograd.Torch.Ops.Ref (m := m) (α := α) s

namespace LeadingAxis
namespace Internal

/-! ## Batch-first derived ops -/

/-- Create a `0 × s` tensor (empty along the leading dimension). -/
def emptyLeadingAxis {α : Type} (s : Shape) : Tensor α (.dim 0 s) :=
  Tensor.dim (fun i : Fin 0 => Fin.elim0 i)

/--
Map a per-sample op over the leading batch dimension.

This adapts the shared leading-axis traversal to the TorchLean `Ops` interface. It is a convenience
for lifting single-sample operations, such as convolution, to batch-first tensors.
-/
def mapLeadingAxis {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch : Nat} {s t : Shape}
    (x : RefTy (m := m) (α := α) (.dim batch s))
    (f : RefTy (m := m) (α := α) s → m (RefTy (m := m) (α := α) t)) :
    m (RefTy (m := m) (α := α) (.dim batch t)) :=
  _root_.Runtime.Autograd.Torch.mapLeadingAxis (m := m) (α := α) f x

end Internal
end LeadingAxis

/-! ## Batch-first primitives (TorchLean user-facing) -/

/--
Batched N-D convolution (channels-first).

Input shape: `(N, inC, spatial...)`.
Output shape: `(N, outC, outSpatial...)`.
-/
def conv {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch d inC outC : Nat}
    {kernel stride padding : Vector Nat d}
    {inSpatial : Vector Nat d}
    {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {_hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (weight : RefTy (m := m) (α := α) (Shape.ofList (outC :: inC :: kernel.toList)))
    (bias : RefTy (m := m) (α := α) (.dim outC .scalar))
    (input : RefTy (m := m) (α := α) (.dim batch (Shape.ofList (inC :: inSpatial.toList)))) :
    m (RefTy (m := m) (α := α)
      (.dim batch
        (Shape.ofList (outC :: (Spec.convOutSpatial inSpatial kernel stride padding).toList)))) :=
  LeadingAxis.Internal.mapLeadingAxis (m := m) (α := α)
    (batch := batch) (s := Shape.ofList (inC :: inSpatial.toList))
    (t := Shape.ofList (outC :: (Spec.convOutSpatial inSpatial kernel stride padding).toList))
    input
    (fun x =>
      _root_.Runtime.Autograd.Torch.conv (m := m) (α := α)
        (d := d) (inC := inC) (outC := outC)
        (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
        (hInC := hInC) (hKernel := hKernel)
        weight bias x)

/--
Batched N-D transpose convolution (channels-first).

Input shape: `(N, inC, spatial...)`.
Output shape: `(N, outC, outSpatial...)`.
-/
def convTranspose {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch d inC outC : Nat}
    {kernel stride padding : Vector Nat d}
    {inSpatial : Vector Nat d}
    {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    (weight : RefTy (m := m) (α := α) (Shape.ofList (inC :: outC :: kernel.toList)))
    (bias : RefTy (m := m) (α := α) (.dim outC .scalar))
    (input : RefTy (m := m) (α := α) (.dim batch (Shape.ofList (inC :: inSpatial.toList)))) :
    m (RefTy (m := m) (α := α)
      (.dim batch
        (Shape.ofList (outC ::
          (Spec.convTransposeOutSpatial inSpatial kernel stride padding).toList)))) :=
  LeadingAxis.Internal.mapLeadingAxis (m := m) (α := α)
    (batch := batch) (s := Shape.ofList (inC :: inSpatial.toList))
    (t := Shape.ofList (outC :: (Spec.convTransposeOutSpatial inSpatial kernel stride padding).toList))
    input
    (fun x =>
      _root_.Runtime.Autograd.Torch.convTranspose (m := m) (α := α)
        (d := d) (inC := inC) (outC := outC)
        (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
        (hInC := hInC) (hKernel := hKernel)
        weight bias x)

/-- Batched max pool (channels-first). Input: `(N,C,spatial...)`. -/
def maxPool {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch d C : Nat}
    {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {_hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (input : RefTy (m := m) (α := α) (.dim batch (Shape.ofList (C :: inSpatial.toList)))) :
    m (RefTy (m := m) (α := α)
      (.dim batch (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))))
    :=
  LeadingAxis.Internal.mapLeadingAxis (m := m) (α := α)
    (batch := batch) (s := Shape.ofList (C :: inSpatial.toList))
    (t := Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))
    input
    (fun x =>
      _root_.Runtime.Autograd.Torch.maxPool (m := m) (α := α)
        (d := d) (C := C) (inSpatial := inSpatial) (kernel := kernel) (stride := stride)
        (padding := padding) (hKernel := hKernel) x)

/-- Batched average pool (channels-first). Input: `(N,C,spatial...)`. -/
def avgPool {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch d C : Nat}
    {inSpatial kernel stride padding : Vector Nat d}
    (hKernel : ∀ i : Fin d, kernel.get i ≠ 0)
    (_hStride : ∀ i : Fin d, stride.get i ≠ 0)
    (input : RefTy (m := m) (α := α) (.dim batch (Shape.ofList (C :: inSpatial.toList)))) :
    m (RefTy (m := m) (α := α)
      (.dim batch (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))))
    :=
  LeadingAxis.Internal.mapLeadingAxis (m := m) (α := α)
    (batch := batch) (s := Shape.ofList (C :: inSpatial.toList))
    (t := Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))
    input
    (fun x =>
      _root_.Runtime.Autograd.Torch.avgPool (m := m) (α := α)
        (d := d) (C := C) (inSpatial := inSpatial) (kernel := kernel) (stride := stride)
        (padding := padding) (hKernel := hKernel) x)

/-- Batched smooth max pool (channels-first). Input: `(N,C,spatial...)`. -/
def smoothMaxPool {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch d C : Nat}
    {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
    {_hStride : ∀ i : Fin d, stride.get i ≠ 0}
    (input : RefTy (m := m) (α := α) (.dim batch (Shape.ofList (C :: inSpatial.toList))))
    (temp : α) :
    m (RefTy (m := m) (α := α)
      (.dim batch (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))))
    :=
  LeadingAxis.Internal.mapLeadingAxis (m := m) (α := α)
    (batch := batch) (s := Shape.ofList (C :: inSpatial.toList))
    (t := Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))
    input
    (fun x =>
      _root_.Runtime.Autograd.Torch.smoothMaxPool (m := m) (α := α)
        (d := d) (C := C) (inSpatial := inSpatial) (kernel := kernel) (stride := stride)
        (padding := padding) (hKernel := hKernel) x temp)

/-- Layer normalization over the final axis of a matrix, including an empty row axis. -/
def layerNorm {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {rows width : Nat} (hWidth : width > 0)
    (x : RefTy (m := m) (α := α) (.dim rows (.dim width .scalar)))
    (gamma : RefTy (m := m) (α := α) (.dim width .scalar))
    (beta : RefTy (m := m) (α := α) (.dim width .scalar)) :
    m (RefTy (m := m) (α := α) (.dim rows (.dim width .scalar))) :=
  match rows with
  | 0 =>
      _root_.Runtime.Autograd.Torch.const (m := m) (α := α)
        (s := .dim 0 (.dim width .scalar))
        (LeadingAxis.Internal.emptyLeadingAxis (α := α) (.dim width .scalar))
  | rows + 1 =>
      _root_.Runtime.Autograd.Torch.layerNorm (m := m) (α := α)
        (seqLen := rows + 1) (embedDim := width) (Nat.succ_pos rows) hWidth x gamma beta

/--
Batched multi-head self-attention.

Input shape: `(batch, n, dModel)`.
Output shape: `(batch, n, dModel)`.
-/
def multiHeadAttention {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
    (wq : RefTy (m := m) (α := α) (.dim dModel (.dim (numHeads * headDim) .scalar)))
    (wk : RefTy (m := m) (α := α) (.dim dModel (.dim (numHeads * headDim) .scalar)))
    (wv : RefTy (m := m) (α := α) (.dim dModel (.dim (numHeads * headDim) .scalar)))
    (wo : RefTy (m := m) (α := α) (.dim (numHeads * headDim) (.dim dModel .scalar)))
    (x : RefTy (m := m) (α := α) (.dim batch (.dim n (.dim dModel .scalar))))
    (mask : Option (Tensor Bool (.dim n (.dim n .scalar))) := none) :
    m (RefTy (m := m) (α := α) (.dim batch (.dim n (.dim dModel .scalar)))) :=
  match batch with
  | 0 =>
      _root_.Runtime.Autograd.Torch.const (m := m) (α := α)
        (s := .dim 0 (.dim n (.dim dModel .scalar)))
        (LeadingAxis.Internal.emptyLeadingAxis (α := α) (.dim n (.dim dModel .scalar)))
  | batch + 1 =>
      _root_.Runtime.Autograd.Torch.batchedMultiHeadAttention (m := m) (α := α)
        (batch := batch + 1) (n := n) (numHeads := numHeads) (dModel := dModel)
        (headDim := headDim) (by simp) h1 wq wk wv wo x (mask := mask)

/-- Multi-head attention followed by a trainable bias on the output feature axis. -/
def multiHeadAttentionOutputBias {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
    (wq : RefTy (m := m) (α := α) (.dim dModel (.dim (numHeads * headDim) .scalar)))
    (wk : RefTy (m := m) (α := α) (.dim dModel (.dim (numHeads * headDim) .scalar)))
    (wv : RefTy (m := m) (α := α) (.dim dModel (.dim (numHeads * headDim) .scalar)))
    (wo : RefTy (m := m) (α := α) (.dim (numHeads * headDim) (.dim dModel .scalar)))
    (bo : RefTy (m := m) (α := α) (.dim dModel .scalar))
    (x : RefTy (m := m) (α := α) (.dim batch (.dim n (.dim dModel .scalar))))
    (mask : Option (Tensor Bool (.dim n (.dim n .scalar))) := none) :
    m (RefTy (m := m) (α := α) (.dim batch (.dim n (.dim dModel .scalar)))) := do
  let y ← multiHeadAttention (m := m) (α := α) h1 wq wk wv wo x mask
  let boFull ← _root_.Runtime.Autograd.Torch.broadcastTo (m := m) (α := α)
    (s₁ := .dim dModel .scalar) (s₂ := .dim batch (.dim n (.dim dModel .scalar)))
    Shape.BroadcastTo.proof bo
  _root_.Runtime.Autograd.Torch.add (m := m) (α := α)
    (s := .dim batch (.dim n (.dim dModel .scalar))) y boFull

/-- A TorchLean program is execution-polymorphic: it can run in any `m` that implements `Ops`.

This is a polymorphic function, not a stored graph. A graph is produced only after choosing a graph
interpreter and running the program with that interpreter.

In practice:
- `m := Runtime.Autograd.Session` gives you eager execution (and an autograd tape),
- `m := Runtime.Autograd.TypedGraph.GraphM.M` records shape-indexed SSA data that can be packaged
  as a reusable `Runtime.Autograd.Torch.TypedGraph`.
-/
abbrev Program (α : Type) [Context α] [DecidableEq Shape] (ss : List Shape) (τ : Shape) : Type 1 :=
  ∀ {m : Type → Type}, [Monad m] → [Ops (m := m) (α := α)] →
    CurriedRef (fun s => RefTy (m := m) (α := α) s) ss (m (RefTy (m := m) (α := α) τ))

/--
An execution-polymorphic program with differentiable tensor inputs followed by discrete tensor
inputs.

The two input lists make the scalar distinction part of the type. Parameters and continuous model
inputs use the backend scalar `α`; token ids, labels, and gather indices use `Nat`. This prevents a
runtime from accidentally treating discrete data as differentiable floating-point values.
-/
abbrev ProgramWithNatInputs (α : Type) [Context α] [DecidableEq Shape]
    (ss natSs : List Shape) (τ : Shape) : Type 1 :=
  ∀ {m : Type → Type}, [Monad m] → [Ops (m := m) (α := α)] →
    CurriedRef (fun s => RefTy (m := m) (α := α) s) ss
      (CurriedRef (fun s => _root_.Runtime.Autograd.Torch.NatTensorRef
        (m := m) (α := α) s) natSs (m (RefTy (m := m) (α := α) τ)))

end TorchLean
end Autograd
end Runtime
