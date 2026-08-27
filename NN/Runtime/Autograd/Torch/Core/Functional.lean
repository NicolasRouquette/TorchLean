/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.TypedGraph
public import NN.Runtime.Autograd.LeadingAxis

/-!
# Execution-Mode-Generic Functional API

The `Ops` interface and curried helper syntax used to write one model once and run it on either the
eager execution path or the typed graph execution path.
-/


@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec
open Tensor
open Proofs.Autograd.Algebra

/-
Execution-mode-generic "one API" layer

Eager execution builds a runtime tape each iteration.
The `GraphM` authoring API provides a shape-indexed executable graph model.

The definitions below let you write a single model/loss once (as a polymorphic program over a
small `Ops` interface) and then choose:
- `execution := .eager`    (build a tape each iteration)
- `execution := .typedGraph` (construct shape-indexed typed graph data)
-/

end Torch
end Autograd
end Runtime

namespace Runtime
namespace Autograd
namespace Torch

open Spec
open Tensor
open Proofs.Autograd.Algebra

namespace Curried

/--
Type of a curried function accepting one tensor argument per shape in `ss`.

For example, `Fn α [s₁, s₂] β` is `Tensor α s₁ → Tensor α s₂ → β`.
-/
def Fn (α : Type) : List Shape → Type → Type
  | [], β => β
  | s :: ss, β => Tensor α s → Fn α ss β

/-- Convert a function on tensor-pack inputs into its curried form. -/
def curry {α : Type} {β : Type} : {ss : List Shape} →
    (_root_.TorchLean.TensorPack α ss → β) → Fn α ss β
  | [], f => f .nil
  | _s :: ss, f => fun x => curry (ss := ss) (fun xs => f (.cons x xs))

/-- Convert a curried function into a function on tensor-pack inputs. -/
def uncurry {α : Type} {β : Type} : {ss : List Shape} →
    Fn α ss β → _root_.TorchLean.TensorPack α ss → β
  | [], f, .nil => f
  | _s :: ss, f, .cons x xs => uncurry (ss := ss) (f x) xs

end Curried

/-!
`RefList` is the reference analogue of `TensorPack`: a dependent pack of `Ref s` values indexed by
a shape sequence.

This is used to write execution-mode-generic code over references (for example, `TensorRef`s in
eager mode or `GraphM.Var`s in typed graph mode).
-/
/-- Reference analogue of `TensorPack`: a dependent reference pack indexed by shapes. -/
inductive RefList (Ref : Shape → Type) : List Shape → Type where
  | nil : RefList Ref []
  | cons {s : Shape} {ss : List Shape} : Ref s → RefList Ref ss → RefList Ref (s :: ss)

namespace RefList

/-- Append two `RefList`s. -/
def append {Ref : Shape → Type} : {ss₁ ss₂ : List Shape} → RefList Ref ss₁ → RefList Ref ss₂ →
  RefList Ref (ss₁ ++ ss₂)
  | [], _ss₂, .nil, ys => ys
  | _s :: ss₁, ss₂, .cons x xs, ys => .cons x (append (ss₁ := ss₁) (ss₂ := ss₂) xs ys)

/-- Split a `RefList Ref (ss₁ ++ ss₂)` into its left and right parts. -/
def split {Ref : Shape → Type} : {ss₁ ss₂ : List Shape} →
    RefList Ref (ss₁ ++ ss₂) → RefList Ref ss₁ × RefList Ref ss₂
  | [], _ss₂, xs => (.nil, xs)
  | _s :: ss₁, ss₂, .cons x xs =>
      let (l, r) := split (Ref := Ref) (ss₁ := ss₁) (ss₂ := ss₂) xs
      (.cons x l, r)

/-- Split a `RefList Ref (ss ++ [τ])` into its prefix and last element. -/
def splitLast {Ref : Shape → Type} : {ss : List Shape} → {τ : Shape} →
    RefList Ref (ss ++ [τ]) → RefList Ref ss × Ref τ
  | [], _τ, .cons x .nil => (.nil, x)
  | _s :: ss, τ, .cons x xs =>
      let (l, last) := splitLast (Ref := Ref) (ss := ss) (τ := τ) xs
      (.cons x l, last)

end RefList

/--
Type of a curried function over references, one `Ref s` argument per shape in `ss`.

This mirrors `Curried.Fn`, but for `Ref`-valued arguments (e.g. `TensorRef`s in eager mode or
`GraphM.Var`s in typed graph mode).
-/
def CurriedRef (Ref : Shape → Type) : List Shape → Type → Type
  | [], β => β
  | s :: ss, β => Ref s → CurriedRef Ref ss β

namespace CurriedRef

/-- Uncurry a curried reference function to accept a `RefList`. -/
def uncurry {Ref : Shape → Type} {β : Type} : {ss : List Shape} → CurriedRef Ref ss β → RefList Ref
  ss → β
  | [], f, .nil => f
  | _s :: ss, f, .cons x xs => uncurry (ss := ss) (f x) xs

/-- Curry a reference function that consumes a `RefList`. -/
def curry {Ref : Shape → Type} {β : Type} : {ss : List Shape} → (RefList Ref ss → β) → CurriedRef
  Ref ss β
  | [], f => f .nil
  | _s :: ss, f => fun x => curry (ss := ss) (fun xs => f (.cons x xs))

/-- Apply a tensor-valued `CurriedRef` to its shape-indexed tensor pack. -/
def uncurryPack {α β : Type} : {ss : List Shape} →
    CurriedRef (Tensor α) ss β → _root_.TorchLean.TensorPack α ss → β
  | [], f, .nil => f
  | _s :: ss, f, .cons x xs => uncurryPack (ss := ss) (f x) xs

/--
Apply a curried reference function to a `GraphM.VarList`.

This is a convenience for the typed graph execution, where leaves/inputs are represented as `Var`s.
-/
def applyVarList {Γ : List Shape} {β : Type} :
    CurriedRef (fun s => Runtime.Autograd.TypedGraph.GraphM.Var s) Γ β →
      Runtime.Autograd.TypedGraph.GraphM.VarList Γ → β
  | f, .nil => f
  | f, .cons v vs => applyVarList (Γ := _) (β := β) (f v) vs

/--
Apply a curried function to the coordinate projections of a tensor pack.

Typed graph module inputs use this to represent each non-differentiable data tensor as a function
of the complete runtime input pack. The projection is typed by the shape list, so neither numeric
offsets nor runtime casts enter the graph definition.
-/
def applyPackProjections {α : Type} {β : Type} {full : List Shape} :
    {rest : List Shape} →
    (_root_.TorchLean.TensorPack α full → _root_.TorchLean.TensorPack α rest) →
    CurriedRef (fun s => _root_.TorchLean.TensorPack α full → Tensor α s) rest β → β
  | [], _drop, f => f
  | _s :: ss, drop, f =>
      let head : _root_.TorchLean.TensorPack α full → Tensor α _ := fun xs =>
        match drop xs with
        | .cons x _ => x
      let tail : _root_.TorchLean.TensorPack α full → _root_.TorchLean.TensorPack α ss := fun xs =>
        match drop xs with
        | .cons _ rest => rest
      applyPackProjections (rest := ss) tail (f head)

end CurriedRef

/--
Execution-mode-generic interface for building and executing tensor programs.

This typeclass lets you write a single model/loss once (polymorphic over `Ops m α`) and then choose:
- eager execution, which evaluates immediately on a runtime tape, or
- typed graph execution that records reusable graph data (`GraphM`) for later evaluation or
  connection to separate correctness theorems.

Each method corresponds to a Tensor op; implementations are expected to match the semantics of the
corresponding `Runtime.Autograd.Tape.*` / `TypedGraph.GraphM.*` operator.
-/
class Ops (m : Type → Type) (α : Type) [Context α] [DecidableEq Shape] where
  Ref : Shape → Type
  /-- Backend representation of non-differentiable tensor data. -/
  DataRef : Type → Shape → Type
  /-- Lift fixed non-differentiable data into the backend representation. -/
  dataConst : {β : Type} → {s : Shape} → Tensor β s → DataRef β s
  /-- Apply a pure transformation to non-differentiable data. -/
  mapData : {β γ : Type} → {s₁ s₂ : Shape} →
    (Tensor β s₁ → Tensor γ s₂) → DataRef β s₁ → DataRef γ s₂
  const : {s : Shape} → Tensor α s → m (Ref s)
  add : {s : Shape} → Ref s → Ref s → m (Ref s)
  sub : {s : Shape} → Ref s → Ref s → m (Ref s)
  mul : {s : Shape} → Ref s → Ref s → m (Ref s)
  scale : {s : Shape} → Ref s → α → m (Ref s)
  abs : {s : Shape} → Ref s → m (Ref s)
  sqrt : {s : Shape} → Ref s → m (Ref s)
  clamp : {s : Shape} → Ref s → α → α → m (Ref s)
  max : {s : Shape} → Ref s → Ref s → m (Ref s)
  min : {s : Shape} → Ref s → Ref s → m (Ref s)
  broadcastTo : {s₁ s₂ : Shape} → Shape.CanBroadcastTo s₁ s₂ → Ref s₁ → m (Ref s₂)
  reshape : {s₁ s₂ : Shape} → Ref s₁ → (h : Spec.Shape.size s₁ = Spec.Shape.size s₂) → m (Ref s₂)
  swapAdjacentAtDepth {s : Shape} : (depth : Nat) → Ref s → m (Ref (s.swapAdjacentAtDepth depth))
  reduceSum {s : Shape} (axis : Nat) [Shape.HasNonemptyAxis axis s] [Shape.WellFormed s] :
    Ref s → m (Ref (shapeAfterSum s axis))
  reduceMean {s : Shape} (axis : Nat) [Shape.HasNonemptyAxis axis s] [Shape.WellFormed s] :
    Ref s → m (Ref (shapeAfterSum s axis))
  select {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s] :
    Ref s → Fin (Shape.axisSize s axis) → m (Ref (s.eraseAxis axis))
  indexSelect {s : Shape} (axis count : Nat) [Shape.AxisInBounds axis s] :
    Ref s → DataRef (Fin (Shape.axisSize s axis)) [count] →
      m (Ref (s.replaceAxis axis count))
  scatterAdd {s : Shape} (axis count : Nat) [Shape.AxisInBounds axis s] :
    Ref s → Ref (s.replaceAxis axis count) →
      DataRef (Fin (Shape.axisSize s axis)) [count] → m (Ref s)
  matmul {batchA batchB batch : Shape} {mDim nDim pDim : Nat}
      [Shape.BroadcastTo batchA batch] [Shape.BroadcastTo batchB batch] :
    Ref (batchA.concat [mDim, nDim]) →
    Ref (batchB.concat [nDim, pDim]) →
    m (Ref (batch.concat [mDim, pDim]))
  concatLeadingAxis {nDim mDim : Nat} {s : Shape} :
    Ref (s.prependDim nDim) →
    Ref (s.prependDim mDim) →
    m (Ref (s.prependDim (nDim + mDim)))
  sliceLeadingAxisRange {nDim : Nat} {s : Shape} :
    (start len : Nat) → (h : start + len ≤ nDim) →
    Ref (s.prependDim nDim) → m (Ref (s.prependDim len))
  maxPool {d C : Nat}
    {inSpatial kernel stride padding : Tensor Nat [d]}
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0} :
    Ref (Shape.ofList (C :: inSpatial.toList)) →
    m (Ref (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList)))
  avgPool {d C : Nat}
    {inSpatial kernel stride padding : Tensor Nat [d]}
    (hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0) :
    Ref (Shape.ofList (C :: inSpatial.toList)) →
    m (Ref (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList)))
  smoothMaxPool {d C : Nat}
    {inSpatial kernel stride padding : Tensor Nat [d]}
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0} [DecidableEq α] :
    Ref (Shape.ofList (C :: inSpatial.toList)) →
    α →
    m (Ref (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList)))
  relu : {s : Shape} → Ref s → m (Ref s)
  sigmoid : {s : Shape} → Ref s → m (Ref s)
  tanh : {s : Shape} → Ref s → m (Ref s)
  gelu : {s : Shape} → Ref s → m (Ref s)
  /-- Backend primitive for softmax over the final tensor dimension. -/
  softmaxLast : {s : Shape} → Ref s → m (Ref s)
  /-- Backend primitive for stable log-softmax over the final tensor dimension. -/
  logSoftmaxLast : {s : Shape} → Ref s → m (Ref s)
  softplus : {s : Shape} → Ref s → m (Ref s)
  exp : {s : Shape} → Ref s → m (Ref s)
  log : {s : Shape} → Ref s → m (Ref s)
  inv : {s : Shape} → Ref s → m (Ref s)
  detach : {s : Shape} → Ref s → m (Ref s)
  safeLog : {s : Shape} → Ref s → α → m (Ref s)
  sum : {s : Shape} → Ref s → m (Ref Shape.scalar)
  flatten : {s : Shape} → Ref s → m (Ref [Spec.Shape.size s])
  linear {inDim outDim : Nat} :
    Ref [outDim, inDim] →
    Ref [outDim] →
    Ref [inDim] →
    m (Ref [outDim])
  mseLoss : {s : Shape} → Ref s → Ref s → m (Ref Shape.scalar)
  layerNorm {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0) :
    Ref [seqLen, embedDim] →
    Ref [embedDim] →
    Ref [embedDim] →
    m (Ref [seqLen, embedDim])
  batchNorm {channels : Nat} {sSpatial : Shape}
    (hWellFormed : (sSpatial.prependDim channels).wellFormed) :
    Ref (sSpatial.prependDim channels) →
    Ref [channels] →
    Ref [channels] →
    m (Ref (sSpatial.prependDim channels))
  multiHeadAttention {n numHeads dModel headDim : Nat} (h1 : n ≠ 0) :
    Ref [dModel, numHeads * headDim] →
    Ref [dModel, numHeads * headDim] →
    Ref [dModel, numHeads * headDim] →
    Ref [numHeads * headDim, dModel] →
    Ref [n, dModel] →
    Option (Tensor Bool [n, n]) →
    m (Ref [n, dModel])
  /--
  Multi-head self-attention with an explicit leading batch axis.

  Its mathematical meaning is the leading-axis map of `multiHeadAttention`; implementations may
  execute the samples together, but may not change the mask convention or the per-sample
  forward/VJP semantics.
  -/
  batchedMultiHeadAttention {batch n numHeads dModel headDim : Nat}
    (hBatch : batch ≠ 0) (h1 : n ≠ 0) :
    Ref [dModel, numHeads * headDim] →
    Ref [dModel, numHeads * headDim] →
    Ref [dModel, numHeads * headDim] →
    Ref [numHeads * headDim, dModel] →
    Ref [batch, n, dModel] →
    Option (Tensor Bool [n, n]) →
    m (Ref [batch, n, dModel])
  conv {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0} :
    Ref (Shape.ofList (outC :: inC :: kernel.toList)) →
    Ref [outC] →
    Ref (Shape.ofList (inC :: inSpatial.toList)) →
    m (Ref (Shape.ofList (outC :: (Spec.convOutSpatial inSpatial kernel stride padding).toList)))
  convTranspose {d inC outC : Nat}
    {kernel stride padding : Tensor Nat [d]}
    {inSpatial : Tensor Nat [d]}
    {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0} :
    Ref (Shape.ofList (inC :: outC :: kernel.toList)) →
    Ref [outC] →
    Ref (Shape.ofList (inC :: inSpatial.toList)) →
    m (Ref (Shape.ofList (outC ::
      (Spec.convTransposeOutSpatial inSpatial kernel stride padding).toList)))
  /-
  Seeded RNG primitives (first-class in TorchLean graphs).

  These are deterministic functions of:
  - the provided `seed` (user-controlled), and
  - backend-specific internal counters (e.g. node id / call index).

  They do not rely on `IO` randomness, so typed graphs remain replayable.
  -/
  randUniform : {s : Shape} → (seed : Nat) → m (Ref s)
  bernoulliMask : {s : Shape} → Ref Shape.scalar → (seed : Nat) → m (Ref s)

section

variable {m : Type → Type} {α : Type} [Context α] [DecidableEq Shape] [Monad m] [Ops (m := m) (α :=
  α)]

/--
Reference type for the current `Ops` instance.

In eager mode this will typically be `TensorRef`; in typed graph mode it will typically be
  `GraphM.Var`.
-/
abbrev Ref (s : Shape) : Type := Ops.Ref (m := m) (α := α) s

/-- Backend representation of a non-differentiable tensor. -/
abbrev DataRef (β : Type) (s : Shape) : Type := Ops.DataRef (m := m) (α := α) β s

/-- Lift fixed non-differentiable data into the current backend. -/
def dataConst {β : Type} {s : Shape} (x : Tensor β s) : DataRef (m := m) (α := α) β s :=
  Ops.dataConst (m := m) (α := α) x

/-- Transform non-differentiable data without adding an autograd node. -/
def mapData {β γ : Type} {s₁ s₂ : Shape} (f : Tensor β s₁ → Tensor γ s₂)
    (x : DataRef (m := m) (α := α) β s₁) : DataRef (m := m) (α := α) γ s₂ :=
  Ops.mapData (m := m) (α := α) f x

/-- Re-export of `Ops.const`. PyTorch: `torch.tensor(...)` / literal constants. -/
def const {s : Shape} (t : Tensor α s) : m (Ref (m := m) (α := α) s) := Ops.const (m := m) (α := α)
  t
/-- Re-export of `Ops.add`. PyTorch: `torch.add` / `+`. -/
def add {s : Shape} (a b : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.add (m :=
  m) (α := α) a b
/-- Re-export of `Ops.sub`. PyTorch: `torch.sub` / `-`. -/
def sub {s : Shape} (a b : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.sub (m :=
  m) (α := α) a b
/-- Re-export of `Ops.mul`. PyTorch: `torch.mul` / `*`. -/
def mul {s : Shape} (a b : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.mul (m :=
  m) (α := α) a b
/-- Re-export of `Ops.scale`. PyTorch: `x * c` for a scalar `c`. -/
def scale {s : Shape} (x : Ref (m := m) (α := α) s) (c : α) : m (Ref (m := m) (α := α) s) :=
  Ops.scale (m := m) (α := α) x c
/-- Re-export of `Ops.abs`. PyTorch: `torch.abs`. -/
def abs {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.abs (m := m)
  (α := α) x
/-- Re-export of `Ops.sqrt`. PyTorch: `torch.sqrt`. -/
def sqrt {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.sqrt (m :=
  m) (α := α) x
/-- Re-export of `Ops.clamp`. PyTorch: `torch.clamp`. -/
def clamp {s : Shape} (x : Ref (m := m) (α := α) s) (minVal maxVal : α) :
    m (Ref (m := m) (α := α) s) :=
  Ops.clamp (m := m) (α := α) (s := s) x minVal maxVal
/-- Re-export of `Ops.max`. PyTorch: `torch.maximum`. -/
def max {s : Shape} (a b : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.max (m :=
  m) (α := α) a b
/-- Re-export of `Ops.min`. PyTorch: `torch.minimum`. -/
def min {s : Shape} (a b : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.min (m :=
  m) (α := α) a b
/-- Re-export of `Ops.broadcastTo`. PyTorch: broadcasting / `expand`. -/
def broadcastTo {s₁ s₂ : Shape} (cb : Shape.CanBroadcastTo s₁ s₂)
  (x : Ref (m := m) (α := α) s₁) : m (Ref (m := m) (α := α) s₂) :=
  Ops.broadcastTo (m := m) (α := α) (s₁ := s₁) (s₂ := s₂) cb x
/-- Re-export of `Ops.reshape`. PyTorch: `reshape` / `view`. -/
def reshape {s₁ s₂ : Shape} (x : Ref (m := m) (α := α) s₁) (h : Spec.Shape.size s₁ = Spec.Shape.size s₂) :
  m (Ref (m := m) (α := α) s₂) :=
  Ops.reshape (m := m) (α := α) (s₁ := s₁) (s₂ := s₂) x h

/-- Restore an axis removed by a reduction and repeat the reduced value along that axis. -/
def broadcastAfterSum (s : Shape) (axis : Nat)
    (x : Ref (m := m) (α := α) (Spec.Tensor.shapeAfterSum s axis)) :
    m (Ref (m := m) (α := α) s) := do
  let kept ← reshape (m := m) (α := α)
    (s₁ := Spec.Tensor.shapeAfterSum s axis)
    (s₂ := Spec.Tensor.shapeAfterSumKeepDim s axis) x
    (Spec.Tensor.shape_after_sum_keep_dim_size s axis).symm
  broadcastTo (m := m) (α := α) (Spec.Tensor.shapeAfterSumKeepDimBroadcast s axis) kept
/-- Re-export of `Ops.swapAdjacentAtDepth` (general adjacent-axis swap). -/
def swapAdjacentAtDepth {s : Shape} (depth : Nat)
  (x : Ref (m := m) (α := α) s) :
  m (Ref (m := m) (α := α) (s.swapAdjacentAtDepth depth)) :=
  Ops.swapAdjacentAtDepth (m := m) (α := α) (s := s) depth x
/-- Re-export of `Ops.reduce_sum`. PyTorch: `torch.sum(..., dim=axis)`. -/
def reduceSum {s : Shape} (axis : Nat) [Shape.HasNonemptyAxis axis s] [Shape.WellFormed s]
  (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) (shapeAfterSum s axis)) :=
  Ops.reduceSum (m := m) (α := α) (s := s) axis x
/-- Re-export of `Ops.reduce_mean`. PyTorch: `torch.mean(..., dim=axis)`. -/
def reduceMean {s : Shape} (axis : Nat) (x : Ref (m := m) (α := α) s)
  [Shape.HasNonemptyAxis axis s] [Shape.WellFormed s] :
  m (Ref (m := m) (α := α) (shapeAfterSum s axis)) :=
  Ops.reduceMean (m := m) (α := α) (s := s) axis x
/-- Select one position along `axis`, removing that axis from the result. -/
def select {s : Shape} (axis : Nat) (x : Ref (m := m) (α := α) s)
    [Shape.AxisInBounds axis s] (index : Fin (Shape.axisSize s axis)) :
    m (Ref (m := m) (α := α) (s.eraseAxis axis)) :=
  Ops.select (m := m) (α := α) (s := s) axis x index

/-- Select positions along `axis`, replacing its extent by the number of indices. -/
def indexSelect {s : Shape} (axis count : Nat) (x : Ref (m := m) (α := α) s)
    [Shape.AxisInBounds axis s]
    (indices : DataRef (m := m) (α := α) (Fin (Shape.axisSize s axis)) [count]) :
    m (Ref (m := m) (α := α) (s.replaceAxis axis count)) :=
  Ops.indexSelect (m := m) (α := α) (s := s) axis count x indices

/-- Add source slices into a base tensor at positions along `axis`. -/
def scatterAdd {s : Shape} (axis count : Nat) (base : Ref (m := m) (α := α) s)
    [Shape.AxisInBounds axis s]
    (source : Ref (m := m) (α := α) (s.replaceAxis axis count))
    (indices : DataRef (m := m) (α := α) (Fin (Shape.axisSize s axis)) [count]) :
    m (Ref (m := m) (α := α) s) :=
  Ops.scatterAdd (m := m) (α := α) (s := s) axis count base source indices
/-- Multiply matrices after broadcasting their batch prefixes to a common shape. -/
def matmul {batchA batchB batch : Shape} {mDim nDim pDim : Nat}
    [Shape.BroadcastTo batchA batch] [Shape.BroadcastTo batchB batch]
    (a : Ref (m := m) (α := α) (batchA.concat [mDim, nDim]))
    (b : Ref (m := m) (α := α) (batchB.concat [nDim, pDim])) :
    m (Ref (m := m) (α := α) (batch.concat [mDim, pDim])) :=
  Ops.matmul (m := m) (α := α) (batchA := batchA) (batchB := batchB)
    (batch := batch) (mDim := mDim) (nDim := nDim) (pDim := pDim) a b
/-- Re-export of `Ops.concat_leading_axis`. PyTorch: `torch.cat(..., dim=0)`. -/
def concatLeadingAxis {nDim mDim : Nat} {s : Shape}
  (a : Ref (m := m) (α := α) (s.prependDim nDim))
  (b : Ref (m := m) (α := α) (s.prependDim mDim)) :
  m (Ref (m := m) (α := α) (s.prependDim (nDim + mDim))) :=
  Ops.concatLeadingAxis (m := m) (α := α) (nDim := nDim) (mDim := mDim) (s := s) a b
/-- Re-export of `Ops.slice_leading_axis_range`. PyTorch: `x[start:start+len]` on the leading dimension. -/
def sliceLeadingAxisRange {nDim : Nat} {s : Shape} (start len : Nat) (h : start + len ≤ nDim)
  (x : Ref (m := m) (α := α) (s.prependDim nDim)) :
  m (Ref (m := m) (α := α) (s.prependDim len)) :=
  Ops.sliceLeadingAxisRange (m := m) (α := α) (nDim := nDim) (s := s) start len h x

/--
Execution-polymorphic leading-axis map.

This is the reference implementation used by public model lifting and by batched primitives whose
backend does not provide a fused implementation.
-/
def mapOuterAxis {σ τ : Shape} (f : Ref (m := m) (α := α) σ →
    m (Ref (m := m) (α := α) τ)) {n : Nat}
    (x : Ref (m := m) (α := α) (σ.prependDim n)) :
    m (Ref (m := m) (α := α) (τ.prependDim n)) :=
  _root_.Runtime.Autograd.mapOuterAxisWith
    (const (m := m) (α := α) (s := τ.prependDim 0) (Spec.fill (0 : α) (τ.prependDim 0)))
    (fun x start len h => sliceLeadingAxisRange (m := m) (α := α) start len h x)
    (fun x h => reshape (m := m) (α := α) x h)
    (fun x y => concatLeadingAxis (m := m) (α := α) x y)
    f x
/--
Re-export of `Ops.max_pool` (generic N-D max pooling, channels-first; no batch axis).

PyTorch comparison: `torch.nn.functional.max_pool1d` / `max_pool2d` / `max_pool3d` depending on the
spatial rank `d`.
-/
def maxPool {d C : Nat}
  {inSpatial kernel stride padding : Tensor Nat [d]}
  {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
  (x : Ref (m := m) (α := α) (Shape.ofList (C :: inSpatial.toList))) :
  m (Ref (m := m) (α := α)
    (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) :=
  Ops.maxPool (m := m) (α := α)
    (d := d) (C := C)
    (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
    (hKernel := hKernel)
    x
/--
Re-export of `Ops.avg_pool` (generic N-D average pooling, channels-first; no batch axis).

PyTorch comparison: `torch.nn.functional.avg_pool1d` / `avg_pool2d` / `avg_pool3d` depending on the
spatial rank `d`.
-/
def avgPool {d C : Nat}
  {inSpatial kernel stride padding : Tensor Nat [d]}
  (hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0)
  (x : Ref (m := m) (α := α) (Shape.ofList (C :: inSpatial.toList))) :
  m (Ref (m := m) (α := α)
    (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) :=
  Ops.avgPool (m := m) (α := α)
    (d := d) (C := C)
    (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
    hKernel
    x
/--
Re-export of `Ops.smooth_max_pool` (generic N-D smooth max pooling, channels-first; no batch axis).

This is a differentiable approximation to max pooling; PyTorch does not expose it as a single
primitive.
-/
def smoothMaxPool {d C : Nat} [DecidableEq α]
  {inSpatial kernel stride padding : Tensor Nat [d]}
  {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
  (x : Ref (m := m) (α := α) (Shape.ofList (C :: inSpatial.toList)))
  (beta : α) :
  m (Ref (m := m) (α := α)
    (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) :=
  Ops.smoothMaxPool (m := m) (α := α)
    (d := d) (C := C)
    (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
    (hKernel := hKernel)
    x beta
/-- Re-export of `Ops.relu`. -/
def relu {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.relu (m :=
  m) (α := α) x
/-- Re-export of `Ops.sigmoid`. -/
def sigmoid {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.sigmoid
  (m := m) (α := α) x
/-- Re-export of `Ops.tanh`. -/
def tanh {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.tanh (m :=
  m) (α := α) x
/-- Apply the backend softmax primitive over the final tensor dimension. -/
def softmaxLast {s : Shape} (x : Ref (m := m) (α := α) s) :
    m (Ref (m := m) (α := α) s) :=
  Ops.softmaxLast (m := m) (α := α) x
/-- Re-export of `Ops.softplus`. -/
def softplus {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.softplus
  (m := m) (α := α) x
/-- Re-export of `Ops.exp`. -/
def exp {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.exp (m := m)
  (α := α) x
/-- Re-export of `Ops.log`. -/
def log {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.log (m := m)
  (α := α) x
/-- Re-export of `Ops.inv` (reciprocal). -/
def inv {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := Ops.inv (m := m)
  (α := α) x
/-- Re-export of `Ops.detach`. PyTorch: `x.detach()`. -/
def detach {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) :=
  Ops.detach (m := m) (α := α) x
/-- Re-export of `Ops.safe_log`. -/
def safeLog {s : Shape} (x : Ref (m := m) (α := α) s) (ε : α := Numbers.epsilon) :
  m (Ref (m := m) (α := α) s) :=
  Ops.safeLog (m := m) (α := α) (s := s) x ε
/-- Re-export of `Ops.rand_uniform` (deterministic seeded RNG). -/
def randUniform {s : Shape} (seed : Nat) : m (Ref (m := m) (α := α) s) :=
  Ops.randUniform (m := m) (α := α) (s := s) seed
/-- Re-export of `Ops.bernoulli_mask` (deterministic dropout-style mask). -/
def bernoulliMask {s : Shape} (keepProb : Ref (m := m) (α := α) Shape.scalar) (seed : Nat) :
    m (Ref (m := m) (α := α) s) :=
  Ops.bernoulliMask (m := m) (α := α) (s := s) keepProb seed

/--
Apply the stable backend log-softmax primitive over the final tensor dimension.

The primitive uses the max-shifted formulation
`x - max(x) - log(sum(exp(x - max(x))))`. Arbitrary-axis user operations permute their selected
dimension to this position before calling the primitive.
-/
def logSoftmaxLast {s : Shape} (x : Ref (m := m) (α := α) s) :
    m (Ref (m := m) (α := α) s) :=
  Ops.logSoftmaxLast (m := m) (α := α) (s := s) x

/-- SiLU / swish: `x * sigmoid(x)`. -/
def silu {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) := do
  let sx ← sigmoid (m := m) (α := α) (s := s) x
  mul (m := m) (α := α) (s := s) x sx

/--
Tanh-approximate GELU:

`0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))`.

GELU is a backend primitive so eager runtimes can execute it without materializing the formula as
a dozen temporary tensors. Every implementation remains responsible for the forward and VJP
semantics in `Activation.geluSpec` and `Activation.geluDerivSpec`.
-/
def gelu {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) s) :=
  Ops.gelu (m := m) (α := α) (s := s) x

/-- Re-export of `Ops.sum`. PyTorch: `x.sum()`. -/
def sum {s : Shape} (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) Shape.scalar) :=
  Ops.sum (m := m) (α := α) (s := s) x
/-- Re-export of `Ops.flatten`. PyTorch: `torch.flatten`. -/
def flatten {s : Shape} (x : Ref (m := m) (α := α) s) :
    m (Ref (m := m) (α := α) [Spec.Shape.size s]) :=
  Ops.flatten (m := m) (α := α) (s := s) x

/-- Re-export of `Ops.linear`. PyTorch: `torch.nn.functional.linear`. -/
def linear {inDim outDim : Nat}
  (w : Ref (m := m) (α := α) [outDim, inDim])
  (b : Ref (m := m) (α := α) [outDim])
  (x : Ref (m := m) (α := α) [inDim]) :
  m (Ref (m := m) (α := α) [outDim]) :=
  Ops.linear (m := m) (α := α) (inDim := inDim) (outDim := outDim) w b x

/-- Re-export of `Ops.mse_loss`. PyTorch: `torch.nn.functional.mse_loss`. -/
def mseLoss {s : Shape} (yhat target : Ref (m := m) (α := α) s) :
  m (Ref (m := m) (α := α) Shape.scalar) :=
  Ops.mseLoss (m := m) (α := α) (s := s) yhat target

/-- Re-export of `Ops.layer_norm`. PyTorch: `nn.LayerNorm` / `functional.layer_norm`. -/
def layerNorm {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (x : Ref (m := m) (α := α) [seqLen, embedDim])
  (gamma : Ref (m := m) (α := α) [embedDim])
  (beta : Ref (m := m) (α := α) [embedDim]) :
  m (Ref (m := m) (α := α) [seqLen, embedDim]) :=
  Ops.layerNorm (m := m) (α := α) (seqLen := seqLen) (embedDim := embedDim)
    h_seq_pos h_embed_pos x gamma beta

/-- Normalize each channel over every spatial axis of a channel-first tensor. -/
def batchNorm {channels : Nat} {sSpatial : Shape}
  (hWellFormed : (sSpatial.prependDim channels).wellFormed)
  (x : Ref (m := m) (α := α) (sSpatial.prependDim channels))
  (gamma : Ref (m := m) (α := α) [channels])
  (beta : Ref (m := m) (α := α) [channels]) :
  m (Ref (m := m) (α := α) (sSpatial.prependDim channels)) :=
  Ops.batchNorm (m := m) (α := α) (channels := channels) (sSpatial := sSpatial)
    hWellFormed x gamma beta

/-- Re-export of `Ops.multi_head_attention`. -/
def multiHeadAttention {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (wq : Ref (m := m) (α := α) [dModel, numHeads * headDim])
  (wk : Ref (m := m) (α := α) [dModel, numHeads * headDim])
  (wv : Ref (m := m) (α := α) [dModel, numHeads * headDim])
  (wo : Ref (m := m) (α := α) [numHeads * headDim, dModel])
  (x : Ref (m := m) (α := α) [n, dModel])
  (mask : Option (Tensor Bool [n, n]) := none) :
  m (Ref (m := m) (α := α) [n, dModel]) :=
  Ops.multiHeadAttention (m := m) (α := α) (n := n) (numHeads := numHeads) (dModel := dModel)
    (headDim := headDim) h1 wq wk wv wo x mask

/--
Batch-aware attention primitive.

Backends must implement the same leading-axis map as `multiHeadAttention`. The separate operation
lets an eager runtime fold `(batch, head)` into one batched contraction without weakening the
typed graph or specification semantics.
-/
def batchedMultiHeadAttention {batch n numHeads dModel headDim : Nat}
  (hBatch : batch ≠ 0) (h1 : n ≠ 0)
  (wq : Ref (m := m) (α := α) [dModel, numHeads * headDim])
  (wk : Ref (m := m) (α := α) [dModel, numHeads * headDim])
  (wv : Ref (m := m) (α := α) [dModel, numHeads * headDim])
  (wo : Ref (m := m) (α := α) [numHeads * headDim, dModel])
  (x : Ref (m := m) (α := α) [batch, n, dModel])
  (mask : Option (Tensor Bool [n, n]) := none) :
  m (Ref (m := m) (α := α) [batch, n, dModel]) :=
  Ops.batchedMultiHeadAttention (m := m) (α := α)
    (batch := batch) (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
    hBatch h1 wq wk wv wo x mask

/--
Re-export of `Ops.conv` (generic N-D convolution, channels-first).

PyTorch comparison: `torch.nn.functional.conv{d}d` specialized to a single sample (no batch axis).
-/
def conv {d inC outC : Nat}
  {kernel stride padding : Tensor Nat [d]}
  {inSpatial : Tensor Nat [d]}
  {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
  (weight : Ref (m := m) (α := α) (Shape.ofList (outC :: inC :: kernel.toList)))
  (bias : Ref (m := m) (α := α) [outC])
  (input : Ref (m := m) (α := α) (Shape.ofList (inC :: inSpatial.toList))) :
  m (Ref (m := m) (α := α)
    (Shape.ofList (outC :: (Spec.convOutSpatial inSpatial kernel stride padding).toList))) :=
  Ops.conv (m := m) (α := α)
    (d := d) (inC := inC) (outC := outC)
    (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
    (hInC := hInC) (hKernel := hKernel)
    weight bias input

/--
Re-export of `Ops.conv_transpose` (generic N-D transpose convolution, channels-first).

PyTorch comparison: `torch.nn.functional.conv_transpose{d}d` specialized to a single sample.
-/
def convTranspose {d inC outC : Nat}
  {kernel stride padding : Tensor Nat [d]}
  {inSpatial : Tensor Nat [d]}
  {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
  (weight : Ref (m := m) (α := α) (Shape.ofList (inC :: outC :: kernel.toList)))
  (bias : Ref (m := m) (α := α) [outC])
  (input : Ref (m := m) (α := α) (Shape.ofList (inC :: inSpatial.toList))) :
  m (Ref (m := m) (α := α)
    (Shape.ofList (outC ::
      (Spec.convTransposeOutSpatial inSpatial kernel stride padding).toList))) :=
  Ops.convTranspose (m := m) (α := α)
    (d := d) (inC := inC) (outC := outC)
    (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
    (hInC := hInC) (hKernel := hKernel)
    weight bias input

end
end Torch
end Autograd
end Runtime
