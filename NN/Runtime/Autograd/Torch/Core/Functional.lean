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

namespace Proofs.Autograd.Algebra.TList

/--
Append two `TList`s.

This is a small utility for bridging between curried APIs and list-of-shapes APIs.
-/
def append {α : Type} : {ss₁ ss₂ : List Spec.Shape} →
    TList α ss₁ → TList α ss₂ → TList α (ss₁ ++ ss₂)
  | [], _ss₂, .nil, ys => ys
  | _s :: ss₁, ss₂, .cons x xs, ys => .cons x (append (ss₁ := ss₁) (ss₂ := ss₂) xs ys)

/--
Split a `TList α (ss₁ ++ ss₂)` into its left and right parts.

This is the inverse of `TList.append`.
-/
def splitAppend {α : Type} : {ss₁ ss₂ : List Spec.Shape} →
    TList α (ss₁ ++ ss₂) → TList α ss₁ × TList α
  ss₂
  | [], _ss₂, xs => (.nil, xs)
  | _s :: ss₁, ss₂, .cons x xs =>
      let (xs₁, xs₂) := splitAppend (α := α) (ss₁ := ss₁) (ss₂ := ss₂) xs
      (.cons x xs₁, xs₂)

/-- Splitting a concatenated typed list recovers its two inputs. -/
@[simp] theorem splitAppend_append {α : Type} {ss₁ ss₂ : List Spec.Shape}
    (xs : TList α ss₁) (ys : TList α ss₂) :
    splitAppend (append xs ys) = (xs, ys) := by
  induction ss₁ with
  | nil => cases xs; rfl
  | cons shape rest ih =>
      cases xs with
      | cons x xs =>
          simp only [append, splitAppend]
          rw [ih xs]

end Proofs.Autograd.Algebra.TList

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

/-- Convert a function on `TList` inputs into its curried form. -/
def curry {α : Type} {β : Type} : {ss : List Shape} → (TList α ss → β) → Fn α ss β
  | [], f => f .nil
  | _s :: ss, f => fun x => curry (ss := ss) (fun xs => f (.cons x xs))

/-- Convert a curried function into a function on `TList` inputs. -/
def uncurry {α : Type} {β : Type} : {ss : List Shape} → Fn α ss β → TList α ss → β
  | [], f, .nil => f
  | _s :: ss, f, .cons x xs => uncurry (ss := ss) (f x) xs

end Curried

/-!
`RefList` is the reference-analogue of `TList`: a heterogeneous list of `Ref s` values indexed by
a shape list.

This is used to write execution-mode-generic code over references (for example, `TensorRef`s in
eager mode or `GraphM.Var`s in typed graph mode).
-/
/-- Reference-analogue of `TList`: a heterogeneous list of `Ref s` values indexed by shapes. -/
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

/-- Apply a tensor-valued `CurriedRef` to its shape-indexed tensor list. -/
def uncurryTList {α β : Type} : {ss : List Shape} →
    CurriedRef (Tensor α) ss β → TList α ss → β
  | [], f, .nil => f
  | _s :: ss, f, .cons x xs => uncurryTList (ss := ss) (f x) xs

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
Apply a curried function to the coordinate projections of a typed tensor list.

Typed graph module inputs use this to represent each non-differentiable natural-number tensor as a
function of the complete runtime input pack. The projection is typed by the shape list, so neither
numeric offsets nor runtime casts enter the graph definition.
-/
def applyTListProjections {α : Type} {β : Type} {full : List Shape} :
    {rest : List Shape} →
    (TList α full → TList α rest) →
    CurriedRef (fun s => TList α full → Tensor α s) rest β → β
  | [], _drop, f => f
  | _s :: ss, drop, f =>
      let head : TList α full → Tensor α _ := fun xs =>
        match drop xs with
        | .cons x _ => x
      let tail : TList α full → TList α ss := fun xs =>
        match drop xs with
        | .cons _ rest => rest
      applyTListProjections (rest := ss) tail (f head)

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
  /-- Reference to non-differentiable natural-number data, such as labels or gather indices. -/
  NatTensorRef : Shape → Type
  /-- Lift a fixed natural-number tensor into the backend's discrete-data representation. -/
  natTensorConst : {s : Shape} → Tensor Nat s → NatTensorRef s
  /-- Apply a pure transformation to non-differentiable natural-number data. -/
  mapNatTensor : {s₁ s₂ : Shape} → (Tensor Nat s₁ → Tensor Nat s₂) → NatTensorRef s₁ →
    NatTensorRef s₂
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
  transpose2d {mDim nDim : Nat} : Ref (.dim mDim (.dim nDim .scalar)) → m (Ref (.dim nDim (.dim mDim
    .scalar)))
  transpose3dFirstToLast {a b c : Nat} :
    Ref (.dim a (.dim b (.dim c .scalar))) → m (Ref (.dim b (.dim c (.dim a .scalar))))
  transpose3dLastToFirst {a b c : Nat} :
    Ref (.dim a (.dim b (.dim c .scalar))) → m (Ref (.dim c (.dim a (.dim b .scalar))))
  transpose3dLastTwo {a b c : Nat} :
    Ref (.dim a (.dim b (.dim c .scalar))) → m (Ref (.dim a (.dim c (.dim b .scalar))))
  swapAdjacentAtDepth {s : Shape} : (depth : Nat) → Ref s → m (Ref (s.swapAdjacentAtDepth depth))
  reduceSum {s : Shape} (axis : Nat) [Shape.HasNonemptyAxis axis s] [Shape.WellFormed s] :
    Ref s → m (Ref (shapeAfterSum s axis))
  reduceMean {s : Shape} (axis : Nat) [Shape.HasNonemptyAxis axis s] [Shape.WellFormed s] :
    Ref s → m (Ref (shapeAfterSum s axis))
  gatherScalar {n : Nat} : Ref (.dim n .scalar) → Fin n → m (Ref Shape.scalar)
  gatherRow {rows cols : Nat} : Ref (.dim rows (.dim cols .scalar)) → Fin rows → m (Ref (.dim cols
    .scalar))
  gatherScalarNatOrZero {n : Nat} : Ref (.dim n .scalar) → Nat → m (Ref Shape.scalar)
  gatherVecNatOrZero {n k : Nat} : Ref (.dim n .scalar) → NatTensorRef (.dim k .scalar) →
    m (Ref (.dim k .scalar))
  gatherRowsNatOrZero {rows cols k : Nat} :
    Ref (.dim rows (.dim cols .scalar)) → NatTensorRef (.dim k .scalar) →
      m (Ref (.dim k (.dim cols .scalar)))
  scatterAddVec {n : Nat} : Ref (.dim n .scalar) → Ref Shape.scalar → Fin n → m (Ref (.dim n
    .scalar))
  scatterAddRow {rows cols : Nat} :
    Ref (.dim rows (.dim cols .scalar)) → Ref (.dim cols .scalar) → Fin rows → m (Ref (.dim rows
      (.dim cols .scalar)))
  matmul {mDim nDim pDim : Nat} :
    Ref (.dim mDim (.dim nDim .scalar)) →
    Ref (.dim nDim (.dim pDim .scalar)) →
    m (Ref (.dim mDim (.dim pDim .scalar)))
  bmm {batch mDim nDim pDim : Nat} :
    Ref (.dim batch (.dim mDim (.dim nDim .scalar))) →
    Ref (.dim batch (.dim nDim (.dim pDim .scalar))) →
    m (Ref (.dim batch (.dim mDim (.dim pDim .scalar))))
  concatLeadingAxis {nDim mDim : Nat} {s : Shape} :
    Ref (.dim nDim s) →
    Ref (.dim mDim s) →
    m (Ref (.dim (nDim + mDim) s))
  sliceLeadingAxisRange {nDim : Nat} {s : Shape} :
    (start len : Nat) → (h : start + len ≤ nDim) →
    Ref (.dim nDim s) → m (Ref (.dim len s))
  maxPool {d C : Nat}
    {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0} :
    Ref (Shape.ofList (C :: inSpatial.toList)) →
    m (Ref (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList)))
  avgPool {d C : Nat}
    {inSpatial kernel stride padding : Vector Nat d}
    (hKernel : ∀ i : Fin d, kernel.get i ≠ 0) :
    Ref (Shape.ofList (C :: inSpatial.toList)) →
    m (Ref (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList)))
  smoothMaxPool {d C : Nat}
    {inSpatial kernel stride padding : Vector Nat d}
    {hKernel : ∀ i : Fin d, kernel.get i ≠ 0} :
    Ref (Shape.ofList (C :: inSpatial.toList)) →
    α →
    m (Ref (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList)))
  maxPool2d {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0} :
    Ref (.dim inC (.dim inH (.dim inW .scalar))) →
    m (Ref (.dim inC (.dim (Spec.poolOutDim inH kH stride 0) (.dim (Spec.poolOutDim inW kW stride 0) .scalar))))
  maxPool2dPad {kH kW inH inW inC stride padding : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0} :
    Ref (.dim inC (.dim inH (.dim inW .scalar))) →
    m (Ref (.dim inC (.dim (Spec.poolOutDim inH kH stride padding)
      (.dim (Spec.poolOutDim inW kW stride padding) .scalar))))
  smoothMaxPool2d {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0} :
    Ref (.dim inC (.dim inH (.dim inW .scalar))) →
    α →
    m (Ref (.dim inC (.dim (Spec.poolOutDim inH kH stride 0) (.dim (Spec.poolOutDim inW kW stride 0) .scalar))))
  avgPool2d {kH kW inH inW inC stride : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0) :
    Ref (.dim inC (.dim inH (.dim inW .scalar))) →
    m (Ref (.dim inC (.dim (Spec.poolOutDim inH kH stride 0) (.dim (Spec.poolOutDim inW kW stride 0) .scalar))))
  avgPool2dPad {kH kW inH inW inC stride padding : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0) :
    Ref (.dim inC (.dim inH (.dim inW .scalar))) →
    m (Ref (.dim inC (.dim (Spec.poolOutDim inH kH stride padding)
      (.dim (Spec.poolOutDim inW kW stride padding) .scalar))))
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
  flatten : {s : Shape} → Ref s → m (Ref (.dim (Spec.Shape.size s) .scalar))
  linear {inDim outDim : Nat} :
    Ref (.dim outDim (.dim inDim .scalar)) →
    Ref (.dim outDim .scalar) →
    Ref (.dim inDim .scalar) →
    m (Ref (.dim outDim .scalar))
  mseLoss : {s : Shape} → Ref s → Ref s → m (Ref Shape.scalar)
  layerNorm {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0) :
    Ref (.dim seqLen (.dim embedDim .scalar)) →
    Ref (.dim embedDim .scalar) →
    Ref (.dim embedDim .scalar) →
    m (Ref (.dim seqLen (.dim embedDim .scalar)))
  batchNormChannelFirst {channels height width : Nat} (h_c : channels > 0) (h_h : height > 0) (h_w
    : width > 0) :
    Ref (.dim channels (.dim height (.dim width .scalar))) →
    Ref (.dim channels .scalar) →
    Ref (.dim channels .scalar) →
    m (Ref (.dim channels (.dim height (.dim width .scalar))))
  multiHeadAttention {n numHeads dModel headDim : Nat} (h1 : n ≠ 0) :
    Ref (.dim dModel (.dim (numHeads * headDim) .scalar)) →
    Ref (.dim dModel (.dim (numHeads * headDim) .scalar)) →
    Ref (.dim dModel (.dim (numHeads * headDim) .scalar)) →
    Ref (.dim (numHeads * headDim) (.dim dModel .scalar)) →
    Ref (.dim n (.dim dModel .scalar)) →
    Option (Tensor Bool (.dim n (.dim n .scalar))) →
    m (Ref (.dim n (.dim dModel .scalar)))
  /--
  Multi-head self-attention with an explicit leading batch axis.

  Its mathematical meaning is the leading-axis map of `multiHeadAttention`; implementations may
  execute the samples together, but may not change the mask convention or the per-sample
  forward/VJP semantics.
  -/
  batchedMultiHeadAttention {batch n numHeads dModel headDim : Nat}
    (hBatch : batch ≠ 0) (h1 : n ≠ 0) :
    Ref (.dim dModel (.dim (numHeads * headDim) .scalar)) →
    Ref (.dim dModel (.dim (numHeads * headDim) .scalar)) →
    Ref (.dim dModel (.dim (numHeads * headDim) .scalar)) →
    Ref (.dim (numHeads * headDim) (.dim dModel .scalar)) →
    Ref (.dim batch (.dim n (.dim dModel .scalar))) →
    Option (Tensor Bool (.dim n (.dim n .scalar))) →
    m (Ref (.dim batch (.dim n (.dim dModel .scalar))))
  conv {d inC outC : Nat}
    {kernel stride padding : Vector Nat d}
    {inSpatial : Vector Nat d}
    {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0} :
    Ref (Shape.ofList (outC :: inC :: kernel.toList)) →
    Ref (.dim outC .scalar) →
    Ref (Shape.ofList (inC :: inSpatial.toList)) →
    m (Ref (Shape.ofList (outC :: (Spec.convOutSpatial inSpatial kernel stride padding).toList)))
  convTranspose {d inC outC : Nat}
    {kernel stride padding : Vector Nat d}
    {inSpatial : Vector Nat d}
    {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0} :
    Ref (Shape.ofList (inC :: outC :: kernel.toList)) →
    Ref (.dim outC .scalar) →
    Ref (Shape.ofList (inC :: inSpatial.toList)) →
    m (Ref (Shape.ofList (outC ::
      (Spec.convTransposeOutSpatial inSpatial kernel stride padding).toList)))
  conv2d {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0} :
    Ref (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))) →
    Ref (.dim outC .scalar) →
    Ref (.dim inC (.dim inH (.dim inW .scalar))) →
    m (Ref (.dim outC (.dim (Spec.Shape.slidingWindowOutDim inH kH stride padding)
      (.dim (Spec.Shape.slidingWindowOutDim inW kW stride padding) .scalar))))

  convTranspose2d {inC outC kH kW stride padding inH inW : Nat}
    {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0} :
    Ref (.dim inC (.dim outC (.dim kH (.dim kW .scalar)))) →
    Ref (.dim outC .scalar) →
    Ref (.dim inC (.dim inH (.dim inW .scalar))) →
    m (Ref (.dim outC (.dim (Spec.convTransposeOutDim inH kH stride padding)
      (.dim (Spec.convTransposeOutDim inW kW stride padding) .scalar))))

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

/-- Backend representation of a non-differentiable natural-number tensor. -/
abbrev NatTensorRef (s : Shape) : Type := Ops.NatTensorRef (m := m) (α := α) s

/-- Lift a fixed natural-number tensor into the current backend. -/
def natTensorConst {s : Shape} (x : Tensor Nat s) : NatTensorRef (m := m) (α := α) s :=
  Ops.natTensorConst (m := m) (α := α) x

/-- Transform a non-differentiable natural-number tensor without adding an autograd node. -/
def mapNatTensor {s₁ s₂ : Shape} (f : Tensor Nat s₁ → Tensor Nat s₂)
    (x : NatTensorRef (m := m) (α := α) s₁) : NatTensorRef (m := m) (α := α) s₂ :=
  Ops.mapNatTensor (m := m) (α := α) f x

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
/-- Re-export of `Ops.transpose2d`. PyTorch: `x.t()` / `transpose`. -/
def transpose2d {mDim nDim : Nat}
  (x : Ref (m := m) (α := α) (.dim mDim (.dim nDim .scalar))) :
  m (Ref (m := m) (α := α) (.dim nDim (.dim mDim .scalar))) :=
  Ops.transpose2d (m := m) (α := α) (mDim := mDim) (nDim := nDim) x
/-- Re-export of `Ops.transpose3d_first_to_last`. PyTorch: `permute(1,2,0)`. -/
def transpose3dFirstToLast {a b c : Nat}
  (x : Ref (m := m) (α := α) (.dim a (.dim b (.dim c .scalar)))) :
  m (Ref (m := m) (α := α) (.dim b (.dim c (.dim a .scalar)))) :=
  Ops.transpose3dFirstToLast (m := m) (α := α) (a := a) (b := b) (c := c) x
/-- Re-export of `Ops.transpose3d_last_to_first`. PyTorch: `permute(2,0,1)`. -/
def transpose3dLastToFirst {a b c : Nat}
  (x : Ref (m := m) (α := α) (.dim a (.dim b (.dim c .scalar)))) :
  m (Ref (m := m) (α := α) (.dim c (.dim a (.dim b .scalar)))) :=
  Ops.transpose3dLastToFirst (m := m) (α := α) (a := a) (b := b) (c := c) x
/-- Re-export of `Ops.transpose3d_last_two`. PyTorch: `transpose(1,2)`. -/
def transpose3dLastTwo {a b c : Nat}
  (x : Ref (m := m) (α := α) (.dim a (.dim b (.dim c .scalar)))) :
  m (Ref (m := m) (α := α) (.dim a (.dim c (.dim b .scalar)))) :=
  Ops.transpose3dLastTwo (m := m) (α := α) (a := a) (b := b) (c := c) x
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
def reduceMean {s : Shape} (axis : Nat) [Shape.HasNonemptyAxis axis s] [Shape.WellFormed s]
  (x : Ref (m := m) (α := α) s) : m (Ref (m := m) (α := α) (shapeAfterSum s axis)) :=
  Ops.reduceMean (m := m) (α := α) (s := s) axis x
/-- Re-export of `Ops.gather_scalar`. PyTorch: `x[i]` (1D). -/
def gatherScalar {n : Nat}
  (x : Ref (m := m) (α := α) (.dim n .scalar)) (i : Fin n) : m (Ref (m := m) (α := α) Shape.scalar)
    :=
  Ops.gatherScalar (m := m) (α := α) (n := n) x i
/-- Re-export of `Ops.gather_row`. PyTorch: `x[i]` (2D row). -/
def gatherRow {rows cols : Nat}
  (x : Ref (m := m) (α := α) (.dim rows (.dim cols .scalar))) (i : Fin rows) :
  m (Ref (m := m) (α := α) (.dim cols .scalar)) :=
  Ops.gatherRow (m := m) (α := α) (rows := rows) (cols := cols) x i
/-- Gather from a vector by a raw natural-number index, returning zero out of bounds. -/
def gatherScalarNatOrZero {n : Nat}
  (x : Ref (m := m) (α := α) (.dim n .scalar)) (i : Nat) : m (Ref (m := m) (α := α) Shape.scalar) :=
  Ops.gatherScalarNatOrZero (m := m) (α := α) (n := n) x i
/-- Gather by an index tensor, returning zero at every out-of-bounds position. -/
def gatherVecNatOrZero {n k : Nat}
  (x : Ref (m := m) (α := α) (.dim n .scalar))
  (idx : NatTensorRef (m := m) (α := α) (.dim k .scalar)) :
  m (Ref (m := m) (α := α) (.dim k .scalar)) :=
  Ops.gatherVecNatOrZero (m := m) (α := α) (n := n) (k := k) x idx
/-- Gather matrix rows by an index tensor, returning a zero row out of bounds. -/
def gatherRowsNatOrZero {rows cols k : Nat}
  (x : Ref (m := m) (α := α) (.dim rows (.dim cols .scalar)))
  (idx : NatTensorRef (m := m) (α := α) (.dim k .scalar)) :
  m (Ref (m := m) (α := α) (.dim k (.dim cols .scalar))) :=
  Ops.gatherRowsNatOrZero (m := m) (α := α) (rows := rows) (cols := cols) (k := k) x idx
/-- Re-export of `Ops.scatter_add_vec`. -/
def scatterAddVec {n : Nat}
  (x : Ref (m := m) (α := α) (.dim n .scalar)) (v : Ref (m := m) (α := α) Shape.scalar) (i : Fin n)
    :
  m (Ref (m := m) (α := α) (.dim n .scalar)) :=
  Ops.scatterAddVec (m := m) (α := α) (n := n) x v i
/-- Re-export of `Ops.scatter_add_row`. -/
def scatterAddRow {rows cols : Nat}
  (x : Ref (m := m) (α := α) (.dim rows (.dim cols .scalar)))
  (v : Ref (m := m) (α := α) (.dim cols .scalar))
  (i : Fin rows) :
  m (Ref (m := m) (α := α) (.dim rows (.dim cols .scalar))) :=
  Ops.scatterAddRow (m := m) (α := α) (rows := rows) (cols := cols) x v i
/--
Multiply two rank-two matrices.

This has the semantics of PyTorch's `torch.mm`. Use `bmm` for a rank-three batch of matrix
products; `mm` never broadcasts batch axes.
-/
def mm {mDim nDim pDim : Nat}
  (a : Ref (m := m) (α := α) (.dim mDim (.dim nDim .scalar)))
  (b : Ref (m := m) (α := α) (.dim nDim (.dim pDim .scalar))) :
  m (Ref (m := m) (α := α) (.dim mDim (.dim pDim .scalar))) :=
  Ops.matmul (m := m) (α := α) (mDim := mDim) (nDim := nDim) (pDim := pDim) a b
/-- Re-export of `Ops.bmm`. PyTorch: `torch.bmm`. -/
def bmm {batch mDim nDim pDim : Nat}
  (a : Ref (m := m) (α := α) (.dim batch (.dim mDim (.dim nDim .scalar))))
  (b : Ref (m := m) (α := α) (.dim batch (.dim nDim (.dim pDim .scalar)))) :
  m (Ref (m := m) (α := α) (.dim batch (.dim mDim (.dim pDim .scalar)))) :=
  Ops.bmm (m := m) (α := α) (batch := batch) (mDim := mDim) (nDim := nDim) (pDim := pDim) a b
/-- Re-export of `Ops.concat_leading_axis`. PyTorch: `torch.cat(..., dim=0)`. -/
def concatLeadingAxis {nDim mDim : Nat} {s : Shape}
  (a : Ref (m := m) (α := α) (.dim nDim s))
  (b : Ref (m := m) (α := α) (.dim mDim s)) :
  m (Ref (m := m) (α := α) (.dim (nDim + mDim) s)) :=
  Ops.concatLeadingAxis (m := m) (α := α) (nDim := nDim) (mDim := mDim) (s := s) a b
/-- Re-export of `Ops.slice_leading_axis_range`. PyTorch: `x[start:start+len]` on the leading dimension. -/
def sliceLeadingAxisRange {nDim : Nat} {s : Shape} (start len : Nat) (h : start + len ≤ nDim)
  (x : Ref (m := m) (α := α) (.dim nDim s)) :
  m (Ref (m := m) (α := α) (.dim len s)) :=
  Ops.sliceLeadingAxisRange (m := m) (α := α) (nDim := nDim) (s := s) start len h x

/--
Execution-polymorphic leading-axis map.

This is the reference implementation used by public model lifting and by batched primitives whose
backend does not provide a fused implementation.
-/
def mapLeadingAxis {σ τ : Shape} (f : Ref (m := m) (α := α) σ →
    m (Ref (m := m) (α := α) τ)) {n : Nat}
    (x : Ref (m := m) (α := α) (.dim n σ)) :
    m (Ref (m := m) (α := α) (.dim n τ)) :=
  _root_.Runtime.Autograd.mapLeadingAxisWith
    (const (m := m) (α := α) (s := .dim 0 τ) (Spec.fill (0 : α) (.dim 0 τ)))
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
  {inSpatial kernel stride padding : Vector Nat d}
  {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
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
  {inSpatial kernel stride padding : Vector Nat d}
  (hKernel : ∀ i : Fin d, kernel.get i ≠ 0)
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
def smoothMaxPool {d C : Nat}
  {inSpatial kernel stride padding : Vector Nat d}
  {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (x : Ref (m := m) (α := α) (Shape.ofList (C :: inSpatial.toList)))
  (beta : α) :
  m (Ref (m := m) (α := α)
    (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) :=
  Ops.smoothMaxPool (m := m) (α := α)
    (d := d) (C := C)
    (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
    (hKernel := hKernel)
    x beta
/-- Re-export of `Ops.max_pool2d`. PyTorch: `torch.nn.functional.max_pool2d`. -/
def maxPool2d {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : Ref (m := m) (α := α) (.dim inC (.dim inH (.dim inW .scalar)))) :
  m (Ref (m := m) (α := α) (.dim inC (.dim (Spec.poolOutDim inH kH stride 0) (.dim (Spec.poolOutDim inW kW stride 0)
    .scalar)))) :=
  Ops.maxPool2d (m := m) (α := α) (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC)
    (stride := stride) (h1 := h1) (h2 := h2) x
/-- Re-export of `Ops.max_pool2d_pad`. PyTorch: `max_pool2d(..., padding=...)`. -/
def maxPool2dPad {kH kW inH inW inC stride padding : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : Ref (m := m) (α := α) (.dim inC (.dim inH (.dim inW .scalar)))) :
  m (Ref (m := m) (α := α) (.dim inC (.dim (Spec.poolOutDim inH kH stride padding)
    (.dim (Spec.poolOutDim inW kW stride padding) .scalar)))) :=
  Ops.maxPool2dPad (m := m) (α := α) (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC)
    (stride := stride) (padding := padding) (h1 := h1) (h2 := h2) x

/-- Re-export of `Ops.smooth_max_pool2d` (softmax pooling). -/
def smoothMaxPool2d {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : Ref (m := m) (α := α) (.dim inC (.dim inH (.dim inW .scalar)))) (beta : α) :
  m (Ref (m := m) (α := α) (.dim inC (.dim (Spec.poolOutDim inH kH stride 0) (.dim (Spec.poolOutDim inW kW stride 0)
    .scalar)))) :=
  Ops.smoothMaxPool2d (m := m) (α := α) (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC :=
    inC)
    (stride := stride) (h1 := h1) (h2 := h2) x beta
/-- Re-export of `Ops.avg_pool2d`. PyTorch: `torch.nn.functional.avg_pool2d`. -/
def avgPool2d {kH kW inH inW inC stride : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  (x : Ref (m := m) (α := α) (.dim inC (.dim inH (.dim inW .scalar)))) :
  m (Ref (m := m) (α := α) (.dim inC (.dim (Spec.poolOutDim inH kH stride 0) (.dim (Spec.poolOutDim inW kW stride 0)
    .scalar)))) :=
  Ops.avgPool2d (m := m) (α := α) (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC)
    (stride := stride) h1 h2 x
/-- Re-export of `Ops.avg_pool2d_pad`. PyTorch: `avg_pool2d(..., padding=...)`. -/
def avgPool2dPad {kH kW inH inW inC stride padding : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  (x : Ref (m := m) (α := α) (.dim inC (.dim inH (.dim inW .scalar)))) :
  m (Ref (m := m) (α := α) (.dim inC (.dim (Spec.poolOutDim inH kH stride padding)
    (.dim (Spec.poolOutDim inW kW stride padding) .scalar)))) :=
  Ops.avgPool2dPad (m := m) (α := α) (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC)
    (stride := stride) (padding := padding) h1 h2 x

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
    m (Ref (m := m) (α := α) (.dim (Spec.Shape.size s) .scalar)) :=
  Ops.flatten (m := m) (α := α) (s := s) x

/-- Re-export of `Ops.linear`. PyTorch: `torch.nn.functional.linear`. -/
def linear {inDim outDim : Nat}
  (w : Ref (m := m) (α := α) (.dim outDim (.dim inDim .scalar)))
  (b : Ref (m := m) (α := α) (.dim outDim .scalar))
  (x : Ref (m := m) (α := α) (.dim inDim .scalar)) :
  m (Ref (m := m) (α := α) (.dim outDim .scalar)) :=
  Ops.linear (m := m) (α := α) (inDim := inDim) (outDim := outDim) w b x

/-- Re-export of `Ops.mse_loss`. PyTorch: `torch.nn.functional.mse_loss`. -/
def mseLoss {s : Shape} (yhat target : Ref (m := m) (α := α) s) :
  m (Ref (m := m) (α := α) Shape.scalar) :=
  Ops.mseLoss (m := m) (α := α) (s := s) yhat target

/-- Re-export of `Ops.layer_norm`. PyTorch: `nn.LayerNorm` / `functional.layer_norm`. -/
def layerNorm {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (x : Ref (m := m) (α := α) (.dim seqLen (.dim embedDim .scalar)))
  (gamma : Ref (m := m) (α := α) (.dim embedDim .scalar))
  (beta : Ref (m := m) (α := α) (.dim embedDim .scalar)) :
  m (Ref (m := m) (α := α) (.dim seqLen (.dim embedDim .scalar))) :=
  Ops.layerNorm (m := m) (α := α) (seqLen := seqLen) (embedDim := embedDim)
    h_seq_pos h_embed_pos x gamma beta

/-- Re-export of `Ops.batchnorm_channel_first`. PyTorch: `nn.BatchNorm2d` (conceptually). -/
def batchNormChannelFirst {channels height width : Nat} (h_c : channels > 0) (h_h : height > 0)
  (h_w : width > 0)
  (x : Ref (m := m) (α := α) (.dim channels (.dim height (.dim width .scalar))))
  (gamma : Ref (m := m) (α := α) (.dim channels .scalar))
  (beta : Ref (m := m) (α := α) (.dim channels .scalar)) :
  m (Ref (m := m) (α := α) (.dim channels (.dim height (.dim width .scalar)))) :=
  Ops.batchNormChannelFirst (m := m) (α := α) (channels := channels) (height := height) (width :=
    width)
    h_c h_h h_w x gamma beta

/-- Re-export of `Ops.multi_head_attention`. -/
def multiHeadAttention {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (wq : Ref (m := m) (α := α) (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wk : Ref (m := m) (α := α) (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wv : Ref (m := m) (α := α) (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wo : Ref (m := m) (α := α) (.dim (numHeads * headDim) (.dim dModel .scalar)))
  (x : Ref (m := m) (α := α) (.dim n (.dim dModel .scalar)))
  (mask : Option (Tensor Bool (.dim n (.dim n .scalar))) := none) :
  m (Ref (m := m) (α := α) (.dim n (.dim dModel .scalar))) :=
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
  (wq : Ref (m := m) (α := α) (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wk : Ref (m := m) (α := α) (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wv : Ref (m := m) (α := α) (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wo : Ref (m := m) (α := α) (.dim (numHeads * headDim) (.dim dModel .scalar)))
  (x : Ref (m := m) (α := α) (.dim batch (.dim n (.dim dModel .scalar))))
  (mask : Option (Tensor Bool (.dim n (.dim n .scalar))) := none) :
  m (Ref (m := m) (α := α) (.dim batch (.dim n (.dim dModel .scalar)))) :=
  Ops.batchedMultiHeadAttention (m := m) (α := α)
    (batch := batch) (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
    hBatch h1 wq wk wv wo x mask

/--
Re-export of `Ops.conv` (generic N-D convolution, channels-first).

PyTorch comparison: `torch.nn.functional.conv{d}d` specialized to a single sample (no batch axis).
-/
def conv {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (weight : Ref (m := m) (α := α) (Shape.ofList (outC :: inC :: kernel.toList)))
  (bias : Ref (m := m) (α := α) (.dim outC .scalar))
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
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (weight : Ref (m := m) (α := α) (Shape.ofList (inC :: outC :: kernel.toList)))
  (bias : Ref (m := m) (α := α) (.dim outC .scalar))
  (input : Ref (m := m) (α := α) (Shape.ofList (inC :: inSpatial.toList))) :
  m (Ref (m := m) (α := α)
    (Shape.ofList (outC ::
      (Spec.convTransposeOutSpatial inSpatial kernel stride padding).toList))) :=
  Ops.convTranspose (m := m) (α := α)
    (d := d) (inC := inC) (outC := outC)
    (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
    (hInC := hInC) (hKernel := hKernel)
    weight bias input

/-- Re-export of `Ops.conv2d`. PyTorch: `torch.nn.functional.conv2d` (conceptually, no batch axis).
  -/
def conv2d {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (kernel : Ref (m := m) (α := α) (.dim outC (.dim inC (.dim kH (.dim kW .scalar)))))
  (bias : Ref (m := m) (α := α) (.dim outC .scalar))
  (input : Ref (m := m) (α := α) (.dim inC (.dim inH (.dim inW .scalar)))) :
  m (Ref (m := m) (α := α) (.dim outC (.dim (Spec.Shape.slidingWindowOutDim inH kH stride padding)
    (.dim (Spec.Shape.slidingWindowOutDim inW kW stride padding) .scalar)))) :=
  Ops.conv2d (m := m) (α := α) (inC := inC) (outC := outC) (kH := kH) (kW := kW)
    (stride := stride) (padding := padding) (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 :=
      h3)
    kernel bias input

/-- Re-export of `Ops.conv_transpose2d`. PyTorch: `torch.nn.functional.conv_transpose2d`. -/
def convTranspose2d {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (kernel : Ref (m := m) (α := α) (.dim inC (.dim outC (.dim kH (.dim kW .scalar)))))
  (bias : Ref (m := m) (α := α) (.dim outC .scalar))
  (input : Ref (m := m) (α := α) (.dim inC (.dim inH (.dim inW .scalar)))) :
  m (Ref (m := m) (α := α) (.dim outC (.dim (Spec.convTransposeOutDim inH kH stride padding)
    (.dim (Spec.convTransposeOutDim inW kW stride padding) .scalar)))) :=
  Ops.convTranspose2d (m := m) (α := α)
    (inC := inC) (outC := outC) (kH := kH) (kW := kW)
    (stride := stride) (padding := padding) (inH := inH) (inW := inW)
    (h1 := h1) (h2 := h2) (h3 := h3)
    kernel bias input

end
end Torch
end Autograd
end Runtime
