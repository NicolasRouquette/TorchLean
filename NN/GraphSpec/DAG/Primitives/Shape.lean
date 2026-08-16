/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.DAG.Primitives.Core

/-!
# DAG Shape and Reduction Primitives

Reductions, transposes, gathers, broadcasting, concatenation, slicing, and shape-preserving views.
-/

@[expose] public section

namespace NN
namespace GraphSpec
namespace DAG

open _root_.NN.Spec
open Spec.Tensor
open NN.Tensor

namespace PrimOp

/-- Sum the leading axis of a nonempty tensor. -/
def reduceLeadingSum (outer : Nat) (inner : Shape) (hOuter : 0 < outer)
    [_root_.Spec.Shape.WellFormed inner] : PrimOp [.dim outer inner] inner :=
  letI : Fact (0 < outer) := ⟨hOuter⟩
  letI : _root_.Spec.Shape.valid_axis_inst 0 (.dim outer inner) :=
    _root_.Spec.Shape.validAxisInstZeroAlt2 hOuter
  { name := s!"reduceLeadingSum({outer})"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons input .nil => _root_.Spec.Tensor.reduceSumAuto (α := α) 0 input
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input =>
        Runtime.Autograd.TorchLean.reduceSum (m := m) (α := α) 0 input }

/-- Pure evaluation of a leading-axis sum. -/
@[simp] theorem reduceLeadingSum_specFwd {outer : Nat} {inner : Shape}
    (hOuter : 0 < outer) [_root_.Spec.Shape.WellFormed inner]
    {α : Type} [Context α] (input : _root_.Spec.Tensor α (.dim outer inner)) :
    (reduceLeadingSum outer inner hOuter).specFwd (.cons input .nil) =
      _root_.Spec.Tensor.reduceSumAuto (α := α)
        (h := _root_.Spec.Shape.validAxisInstZeroAlt2 hOuter) 0 input := by
  rfl

/-- Average the leading axis of a nonempty tensor. -/
def reduceLeadingMean (outer : Nat) (inner : Shape) (hOuter : 0 < outer)
    [_root_.Spec.Shape.WellFormed inner] : PrimOp [.dim outer inner] inner :=
  letI : Fact (0 < outer) := ⟨hOuter⟩
  letI : _root_.Spec.Shape.valid_axis_inst 0 (.dim outer inner) :=
    _root_.Spec.Shape.validAxisInstZeroAlt2 hOuter
  { name := s!"reduceLeadingMean({outer})"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons input .nil => _root_.Spec.Tensor.reduceMeanAuto (α := α) 0 inferInstance input
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input =>
        Runtime.Autograd.TorchLean.reduceMean (m := m) (α := α) 0 input }

/-- Pure evaluation of leading-axis mean reduction. -/
@[simp] theorem reduceLeadingMean_specFwd {outer : Nat} {inner : Shape}
    (hOuter : 0 < outer) [_root_.Spec.Shape.WellFormed inner]
    {α : Type} [Context α] (input : _root_.Spec.Tensor α (.dim outer inner)) :
    (reduceLeadingMean outer inner hOuter).specFwd (.cons input .nil) =
      _root_.Spec.Tensor.reduceMeanAuto (α := α) 0
        (_root_.Spec.Shape.validAxisInstZeroAlt2 hOuter) input := by
  rfl

/-- Transpose a two-dimensional tensor. -/
def transpose2d (rows columns : Nat) :
    PrimOp [.dim rows (.dim columns .scalar)] (.dim columns (.dim rows .scalar)) :=
  { name := s!"transpose2d({rows},{columns})"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons matrix .nil => _root_.Spec.Tensor.matrixTransposeSpec (α := α) matrix
    program := fun {α} _ _ =>
      fun {m} _ _ => fun matrix =>
        Runtime.Autograd.TorchLean.transpose2d (m := m) (α := α)
          (mDim := rows) (nDim := columns) matrix }

/-- Swap the last two axes of a rank-three tensor. -/
def transpose3dLastTwo (batch rows columns : Nat) :
    PrimOp
      [.dim batch (.dim rows (.dim columns .scalar))]
      (.dim batch (.dim columns (.dim rows .scalar))) :=
  { name := s!"transpose3dLastTwo({batch},{rows},{columns})"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons input .nil => _root_.Spec.Tensor.transpose3DLastTwoSpec (α := α) input
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input =>
        Runtime.Autograd.TorchLean.transpose3dLastTwo (m := m) (α := α)
          (a := batch) (b := rows) (c := columns) input }

/-- Swap adjacent tensor axes at a statically chosen depth. -/
def swapAdjacentAtDepth (s : Shape) (depth : Nat) :
    PrimOp [s] (s.swapAdjacentAtDepth depth) :=
  { name := s!"swapAdjacentAtDepth({depth})"
    specFwd := fun {_α} _ xs =>
      match xs with
      | .cons input .nil => _root_.Spec.Tensor.swapAtDepthHelper input depth
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input =>
        Runtime.Autograd.TorchLean.swapAdjacentAtDepth (m := m) (α := α) depth input }

/-- Pure evaluation of an adjacent-axis swap. -/
@[simp] theorem swapAdjacentAtDepth_specFwd {α : Type} [Context α]
    (s : Shape) (depth : Nat) (input : _root_.Spec.Tensor α s) :
    (swapAdjacentAtDepth s depth).specFwd (.cons input .nil) =
      _root_.Spec.Tensor.swapAtDepthHelper input depth := by
  rfl

/-- Select one row of a statically shaped matrix. -/
def gatherRow (rows columns : Nat) (row : Fin rows) :
    PrimOp [.dim rows (.dim columns .scalar)] (.dim columns .scalar) :=
  { name := s!"gatherRow({row.val})"
    specFwd := fun {_α} _ xs =>
      match xs with
      | .cons input .nil => _root_.Spec.sliceSpec input row
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input =>
        Runtime.Autograd.TorchLean.gatherRow (m := m) (α := α) input row }

/-- Select one slice along the leading axis of an arbitrary tensor.

The executable path views each slice as one flattened row, uses the runtime's differentiable row
gather, and restores the inner shape. This is useful for packed parameter banks such as embeddings
and mixture-of-experts weights.
-/
def gatherLeading (rows : Nat) (inner : Shape) (row : Fin rows) :
    PrimOp [.dim rows inner] inner :=
  { name := s!"gatherLeading({row.val})"
    specFwd := fun {_α} _ xs =>
      match xs with
      | .cons input .nil => _root_.Spec.sliceSpec input row
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input =>
        (do
          let matrix ← Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₂ := .dim rows (.dim inner.size .scalar)) input
            (by simp [_root_.Spec.Shape.size])
          let flat ← Runtime.Autograd.TorchLean.gatherRow (m := m) (α := α) matrix row
          Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₂ := inner) flat (by simp [_root_.Spec.Shape.size]) :
          m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) inner)) }

/-- Pure semantics of leading-axis gather, stated through the standard tensor indexing API. -/
@[simp] theorem gatherLeading_specFwd {rows : Nat} {inner : Shape} {row : Fin rows}
    {α : Type} [Context α] (input : _root_.Spec.Tensor α (.dim rows inner)) :
    (gatherLeading rows inner row).specFwd (.cons input .nil) = _root_.Spec.get input row := by
  cases input
  rfl

/-- Expand singleton or missing dimensions according to a checked broadcasting derivation. -/
def broadcast {source target : Shape} (proof : source.CanBroadcastTo target) :
    PrimOp [source] target :=
  { name := "broadcast"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons input .nil => _root_.Spec.Tensor.broadcastTo (α := α) proof input
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input =>
        Runtime.Autograd.TorchLean.broadcastTo (m := m) (α := α) proof input }

/-- Concatenate tensors along their leading axis. -/
def concatLeadingAxis (left right : Nat) (inner : Shape) :
    PrimOp [.dim left inner, .dim right inner] (.dim (left + right) inner) :=
  { name := s!"concatLeadingAxis({left},{right})"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons a (.cons b .nil) =>
          _root_.Spec.Tensor.concatLeadingAxisSpec (α := α) a b
    program := fun {α} _ _ =>
      fun {m} _ _ => fun a b =>
        Runtime.Autograd.TorchLean.concatLeadingAxis (m := m) (α := α)
          (nDim := left) (mDim := right) (s := inner) a b }

/-- Pure semantics of concatenation along the leading tensor axis. -/
@[simp] theorem concatLeadingAxis_specFwd {left right : Nat} {inner : Shape}
    {α : Type} [Context α]
    (a : _root_.Spec.Tensor α (.dim left inner))
    (b : _root_.Spec.Tensor α (.dim right inner)) :
    (concatLeadingAxis left right inner).specFwd (.cons a (.cons b .nil)) =
      _root_.Spec.Tensor.concatLeadingAxisSpec a b := by
  rfl

/-- Select a contiguous range along the leading axis.

The range proof makes the output shape total at graph-construction time. The operation is useful
for rolling sequence windows and cache prefixes, and lowers to the existing differentiable
TorchLean slice rather than introducing a backend-specific indexing node.
-/
def sliceLeadingAxisRange (total start length : Nat) (inner : Shape)
    (hRange : length + start ≤ total) :
    PrimOp [.dim total inner] (.dim length inner) :=
  { name := s!"sliceLeadingAxisRange({start},{length})"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons input .nil =>
          _root_.Spec.Tensor.sliceLeadingAxisRangeSpec
            (α := α) (n := total) (s := inner) start length hRange input
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input =>
        Runtime.Autograd.TorchLean.sliceLeadingAxisRange (m := m) (α := α)
          (nDim := total) (s := inner) start length hRange input }

/-- Pure semantics of a checked contiguous slice of the leading tensor axis. -/
@[simp] theorem sliceLeadingAxisRange_specFwd {total start length : Nat} {inner : Shape}
    (hRange : length + start ≤ total) {α : Type} [Context α]
    (input : _root_.Spec.Tensor α (.dim total inner)) :
    (sliceLeadingAxisRange total start length inner hRange).specFwd (.cons input .nil) =
      _root_.Spec.Tensor.sliceLeadingAxisRangeSpec start length hRange input := by
  rfl


/-- Reshape a tensor while preserving its statically known number of entries. -/
def reshape (source target : Shape) (hSize : source.size = target.size) :
    PrimOp [source] target :=
  { name := "reshape"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons input .nil => _root_.Spec.Tensor.reshapeSpec (α := α) input hSize
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input =>
        Runtime.Autograd.TorchLean.reshape (m := m) (α := α) input hSize }

/-- Pure evaluation of a shape-preserving view. -/
@[simp] theorem reshape_specFwd {source target : Shape}
    (hSize : source.size = target.size) {α : Type} [Context α]
    (input : _root_.Spec.Tensor α source) :
    (reshape source target hSize).specFwd (.cons input .nil) =
      _root_.Spec.Tensor.reshapeSpec input hSize := by
  rfl


end PrimOp

end DAG
end GraphSpec
end NN
