/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.DAG.Primitives.Shape
public import NN.Spec.Layers.Normalization
public import NN.Spec.Models.CommonHelpers

/-!
# DAG Normalization Primitives

RMS normalization and regularized L2 normalization with explicit shape and positivity contracts.
-/

@[expose] public section

namespace NN
namespace GraphSpec
namespace DAG

open _root_.NN.Spec
open Spec.Tensor
open NN.Tensor

namespace PrimOp

/-- Root-mean-square normalization along the last axis of a matrix. -/
def rmsNorm (rows width : Nat) (hRows : 0 < rows) (hWidth : 0 < width) :
    PrimOp
      [.dim rows (.dim width .scalar), .dim width .scalar]
      (.dim rows (.dim width .scalar)) :=
  { name := s!"rmsNorm({rows},{width})"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons input (.cons gamma .nil) =>
          _root_.Spec.rmsNorm (α := α) input gamma hRows hWidth
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input gamma =>
        Runtime.Autograd.TorchLean.Norm.rmsNormLast (m := m) (α := α)
          hRows hWidth input gamma }

/-- Pure evaluation of matrix RMS normalization. -/
@[simp] theorem rmsNorm_specFwd {rows width : Nat}
    (hRows : 0 < rows) (hWidth : 0 < width)
    {α : Type} [Context α]
    (input : _root_.Spec.Tensor α (.dim rows (.dim width .scalar)))
    (gamma : _root_.Spec.Tensor α (.dim width .scalar)) :
    (rmsNorm rows width hRows hWidth).specFwd (.cons input (.cons gamma .nil)) =
      _root_.Spec.rmsNorm input gamma hRows hWidth := by
  rfl

/--
Interpret vector RMSNorm through the canonical matrix specification.

This is a shape adapter, not a second RMSNorm definition: it inserts one leading row, applies
`Spec.rmsNorm`, and selects that row again.
-/
def rmsNormVectorSemantics {α : Type} [Context α] {width : Nat}
    (hWidth : 0 < width) (input gamma : _root_.Spec.Tensor α (.dim width .scalar)) :
    _root_.Spec.Tensor α (.dim width .scalar) :=
  _root_.Spec.getAtSpec
    (_root_.Spec.rmsNorm (α := α) (.dim fun _ => input) gamma
      (Nat.zero_lt_succ 0) hWidth)
    ⟨0, Nat.zero_lt_succ 0⟩

/-- Root-mean-square normalization of one vector.

Single-token decoders and recurrent layers naturally carry a vector rather than a singleton
matrix. The pure semantics uses `rmsNormVectorSemantics`; the executable path reshapes the vector
to one row, calls the same RMSNorm runtime operation used by batched models, and restores the
vector shape. This keeps one normalization convention without exposing layout plumbing to graph
authors.
-/
def rmsNormVector (width : Nat) (hWidth : 0 < width) :
    PrimOp [.dim width .scalar, .dim width .scalar] (.dim width .scalar) :=
  { name := s!"rmsNormVector({width})"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons input (.cons gamma .nil) =>
          rmsNormVectorSemantics hWidth input gamma
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input gamma =>
        (do
          let row ← Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₁ := .dim width .scalar) (s₂ := .dim 1 (.dim width .scalar)) input (by
              simp [_root_.Spec.Shape.size])
          let normalized ← Runtime.Autograd.TorchLean.Norm.rmsNormLast (m := m) (α := α)
            (Nat.zero_lt_succ 0) hWidth row gamma
          Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₁ := .dim 1 (.dim width .scalar)) (s₂ := .dim width .scalar) normalized (by
              simp [_root_.Spec.Shape.size]) :
          m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) (.dim width .scalar))) }

/-- The vector RMSNorm primitive exposes the canonical vector specification directly. -/
@[simp] theorem rmsNormVector_specFwd {width : Nat}
    {α : Type} [Context α] (hWidth : 0 < width)
    (input gamma : _root_.Spec.Tensor α (.dim width .scalar)) :
    (rmsNormVector width hWidth).specFwd (.cons input (.cons gamma .nil)) =
      rmsNormVectorSemantics hWidth input gamma := by
  rfl

/-- Root-mean-square normalization with a separate learned scale for every row.

Unlike `rmsNorm`, whose scale vector is shared across rows, both inputs have shape
`rows × width`.  Row `i` is normalized along its final axis and then multiplied coordinatewise by
row `i` of `gamma`.  This covers head-specific normalization without encoding the head axis in a
special-purpose primitive. -/
def rmsNormRows (rows width : Nat) (hRows : 0 < rows) (hWidth : 0 < width) :
    PrimOp
      [.dim rows (.dim width .scalar), .dim rows (.dim width .scalar)]
      (.dim rows (.dim width .scalar)) :=
  { name := s!"rmsNormRows({rows},{width})"
    specFwd := fun {_} _ xs =>
      match xs with
      | .cons input (.cons gamma .nil) =>
          .dim fun row => rmsNormVectorSemantics hWidth
            (_root_.Spec.get input row) (_root_.Spec.get gamma row)
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input gamma =>
        (do
          let ones ← Runtime.Autograd.TorchLean.const (m := m) (α := α)
            (_root_.Spec.fill 1 (.dim width .scalar))
          let normalized ← Runtime.Autograd.TorchLean.Norm.rmsNormLast (m := m) (α := α)
            hRows hWidth input ones
          Runtime.Autograd.TorchLean.mul (m := m) (α := α)
            (s := .dim rows (.dim width .scalar)) normalized gamma :
          m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α)
            (.dim rows (.dim width .scalar)))) }

/-- Pure evaluation of RMS normalization with one scale vector per row. -/
@[simp] theorem rmsNormRows_specFwd {rows width : Nat}
    (hRows : 0 < rows) (hWidth : 0 < width) {α : Type} [Context α]
    (input gamma : _root_.Spec.Tensor α (.dim rows (.dim width .scalar))) :
    (rmsNormRows rows width hRows hWidth).specFwd (.cons input (.cons gamma .nil)) =
      .dim (fun row => rmsNormVectorSemantics hWidth
        (_root_.Spec.get input row) (_root_.Spec.get gamma row)) := by
  rfl


/-- Normalize every matrix row by `sqrt(sum (x * x) + epsilon)`.

The stabilizer is a scalar graph input rather than a declaration parameter. This keeps the exact
numerical convention visible in captured graphs and permits architectures to select their own
positive epsilon without adding another primitive.
-/
def l2Normalize (rows width : Nat) (hRows : 0 < rows) (hWidth : 0 < width) :
    PrimOp
      [.dim rows (.dim width .scalar), .scalar]
      (.dim rows (.dim width .scalar)) :=
  letI : Fact (0 < rows) := ⟨hRows⟩
  letI : Fact (0 < width) := ⟨hWidth⟩
  letI : _root_.Spec.Shape.valid_axis_inst 1
      (.dim rows (.dim width .scalar)) :=
    _root_.Spec.Shape.validAxisInstOne (Nat.ne_of_gt hRows) (Nat.ne_of_gt hWidth)
  { name := s!"l2Normalize({rows},{width})"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons input (.cons (.scalar epsilon) .nil) =>
          _root_.Spec.Tensor.dim fun row =>
            _root_.Spec.normalizeL2RegularizedSpec (_root_.Spec.get input row) epsilon
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input epsilon =>
        (do
          let squared ← Runtime.Autograd.TorchLean.mul (m := m) (α := α)
            (s := .dim rows (.dim width .scalar)) input input
          let normSquared ← Runtime.Autograd.TorchLean.reduceSum (m := m) (α := α)
            (s := .dim rows (.dim width .scalar)) 1 squared
          let epsilonRows ← Runtime.Autograd.TorchLean.broadcastTo (m := m) (α := α)
            (s₁ := .scalar) (s₂ := .dim rows .scalar)
            (_root_.Spec.Shape.CanBroadcastTo.scalar_to_any (.dim rows .scalar)) epsilon
          let shifted ← Runtime.Autograd.TorchLean.add (m := m) (α := α)
            (s := .dim rows .scalar) normSquared epsilonRows
          let denominator ← Runtime.Autograd.TorchLean.sqrt (m := m) (α := α)
            (s := .dim rows .scalar) shifted
          let inverse ← Runtime.Autograd.TorchLean.inv (m := m) (α := α)
            (s := .dim rows .scalar) denominator
          let inverseColumn ← Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₂ := .dim rows (.dim 1 .scalar)) inverse (by simp [_root_.Spec.Shape.size])
          let inverseMatrix ← Runtime.Autograd.TorchLean.broadcastTo (m := m) (α := α)
            (s₁ := .dim rows (.dim 1 .scalar))
            (s₂ := .dim rows (.dim width .scalar))
            (_root_.Spec.Shape.CanBroadcastTo.dim_eq
              (_root_.Spec.Shape.CanBroadcastTo.dim_1_to_n
                (_root_.Spec.Shape.CanBroadcastTo.scalar_to_any .scalar))) inverseColumn
          Runtime.Autograd.TorchLean.mul (m := m) (α := α)
            (s := .dim rows (.dim width .scalar)) input inverseMatrix :
          m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α)
            (.dim rows (.dim width .scalar)))) }

/-- Pure evaluation of row-wise regularized L2 normalization. -/
@[simp] theorem l2Normalize_specFwd {rows width : Nat}
    (hRows : 0 < rows) (hWidth : 0 < width) {α : Type} [Context α]
    (input : _root_.Spec.Tensor α (.dim rows (.dim width .scalar))) (epsilon : α) :
    (l2Normalize rows width hRows hWidth).specFwd
        (.cons input (.cons (.scalar epsilon) .nil)) =
      .dim (fun row => _root_.Spec.normalizeL2RegularizedSpec
        (_root_.Spec.get input row) epsilon) := by
  rfl


end PrimOp

end DAG
end GraphSpec
end NN
