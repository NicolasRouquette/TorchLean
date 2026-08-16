/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.DAG.Primitives.Core

/-!
# DAG Linear Algebra Primitives

Typed matrix, vector, batched contraction, and scaling operations for general computation graphs.
-/

@[expose] public section

namespace NN
namespace GraphSpec
namespace DAG

open _root_.NN.Spec
open Spec.Tensor
open NN.Tensor

namespace PrimOp

/-- Matrix multiplication with statically checked inner dimensions. -/
def matmul (mDim nDim pDim : Nat) :
    PrimOp
      [.dim mDim (.dim nDim .scalar), .dim nDim (.dim pDim .scalar)]
      (.dim mDim (.dim pDim .scalar)) :=
  { name := s!"matmul({mDim},{nDim},{pDim})"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons a (.cons b .nil) => _root_.Spec.matMulSpec (α := α) a b
    program := fun {α} _ _ =>
      fun {m} _ _ => fun a b =>
        Runtime.Autograd.TorchLean.mm (m := m) (α := α)
          (mDim := mDim) (nDim := nDim) (pDim := pDim) a b }

/-- Pure evaluation of statically shaped matrix multiplication. -/
@[simp] theorem matmul_specFwd {mDim nDim pDim : Nat}
    {α : Type} [Context α]
    (left : _root_.Spec.Tensor α (.dim mDim (.dim nDim .scalar)))
    (right : _root_.Spec.Tensor α (.dim nDim (.dim pDim .scalar))) :
    (matmul mDim nDim pDim).specFwd (.cons left (.cons right .nil)) =
      _root_.Spec.matMulSpec left right := by
  rfl

/-- Batched matrix multiplication with a statically shared batch dimension. -/
def bmm (batch mDim nDim pDim : Nat) :
    PrimOp
      [ .dim batch (.dim mDim (.dim nDim .scalar)),
        .dim batch (.dim nDim (.dim pDim .scalar)) ]
      (.dim batch (.dim mDim (.dim pDim .scalar))) :=
  { name := s!"bmm({batch},{mDim},{nDim},{pDim})"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons a (.cons b .nil) => _root_.Spec.Tensor.bmmSpec (α := α) a b
    program := fun {α} _ _ =>
      fun {m} _ _ => fun a b =>
        Runtime.Autograd.TorchLean.bmm (m := m) (α := α)
          (batch := batch) (mDim := mDim) (nDim := nDim) (pDim := pDim) a b }

/-- Multiply one shared matrix by a batch of matrices.

The pure semantics expose the operation as a function over the batch axis.  The executable path
broadcasts the shared matrix once and delegates to batched matrix multiplication.  This is useful
whenever every head or expert consumes the same activation matrix but owns distinct weights.
-/
def batchedSharedMatMul (batch mDim nDim pDim : Nat) :
    PrimOp
      [ .dim mDim (.dim nDim .scalar),
        .dim batch (.dim nDim (.dim pDim .scalar)) ]
      (.dim batch (.dim mDim (.dim pDim .scalar))) :=
  { name := s!"batchedSharedMatMul({batch},{mDim},{nDim},{pDim})"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons shared (.cons (.dim weights) .nil) =>
          .dim fun i => _root_.Spec.matMulSpec (α := α) shared (weights i)
    program := fun {α} _ _ =>
      fun {m} _ _ => fun shared weights =>
        (do
          let singleton ← Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₂ := .dim 1 (.dim mDim (.dim nDim .scalar))) shared
            (by simp [_root_.Spec.Shape.size])
          let expanded ← Runtime.Autograd.TorchLean.broadcastTo (m := m) (α := α)
            (s₂ := .dim batch (.dim mDim (.dim nDim .scalar)))
            (_root_.Spec.Shape.CanBroadcastTo.dim_1_to_n
              (_root_.Spec.Shape.CanBroadcastTo.dim_eq
                (_root_.Spec.Shape.CanBroadcastTo.dim_eq
                  (_root_.Spec.Shape.CanBroadcastTo.scalar_to_any .scalar)))) singleton
          Runtime.Autograd.TorchLean.bmm (m := m) (α := α)
            (batch := batch) (mDim := mDim) (nDim := nDim) (pDim := pDim)
            expanded weights :
          m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α)
            (.dim batch (.dim mDim (.dim pDim .scalar))))) }

/-- Pure evaluation of a shared matrix against a batch of matrices. -/
@[simp] theorem batchedSharedMatMul_specFwd {batch mDim nDim pDim : Nat}
    {α : Type} [Context α]
    (shared : _root_.Spec.Tensor α (.dim mDim (.dim nDim .scalar)))
    (weights : _root_.Spec.Tensor α (.dim batch (.dim nDim (.dim pDim .scalar)))) :
    (batchedSharedMatMul batch mDim nDim pDim).specFwd
        (.cons shared (.cons weights .nil)) =
      .dim (fun i => _root_.Spec.matMulSpec shared (_root_.Spec.get weights i)) := by
  cases weights
  rfl

/-- Multiply one shared vector by a batch of matrices. -/
def batchedSharedVecMat (batch rows columns : Nat) :
    PrimOp
      [.dim rows .scalar, .dim batch (.dim rows (.dim columns .scalar))]
      (.dim batch (.dim columns .scalar)) :=
  { name := s!"batchedSharedVecMat({batch},{rows},{columns})"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons vector (.cons (.dim matrices) .nil) =>
          .dim fun i => _root_.Spec.vecMatMulSpec (α := α) vector (matrices i)
    program := fun {α} _ _ =>
      fun {m} _ _ => fun vector matrices =>
        (do
          let singleton ← Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₂ := .dim 1 (.dim 1 (.dim rows .scalar))) vector
            (by simp [_root_.Spec.Shape.size])
          let expanded ← Runtime.Autograd.TorchLean.broadcastTo (m := m) (α := α)
            (s₂ := .dim batch (.dim 1 (.dim rows .scalar)))
            (_root_.Spec.Shape.CanBroadcastTo.dim_1_to_n
              (_root_.Spec.Shape.CanBroadcastTo.dim_eq
                (_root_.Spec.Shape.CanBroadcastTo.dim_eq
                  (_root_.Spec.Shape.CanBroadcastTo.scalar_to_any .scalar)))) singleton
          let product ← Runtime.Autograd.TorchLean.bmm (m := m) (α := α)
            (batch := batch) (mDim := 1) (nDim := rows) (pDim := columns)
            expanded matrices
          Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₂ := .dim batch (.dim columns .scalar)) product
            (by simp [_root_.Spec.Shape.size]) :
          m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α)
            (.dim batch (.dim columns .scalar)))) }

/-- Pure evaluation of a shared vector against a batch of matrices. -/
@[simp] theorem batchedSharedVecMat_specFwd {batch rows columns : Nat}
    {α : Type} [Context α]
    (vector : _root_.Spec.Tensor α (.dim rows .scalar))
    (matrices : _root_.Spec.Tensor α (.dim batch (.dim rows (.dim columns .scalar)))) :
    (batchedSharedVecMat batch rows columns).specFwd
        (.cons vector (.cons matrices .nil)) =
      .dim (fun i => _root_.Spec.vecMatMulSpec vector (_root_.Spec.get matrices i)) := by
  cases matrices
  rfl

/-- Multiply each vector in a batch by the corresponding matrix. -/
def batchedVecMat (batch rows columns : Nat) :
    PrimOp
      [.dim batch (.dim rows .scalar), .dim batch (.dim rows (.dim columns .scalar))]
      (.dim batch (.dim columns .scalar)) :=
  { name := s!"batchedVecMat({batch},{rows},{columns})"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons (.dim vectors) (.cons (.dim matrices) .nil) =>
          .dim fun i => _root_.Spec.vecMatMulSpec (α := α) (vectors i) (matrices i)
    program := fun {α} _ _ =>
      fun {m} _ _ => fun vectors matrices =>
        (do
          let rows' ← Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₂ := .dim batch (.dim 1 (.dim rows .scalar))) vectors
            (by simp [_root_.Spec.Shape.size])
          let product ← Runtime.Autograd.TorchLean.bmm (m := m) (α := α)
            (batch := batch) (mDim := 1) (nDim := rows) (pDim := columns)
            rows' matrices
          Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₂ := .dim batch (.dim columns .scalar)) product
            (by simp [_root_.Spec.Shape.size]) :
          m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α)
            (.dim batch (.dim columns .scalar)))) }

/-- Pure evaluation of pairwise batched vector-matrix multiplication. -/
@[simp] theorem batchedVecMat_specFwd {batch rows columns : Nat}
    {α : Type} [Context α]
    (vectors : _root_.Spec.Tensor α (.dim batch (.dim rows .scalar)))
    (matrices : _root_.Spec.Tensor α (.dim batch (.dim rows (.dim columns .scalar)))) :
    (batchedVecMat batch rows columns).specFwd (.cons vectors (.cons matrices .nil)) =
      .dim (fun i => _root_.Spec.vecMatMulSpec
        (_root_.Spec.get vectors i) (_root_.Spec.get matrices i)) := by
  cases vectors
  cases matrices
  rfl

/-- Dot one shared vector with every vector in a batch. -/
def batchedSharedDot (batch width : Nat) :
    PrimOp [.dim width .scalar, .dim batch (.dim width .scalar)] (.dim batch .scalar) :=
  { name := s!"batchedSharedDot({batch},{width})"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons vector (.cons (.dim vectors) .nil) =>
          .dim fun i => .scalar (_root_.Spec.Tensor.dotSpec (α := α) vector (vectors i))
    program := fun {α} _ _ =>
      fun {m} _ _ => fun vector vectors =>
        (do
          let column ← Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₂ := .dim width (.dim 1 .scalar)) vector
            (by simp [_root_.Spec.Shape.size])
          Runtime.Autograd.TorchLean.mm (m := m) (α := α)
            (mDim := batch) (nDim := width) (pDim := 1) vectors column >>= fun result =>
          Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₂ := .dim batch .scalar) result (by simp [_root_.Spec.Shape.size]) :
          m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) (.dim batch .scalar))) }

/-- Apply a depthwise weighted reduction independently to every element of a batch.

Both inputs use the layout `batch × width × channels`.  For each batch index and channel, the
result is the sum over `width` of the pointwise product.  The executable program implements that
equation with elementwise multiplication, an axis swap, and a leading-axis reduction; exposing the
operation here avoids making every caller reproduce that layout plumbing.
-/
def batchedDepthwiseWeightedSum (batch width channels : Nat)
    (hBatch : 0 < batch) (hWidth : 0 < width) (hChannels : 0 < channels) :
    PrimOp
      [ .dim batch (.dim width (.dim channels .scalar)),
        .dim batch (.dim width (.dim channels .scalar)) ]
      (.dim batch (.dim channels .scalar)) :=
  letI : Fact (0 < batch) := ⟨hBatch⟩
  letI : Fact (0 < width) := ⟨hWidth⟩
  letI : Fact (0 < channels) := ⟨hChannels⟩
  letI : _root_.Spec.Shape.valid_axis_inst 0
      (.dim width (.dim channels .scalar)) :=
    _root_.Spec.Shape.validAxisInstZeroAlt2 hWidth
  letI : _root_.Spec.Shape.valid_axis_inst 0
      (.dim width (.dim batch (.dim channels .scalar))) :=
    _root_.Spec.Shape.validAxisInstZeroAlt2 hWidth
  { name := s!"batchedDepthwiseWeightedSum({batch},{width},{channels})"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons (.dim values) (.cons (.dim weights) .nil) =>
          .dim fun i => _root_.Spec.Tensor.reduceSumAuto (α := α) 0
            (_root_.Spec.Tensor.mulSpec (values i) (weights i))
    program := fun {α} _ _ =>
      fun {m} _ _ => fun values weights =>
        (do
          let weighted ← Runtime.Autograd.TorchLean.mul (m := m) (α := α)
            (s := .dim batch (.dim width (.dim channels .scalar))) values weights
          let tapsFirst : Runtime.Autograd.TorchLean.RefTy (m := m) (α := α)
              (.dim width (.dim batch (.dim channels .scalar))) ←
            Runtime.Autograd.TorchLean.swapAdjacentAtDepth (m := m) (α := α) 0 weighted
          Runtime.Autograd.TorchLean.reduceSum (m := m) (α := α) 0 tapsFirst :
          m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α)
            (.dim batch (.dim channels .scalar)))) }

/-- Pure evaluation of a batched depthwise weighted sum. -/
@[simp] theorem batchedDepthwiseWeightedSum_specFwd
    {batch width channels : Nat}
    (hBatch : 0 < batch) (hWidth : 0 < width) (hChannels : 0 < channels)
    {α : Type} [Context α]
    (values weights :
      _root_.Spec.Tensor α (.dim batch (.dim width (.dim channels .scalar)))) :
    (batchedDepthwiseWeightedSum batch width channels hBatch hWidth hChannels).specFwd
        (.cons values (.cons weights .nil)) =
      .dim (fun i => _root_.Spec.Tensor.reduceSumAuto
        (h := _root_.Spec.Shape.validAxisInstZeroAlt2 hWidth) 0
        (_root_.Spec.Tensor.mulSpec (_root_.Spec.get values i) (_root_.Spec.get weights i))) := by
  cases values
  cases weights
  rfl


/-- Form one outer product for every pair of vectors in a batch. -/
def batchedOuter (batch rows columns : Nat) :
    PrimOp
      [.dim batch (.dim rows .scalar), .dim batch (.dim columns .scalar)]
      (.dim batch (.dim rows (.dim columns .scalar))) :=
  { name := s!"batchedOuter({batch},{rows},{columns})"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons (.dim left) (.cons (.dim right) .nil) =>
          .dim fun i => _root_.Spec.outerProductSpec (α := α) (left i) (right i)
    program := fun {α} _ _ =>
      fun {m} _ _ => fun left right =>
        (do
          let columns' ← Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₂ := .dim batch (.dim rows (.dim 1 .scalar))) left
            (by simp [_root_.Spec.Shape.size])
          let rows' ← Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₂ := .dim batch (.dim 1 (.dim columns .scalar))) right
            (by simp [_root_.Spec.Shape.size])
          Runtime.Autograd.TorchLean.bmm (m := m) (α := α)
            (batch := batch) (mDim := rows) (nDim := 1) (pDim := columns)
            columns' rows' :
          m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α)
            (.dim batch (.dim rows (.dim columns .scalar))))) }

/-- Pure evaluation of pairwise batched outer products. -/
@[simp] theorem batchedOuter_specFwd {batch rows columns : Nat}
    {α : Type} [Context α]
    (left : _root_.Spec.Tensor α (.dim batch (.dim rows .scalar)))
    (right : _root_.Spec.Tensor α (.dim batch (.dim columns .scalar))) :
    (batchedOuter batch rows columns).specFwd (.cons left (.cons right .nil)) =
      .dim (fun i => _root_.Spec.outerProductSpec
        (_root_.Spec.get left i) (_root_.Spec.get right i)) := by
  cases left
  cases right
  rfl

/-- Scale every matrix row by the corresponding coordinate of a batched vector. -/
def batchedRowScale (batch rows columns : Nat) :
    PrimOp
      [.dim batch (.dim rows .scalar), .dim batch (.dim rows (.dim columns .scalar))]
      (.dim batch (.dim rows (.dim columns .scalar))) :=
  { name := s!"batchedRowScale({batch},{rows},{columns})"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons (.dim scales) (.cons (.dim matrices) .nil) =>
          .dim fun i => .dim fun row => .dim fun column => .scalar <|
            _root_.Spec.Tensor.vecGet (scales i) row *
              _root_.Spec.get2 (matrices i) row column
    program := fun {α} _ _ =>
      fun {m} _ _ => fun scales matrices =>
        (do
          let columns' ← Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₂ := .dim batch (.dim rows (.dim 1 .scalar))) scales
            (by simp [_root_.Spec.Shape.size])
          let expanded ← Runtime.Autograd.TorchLean.broadcastTo (m := m) (α := α)
            (s₂ := .dim batch (.dim rows (.dim columns .scalar)))
            (_root_.Spec.Shape.CanBroadcastTo.dim_eq
              (_root_.Spec.Shape.CanBroadcastTo.dim_eq
                (_root_.Spec.Shape.CanBroadcastTo.dim_1_to_n
                  (_root_.Spec.Shape.CanBroadcastTo.scalar_to_any .scalar)))) columns'
          Runtime.Autograd.TorchLean.mul (m := m) (α := α) expanded matrices :
          m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α)
            (.dim batch (.dim rows (.dim columns .scalar))))) }

/-- Pure evaluation of batched row scaling. -/
@[simp] theorem batchedRowScale_specFwd {batch rows columns : Nat}
    {α : Type} [Context α]
    (scales : _root_.Spec.Tensor α (.dim batch (.dim rows .scalar)))
    (matrices : _root_.Spec.Tensor α (.dim batch (.dim rows (.dim columns .scalar)))) :
    (batchedRowScale batch rows columns).specFwd (.cons scales (.cons matrices .nil)) =
      .dim (fun i => .dim (fun row => .dim (fun column => .scalar
        (_root_.Spec.Tensor.vecGet (_root_.Spec.get scales i) row *
          _root_.Spec.get2 (_root_.Spec.get matrices i) row column)))) := by
  cases scales
  cases matrices
  rfl

/-- Scale every tensor in a batch by its corresponding scalar coefficient.

The trailing tensor shape is unrestricted. This one primitive therefore covers vectors, matrices,
and higher-rank values without introducing rank-specific operation names. -/
def batchedScale (batch : Nat) (elementShape : Shape) :
    PrimOp [.dim batch .scalar, .dim batch elementShape] (.dim batch elementShape) :=
  { name := s!"batchedScale({batch})"
    specFwd := fun {_} _ xs =>
      match xs with
      | .cons (.dim coefficients) (.cons (.dim values) .nil) =>
          .dim fun i => _root_.Spec.Tensor.mulSpec
            (_root_.Spec.fill (_root_.Spec.Tensor.toScalar (coefficients i))
              elementShape) (values i)
    program := fun {α} _ _ =>
      fun {m} _ _ => fun coefficients values =>
        (do
          let expanded ← Runtime.Autograd.TorchLean.broadcastTo (m := m) (α := α)
            (s₂ := .dim batch elementShape)
            (_root_.Spec.Shape.CanBroadcastTo.dim_eq
              (_root_.Spec.Shape.CanBroadcastTo.scalar_to_any elementShape)) coefficients
          Runtime.Autograd.TorchLean.mul (m := m) (α := α) expanded values :
          m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α)
            (.dim batch elementShape))) }

/-- Pure evaluation of one scalar coefficient per batched tensor. -/
@[simp] theorem batchedScale_specFwd {batch : Nat} {elementShape : Shape}
    {α : Type} [Context α]
    (coefficients : _root_.Spec.Tensor α (.dim batch .scalar))
    (values : _root_.Spec.Tensor α (.dim batch elementShape)) :
    (batchedScale batch elementShape).specFwd (.cons coefficients (.cons values .nil)) =
      .dim (fun i => _root_.Spec.Tensor.mulSpec
        (_root_.Spec.fill (_root_.Spec.Tensor.toScalar (_root_.Spec.get coefficients i))
          elementShape)
        (_root_.Spec.get values i)) := by
  cases coefficients
  cases values
  rfl


/-- Multiply a row vector by a matrix. -/
def vecMat (rows columns : Nat) :
    PrimOp
      [.dim rows .scalar, .dim rows (.dim columns .scalar)]
      (.dim columns .scalar) :=
  { name := s!"vecMat({rows},{columns})"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons vector (.cons matrix .nil) => _root_.Spec.vecMatMulSpec (α := α) vector matrix
    program := fun {α} _ _ =>
      fun {m} _ _ => fun vector matrix =>
        (do
          let row ← Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₂ := .dim 1 (.dim rows .scalar)) vector (by simp [Spec.Shape.size])
          let product ← Runtime.Autograd.TorchLean.mm (m := m) (α := α)
            (mDim := 1) (nDim := rows) (pDim := columns) row matrix
          Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₂ := .dim columns .scalar) product (by simp [Spec.Shape.size]) :
          m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) (.dim columns .scalar))) }

/-- Multiply each row of a matrix by the corresponding vector coordinate. -/
def rowScale (rows columns : Nat) :
    PrimOp
      [.dim rows .scalar, .dim rows (.dim columns .scalar)]
      (.dim rows (.dim columns .scalar)) :=
  { name := s!"rowScale({rows},{columns})"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons scales (.cons matrix .nil) =>
          _root_.Spec.Tensor.dim fun row => _root_.Spec.Tensor.dim fun column =>
            _root_.Spec.Tensor.scalar <|
              _root_.Spec.Tensor.vecGet scales row * _root_.Spec.get2 matrix row column
    program := fun {α} _ _ =>
      fun {m} _ _ => fun scales matrix =>
        (do
          let column ← Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₂ := .dim rows (.dim 1 .scalar)) scales (by simp [Spec.Shape.size])
          let expanded ← Runtime.Autograd.TorchLean.broadcastTo (m := m) (α := α)
            (s₂ := .dim rows (.dim columns .scalar))
            (_root_.Spec.Shape.CanBroadcastTo.dim_eq
              (_root_.Spec.Shape.CanBroadcastTo.dim_1_to_n
                (_root_.Spec.Shape.CanBroadcastTo.scalar_to_any .scalar))) column
          Runtime.Autograd.TorchLean.mul (m := m) (α := α) expanded matrix :
          m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α)
            (.dim rows (.dim columns .scalar)))) }

/-- Form the outer product of two vectors. -/
def outer (rows columns : Nat) :
    PrimOp
      [.dim rows .scalar, .dim columns .scalar]
      (.dim rows (.dim columns .scalar)) :=
  { name := s!"outer({rows},{columns})"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons left (.cons right .nil) => _root_.Spec.outerProductSpec (α := α) left right
    program := fun {α} _ _ =>
      fun {m} _ _ => fun left right =>
        (do
          let leftColumn ← Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₂ := .dim rows (.dim 1 .scalar)) left (by simp [Spec.Shape.size])
          let rightRow ← Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₂ := .dim 1 (.dim columns .scalar)) right (by simp [Spec.Shape.size])
          Runtime.Autograd.TorchLean.mm (m := m) (α := α)
            (mDim := rows) (nDim := 1) (pDim := columns) leftColumn rightRow :
          m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α)
            (.dim rows (.dim columns .scalar)))) }

/-- Multiply a tensor by a scalar supplied as a graph input. -/
def scalarMul (s : Shape) : PrimOp [.scalar, s] s :=
  { name := "scalarMul"
    specFwd := fun {_α} _ xs =>
      match xs with
      | .cons (.scalar coefficient) (.cons input .nil) =>
          _root_.Spec.Tensor.mulSpec (_root_.Spec.fill coefficient s) input
    program := fun {α} _ _ =>
      fun {m} _ _ => fun scalar input =>
        (do
          let expanded ← Runtime.Autograd.TorchLean.broadcastTo (m := m) (α := α)
            (_root_.Spec.Shape.CanBroadcastTo.scalar_to_any s) scalar
          Runtime.Autograd.TorchLean.mul (m := m) (α := α) expanded input :
          m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) s)) }

/-- Scalar multiplication depends only on the value carried by its scalar-shaped input. -/
@[simp] theorem scalarMul_specFwd {α : Type} [Context α] {s : Shape}
    (coefficient : _root_.Spec.Tensor α .scalar) (input : _root_.Spec.Tensor α s) :
    (scalarMul s).specFwd (.cons coefficient (.cons input .nil)) =
      _root_.Spec.Tensor.mapSpec (fun value => coefficient.toScalar * value) input := by
  cases coefficient with
  | scalar value =>
      exact _root_.Spec.Tensor.mulSpec_fill_left value input


/-- Sum every scalar entry of a tensor. -/
def sum (s : Shape) : PrimOp [s] .scalar :=
  { name := "sum"
    specFwd := fun {_α} _ xs =>
      match xs with
      | .cons input .nil => .scalar (_root_.Spec.Tensor.sumSpec input)
    program := fun {α} _ _ =>
      fun {m} _ _ => fun input =>
        Runtime.Autograd.TorchLean.sum (m := m) (α := α) (s := s) input }


end PrimOp

end DAG
end GraphSpec
end NN
