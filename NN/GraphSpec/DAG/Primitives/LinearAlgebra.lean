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

open _root_.Spec
open Spec.Tensor
open _root_.TorchLean.Tensor

namespace PrimOp

/-- Multiply matrices after broadcasting their batch prefixes to a common shape. -/
def matmul (batchA batchB batch : Shape) (mDim nDim pDim : Nat)
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch) :
    PrimOp
      [batchA.concat [mDim, nDim], batchB.concat [nDim, pDim]]
      (batch.concat [mDim, pDim]) :=
  { name := s!"matmul({mDim},{nDim},{pDim})"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons a (.cons b .nil) =>
          _root_.Spec.Tensor.matmulSpec broadcastA broadcastB a b
    program := fun {α} _ _ =>
      fun {m} _ _ => fun a b => by
        letI : Shape.BroadcastTo batchA batch := ⟨broadcastA⟩
        letI : Shape.BroadcastTo batchB batch := ⟨broadcastB⟩
        exact
        Runtime.Autograd.TorchLean.matmul (m := m) (α := α)
          (batchA := batchA) (batchB := batchB) (batch := batch)
          (mDim := mDim) (nDim := nDim) (pDim := pDim) a b }

/-- Pure evaluation of matrix multiplication with explicitly broadcast batch prefixes. -/
@[simp] theorem matmul_specFwd {batchA batchB batch : Shape} {mDim nDim pDim : Nat}
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch)
    {α : Type} [Context α]
    (left : _root_.Spec.Tensor α (batchA.concat [mDim, nDim]))
    (right : _root_.Spec.Tensor α (batchB.concat [nDim, pDim])) :
    (matmul batchA batchB batch mDim nDim pDim broadcastA broadcastB).specFwd
        (.cons left (.cons right .nil)) =
      _root_.Spec.Tensor.matmulSpec broadcastA broadcastB left right := by
  rfl

namespace Internal

/-- Multiply vectors and matrices pointwise over a common batch shape. -/
def vecMatCommonBatchSpec {α : Type} [Context α] {rows columns : Nat} :
    (batch : Shape) →
      _root_.Spec.Tensor α (batch.concat [rows]) →
      _root_.Spec.Tensor α (batch.concat [rows, columns]) →
      _root_.Spec.Tensor α (batch.concat [columns])
  | .scalar, vector, matrix => _root_.Spec.vecMatMulSpec vector matrix
  | .dim _ rest, .dim vectors, .dim matrices =>
      .dim fun index => vecMatCommonBatchSpec rest (vectors index) (matrices index)

@[simp] theorem vecMatCommonBatchSpec_scalar {α : Type} [Context α]
    {rows columns : Nat} (vector : _root_.Spec.Tensor α [rows])
    (matrix : _root_.Spec.Tensor α [rows, columns]) :
    vecMatCommonBatchSpec .scalar vector matrix =
      _root_.Spec.vecMatMulSpec vector matrix := by
  rfl

@[simp] theorem vecMatCommonBatchSpec_dim {α : Type} [Context α]
    {count rows columns : Nat} {rest : Shape}
    (vectors : Fin count → _root_.Spec.Tensor α (rest.concat [rows]))
    (matrices : Fin count → _root_.Spec.Tensor α (rest.concat [rows, columns])) :
    vecMatCommonBatchSpec (.dim count rest) (.dim vectors) (.dim matrices) =
      .dim (fun index => vecMatCommonBatchSpec rest (vectors index) (matrices index)) := by
  rfl

end Internal

/-- Multiply vectors by matrices after broadcasting their batch prefixes to a common shape. -/
def broadcastVecMat (vectorBatch matrixBatch batch : Shape) (rows columns : Nat)
    (broadcastVector : Shape.CanBroadcastTo vectorBatch batch)
    (broadcastMatrix : Shape.CanBroadcastTo matrixBatch batch) :
    PrimOp
      [vectorBatch.concat [rows], matrixBatch.concat [rows, columns]]
      (batch.concat [columns]) :=
  { name := s!"vecMat({rows},{columns})"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons vector (.cons matrix .nil) =>
          let commonVector := _root_.Spec.Tensor.broadcastTo
            (_root_.Spec.Tensor.Internal.extendBroadcastSuffix [rows] broadcastVector) vector
          let commonMatrix := _root_.Spec.Tensor.broadcastTo
            (_root_.Spec.Tensor.Internal.extendBroadcastSuffix [rows, columns] broadcastMatrix)
            matrix
          Internal.vecMatCommonBatchSpec batch commonVector commonMatrix
    program := fun {α} _ _ =>
      fun {m} _ _ => fun vector matrix => by
        letI : Shape.BroadcastTo vectorBatch batch := ⟨broadcastVector⟩
        letI : Shape.BroadcastTo matrixBatch batch := ⟨broadcastMatrix⟩
        exact (do
          let rowVector ← Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₂ := vectorBatch.concat [1, rows]) vector
            (by simp [_root_.Spec.Shape.size_concat, _root_.Spec.Shape.size])
          let product ← Runtime.Autograd.TorchLean.matmul (m := m) (α := α)
            (batchA := vectorBatch) (batchB := matrixBatch) (batch := batch)
            (mDim := 1) (nDim := rows) (pDim := columns)
            rowVector matrix
          Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₂ := batch.concat [columns]) product
            (by simp [_root_.Spec.Shape.size_concat, _root_.Spec.Shape.size]) :
          m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α)
            (batch.concat [columns]))) }

/-- Pure evaluation of broadcasted vector–matrix multiplication. -/
@[simp] theorem broadcastVecMat_specFwd
    {vectorBatch matrixBatch batch : Shape} {rows columns : Nat}
    (broadcastVector : Shape.CanBroadcastTo vectorBatch batch)
    (broadcastMatrix : Shape.CanBroadcastTo matrixBatch batch)
    {α : Type} [Context α]
    (vector : _root_.Spec.Tensor α (vectorBatch.concat [rows]))
    (matrix : _root_.Spec.Tensor α (matrixBatch.concat [rows, columns])) :
    (broadcastVecMat vectorBatch matrixBatch batch rows columns
      broadcastVector broadcastMatrix).specFwd (.cons vector (.cons matrix .nil)) =
      Internal.vecMatCommonBatchSpec batch
        (_root_.Spec.Tensor.broadcastTo
          (_root_.Spec.Tensor.Internal.extendBroadcastSuffix [rows] broadcastVector) vector)
        (_root_.Spec.Tensor.broadcastTo
          (_root_.Spec.Tensor.Internal.extendBroadcastSuffix [rows, columns] broadcastMatrix)
          matrix) := by
  rfl

/-- Dot one shared vector with every vector in a batch. -/
def batchedSharedDot (batch width : Nat) :
    PrimOp [[width], [batch, width]] [batch] :=
  { name := s!"batchedSharedDot({batch},{width})"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons v (.cons (.dim vectors) .nil) =>
          .dim fun i => .scalar (_root_.Spec.Tensor.dotSpec (α := α) v (vectors i))
    program := fun {α} _ _ =>
      fun {m} _ _ => fun vector vectors =>
        (do
          let column ← Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₂ := [width, 1]) vector
            (by simp [_root_.Spec.Shape.size])
          Runtime.Autograd.TorchLean.matmul (m := m) (α := α)
            (batchA := .scalar) (batchB := .scalar) (batch := .scalar)
            (mDim := batch) (nDim := width) (pDim := 1) vectors column >>= fun result =>
          Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₂ := [batch]) result (by simp [_root_.Spec.Shape.size]) :
          m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) [batch])) }

/-- Apply a depthwise weighted reduction independently to every element of a batch.

Both inputs use the layout `batch × width × channels`.  For each batch index and channel, the
result is the sum over `width` of the pointwise product.  The executable program implements that
equation with elementwise multiplication, an axis swap, and a leading-axis reduction; exposing the
operation here avoids making every caller reproduce that layout plumbing.
-/
def batchedDepthwiseWeightedSum (batch width channels : Nat)
    (hBatch : 0 < batch) (hWidth : 0 < width) (hChannels : 0 < channels) :
    PrimOp
      [[batch, width, channels], [batch, width, channels]]
      [batch, channels] :=
  letI : NeZero batch := ⟨Nat.ne_of_gt hBatch⟩
  letI : NeZero width := ⟨Nat.ne_of_gt hWidth⟩
  letI : NeZero channels := ⟨Nat.ne_of_gt hChannels⟩
  letI : _root_.Spec.Shape.HasNonemptyAxis 0
      [width, channels] :=
    _root_.Spec.Shape.hasNonemptyAxisZeroOfPos hWidth
  letI : _root_.Spec.Shape.HasNonemptyAxis 0
      [width, batch, channels] :=
    _root_.Spec.Shape.hasNonemptyAxisZeroOfPos hWidth
  { name := s!"batchedDepthwiseWeightedSum({batch},{width},{channels})"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons (.dim values) (.cons (.dim weights) .nil) =>
          .dim fun i => _root_.Spec.Tensor.reduceSum (α := α) 0
            (_root_.Spec.Tensor.mulSpec (values i) (weights i))
            (_root_.Spec.Shape.hasNonemptyAxisZeroOfPos hWidth).proof
    program := fun {α} _ _ =>
      fun {m} _ _ => fun values weights =>
        (do
          let weighted ← Runtime.Autograd.TorchLean.mul (m := m) (α := α)
            (s := [batch, width, channels]) values weights
          let tapsFirst : Runtime.Autograd.TorchLean.RefTy (m := m) (α := α)
              [width, batch, channels] ←
            Runtime.Autograd.TorchLean.swapAdjacentAtDepth (m := m) (α := α) 0 weighted
          Runtime.Autograd.TorchLean.reduceSum (m := m) (α := α) 0 tapsFirst :
          m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α)
            [batch, channels])) }

/-- Pure evaluation of a batched depthwise weighted sum. -/
@[simp] theorem batchedDepthwiseWeightedSum_specFwd
    {batch width channels : Nat}
    (hBatch : 0 < batch) (hWidth : 0 < width) (hChannels : 0 < channels)
    {α : Type} [Context α]
    (values weights :
      _root_.Spec.Tensor α [batch, width, channels]) :
    (batchedDepthwiseWeightedSum batch width channels hBatch hWidth hChannels).specFwd
        (.cons values (.cons weights .nil)) =
      .dim (fun i => _root_.Spec.Tensor.reduceSum 0
        (_root_.Spec.Tensor.mulSpec (_root_.Spec.get values i) (_root_.Spec.get weights i))
        (_root_.Spec.Shape.hasNonemptyAxisZeroOfPos hWidth).proof) := by
  cases values
  cases weights
  rfl


/-- Form one outer product for every pair of vectors in a batch. -/
def batchedOuter (batch rows columns : Nat) :
    PrimOp
      [[batch, rows], [batch, columns]]
      [batch, rows, columns] :=
  { name := s!"batchedOuter({batch},{rows},{columns})"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons (.dim left) (.cons (.dim right) .nil) =>
          .dim fun i => _root_.Spec.outerProductSpec (α := α) (left i) (right i)
    program := fun {α} _ _ =>
      fun {m} _ _ => fun left right =>
        (do
          let columns' ← Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₂ := [batch, rows, 1]) left
            (by simp [_root_.Spec.Shape.size])
          let rows' ← Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₂ := [batch, 1, columns]) right
            (by simp [_root_.Spec.Shape.size])
          Runtime.Autograd.TorchLean.matmul (m := m) (α := α)
            (batchA := [batch]) (batchB := [batch]) (batch := [batch])
            (mDim := rows) (nDim := 1) (pDim := columns)
            columns' rows' :
          m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α)
            [batch, rows, columns])) }

/-- Pure evaluation of pairwise batched outer products. -/
@[simp] theorem batchedOuter_specFwd {batch rows columns : Nat}
    {α : Type} [Context α]
    (left : _root_.Spec.Tensor α [batch, rows])
    (right : _root_.Spec.Tensor α [batch, columns]) :
    (batchedOuter batch rows columns).specFwd (.cons left (.cons right .nil)) =
      .dim (fun i => _root_.Spec.outerProductSpec
        (_root_.Spec.get left i) (_root_.Spec.get right i)) := by
  cases left
  cases right
  rfl

/-- Scale every matrix row by the corresponding coordinate of a batched vector. -/
def batchedRowScale (batch rows columns : Nat) :
    PrimOp [[batch, rows], [batch, rows, columns]] [batch, rows, columns] :=
  { name := s!"batchedRowScale({batch},{rows},{columns})"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons (.dim scales) (.cons (.dim matrices) .nil) =>
          .dim fun i => .dim fun row => .dim fun column => .scalar <|
            _root_.Spec.Tensor.getScalar (scales i) row *
              _root_.Spec.get2 (matrices i) row column
    program := fun {α} _ _ =>
      fun {m} _ _ => fun scales matrices =>
        (do
          let columns' ← Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₂ := [batch, rows, 1]) scales
            (by simp [_root_.Spec.Shape.size])
          let expanded ← Runtime.Autograd.TorchLean.broadcastTo (m := m) (α := α)
            (s₂ := [batch, rows, columns])
            (_root_.Spec.Shape.CanBroadcastTo.dim_eq
              (_root_.Spec.Shape.CanBroadcastTo.dim_eq
                (_root_.Spec.Shape.CanBroadcastTo.dim_1_to_n
                  _root_.Spec.Shape.CanBroadcastTo.scalar))) columns'
          Runtime.Autograd.TorchLean.mul (m := m) (α := α) expanded matrices :
          m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α)
            [batch, rows, columns])) }

/-- Pure evaluation of batched row scaling. -/
@[simp] theorem batchedRowScale_specFwd {batch rows columns : Nat}
    {α : Type} [Context α]
    (scales : _root_.Spec.Tensor α [batch, rows])
    (matrices : _root_.Spec.Tensor α [batch, rows, columns]) :
    (batchedRowScale batch rows columns).specFwd (.cons scales (.cons matrices .nil)) =
      .dim (fun i => .dim (fun row => .dim (fun column => .scalar
        (_root_.Spec.Tensor.getScalar (_root_.Spec.get scales i) row *
          _root_.Spec.get2 (_root_.Spec.get matrices i) row column)))) := by
  cases scales
  cases matrices
  rfl

/-- Scale every tensor in a batch by its corresponding scalar coefficient.

The trailing tensor shape is unrestricted. This one primitive therefore covers vectors, matrices,
and higher-rank values without introducing rank-specific operation names. -/
def batchedScale (batch : Nat) (elementShape : Shape) :
    PrimOp [[batch], .dim batch elementShape] (.dim batch elementShape) :=
  { name := s!"batchedScale({batch})"
    specFwd := fun {_} _ xs =>
      match xs with
      | .cons (.dim coefficients) (.cons (.dim values) .nil) =>
          .dim fun i => _root_.Spec.Tensor.mulSpec
            (_root_.Spec.fill (_root_.Spec.Tensor.item (coefficients i))
              elementShape) (values i)
    program := fun {α} _ _ =>
      fun {m} _ _ => fun coefficients values =>
        (do
          let coefficientShape : Shape :=
            .dim batch (_root_.Spec.Shape.singletonAxes elementShape)
          let coefficients' ← Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₁ := [batch]) (s₂ := coefficientShape) coefficients (by
              simp [coefficientShape, _root_.Spec.Shape.size])
          let _ : _root_.Spec.Shape.SameRank
              (_root_.Spec.Shape.singletonAxes elementShape) elementShape :=
            ⟨_root_.Spec.Shape.rank_singletonAxes elementShape⟩
          let expanded ← Runtime.Autograd.TorchLean.broadcastTo (m := m) (α := α)
            (s₁ := coefficientShape) (s₂ := .dim batch elementShape)
            (_root_.Spec.Shape.CanBroadcastTo.dim_eq
              (_root_.Spec.Shape.CanBroadcastTo.singletonAxes elementShape)) coefficients'
          Runtime.Autograd.TorchLean.mul (m := m) (α := α) expanded values :
          m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α)
            (.dim batch elementShape))) }

/-- Pure evaluation of one scalar coefficient per batched tensor. -/
@[simp] theorem batchedScale_specFwd {batch : Nat} {elementShape : Shape}
    {α : Type} [Context α]
    (coefficients : _root_.Spec.Tensor α [batch])
    (values : _root_.Spec.Tensor α (.dim batch elementShape)) :
    (batchedScale batch elementShape).specFwd (.cons coefficients (.cons values .nil)) =
      .dim (fun i => _root_.Spec.Tensor.mulSpec
        (_root_.Spec.fill (_root_.Spec.Tensor.item (_root_.Spec.get coefficients i))
          elementShape)
        (_root_.Spec.get values i)) := by
  cases coefficients
  cases values
  rfl


/-- Multiply a row vector by a matrix. -/
def vecMat (rows columns : Nat) :
    PrimOp
      [[rows], [rows, columns]]
      [columns] :=
  { name := s!"vecMat({rows},{columns})"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons v (.cons mtx .nil) => _root_.Spec.vecMatMulSpec (α := α) v mtx
    program := fun {α} _ _ =>
      fun {m} _ _ => fun vector matrix =>
        (do
          let row ← Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₂ := [1, rows]) vector (by simp [Spec.Shape.size])
          let product ← Runtime.Autograd.TorchLean.matmul (m := m) (α := α)
            (batchA := .scalar) (batchB := .scalar) (batch := .scalar)
            (mDim := 1) (nDim := rows) (pDim := columns) row matrix
          Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₂ := [columns]) product (by simp [Spec.Shape.size]) :
          m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) [columns])) }

/-- Multiply each row of a matrix by the corresponding vector coordinate. -/
def rowScale (rows columns : Nat) :
    PrimOp
      [[rows], [rows, columns]]
      [rows, columns] :=
  { name := s!"rowScale({rows},{columns})"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons scales (.cons mtx .nil) =>
          _root_.Spec.Tensor.dim fun row => _root_.Spec.Tensor.dim fun column =>
            _root_.Spec.Tensor.scalar <|
              _root_.Spec.Tensor.getScalar scales row * _root_.Spec.get2 mtx row column
    program := fun {α} _ _ =>
      fun {m} _ _ => fun scales matrix =>
        (do
          let column ← Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₂ := [rows, 1]) scales (by simp [Spec.Shape.size])
          let expanded ← Runtime.Autograd.TorchLean.broadcastTo (m := m) (α := α)
            (s₂ := [rows, columns])
            (_root_.Spec.Shape.CanBroadcastTo.dim_eq
              (_root_.Spec.Shape.CanBroadcastTo.dim_1_to_n
                _root_.Spec.Shape.CanBroadcastTo.scalar)) column
          Runtime.Autograd.TorchLean.mul (m := m) (α := α) expanded matrix :
          m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α)
            [rows, columns])) }

/-- Form the outer product of two vectors. -/
def outer (rows columns : Nat) :
    PrimOp
      [[rows], [columns]]
      [rows, columns] :=
  { name := s!"outer({rows},{columns})"
    specFwd := fun {α} _ xs =>
      match xs with
      | .cons left (.cons right .nil) => _root_.Spec.outerProductSpec (α := α) left right
    program := fun {α} _ _ =>
      fun {m} _ _ => fun left right =>
        (do
          let leftColumn ← Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₂ := [rows, 1]) left (by simp [Spec.Shape.size])
          let rightRow ← Runtime.Autograd.TorchLean.reshape (m := m) (α := α)
            (s₂ := [1, columns]) right (by simp [Spec.Shape.size])
          Runtime.Autograd.TorchLean.matmul (m := m) (α := α)
            (batchA := .scalar) (batchB := .scalar) (batch := .scalar)
            (mDim := rows) (nDim := 1) (pDim := columns) leftColumn rightRow :
          m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α)
            [rows, columns])) }

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
            (_root_.Spec.Shape.CanBroadcastTo.scalarTo s) scalar
          Runtime.Autograd.TorchLean.mul (m := m) (α := α) expanded input :
          m (Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) s)) }

/-- Scalar multiplication depends only on the value carried by its scalar-shaped input. -/
@[simp] theorem scalarMul_specFwd {α : Type} [Context α] {s : Shape}
    (coefficient : _root_.Spec.Tensor α .scalar) (input : _root_.Spec.Tensor α s) :
    (scalarMul s).specFwd (.cons coefficient (.cons input .nil)) =
      _root_.Spec.Tensor.mapSpec (fun value => coefficient.item * value) input := by
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
