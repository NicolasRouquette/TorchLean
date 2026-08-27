/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape.Reductions

@[expose] public section


namespace Spec
namespace Tensor

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-!
# Linear Algebra Helpers

Rank-polymorphic axis permutations, broadcasted matmul, and shape matching.
-/

/-- Swap adjacent tensor axes at `depth` and `depth + 1`. -/
def swapAdjacentAxes {β : Type} {shape : Shape} (tensor : Tensor β shape) (d : Nat) :
      Tensor β (shape.swapAdjacentAtDepth d) :=
      match d, shape, tensor with
      | 0, .dim m (.dim k rest), .dim g =>
        -- Swap axes 0 and 1 at this level.
        .dim fun j =>
          .dim fun i =>
            match g i with
            | .dim h => h j
      | d + 1, .dim m rest, .dim g =>
        -- Preserve this axis and recurse into the remaining shape.
        .dim fun i => swapAdjacentAxes (g i) d
      | d, .scalar, .scalar x =>
        -- A scalar has no axes to swap.
        by cases d <;> exact .scalar x
      | 0, .dim _ .scalar, .dim g =>
        -- A vector has no adjacent pair at this level.
        .dim g

/-- Apply adjacent-axis swaps while retaining the resulting shape in the return type. -/
def permuteByAdjacentSwaps {β : Type} {s : Shape} (tensor : Tensor β s) :
    (depths : List Nat) → Tensor β (s.applyAdjacentSwaps depths)
  | [] => tensor
  | depth :: depths =>
      permuteByAdjacentSwaps (swapAdjacentAxes tensor depth) depths

/-- Swapping at depth zero exchanges the two leading axes. -/
theorem swapAdjacentAxes_zero {β : Type} {m n : Nat} {s : Shape}
    (tensor : Tensor β (.dim m (.dim n s))) :
    swapAdjacentAxes tensor 0 =
      .dim (fun j => .dim (fun i => _root_.Spec.get (_root_.Spec.get tensor i) j)) := by
  cases tensor with
  | dim values =>
      apply congrArg Tensor.dim
      funext j
      apply congrArg Tensor.dim
      funext i
      cases hvalue : values i
      simp [hvalue, _root_.Spec.get]

namespace Internal

theorem rank_concat (left right : Shape) :
    (left.concat right).rank = left.rank + right.rank := by
  induction left with
  | scalar => simp [Shape.rank]
  | dim _ tail ih => simp [Shape.rank, ih, Nat.add_assoc]

theorem sameRankConcatRight (left right suffix : Shape) (same : Shape.SameRank left right) :
    Shape.SameRank (left.concat suffix) (right.concat suffix) :=
  ⟨by simp only [rank_concat, same.rank_eq]⟩

/-- Extend prefix-broadcast evidence across a fixed non-broadcasted tensor suffix. -/
def extendBroadcastSuffix {source target : Shape} (suffix : Shape)
    (broadcast : Shape.CanBroadcastTo source target) :
    Shape.CanBroadcastTo (source.concat suffix) (target.concat suffix) :=
  match broadcast with
  | .scalar => .refl suffix
  | @Shape.CanBroadcastTo.dim_eq _ source target same tail =>
      letI : Shape.SameRank (source.concat suffix) (target.concat suffix) :=
        sameRankConcatRight source target suffix same
      .dim_eq (extendBroadcastSuffix suffix tail)
  | @Shape.CanBroadcastTo.dim_1_to_n _ source target same tail =>
      letI : Shape.SameRank (source.concat suffix) (target.concat suffix) :=
        sameRankConcatRight source target suffix same
      .dim_1_to_n (extendBroadcastSuffix suffix tail)
  | .expand_dims tail => .expand_dims (extendBroadcastSuffix suffix tail)

@[simp] theorem broadcastTo_refl {α : Type} [Inhabited α] {shape : Shape}
    (tensor : Tensor α shape) :
    broadcastTo (Shape.CanBroadcastTo.refl shape) tensor = tensor := by
  induction shape with
  | scalar => rfl
  | dim count rest ih =>
      cases tensor with
      | dim values =>
          simp only [Shape.CanBroadcastTo.refl, broadcastTo]
          congr
          funext index
          exact ih (values index)

@[simp] theorem broadcastTo_extendBroadcastSuffix_refl
    {α : Type} [Inhabited α] (batch suffix : Shape)
    (tensor : Tensor α (batch.concat suffix)) :
    broadcastTo (extendBroadcastSuffix suffix (Shape.CanBroadcastTo.refl batch)) tensor =
      tensor := by
  induction batch with
  | scalar => exact broadcastTo_refl tensor
  | dim count rest ih =>
      cases tensor with
      | dim values =>
          simp only [Shape.CanBroadcastTo.refl, extendBroadcastSuffix, broadcastTo]
          congr
          funext index
          exact ih (values index)

@[simp] theorem broadcastTo_extendBroadcastSuffix_scalarTo_dim
    {α : Type} [Inhabited α] (count : Nat) (suffix : Shape)
    (tensor : Tensor α suffix) :
    broadcastTo
        (extendBroadcastSuffix suffix (Shape.CanBroadcastTo.scalarTo [count])) tensor =
      .dim (fun _ => tensor) := by
  change (Tensor.dim fun _ => broadcastTo (Shape.CanBroadcastTo.refl suffix) tensor) = _
  congr
  funext index
  exact broadcastTo_refl tensor

theorem flattenBatchMatrix_size (batch : Shape) (m n : Nat) :
    (batch.concat [m, n]).size = Shape.size [batch.size, m, n] := by
  simp [Shape.size_concat, Shape.size]

def flattenBatchMatrix {α : Type} [Inhabited α] {batch : Shape} {m n : Nat}
    (tensor : Tensor α (batch.concat [m, n])) : Tensor α [batch.size, m, n] :=
  reshapeSpec tensor (flattenBatchMatrix_size batch m n)

def restoreBatchMatrix {α : Type} [Inhabited α] {batch : Shape} {m n : Nat}
    (tensor : Tensor α [batch.size, m, n]) : Tensor α (batch.concat [m, n]) :=
  reshapeSpec tensor (flattenBatchMatrix_size batch m n).symm

def bmmLikeSpec {α : Type} [Add α] [Mul α] [Zero α]
    {batch m n p : Nat} (A : Tensor α [batch, m, n]) (B : Tensor α [batch, n, p]) :
    Tensor α [batch, m, p] :=
  match A, B with
  | .dim fA, .dim fB => .dim fun i => matMulSpec (fA i) (fB i)

def matmulCommonBatchSpec {α : Type} [Inhabited α] [Add α] [Mul α] [Zero α]
    {batch : Shape} {m n p : Nat}
    (A : Tensor α (batch.concat [m, n])) (B : Tensor α (batch.concat [n, p])) :
    Tensor α (batch.concat [m, p]) :=
  match batch, A, B with
  | .scalar, A, B => matMulSpec A B
  | .dim _ rest, .dim left, .dim right =>
      .dim fun index => matmulCommonBatchSpec (left index) (right index)

@[simp] theorem matmulCommonBatchSpec_scalar {α : Type}
    [Inhabited α] [Add α] [Mul α] [Zero α] {m n p : Nat}
    (left : Tensor α [m, n]) (right : Tensor α [n, p]) :
    matmulCommonBatchSpec (batch := .scalar) left right = matMulSpec left right := by
  rfl

@[simp] theorem matmulCommonBatchSpec_dim {α : Type}
    [Inhabited α] [Add α] [Mul α] [Zero α]
    {count m n p : Nat} {rest : Shape}
    (left : Fin count → Tensor α (rest.concat [m, n]))
    (right : Fin count → Tensor α (rest.concat [n, p])) :
    matmulCommonBatchSpec (batch := .dim count rest) (.dim left) (.dim right) =
      .dim (fun index => matmulCommonBatchSpec (left index) (right index)) := by
  rfl

end Internal

/-- Matrix-rank matmul with explicit broadcasting of both batch prefixes.

`A` has shape `batchA ++ [m, n]`, `B` has shape `batchB ++ [n, p]`, and both batch
prefixes broadcast to `batch`. The result has shape `batch ++ [m, p]`. -/
def matmulSpec {α : Type} [Inhabited α] [Add α] [Mul α] [Zero α]
    {batchA batchB batch : Shape} {m n p : Nat}
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch)
    (A : Tensor α (batchA.concat [m, n])) (B : Tensor α (batchB.concat [n, p])) :
    Tensor α (batch.concat [m, p]) :=
  let commonA := broadcastTo (Internal.extendBroadcastSuffix [m, n] broadcastA) A
  let commonB := broadcastTo (Internal.extendBroadcastSuffix [n, p] broadcastB) B
  Internal.matmulCommonBatchSpec commonA commonB

/-- Reverse-mode derivatives for matrix-rank matmul with broadcasted batch prefixes. -/
def matmulBackwardSpec {α : Type} [Inhabited α] [Add α] [Mul α] [Zero α]
    {batchA batchB batch : Shape} {m n p : Nat}
    (broadcastA : Shape.CanBroadcastTo batchA batch)
    (broadcastB : Shape.CanBroadcastTo batchB batch)
    (A : Tensor α (batchA.concat [m, n])) (B : Tensor α (batchB.concat [n, p]))
    (dC : Tensor α (batch.concat [m, p])) :
    Tensor α (batchA.concat [m, n]) × Tensor α (batchB.concat [n, p]) :=
  let commonA := broadcastTo (Internal.extendBroadcastSuffix [m, n] broadcastA) A
  let commonB := broadcastTo (Internal.extendBroadcastSuffix [n, p] broadcastB) B
  let commonBT : Tensor α (batch.concat [p, n]) := by
    simpa using swapAdjacentAxes commonB batch.rank
  let commonAT : Tensor α (batch.concat [n, m]) := by
    simpa using swapAdjacentAxes commonA batch.rank
  let dCommonA := Internal.matmulCommonBatchSpec dC commonBT
  let dCommonB := Internal.matmulCommonBatchSpec commonAT dC
  let dA :=
    reduceFromBroadcastTo (Internal.extendBroadcastSuffix [m, n] broadcastA) dCommonA
  let dB :=
    reduceFromBroadcastTo (Internal.extendBroadcastSuffix [n, p] broadcastB) dCommonB
  (dA, dB)

end Tensor
end Spec
