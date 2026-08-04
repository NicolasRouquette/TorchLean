/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Correctness.Eval.Elementwise

/-!
# Shape-Changing IR Evaluation

Local semantics for the shape-oriented IR nodes emitted by PyTorch/ONNX import paths.  These facts
pin the executable IR evaluator to the corresponding typed tensor operations.
-/

@[expose] public section

namespace NN.Verification.TorchLean.Proved

open _root_.Spec
open _root_.Spec.Tensor
open NN.IR

namespace Correctness

namespace IRStep

/-- Shape-changing unary operations with statically typed output tensors. -/
inductive ShapeOperation : Shape → Shape → Type where
  /-- Reshape between shapes with equal element counts. -/
  | reshape (inShape outShape : Shape) (hsize : Spec.Shape.size inShape = Spec.Shape.size outShape) :
      ShapeOperation inShape outShape
  /-- Flatten a tensor to one dimension. -/
  | flatten (s : Shape) : ShapeOperation s (.dim (Spec.Shape.size s) .scalar)
  /-- Broadcast to a compatible target shape. -/
  | broadcastTo (s₁ s₂ : Shape) (witness : Shape.CanBroadcastTo s₁ s₂)
      (accepted : OpContracts.mkCanBroadcastTo? s₁ s₂ = some witness) : ShapeOperation s₁ s₂
  /-- Sum every coordinate to a scalar. -/
  | sum (s : Shape) : ShapeOperation s .scalar

/-- IR opcode represented by a typed shape operation. -/
def ShapeOperation.toOpKind {inShape outShape : Shape} :
    ShapeOperation inShape outShape → OpKind
  | .reshape _ _ _ => .reshape inShape outShape
  | .flatten _ => .flatten inShape
  | .broadcastTo _ _ _ _ => .broadcastTo inShape outShape
  | .sum _ => .sum

/-- Typed denotation of a shape operation. -/
def ShapeOperation.denote
    {α : Type} [Context α] {inShape outShape : Shape}
    (op : ShapeOperation inShape outShape) (x : Tensor α inShape) : Tensor α outShape :=
  match op with
  | .reshape _ _ hsize => Tensor.reshapeSpec (α := α) x hsize
  | .flatten _ => Tensor.flattenSpec (α := α) x
  | .broadcastTo _ _ witness _ => Tensor.broadcastTo (α := α) witness x
  | .sum _ => Tensor.scalar (Tensor.sumSpec (α := α) x)

/-- Evaluate any typed shape operation in its canonical two-node graph. -/
theorem evalAt_shapeOperation_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {inShape outShape : Shape} (op : ShapeOperation inShape outShape)
    (x : Tensor α inShape) :
    Graph.evalAt (α := α) (g := unaryGraphOut op.toOpKind inShape outShape)
        (payload := {})
        (input := DVal.mk (α := α) inShape x)
        (vals := #[DVal.mk (α := α) inShape x]) (i := 1)
      =
      Except.ok (DVal.mk (α := α) outShape (op.denote x)) := by
  cases op <;>
    simp_all [ShapeOperation.toOpKind, ShapeOperation.denote, Graph.evalAt, Graph.evalNode, unaryGraphOut,
      unaryNodeOut, Graph.getNode, Graph.getNode?, Graph.expectShape, Bind.bind, Except.bind,
      Pure.pure, Except.pure]

/-- Local IR semantics for `reshape` when the element counts match. -/
theorem evalAt_reshape_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {inShape outShape : Shape} (x : Tensor α inShape)
    (hsize : Spec.Shape.size inShape = Spec.Shape.size outShape) :
    Graph.evalAt (α := α) (g := unaryGraphOut (.reshape inShape outShape) inShape outShape)
        (payload := {})
        (input := DVal.mk (α := α) inShape x)
        (vals := #[DVal.mk (α := α) inShape x]) (i := 1)
      =
      Except.ok
        (DVal.mk (α := α) outShape
          (Tensor.reshapeSpec (α := α) (s₁ := inShape) (s₂ := outShape) x hsize)) := by
  exact evalAt_shapeOperation_eq (.reshape inShape outShape hsize) x

/-- Local IR semantics for `flatten`. -/
theorem evalAt_flatten_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraphOut (.flatten s) s (.dim (Spec.Shape.size s) .scalar))
        (payload := {})
        (input := DVal.mk (α := α) s x)
        (vals := #[DVal.mk (α := α) s x]) (i := 1)
      =
      Except.ok
        (DVal.mk (α := α) (.dim (Spec.Shape.size s) .scalar)
          (Tensor.flattenSpec (α := α) (s := s) x)) := by
  exact evalAt_shapeOperation_eq (.flatten s) x

/-- Local IR semantics for `broadcastTo` when the broadcast witness is accepted by the contract. -/
theorem evalAt_broadcastTo_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s₁ s₂ : Shape} (x : Tensor α s₁) (cb : Shape.CanBroadcastTo s₁ s₂)
    (hcb : OpContracts.mkCanBroadcastTo? s₁ s₂ = some cb) :
    Graph.evalAt (α := α) (g := unaryGraphOut (.broadcastTo s₁ s₂) s₁ s₂)
        (payload := {})
        (input := DVal.mk (α := α) s₁ x)
        (vals := #[DVal.mk (α := α) s₁ x]) (i := 1)
      =
      Except.ok
        (DVal.mk (α := α) s₂
          (Tensor.broadcastTo (α := α) (s₁ := s₁) (s₂ := s₂) cb x)) := by
  exact evalAt_shapeOperation_eq (.broadcastTo s₁ s₂ cb hcb) x

/-- Local IR semantics for reduction to a scalar sum. -/
theorem evalAt_sum_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraphOut .sum s .scalar)
        (payload := {})
        (input := DVal.mk (α := α) s x)
        (vals := #[DVal.mk (α := α) s x]) (i := 1)
      =
      Except.ok
        (DVal.mk (α := α) .scalar (Tensor.scalar (Tensor.sumSpec (α := α) x))) := by
  exact evalAt_shapeOperation_eq (.sum s) x

end IRStep

end Correctness

end NN.Verification.TorchLean.Proved
