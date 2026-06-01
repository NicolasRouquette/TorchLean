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

/-- Local IR semantics for `reshape` when the element counts match. -/
theorem evalAt_reshape_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {inShape outShape : Shape} (x : Tensor α inShape)
    (hsize : Shape.size inShape = Shape.size outShape) :
    Graph.evalAt (α := α) (g := unaryGraphOut (.reshape inShape outShape) inShape outShape)
        (payload := {})
        (input := DVal.mk (α := α) inShape x)
        (vals := #[DVal.mk (α := α) inShape x]) (i := 1)
      =
      Except.ok
        (DVal.mk (α := α) outShape
          (Tensor.reshapeSpec (α := α) (s₁ := inShape) (s₂ := outShape) x hsize)) := by
  simp [Graph.evalAt, unaryGraphOut, unaryNodeOut, Graph.getNode, Graph.getNode?,
    Graph.expectShape, hsize, Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Local IR semantics for `flatten`. -/
theorem evalAt_flatten_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraphOut (.flatten s) s (.dim (Shape.size s) .scalar))
        (payload := {})
        (input := DVal.mk (α := α) s x)
        (vals := #[DVal.mk (α := α) s x]) (i := 1)
      =
      Except.ok
        (DVal.mk (α := α) (.dim (Shape.size s) .scalar)
          (Tensor.flattenSpec (α := α) (s := s) x)) := by
  simp [Graph.evalAt, unaryGraphOut, unaryNodeOut, Graph.getNode, Graph.getNode?,
    Graph.expectShape, Bind.bind, Except.bind, Pure.pure, Except.pure]

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
  simp [Graph.evalAt, unaryGraphOut, unaryNodeOut, Graph.getNode, Graph.getNode?,
    Graph.expectShape, hcb, Bind.bind, Except.bind, Pure.pure, Except.pure]

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
  simp [Graph.evalAt, unaryGraphOut, unaryNodeOut, Graph.getNode, Graph.getNode?,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

end IRStep

end Correctness

end NN.Verification.TorchLean.Proved
