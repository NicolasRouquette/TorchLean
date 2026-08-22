/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Correctness.Eval.Permutation

/-!
# Softmax IR Evaluation

The IR evaluator gives every in-bounds softmax axis the canonical axis-indexed tensor semantics.
-/

@[expose] public section

namespace NN.Verification.TorchLean.Proved

open _root_.Spec
open _root_.Spec.Tensor
open NN.IR

namespace Correctness

namespace IRStep

/-- Local IR semantics for stable last-axis softmax with a hard Boolean mask. -/
theorem evalAt_hardMaskedSoftmax_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (scores : Tensor α s) (allowed : Tensor Bool s) :
    Graph.evalAt (α := α)
        (g := unaryGraphOut (.hardMaskedSoftmax (NN.IR.HardMask.ofTensor allowed)) s s)
        (payload := {})
        (input := Spec.PackedTensor.mk (α := α) s scores)
        (vals := #[Spec.PackedTensor.mk (α := α) s scores]) (i := 1)
      =
      Except.ok
        (Spec.PackedTensor.mk (α := α) s (Spec.hardMaskedSoftmaxLastSpec scores allowed)) := by
  simp [Graph.evalAt, Graph.evalNode, Graph.normalizeNodeOutput, unaryGraphOut, unaryNodeOut, Graph.getNode, Graph.getNode?,
    Graph.expectShape, Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Local IR semantics for softmax along any in-bounds tensor dimension. -/
theorem evalAt_softmax_axis_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s] (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraphOut (.softmax axis) s s)
        (payload := {})
        (input := Spec.PackedTensor.mk (α := α) s x)
        (vals := #[Spec.PackedTensor.mk (α := α) s x]) (i := 1)
      =
      Except.ok (Spec.PackedTensor.mk (α := α) s (Activation.softmaxSpec (α := α) axis x)) := by
  cases hAxis : Shape.axisInBounds? axis s with
  | none =>
      have hSome := Shape.axisInBounds?_isSome (axis := axis) (s := s)
      simp [hAxis] at hSome
  | some h =>
      simp [Graph.evalAt, Graph.evalNode, Graph.normalizeNodeOutput, unaryGraphOut,
        unaryNodeOut, Graph.getNode, Graph.getNode?, Graph.expectShape, hAxis,
        Bind.bind, Except.bind, Pure.pure, Except.pure]

end IRStep

end Correctness

end NN.Verification.TorchLean.Proved
