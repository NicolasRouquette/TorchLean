/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Correctness.Eval.Permutation

/-!
# Softmax IR Evaluation

The IR evaluator supports arbitrary softmax axes by moving the requested axis to the end, applying
the spec softmax, and moving the result back.
-/

@[expose] public section

namespace NN.Verification.TorchLean.Proved

open _root_.Spec
open _root_.Spec.Tensor
open NN.IR

namespace Correctness

namespace IRStep

/-- Local IR semantics for last-axis softmax. -/
theorem evalAt_softmax_last_axis_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (axis : Nat) (x : Tensor α s)
    (hAxis : OpContracts.checkAxisValid axis s = .ok ())
    (hLast : axis + 1 = Shape.rank s) :
    Graph.evalAt (α := α) (g := unaryGraphOut (.softmax axis) s s)
        (payload := {})
        (input := DVal.mk (α := α) s x)
        (vals := #[DVal.mk (α := α) s x]) (i := 1)
      =
      Except.ok (DVal.mk (α := α) s (Activation.softmaxSpec (α := α) x)) := by
  simp [Graph.evalAt, unaryGraphOut, unaryNodeOut, Graph.getNode, Graph.getNode?,
    Graph.expectShape, hAxis, hLast, Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Local IR semantics for non-last-axis softmax through the evaluator's permutation path. -/
theorem evalAt_softmax_permuted_axis_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (axis : Nat) (x : Tensor α s)
    (permToLast permBack : List Nat) (xLast yBack : DVal α) (y : Tensor α s)
    (hAxis : OpContracts.checkAxisValid axis s = .ok ())
    (hNotLast : ¬ axis + 1 = Shape.rank s)
    (hToLast : OpContracts.permMoveAxisToLast axis s = .ok permToLast)
    (hBack : OpContracts.inversePerm permToLast = .ok permBack)
    (hXLast :
      Graph.permuteDVal (α := α) (v := DVal.mk (α := α) s x) permToLast =
        .ok xLast)
    (hYBack :
      Graph.permuteDVal (α := α)
          (v := match xLast with
            | ⟨sLast, tLast⟩ =>
                DVal.mk (α := α) sLast (Activation.softmaxSpec (α := α) tLast))
          permBack =
        .ok yBack)
    (hYShape : Graph.expectShape (α := α) (expected := s) yBack = .ok y) :
    Graph.evalAt (α := α) (g := unaryGraphOut (.softmax axis) s s)
        (payload := {})
        (input := DVal.mk (α := α) s x)
        (vals := #[DVal.mk (α := α) s x]) (i := 1)
      =
      Except.ok (DVal.mk (α := α) s y) := by
  simp [Graph.evalAt, unaryGraphOut, unaryNodeOut, Graph.getNode, Graph.getNode?,
    Graph.expectShape, hAxis, hNotLast, hToLast, hBack,
    Bind.bind, Except.bind, Pure.pure, Except.pure]
  have hXLast' : Graph.permuteDVal (α := α) (v := ⟨s, x⟩) permToLast = .ok xLast := by
    simpa [DVal.mk] using hXLast
  rw [hXLast']
  cases xLast with
  | mk sLast tLast =>
      simp [DVal.mk] at hYBack ⊢
      rw [hYBack]
      simp
      have hYShape' :
          (if h : yBack.1 = s then Except.ok (h ▸ yBack.2)
            else
              (throw
                (s!"IR eval: shape mismatch: expected {repr s}, got {repr yBack.1}") :
                Except String (Tensor α s))) =
            Except.ok y := by
        simpa [Graph.expectShape] using hYShape
      rw [hYShape']

end IRStep

end Correctness

end NN.Verification.TorchLean.Proved
