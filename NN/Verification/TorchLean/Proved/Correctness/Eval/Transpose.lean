/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Correctness.Eval.ShapeOps

/-!
# Transpose IR Evaluation

Local semantics for the two transpose forms currently accepted by the PyTorch/ONNX import bridge.
-/

@[expose] public section

namespace NN.Verification.TorchLean.Proved

open _root_.Spec
open _root_.Spec.Tensor
open NN.IR

namespace Correctness

namespace IRStep

/-- Local IR semantics for swapping the first two axes. -/
theorem evalAt_swap_first_two_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {m n : Nat} {rest : Shape}
    (x : Tensor α (.dim m (.dim n rest))) :
    Graph.evalAt (α := α)
        (g := unaryGraphOut .swap_first_two
          (.dim m (.dim n rest))
          (.dim n (.dim m rest)))
        (payload := {})
        (input := DVal.mk (α := α) (.dim m (.dim n rest)) x)
        (vals := #[DVal.mk (α := α) (.dim m (.dim n rest)) x]) (i := 1)
      =
      Except.ok
        (DVal.mk (α := α) (.dim n (.dim m rest))
          (Tensor.swapFirstTwoSpec (α := α) (m := m) (n := n) (s := rest) x)) := by
  simp [Graph.evalAt, unaryGraphOut, unaryNodeOut, Graph.getNode, Graph.getNode?,
    Graph.expectShape, Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Local IR semantics for swapping the last two axes of a rank-3 tensor. -/
theorem evalAt_transpose3dLastTwo_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {a b c : Nat}
    (x : Tensor α (.dim a (.dim b (.dim c .scalar)))) :
    Graph.evalAt (α := α)
        (g := unaryGraphOut .transpose3dLastTwo
          (.dim a (.dim b (.dim c .scalar)))
          (.dim a (.dim c (.dim b .scalar))))
        (payload := {})
        (input := DVal.mk (α := α) (.dim a (.dim b (.dim c .scalar))) x)
        (vals := #[DVal.mk (α := α) (.dim a (.dim b (.dim c .scalar))) x]) (i := 1)
      =
      Except.ok
        (DVal.mk (α := α) (.dim a (.dim c (.dim b .scalar)))
          (Tensor.transpose3DLastTwoSpec (α := α) (a := a) (b := b) (c := c) x)) := by
  simp [Graph.evalAt, unaryGraphOut, unaryNodeOut, Graph.getNode, Graph.getNode?,
    Graph.expectShape, Bind.bind, Except.bind, Pure.pure, Except.pure]

end IRStep

end Correctness

end NN.Verification.TorchLean.Proved
