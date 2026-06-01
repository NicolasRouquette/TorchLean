/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Correctness.Eval.ShapeOps

/-!
# Reduction IR Evaluation

Local semantics for reduction nodes accepted by the shared IR importer.
-/

@[expose] public section

namespace NN.Verification.TorchLean.Proved

open _root_.Spec
open _root_.Spec.Tensor
open NN.IR

namespace Correctness

namespace IRStep

/-- Local IR semantics for `reduce_sum` along a valid axis. -/
theorem evalAt_reduceSum_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (axis : Nat) (x : Tensor α s)
    (hAxis : PLift (Shape.valid_axis axis s))
    (hAxisLookup : Graph.mkValidAxis? (axis := axis) s = some hAxis) :
    Graph.evalAt (α := α) (g := unaryGraphOut (.reduceSum axis) s (Tensor.shapeAfterSum s axis))
        (payload := {})
        (input := DVal.mk (α := α) s x)
        (vals := #[DVal.mk (α := α) s x]) (i := 1)
      =
      Except.ok
        (DVal.mk (α := α) (Tensor.shapeAfterSum s axis)
          (Tensor.reduceSum (α := α) (s := s) axis x
            (Shape.proveReducibleAlong axis s hAxis.down))) := by
  simp [Graph.evalAt, unaryGraphOut, unaryNodeOut, Graph.getNode, Graph.getNode?,
    hAxisLookup, Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Local IR semantics for `reduce_mean` along a valid axis. -/
theorem evalAt_reduceMean_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (axis : Nat) (x : Tensor α s)
    (hAxis : PLift (Shape.valid_axis axis s))
    (hAxisLookup : Graph.mkValidAxis? (axis := axis) s = some hAxis) :
    Graph.evalAt (α := α) (g := unaryGraphOut (.reduceMean axis) s (Tensor.shapeAfterSum s axis))
        (payload := {})
        (input := DVal.mk (α := α) s x)
        (vals := #[DVal.mk (α := α) s x]) (i := 1)
      =
      Except.ok
        (DVal.mk (α := α) (Tensor.shapeAfterSum s axis)
          (Tensor.reduceMean (α := α) (s := s) axis x
            (Shape.proveReducibleAlong axis s hAxis.down))) := by
  simp [Graph.evalAt, unaryGraphOut, unaryNodeOut, Graph.getNode, Graph.getNode?,
    hAxisLookup, Bind.bind, Except.bind, Pure.pure, Except.pure]

end IRStep

end Correctness

end NN.Verification.TorchLean.Proved
