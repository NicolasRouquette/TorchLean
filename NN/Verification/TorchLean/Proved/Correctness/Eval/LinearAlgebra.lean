/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Correctness.Eval.ShapeOps

/-!
# Linear Algebra IR Evaluation

Local semantics for matrix multiplication nodes accepted by the shared IR importer.
-/

@[expose] public section

namespace NN.Verification.TorchLean.Proved

open _root_.Spec
open _root_.Spec.Tensor
open NN.IR

namespace Correctness

namespace IRStep

/-- Local IR semantics for rank-2 matrix multiplication. -/
theorem evalAt_matmul2d_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {m n p : Nat}
    (a : Tensor α (.dim m (.dim n .scalar)))
    (b : Tensor α (.dim n (.dim p .scalar))) :
    Graph.evalAt (α := α)
        (g := binaryGraphOut .matmul
          (.dim m (.dim n .scalar))
          (.dim n (.dim p .scalar))
          (.dim m (.dim p .scalar)))
        (payload := {})
        (input := DVal.mk (α := α) (.dim m (.dim n .scalar)) a)
        (vals := #[
          DVal.mk (α := α) (.dim m (.dim n .scalar)) a,
          DVal.mk (α := α) (.dim n (.dim p .scalar)) b
        ]) (i := 2)
      =
      Except.ok
        (DVal.mk (α := α) (.dim m (.dim p .scalar))
          (Tensor.matMulSpec (α := α) a b)) := by
  simp [Graph.evalAt, binaryGraphOut, binaryNodeOut, Graph.getNode, Graph.getNode?,
    Graph.expectShape, Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Local IR semantics for rank-3 batched matrix multiplication. -/
theorem evalAt_bmm_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {batch m n p : Nat}
    (a : Tensor α (.dim batch (.dim m (.dim n .scalar))))
    (b : Tensor α (.dim batch (.dim n (.dim p .scalar)))) :
    Graph.evalAt (α := α)
        (g := binaryGraphOut .matmul
          (.dim batch (.dim m (.dim n .scalar)))
          (.dim batch (.dim n (.dim p .scalar)))
          (.dim batch (.dim m (.dim p .scalar))))
        (payload := {})
        (input := DVal.mk (α := α) (.dim batch (.dim m (.dim n .scalar))) a)
        (vals := #[
          DVal.mk (α := α) (.dim batch (.dim m (.dim n .scalar))) a,
          DVal.mk (α := α) (.dim batch (.dim n (.dim p .scalar))) b
        ]) (i := 2)
      =
      Except.ok
        (DVal.mk (α := α) (.dim batch (.dim m (.dim p .scalar)))
          (Tensor.bmmSpec (α := α) (batch := batch) (m := m) (n := n) (p := p) a b)) := by
  simp [Graph.evalAt, binaryGraphOut, binaryNodeOut, Graph.getNode, Graph.getNode?,
    Graph.expectShape, Bind.bind, Except.bind, Pure.pure, Except.pure]

end IRStep

end Correctness

end NN.Verification.TorchLean.Proved
