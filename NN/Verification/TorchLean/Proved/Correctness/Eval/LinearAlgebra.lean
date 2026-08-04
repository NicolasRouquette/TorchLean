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

/-- Rank-specialized matrix multiplication operations accepted by the IR evaluator. -/
inductive MatmulOperation : Shape → Shape → Shape → Type where
  /-- Rank-2 matrix multiplication. -/
  | matrix (m n p : Nat) :
      MatmulOperation (.dim m (.dim n .scalar)) (.dim n (.dim p .scalar))
        (.dim m (.dim p .scalar))
  /-- Rank-3 batched matrix multiplication. -/
  | batched (batch m n p : Nat) :
      MatmulOperation (.dim batch (.dim m (.dim n .scalar)))
        (.dim batch (.dim n (.dim p .scalar)))
        (.dim batch (.dim m (.dim p .scalar)))

/-- Typed denotation of a supported matrix multiplication operation. -/
def MatmulOperation.denote
    {α : Type} [Context α] {leftShape rightShape outShape : Shape}
    (op : MatmulOperation leftShape rightShape outShape)
    (left : Tensor α leftShape) (right : Tensor α rightShape) : Tensor α outShape :=
  match op with
  | .matrix _ _ _ => Tensor.matMulSpec (α := α) left right
  | .batched batch m n p =>
      Tensor.bmmSpec (α := α) (batch := batch) (m := m) (n := n) (p := p) left right

/-- Evaluate any supported matrix multiplication in its canonical three-node graph. -/
theorem evalAt_matmul_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {leftShape rightShape outShape : Shape}
    (op : MatmulOperation leftShape rightShape outShape)
    (left : Tensor α leftShape) (right : Tensor α rightShape) :
    Graph.evalAt (α := α)
        (g := binaryGraphOut .matmul leftShape rightShape outShape)
        (payload := {})
        (input := DVal.mk (α := α) leftShape left)
        (vals := #[DVal.mk (α := α) leftShape left, DVal.mk (α := α) rightShape right])
        (i := 2)
      =
      Except.ok (DVal.mk (α := α) outShape (op.denote left right)) := by
  cases op <;>
    simp [MatmulOperation.denote, Graph.evalAt, Graph.evalNode, Graph.normalizeNodeOutput, binaryGraphOut, binaryNodeOut, Graph.getNode,
      Graph.getNode?, Graph.expectShape, Bind.bind, Except.bind, Pure.pure, Except.pure]

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
  exact evalAt_matmul_eq (.matrix m n p) a b

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
  exact evalAt_matmul_eq (.batched batch m n p) a b

end IRStep

end Correctness

end NN.Verification.TorchLean.Proved
