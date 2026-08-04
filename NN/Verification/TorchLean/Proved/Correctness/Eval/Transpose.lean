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

/-- Typed transpose cases whose runtime semantics are proved in this module. -/
inductive TransposeOperation : Shape → Shape → Type where
  | firstTwo (m n : Nat) (rest : Shape) :
      TransposeOperation (.dim m (.dim n rest)) (.dim n (.dim m rest))
  | lastTwo3D (a b c : Nat) :
      TransposeOperation (.dim a (.dim b (.dim c .scalar)))
        (.dim a (.dim c (.dim b .scalar)))

/-- IR operation kind selected by a typed transpose case. -/
def TransposeOperation.toOpKind {inputShape outputShape : Shape}
    (operation : TransposeOperation inputShape outputShape) : OpKind :=
  match operation with
  | .firstTwo .. => .swap_first_two
  | .lastTwo3D .. => .transpose3dLastTwo

/-- Tensor denotation of a typed transpose case. -/
def TransposeOperation.denote
    {α : Type} [Context α] {inputShape outputShape : Shape}
    (operation : TransposeOperation inputShape outputShape)
    (input : Tensor α inputShape) : Tensor α outputShape :=
  match operation with
  | .firstTwo m n rest =>
      Tensor.swapFirstTwoSpec (α := α) (m := m) (n := n) (s := rest) input
  | .lastTwo3D a b c =>
      Tensor.transpose3DLastTwoSpec (α := α) (a := a) (b := b) (c := c) input

/-- Evaluate either typed transpose case in its canonical unary graph. -/
theorem evalAt_transposeOperation_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {inputShape outputShape : Shape}
    (operation : TransposeOperation inputShape outputShape)
    (input : Tensor α inputShape) :
    Graph.evalAt (α := α)
        (g := unaryGraphOut operation.toOpKind inputShape outputShape)
        (payload := {})
        (input := DVal.mk (α := α) inputShape input)
        (vals := #[DVal.mk (α := α) inputShape input]) (i := 1)
      =
      Except.ok (DVal.mk (α := α) outputShape (operation.denote input)) := by
  cases operation <;>
    simp [TransposeOperation.toOpKind, TransposeOperation.denote, Graph.evalAt, Graph.evalNode, Graph.normalizeNodeOutput,
      unaryGraphOut, unaryNodeOut, Graph.getNode, Graph.getNode?, Graph.expectShape,
      Bind.bind, Except.bind, Pure.pure, Except.pure]

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
  exact evalAt_transposeOperation_eq (.firstTwo m n rest) x

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
  exact evalAt_transposeOperation_eq (.lastTwo3D a b c) x

end IRStep

end Correctness

end NN.Verification.TorchLean.Proved
