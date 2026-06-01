/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Correctness.Eval.LinearAlgebra

/-!
# Concat IR Evaluation

Local semantics for IR concat.  The evaluator keeps the generic-axis implementation in the shared
`Graph.evalConcat` helper, which moves the requested axis to the front, folds
`Tensor.concatDim0Spec`, and moves the result back.
-/

@[expose] public section

namespace NN.Verification.TorchLean.Proved

open _root_.Spec
open _root_.Spec.Tensor
open NN.IR

namespace Correctness

namespace IRStep

/-- A three-input node with parents `0`, `1`, and `2`. -/
def ternaryNodeOut (kind : OpKind) (outShape : Shape) : NN.IR.Node :=
  { id := 3, parents := [0, 1, 2], kind := kind, outShape := outShape }

/-- A graph table containing only a ternary node at index `3`; earlier indices are parent slots. -/
def ternaryGraphOut (kind : OpKind) (left mid right outShape : Shape) : Graph :=
  { nodes := #[
      { id := 0, parents := [], kind := .input, outShape := left },
      { id := 1, parents := [], kind := .input, outShape := mid },
      { id := 2, parents := [], kind := .input, outShape := right },
      ternaryNodeOut kind outShape
    ] }

/-- A four-input node with parents `0`, `1`, `2`, and `3`. -/
def quaternaryNodeOut (kind : OpKind) (outShape : Shape) : NN.IR.Node :=
  { id := 4, parents := [0, 1, 2, 3], kind := kind, outShape := outShape }

/-- A graph table containing only a quaternary node at index `4`; earlier indices are parent slots. -/
def quaternaryGraphOut
    (kind : OpKind) (s₀ s₁ s₂ s₃ outShape : Shape) : Graph :=
  { nodes := #[
      { id := 0, parents := [], kind := .input, outShape := s₀ },
      { id := 1, parents := [], kind := .input, outShape := s₁ },
      { id := 2, parents := [], kind := .input, outShape := s₂ },
      { id := 3, parents := [], kind := .input, outShape := s₃ },
      quaternaryNodeOut kind outShape
    ] }

/-- Local IR semantics for binary concat, pinned to the shared generic concat interpreter. -/
theorem evalAt_concat_binary_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s₁ s₂ out : Shape} (axis : Nat)
    (lhs : Tensor α s₁) (rhs : Tensor α s₂) :
    Graph.evalAt (α := α)
        (g := binaryGraphOut (.concat axis) s₁ s₂ out)
        (payload := {})
        (input := DVal.mk (α := α) s₁ lhs)
        (vals := #[
          DVal.mk (α := α) s₁ lhs,
          DVal.mk (α := α) s₂ rhs
        ]) (i := 2)
      =
      (Graph.evalConcat (α := α) 2 (binaryNodeOut (.concat axis) out) axis
          [DVal.mk (α := α) s₁ lhs, DVal.mk (α := α) s₂ rhs]).bind
        (Graph.normalizeNodeOutput (α := α) 2 (binaryNodeOut (.concat axis) out)) := by
  simp [Graph.evalAt, binaryGraphOut, binaryNodeOut, Graph.getNode, Graph.getNode?,
    Graph.normalizeNodeOutput, Bind.bind, Except.bind, Pure.pure, Except.pure]

/--
Successful binary concat evaluation, once the shared concat interpreter has produced a value with
the node's declared output shape.
-/
theorem evalAt_concat_binary_ok
    {α : Type} [Context α] [DecidableEq Shape]
    {s₁ s₂ out : Shape} (axis : Nat)
    (lhs : Tensor α s₁) (rhs : Tensor α s₂) (y : Tensor α out)
    (hConcat :
      Graph.evalConcat (α := α) 2 (binaryNodeOut (.concat axis) out) axis
          [DVal.mk (α := α) s₁ lhs, DVal.mk (α := α) s₂ rhs] =
        .ok (DVal.mk (α := α) out y)) :
    Graph.evalAt (α := α)
        (g := binaryGraphOut (.concat axis) s₁ s₂ out)
        (payload := {})
        (input := DVal.mk (α := α) s₁ lhs)
        (vals := #[
          DVal.mk (α := α) s₁ lhs,
          DVal.mk (α := α) s₂ rhs
        ]) (i := 2)
      =
      .ok (DVal.mk (α := α) out y) := by
  rw [evalAt_concat_binary_eq]
  rw [hConcat]
  simp [Graph.normalizeNodeOutput, binaryNodeOut, Except.bind, Pure.pure, Except.pure]

/-- Binary concat evaluation rejects the node whenever the shared concat interpreter rejects it. -/
theorem evalAt_concat_binary_error
    {α : Type} [Context α] [DecidableEq Shape]
    {s₁ s₂ out : Shape} (axis : Nat)
    (lhs : Tensor α s₁) (rhs : Tensor α s₂) (msg : String)
    (hConcat :
      Graph.evalConcat (α := α) 2 (binaryNodeOut (.concat axis) out) axis
          [DVal.mk (α := α) s₁ lhs, DVal.mk (α := α) s₂ rhs] =
        .error msg) :
    Graph.evalAt (α := α)
        (g := binaryGraphOut (.concat axis) s₁ s₂ out)
        (payload := {})
        (input := DVal.mk (α := α) s₁ lhs)
        (vals := #[
          DVal.mk (α := α) s₁ lhs,
          DVal.mk (α := α) s₂ rhs
        ]) (i := 2)
      =
      .error msg := by
  rw [evalAt_concat_binary_eq]
  rw [hConcat]
  rfl

/-- The leading-axis concat fold agrees with `Tensor.concatDim0Spec` for binary concat. -/
theorem evalConcatAxis0Fold_binary_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {n m : Nat} {rest : Shape}
    (lhs : Tensor α (.dim n rest)) (rhs : Tensor α (.dim m rest)) :
    Graph.evalConcatAxis0Fold (α := α) 2 (n + m) rest
        [DVal.mk (α := α) (.dim n rest) lhs,
          DVal.mk (α := α) (.dim m rest) rhs]
      =
      .ok
        (DVal.mk (α := α) (.dim (n + m) rest)
          (Tensor.concatDim0Spec (α := α) (n := n) (m := m) (s := rest) lhs rhs)) := by
  simp [Graph.evalConcatAxis0Fold,
    DVal.mk, DVal.shape, DVal.tensor, Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- The leading-axis concat fold agrees with two `Tensor.concatDim0Spec` steps for ternary concat. -/
theorem evalConcatAxis0Fold_ternary_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {n m k : Nat} {rest : Shape}
    (x : Tensor α (.dim n rest)) (y : Tensor α (.dim m rest))
    (z : Tensor α (.dim k rest)) :
    Graph.evalConcatAxis0Fold (α := α) 3 (n + m + k) rest
        [DVal.mk (α := α) (.dim n rest) x,
          DVal.mk (α := α) (.dim m rest) y,
          DVal.mk (α := α) (.dim k rest) z]
      =
      .ok
        (DVal.mk (α := α) (.dim (n + m + k) rest)
          (Tensor.concatDim0Spec (α := α) (n := n + m) (m := k) (s := rest)
            (Tensor.concatDim0Spec (α := α) (n := n) (m := m) (s := rest) x y) z)) := by
  simp [Graph.evalConcatAxis0Fold,
    DVal.mk, DVal.shape, DVal.tensor, Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- The leading-axis concat fold agrees with three `Tensor.concatDim0Spec` steps for four inputs. -/
theorem evalConcatAxis0Fold_quaternary_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {n m k l : Nat} {rest : Shape}
    (x : Tensor α (.dim n rest)) (y : Tensor α (.dim m rest))
    (z : Tensor α (.dim k rest)) (w : Tensor α (.dim l rest)) :
    Graph.evalConcatAxis0Fold (α := α) 4 (n + m + k + l) rest
        [DVal.mk (α := α) (.dim n rest) x,
          DVal.mk (α := α) (.dim m rest) y,
          DVal.mk (α := α) (.dim k rest) z,
          DVal.mk (α := α) (.dim l rest) w]
      =
      .ok
        (DVal.mk (α := α) (.dim (n + m + k + l) rest)
          (Tensor.concatDim0Spec (α := α) (n := n + m + k) (m := l) (s := rest)
            (Tensor.concatDim0Spec (α := α) (n := n + m) (m := k) (s := rest)
              (Tensor.concatDim0Spec (α := α) (n := n) (m := m) (s := rest) x y) z) w)) := by
  simp [Graph.evalConcatAxis0Fold,
    DVal.mk, DVal.shape, DVal.tensor, Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Shape inference for binary concat along axis 0. -/
theorem inferConcatOutShape_axis0_binary_eq
    {n m : Nat} {rest : Shape} :
    OpContracts.inferConcatOutShape 0 [.dim n rest, .dim m rest] =
      .ok (.dim (n + m) rest) := by
  have hAxis : OpContracts.checkAxisValid 0 (.dim n rest) = .ok () := by
    unfold OpContracts.checkAxisValid
    change (if 0 < 1 + Shape.rank rest then Except.ok () else
      Except.error s!"invalid axis {0} for rank {Shape.rank (.dim n rest)}") = Except.ok ()
    simp
  have hRank : Shape.rank (.dim m rest) = Shape.rank (.dim n rest) := rfl
  have hTail : (rest != rest) = false := shapeBNe_refl rest
  simp [OpContracts.inferConcatOutShape, OpContracts.inferConcatOutShape.go,
    hAxis, hRank, hTail, Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Shape inference for ternary concat along axis 0. -/
theorem inferConcatOutShape_axis0_ternary_eq
    {n m k : Nat} {rest : Shape} :
    OpContracts.inferConcatOutShape 0 [.dim n rest, .dim m rest, .dim k rest] =
      .ok (.dim (n + m + k) rest) := by
  have hAxis : OpContracts.checkAxisValid 0 (.dim n rest) = .ok () := by
    unfold OpContracts.checkAxisValid
    change (if 0 < 1 + Shape.rank rest then Except.ok () else
      Except.error s!"invalid axis {0} for rank {Shape.rank (.dim n rest)}") = Except.ok ()
    simp
  have hRankM : Shape.rank (.dim m rest) = Shape.rank (.dim n rest) := rfl
  have hRankK : Shape.rank (.dim k rest) = Shape.rank (.dim n rest) := rfl
  have hTail : (rest != rest) = false := shapeBNe_refl rest
  simp [OpContracts.inferConcatOutShape, OpContracts.inferConcatOutShape.go,
    hAxis, hRankM, hRankK, hTail, Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Shape inference for four-input concat along axis 0. -/
theorem inferConcatOutShape_axis0_quaternary_eq
    {n m k l : Nat} {rest : Shape} :
    OpContracts.inferConcatOutShape 0 [.dim n rest, .dim m rest, .dim k rest, .dim l rest] =
      .ok (.dim (n + m + k + l) rest) := by
  have hAxis : OpContracts.checkAxisValid 0 (.dim n rest) = .ok () := by
    unfold OpContracts.checkAxisValid
    change (if 0 < 1 + Shape.rank rest then Except.ok () else
      Except.error s!"invalid axis {0} for rank {Shape.rank (.dim n rest)}") = Except.ok ()
    simp
  have hRankM : Shape.rank (.dim m rest) = Shape.rank (.dim n rest) := rfl
  have hRankK : Shape.rank (.dim k rest) = Shape.rank (.dim n rest) := rfl
  have hRankL : Shape.rank (.dim l rest) = Shape.rank (.dim n rest) := rfl
  have hTail : (rest != rest) = false := shapeBNe_refl rest
  simp [OpContracts.inferConcatOutShape, OpContracts.inferConcatOutShape.go,
    hAxis, hRankM, hRankK, hRankL, hTail, Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- The shared concat interpreter takes the direct leading-axis path when shape inference agrees. -/
theorem evalConcat_axis0_binary_eq_of_infer
    {α : Type} [Context α] [DecidableEq Shape]
    {s₁ s₂ out : Shape} (lhs : Tensor α s₁) (rhs : Tensor α s₂)
    {nOut : Nat} {rest : Shape}
    (hOut : out = .dim nOut rest)
    (hInfer : OpContracts.inferConcatOutShape 0 [s₁, s₂] = .ok out) :
    Graph.evalConcat (α := α) 2 (binaryNodeOut (.concat 0) out) 0
        [DVal.mk (α := α) s₁ lhs, DVal.mk (α := α) s₂ rhs]
      =
      Graph.evalConcatAxis0Fold (α := α) 2 nOut rest
        [DVal.mk (α := α) s₁ lhs, DVal.mk (α := α) s₂ rhs] := by
  subst hOut
  have hSame : (Shape.dim nOut rest != Shape.dim nOut rest) = false :=
    shapeBNe_refl (.dim nOut rest)
  simp [Graph.evalConcat, binaryNodeOut, hInfer, Graph.evalConcatAxis0Fold,
    hSame, Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Local IR semantics for leading-axis binary concat when shape inference accepts the declared shape. -/
theorem evalAt_concat_axis0_binary_eq_of_infer
    {α : Type} [Context α] [DecidableEq Shape]
    {n m : Nat} {rest : Shape}
    (lhs : Tensor α (.dim n rest)) (rhs : Tensor α (.dim m rest))
    (hInfer :
      OpContracts.inferConcatOutShape 0 [.dim n rest, .dim m rest] =
        .ok (.dim (n + m) rest)) :
    Graph.evalAt (α := α)
        (g := binaryGraphOut (.concat 0) (.dim n rest) (.dim m rest) (.dim (n + m) rest))
        (payload := {})
        (input := DVal.mk (α := α) (.dim n rest) lhs)
        (vals := #[
          DVal.mk (α := α) (.dim n rest) lhs,
          DVal.mk (α := α) (.dim m rest) rhs
        ]) (i := 2)
      =
      .ok
        (DVal.mk (α := α) (.dim (n + m) rest)
          (Tensor.concatDim0Spec (α := α) (n := n) (m := m) (s := rest) lhs rhs)) := by
  apply evalAt_concat_binary_ok
  rw [evalConcat_axis0_binary_eq_of_infer (lhs := lhs) (rhs := rhs)
    (hOut := rfl) (hInfer := hInfer)]
  exact evalConcatAxis0Fold_binary_eq (α := α) lhs rhs

/-- Local IR semantics for binary concat along the leading axis. -/
theorem evalAt_concat_axis0_binary_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {n m : Nat} {rest : Shape}
    (lhs : Tensor α (.dim n rest)) (rhs : Tensor α (.dim m rest)) :
    Graph.evalAt (α := α)
        (g := binaryGraphOut (.concat 0) (.dim n rest) (.dim m rest) (.dim (n + m) rest))
        (payload := {})
        (input := DVal.mk (α := α) (.dim n rest) lhs)
        (vals := #[
          DVal.mk (α := α) (.dim n rest) lhs,
          DVal.mk (α := α) (.dim m rest) rhs
        ]) (i := 2)
      =
      .ok
        (DVal.mk (α := α) (.dim (n + m) rest)
          (Tensor.concatDim0Spec (α := α) (n := n) (m := m) (s := rest) lhs rhs)) := by
  exact evalAt_concat_axis0_binary_eq_of_infer (α := α) lhs rhs
    inferConcatOutShape_axis0_binary_eq

/-- Local IR semantics for ternary concat along the leading axis. -/
theorem evalAt_concat_axis0_ternary_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {n m k : Nat} {rest : Shape}
    (x : Tensor α (.dim n rest)) (y : Tensor α (.dim m rest))
    (z : Tensor α (.dim k rest)) :
    Graph.evalAt (α := α)
        (g := ternaryGraphOut (.concat 0) (.dim n rest) (.dim m rest) (.dim k rest)
          (.dim (n + m + k) rest))
        (payload := {})
        (input := DVal.mk (α := α) (.dim n rest) x)
        (vals := #[
          DVal.mk (α := α) (.dim n rest) x,
          DVal.mk (α := α) (.dim m rest) y,
          DVal.mk (α := α) (.dim k rest) z
        ]) (i := 3)
      =
      .ok
        (DVal.mk (α := α) (.dim (n + m + k) rest)
          (Tensor.concatDim0Spec (α := α) (n := n + m) (m := k) (s := rest)
            (Tensor.concatDim0Spec (α := α) (n := n) (m := m) (s := rest) x y) z)) := by
  have hSame : (Shape.dim (n + m + k) rest != Shape.dim (n + m + k) rest) = false :=
    shapeBNe_refl (.dim (n + m + k) rest)
  simp [Graph.evalAt, ternaryGraphOut, ternaryNodeOut, Graph.getNode, Graph.getNode?,
    Graph.evalConcat, inferConcatOutShape_axis0_ternary_eq, hSame,
    Bind.bind, Except.bind, Pure.pure, Except.pure]
  have hFold := evalConcatAxis0Fold_ternary_eq (α := α) x y z
  simp [DVal.mk] at hFold
  rw [hFold]
  simp

/-- Local IR semantics for four-input concat along the leading axis. -/
theorem evalAt_concat_axis0_quaternary_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {n m k l : Nat} {rest : Shape}
    (x : Tensor α (.dim n rest)) (y : Tensor α (.dim m rest))
    (z : Tensor α (.dim k rest)) (w : Tensor α (.dim l rest)) :
    Graph.evalAt (α := α)
        (g := quaternaryGraphOut (.concat 0) (.dim n rest) (.dim m rest)
          (.dim k rest) (.dim l rest) (.dim (n + m + k + l) rest))
        (payload := {})
        (input := DVal.mk (α := α) (.dim n rest) x)
        (vals := #[
          DVal.mk (α := α) (.dim n rest) x,
          DVal.mk (α := α) (.dim m rest) y,
          DVal.mk (α := α) (.dim k rest) z,
          DVal.mk (α := α) (.dim l rest) w
        ]) (i := 4)
      =
      .ok
        (DVal.mk (α := α) (.dim (n + m + k + l) rest)
          (Tensor.concatDim0Spec (α := α) (n := n + m + k) (m := l) (s := rest)
            (Tensor.concatDim0Spec (α := α) (n := n + m) (m := k) (s := rest)
              (Tensor.concatDim0Spec (α := α) (n := n) (m := m) (s := rest) x y) z) w)) := by
  have hSame :
      (Shape.dim (n + m + k + l) rest != Shape.dim (n + m + k + l) rest) = false :=
    shapeBNe_refl (.dim (n + m + k + l) rest)
  simp [Graph.evalAt, quaternaryGraphOut, quaternaryNodeOut, Graph.getNode, Graph.getNode?,
    Graph.evalConcat, inferConcatOutShape_axis0_quaternary_eq, hSame,
    Bind.bind, Except.bind, Pure.pure, Except.pure]
  have hFold := evalConcatAxis0Fold_quaternary_eq (α := α) x y z w
  simp [DVal.mk] at hFold
  rw [hFold]
  simp

end IRStep

end Correctness

end NN.Verification.TorchLean.Proved
