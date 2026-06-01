/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Correctness.Eval.Core
public import NN.IR.Semantics

/-!
# Elementwise IR Evaluation

These lemmas cover the common elementwise operators emitted by the PyTorch and ONNX bridges.  Each
statement is local to one IR node: if the parent values are already present in the evaluator table,
`Graph.evalAt` returns the corresponding spec tensor operation.
-/

@[expose] public section

namespace NN.Verification.TorchLean.Proved

open _root_.Spec
open _root_.Spec.Tensor
open NN.IR

namespace Correctness

namespace IRStep

/-- Local IR semantics for elementwise addition. -/
theorem evalAt_add_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (a b : Tensor α s) :
    Graph.evalAt (α := α) (g := binaryGraph .add s) (payload := {})
        (input := DVal.mk (α := α) s a)
        (vals := #[DVal.mk (α := α) s a, DVal.mk (α := α) s b]) (i := 2)
      =
      Except.ok (DVal.mk (α := α) s (Tensor.addSpec (α := α) a b)) := by
  simp [Graph.evalAt, binaryGraph, binaryNode, Graph.getNode, Graph.getNode?, Graph.expectShape,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Local IR semantics for elementwise subtraction. -/
theorem evalAt_sub_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (a b : Tensor α s) :
    Graph.evalAt (α := α) (g := binaryGraph .sub s) (payload := {})
        (input := DVal.mk (α := α) s a)
        (vals := #[DVal.mk (α := α) s a, DVal.mk (α := α) s b]) (i := 2)
      =
      Except.ok (DVal.mk (α := α) s (Tensor.subSpec (α := α) a b)) := by
  simp [Graph.evalAt, binaryGraph, binaryNode, Graph.getNode, Graph.getNode?, Graph.expectShape,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Local IR semantics for elementwise multiplication. -/
theorem evalAt_mul_elem_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (a b : Tensor α s) :
    Graph.evalAt (α := α) (g := binaryGraph .mul_elem s) (payload := {})
        (input := DVal.mk (α := α) s a)
        (vals := #[DVal.mk (α := α) s a, DVal.mk (α := α) s b]) (i := 2)
      =
      Except.ok (DVal.mk (α := α) s (Tensor.mulSpec (α := α) a b)) := by
  simp [Graph.evalAt, binaryGraph, binaryNode, Graph.getNode, Graph.getNode?, Graph.expectShape,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Local IR semantics for elementwise maximum. -/
theorem evalAt_maxElem_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (a b : Tensor α s) :
    Graph.evalAt (α := α) (g := binaryGraph .maxElem s) (payload := {})
        (input := DVal.mk (α := α) s a)
        (vals := #[DVal.mk (α := α) s a, DVal.mk (α := α) s b]) (i := 2)
      =
      Except.ok (DVal.mk (α := α) s (Tensor.maxSpec (α := α) a b)) := by
  simp [Graph.evalAt, binaryGraph, binaryNode, Graph.getNode, Graph.getNode?, Graph.expectShape,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Local IR semantics for elementwise minimum. -/
theorem evalAt_minElem_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (a b : Tensor α s) :
    Graph.evalAt (α := α) (g := binaryGraph .minElem s) (payload := {})
        (input := DVal.mk (α := α) s a)
        (vals := #[DVal.mk (α := α) s a, DVal.mk (α := α) s b]) (i := 2)
      =
      Except.ok (DVal.mk (α := α) s (Tensor.minSpec (α := α) a b)) := by
  simp [Graph.evalAt, binaryGraph, binaryNode, Graph.getNode, Graph.getNode?, Graph.expectShape,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Local IR semantics for elementwise absolute value. -/
theorem evalAt_abs_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph .abs s) (payload := {})
        (input := DVal.mk (α := α) s x)
        (vals := #[DVal.mk (α := α) s x]) (i := 1)
      =
      Except.ok (DVal.mk (α := α) s (Tensor.absSpec (α := α) x)) := by
  simp [Graph.evalAt, unaryGraph, unaryNode, Graph.getNode, Graph.getNode?, Graph.expectShape,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Local IR semantics for elementwise square root. -/
theorem evalAt_sqrt_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph .sqrt s) (payload := {})
        (input := DVal.mk (α := α) s x)
        (vals := #[DVal.mk (α := α) s x]) (i := 1)
      =
      Except.ok (DVal.mk (α := α) s (Tensor.sqrtSpec (α := α) x)) := by
  simp [Graph.evalAt, unaryGraph, unaryNode, Graph.getNode, Graph.getNode?, Graph.expectShape,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Local IR semantics for elementwise reciprocal. -/
theorem evalAt_inv_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph .inv s) (payload := {})
        (input := DVal.mk (α := α) s x)
        (vals := #[DVal.mk (α := α) s x]) (i := 1)
      =
      Except.ok (DVal.mk (α := α) s (Tensor.invSpec (α := α) x)) := by
  simp [Graph.evalAt, unaryGraph, unaryNode, Graph.getNode, Graph.getNode?, Graph.expectShape,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Local IR semantics for ReLU. -/
theorem evalAt_relu_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph .relu s) (payload := {})
        (input := DVal.mk (α := α) s x)
        (vals := #[DVal.mk (α := α) s x]) (i := 1)
      =
      Except.ok (DVal.mk (α := α) s (Activation.reluSpec (α := α) x)) := by
  simp [Graph.evalAt, unaryGraph, unaryNode, Graph.getNode, Graph.getNode?, Graph.expectShape,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Local IR semantics for tanh. -/
theorem evalAt_tanh_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph .tanh s) (payload := {})
        (input := DVal.mk (α := α) s x)
        (vals := #[DVal.mk (α := α) s x]) (i := 1)
      =
      Except.ok (DVal.mk (α := α) s (Activation.tanhSpec (α := α) x)) := by
  simp [Graph.evalAt, unaryGraph, unaryNode, Graph.getNode, Graph.getNode?, Graph.expectShape,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Local IR semantics for sigmoid. -/
theorem evalAt_sigmoid_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph .sigmoid s) (payload := {})
        (input := DVal.mk (α := α) s x)
        (vals := #[DVal.mk (α := α) s x]) (i := 1)
      =
      Except.ok (DVal.mk (α := α) s (Activation.sigmoidSpec (α := α) x)) := by
  simp [Graph.evalAt, unaryGraph, unaryNode, Graph.getNode, Graph.getNode?, Graph.expectShape,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Local IR semantics for exp. -/
theorem evalAt_exp_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph .exp s) (payload := {})
        (input := DVal.mk (α := α) s x)
        (vals := #[DVal.mk (α := α) s x]) (i := 1)
      =
      Except.ok (DVal.mk (α := α) s (Tensor.expSpec (α := α) x)) := by
  simp [Graph.evalAt, unaryGraph, unaryNode, Graph.getNode, Graph.getNode?, Graph.expectShape,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Local IR semantics for sin. -/
theorem evalAt_sin_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph .sin s) (payload := {})
        (input := DVal.mk (α := α) s x)
        (vals := #[DVal.mk (α := α) s x]) (i := 1)
      =
      Except.ok (DVal.mk (α := α) s (Tensor.mapSpec (fun v => MathFunctions.sin v) x)) := by
  simp [Graph.evalAt, unaryGraph, unaryNode, Graph.getNode, Graph.getNode?, Graph.expectShape,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Local IR semantics for cos. -/
theorem evalAt_cos_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (x : Tensor α s) :
    Graph.evalAt (α := α) (g := unaryGraph .cos s) (payload := {})
        (input := DVal.mk (α := α) s x)
        (vals := #[DVal.mk (α := α) s x]) (i := 1)
      =
      Except.ok (DVal.mk (α := α) s (Tensor.mapSpec (fun v => MathFunctions.cos v) x)) := by
  simp [Graph.evalAt, unaryGraph, unaryNode, Graph.getNode, Graph.getNode?, Graph.expectShape,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

/-- Local IR semantics for log on inputs satisfying the IR positivity side condition. -/
theorem evalAt_log_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (x : Tensor α s)
    (hpos : Tensor.allSpec (α := α) (s := s) (fun v => decide (0 < v)) x = true) :
    Graph.evalAt (α := α) (g := unaryGraph .log s) (payload := {})
        (input := DVal.mk (α := α) s x)
        (vals := #[DVal.mk (α := α) s x]) (i := 1)
      =
      Except.ok (DVal.mk (α := α) s (Tensor.logSpec (α := α) x)) := by
  simp [Graph.evalAt, unaryGraph, unaryNode, Graph.getNode, Graph.getNode?, Graph.expectShape, hpos,
    Bind.bind, Except.bind, Pure.pure, Except.pure]

end IRStep

end Correctness

end NN.Verification.TorchLean.Proved
