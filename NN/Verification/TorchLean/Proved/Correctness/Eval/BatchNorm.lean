/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Correctness.Eval.PayloadOps

/-!
# BatchNorm IR Evaluation

The TorchLean compiler, PyTorch importer, verifier, and PyTorch exporter all meet at the IR
BatchNorm node.  This file records the small semantic fact that matters at that boundary: once the
payload is present and the input has matching NCHW channels, IR evaluation is the standard
eval-mode BatchNorm formula applied independently at each `(N,C,H,W)` coordinate.
-/

@[expose] public section

namespace NN.Verification.TorchLean.Proved

open _root_.Spec
open _root_.Spec.Tensor
open NN.IR

namespace Correctness

namespace IRStep

/--
The scalar formula used by eval-mode NCHW BatchNorm in the IR.

Keeping this formula named gives import/export proofs, regression tests, and documentation a single
place to point to instead of repeating the normalization expression in several files.
-/
def batchNorm2dNchwEvalScalar {α : Type} [Context α]
    (x gamma beta mean var eps : α) : α :=
  let denom := MathFunctions.sqrt (max var (0 : α) + eps)
  (((x - mean) / denom) * gamma + beta)

/-- The NCHW tensor obtained by applying `batchNorm2dNchwEvalScalar` channel-wise. -/
def batchNorm2dNchwEvalTensor {α : Type} [Context α]
    {n c h w : Nat}
    (gamma beta mean var : Tensor α (.dim c .scalar))
    (eps : α)
    (x : Tensor α (.dim n (.dim c (.dim h (.dim w .scalar))))) :
    Tensor α (.dim n (.dim c (.dim h (.dim w .scalar)))) :=
  Tensor.dim fun ni =>
    Tensor.dim fun ci =>
      Tensor.dim fun hi =>
        Tensor.dim fun wi =>
          match getAtSpec (getAtSpec (getAtSpec (getAtSpec x ni) ci) hi) wi,
              getAtSpec gamma ci, getAtSpec beta ci, getAtSpec mean ci, getAtSpec var ci with
          | .scalar xv, .scalar g, .scalar b, .scalar m, .scalar v =>
              Tensor.scalar (batchNorm2dNchwEvalScalar (α := α) xv g b m v eps)

/--
Coordinate-level semantics for `Graph.evalBatchNorm2DNchwEval`.

This is the proof-facing version of the BatchNorm parity test: the payload-backed IR evaluator is
definitionally the channel-wise eval-mode BatchNorm equation over NCHW tensors.
-/
theorem evalBatchNorm2DNchwEval_eq_nchw_formula
    {α : Type} [Context α] [DecidableEq Shape]
    (id n c h w : Nat)
    (gamma beta mean var : Tensor α (.dim c .scalar))
    (eps : α)
    (x : Tensor α (.dim n (.dim c (.dim h (.dim w .scalar))))) :
    let cfg : BatchNorm2DNchwEvalParams α :=
      { c := c, gamma := gamma, beta := beta, mean := mean, var := var, eps := eps }
    let payload : Payload α :=
      singletonBatchNorm2DNchwEvalPayload (α := α) id cfg
    Graph.evalBatchNorm2DNchwEval (α := α) (payload := payload) (id := id)
        (x := DVal.mk (α := α) (.dim n (.dim c (.dim h (.dim w .scalar)))) x
      )
      =
      Except.ok
        (DVal.mk (α := α) (.dim n (.dim c (.dim h (.dim w .scalar))))
          (batchNorm2dNchwEvalTensor (α := α) gamma beta mean var eps x)) := by
  simp [Graph.evalBatchNorm2DNchwEval, singletonBatchNorm2DNchwEvalPayload, Graph.expectShape,
    batchNorm2dNchwEvalTensor, batchNorm2dNchwEvalScalar, Bind.bind, Except.bind, Pure.pure,
    Except.pure]
  rfl

/-- Local IR semantics for payload-backed eval-mode NCHW BatchNorm. -/
theorem evalAt_batchNorm2dNchwEval_eq
    {α : Type} [Context α] [DecidableEq Shape]
    (n c h w : Nat)
    (gamma beta mean var : Tensor α (.dim c .scalar))
    (eps : α)
    (x : Tensor α (.dim n (.dim c (.dim h (.dim w .scalar))))) :
    let cfg : BatchNorm2DNchwEvalParams α :=
      { c := c, gamma := gamma, beta := beta, mean := mean, var := var, eps := eps }
    Graph.evalAt (α := α)
        (g := unaryGraphOut (.batchNorm2dNchwEval c)
          (.dim n (.dim c (.dim h (.dim w .scalar))))
          (.dim n (.dim c (.dim h (.dim w .scalar)))))
        (payload := singletonBatchNorm2DNchwEvalPayload (α := α) 1 cfg)
        (input := DVal.mk (α := α) (.dim n (.dim c (.dim h (.dim w .scalar)))) x)
        (vals := #[DVal.mk (α := α) (.dim n (.dim c (.dim h (.dim w .scalar)))) x])
        (i := 1)
      =
      Except.ok
        (DVal.mk (α := α) (.dim n (.dim c (.dim h (.dim w .scalar))))
          (batchNorm2dNchwEvalTensor (α := α) gamma beta mean var eps x)) := by
  simp [Graph.evalAt, unaryGraphOut, unaryNodeOut, Graph.getNode, Graph.getNode?,
    Graph.evalBatchNorm2DNchwEval, singletonBatchNorm2DNchwEvalPayload, Graph.expectShape,
    batchNorm2dNchwEvalTensor, batchNorm2dNchwEvalScalar, shapeBNe_refl,
    Bind.bind, Except.bind, Pure.pure, Except.pure]
  rfl

/-- Missing BatchNorm payloads are rejected before any tensor computation happens. -/
theorem evalBatchNorm2DNchwEval_missing_payload
    {α : Type} [Context α] [DecidableEq Shape]
    (payload : Payload α) (id : Nat)
    (hMissing : payload.batchNorm2dNchwEval? id = none)
    (x : DVal α) :
    Graph.evalBatchNorm2DNchwEval (α := α) (payload := payload) (id := id) (x := x)
      =
      Except.error s!"IR eval: missing batch_norm2d_nchw_eval payload for node {id}" := by
  simp [Graph.evalBatchNorm2DNchwEval, hMissing]
  rfl

/--
Shape inference for eval-mode BatchNorm is the identity on well-formed NCHW inputs with matching
channels.
-/
theorem inferBatchNorm2dNchwEvalOutShape_eq_self
    (n c h w : Nat)
    (hc : c ≠ 0) (hn : n ≠ 0) (hh : h ≠ 0) (hw : w ≠ 0) :
    OpContracts.inferBatchNorm2dNchwEvalOutShape c
        (.dim n (.dim c (.dim h (.dim w .scalar))))
      =
      Except.ok (.dim n (.dim c (.dim h (.dim w .scalar)))) := by
  simp [OpContracts.inferBatchNorm2dNchwEvalOutShape, OpContracts.checkPositive,
    hc, hn, hh, hw, Bind.bind, Except.bind, Pure.pure, Except.pure]

end IRStep

end Correctness

end NN.Verification.TorchLean.Proved
