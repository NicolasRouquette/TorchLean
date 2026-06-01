/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Correctness.Eval.Pooling

/-!
# LayerNorm IR Evaluation

The IR `layernorm axis` node interprets `axis` as the start of the normalized suffix.  Evaluation
reshapes the tensor to `(seqLen, embedDim)`, runs the spec 2D LayerNorm with `gamma=1` and
`beta=0`, then reshapes back.
-/

@[expose] public section

namespace NN.Verification.TorchLean.Proved

open _root_.Spec
open _root_.Spec.Tensor
open NN.IR

namespace Correctness

namespace IRStep

/-- `layernormPure` is exactly spec LayerNorm with unit scale and zero bias. -/
theorem layernormPure_eq_spec
    {α : Type} [Context α]
    (seqLen embedDim : Nat)
    (x : Tensor α (.dim seqLen (.dim embedDim .scalar)))
    (hSeq : seqLen > 0)
    (hEmb : embedDim > 0) :
    Graph.layernormPure (α := α) (seqLen := seqLen) (embedDim := embedDim) x
      =
      Except.ok
        (Spec.layerNorm (α := α) (seqLen := seqLen) (embedDim := embedDim)
          (x := x)
          (gamma := Spec.fill (α := α) 1 (.dim embedDim .scalar))
          (beta := Spec.fill (α := α) 0 (.dim embedDim .scalar))
          (h_seq_pos := hSeq) (h_embed_pos := hEmb)) := by
  simp [Graph.layernormPure, hSeq, hEmb]
  rfl

/--
Local IR semantics for `layernorm axis`.

The hypotheses are the same contracts checked by the evaluator: `axis` must produce a valid
`(seqLen, embedDim)` view, the reshape must preserve element count, and the pure 2D LayerNorm step
must succeed.
-/
theorem evalAt_layernorm_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (axis seqLen embedDim : Nat)
    (x : Tensor α s)
    (hParams : OpContracts.layerNorm2DParams axis s = .ok (seqLen, embedDim))
    (hNumel : Shape.size s = Shape.size (.dim seqLen (.dim embedDim .scalar)))
    (y2D : Tensor α (.dim seqLen (.dim embedDim .scalar)))
    (hPure :
      Graph.layernormPure (α := α) (seqLen := seqLen) (embedDim := embedDim)
        (Tensor.reshapeSpec (α := α) (s₁ := s)
          (s₂ := .dim seqLen (.dim embedDim .scalar)) x hNumel)
        =
        Except.ok y2D) :
    Graph.evalAt (α := α) (g := unaryGraphOut (.layernorm axis) s s)
        (payload := {})
        (input := DVal.mk (α := α) s x)
        (vals := #[DVal.mk (α := α) s x]) (i := 1)
      =
      Except.ok
        (DVal.mk (α := α) s
          (Tensor.reshapeSpec (α := α)
            (s₁ := .dim seqLen (.dim embedDim .scalar)) (s₂ := s) y2D hNumel.symm)) := by
  simp [Graph.evalAt, unaryGraphOut, unaryNodeOut, Graph.getNode, Graph.getNode?,
    Graph.expectShape, hParams, hNumel, hPure, Bind.bind, Except.bind, Pure.pure, Except.pure]

end IRStep

end Correctness

end NN.Verification.TorchLean.Proved
