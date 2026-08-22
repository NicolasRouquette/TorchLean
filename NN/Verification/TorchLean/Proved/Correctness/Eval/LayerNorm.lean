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

/-- Affine-free IR LayerNorm is exactly spec LayerNorm with unit scale and zero bias. -/
theorem layerNormWithoutAffine_eq_spec
    {α : Type} [Context α]
    (seqLen embedDim : Nat)
    (x : Tensor α (.dim seqLen (.dim embedDim .scalar)))
    (hSeq : seqLen > 0)
    (hEmb : embedDim > 0) :
    Graph.layerNormWithoutAffine (α := α) (seqLen := seqLen) (embedDim := embedDim) x
      =
      Except.ok
        (Spec.layerNorm (α := α) (seqLen := seqLen) (embedDim := embedDim)
          (x := x)
          (gamma := Spec.fill (α := α) 1 (.dim embedDim .scalar))
          (beta := Spec.fill (α := α) 0 (.dim embedDim .scalar))
          (h_seq_pos := hSeq) (h_embed_pos := hEmb)) := by
  simp [Graph.layerNormWithoutAffine, hSeq, hEmb]
  rfl

/--
Local IR semantics for `layernorm axis`.

The hypotheses are the same contracts checked by the evaluator: `axis` must produce a valid
`(seqLen, embedDim)` view, the reshape must preserve element count, and the affine-free LayerNorm step
must succeed.
-/
theorem evalAt_layernorm_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s : Shape} (axis seqLen embedDim : Nat)
    (x : Tensor α s)
    (hParams : OpContracts.layerNormMatrixDims axis s = .ok (seqLen, embedDim))
    (hNumel : Spec.Shape.size s = Spec.Shape.size (.dim seqLen (.dim embedDim .scalar)))
    (y2d : Tensor α (.dim seqLen (.dim embedDim .scalar)))
    (hLayerNorm :
      Graph.layerNormWithoutAffine (α := α) (seqLen := seqLen) (embedDim := embedDim)
        (Tensor.reshapeSpec (α := α) (s₁ := s)
          (s₂ := .dim seqLen (.dim embedDim .scalar)) x hNumel)
        =
        Except.ok y2d) :
    Graph.evalAt (α := α) (g := unaryGraphOut (.layernorm axis) s s)
        (payload := {})
        (input := Spec.PackedTensor.mk (α := α) s x)
        (vals := #[Spec.PackedTensor.mk (α := α) s x]) (i := 1)
      =
      Except.ok
        (Spec.PackedTensor.mk (α := α) s
          (Tensor.reshapeSpec (α := α)
            (s₁ := .dim seqLen (.dim embedDim .scalar)) (s₂ := s) y2d hNumel.symm)) := by
  simp [Graph.evalAt, Graph.evalNode, Graph.normalizeNodeOutput, unaryGraphOut, unaryNodeOut, Graph.getNode, Graph.getNode?,
    Graph.expectShape, hParams, hNumel, hLayerNorm, Bind.bind, Except.bind, Pure.pure, Except.pure]

end IRStep

end Correctness

end NN.Verification.TorchLean.Proved
