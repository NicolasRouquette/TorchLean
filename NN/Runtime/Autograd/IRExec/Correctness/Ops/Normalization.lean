/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Correctness.SemanticEquivalenceCommon

/-!
# Normalization

Normalization correctness lemmas for the IR-to-forward-executor lowering.

TorchLean’s LayerNorm correctness argument has two layers:

* the spec layer (`Spec.layerNorm`) defines the mathematical normalization over a tensor axis,
  matching the original Layer Normalization formulation from Ba et al. (2016) and the public
  PyTorch `LayerNorm` API;
* the runtime/lowering pass layer (`IRExec.Internal.buildFrom`) lowers `.layernorm axis` IR nodes
  into SSA nodes
  whose `forward` closure computes the same result on the forward-graph execution path.

This file proves the forward-correctness lemmas for LayerNorm and eval-mode BatchNorm2d lowering:
when `buildFrom` succeeds at IR node position `i`, the IR evaluator and the forward-graph evaluator
append the same output tensor.

The proof is shape-driven: it follows the same dependent matches and checks as the lowering pass, so
failed preconditions discharge as contradictions.

References:
* Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton, "Layer Normalization", arXiv:1607.06450.
* PyTorch `LayerNorm` documentation:
  https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html

## Main definitions

- `buildFrom_denoteAllFrom_layernorm`: correctness step for `.layernorm axis` lowering.
- `buildFrom_denoteAllFrom_batchNorm2dNchwEval`: correctness step for eval-mode NCHW BatchNorm2d.

## Implementation notes

- This proof is shape-driven and follows the same checks as lowering, which keeps it
  maintainable as layernorm contracts evolve.
- Branches that fail preconditions are discharged as contradictions close to where they arise,
  keeping the successful path readable.
- LayerNorm carries axis constraints through both the shape discipline and the tensor computation.
  Keep axis-validity and shape-cast facts in small helper lemmas so the semantic theorem stays
  focused on agreement between the lowered node and the evaluator.

## Tags

layernorm, correctness, ir, runtime, semantic equivalence
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace IRExec

open Spec
open Tensor
open Proofs.Autograd.Algebra
open NN.IR
open Internal

/-- Correctness lemma for the `.layernorm` node lowering pass. -/
theorem buildFrom_denoteAllFrom_layernorm
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node)
    (axis : Nat)
    (hN : g.getNode i = .ok n) (hk : n.kind = .layernorm axis) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.PackedTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.PackedTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (Spec.PackedTensor α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TList α ([inShape] ++ ss) :=
    ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)
  let input : Spec.PackedTensor α := Spec.PackedTensor.mk (α := α) inShape x

  -- Unfold the lowering pass step and specialize to the `.layernorm axis` branch.
  unfold buildFrom at hBuild
  simp [hi, hN] at hBuild
  simp (config := { failIfUnchanged := false }) [hk] at hBuild

  -- `layernorm` is unary.
  cases hp : n.parents with
  | nil =>
      exact False.elim <| throw_bind_ne_ok (by simpa [hp] using hBuild)
  | cons pId ps =>
      cases ps with
      | cons _ _ =>
          exact False.elim <| throw_bind_ne_ok (by simpa [hp] using hBuild)
      | nil =>
          -- Compute the 2D view parameters used by the lowering pass and the IR evaluator.
          cases hParams : OpContracts.layerNormMatrixDims axis n.outShape with
          | error msg =>
              exact False.elim <| throw_bind_ne_ok (by simpa [hp, hParams] using hBuild)
          | ok p =>
              rcases p with ⟨seqLen, embedDim⟩
              let view2d : Shape := .dim seqLen (.dim embedDim .scalar)

              by_cases hNumel : Spec.Shape.size n.outShape = Spec.Shape.size view2d
              · by_cases hSeq : seqLen > 0
                · by_cases hEmb : embedDim > 0
                  · cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId n.outShape with
                    | error msg =>
                        have hFalse : False := by
                          simp [hp, hParams, view2d, hNumel, hSeq, hEmb, hIdx] at hBuild
                        cases hFalse
                    | ok ip =>
                        -- Reduce `hBuild` to the recursive lowering call.
                        simp [hp, hParams, view2d, hNumel, hSeq, hEmb, hIdx] at hBuild

                        let gamma : Tensor α (.dim embedDim .scalar) :=
                          Spec.fill (α := α) 1 (.dim embedDim .scalar)
                        let beta : Tensor α (.dim embedDim .scalar) :=
                          Spec.fill (α := α) 0 (.dim embedDim .scalar)
                        let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                          mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                            let x : Tensor α n.outShape := getIdx (α := α) (xs := ctx) ip
                            let x2d : Tensor α view2d :=
                              Tensor.reshapeSpec (α := α) (s₁ := n.outShape) (s₂ := view2d) x
                                hNumel
                            let y2d : Tensor α view2d :=
                              Spec.layerNorm (α := α) (seqLen := seqLen) (embedDim := embedDim)
                                (x := x2d) (gamma := gamma) (beta := beta)
                                (h_seq_pos := hSeq) (h_embed_pos := hEmb)
                            Tensor.reshapeSpec (α := α) (s₁ := view2d) (s₂ := n.outShape) y2d
                              hNumel.symm)
                        let st1 : State α inShape :=
                          ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩

                        have hRec :
                            buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                                (i := i + 1) st1 = .ok st' := by
                          simpa [st1, nodeData, gamma, beta] using hBuild

                        have hGet :
                            vals0[pId]? = some (Spec.PackedTensor.mk (α := α) n.outShape
                                (getIdx (α := α) (xs := ctx) ip)) := by
                          simpa [vals0, ctx] using
                            (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                              (gd := gd) (x := x) (pid := pId) (s := n.outShape) (idx := ip) hIdx)

                        have hLN :
                            NN.IR.Graph.layerNormWithoutAffine (α := α) (seqLen := seqLen) (embedDim :=
                              embedDim)
                                (Tensor.reshapeSpec (α := α) (s₁ := n.outShape) (s₂ := view2d)
                                  (getIdx (α := α) (xs := ctx) ip) hNumel) =
                              .ok
                                (Spec.layerNorm (α := α) (seqLen := seqLen) (embedDim := embedDim)
                                  (x := Tensor.reshapeSpec (α := α) (s₁ := n.outShape) (s₂ :=
                                    view2d)
                                    (getIdx (α := α) (xs := ctx) ip) hNumel)
                                  (gamma := gamma) (beta := beta)
                                  (h_seq_pos := hSeq) (h_embed_pos := hEmb)) := by
                          -- This operation is exactly `Spec.layerNorm` with `gamma=1` and
                          -- `beta=0`.
                          simp [NN.IR.Graph.layerNormWithoutAffine, hSeq, hEmb, gamma, beta]
                          rfl

                        have hEval :
                            NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                                (input := input) (vals := vals0) (i := i) =
                              .ok (Spec.PackedTensor.mk (α := α) n.outShape (nodeData.eval ctx)) :=
                                by
                          -- Focused simplification of the `.layernorm` branch of the evaluator.
                          simp (config := { failIfUnchanged := false })
                            [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                              NN.IR.Graph.normalizeNodeOutput, hN, hk, hp, hGet, hParams,
                              view2d, hNumel,
                              throw_eq_error,
                              nodeData, mkForwardNode]
                          simpa using
                            congrArg
                              (fun e =>
                                (fun a : Tensor α view2d =>
                                  Spec.PackedTensor.mk (α := α) n.outShape
                                    (Tensor.reshapeSpec (α := α) (s₁ := view2d) (s₂ := n.outShape)
                                      a hNumel.symm)) <$> e)
                              hLN

                        have hStep :
                            denoteAllState (α := α) inShape st1 x =
                              vals0.push
                                (Spec.PackedTensor.mk (α := α) n.outShape (nodeData.eval ctx)) := by
                          simpa [vals0, st1, nodeData, ctx] using
                            (denoteAllState_snoc (α := α) (inShape := inShape) (ss := ss)
                              (τ := n.outShape) (gd := gd) (nodeData := nodeData) (x := x))

                        have hTail := ih st1 hRec
                        exact buildFrom_denoteAllFrom_finish (α := α) (g := g) (payload := payload)
                          (i := i) (x := x) (hi := hi) (τ := n.outShape)
                          (nodeData := nodeData) (st1 := st1) (st' := st')
                          (ctx := ctx) (vals0 := vals0) (input := input) hTail hEval hStep
                  · exact False.elim <|
                      throw_bind_ne_ok (by simpa [hp, hParams, view2d, hNumel, hSeq, hEmb] using
                        hBuild)
                · exact False.elim <|
                    throw_bind_ne_ok (by simpa [hp, hParams, view2d, hNumel, hSeq] using hBuild)
              · exact False.elim <|
                  throw_bind_ne_ok (by simpa [hp, hParams, view2d, hNumel] using hBuild)

/-- Correctness lemma for eval-mode NCHW BatchNorm2d lowering. -/
theorem buildFrom_denoteAllFrom_batchNorm2dNchwEval
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node) (channels : Nat)
    (hN : g.getNode i = .ok n) (hk : n.kind = .batchNorm2dNchwEval channels)
    (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.PackedTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.PackedTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape
        (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (Spec.PackedTensor α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TList α ([inShape] ++ ss) :=
    ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)
  let input : Spec.PackedTensor α := Spec.PackedTensor.mk (α := α) inShape x

  unfold buildFrom at hBuild
  simp [hi, hN] at hBuild
  simp (config := { failIfUnchanged := false }) [hk] at hBuild
  cases hp : n.parents with
  | nil =>
      exact False.elim <| throw_bind_ne_ok (by simpa [hp] using hBuild)
  | cons pId ps =>
      cases ps with
      | cons _ _ =>
          exact False.elim <| throw_bind_ne_ok (by simpa [hp] using hBuild)
      | nil =>
          cases hCfg : payload.batchNorm2dNchwEval? n.id with
          | none =>
              exact False.elim <| throw_bind_ne_ok (by simpa [hp, hCfg] using hBuild)
          | some cfg =>
              simp [hp, hCfg] at hBuild
              split at hBuild
              next nBatch c height width hShape =>
                by_cases hChannels : c = cfg.c
                · simp [hChannels] at hBuild
                  let expected : Shape :=
                    .dim nBatch (.dim cfg.c (.dim height (.dim width .scalar)))
                  cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId expected with
                  | error msg =>
                      have : False := by
                        simp [expected, hIdx] at hBuild
                      exact False.elim this
                  | ok ip =>
                      by_cases hOut : expected = n.outShape
                      · simp [expected, hIdx, hOut] at hBuild
                        let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                          mkForwardNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                            let parent := getIRValue (α := α) (ctx := ctx) ip
                            let y : Tensor α expected :=
                              NN.IR.Graph.batchNorm2dEvalTensor (α := α) cfg parent
                            hOut ▸ y)
                        let st1 : State α inShape :=
                          ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                        have hRec :
                            buildFrom (α := α) (g := g) (payload := payload)
                              (inShape := inShape) (i := i + 1) st1 = .ok st' := by
                          convert hBuild using 1
                          all_goals simp [st1, nodeData]
                        have hGet :
                            vals0[pId]? = some (Spec.PackedTensor.mk (α := α) expected
                              (getIdx (α := α) (xs := ctx) ip)) := by
                          simpa [vals0, ctx, expected] using
                            (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                              (gd := gd) (x := x) (pid := pId) (s := expected) (idx := ip) hIdx)
                        let normalized : Tensor α expected :=
                          NN.IR.Graph.batchNorm2dEvalTensor (α := α) cfg
                            (getIdx (α := α) (xs := ctx) ip)
                        have hBatch :
                            NN.IR.Graph.evalBatchNorm2dNchwEval (α := α) (payload := payload)
                                (id := n.id)
                                (x := Spec.PackedTensor.mk (α := α) expected
                                  (getIdx (α := α) (xs := ctx) ip)) =
                              .ok (Spec.PackedTensor.mk (α := α) expected normalized) := by
                          simp [NN.IR.Graph.evalBatchNorm2dNchwEval, hCfg, expected,
                            normalized, NN.IR.Graph.expectShape]
                          rfl
                        have hEval :
                            NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                                (input := input) (vals := vals0) (i := i) =
                              .ok (Spec.PackedTensor.mk (α := α) n.outShape
                                (nodeData.eval ctx)) := by
                          simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                            NN.IR.Graph.normalizeNodeOutput, hN, hk, hp, hGet, hBatch,
                            expected, hOut, normalized, nodeData, getIRValue, shape_bne_refl]
                          congr 1
                        have hStep :
                            denoteAllState (α := α) inShape st1 x =
                              vals0.push (Spec.PackedTensor.mk (α := α) n.outShape
                                (nodeData.eval ctx)) := by
                          simpa [vals0, st1, nodeData, ctx] using
                            (denoteAllState_snoc (α := α) (inShape := inShape) (ss := ss)
                              (τ := n.outShape) (gd := gd) (nodeData := nodeData) (x := x))
                        have hTail := ih st1 hRec
                        exact buildFrom_denoteAllFrom_finish (α := α) (g := g)
                          (payload := payload) (i := i) (x := x) (hi := hi)
                          (τ := n.outShape) (nodeData := nodeData) (st1 := st1) (st' := st')
                          (ctx := ctx) (vals0 := vals0) (input := input) hTail hEval hStep
                      · exact False.elim <| throw_bind_ne_ok
                          (by simpa [expected, hIdx, hOut] using hBuild)
                · exact False.elim <| throw_bind_ne_ok
                    (by simpa [hChannels] using hBuild)
              next hShape =>
                exact False.elim <| throw_bind_ne_ok (by simpa using hBuild)

end IRExec
end Autograd
end Runtime
