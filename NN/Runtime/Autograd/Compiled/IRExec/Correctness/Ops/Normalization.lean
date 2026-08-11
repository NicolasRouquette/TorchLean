/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Compiled.IRExec.Correctness.SemanticEquivalenceCommon

/-!
# Normalization

Normalization correctness lemmas for the IR -> compiled runtime bridge.

TorchLean’s LayerNorm correctness argument has two layers:

* the spec layer (`Spec.layerNorm`) defines the mathematical normalization over a tensor axis,
  matching the original Layer Normalization formulation from Ba et al. (2016) and the public
  PyTorch `LayerNorm` API;
* the runtime/compiler layer (`IRExec.buildFrom`) lowers `.layernorm axis` IR nodes into SSA nodes
  whose `forward` closure computes the same result on the compiled execution path.

This file proves the forward-correctness lemmas for LayerNorm and eval-mode BatchNorm2d compilation:
when `buildFrom` succeeds at IR node position `i`, the IR evaluator and the compiled evaluator
append the same output tensor.

The proof is shape-driven: it follows the same dependent matches and checks as the compiler, so
failed preconditions discharge as contradictions.

References:
* Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton, "Layer Normalization", arXiv:1607.06450.
* PyTorch `LayerNorm` documentation:
  https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html

## Main definitions

- `buildFrom_denoteAllFrom_layernorm`: correctness step for `.layernorm axis` lowering.
- `buildFrom_denoteAllFrom_batchNorm2dNchwEval`: correctness step for eval-mode NCHW BatchNorm2d.

## Implementation notes

- This proof is shape-driven and follows the same checks as compilation, which keeps it
  maintainable as layernorm contracts evolve.
- Branches that fail preconditions are discharged as contradictions close to where they arise,
  keeping the successful path readable.
- LayerNorm carries axis constraints through both the shape discipline and the tensor computation.
  Keep axis-validity and shape-cast facts in small helper lemmas so the semantic theorem stays
  focused on the compiled/evaluator agreement.

## Tags

layernorm, correctness, ir, runtime, semantic equivalence
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Compiled

open Spec
open Tensor
open Proofs.Autograd.Algebra
open NN.IR
open IRExec

/-- Correctness lemma for the `.layernorm` node compiler. -/
theorem buildFrom_denoteAllFrom_layernorm
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
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
          (input := NN.IR.DVal.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := NN.IR.DVal.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (NN.IR.DVal α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TList α ([inShape] ++ ss) :=
    GraphData.eval (α := α) (Δ := Unit) (Γ := [inShape]) (ss := ss) gd (.cons x .nil) ()
  let input : NN.IR.DVal α := NN.IR.DVal.mk (α := α) inShape x

  -- Unfold the compiler step and specialize to the `.layernorm axis` branch.
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
          -- Compute the 2D view parameters used by the compiler and the IR evaluator.
          cases hParams : OpContracts.layerNorm2DParams axis n.outShape with
          | error msg =>
              exact False.elim <| throw_bind_ne_ok (by simpa [hp, hParams] using hBuild)
          | ok p =>
              rcases p with ⟨seqLen, embedDim⟩
              let view2D : Shape := .dim seqLen (.dim embedDim .scalar)

              by_cases hNumel : Spec.Shape.size n.outShape = Spec.Shape.size view2D
              · by_cases hSeq : seqLen > 0
                · by_cases hEmb : embedDim > 0
                  · cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId n.outShape with
                    | error msg =>
                        have hFalse : False := by
                          simp [hp, hParams, view2D, hNumel, hSeq, hEmb, hIdx] at hBuild
                        cases hFalse
                    | ok ip =>
                        -- Reduce `hBuild` to the recursive compilation call.
                        simp [hp, hParams, view2D, hNumel, hSeq, hEmb, hIdx] at hBuild

                        let gamma : Tensor α (.dim embedDim .scalar) :=
                          Spec.fill (α := α) 1 (.dim embedDim .scalar)
                        let beta : Tensor α (.dim embedDim .scalar) :=
                          Spec.fill (α := α) 0 (.dim embedDim .scalar)
                        let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape :=
                          mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
                            let x : Tensor α n.outShape := getIdx (α := α) (xs := ctx) ip
                            let x2D : Tensor α view2D :=
                              Tensor.reshapeSpec (α := α) (s₁ := n.outShape) (s₂ := view2D) x
                                hNumel
                            let y2D : Tensor α view2D :=
                              Spec.layerNorm (α := α) (seqLen := seqLen) (embedDim := embedDim)
                                (x := x2D) (gamma := gamma) (beta := beta)
                                (h_seq_pos := hSeq) (h_embed_pos := hEmb)
                            Tensor.reshapeSpec (α := α) (s₁ := view2D) (s₂ := n.outShape) y2D
                              hNumel.symm)
                        let st1 : State α inShape :=
                          ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩

                        have hRec :
                            buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
                                (i := i + 1) st1 = .ok st' := by
                          simpa [st1, nodeData, gamma, beta] using hBuild

                        have hGet :
                            vals0[pId]! =
                              NN.IR.DVal.mk (α := α) n.outShape
                                (getIdx (α := α) (xs := ctx) ip) := by
                          simpa [vals0, ctx] using
                            (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                              (gd := gd) (x := x) (pid := pId) (s := n.outShape) (idx := ip) hIdx)

                        have hExp :
                            NN.IR.Graph.expectShape (α := α) (expected := n.outShape) vals0[pId]! =
                              .ok (getIdx (α := α) (xs := ctx) ip) := by
                          simp [hGet, Graph.expectShape_sigma, NN.IR.DVal.mk]

                        have hLN :
                            NN.IR.Graph.layernormPure (α := α) (seqLen := seqLen) (embedDim :=
                              embedDim)
                                (Tensor.reshapeSpec (α := α) (s₁ := n.outShape) (s₂ := view2D)
                                  (getIdx (α := α) (xs := ctx) ip) hNumel) =
                              .ok
                                (Spec.layerNorm (α := α) (seqLen := seqLen) (embedDim := embedDim)
                                  (x := Tensor.reshapeSpec (α := α) (s₁ := n.outShape) (s₂ :=
                                    view2D)
                                    (getIdx (α := α) (xs := ctx) ip) hNumel)
                                  (gamma := gamma) (beta := beta)
                                  (h_seq_pos := hSeq) (h_embed_pos := hEmb)) := by
                          -- `layernormPure` is exactly `Spec.layerNorm` with `gamma=1` and
                          -- `beta=0`.
                          simp [NN.IR.Graph.layernormPure, hSeq, hEmb, gamma, beta]
                          rfl

                        have hEval :
                            NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                                (input := input) (vals := vals0) (i := i) =
                              .ok (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) :=
                                by
                          -- Focused simplification of the `.layernorm` branch of the evaluator.
                          simp (config := { failIfUnchanged := false })
                            [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput, hN, hk, hp, hExp, hParams, view2D, hNumel,
                              NN.IR.DVal.shape, NN.IR.DVal.tensor, NN.IR.DVal.mk,
                              throw_eq_error,
                              nodeData, mkFwdNode]
                          simpa [NN.IR.DVal.mk] using
                            congrArg
                              (fun e =>
                                (fun a : Tensor α view2D =>
                                  NN.IR.DVal.mk (α := α) n.outShape
                                    (Tensor.reshapeSpec (α := α) (s₁ := view2D) (s₂ := n.outShape)
                                      a hNumel.symm)) <$> e)
                              hLN

                        have hStep :
                            denoteAllState (α := α) inShape st1 x =
                              vals0.push
                                (NN.IR.DVal.mk (α := α) n.outShape (nodeData.forward ctx ())) := by
                          simpa [vals0, st1, nodeData, ctx] using
                            (denoteAllState_snoc (α := α) (inShape := inShape) (ss := ss)
                              (τ := n.outShape) (gd := gd) (nodeData := nodeData) (x := x))

                        have hTail := ih st1 hRec
                        exact buildFrom_denoteAllFrom_finish (α := α) (g := g) (payload := payload)
                          (i := i) (x := x) (hi := hi) (τ := n.outShape)
                          (nodeData := nodeData) (st1 := st1) (st' := st')
                          (ctx := ctx) (vals0 := vals0) (input := input) hTail hEval hStep
                  · exact False.elim <|
                      throw_bind_ne_ok (by simpa [hp, hParams, view2D, hNumel, hSeq, hEmb] using
                        hBuild)
                · exact False.elim <|
                    throw_bind_ne_ok (by simpa [hp, hParams, view2D, hNumel, hSeq] using hBuild)
              · exact False.elim <|
                  throw_bind_ne_ok (by simpa [hp, hParams, view2D, hNumel] using hBuild)

/-- Correctness lemma for eval-mode NCHW BatchNorm2d lowering. -/
theorem buildFrom_denoteAllFrom_batchNorm2dNchwEval
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (i : Nat) (st' : State α inShape)
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
          (input := NN.IR.DVal.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := NN.IR.DVal.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape
        (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 : Array (NN.IR.DVal α) :=
    denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx : TList α ([inShape] ++ ss) :=
    GraphData.eval (α := α) (Δ := Unit) (Γ := [inShape]) (ss := ss) gd (.cons x .nil) ()
  let input : NN.IR.DVal α := NN.IR.DVal.mk (α := α) inShape x

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
                        let nodeData : NodeData α Unit ([inShape] ++ ss) n.outShape :=
                          mkFwdNode (α := α) (Γ := [inShape] ++ ss) (τ := n.outShape) (fun ctx =>
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
                            vals0[pId]! = NN.IR.DVal.mk (α := α) expected
                              (getIdx (α := α) (xs := ctx) ip) := by
                          simpa [vals0, ctx, expected] using
                            (denoteAllState_get_mkIdx (inShape := inShape) (ss := ss)
                              (gd := gd) (x := x) (pid := pId) (s := expected) (idx := ip) hIdx)
                        let normalized : Tensor α expected :=
                          NN.IR.Graph.batchNorm2dEvalTensor (α := α) cfg
                            (getIdx (α := α) (xs := ctx) ip)
                        have hBatch :
                            NN.IR.Graph.evalBatchNorm2DNchwEval (α := α) (payload := payload)
                                (id := n.id) (x := vals0[pId]!) =
                              .ok (NN.IR.DVal.mk (α := α) expected normalized) := by
                          have hCast :=
                            dval_snd_cast_of_eq_mk (α := α) (v := vals0[pId]!)
                              (s := expected) (t := getIdx (α := α) (xs := ctx) ip) hGet
                          simp [NN.IR.Graph.evalBatchNorm2DNchwEval, hCfg, hGet, expected,
                            normalized, NN.IR.Graph.expectShape, NN.IR.DVal.shape,
                            NN.IR.DVal.tensor, NN.IR.DVal.mk, hCast]
                          rfl
                        have hEval :
                            NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                                (input := input) (vals := vals0) (i := i) =
                              .ok (NN.IR.DVal.mk (α := α) n.outShape
                                (nodeData.forward ctx ())) := by
                          simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput, hN, hk, hp, hBatch, expected, hOut,
                            normalized, nodeData, getIRValue, shape_bne_refl, NN.IR.DVal.shape,
                            NN.IR.DVal.tensor, NN.IR.DVal.mk]
                          congr 1
                        have hStep :
                            denoteAllState (α := α) inShape st1 x =
                              vals0.push (NN.IR.DVal.mk (α := α) n.outShape
                                (nodeData.forward ctx ())) := by
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

end Compiled
end Autograd
end Runtime
