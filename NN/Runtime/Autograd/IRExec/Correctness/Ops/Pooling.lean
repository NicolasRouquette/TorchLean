/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Correctness.SemanticEquivalenceCommon

/-!
# Pooling Correctness

Semantic preservation for lowering arbitrary-rank max and average pooling. A `WindowConfig`
selects a spatial suffix and gives a separate kernel, stride, and symmetric padding for every
pooled axis. All leading axes are preserved, including any number of batch dimensions.

Shape inference, denotational evaluation, and executable lowering share `OpContracts.PoolPlan`.
Consequently, successful lowering records exactly the rank split and positivity evidence consumed
by the typed pooling specification; there are no separate padded or two-dimensional proof paths.
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

/-- Lowering arbitrary-rank max pooling preserves the IR denotation. -/
theorem buildFrom_denoteAllFrom_maxPool
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node) (config : WindowConfig)
    (hN : g.getNode i = .ok n) (hk : n.kind = .maxPool config) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.SomeTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.SomeTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape
        (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx := ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)
  unfold buildFrom at hBuild
  simp [hi, hN] at hBuild
  simp (config := { failIfUnchanged := false }) [hk] at hBuild
  cases hp : unaryParent? n.parents with
  | none => exact False.elim <| throw_bind_ne_ok (by simpa [hp] using hBuild)
  | some pId =>
          cases hParent : g.getNode pId with
          | error message => simp [hp, hParent] at hBuild
          | ok parentNode =>
              let parentShape := parentNode.outShape
              cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId parentShape with
              | error message => simp [hp, hParent, parentShape, hIdx] at hBuild
              | ok parentIdx =>
                  cases hPlan : OpContracts.planPool "max_pool" config parentShape with
                  | error message => simp [hp, hParent, parentShape, hIdx, hPlan] at hBuild
                  | ok plan =>
                      let expected := plan.outShape
                      by_cases hOut : expected = n.outShape
                      · simp [hp, hParent, parentShape, hIdx, hPlan, expected, hOut] at hBuild
                        let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                          mkForwardNode (fun context =>
                            let parent := getIdx (α := α) (xs := context) parentIdx
                            packedResultOrPanic (α := α) n.outShape <|
                              NN.IR.Graph.evalMaxPool config (Spec.SomeTensor.ofTensor parent))
                        let st1 : State α inShape :=
                          ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                        have hRec :
                            buildFrom (α := α) (g := g) (payload := payload)
                              (inShape := inShape) (i := i + 1) st1 = .ok st' := by
                          simpa [st1, nodeData] using hBuild
                        have hGet :
                            vals0[pId]? = some (Spec.SomeTensor.mk (α := α) parentShape
                              (getIdx (α := α) (xs := ctx) parentIdx)) := by
                          simpa [vals0, ctx, parentShape] using
                            (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                              (gd := gd) (x := x) (pid := pId) (s := parentShape)
                              (idx := parentIdx) hIdx)
                        let parent := getIdx (α := α) (xs := ctx) parentIdx
                        let input : Tensor α
                            (plan.leading.concat (Shape.ofList plan.spatial.toList)) :=
                          plan.concat_eq.symm ▸ parent
                        let layer : Spec.MaxPoolSpec config.spatialRank config.kernel config.stride
                            config.padding plan.kernelNonzero plan.strideNonzero := {}
                        let output := Tensor.mapEach plan.leading
                          (Spec.maxPoolSpatialSpec (α := α) (inSpatial := plan.spatial) layer) input
                        have hPool :
                            NN.IR.Graph.evalMaxPool config
                                (Spec.SomeTensor.mk (α := α) parentShape parent) =
                              .ok (Spec.SomeTensor.ofTensor output) := by
                          simp [NN.IR.Graph.evalMaxPool, hPlan, input, layer, output]
                        have hOutputShape : (Spec.SomeTensor.ofTensor output).shape = expected := by
                          rfl
                        have hResultShape : (Spec.SomeTensor.ofTensor output).shape = n.outShape :=
                          hOutputShape.trans hOut
                        have hEval :
                            NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                                (input := Spec.SomeTensor.mk (α := α) inShape x)
                                (vals := vals0) (i := i) =
                              .ok (Spec.SomeTensor.mk (α := α) n.outShape
                                (nodeData.eval ctx)) := by
                          simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                            NN.IR.Graph.normalizeNodeOutput, hN, hk, hp, hGet, hPool,
                            nodeData, parent, packedResultOrPanic]
                          split
                          · simp_all
                            congr 2
                          · simp_all
                        apply buildFrom_denoteAllFrom_nodeData_exact (α := α) (g := g)
                          (payload := payload) (gd := gd) (i := i) (st' := st') (x := x)
                          (hi := hi) (nodeData := nodeData)
                        · exact ih st1 hRec
                        · exact hEval
                      · exact False.elim <| throw_bind_ne_ok <| by
                          simpa [hp, hParent, parentShape, hIdx, hPlan, expected, hOut] using hBuild

/-- Lowering arbitrary-rank average pooling preserves the IR denotation. -/
theorem buildFrom_denoteAllFrom_avgPool
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node) (config : WindowConfig)
    (hN : g.getNode i = .ok n) (hk : n.kind = .avgPool config) (hi : i < g.nodes.size)
    (hBuild :
      buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
        (i := i) (st := (⟨ss, gd⟩ : State α inShape)) = .ok st')
    (ih :
      ∀ (st1 : State α inShape),
        buildFrom (α := α) (g := g) (payload := payload) (inShape := inShape)
          (i := i + 1) st1 = .ok st' →
        NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := Spec.SomeTensor.mk (α := α) inShape x)
          (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
          .ok (denoteAllState (α := α) inShape st' x)) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
      (input := Spec.SomeTensor.mk (α := α) inShape x)
      (i := i) (vals := denoteAllState (α := α) inShape
        (st := (⟨ss, gd⟩ : State α inShape)) x) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  let vals0 := denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x
  let ctx := ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)
  unfold buildFrom at hBuild
  simp [hi, hN] at hBuild
  simp (config := { failIfUnchanged := false }) [hk] at hBuild
  cases hp : unaryParent? n.parents with
  | none => exact False.elim <| throw_bind_ne_ok (by simpa [hp] using hBuild)
  | some pId =>
          cases hParent : g.getNode pId with
          | error message => simp [hp, hParent] at hBuild
          | ok parentNode =>
              let parentShape := parentNode.outShape
              cases hIdx : mkIdx (inShape := inShape) (ss := ss) pId parentShape with
              | error message => simp [hp, hParent, parentShape, hIdx] at hBuild
              | ok parentIdx =>
                  cases hPlan : OpContracts.planPool "avg_pool" config parentShape with
                  | error message => simp [hp, hParent, parentShape, hIdx, hPlan] at hBuild
                  | ok plan =>
                      let expected := plan.outShape
                      by_cases hOut : expected = n.outShape
                      · simp [hp, hParent, parentShape, hIdx, hPlan, expected, hOut] at hBuild
                        let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                          mkForwardNode (fun context =>
                            let parent := getIdx (α := α) (xs := context) parentIdx
                            packedResultOrPanic (α := α) n.outShape <|
                              NN.IR.Graph.evalAvgPool config (Spec.SomeTensor.ofTensor parent))
                        let st1 : State α inShape :=
                          ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                        have hRec :
                            buildFrom (α := α) (g := g) (payload := payload)
                              (inShape := inShape) (i := i + 1) st1 = .ok st' := by
                          simpa [st1, nodeData] using hBuild
                        have hGet :
                            vals0[pId]? = some (Spec.SomeTensor.mk (α := α) parentShape
                              (getIdx (α := α) (xs := ctx) parentIdx)) := by
                          simpa [vals0, ctx, parentShape] using
                            (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                              (gd := gd) (x := x) (pid := pId) (s := parentShape)
                              (idx := parentIdx) hIdx)
                        let parent := getIdx (α := α) (xs := ctx) parentIdx
                        let input : Tensor α
                            (plan.leading.concat (Shape.ofList plan.spatial.toList)) :=
                          plan.concat_eq.symm ▸ parent
                        let layer : Spec.AvgPoolSpec config.spatialRank config.kernel config.stride
                            config.padding plan.kernelNonzero plan.strideNonzero := {}
                        let output := Tensor.mapEach plan.leading
                          (Spec.avgPoolSpatialSpec (α := α) (inSpatial := plan.spatial) layer) input
                        have hPool :
                            NN.IR.Graph.evalAvgPool config
                                (Spec.SomeTensor.mk (α := α) parentShape parent) =
                              .ok (Spec.SomeTensor.ofTensor output) := by
                          simp [NN.IR.Graph.evalAvgPool, hPlan, input, layer, output]
                        have hOutputShape : (Spec.SomeTensor.ofTensor output).shape = expected := by
                          rfl
                        have hResultShape : (Spec.SomeTensor.ofTensor output).shape = n.outShape :=
                          hOutputShape.trans hOut
                        have hEval :
                            NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                                (input := Spec.SomeTensor.mk (α := α) inShape x)
                                (vals := vals0) (i := i) =
                              .ok (Spec.SomeTensor.mk (α := α) n.outShape
                                (nodeData.eval ctx)) := by
                          simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                            NN.IR.Graph.normalizeNodeOutput, hN, hk, hp, hGet, hPool,
                            nodeData, parent, packedResultOrPanic]
                          split
                          · simp_all
                            congr 2
                          · simp_all
                        apply buildFrom_denoteAllFrom_nodeData_exact (α := α) (g := g)
                          (payload := payload) (gd := gd) (i := i) (st' := st') (x := x)
                          (hi := hi) (nodeData := nodeData)
                        · exact ih st1 hRec
                        · exact hEval
                      · exact False.elim <| throw_bind_ne_ok <| by
                          simpa [hp, hParent, parentShape, hIdx, hPlan, expected, hOut] using hBuild

end IRExec
end Autograd
end Runtime
