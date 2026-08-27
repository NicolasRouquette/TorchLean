/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Correctness.SemanticEquivalenceCommon

/-!
# Convolution Correctness

Semantic preservation for arbitrary-rank convolution. `ConvConfig` records the channel axis and
one kernel, stride, and padding value for every spatial axis. Axes before the channel axis are
preserved, so the same theorem covers unbatched tensors and tensors with any number of leading
batch dimensions.
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

/-- Successful convolution evaluation agrees with the generalized typed convolution semantics. -/
theorem evalConv_eq_generalSpec
    {α : Type} [Context α]
    (payload : Payload α) (id : Nat) (config : ConvConfig) (params : ConvParams α)
    {parentShape : Shape} (parent : Tensor α parentShape) (leading outShape : Shape)
    (hInfer : OpContracts.inferConvConfigOutShape "conv" config parentShape =
      .ok outShape)
    (hParams : payload.conv? id = some params)
    (hMatches : params.matchesConfig config = true)
    (hLeading : Shape.ofList (parentShape.toList.take config.channelAxis) = leading)
    (hInput : parentShape = params.inputShape leading) :
    NN.IR.Graph.evalConv payload id config (Spec.SomeTensor.ofTensor parent) =
      .ok (Spec.SomeTensor.ofTensor <|
        Tensor.mapEach leading
          (Spec.groupedConvSpec (α := α) (stride := params.stride)
            (dilation := params.dilation) (paddingBefore := params.padding)
            (paddingAfter := params.paddingAfter) params.groups params.spec.kernel params.spec.bias)
          (hInput ▸ parent)) := by
  unfold NN.IR.Graph.evalConv
  simp only [Spec.SomeTensor.shape_ofTensor, Spec.SomeTensor.tensor_ofTensor]
  rw [hInfer]
  simp only [hParams, hMatches, if_true]
  rw [hLeading]
  split
  · congr
  · contradiction

set_option maxRecDepth 10000 in
/-- Lowering an arbitrary-rank convolution preserves the IR denotation. -/
theorem buildFrom_denoteAllFrom_conv
    {α : Type} [Context α] [shapeDecidable : DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (i : Nat) (st' : State α inShape)
    (x : Tensor α inShape) (n : NN.IR.Node) (config : ConvConfig)
    (hN : g.getNode i = .ok n) (hk : n.kind = .conv config) (hi : i < g.nodes.size)
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
  | some parentId =>
          cases hParent : g.getNode parentId with
          | error message => simp [hp, hParent] at hBuild
          | ok parentNode =>
              let parentShape := parentNode.outShape
              cases hIdx : mkIdx (inShape := inShape) (ss := ss) parentId parentShape with
              | error message => simp [hp, hParent, parentShape, hIdx] at hBuild
              | ok parentIdx =>
                  cases hExpected :
                      OpContracts.inferConvConfigOutShape "conv" config parentShape with
                  | error message =>
                      simp [hp, hParent, parentShape, hIdx, hExpected] at hBuild
                  | ok expected =>
                      cases hParams : payload.conv? n.id with
                      | none =>
                          exact False.elim <| throw_bind_ne_ok <| by
                            simpa [hp, hParent, parentShape, hIdx, hExpected, hParams] using hBuild
                      | some params =>
                          by_cases hMatches : params.matchesConfig config = true
                          · simp [hp, hParent, parentShape, hIdx, hExpected, hParams, hMatches]
                              at hBuild
                            let leading := Shape.ofList
                              (parentShape.toList.take config.channelAxis)
                            let payloadShape := params.inputShape leading
                            cases hInputDecision :
                                shapeDecidable parentNode.outShape
                                  (params.inputShape (Shape.ofList
                                    (parentNode.outShape.toList.take config.channelAxis))) with
                            | isFalse hInputRaw =>
                                rw [hInputDecision] at hBuild
                                exact False.elim <| throw_bind_ne_ok <| by
                                  simpa [hp, hParent, parentShape, hIdx, hExpected, hParams,
                                    hMatches] using hBuild
                            | isTrue hInputRaw =>
                                rw [hInputDecision] at hBuild
                                have hInput : parentShape = payloadShape := by
                                  simpa [parentShape, leading, payloadShape] using hInputRaw
                                by_cases hPayloadOut : params.outputShape leading = expected
                                · by_cases hOut : expected = n.outShape
                                  · simp [parentShape, leading, hPayloadOut, hOut] at hBuild
                                    let nodeData : ForwardNode α ([inShape] ++ ss) n.outShape :=
                                      mkForwardNode (fun context =>
                                        let parent := getIRValue (α := α) context parentIdx
                                        packedResultOrPanic (α := α) n.outShape <|
                                          NN.IR.Graph.evalConv payload n.id config
                                            (Spec.SomeTensor.ofTensor parent))
                                    let st1 : State α inShape :=
                                      ⟨ss ++ [n.outShape], .snoc (ss := ss) gd nodeData⟩
                                    have hRec :
                                        buildFrom (α := α) (g := g) (payload := payload)
                                            (inShape := inShape) (i := i + 1) st1 = .ok st' := by
                                      simpa [st1, nodeData, parentShape, getIRValue] using hBuild
                                    have hGet :
                                        vals0[parentId]? = some
                                          (Spec.SomeTensor.mk (α := α) parentShape
                                            (getIdx (α := α) (xs := ctx) parentIdx)) := by
                                      simpa [vals0, ctx, parentShape, getIRValue] using
                                        (denoteAllState_get_mkIdx? (inShape := inShape) (ss := ss)
                                          (gd := gd) (x := x) (pid := parentId) (s := parentShape)
                                          (idx := parentIdx) hIdx)
                                    let parent := getIdx (α := α) (xs := ctx) parentIdx
                                    let input : Tensor α payloadShape := hInput ▸ parent
                                    let output := Tensor.mapEach leading
                                      (Spec.groupedConvSpec (α := α) (stride := params.stride)
                                        (dilation := params.dilation)
                                        (paddingBefore := params.padding)
                                        (paddingAfter := params.paddingAfter) params.groups
                                        params.spec.kernel params.spec.bias) input
                                    have hConv :
                                        NN.IR.Graph.evalConv payload n.id config
                                            (Spec.SomeTensor.mk (α := α) parentShape parent) =
                                          .ok (Spec.SomeTensor.ofTensor output) := by
                                      simpa [leading, payloadShape, input, output] using
                                        (evalConv_eq_generalSpec (payload := payload) (id := n.id)
                                          (config := config) (params := params) (parent := parent)
                                          (leading := leading) (hInfer := hExpected)
                                          (hParams := hParams) (hMatches := hMatches)
                                          (hLeading := rfl) (hInput := hInput))
                                    have hOutputShape :
                                        (Spec.SomeTensor.ofTensor output).shape = expected := by
                                      simpa [output, ConvParams.outputShape] using hPayloadOut
                                    have hResultShape :
                                        (Spec.SomeTensor.ofTensor output).shape = n.outShape :=
                                      hOutputShape.trans hOut
                                    have hResultBne :
                                        ((Spec.SomeTensor.ofTensor output).shape != n.outShape) =
                                          false :=
                                      shape_bne_eq_false_of_eq hResultShape
                                    have hEval :
                                        NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
                                            (input := Spec.SomeTensor.mk (α := α) inShape x)
                                            (vals := vals0) (i := i) =
                                          .ok (Spec.SomeTensor.mk (α := α) n.outShape
                                            (nodeData.eval ctx)) := by
                                      simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode,
                                        NN.IR.Graph.normalizeNodeOutput, hN, hk, hp, hGet, hConv,
                                        nodeData, parent, getIRValue, packedResultOrPanic]
                                      split
                                      · simp_all
                                      · simp_all
                                        congr 2
                                    apply buildFrom_denoteAllFrom_nodeData_exact (α := α) (g := g)
                                      (payload := payload) (gd := gd) (i := i) (st' := st')
                                      (x := x) (hi := hi) (nodeData := nodeData)
                                    · exact ih st1 hRec
                                    · exact hEval
                                  · exact False.elim <| throw_bind_ne_ok <| by
                                      simpa [hp, hParent, parentShape, hIdx, hExpected, hParams,
                                        hMatches, leading, hPayloadOut, hOut]
                                        using hBuild
                                · exact False.elim <| throw_bind_ne_ok <| by
                                    simpa [hp, hParent, parentShape, hIdx, hExpected, hParams,
                                      hMatches, leading, hPayloadOut] using hBuild
                          · exact False.elim <| throw_bind_ne_ok <| by
                              simpa [hp, hParent, parentShape, hIdx, hExpected, hParams,
                                hMatches] using hBuild

end IRExec
end Autograd
end Runtime
