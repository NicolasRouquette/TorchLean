/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Lowering.Common

/-!
# Convolution and Normalization IR Lowering

Checked lowering for pooling, convolution, batch normalization, and layer normalization.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace IRExec

open Spec
open Tensor
open Proofs.Autograd.Algebra
open NN.IR

namespace Internal

/-- Checked lowering for pooling, convolution, batch normalization, and layer normalization. -/
@[simp] def lowerConvolutionNormalization {α : Type} [Context α]
    [shapeDecidable : DecidableEq Shape]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (kind : OpKind) :
    NodeLoweringResult ctx := do
  let g := ctx.graph
  let payload := ctx.payload
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : _root_.TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  let getIRValue {s : Shape} (context : _root_.TorchLean.TensorPack α Γ)
      (idx : Idx Γ s) : Tensor α s :=
    getIdx (α := α) (xs := context) idx
  match kind with
  | .maxPool config =>
      match unaryParent? n.parents with
      | some pId =>
          let pNode ← g.getNode pId
          let sIn := pNode.outShape
          let ip ← parentIdx pId sIn
          let plan ← OpContracts.planPool "max_pool" config sIn
          let expected := plan.outShape
          if _hOut : expected = τ then
            let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
              let x := getIdx (α := α) (xs := ctx) ip
              packedResultOrPanic (α := α) τ <|
                NN.IR.Graph.evalMaxPool config (Spec.SomeTensor.ofTensor x)
            pure <| fwd forward
          else
            throw s!"IRExec: node {i}: max_pool outShape mismatch ({n.summary})"
      | _ => throw s!"IRExec: node {i}: max_pool expects 1 parent ({n.summary})"
  | .avgPool config =>
      match unaryParent? n.parents with
      | some pId =>
          let pNode ← g.getNode pId
          let sIn := pNode.outShape
          let ip ← parentIdx pId sIn
          let plan ← OpContracts.planPool "avg_pool" config sIn
          let expected := plan.outShape
          if _hOut : expected = τ then
            let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
              let x := getIdx (α := α) (xs := ctx) ip
              packedResultOrPanic (α := α) τ <|
                NN.IR.Graph.evalAvgPool config (Spec.SomeTensor.ofTensor x)
            pure <| fwd forward
          else
            throw s!"IRExec: node {i}: avg_pool outShape mismatch ({n.summary})"
      | _ => throw s!"IRExec: node {i}: avg_pool expects 1 parent ({n.summary})"
  | .conv config =>
      match unaryParent? n.parents with
      | some xId =>
          let xNode ← g.getNode xId
          let expectedIn : Shape := xNode.outShape
          let ix ← parentIdx xId expectedIn
          let expected ← OpContracts.inferConvConfigOutShape "conv" config expectedIn
          match payload.conv? n.id with
          | none => throw s!"IRExec: missing convolution payload for node {n.id}"
          | some params =>
              if params.matchesConfig config then
                let dims := expectedIn.toList
                let leading : Shape := Shape.ofList (dims.take config.channelAxis)
                let payloadShape := params.inputShape leading
                let shapeDecision : Decidable (expectedIn = payloadShape) :=
                  shapeDecidable expectedIn payloadShape
                match shapeDecision with
                | .isTrue _ =>
                    if _hPayloadOut : params.outputShape leading = expected then
                      if _hOut : expected = τ then
                        let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
                          let x := getIRValue ctx ix
                          packedResultOrPanic (α := α) τ <|
                            NN.IR.Graph.evalConv payload n.id config
                              (Spec.SomeTensor.ofTensor x)
                        pure <| fwd forward
                      else
                        throw s!"IRExec: node {i}: conv outShape mismatch ({n.summary})"
                    else
                      throw <|
                        s!"IRExec: node {i}: convolution payload output shape " ++
                          s!"{repr (params.outputShape leading)} does not match inferred shape " ++
                          s!"{repr expected} ({n.summary})"
                | .isFalse _ =>
                    throw <|
                      s!"IRExec: node {i}: convolution payload shape {repr payloadShape} " ++
                        s!"does not match parent shape {repr expectedIn} ({n.summary})"
              else
                throw s!"IRExec: node {i}: convolution payload does not match the node configuration"
      | _ => throw s!"IRExec: node {i}: conv expects 1 parent ({n.summary})"
  | .batchNormEval channelAxis channels =>
      match unaryParent? n.parents with
      | some xId =>
          let xNode ← g.getNode xId
          let expectedIn : Shape := xNode.outShape
          let ix ← parentIdx xId expectedIn
          let _ ← OpContracts.inferBatchNormEvalOutShape channelAxis channels expectedIn
          match payload.batchNormEval? n.id with
          | none => throw s!"IRExec: missing batch_norm_eval payload for node {n.id}"
          | some params =>
              if _hChannels : params.c = channels then
                let dims := expectedIn.toList
                let leading : Shape := Shape.ofList (dims.take channelAxis)
                let spatial : Shape := Shape.ofList (dims.drop (channelAxis + 1))
                let payloadShape : Shape := leading.concat (.dim params.c spatial)
                let shapeDecision : Decidable (expectedIn = payloadShape) :=
                  shapeDecidable expectedIn payloadShape
                match shapeDecision with
                | .isTrue _ =>
                    if _hOut : @Eq Shape expectedIn τ then
                      let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
                        let parent : Tensor α expectedIn := getIdx (α := α) (xs := ctx) ix
                        packedResultOrPanic (α := α) τ <|
                          NN.IR.Graph.evalBatchNorm payload n.id channelAxis channels
                            (Spec.SomeTensor.ofTensor parent)
                      pure <| fwd forward
                    else
                      throw s!"IRExec: node {i}: batch_norm_eval outShape mismatch ({n.summary})"
                | .isFalse _ =>
                  throw <|
                    s!"IRExec: node {i}: batch_norm_eval payload shape {repr payloadShape} " ++
                      s!"does not match parent shape {repr expectedIn} ({n.summary})"
              else
                throw <|
                  s!"IRExec: node {i}: batch_norm_eval payload channels {params.c} do not " ++
                    s!"match node channels {channels} ({n.summary})"
      | _ => throw s!"IRExec: node {i}: batch_norm_eval expects 1 parent ({n.summary})"
  | .layernorm axis =>
      match unaryParent? n.parents with
      | some pId => do
          let (seqLen, embedDim) ←
            match OpContracts.layerNormMatrixDims axis τ with
            | .ok p => pure p
            | .error msg => throw s!"IRExec: node {i}: layernorm: {msg} ({n.summary})"
          let view2d : Shape := .dim seqLen (.dim embedDim .scalar)
          if hNumel : Spec.Shape.size τ = Spec.Shape.size view2d then
            if hSeq : seqLen > 0 then
              if hEmb : embedDim > 0 then
                let ip ← parentIdx pId τ
                let affine ←
                  NN.IR.Graph.resolveLayerNormAffine payload i axis τ embedDim
                let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
                  let x : Tensor α τ := getIdx (α := α) (xs := ctx) ip
                  let x2d : Tensor α view2d :=
                    Tensor.reshapeSpec (α := α) (s₁ := τ) (s₂ := view2d) x hNumel
                  let y2d : Tensor α view2d :=
                    Spec.layerNorm (α := α) (seqLen := seqLen) (embedDim := embedDim)
                      (x := x2d) (gamma := affine.gamma) (beta := affine.beta)
                      (h_seq_pos := hSeq) (h_embed_pos := hEmb) (epsilon := affine.epsilon)
                  Tensor.reshapeSpec (α := α) (s₁ := view2d) (s₂ := τ) y2d hNumel.symm
                pure <| fwd forward
              else
                throw s!"IRExec: node {i}: layernorm embedDim must be > 0 (got {embedDim})"
            else
              throw s!"IRExec: node {i}: layernorm seqLen must be > 0 (got {seqLen})"
          else
            throw <|
              s!"IRExec: node {i}: layernorm internal error: bad reshape sizes " ++
                s!"({Spec.Shape.size τ} vs {Spec.Shape.size view2d}) ({n.summary})"
      | _ =>
          throw s!"IRExec: node {i}: layernorm expects 1 parent ({n.summary})"
  | _ => throw s!"IRExec: internal error: operation routed to lowerConvolutionNormalization"


end Internal
end IRExec
end Autograd
end Runtime
