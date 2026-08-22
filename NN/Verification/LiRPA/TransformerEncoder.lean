/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.LiRPA.ExampleInputs

/-!
# LiRPA transformer encoder certificate checker

LiRPA/IBP certificate checker: transformer-encoder-like graph.

This transformer encoder block includes:
- attention-like `softmax` flow,
- residual additions,
- `layernorm`, and
- a 2-layer feed-forward network with `relu`.

It exists primarily to exercise certificate checking across a wider set of nonlinear ops than the
MLP/CNN workflows.

References:
- IBP: arXiv:1810.12715 `https://arxiv.org/abs/1810.12715`
- CROWN background: `https://arxiv.org/abs/1811.00866`
- auto_LiRPA (reference implementation / exporter inspiration):
  `https://github.com/Verified-Intelligence/auto_LiRPA`

Export (Python):
`python3.12 scripts/verification/lirpa/export_crown_cert.py`

Run (Lean):
`lake exe verify -- lirpa-encoder [NN/Examples/Verification/LiRPA/transformer_encoder_cert.json]`
-/

@[expose] public section


namespace NN.Verification.LiRPA.TransformerEncoder

open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN
open _root_.Spec
open _root_.Spec.Tensor

/-- Small fixed graph with residual + layernorm + FFN (see module doc). -/
def buildGraph : Graph :=
  let nModel := 4
  let scoresDim := 5
  let nHidden := 6
  let inputNode : Node := { id := 0, parents := [], kind := .input, outShape := .dim nModel .scalar }
  let scoreNode : Node := { id := 1, parents := [0], kind := .linear, outShape := .dim scoresDim .scalar }
  let softmaxNode : Node :=
    { id := 2, parents := [1], kind := .softmax (axis := 0), outShape := .dim scoresDim .scalar }
  let attentionValueNode : Node := { id := 3, parents := [2], kind := .matmul, outShape := .dim nModel .scalar }
  let attentionResidualNode : Node :=
    { id := 4, parents := [0, 3], kind := .add, outShape := .dim nModel .scalar }
  let firstLayerNormNode : Node :=
    { id := 5, parents := [4], kind := .layernorm (axis := 0), outShape := .dim nModel .scalar }
  let feedForwardHiddenNode : Node :=
    { id := 6, parents := [5], kind := .linear, outShape := .dim nHidden .scalar }
  let feedForwardReluNode : Node :=
    { id := 7, parents := [6], kind := .relu, outShape := .dim nHidden .scalar }
  let feedForwardOutputNode : Node :=
    { id := 8, parents := [7], kind := .linear, outShape := .dim nModel .scalar }
  let feedForwardResidualNode : Node :=
    { id := 9, parents := [5, 8], kind := .add, outShape := .dim nModel .scalar }
  let finalLayerNormNode : Node :=
    { id := 10, parents := [9], kind := .layernorm (axis := 0), outShape := .dim nModel .scalar }
  { nodes :=
      #[ inputNode
       , scoreNode
       , softmaxNode
       , attentionValueNode
       , attentionResidualNode
       , firstLayerNormNode
       , feedForwardHiddenNode
       , feedForwardReluNode
       , feedForwardOutputNode
       , feedForwardResidualNode
       , finalLayerNormNode ] }

/-- Seed deterministic parameters for the `.linear` / `.matmul` nodes in `buildGraph`. -/
def seedParamsFloat : ParamStore Float :=
  let nModel := 4; let scoresDim := 5; let nHidden := 6
  let scoreWeight : Tensor Float (.dim scoresDim (.dim nModel .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (Float.ofNat (1 + (i.val + 2*j.val)))))
  let scoreBias : Tensor Float (.dim scoresDim .scalar) := Tensor.dim (fun i => Tensor.scalar (0.1 *
    Float.ofNat i.val))
  let valueWeight : Tensor Float (.dim nModel (.dim scoresDim .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (Float.ofNat (2 + (i.val + j.val)))))
  let feedForwardHiddenWeight : Tensor Float (.dim nHidden (.dim nModel .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (Float.ofNat (1 + ((i.val + j.val) %
      3)))))
  let feedForwardHiddenBias : Tensor Float (.dim nHidden .scalar) := Tensor.dim (fun i => Tensor.scalar (0.05 *
    Float.ofNat i.val))
  let feedForwardOutputWeight : Tensor Float (.dim nModel (.dim nHidden .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (Float.ofNat (2 + ((i.val + j.val) %
      4)))))
  let feedForwardOutputBias : Tensor Float (.dim nModel .scalar) := Tensor.dim (fun i => Tensor.scalar (0.02 *
    Float.ofNat i.val))
  let emptyStore : ParamStore Float := {}
  let withScoreLinear :=
    { emptyStore with
      linearWB :=
        emptyStore.linearWB.insert 1
          { m := scoresDim
            n := nModel
            w := scoreWeight
            b := scoreBias } }
  let withValueProjection :=
    { withScoreLinear with
      matmulW :=
        withScoreLinear.matmulW.insert 3
          ({ m := nModel, n := scoresDim, w := valueWeight }) }
  let withFeedForwardHidden :=
    { withValueProjection with
      linearWB :=
        withValueProjection.linearWB.insert 6
          ({ m := nHidden, n := nModel, w := feedForwardHiddenWeight, b := feedForwardHiddenBias }) }
  let withFeedForwardOutput :=
    { withFeedForwardHidden with
      linearWB :=
        withFeedForwardHidden.linearWB.insert 8
          ({ m := nModel, n := nHidden, w := feedForwardOutputWeight, b := feedForwardOutputBias }) }
  withFeedForwardOutput

/-- Insert an $L^\infty$ input box of radius `eps` around a fixed center point. -/
def seedInputFloat (ps : ParamStore Float) (eps : Float) : ParamStore Float :=
  NN.Verification.LiRPA.ExampleInputs.seedNaturalInputBox 0 4 eps ps

/--
Check an IBP certificate JSON against this transformer-encoder graph.

This is wired into `lake exe verify -- lirpa-encoder [path]`.
-/
def verifyCert (path : String) : IO Unit := do
  let g := buildGraph
  let ps := seedInputFloat (seedParamsFloat) (eps := (0.5))
  NN.Verification.IBPCert.checkOrThrow g ps (outId := 10) path

end NN.Verification.LiRPA.TransformerEncoder
