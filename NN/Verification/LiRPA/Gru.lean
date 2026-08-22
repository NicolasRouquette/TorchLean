/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.LiRPA.ExampleInputs

/-!
# LiRPA GRU gate certificate checker

LiRPA/IBP certificate checker: GRU-style gate fragment.

This module builds a small nonlinear graph with `sigmoid`, `tanh`, and an elementwise multiply:
`x -> linear -> sigmoid`
`x -> linear -> tanh`
`mul_elem(sigmoid(x), tanh(x))`

It is a focused fragment that exercises common RNN nonlinearities in the
LiRPA certificate checker.

References:
- IBP: arXiv:1810.12715 `https://arxiv.org/abs/1810.12715`
- auto_LiRPA (reference implementation / exporter inspiration):
  `https://github.com/Verified-Intelligence/auto_LiRPA`

Export (Python):
`python3.12 scripts/verification/lirpa/export_gru_cert.py`

Run (Lean):
`lake exe verify -- lirpa-gru [NN/Examples/Verification/LiRPA/gru_gate_cert.json]`
-/

@[expose] public section


namespace NN.Verification.LiRPA.Gru

open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Graph
open _root_.Spec
open _root_.Spec.Tensor

/-- Small nonlinear graph exercising `sigmoid`, `tanh`, and `mul_elem`. -/
def buildGraph : Graph :=
  let n := 3
  let inputNode : Node := { id := 0, parents := [], kind := .input, outShape := .dim n .scalar }
  let gateLinearNode : Node := { id := 1, parents := [0], kind := .linear, outShape := .dim n .scalar }
  let sigmoidGateNode : Node := { id := 2, parents := [1], kind := .sigmoid, outShape := .dim n .scalar }
  let candidateLinearNode : Node := { id := 3, parents := [0], kind := .linear, outShape := .dim n .scalar }
  let candidateTanhNode : Node := { id := 4, parents := [3], kind := .tanh, outShape := .dim n .scalar }
  let gatedCandidateNode : Node :=
    { id := 5, parents := [2, 4], kind := .mul_elem, outShape := .dim n .scalar }
  { nodes := #[inputNode, gateLinearNode, sigmoidGateNode, candidateLinearNode,
      candidateTanhNode, gatedCandidateNode] }

/-- Seed deterministic linear weights for the two `.linear` nodes in `buildGraph`. -/
def seedParamsFloat : ParamStore Float :=
  let n := 3
  let weight : Tensor Float (.dim n (.dim n .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (Float.ofNat (1 + (i.val + j.val)))))
  let bias : Tensor Float (.dim n .scalar) := Tensor.dim (fun i => Tensor.scalar (Float.ofNat (i.val)))
  let emptyStore : ParamStore Float := {}
  let withGateLinear :=
    { emptyStore with
      linearWB := emptyStore.linearWB.insert 1 ({ m := n, n := n, w := weight, b := bias }) }
  let withCandidateLinear :=
    { withGateLinear with
      linearWB := withGateLinear.linearWB.insert 3 ({ m := n, n := n, w := weight, b := bias }) }
  withCandidateLinear

/-- Insert an $L^\infty$ input box of radius `eps` around a fixed center point. -/
def seedInputFloat (ps : ParamStore Float) (eps : Float) : ParamStore Float :=
  NN.Verification.LiRPA.ExampleInputs.seedNaturalInputBox 0 3 eps ps

/--
Check an IBP certificate JSON against this GRU-fragment graph.

This is wired into `lake exe verify -- lirpa-gru [path]`.
-/
def verifyCert (path : String) : IO Unit := do
  let g := buildGraph
  let ps := seedInputFloat seedParamsFloat (eps := 0.5)
  NN.Verification.IBPCert.checkOrThrow g ps (outId := 5) path

end NN.Verification.LiRPA.Gru
