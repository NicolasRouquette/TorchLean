/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.LiRPA.ExampleInputs

/-!
# LiRPA MLP certificate checker

This module is a compact end-to-end example of *checking* an IBP certificate for a
compact MLP in TorchLean's graph-based verifier.

It does three things:
1. Defines a compact graph (`buildGraph`),
2. Seeds deterministic parameters and an input box,
3. Checks a JSON certificate produced by an external tool (LiRPA-style workflow).

Run via the unified CLI registry:

- `lake exe verify -- lirpa-mlp [path]`

If `path` is omitted, the CLI uses the default example certificate at:
`NN/Examples/Verification/LiRPA/mlp_cert.json`.
-/

@[expose] public section


namespace NN.Verification.LiRPA.Mlp

open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN
open _root_.Spec
open _root_.Spec.Tensor

/-- Four-node `3 → 4 → 2` ReLU MLP graph used by the compact LiRPA certificate example. -/
def buildGraph : Graph :=
  let inputNode : Node := { id := 0, parents := [], kind := .input, outShape := .dim 3 .scalar }
  let hiddenLinearNode : Node := { id := 1, parents := [0], kind := .linear, outShape := .dim 4 .scalar }
  let reluNode : Node := { id := 2, parents := [1], kind := .relu, outShape := .dim 4 .scalar }
  let outputLinearNode : Node := { id := 3, parents := [2], kind := .linear, outShape := .dim 2 .scalar }
  { nodes := #[inputNode, hiddenLinearNode, reluNode, outputLinearNode] }

/-- Deterministic Float weights and biases for both linear nodes in `buildGraph`. -/
def seedParamsFloat : ParamStore Float :=
  let hiddenWeight : Tensor Float (.dim 4 (.dim 3 .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (Float.ofNat (1 + (i.val + j.val)))))
  let hiddenBias : Tensor Float (.dim 4 .scalar) := Tensor.dim (fun i => Tensor.scalar (Float.ofNat (i.val +
    1)))
  let outputWeight : Tensor Float (.dim 2 (.dim 4 .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (Float.ofNat (2 + (i.val + j.val)))))
  let outputBias : Tensor Float (.dim 2 .scalar) := Tensor.dim (fun i => Tensor.scalar (Float.ofNat
    (i.val)))
  let emptyStore : ParamStore Float := {}
  let withHiddenLayer :=
    { emptyStore with
      linearWB := emptyStore.linearWB.insert 1
        ({ m := 4, n := 3, w := hiddenWeight, b := hiddenBias }) }
  let withOutputLayer :=
    { withHiddenLayer with
      linearWB := withHiddenLayer.linearWB.insert 3
        ({ m := 2, n := 4, w := outputWeight, b := outputBias }) }
  withOutputLayer

/-- Add the example's natural input box of radius `eps` to a parameter store. -/
def seedInputFloat (ps : ParamStore Float) (eps : Float) : ParamStore Float :=
  NN.Verification.LiRPA.ExampleInputs.seedNaturalInputBox 0 3 eps ps

/-- Check an IBP certificate JSON file and throw an error if it does not match recomputed bounds. -/
def verifyCert (path : String) : IO Unit := do
  let g := buildGraph
  let ps := seedInputFloat (seedParamsFloat) (eps := (1.0))
  NN.Verification.IBPCert.checkOrThrow g ps (outId := 3) path

end NN.Verification.LiRPA.Mlp
