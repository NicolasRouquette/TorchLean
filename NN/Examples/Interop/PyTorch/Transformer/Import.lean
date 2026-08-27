/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Import.Core
public import NN.Spec.Models.Transformer

/-!
# Transformer PyTorch Reference Import

Transformer reference weight import from JSON.

In the spec layer, our transformer encoder parameters are explicit tensors (query/key/value/output
projections, feed-forward weights, and LayerNorm affine parameters). In PyTorch these are usually
spread across multiple `nn.Linear` and `nn.LayerNorm` submodules.

For round-trip examples we accept a *stable, explicit key format* in JSON:
`Wq`, `Wk`, `Wv`, `Wo`, `W1`, `W2`, `b1`, `b2`, `norm1_gamma`, `norm1_beta`, `norm2_gamma`,
  `norm2_beta`.

We also accept the nested PyTorch module keys emitted by
`Export.TransformerPyTorch.generateTransformerEncoderWithWeights`, such as
`layers.0.mha.q_proj.weight`. That keeps generated export state dicts loadable by both PyTorch and
this Lean importer.
-/

@[expose] public section


namespace Import
namespace TransformerPyTorch
open PyTorch

open Spec
open Tensor
open Shape
open Lean
open Data
open Json

-- Transformer Encoder state dict structure (simplified, for one layer)
/-- Typed view of a single-layer Transformer encoder `state_dict` (Float tensors).

This is the normalized typed view returned by the JSON loader. The loader accepts both TorchLean's
explicit keys and the nested PyTorch module keys emitted by the exporter.
-/
structure TransformerEncoderStateDict (embedDim headCount hiddenDim : Nat) where
  /-- Query projection matrix. -/
  queryWeight : Tensor Float [embedDim, embedDim]
  /-- Key projection matrix. -/
  keyWeight : Tensor Float [embedDim, embedDim]
  /-- Value projection matrix. -/
  valueWeight : Tensor Float [embedDim, embedDim]
  /-- Output projection matrix. -/
  outputWeight : Tensor Float [embedDim, embedDim]
  /-- Input feed-forward projection matrix. -/
  feedForwardInputWeight : Tensor Float [embedDim, hiddenDim]
  /-- Output feed-forward projection matrix. -/
  feedForwardOutputWeight : Tensor Float [hiddenDim, embedDim]
  /-- Input feed-forward projection bias. -/
  feedForwardInputBias : Tensor Float [hiddenDim]
  /-- Output feed-forward projection bias. -/
  feedForwardOutputBias : Tensor Float [embedDim]
  /-- First LayerNorm scale. -/
  norm1Scale : Tensor Float [embedDim]
  /-- First LayerNorm bias. -/
  norm1Bias : Tensor Float [embedDim]
  /-- Second LayerNorm scale. -/
  norm2Scale : Tensor Float [embedDim]
  /-- Second LayerNorm bias. -/
  norm2Bias : Tensor Float [embedDim]

def getTensorAny? (o : StateDict) (s : Shape) (keys : List String) :
    Option (Tensor Float s) :=
  match keys with
  | [] => none
  | k :: ks =>
      match getTensor? o k s with
      | some t => some t
      | none => getTensorAny? o s ks

/-- Load Transformer Encoder state dict from JSON matching either supported export key format. -/
def loadTransformerEncoderStateDict (embedDim headCount hiddenDim : Nat) (j : Json) : Option
  (TransformerEncoderStateDict embedDim headCount hiddenDim) :=
  let _ := headCount
  let projectionShape : Shape := [embedDim, embedDim]
  let feedForwardInputWeightShape : Shape := [embedDim, hiddenDim]
  let feedForwardOutputWeightShape : Shape := [hiddenDim, embedDim]
  let feedForwardInputBiasShape : Shape := [hiddenDim]
  let feedForwardOutputBiasShape : Shape := [embedDim]
  let normShape : Shape := [embedDim]
  do
    -- Accepts both `{...}` and `{ "params": {...} }`.
    let o ← loadWeights? j
    let queryWeight ←
      getTensorAny? o projectionShape ["Wq", "layers.0.mha.q_proj.weight"]
    let keyWeight ← getTensorAny? o projectionShape ["Wk", "layers.0.mha.k_proj.weight"]
    let valueWeight ←
      getTensorAny? o projectionShape ["Wv", "layers.0.mha.v_proj.weight"]
    let outputWeight ←
      getTensorAny? o projectionShape ["Wo", "layers.0.mha.out_proj.weight"]
    let feedForwardInputWeight ← getTensorAny? o feedForwardInputWeightShape
      ["W1", "layers.0.ffn.fc1.weight"]
    let feedForwardOutputWeight ← getTensorAny? o feedForwardOutputWeightShape
      ["W2", "layers.0.ffn.fc2.weight"]
    let feedForwardInputBias ← getTensorAny? o feedForwardInputBiasShape
      ["b1", "layers.0.ffn.fc1.bias"]
    let feedForwardOutputBias ← getTensorAny? o feedForwardOutputBiasShape
      ["b2", "layers.0.ffn.fc2.bias"]
    let norm1Scale ← getTensorAny? o normShape ["norm1_gamma", "layers.0.norm1.weight"]
    let norm1Bias ← getTensorAny? o normShape ["norm1_beta", "layers.0.norm1.bias"]
    let norm2Scale ← getTensorAny? o normShape ["norm2_gamma", "layers.0.norm2.weight"]
    let norm2Bias ← getTensorAny? o normShape ["norm2_beta", "layers.0.norm2.bias"]
    pure {
      queryWeight, keyWeight, valueWeight, outputWeight
      feedForwardInputWeight, feedForwardOutputWeight
      feedForwardInputBias, feedForwardOutputBias
      norm1Scale, norm1Bias, norm2Scale, norm2Bias
    }

end TransformerPyTorch
end Import
