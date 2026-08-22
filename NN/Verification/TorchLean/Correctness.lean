/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Semantics
public import NN.Verification.TorchLean.Lowering

/-!
# Correctness

TorchLean→IR correctness helpers.

This file does **not** (yet) contain a full lowering-correctness theorem for arbitrary
`TorchLean.Program`s (the current embedding is higher-order). It provides the small, reusable
bridges needed by concrete model-correctness theorems:

- convert a verifier `ParamStore` into an IR `Payload` for `NN.IR.Graph.denote`;
- evaluate a `LoweredIR` graph on a concrete input.
-/

@[expose] public section


namespace NN.Verification.TorchLean

open NN.IR

/--
Convert a verifier `ParamStore` into an IR `Payload` for `NN.IR.Graph.denote`.

This is the bridge between the CROWN/LiRPA parameter representation used by the verification
pipeline and the executable IR semantics.
-/
def payloadOfParamStore {α : Type} [Context α] (ps : NN.MLTheory.CROWN.Graph.ParamStore α) : Payload
  α :=
  { const? := fun id =>
      (ps.constVals.get? id).map (fun c =>
        { n := c.n, v := c.v })
    linear? := fun id =>
      (ps.linearWB.get? id).map (fun p =>
        { outDim := p.m, inDim := p.n, W := p.w, b := p.b })
    conv2d? := ps.conv2dCfg.get?
    batchNorm2dNchwEval? := ps.batchNorm2dNchwEval.get? }

/-- Cast a tensor across a proved shape equality. -/
def castTensor {α : Type} [Context α] {s s' : Spec.Shape} (h : s = s')
    (t : Spec.Tensor α s) : Spec.Tensor α s' :=
  cast (congrArg (fun s : Spec.Shape => Spec.Tensor α s) h) t

/-- Evaluate a `LoweredIR` forward graph on an input tensor, returning a shape-checked tensor. -/
def runForwardIR
    {α : Type} [Context α] [DecidableEq Spec.Shape]
    {inShape outShape : Spec.Shape}
    (c : LoweredIR α) (x : Spec.Tensor α inShape) : Except String (Spec.Tensor α outShape) := do
  let input : Spec.PackedTensor α := Spec.PackedTensor.mk (α := α) inShape x
  let out ←
    Graph.denote (α := α) (g := c.graph) (payload := payloadOfParamStore (α := α) c.ps)
      (input := input) (outputId := c.outputId)
  if h : out.shape = outShape then
    pure (h ▸ out.tensor)
  else
    throw <|
      s!"TorchLeanCorrectness: output shape mismatch: " ++
        s!"produced={repr out.shape}, expected={repr outShape}"

end NN.Verification.TorchLean
