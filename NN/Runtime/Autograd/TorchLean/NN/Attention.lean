/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.NN.ConvPool

/-!
# TorchLean NN: Attention and Sequence Syntax
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor
open Proofs.Autograd.Algebra

namespace NN

/--
Multi-head self-attention layer for a sequence `(n × dModel) → (n × dModel)`.

This layer packs the four projection matrices `(Wq, Wk, Wv, Wo)` and calls the TorchLean attention
primitive. An optional boolean mask of shape `(n × n)` can be provided (e.g. causal masking).

PyTorch analogy: `torch.nn.MultiheadAttention(embed_dim=dModel, num_heads=numHeads)` in
  self-attention
mode (shape conventions differ; TorchLean uses explicit `n × dModel` tensors).
-/
def multiHeadAttention
    (batch n dModel numHeads headDim : Nat)
    {h1 : n ≠ 0}
    (seedW : Nat := 0)
    (weightInit? : Option Torch.Init.Scheme := none)
    (outputWeightInit? : Option Torch.Init.Scheme := none)
    (mask : Option (Tensor Bool (.dim n (.dim n .scalar))) := none) :
    LayerDef (.dim batch (.dim n (.dim dModel .scalar)))
      (.dim batch (.dim n (.dim dModel .scalar))) :=
  let projDim := numHeads * headDim
  let wProjShape : Shape := .dim dModel (.dim projDim .scalar)
  let wOShape : Shape := .dim projDim (.dim dModel .scalar)
  let projInit := weightInit?.getD (.xavierUniform dModel projDim)
  let outInit := outputWeightInit?.orElse (fun _ => weightInit?)
    |>.getD (.xavierUniform projDim dModel)
  let wq0 : Tensor Float wProjShape := Torch.Init.tensor projInit (seed := seedW)
  let wk0 : Tensor Float wProjShape := Torch.Init.tensor projInit (seed := seedW + 1)
  let wv0 : Tensor Float wProjShape := Torch.Init.tensor projInit (seed := seedW + 2)
  let wo0 : Tensor Float wOShape := Torch.Init.tensor outInit (seed := seedW + 3)
  { kind := s!"MultiHeadAttention(heads={numHeads}, headDim={headDim})"
    paramShapes := [wProjShape, wProjShape, wProjShape, wOShape]
    initParams := Torch.tlistQuad wq0 wk0 wv0 wo0
    runtimeInit := some <|
      .cons (Module.RuntimeInit.FloatInit.ofScheme projInit (seedW + 0)) <|
      .cons (Module.RuntimeInit.FloatInit.ofScheme projInit (seedW + 1)) <|
      .cons (Module.RuntimeInit.FloatInit.ofScheme projInit (seedW + 2)) <|
      .cons (Module.RuntimeInit.FloatInit.ofScheme outInit (seedW + 3)) .nil
    paramRequiresGrad := [true, true, true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun wq wk wv wo x =>
          TorchLean.multiHeadAttention (m := m) (α := α)
            (batch := batch) (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
            (h1 := h1)
            wq wk wv wo x (mask := mask)
  }

/--
Multi-head self-attention with a trainable bias on the final output projection.

The Q/K/V projections remain bias-free.  This is the parameterization used in Karpathy's
educational GPT implementation: the three per-head projections are linear maps without bias,
while the projection applied after concatenating the heads is affine.
-/
def multiHeadAttentionOutputBias
    (batch n dModel numHeads headDim : Nat)
    {h1 : n ≠ 0}
    (seedW : Nat := 0)
    (weightInit? : Option Torch.Init.Scheme := none)
    (outputWeightInit? : Option Torch.Init.Scheme := none)
    (mask : Option (Tensor Bool (.dim n (.dim n .scalar))) := none) :
    LayerDef (.dim batch (.dim n (.dim dModel .scalar)))
      (.dim batch (.dim n (.dim dModel .scalar))) :=
  let projDim := numHeads * headDim
  let wProjShape : Shape := .dim dModel (.dim projDim .scalar)
  let wOShape : Shape := .dim projDim (.dim dModel .scalar)
  let bOShape : Shape := .dim dModel .scalar
  let projInit := weightInit?.getD (.xavierUniform dModel projDim)
  let outInit := outputWeightInit?.orElse (fun _ => weightInit?)
    |>.getD (.xavierUniform projDim dModel)
  let wq0 : Tensor Float wProjShape := Torch.Init.tensor projInit (seed := seedW)
  let wk0 : Tensor Float wProjShape := Torch.Init.tensor projInit (seed := seedW + 1)
  let wv0 : Tensor Float wProjShape := Torch.Init.tensor projInit (seed := seedW + 2)
  let wo0 : Tensor Float wOShape := Torch.Init.tensor outInit (seed := seedW + 3)
  let bo0 : Tensor Float bOShape := Spec.zeros Float bOShape
  { kind := s!"MultiHeadAttention(heads={numHeads}, headDim={headDim}, outputBias=true)"
    paramShapes := [wProjShape, wProjShape, wProjShape, wOShape, bOShape]
    initParams := .cons wq0 (.cons wk0 (.cons wv0 (.cons wo0 (.cons bo0 .nil))))
    runtimeInit := some <|
      .cons (Module.RuntimeInit.FloatInit.ofScheme projInit (seedW + 0)) <|
      .cons (Module.RuntimeInit.FloatInit.ofScheme projInit (seedW + 1)) <|
      .cons (Module.RuntimeInit.FloatInit.ofScheme projInit (seedW + 2)) <|
      .cons (Module.RuntimeInit.FloatInit.ofScheme outInit (seedW + 3)) <|
      .cons .zeros .nil
    paramRequiresGrad := [true, true, true, true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun wq wk wv wo bo x =>
          TorchLean.multiHeadAttentionOutputBias (m := m) (α := α)
            (batch := batch) (n := n) (numHeads := numHeads) (dModel := dModel)
            (headDim := headDim) h1 wq wk wv wo bo x (mask := mask) }

/-- Lift a single layer into a 1-layer sequential model. -/
def singleLayer {σ τ : Shape} (l : LayerDef σ τ) : Seq σ τ :=
  .cons l (.id τ)

/-!
## Sequential model literal (`tlseq[...]`)

When writing compact models, chaining `singleLayer` with `>>>` is a bit verbose:

```lean
TorchLean.NN.singleLayer (TorchLean.NN.linear inDim hidDim) >>>
TorchLean.NN.singleLayer TorchLean.NN.tanh >>>
TorchLean.NN.singleLayer (TorchLean.NN.linear hidDim outDim)
```

This macro provides a compact, explicit alternative:

```lean
tlseq[
  TorchLean.NN.linear inDim hidDim,
  TorchLean.NN.tanh,
  TorchLean.NN.linear hidDim outDim
]
```

It expands to `singleLayer ... >>> singleLayer ... >>> ...`.
The syntax is namespaced to avoid colliding with other libraries.
-/

syntax (name := torchLeanSeqLit) "tlseq" "[" term,+ "]" : term

macro_rules
  | `(tlseq[$l]) => `(TorchLean.NN.singleLayer $l)
  | `(tlseq[$l, $ls,*]) => `(TorchLean.NN.singleLayer $l >>> tlseq[$ls,*])

end NN

end TorchLean
end Autograd
end Runtime
