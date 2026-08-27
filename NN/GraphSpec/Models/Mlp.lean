/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Core

/-!
# GraphSpec MLP Example

This file contains the smallest GraphSpec architecture example:

`Linear(in,hid) → ReLU → Linear(hid,out)`.

This does not duplicate TorchLean's executable MLP helper. That constructor lives under
`NN.GraphSpec.Models.TorchLean.Mlp`; application code reaches the corresponding public constructor
through `TorchLean.nn.models`.
The point here is narrower and proof-oriented:

- show the sequential `Chain` DSL in its simplest useful form;
- make the parameter ABI visible in the type;
- provide a stable target for GraphSpec equivalence and deterministic-init proofs.

Because this is a pure sequential chain, it is authored with `Chain` and `>>>`. The companion
`mlpDAGModelZeroInit` lowers the same chain to the general DAG model representation so DAG-only
tooling can consume it.
-/

@[expose] public section

namespace NN
namespace GraphSpec
namespace Models

open _root_.Spec
open _root_.TorchLean.Tensor

/--
2-layer MLP: `Linear(in,hid) → ReLU → Linear(hid,out)`.

Notice how the parameter interface is explicit in the type:

- the first `Linear(in,hid)` contributes tensors `W₁ : Tensor α [hid, in]` and
  `b₁ : Tensor α [hid]`,
- the second `Linear(hid,out)` contributes tensors `W₂ : Tensor α [out, hid]` and
  `b₂ : Tensor α [out]`,
- and `ReLU` contributes no parameters.

So the overall parameter list is exactly:
`[[hid, in], [hid], [out, hid], [out]]`.
-/
def mlp (inDim hidDim outDim : Nat) :
    Chain
      [[hidDim, inDim], [hidDim], [outDim, hidDim], [outDim]]
      [inDim] [outDim] :=
  Chain.linear inDim hidDim >>>
  Chain.relu [hidDim] >>>
  Chain.linear hidDim outDim

/--
The same 2-layer MLP, but exposed as a DAG `Model` via the structural lowering
`LowerToDAG.Chain.toDAGModelZeroInit`.

This is mainly for GraphSpec example ergonomics: downstream tooling that expects DAG terms can
consume this even though it was authored using the sequential `>>>` syntax.

Initialization: all-zero parameters (see `LowerToDAG.Chain.toDAGModelZeroInit`).
-/
def mlpDAGModelZeroInit (inDim hidDim outDim : Nat) :
    DAG.Model
      [[hidDim, inDim], [hidDim], [outDim, hidDim], [outDim]]
      [[inDim]]
      [outDim] :=
  LowerToDAG.Chain.toDAGModelZeroInit (mlp (inDim := inDim) (hidDim := hidDim) (outDim := outDim))

/-!
## Example Usage

You can build a simple classifier head by appending a softmax:

```lean
def g (inDim hidDim outDim : Nat) :
    Chain
      [[hidDim, inDim], [hidDim], [outDim, hidDim], [outDim]]
      [inDim] [outDim] :=
  Models.mlp inDim hidDim outDim >>> Chain.softmax [outDim] 0
```

Then:

- `Interp.spec (g …)` maps a parameter pack and input tensor to an output tensor;
- `Chain.toProgram (g …)` is an executable TorchLean `Program` with arguments
  `params ++ [input]`.
-/

end Models
end GraphSpec
end NN
