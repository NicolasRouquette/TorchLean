/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor
public import NN.Runtime.Autograd.Torch.Initialization

/-!
# Tensor Initialization

Deterministic tensor initialization helpers (Xavier/Kaiming, etc.) that return TorchLean
`Spec.Tensor`s with shape tracked in the type.
-/

@[expose] public section

namespace TorchLean
namespace Init

open Spec

/-!
## Tensor Initialization Helpers

This module exposes `Runtime.Autograd.Torch.Init` through TorchLean
`Spec.Tensor`s with shapes tracked in the type.

All initializers are deterministic given an explicit `seed : Nat`, which is convenient for:
- reproducible examples
- stable tests
- proofs that want to fix concrete initial values

### PyTorch Mapping

The names mirror common PyTorch initializers:
- Xavier/Glorot: `torch.nn.init.xavier_uniform_`, `torch.nn.init.xavier_normal_`
- Kaiming/He: `torch.nn.init.kaiming_uniform_`, `torch.nn.init.kaiming_normal_`

See the PyTorch init docs:
`https://pytorch.org/docs/stable/nn.init.html`
-/

export _root_.Runtime.Autograd.Torch.Init (Scheme)

/--
Initialize a `Float` tensor using the given scheme.

Most user code should prefer `tensor`, which casts the generated values into its chosen scalar
semantics. The dimension list is normally inferred from the expected tensor type.
-/
def tensorFloat {shape : List Nat} (sch : Scheme) (seed : Nat := 0) : Tensor Float shape :=
  _root_.Runtime.Autograd.Torch.Init.tensor
    (sch := sch) (seed := seed) (s := Shape.ofList shape)

/--
Initialize a tensor under an arbitrary scalar semantics `α`, by first generating `Float`s and
then casting elementwise via `cast : Float → α`.
-/
def tensor {α : Type} [Context α] {shape : List Nat}
    (cast : Float → α) (sch : Scheme) (seed : Nat := 0) : Tensor α shape :=
  TorchLean.Tensor.map cast (tensorFloat (sch := sch) (seed := seed))

/--
Xavier/Glorot initializer for a linear weight matrix of shape `(outDim, inDim)`.

PyTorch analogue: `torch.nn.init.xavier_uniform_` (for example).
-/
def xavierW {α : Type} [Context α] (cast : Float → α) (outDim inDim : Nat) (seed : Nat := 0) :
    Tensor α [outDim, inDim] :=
  TorchLean.Tensor.map cast (_root_.Runtime.Autograd.Torch.Init.xavierW outDim inDim seed)

/--
Kaiming/He initializer for a linear weight matrix of shape `(outDim, inDim)`.

PyTorch analogue: `torch.nn.init.kaiming_uniform_` (for example).
-/
def kaimingW {α : Type} [Context α] (cast : Float → α) (outDim inDim : Nat) (seed : Nat := 0) :
    Tensor α [outDim, inDim] :=
  TorchLean.Tensor.map cast (_root_.Runtime.Autograd.Torch.Init.kaimingW outDim inDim seed)

end Init
end TorchLean
