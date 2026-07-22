/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.NN

/-!
# TorchLean Adapter API

Parameter-efficient adapter operations exposed by the `NN` umbrella.

Adapters change the parameterization of a model or layer.  They are not optimizers: the optimizer
still acts on whichever adapter/base parameters the caller chooses to train.  For LoRA, the public
the API here builds the low-rank delta and effective linear weight; freezing base weights and
choosing an optimizer are separate training-loop decisions.
-/

@[expose] public section

namespace TorchLean

namespace Adapters

namespace LoRA

@[inherit_doc NN.API.Adapters.LoRA.Params]
abbrev Params := NN.API.Adapters.LoRA.Params

@[inherit_doc NN.API.Adapters.LoRA.delta]
def delta {α : Type} [Add α] [Mul α] [Zero α]
    {inDim rank outDim : Nat} (p : Params α inDim rank outDim) (scale : α) :
    _root_.Spec.Tensor α (.dim inDim (.dim outDim .scalar)) :=
  NN.API.Adapters.LoRA.delta p scale

@[inherit_doc NN.API.Adapters.LoRA.effectiveWeight]
def effectiveWeight {α : Type} [Add α] [Mul α] [Sub α] [Zero α]
    {inDim rank outDim : Nat}
    (base : _root_.Spec.Tensor α (.dim inDim (.dim outDim .scalar)))
    (p : Params α inDim rank outDim) (scale : α) :
    _root_.Spec.Tensor α (.dim inDim (.dim outDim .scalar)) :=
  NN.API.Adapters.LoRA.effectiveWeight base p scale

@[inherit_doc NN.API.Adapters.LoRA.linear]
def linear {α : Type} [Add α] [Mul α] [Sub α] [Zero α]
    {batch inDim rank outDim : Nat}
    (x : _root_.Spec.Tensor α (.dim batch (.dim inDim .scalar)))
    (base : _root_.Spec.Tensor α (.dim inDim (.dim outDim .scalar)))
    (p : Params α inDim rank outDim) (scale : α) :
    _root_.Spec.Tensor α (.dim batch (.dim outDim .scalar)) :=
  NN.API.Adapters.LoRA.linear x base p scale

end LoRA

end Adapters

end TorchLean
