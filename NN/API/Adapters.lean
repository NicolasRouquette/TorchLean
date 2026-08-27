/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor

/-!
# Low-Rank Adapters

LoRA represents a linear-weight update as two smaller matrices. For a base weight
$W : \mathbb{R}^{d_{in}\times d_{out}}$, an adapter of rank $r$ uses
$A : \mathbb{R}^{d_{in}\times r}$ and $B : \mathbb{R}^{r\times d_{out}}$:

$$W_{eff}=W+sAB.$$

The matrix orientation agrees with TorchLean's row-batch linear layers. This module defines the
typed update and its action on a batch; the training code decides which parameters to optimize.

Reference: Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models” (2021),
https://arxiv.org/abs/2106.09685.
-/

@[expose] public section

namespace TorchLean.Adapters.LoRA

open _root_.Spec
open _root_.Spec.Tensor

/-- LoRA factors for a linear weight of shape `inDim × outDim`. -/
structure Params (α : Type) (inDim rank outDim : Nat) where
  /-- Projection from the input dimension to the adapter rank. -/
  A : Tensor α [inDim, rank]
  /-- Projection from the adapter rank to the output dimension. -/
  B : Tensor α [rank, outDim]

/-- The scaled low-rank update $sAB$. -/
def delta {α : Type} [Add α] [Mul α] [Zero α]
    {inDim rank outDim : Nat} (p : Params α inDim rank outDim) (scale : α) :
    Tensor α [inDim, outDim] :=
  scaleSpec (matMulSpec p.A p.B) scale

/-- Add a LoRA update to a base linear weight. -/
def effectiveWeight {α : Type} [Add α] [Mul α] [Sub α] [Zero α]
    {inDim rank outDim : Nat}
    (base : Tensor α [inDim, outDim])
    (p : Params α inDim rank outDim) (scale : α) :
    Tensor α [inDim, outDim] :=
  addSpec base (delta p scale)

/-- Apply a linear map whose weight is augmented by a LoRA update at every leading index. -/
def linear {α : Type} [Add α] [Mul α] [Sub α] [Zero α]
    {leading : List Nat} {inDim rank outDim : Nat}
    (x : Tensor α (leading ++ [inDim]))
    (base : Tensor α [inDim, outDim])
    (p : Params α inDim rank outDim) (scale : α) :
    Tensor α (leading ++ [outDim]) := by
  let x' : Tensor α ((Shape.ofList leading).concat (Shape.ofList [inDim])) := by
    simpa only [Shape.ofList_append] using x
  simpa only [Shape.ofList_append] using
    Spec.Tensor.mapEach (Shape.ofList leading)
      (fun row => vecMatMulSpec row (effectiveWeight base p scale)) x'

end TorchLean.Adapters.LoRA
