/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Pooling
public import NN.Spec.Module.Core

/-!
# Pooling Modules

The wrappers in this file preserve a leading channel dimension and pool over an arbitrary vector
of spatial dimensions. Padding, stride, and window extents are independent on every axis.
-/

@[expose] public section


namespace Spec.Module
open Tensor

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-- Wrap arbitrary-rank channels-first max pooling as a `Spec.Module`. -/
def maxPool {d C : Nat} {inSpatial kernel stride padding : Spec.Tensor Nat [d]}
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    {hStride : ∀ i : Fin d, stride.getScalar i ≠ 0}
    (m : MaxPoolSpec d kernel stride padding hKernel hStride) :
    Spec.Module α
      (Shape.ofList (C :: inSpatial.toList))
      (Shape.ofList (C :: (poolOutSpatialPad inSpatial kernel stride padding).toList)) :=
  { forward := maxPoolSpec m
    kind := "MaxPool"
    pythonExpr := "nn.MaxPool(...)" }

/-- Wrap arbitrary-rank channels-first average pooling as a `Spec.Module`. -/
def avgPool {d C : Nat} {inSpatial kernel stride padding : Spec.Tensor Nat [d]}
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    {hStride : ∀ i : Fin d, stride.getScalar i ≠ 0}
    (m : AvgPoolSpec d kernel stride padding hKernel hStride) :
    Spec.Module α
      (Shape.ofList (C :: inSpatial.toList))
      (Shape.ofList (C :: (poolOutSpatialPad inSpatial kernel stride padding).toList)) :=
  { forward := avgPoolSpec m
    kind := "AvgPool"
    pythonExpr := "nn.AvgPool(...)" }

end Spec.Module
