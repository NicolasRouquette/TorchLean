/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Conv
public import NN.Spec.Module.Core

/-!
# Convolution Modules

Convolution modules are parameterized by vectors of spatial extents. The same definitions cover
one-dimensional sequences, images, volumes, and higher-rank spatial data.
-/

@[expose] public section

namespace Spec.Module

open Tensor

/-- Wrap an arbitrary-rank channels-first convolution as a `Spec.Module`. -/
def conv {α : Type} [Context α]
    {d inC outC : Nat} {kernel stride padding inSpatial : Spec.Tensor Nat [d]}
    (m : ConvSpec d inC outC kernel stride padding α) :
    Spec.Module α
      (Shape.ofList (inC :: inSpatial.toList))
      (Shape.ofList (outC :: (convOutSpatial inSpatial kernel stride padding).toList)) :=
  { forward := convSpec m
    kind := "Conv"
    pythonExpr := "nn.Conv(...)" }

/-- Wrap an arbitrary-rank channels-first transposed convolution as a `Spec.Module`. -/
def convTranspose {α : Type} [Context α]
    {d inC outC : Nat} {kernel stride padding inSpatial : Spec.Tensor Nat [d]}
    (m : ConvTransposeSpec d inC outC kernel stride padding α) :
    Spec.Module α
      (Shape.ofList (inC :: inSpatial.toList))
      (Shape.ofList (outC :: (convTransposeOutSpatial inSpatial kernel stride padding).toList)) :=
  { forward := convTransposeSpec m
    kind := "ConvTranspose"
    pythonExpr := "nn.ConvTranspose(...)" }

end Spec.Module
