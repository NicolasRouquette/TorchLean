/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Conv
public import NN.Spec.Module.Core

/-!
# Convolution module wrappers

This file exposes conv specs as `Spec.Module`s.

The wrappers for `Conv2d` and `ConvTranspose2d` are consolidated here with their public names.
-/

@[expose] public section

namespace Spec.Module

open Tensor

/-!
## Conv2d
-/

/-- Wrap `conv2d_spec` as an `Spec.Module`, with the output shape computed in the type. -/
def conv2d {α : Type} [Context α] {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (m : Conv2dSpec inC outC kH kW stride padding α h1 h2 h3) :
  Spec.Module α
    (.dim inC (.dim inH (.dim inW .scalar)))
    (.dim outC
      (.dim (Shape.slidingWindowOutDim inH kH stride padding)
        (.dim (Shape.slidingWindowOutDim inW kW stride padding) .scalar))) :=
{ forward := fun x => conv2dSpec m x, kind := "Conv2d", pythonExpr := s!"nn.Conv2d({inC}, {outC}, kernel_size=({kH}, {kW}), stride={stride}, padding={padding})" }

/-!
## ConvTranspose2d
-/

/-- ConvTranspose2d wrapper as an `Spec.Module` (output shape encoded at the type level). -/
def convTranspose2d {α : Type} [Context α]
  {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC > 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (m : ConvTranspose2dSpec inC outC kH kW stride padding α h1 h2 h3) :
  Spec.Module α
    (.dim inC (.dim inH (.dim inW .scalar)))
    (.dim outC
      (.dim (convTransposeOutDim inH kH stride padding)
        (.dim (convTransposeOutDim inW kW stride padding) .scalar))) :=
{ forward := fun x => convTranspose2dSpec (inC := inC) (outC := outC)
    (kH := kH) (kW := kW) (stride := stride) (padding := padding)
    (inH := inH) (inW := inW) m x
  kind := "ConvTranspose2d"
  pythonExpr := s!"nn.ConvTranspose2d({inC}, {outC}, kernel_size=({kH}, {kW}), " ++
        s!"stride={stride}, padding={padding})" }

end Spec.Module
