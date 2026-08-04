/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.SelfSupervised.BlockMask
public import NN.Tensor.API

/-!
# Block-Mask API Tests

These checks exercise the runtime coordinate plumbing behind the general block-mask theorem. They
cover a one-dimensional signal and a three-dimensional volume so a later implementation change
cannot accidentally restore image-specific rank assumptions.
-/

@[expose] public section

namespace NN.Tests.API.SelfSupervised.BlockMask

open TorchLean TorchLean.ssl NN.Tensor

def expect (tag : String) (ok : Bool) : IO Unit := do
  unless ok do
    throw <| IO.userError s!"block-mask check failed: {tag}"

def signal : Spec.Tensor Float (Spec.Shape.ofList [8]) :=
  tensor! (ty := Float) [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

def volume : Spec.Tensor Float (Spec.Shape.ofList [2, 2, 2]) :=
  tensor! (ty := Float) [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]

def run : IO Unit := do
  let signalMasked := blockMask #v[8] #v[some 2] 2 0 signal
  expect "signal hidden block"
    (scalarAt [8] signalMasked [0] == some 0.0)
  expect "signal visible block"
    (scalarAt [8] signalMasked [2] == some 3.0)

  let volumeMasked := blockMask #v[2, 2, 2] #v[some 1, some 1, some 1] 2 0 volume
  expect "volume hidden block"
    (scalarAt [2, 2, 2] volumeMasked [0, 0, 0] == some 0.0)
  expect "volume visible block"
    (scalarAt [2, 2, 2] volumeMasked [0, 0, 1] == some 2.0)

end NN.Tests.API.SelfSupervised.BlockMask
