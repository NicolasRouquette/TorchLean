/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.SelfSupervised.BlockMask
public import NN.Tensor

/-!
# Block-Mask API Tests

These checks exercise the runtime coordinate plumbing behind the general block-mask theorem. They
cover a one-dimensional signal and a three-dimensional volume so a later implementation change
cannot accidentally restore image-specific rank assumptions.
-/

@[expose] public section

namespace NN.Tests.API.SelfSupervised.BlockMask

open TorchLean

def expect (tag : String) (ok : Bool) : IO Unit := do
  unless ok do
    throw <| IO.userError s!"block-mask check failed: {tag}"

def signal : Spec.Tensor Float [8] :=
  tensor! (ty := Float) [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]

def volume : Spec.Tensor Float [2, 2, 2] :=
  tensor! (ty := Float) [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]

def run : IO Unit := do
  let signalMasked := TorchLean.ssl.BlockMask.apply (tensor! [some 2]) 2 0 signal
  expect "signal hidden block"
    (TorchLean.Tensor.at? signalMasked #[0] == some 0.0)
  expect "signal visible block"
    (TorchLean.Tensor.at? signalMasked #[2] == some 3.0)

  let volumeMasked := TorchLean.ssl.BlockMask.apply (tensor! [some 1, some 1, some 1]) 2 0 volume
  expect "volume hidden block"
    (TorchLean.Tensor.at? volumeMasked #[0, 0, 0] == some 0.0)
  expect "volume visible block"
    (TorchLean.Tensor.at? volumeMasked #[0, 0, 1] == some 2.0)

end NN.Tests.API.SelfSupervised.BlockMask
