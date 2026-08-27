/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.PyTorch.Import.Core
public import NN.Spec.Module.Conv
public import NN.Spec.Module.Linear

/-!
# CNN PyTorch Reference Import

CNN reference weight import from a PyTorch-style `state_dict`.

We mirror the common PyTorch naming convention for modules:

- `conv1.weight`, `conv1.bias`
- `conv2.weight`, `conv2.bias`
- `fc.weight`, `fc.bias`

Each tensor is expected to be encoded as nested JSON arrays whose shape matches the expected
TorchLean `Shape`.
-/

@[expose] public section


namespace Import
namespace CNNPyTorch
open PyTorch

open Spec
open Tensor
open Shape
open Lean
open Data
open Json

/-- Typed view of a PyTorch `state_dict` for the example 2-block CNN.

This matches the keys used by the exporter (`conv1.*`, `conv2.*`, `fc.*`) and pins down the exact
shapes expected by TorchLean.
-/
structure CnnStateDict (inC outC kH kW flatSize : Nat) where
  /-- First convolution kernel, PyTorch shape `(outC, inC, kH, kW)`. -/
  convW1 : Tensor Float [outC, inC, kH, kW]
  /-- First convolution bias. -/
  convB1 : Tensor Float [outC]
  /-- Second convolution kernel, PyTorch shape `(outC, outC, kH, kW)`. -/
  convW2 : Tensor Float [outC, outC, kH, kW]
  /-- Second convolution bias. -/
  convB2 : Tensor Float [outC]
  /-- Classifier weight, PyTorch shape `(outC, flatSize)`. -/
  linearW : Tensor Float [outC, flatSize]
  /-- Classifier bias. -/
  linearB : Tensor Float [outC]

/-- Load a CNN state dict from JSON (PyTorch `state_dict`-style keys). -/
def loadCnnStateDict (inC outC kH kW flatSize : Nat) (j : Json) : Option (CnnStateDict inC outC kH
  kW flatSize) :=
  let convW1Shape : Shape := [outC, inC, kH, kW]
  let convB1Shape : Shape := [outC]
  let convW2Shape : Shape := [outC, outC, kH, kW]
  let convB2Shape : Shape := [outC]
  let linearWShape : Shape := [outC, flatSize]
  let linearBShape : Shape := [outC]
  do
    -- Accepts both `{...}` and `{ "params": {...} }`.
    let o ← loadWeights? j
    let convW1 ← getTensor? o "conv1.weight" convW1Shape
    let convB1 ← getTensor? o "conv1.bias" convB1Shape
    let convW2 ← getTensor? o "conv2.weight" convW2Shape
    let convB2 ← getTensor? o "conv2.bias" convB2Shape
    let linearW ← getTensor? o "fc.weight" linearWShape
    let linearB ← getTensor? o "fc.bias" linearBShape
    pure { convW1 := convW1, convB1 := convB1, convW2 := convW2, convB2 := convB2, linearW :=
      linearW, linearB := linearB }

end CNNPyTorch
end Import
