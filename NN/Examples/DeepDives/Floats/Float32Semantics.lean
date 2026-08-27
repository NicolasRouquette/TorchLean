/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Batteries.Data.Float.Basic
public import NN.API
public import NN.Floats
public import NN.Widgets

/-!
# Float32 Semantics

This tutorial runs the same compact MLP under two executable binary32 semantics:

- `Float32`: Lean's native binary32 type and runtime implementation;
- `TorchLean.Floats.IEEE32Exec`: TorchLean's executable bit-level IEEE-754 binary32 model.

We run a single forward pass and a single reverse-mode VJP (seeded with `1.0`) and then report
`max_abs_diff` between native Float32 and IEEE32Exec after converting both results to host `Float`
for display.

PyTorch comparison:

```python
import torch
from torch import nn

model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 1)).to(torch.float32)

# PyTorch runs the model through its selected float32 kernels.
```

The TorchLean example adds an independent second execution. Lean's native `Float32` supplies the
ordinary runtime path, while `IEEE32Exec` computes from explicit binary32 fields and rounding rules.
The model and autograd program stay unchanged because their scalar type is generic.

Run:
  `lake exe torchlean float32_semantics`

For editor inspection, put the cursor on the `#float32_*` commands below. Those widgets are for
visualization only; the actual tutorial code uses ordinary `def` and `IO` definitions.
-/

@[expose] public section


namespace NN.Examples.DeepDives.Floats.Float32Semantics

open _root_.Spec
open _root_.Spec.Tensor
open _root_.TorchLean
open _root_.TorchLean.Floats.IEEE754

/-
This tutorial uses the public TorchLean surface:

- models: `nn.Sequential` built from `nn.*`
- execution: `nn.lowerToTypedGraph model` + `graph.forward`
- autodiff: `autograd.model.vjpState` / `vjpInput`

No `Runtime.Autograd.*` tape/session machinery appears in this file.
-/

/-! ## Float32 widget probes -/

/--
`0.1` is the canonical "not exactly representable" decimal. The widget shows the binary32 value
that `IEEE32Exec` receives after rounding from the host literal.

PyTorch analogue:

```python
torch.tensor(0.1, dtype=torch.float32)
```
-/
def decimalTenth : Float := 0.1

/-- A simple finite binary32 value for the bit-layout widget. -/
def one32 : IEEE32Exec := IEEE32Exec.ofFloat 1.0

/-- Canonical quiet NaN, useful for showing classification and comparison behavior. -/
def quietNaN32 : IEEE32Exec := IEEE32Exec.canonicalNaN

#float32_round_view decimalTenth
#float32_view one32
#float32_view quietNaN32
#float32_compare_view one32, quietNaN32

def model : nn.Sequential [2] [1] :=
  -- A compact 2-layer MLP with ReLU:
  --   Linear(2 -> 3) -> ReLU -> Linear(3 -> 1)
  --
  -- This uses the public `nn` surface (PyTorch-like layer stacking and named configs).
  --
  -- Note: this tutorial supplies *explicit* parameter tensors below, so the init seeds are irrelevant
  -- here.
  nn.build 0 <| nn.blocks.mlp 2 1 { hidden := [3], activation := .relu }

-- This tutorial returns a single typed bundle so we can:
-- - print everything in one place, and
-- - compare native Float32 with IEEE32Exec numerically at the end.
def OutShapes : List Spec.Shape :=
  [[1], [3, 2], [3],
   [1, 3], [1], [2]]
abbrev OutPack (α : Type) :=
  _root_.TorchLean.TensorPack α OutShapes

def runOnce {α : Type}
    [_root_.Context α] [DecidableEq Spec.Shape] [ToString α] [Runtime.FromFloat α]
    (tag : String) : IO (OutPack α) := do
  -- Public data boundaries use `Float` literals and convert them into the selected scalar `α`.
  let cast : Float → α := Runtime.ofFloat

  /-
  ### 1. Explicit parameters

  PyTorch analogue:

  ```python
  with torch.no_grad():
      model[0].weight.copy_(torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
      model[0].bias.copy_(torch.tensor([0.1, 0.2, 0.3]))
      model[2].weight.copy_(torch.tensor([[0.7, 0.8, 0.9]]))
      model[2].bias.copy_(torch.tensor([0.4]))
  ```

  `autograd.model.State model α` is a typed tensor pack (`_root_.TorchLean.TensorPack`). The model
  determines its shapes, so the parameter order cannot be silently permuted.
  -/
  let params : autograd.model.State model α :=
    _root_.TorchLean.TensorPack!
      (Spec.Tensor.map cast (tensorOfArray! (ty := Float) [3, 2] #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6])),
      (Spec.Tensor.map cast (tensorOfArray! (ty := Float) [3] #[0.1, 0.2, 0.3])),
      (Spec.Tensor.map cast (tensorOfArray! (ty := Float) [1, 3] #[0.7, 0.8, 0.9])),
      (Spec.Tensor.map cast (tensorOfArray! (ty := Float) [1] #[0.4]))

  -- One input vector x in R^2.
  let x : Tensor α [2] :=
    Spec.Tensor.map cast (tensorOfArray! (ty := Float) [2] #[0.5, 0.8])

  /-
  ### 2. Forward pass

  PyTorch analogue:

  ```python
  y = model(x)
  ```

  `nn.lowerToTypedGraph` records the model's forward program as a reusable typed SSA graph. TorchLean
  models are scalar-generic, so this step chooses the scalar `α` used by that executable graph.
  -/
  let graph ← nn.lowerToTypedGraph model (α := α)
  -- Run `y = model(params, x)` (single-example predict; no batching here).
  let y := nn.TypedGraphModel.forward graph params x

  /-
  ### 3. Reverse-mode VJP

  PyTorch analogue:

  ```python
  y.sum().backward()
  dState = [p.grad for p in model.parameters()]
  inputGrad = x.grad
  ```

  A VJP needs an output cotangent seed. Since the output shape is `[1]`, seeding with `[1]`
  computes the same gradient as differentiating `sum(y)`.
  -/
  let seedOut : Tensor α [1] :=
    Spec.fill (α := α) (cast 1.0) [1]

  -- Gradients w.r.t. *parameters* (same `_root_.TorchLean.TensorPack` structure/order as `params`).
  let dState ← autograd.model.vjpState (α := α) model params x seedOut
  -- Gradients w.r.t. *inputs* (here: just the input vector `inputGrad`, no tensor-pack noise).
  let inputGrad ← autograd.model.vjpInput (α := α) model params x seedOut

  /- TorchLean keeps each parameter-gradient shape in the type of the gradient pack. -/
  let .cons hiddenWeightGrad
      (.cons hiddenBiasGrad (.cons outputWeightGrad (.cons outputBiasGrad .nil))) := dState

  IO.println s!"== {tag} =="
  IO.println s!"y   = {Spec.pretty y}"
  IO.println s!"hiddenWeightGrad = {Spec.pretty hiddenWeightGrad}"
  IO.println s!"hiddenBiasGrad = {Spec.pretty hiddenBiasGrad}"
  IO.println s!"outputWeightGrad = {Spec.pretty outputWeightGrad}"
  IO.println s!"outputBiasGrad = {Spec.pretty outputBiasGrad}"
  IO.println s!"inputGrad  = {Spec.pretty inputGrad}"

  pure (_root_.TorchLean.TensorPack! y, hiddenWeightGrad, hiddenBiasGrad, outputWeightGrad, outputBiasGrad, inputGrad)

def maxAbsDiffTensor {s : Spec.Shape} (a b : Spec.Tensor Float s) : Float :=
  let diffs :=
    (Spec.Tensor.toArray a).zip (Spec.Tensor.toArray b) |>.map (fun (x, y) => Float.abs (x - y))
  diffs.foldl max 0.0

def unpackOutPack {α : Type} (p : OutPack α) :
    Tensor α [1] ×
      Tensor α [3, 2] ×
      Tensor α [3] ×
      Tensor α [1, 3] ×
      Tensor α [1] ×
      Tensor α [2] :=
  match p with
  | .cons y
      (.cons hiddenWeightGrad
        (.cons hiddenBiasGrad (.cons outputWeightGrad (.cons outputBiasGrad (.cons inputGrad .nil))))) =>
      (y, hiddenWeightGrad, hiddenBiasGrad, outputWeightGrad, outputBiasGrad, inputGrad)

def maxAbsDiffPack (a b : OutPack Float) : Float :=
  let (ay, aHiddenWeightGrad, aHiddenBiasGrad, aOutputWeightGrad, aOutputBiasGrad, aInputGrad) :=
    unpackOutPack a
  let (by_, bHiddenWeightGrad, bHiddenBiasGrad, bOutputWeightGrad, bOutputBiasGrad, bInputGrad) :=
    unpackOutPack b
  max
    (max (maxAbsDiffTensor ay by_) (maxAbsDiffTensor aHiddenWeightGrad bHiddenWeightGrad))
    (max
      (max (maxAbsDiffTensor aHiddenBiasGrad bHiddenBiasGrad)
        (maxAbsDiffTensor aOutputWeightGrad bOutputWeightGrad))
      (max (maxAbsDiffTensor aOutputBiasGrad bOutputBiasGrad) (maxAbsDiffTensor aInputGrad bInputGrad)))

/-- Command-line help for the Float32 semantics tutorial. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean Float32 semantics tutorial"
    , ""
    , "Usage:"
    , "  lake exe torchlean float32_semantics"
    , ""
    , "This demo has no tutorial-specific flags."
    ]

def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return
  CLI.requireNoArgs "float32_semantics" args
  IO.println "== Float32 semantics tutorial =="
  IO.println
    "Note: rounded-real binary32 is proof-only and is selected directly in theorem statements."
  IO.println "[TorchLean] FP32: finite rounded-real proof model"
  IO.println "[TorchLean] IEEE32Exec: bit-level binary32 reference"

  let rNative ← runOnce (α := Float32) "Float32 (native runtime)"
  let r32 ← runOnce (α := TorchLean.Floats.IEEE32Exec) "IEEE32Exec"

  let rNativeF : OutPack Float :=
    _root_.TorchLean.TensorPack.map (α := Float32) (β := Float)
      (fun {_s} t => Tensor.map Float32.toFloat t)
      rNative
  let r32F : OutPack Float :=
    _root_.TorchLean.TensorPack.map (α := TorchLean.Floats.IEEE32Exec) (β := Float)
      (fun {_s} t => Tensor.map TorchLean.Floats.IEEE754.IEEE32Exec.toFloat t)
      r32

  let diff := maxAbsDiffPack rNativeF r32F
  IO.println s!"max_abs_diff(Float32 vs IEEE32Exec) = {diff.toStringFull}"

end NN.Examples.DeepDives.Floats.Float32Semantics
