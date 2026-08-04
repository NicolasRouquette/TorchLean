/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API

/-!
# Quickstart: Autograd Basics

Tour of the public autograd APIs beyond ordinary training:

- `autograd.model.*` for model-level VJP, Jacobian, and loss gradients.
- `autograd.model.OutputLoss.*` for reusable scalar losses on model outputs.
- `autograd.func.*` for Jacobian / Hessian APIs on single-input tensor functions.
- `nn.functional.detach` for stop-gradient behavior.

Run:
  `lake exe torchlean quickstart_autograd`
-/

@[expose] public section


namespace NN.Examples.Quickstart.AutogradBasics

open TorchLean

abbrev XShape := Shape.scalar.appendDim 2
abbrev YShape := Shape.scalar.appendDim 3
abbrev WShape : Shape := .dim 3 (.dim 2 .scalar)
abbrev BShape : Shape := .dim 3 .scalar

/-!
The tour stays on the public Float autodiff surface. It avoids:

- runtime tape/session code,
- hand-written parameter-shape bookkeeping,
- runtime scalar dispatch.

Instead, it uses `TorchLean.autograd.*` directly on a tiny fixed payload.
-/

def model : nn.Sequential XShape YShape :=
  -- One Linear layer: y = x ↦ W*x + b.
  nn.deterministic.linear 2 3 0 1

def mseLoss : autograd.model.OutputLoss YShape YShape :=
  -- Reusable scalar loss on model outputs: MSE(pred, target).
  autograd.model.OutputLoss.mse (τ := YShape)

def detachedMSELoss : autograd.model.OutputLoss YShape YShape :=
  -- Same forward value as `mseLoss`, but all gradients are zero (stop-gradient / detach).
  autograd.model.OutputLoss.detach mseLoss

def squareFn : autograd.func.Fn XShape XShape :=
  fun x => nn.functional.square x

def sumsqFn : autograd.func.Fn XShape Shape.scalar :=
  fun x => do
    let y ← nn.functional.square x
    -- `mean` is a convenient scalar reduction, like `torch.mean`.
    nn.functional.mean y

namespace Internal

/--
Deterministic model/sample payload for the autograd walkthrough.

The example stays small: one Linear layer, one input vector, one target vector, and one
fixed parameter-direction for JVP/HVP queries.
-/
structure DemoPayload (α : Type) where
  W : Tensor.T α WShape
  b : Tensor.T α BShape
  x : Tensor.T α XShape
  y : Tensor.T α YShape
  vW : Tensor.T α WShape
  vb : Tensor.T α BShape

/-- Fixed Float tensors used by the walkthrough. -/
def demoPayloadF : DemoPayload Float :=
  { W := tensorOfList! [3, 2] [
      0.2, -0.1,
      0.0,  0.3,
     -0.4,  0.1
    ]
    b := tensorOfList! [3] [0.01, -0.02, 0.03]
    x := tensorOfList! [2] [0.5, -1.2]
    y := tensorOfList! [3] [0.7, 0.1, -0.5]
    vW := Tensor.fill 0.1 WShape
    vb := Tensor.fill (-0.2) BShape }

/-- Parameter pack for the single Linear layer in `model`. -/
def modelParams {α : Type} (payload : DemoPayload α) :
    autograd.model.Params model α :=
  .cons payload.W (.cons payload.b .nil)

/-- Direction vector in parameter space used for JVP/HVP examples. -/
def paramDirection {α : Type} (payload : DemoPayload α) :
    autograd.model.Params model α :=
  .cons payload.vW (.cons payload.vb .nil)

/-- Unpack this tutorial's single Linear-layer parameter pack. -/
def unpackLinearParams {α : Type} (params : autograd.model.Params model α) :
    Tensor.T α WShape × Tensor.T α BShape := by
  simpa [autograd.model.Params, model, BShape] using tensorpack.unpackPair params

/-- Run the Float autograd walkthrough. -/
def runDemo : IO Unit := do
  let payload := demoPayloadF
  let params := modelParams payload
  let vparams := paramDirection payload

  -- ------------------------------------------------------------
  -- 1) Tensor-output compilation: VJP with an explicit output seed
  -- ------------------------------------------------------------
  --
  -- `vjpParams` computes a vector-Jacobian product (reverse-mode) for a tensor-output model.
  -- You provide an explicit output cotangent `seedOut` and get cotangents for the parameters.
  --
  -- Here the model output is `Vec 3`, and we choose `seedOut = ones`.
  -- Intuition: we backprop `sum(y)` w.r.t. parameters.
  let seedOut : Tensor.T Float YShape := Tensor.fill 1.0 YShape
  let vjpParams ←
    autograd.model.vjpParams (α := Float) model params payload.x seedOut

  let (dW, db) := unpackLinearParams vjpParams
  IO.println s!"vjpOutParams (seed=ones) dW = {Tensor.pretty dW}"
  IO.println s!"vjpOutParams (seed=ones) db = {Tensor.pretty db}"

  -- `jacrevParams` returns the full Jacobian of the model output w.r.t. parameters:
  -- one row per output coordinate. Each row is itself a typed list matching the parameter
  -- structure.
  let jacRows ←
    autograd.model.jacrevParams (α := Float) model params payload.x
  IO.println s!"jacrevOutParams rows = {jacRows.size} (should be size(out)=3)"
  for i in List.finRange jacRows.size do
    let row := jacRows[i.1]'i.2
    let (dWi, dbi) := unpackLinearParams row
    IO.println s!"  row[{i.1}] dW = {Tensor.pretty dWi}; db = {Tensor.pretty dbi}"

  -- ------------------------------------------------------------
  -- 2) Reverse-mode grad for scalar loss
  -- ------------------------------------------------------------
  --
  -- This is the "PyTorch-style training" case: differentiate a scalar loss.
  -- `valueAndGradParamsScalar` is the one-liner: it runs forward+backward and returns:
  -- - the scalar loss value
  -- - parameter gradients (same typed-list structure/order as `params`)
  let (lossMSE, gParams) ←
    autograd.model.valueAndGradParamsScalar (α := Float) model mseLoss params payload.x payload.y
  let (gW, gb) := unpackLinearParams gParams
  IO.println s!"loss(mse) = {lossMSE}"
  IO.println s!"gradParams (mse) gW = {Tensor.pretty gW}"
  IO.println s!"gradParams (mse) gb = {Tensor.pretty gb}"

  -- ------------------------------------------------------------
  -- 3) Detach semantics: same forward value, zero gradient
  -- ------------------------------------------------------------
  --
  -- This matches PyTorch `detach()`: stop-gradient on the loss computation.
  let (lossDetached, gParamsDetached) ←
    autograd.model.valueAndGradParamsScalar
      (α := Float) model detachedMSELoss params payload.x payload.y
  let (gW0, gb0) := unpackLinearParams gParamsDetached
  IO.println s!"loss(mse ∘ detach) = {lossDetached}"
  IO.println s!"gradParams (mse ∘ detach) gW = {Tensor.pretty gW0}"
  IO.println s!"gradParams (mse ∘ detach) gb = {Tensor.pretty gb0}"

  -- ------------------------------------------------------------
  -- 4) Forward-mode JVP of the scalar loss along a parameter direction
  -- ------------------------------------------------------------
  --
  -- JVP = Jacobian-vector product (forward-mode). Here we compute the directional derivative
  -- of the scalar loss along a parameter perturbation direction `vparams`.
  let dl ←
    autograd.model.jvpParams (α := Float) model mseLoss params payload.x payload.y vparams
  IO.println s!"jvpLossParams dl = {dl}"

  -- ------------------------------------------------------------
  -- 5) Hessian-vector product (HVP) w.r.t. parameters
  -- ------------------------------------------------------------
  --
  -- HVP = (Hessian of loss) applied to a direction vector, without materializing the full
  -- Hessian.
  let hvp ←
    autograd.model.hvpParams (α := Float) model mseLoss params payload.x payload.y vparams
  let (hW, hb) := unpackLinearParams hvp
  IO.println s!"hvpParams hW = {Tensor.pretty hW}"
  IO.println s!"hvpParams hb = {Tensor.pretty hb}"

  -- ------------------------------------------------------------
  -- 6) jacfwd/jacrev/hessian for a function of a single tensor input
  -- ------------------------------------------------------------
  let jacCols ← autograd.func.jacfwd (α := Float) squareFn payload.x
  IO.println s!"jacfwdInput(square) cols = {jacCols.size} (should be size(in)=2)"
  for i in List.finRange jacCols.size do
    let col := jacCols[i.1]'i.2
    IO.println s!"  col[{i.1}] = {Tensor.pretty col}"

  let hessCols ← autograd.func.hessian (α := Float) sumsqFn payload.x
  IO.println s!"hessianInput(mean(x^2)) cols = {hessCols.size} (should be size(in)=2)"
  for i in List.finRange hessCols.size do
    let col := hessCols[i.1]'i.2
    IO.println s!"  H*e[{i.1}] = {Tensor.pretty col}"

  -- ------------------------------------------------------------
  -- 7) One-liners: vjp / jacrev / grad / valueAndGrad
  -- ------------------------------------------------------------
  --
  -- These entrypoints cover the common case of a single tensor input.
  let seedSq : Tensor.T Float XShape := Tensor.fill 1.0 XShape
  let vjpSq ← autograd.func.vjp (α := Float) squareFn payload.x seedSq
  IO.println s!"vjp(square, seed=ones) = {Tensor.pretty vjpSq}"

  let jacRowsSq ← autograd.func.jacrev (α := Float) squareFn payload.x
  IO.println s!"jacrev1(square) rows = {jacRowsSq.size} (should be size(out)=2)"
  for i in List.finRange jacRowsSq.size do
    let row := jacRowsSq[i.1]'i.2
    IO.println s!"  row[{i.1}] = {Tensor.pretty row}"

  let gSumsq ← autograd.func.grad (α := Float) sumsqFn payload.x
  IO.println s!"grad1(mean(x^2)) = {Tensor.pretty gSumsq}"

  let (valSumsq, gSumsq2) ← autograd.func.valueAndGradScalar (α := Float) sumsqFn payload.x
  IO.println s!"valueAndGradScalar(mean(x^2)) value = {valSumsq}, grad = {Tensor.pretty gSumsq2}"

end Internal

/-- Command-line help for the Float autograd quickstart. -/
def usage : String :=
  String.intercalate "\n"
    [ "TorchLean autograd quickstart"
    , ""
    , "Usage:"
    , "  lake exe torchlean quickstart_autograd"
    , ""
    , "This demo has no tutorial-specific flags."
    ]

/-- CLI entrypoint for the Float autograd quickstart. -/
def main (args : List String) : IO Unit := do
  let args := CLI.dropDashDash args
  if CLI.hasHelp args then
    IO.println usage
    return
  CLI.requireNoArgs "quickstart_autograd" args
  Internal.runDemo

end NN.Examples.Quickstart.AutogradBasics
