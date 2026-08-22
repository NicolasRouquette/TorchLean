/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.Verification.TorchLean.Lowering

/-!
# TorchLean CROWN Ops Workflow

Running CROWN end-to-end on small TorchLean graphs.

We lower TorchLean programs to the verifier IR (`NN.IR.Graph`), then run:
- IBP (`runIBP`)
- basic CROWN forward bounds (`runCROWN`)
- objective-dependent backward/dual CROWN (`runCROWNBackwardObjective`)

The workflow gives compact, fast coverage for nonlinear ops added to CROWN:
- `softmax` (vector)
- `mse_loss` (vector → scalar)

For attention + `layer_norm`, see
  `NN/Verification/TorchLean/TransformerIBPWorkflow.lean`.

Run:
  `lake exe verify -- torchlean-crown-ops`
  `lake exe verify -- torchlean-crown-ops --scalar ieee32-exec`
-/

@[expose] public section


namespace NN.Verification.TorchLean.CrownOpsWorkflow

open _root_.Spec
open _root_.Spec.Tensor
open _root_.TorchLean

open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN

/-- Input dimension for the softmax workflow model. -/
def softmaxInDim : Nat := 2
/-- Output dimension for the softmax workflow model. -/
def softmaxOutDim : Nat := 3

/-- Input shape for the softmax workflow model. -/
def softmaxXShape : Spec.Shape := .dim softmaxInDim .scalar
/-- Output shape for the softmax workflow model. -/
def softmaxYShape : Spec.Shape := .dim softmaxOutDim .scalar

/-- TorchLean model: `Linear -> Softmax`. -/
def softmaxModel : nn.Sequential softmaxXShape softmaxYShape := by
  letI : Spec.Shape.AxisInBounds 0 softmaxYShape := by
    unfold softmaxYShape softmaxOutDim
    infer_instance
  exact
  nn.build 0 <|
    nn.Sequential![
      nn.linear softmaxInDim softmaxOutDim,
      nn.softmax 0
    ]

/-- Parameter shapes for `softmaxModel`. -/
def softmaxParamShapes : List Spec.Shape := nn.stateShapes softmaxModel

/-- Example margin functional on softmax outputs
($\mathrm{lo}_0-\mathrm{hi}_1$). -/
def softmaxMargin {α : Type} [Context α]
    (lo hi : Spec.Tensor α softmaxYShape) : α :=
  let lo0 := _root_.Spec.Tensor.vecGet lo fin0!
  let hi1 := _root_.Spec.Tensor.vecGet hi fin1!
  lo0 - hi1

/--
Run the softmax workflow under a chosen scalar backend `α`.

This lowers the TorchLean model to verifier IR and prints IBP/CROWN bounds.
-/
def runSoftmax {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    [Runtime.FromFloat α] [BoundOps α] : IO Unit := do
  IO.println "== Workflow 1: linear -> softmax (vector) =="
  let cast : Float → α := Runtime.ofFloat

  let params : TensorPack α softmaxParamShapes :=
    TensorPack!
      (NN.Tensor.ofListOfLength (α := α) [3, 2]
        [ cast 1.0, cast (-0.5)
        , cast 0.2, cast 0.7
        , cast (-0.3), cast 0.1
        ] (by rfl)),
      (NN.Tensor.ofListOfLength (α := α) [3] [cast 0.1, cast (-0.2), cast 0.0] (by rfl))

  let lowered ←
    match Verification.lowerProgramToIR
          (α := α) (paramShapes := softmaxParamShapes) (σ := softmaxXShape) (τ := softmaxYShape)
          (nn.forward softmaxModel (α := α)) params with
    | .ok c => pure c
    | .error e => throw <| IO.userError e

  IO.println s!"lowered IR nodes: {lowered.graph.nodes.size}"

  let x0 : Spec.Tensor α softmaxXShape :=
    NN.Tensor.ofListOfLength (α := α) [2] [cast 0.2, cast (-0.1)] (by rfl)
  let eps : α := Runtime.ofFloat 0.05
  let xB : FlatBox α := Verification.lInfBall (α := α) x0 eps
  let ps : ParamStore α := lowered.seedInputBox xB

  -- IBP
  let ibp := lowered.runIBP ps
  let outB ← lowered.outputBoxOrThrow ibp
  if hDim : outB.dim = softmaxOutDim then
    let loY : Spec.Tensor α softmaxYShape := by
      simpa [softmaxYShape] using outB.loAsDim hDim
    let hiY : Spec.Tensor α softmaxYShape := by
      simpa [softmaxYShape] using outB.hiAsDim hDim
    IO.println s!"[IBP] p lo = {pretty loY}"
    IO.println s!"[IBP] p hi = {pretty hiY}"
    IO.println s!"[IBP] margin(p0 - p1) = {softmaxMargin (α := α) loY hiY}"
  else
    IO.println s!"[IBP] unexpected output dim {outB.dim} (expected {softmaxOutDim})"

  -- CROWN (forward, affine lower+upper)
  match lowered.outputBoxCROWN? ps xB with
  | .ok outC =>
      if hOut : outC.dim = softmaxOutDim then
          let loY : Spec.Tensor α softmaxYShape := by
            simpa [softmaxYShape] using outC.loAsDim hOut
          let hiY : Spec.Tensor α softmaxYShape := by
            simpa [softmaxYShape] using outC.hiAsDim hOut
          IO.println s!"[CROWN] p lo = {pretty loY}"
          IO.println s!"[CROWN] p hi = {pretty hiY}"
          IO.println s!"[CROWN] margin(p0 - p1) = {softmaxMargin (α := α) loY hiY}"
      else
        IO.println s!"[CROWN] unexpected output dim {outC.dim} (expected {softmaxOutDim})"
  | .error msg =>
      IO.println s!"[CROWN] {msg}"

  -- Backward/dual CROWN for the margin objective: p0 - p1.
  let objV : Spec.Tensor α (.dim softmaxOutDim .scalar) :=
    NN.Tensor.ofListOfLength (α := α) [3] [cast 1.0, cast (-1.0), cast 0.0] (by rfl)
  let obj : FlatVec α := { n := softmaxOutDim, v := objV }
  match lowered.backwardObjectiveBox? ps ibp xB obj with
  | .ok outC =>
      let loM : α := getAtOrZero outC.lo [0]
      let hiM : α := getAtOrZero outC.hi [0]
      IO.println s!"[CROWN-backward] margin lo = {loM}"
      IO.println s!"[CROWN-backward] margin hi = {hiM}"
  | .error msg =>
      IO.println s!"[CROWN-backward] {msg}"

/-- Input dimension for the MSE-loss workflow model. -/
def mseInDim : Nat := 2
/-- Output dimension for the MSE-loss workflow model. -/
def mseOutDim : Nat := 2

/-- Weight shape for the MSE-loss workflow's linear layer. -/
def mseWShape : Spec.Shape := .dim mseOutDim (.dim mseInDim .scalar)
/-- Bias shape for the MSE-loss workflow's linear layer. -/
def mseBShape : Spec.Shape := .dim mseOutDim .scalar
/-- Input shape for the MSE-loss workflow. -/
def mseXShape : Spec.Shape := .dim mseInDim .scalar
/-- Output shape for the MSE-loss workflow. -/
def mseYShape : Spec.Shape := .dim mseOutDim .scalar

/-- Parameter shapes for the MSE-loss workflow (`[W,b,target]`). -/
def mseParamShapes : List Spec.Shape := [mseWShape, mseBShape, mseYShape]

/-- TorchLean forward program computing
$\widehat{y}=\operatorname{linear}(x)$ and
$\operatorname{mse\_loss}(\widehat{y},\mathrm{target})$, returning a scalar. -/
def mseLossModel {α : Type} [Context α] [DecidableEq Spec.Shape] :
    _root_.Runtime.Autograd.TorchLean.Program α (mseParamShapes ++ [mseXShape])
      Spec.Shape.scalar :=
  fun {m} _ _ =>
    fun w b target x =>
      (do
        let yhat ← Ops.linear (m := m) (α := α) (inDim := mseInDim) (outDim := mseOutDim) w b x
        Ops.mseLoss (m := m) (α := α) (s := mseYShape) yhat target
        : m (Ops.RefTy (m := m) (α := α) Spec.Shape.scalar))

/--
Run the MSE-loss workflow under a chosen scalar backend `α`.

This lowers the TorchLean forward computation to verifier IR and prints IBP/CROWN bounds for the scalar
  loss.
-/
def runMSE {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    [Runtime.FromFloat α] [BoundOps α] : IO Unit := do
  IO.println "== Workflow 2: linear -> mse_loss (scalar) =="
  let cast : Float → α := Runtime.ofFloat

  let params : TensorPack α mseParamShapes :=
    TensorPack!
      (NN.Tensor.ofListOfLength (α := α) [2, 2]
        [cast 0.4, cast (-0.3), cast 1.2, cast 0.1] (by rfl)),
      (NN.Tensor.ofListOfLength (α := α) [2] [cast 0.05, cast (-0.02)] (by rfl)),
      (NN.Tensor.ofListOfLength (α := α) [2] [cast 0.0, cast 1.0] (by rfl))

  let lowered ←
    match Verification.lowerProgramToIR
          (α := α) (paramShapes := mseParamShapes) (σ := mseXShape)
          (τ := Spec.Shape.scalar)
          (mseLossModel (α := α)) params with
    | .ok c => pure c
    | .error e => throw <| IO.userError e

  IO.println s!"lowered IR nodes: {lowered.graph.nodes.size}"

  let x0 : Spec.Tensor α mseXShape :=
    NN.Tensor.ofListOfLength (α := α) [2] [cast 0.3, cast (-0.4)] (by rfl)
  let eps : α := Runtime.ofFloat 0.05
  let xB : FlatBox α := Verification.lInfBall (α := α) x0 eps
  let ps : ParamStore α := lowered.seedInputBox xB

  -- IBP
  let ibp := lowered.runIBP ps
  let outB ← lowered.outputBoxOrThrow ibp
  IO.println s!"[IBP] loss lo = {pretty outB.lo}"
  IO.println s!"[IBP] loss hi = {pretty outB.hi}"

  -- CROWN forward bounds on the scalar loss.
  match lowered.outputBoxCROWN? ps xB with
  | .ok outC =>
      if hOut : outC.dim = 1 then
          IO.println s!"[CROWN] loss lo = {pretty outC.lo}"
          IO.println s!"[CROWN] loss hi = {pretty outC.hi}"
      else
        IO.println s!"[CROWN] unexpected output dim {outC.dim} (expected 1)"
  | .error msg =>
      IO.println s!"[CROWN] {msg}"

  -- Backward/dual CROWN for the loss objective itself (obj = 1).
  let obj : FlatVec α := { n := 1, v := Spec.fill (α := α) Numbers.one (.dim 1 .scalar) }
  match lowered.backwardObjectiveBox? ps ibp xB obj with
  | .ok outC =>
      IO.println s!"[CROWN-backward] loss lo = {pretty outC.lo}"
      IO.println s!"[CROWN-backward] loss hi = {pretty outC.hi}"
  | .error msg =>
      IO.println s!"[CROWN-backward] {msg}"

/-- Run all CROWN-ops workflows (softmax + mse_loss) under a chosen scalar backend `α`. -/
def runOnce {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    [Runtime.FromFloat α] [BoundOps α] : IO Unit := do
  runSoftmax (α := α)
  IO.println ""
  runMSE (α := α)

/--
CLI entry point for the CROWN-ops workflow.

This is wired into `lake exe verify -- torchlean-crown-ops`.
-/
def main (args : List String) : IO Unit :=
  NN.Verification.TorchLean.runWithBoundScalar
    "TorchLean → IR → IBP + CROWN (ops: softmax/mse_loss)" args
    (@runOnce)

end NN.Verification.TorchLean.CrownOpsWorkflow
