/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.API.Verification
public import NN.Verification.TorchLean.Lowering

/-!
# TorchLean Transformer IBP Workflow

Small end-to-end workflow:

TorchLean (MHA + LayerNorm + MSE) → lower to `NN.IR.Graph` → run:
- IBP (`runIBP`)
- basic CROWN forward bounds (`runCROWN`)
- objective-dependent backward/dual CROWN (`runCROWNBackwardObjective`)

Run:
  `lake exe verify -- torchlean-transformer-ibp`
  `lake exe verify -- torchlean-transformer-ibp --with-crown`
  `lake exe verify -- torchlean-transformer-ibp --scalar ieee32-exec`
-/

@[expose] public section


namespace NN.Verification.TorchLean.TransformerIBPWorkflow

open _root_.Spec
open _root_.Spec.Tensor
open _root_.TorchLean

open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN

/-- Sequence length for the transformer verification example. -/
def n : Nat := 2
/-- Model embedding dimension. -/
def dModel : Nat := 2
/-- Number of attention heads. -/
def numHeads : Nat := 1
/-- Per-head embedding dimension. -/
def headDim : Nat := 2
/-- Batch size for the transformer verification example. -/
def batch : Nat := 1

/-- Input shape `(batch × n × dModel)`. -/
def xShape : Spec.Shape := .dim batch (.dim n (.dim dModel .scalar))
/-- Projection weight shape for Q/K/V: `(dModel × (numHeads*headDim))`. -/
def wProjShape : Spec.Shape := .dim dModel (.dim (numHeads * headDim) .scalar)
/-- Output projection weight shape: `((numHeads*headDim) × dModel)`. -/
def wOShape : Spec.Shape := .dim (numHeads * headDim) (.dim dModel .scalar)
/-- LayerNorm scale parameter shape, matching the feature dimension. -/
def gammaShape : Spec.Shape := .dim dModel .scalar
/-- LayerNorm beta shape, matching the feature dimension. -/
def betaShape : Spec.Shape := .dim dModel .scalar
/-- MSE target shape (matches the model output shape). -/
def targetShape : Spec.Shape := xShape

/-- Parameter shapes list for `modelLoss` (`Wq,Wk,Wv,Wo,gamma,beta,target`). -/
def paramShapes : List Spec.Shape :=
  [wProjShape, wProjShape, wProjShape, wOShape, gammaShape, betaShape, targetShape]

/-- TorchLean program: `mha -> layer_norm -> mse_loss`, returning a scalar loss. -/
def modelLoss {α : Type} [Context α] [DecidableEq Spec.Shape] :
    _root_.Runtime.Autograd.TorchLean.Program α (paramShapes ++ [xShape])
      Spec.Shape.scalar :=
  fun {m} _ _ =>
    fun wq wk wv wo gamma beta target x =>
      (do
        let y ← _root_.Runtime.Autograd.TorchLean.multiHeadAttention (m := m) (α := α)
          (leadingShape := [batch]) (n := n) (numHeads := numHeads) (dModel := dModel)
          (headDim := headDim)
          (hN := by decide) wq wk wv wo x (mask := none)
        let yLn ← _root_.Runtime.Autograd.TorchLean.layerNorm (m := m) (α := α)
          (leading := [batch, n]) (width := dModel) (hWidth := by decide) y gamma beta
        Runtime.mseLoss (m := m) (α := α) (s := xShape) yLn target
        : m (Runtime.ValueRef (m := m) (α := α) Spec.Shape.scalar))

/-- Runtime-selected typed runner used by the CLI entrypoint. -/
def runMain {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    [Runtime.FromFloat α] [BoundOps α] (withCrown : Bool) : IO Unit := do
  let cast : Float → α := Runtime.ofFloat
  let params : _root_.TorchLean.TensorPack α paramShapes :=
    _root_.TorchLean.TensorPack!
      (Spec.Tensor.map cast (tensorOfArray! (ty := Float) [2, 2] #[1.0, 0.0, 0.0, 1.0])),
      (Spec.Tensor.map cast (tensorOfArray! (ty := Float) [2, 2] #[1.0, 0.0, 0.0, 1.0])),
      (Spec.Tensor.map cast (tensorOfArray! (ty := Float) [2, 2] #[1.0, 0.0, 0.0, 1.0])),
      (Spec.Tensor.map cast (tensorOfArray! (ty := Float) [2, 2] #[1.0, 0.0, 0.0, 1.0])),
      (Spec.Tensor.map cast (tensorOfArray! (ty := Float) [2] #[1.0, 1.0])),
      (Spec.Tensor.map cast (tensorOfArray! (ty := Float) [2] #[0.0, 0.0])),
      (Spec.Tensor.map cast (tensorOfArray! (ty := Float) [1, 2, 2] #[0.0, 0.0, 0.0, 0.0]))

  let lowered ←
    match Verification.lowerProgramToIR
          (α := α) (paramShapes := paramShapes) (σ := xShape)
          (τ := Spec.Shape.scalar)
          (modelLoss (α := α)) params with
    | .ok c => pure c
    | .error e => throw <| IO.userError e

  IO.println s!"lowered IR nodes: {lowered.graph.nodes.size}"

  let x0 : Spec.Tensor α xShape :=
    Spec.Tensor.map cast (tensorOfArray! (ty := Float) [1, 2, 2] #[0.2, -0.3, 0.7, 0.1])
  let eps : α := Runtime.ofFloat 0.05
  let xB : FlatBox α := NN.Verification.TorchLean.lInfBall (α := α) x0 eps
  let ps : ParamStore α := lowered.seedInputBox xB

  let boxes := lowered.runIBP ps
  let outB ← lowered.outputBoxOrThrow boxes
  IO.println s!"[IBP] loss lo: {pretty outB.lo}"
  IO.println s!"[IBP] loss hi: {pretty outB.hi}"

  if !withCrown then
    IO.println "[CROWN] skipped for the default runtime-check path; pass --with-crown for the heavier transformer CROWN run"
    return ()

  IO.println "[CROWN] running transformer-scale forward CROWN; this experimental path can take minutes"
  let inputDim := Spec.Shape.size xShape
  match lowered.outputBoxCROWN? ps xB with
  | .ok outC =>
      if hOut : outC.dim = 1 then
          IO.println s!"[CROWN] loss lo: {pretty outC.lo}"
          IO.println s!"[CROWN] loss hi: {pretty outC.hi}"
      else
        IO.println s!"[CROWN] unexpected output dim {outC.dim} (expected 1)"
  | .error msg =>
      IO.println s!"[CROWN] {msg}"

  IO.println "[CROWN-backward] running objective-dependent backward CROWN"
  let obj : FlatTensor α := { n := 1, v := Spec.fill (α := α) Numbers.one (.dim 1 .scalar) }
  match lowered.backwardObjectiveBox? ps boxes xB obj with
  | .ok outC =>
      IO.println s!"[CROWN-backward] loss lo: {pretty outC.lo}"
      IO.println s!"[CROWN-backward] loss hi: {pretty outC.hi}"
  | .error msg =>
      IO.println s!"[CROWN-backward] {msg}"

/-- Runtime-selected typed runner for the default IBP-only path. -/
def runMainDefault {α : Type}
    [_root_.Context α] [DecidableEq Spec.Shape] [ToString α] [Runtime.FromFloat α]
    [BoundOps α] : IO Unit :=
  runMain (α := α) false

/-- Runtime-selected typed runner for the heavier IBP+CROWN path. -/
def runMainWithCrown {α : Type}
    [_root_.Context α] [DecidableEq Spec.Shape] [ToString α] [Runtime.FromFloat α]
    [BoundOps α] : IO Unit :=
  runMain (α := α) true

/--
CLI entry point for the transformer-IBP workflow.

This is wired into `lake exe verify -- torchlean-transformer-ibp`.

By default this command is a fast validation check: lower the TorchLean transformer fragment to the
verification IR and run IBP on the scalar loss. Pass `--with-crown` to also run the experimental
transformer-scale CROWN passes. The separate `torchlean-crown-ops` command keeps CROWN itself in the
standard check suite on compact graphs, while this file focuses on the heavier attention/layer-norm
front-end path.
-/
def main (args : List String) : IO Unit := do
  let parsedWithCrown : Bool × List String ←
    match CLI.takeBoolFlagOnce args "with-crown" with
    | .ok parsed => pure parsed
    | .error msg => throw <| IO.userError msg
  let withCrown : Bool := parsedWithCrown.1
  let restArgs : List String := parsedWithCrown.2
  if withCrown then
    NN.Verification.TorchLean.runWithBoundScalar
      "TorchLean (MHA+LayerNorm+MSE) → IR → IBP" restArgs
      (@runMainWithCrown)
  else
    NN.Verification.TorchLean.runWithBoundScalar
      "TorchLean (MHA+LayerNorm+MSE) → IR → IBP" restArgs
      (@runMainDefault)

end NN.Verification.TorchLean.TransformerIBPWorkflow
