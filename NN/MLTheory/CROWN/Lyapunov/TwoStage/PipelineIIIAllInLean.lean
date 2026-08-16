/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec
public import NN.API.Macros
public import NN.API.TensorPack
public import NN.MLTheory.CROWN.Core
public import NN.MLTheory.CROWN.Graph
public import NN.MLTheory.CROWN.Lyapunov.TwoStage.Core
public import NN.MLTheory.CROWN.Lyapunov.TwoStage.LossAnalysis
public import NN.MLTheory.CROWN.Lyapunov.TwoStage.ExecUtils
public import NN.Runtime.Autograd.TorchLean.Autodiff
public import NN.Runtime.Autograd.TorchLean.Program
public import NN.Runtime.Autograd.TorchLean.Module
public import NN.Verification.TorchLean.Lowering

/-!
# Pipeline (iii): All-in-Lean TwoStage refinement + IBP/CROWN check

This file corresponds to **Figure 7 (iii)** in the TorchLean paper (`arXiv:2602.22631`).

Everything runs *inside Lean*:
- Stage 1: sample training points in a box and train parameters (SGD) under exact `IEEE32Exec`.
- Stage 2: for each round, run a small PGD loop on the input `x` to find “counterexample-ish”
  points, then train on them (CEGIS flavor).
- Final: lower the same TorchLean loss program to the shared verifier IR and run in-repo IBP/CROWN
  bound propagation to check the loss on a small box around the origin.

The workflow uses the in-repo IBP/CROWN engine rather than reproducing every optimization from
external α/β-CROWN systems. The complete computation, including its float32 semantics, runs inside
Lean and requires no external verifier.

Run:
`lake exe verify -- twostage-torchlean-cegis-van`
-/

@[expose] public section


open Spec
open Tensor

namespace NN.MLTheory.CROWN.Lyapunov.TwoStage.PipelineIII.AllInLean

open _root_.TorchLean.Floats.IEEE754
open _root_.Runtime
open _root_.Runtime.Autograd
open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN

open NN.MLTheory.CROWN.Lyapunov.TwoStage.Core
open NN.MLTheory.CROWN.Lyapunov.TwoStage.ExecUtils

abbrev α : Type := IEEE32Exec

abbrev xDim : Nat := Core.xDim
abbrev uDim : Nat := Core.uDim

abbrev xShape : Shape := Core.xShape
abbrev uShape : Shape := Core.uShape

abbrev paramShapes (width : Nat) : List Shape :=
  Core.paramShapes width

/-- Local alias for `ExecUtils.nat` (coercion `Nat → IEEE32Exec`). -/
abbrev nat (k : Nat) : α := ExecUtils.nat k

/-- Learning rate for the stage-1 and stage-2 SGD loops. -/
def lr : α := ExecUtils.defaultLr

/-- PGD step size when searching for counterexample-ish inputs. -/
def pgdStepSize : α := ExecUtils.defaultPgdStepSize

/-- Radius of the training box `[-rad, rad]^2` (also used for clamping PGD iterates). -/
def rad : α := ExecUtils.defaultRad

/-- Half-width of the small box around the origin used for the final IBP/CROWN post-check. -/
def epsCheck : α := ExecUtils.defaultEpsCheck

def lossProg (width : Nat) :
    ∀ {β : Type}, [Context β] → [DecidableEq Shape] →
      TorchLean.Program β (paramShapes width ++ [xShape]) Shape.scalar :=
  Core.lossProgram width

def initParamsF (width : Nat) : _root_.TorchLean.TensorPack Float (paramShapes width) :=
  let wC : Tensor Float (.dim uDim (.dim xDim .scalar)) :=
    _root_.Runtime.Autograd.Torch.Init.xavierW uDim xDim (seed := 0)
  let bC : Tensor Float (.dim uDim .scalar) :=
    _root_.Runtime.Autograd.Torch.Init.tensor (s := .dim uDim .scalar) (sch := .zeros) (seed := 1)
  let w1 : Tensor Float (.dim width (.dim xDim .scalar)) :=
    _root_.Runtime.Autograd.Torch.Init.xavierW width xDim (seed := 2)
  let b1 : Tensor Float (.dim width .scalar) :=
    _root_.Runtime.Autograd.Torch.Init.tensor (s := .dim width .scalar) (sch := .zeros) (seed := 3)
  let w2 : Tensor Float (.dim 1 (.dim width .scalar)) :=
    _root_.Runtime.Autograd.Torch.Init.xavierW 1 width (seed := 4)
  let b2 : Tensor Float (.dim 1 .scalar) :=
    _root_.Runtime.Autograd.Torch.Init.tensor (s := .dim 1 .scalar) (sch := .zeros) (seed := 5)
  tensorpack! wC, bC, w1, b1, w2, b2

def objectiveDef (width : Nat) : TorchLean.Module.ObjectiveDef (paramShapes width) [xShape]
  :=
  { initState := initParamsF width
    loss := Core.lossProgram width }

/-- Main entrypoint for the all-in-Lean pipeline (width is a parameter; CLI default is
  `defaultWidth`). -/
def run (width : Nat) (args : List String) : IO Unit := do
  let longRun : Bool := args.any (· = "--long")
  let twoRun : Bool := args.any (· = "--two")
  let paperRun : Bool := args.any (· = "--paper")
  let stage1Steps : Nat := (if longRun then 20 else if paperRun then 10 else if twoRun then 2 else
    1)
  let stage2Rounds : Nat := (if longRun then 10 else if paperRun then 10 else 1)
  let pgdSteps : Nat := (if longRun then 20 else if paperRun then 10 else 1)

  IO.println "== TwoStage TorchLean CEGIS workflow (IEEE32Exec) =="
  IO.println
    s!"width={width} stage1Steps={stage1Steps} stage2Rounds={stage2Rounds} pgdSteps={pgdSteps}"

  let mod ← TorchLean.Module.ObjectiveDef.instantiate (α := α) (objectiveDef width)
    IEEE32Exec.ofFloat .typedGraph
  let tr := mod.trainer
  let cLoss ← TorchLean.Autodiff.lowerScalarToTypedGraph
    (α := α) (paramShapes := paramShapes width) (inputShapes := [xShape]) (lossProg width)

  -- Stage 1: initialization pass on random x in [-rad, rad]^2
  let mut seed : UInt64 := 1
  for i in [0:stage1Steps] do
    let (seed', x) := sampleStateVector seed rad
    seed := seed'
    let xs : _root_.TorchLean.TensorPack α [xShape] := tensorpack! x
    let currentLoss := _root_.Runtime.Autograd.Torch.scalarOf (←
      _root_.Runtime.Autograd.Torch.ScalarTrainer.lossPacked tr xs .nil)
    _root_.Runtime.Autograd.Torch.ScalarTrainer.stepPacked tr lr xs .nil
    if i % 5 = 0 then
      IO.println s!"[stage1] step {i}: loss={currentLoss}"

  -- Stage 2: PGD on x to find violations, then train on them
  for round in [0:stage2Rounds] do
    let (seed', x0) := sampleStateVector seed rad
    seed := seed'
    let params ← tr.getState
    let mut x := x0
    for _k in [0:pgdSteps] do
      x := LossAnalysis.projectedGradientStep
        width cLoss params x pgdStepSize rad
    let xs : _root_.TorchLean.TensorPack α [xShape] := tensorpack! x
    let lossFound := _root_.Runtime.Autograd.Torch.scalarOf (←
      _root_.Runtime.Autograd.Torch.ScalarTrainer.lossPacked tr xs .nil)
    _root_.Runtime.Autograd.Torch.ScalarTrainer.stepPacked tr lr xs .nil
    IO.println s!"[stage2] round {round}: loss={lossFound}"

  let params ← tr.getState
  LossAnalysis.checkLossBox width params epsCheck

/-- Default hidden width used by the Pipeline III all-in-Lean workflow. -/
def defaultWidth : Nat := 100

/-- CLI entrypoint (all-in-Lean pipeline, default width). -/
def main (args : List String) : IO Unit :=
  run (width := defaultWidth) args

end NN.MLTheory.CROWN.Lyapunov.TwoStage.PipelineIII.AllInLean
