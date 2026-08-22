/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec
public import NN.API.CLI
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
public import NN.Verification.Util.Json

/-!
# Pipeline (ii): Hybrid PyTorch → TorchLean (exact float32) → IBP/CROWN post-check

This file corresponds to **Figure 7 (ii)** in the TorchLean paper (`arXiv:2602.22631`).

What is “hybrid” here:
- **Stage 1** (outside Lean): train in PyTorch using ordinary float32.
  The output is exported as *bit-exact* float32 parameters (a JSON array of uint32 bit patterns).
- **Stage 2** (inside Lean): load those exact bits into `α = IEEE32Exec` (Lean's executable model of
  IEEE-754 float32), run a small refinement loop (PGD on input, SGD on parameters), and then
  lower the TorchLean loss to the shared verifier IR and run in-repo IBP/CROWN bounds on a box.

Trust boundary:
- Stage-1 training is untrusted and only provides an initialization.
- The Stage-2 computation and the final IBP/CROWN bound propagation are *inside Lean*.

Stage-1 export script:
`python3 scripts/verification/two_stage/export_van_stage1_bits.py --width 100
  --steps 10`

Run this pipeline:
`lake exe verify -- twostage-hybrid-van-stage2`

Convenience flags for this workflow runner:
- omit the weights path to auto-use `_external/van_stage1_w{width}_bits.json`
- if the weights file is missing (or `--stage1` is passed), we run the stage-1 exporter:
  `python3 scripts/verification/two_stage/export_van_stage1_bits.py ...`
- `--stage1-steps=N` controls stage-1 SGD steps (default: 10)
- `--weights=PATH` overrides the weights JSON path
-/

@[expose] public section


open Spec
open Tensor

namespace NN.MLTheory.CROWN.Lyapunov.TwoStage.PipelineII.Hybrid

open Lean
open Lean.Data
open Lean.Json
open NN.Verification.Json

open _root_.TorchLean.Floats.IEEE754
open _root_.Runtime
open _root_.Runtime.Autograd
open NN.MLTheory.CROWN.Graph
open NN.MLTheory.CROWN

open NN.MLTheory.CROWN.Lyapunov.TwoStage.Core
open NN.MLTheory.CROWN.Lyapunov.TwoStage.ExecUtils

local notation "Scalar" => IEEE32Exec

/-- Learning rate for the stage-2 SGD loop. -/
def lr : Scalar := ExecUtils.defaultLr

/-- PGD step size when searching for counterexample-ish inputs. -/
def pgdStepSize : Scalar := ExecUtils.defaultPgdStepSize

/-- Radius of the training box `[-rad, rad]^2` (also used for clamping PGD iterates). -/
def rad : Scalar := ExecUtils.defaultRad

/-- Half-width of the small box around the origin used for the final IBP/CROWN post-check. -/
def epsCheck : Scalar := ExecUtils.defaultEpsCheck

def lossProg (width : Nat) :
    ∀ {β : Type}, [Context β] → [DecidableEq Shape] →
      TorchLean.Program β (Core.paramShapes width ++ [Core.xShape]) Shape.scalar :=
  Core.lossProgram width

/-- Convert a `Nat` JSON payload into a `UInt32`, or raise a user-facing error with context. -/
def expectU32 (ctx : String) (n : Nat) : IO UInt32 := do
  let limit : Nat := 4294967296 -- 2^32
  if _h : n < limit then
    pure (UInt32.ofNat n)
  else
    throw <| IO.userError s!"{ctx}: expected uint32 in [0,2^32), got {n}"

/-- Parse an array of float32 bit patterns (`UInt32`) encoded as nat/decimal-strings in JSON. -/
def parseBitsArray (j : Json) (ctx : String) : IO (Array UInt32) := do
  let arr ←
    match j with
    | .arr xs => pure xs
    | _ => throw <| IO.userError s!"{ctx}: expected JSON array"
  let ns ←
    match arr.mapM asNat? with
    | some ns => pure ns
    | none => throw <| IO.userError s!"{ctx}: expected nat/decimal-string array"
  ns.mapM (expectU32 ctx)

/-- Turn `UInt32` float32 bit patterns into executable float32 values (`IEEE32Exec`). -/
def bitsToScalar (bs : Array UInt32) : Array Scalar :=
  bs.map IEEE32Exec.ofBits

/-- Build a length-`n` vector tensor from an array (with a length check). -/
def mkVec (n : Nat) (xs : Array Scalar) : IO (Tensor Scalar (.dim n .scalar)) := do
  if xs.size != n then
    throw <| IO.userError s!"expected length {n}, got {xs.size}"
  pure <| Tensor.dim (n := n) (s := .scalar) (fun i => Tensor.scalar xs[i.val]!)

/-- Build an `m×n` matrix tensor from a flat array (row-major, with a length check). -/
def mkMat (m n : Nat) (xs : Array Scalar) : IO (Tensor Scalar (.dim m (.dim n .scalar))) := do
  let expected := m * n
  if xs.size != expected then
    throw <| IO.userError s!"expected length {expected} (matrix {m}x{n}), got {xs.size}"
  pure <|
    Tensor.dim (n := m) (s := .dim n .scalar) (fun i =>
      Tensor.dim (n := n) (s := .scalar) (fun j =>
        let idx := i.val * n + j.val
        Tensor.scalar xs[idx]!))

/--
Load stage-1 parameters exported by PyTorch as *float32 bit patterns*.

We do this (instead of parsing JSON floats) so stage-2 runs under *bit-exact* float32 semantics
(`IEEE32Exec`) without decimal conversion error.
-/
def loadFirstStageParams (width : Nat) (path : String) : IO (_root_.TorchLean.TensorPack Scalar (Core.paramShapes
  width)) := do
  let jsonStr ← IO.FS.readFile path
  let j ← match Json.parse jsonStr with
    | .ok j => pure j
    | .error e => throw <| IO.userError s!"Bad JSON: {e}"
  let top ← expectObj j "top-level"

  let wJ ← expectField top "width" "top-level"
  let some w := asNat? wJ | throw <| IO.userError "top-level.width must be nat/decimal-string"
  if w != width then
    throw <| IO.userError s!"width mismatch: file has {w}, workflow expects {width}"

  let wCBits ← parseBitsArray (← expectField top "wC" "top-level") "wC"
  let bCBits ← parseBitsArray (← expectField top "bC" "top-level") "bC"
  let w1Bits ← parseBitsArray (← expectField top "w1" "top-level") "w1"
  let b1Bits ← parseBitsArray (← expectField top "b1" "top-level") "b1"
  let w2Bits ← parseBitsArray (← expectField top "w2" "top-level") "w2"
  let b2Bits ← parseBitsArray (← expectField top "b2" "top-level") "b2"

  let wC ← mkMat Core.uDim Core.xDim (bitsToScalar wCBits)
  let bC ← mkVec Core.uDim (bitsToScalar bCBits)
  let w1 ← mkMat width Core.xDim (bitsToScalar w1Bits)
  let b1 ← mkVec width (bitsToScalar b1Bits)
  let w2 ← mkMat 1 width (bitsToScalar w2Bits)
  let b2 ← mkVec 1 (bitsToScalar b2Bits)

  pure <| TensorPack! wC, bC, w1, b1, w2, b2

/-- Parsed command options for the hybrid two-stage runner. -/
structure HybridCliOptions where
  weightsPath : String
  forceStage1 : Bool
  stage1Steps : Nat
  longRun : Bool
  paperRun : Bool
  candidates : Nat
deriving Repr

/-- Parse all CLI flags once, so the runner and stage-1 bootstrap cannot disagree. -/
def parseHybridCliOptions (width : Nat) (args : List String) : IO HybridCliOptions := do
  let defaultPath : String := s!"_external/van_stage1_w{width}_bits.json"
  let args := TorchLean.CLI.dropDashDash args
  let (weightsFlag?, args) ← TorchLean.CLI.orThrowIO <| TorchLean.CLI.takeFlagValueOnce args "weights"
  let (positionalWeights, args) ← TorchLean.CLI.orThrowIO <|
    TorchLean.CLI.takePositionalDefault args defaultPath
  let (forceStage1, args) ← TorchLean.CLI.orThrowIO <| TorchLean.CLI.takeBoolFlagOnce args "stage1"
  let (stage1Steps, args) ← TorchLean.CLI.orThrowIO <|
    TorchLean.CLI.takeNatFlagDefault args "stage1-steps" 10
  let (longRun, args) ← TorchLean.CLI.orThrowIO <| TorchLean.CLI.takeBoolFlagOnce args "long"
  let (paperRun, args) ← TorchLean.CLI.orThrowIO <| TorchLean.CLI.takeBoolFlagOnce args "paper"
  let (candidates, args) ← TorchLean.CLI.orThrowIO <| TorchLean.CLI.takeNatFlagDefault args "candidates" 1
  TorchLean.CLI.orThrowIO <| TorchLean.CLI.checkNoArgs args
  pure
    { weightsPath := weightsFlag?.getD positionalWeights
      forceStage1 := forceStage1
      stage1Steps := stage1Steps
      longRun := longRun
      paperRun := paperRun
      candidates := candidates }

/--
Run the external PyTorch Stage-1 exporter (if needed) and return the JSON path.

This is the only place pipeline (ii) depends on Python. The trust boundary is still clean:
- Stage 1 provides an **initialization only** (untrusted),
- Stage 2 and the IBP/CROWN post-check run inside Lean under exact `IEEE32Exec` semantics.
-/
def ensureFirstStageWeights (width : Nat) (opts : HybridCliOptions) : IO String := do
  let weightsPath := opts.weightsPath
  let weightsExists := (← System.FilePath.pathExists (System.FilePath.mk weightsPath))
  if weightsExists && !opts.forceStage1 then
    return weightsPath

  IO.println s!"[stage1] running PyTorch exporter (width={width} steps={opts.stage1Steps}) → {weightsPath}"
  let script : String :=
    "scripts/verification/two_stage/export_van_stage1_bits.py"
  let proc := (← IO.Process.spawn
    { cmd := "python3"
      args := #[script, "--width", toString width, "--steps", toString opts.stage1Steps,
        "--out", weightsPath]
      stdout := .inherit
      stderr := .inherit })
  let code := (← proc.wait)
  if code != 0 then
    throw <| IO.userError s!"Stage-1 exporter failed with exit code {code}"
  let weightsExists' := (← System.FilePath.pathExists (System.FilePath.mk weightsPath))
  if !weightsExists' then
    throw <| IO.userError s!"Stage-1 exporter did not produce expected file: {weightsPath}"
  return weightsPath

/-- Main entrypoint for the hybrid pipeline (width is a parameter; CLI default is `defaultWidth`).
  -/
def run (width : Nat) (args : List String) : IO Unit := do
  let opts ← parseHybridCliOptions width args
  let weightsPath ← ensureFirstStageWeights width opts

  let stage2Rounds : Nat := (if opts.longRun then 10 else if opts.paperRun then 10 else 1)
  let pgdSteps : Nat := (if opts.longRun then 20 else if opts.paperRun then 10 else 1)

  IO.println "== TwoStage Hybrid workflow: Stage1=PyTorch (bits), Stage2=TorchLean (IEEE32Exec) =="
  IO.println
    (s!"weights={weightsPath} width={width} stage2Rounds={stage2Rounds} " ++
      s!"candidates={opts.candidates} pgdSteps={pgdSteps}")

  let initParams ← loadFirstStageParams width weightsPath
  let mod ← _root_.Runtime.Autograd.TorchLean.Module.Objective.create
    (α := Scalar) (stateShapes := Core.paramShapes width) (inputShapes := [Core.xShape])
    (natInputShapes := [])
    (opts := { execution := .typedGraph })
    (requiresGrad := List.replicate (Core.paramShapes width).length true)
    (loss := lossProg width (β := Scalar))
    (initState := initParams)
  let tr := _root_.Runtime.Autograd.TorchLean.Module.Objective.trainer mod

  let cLoss ← TorchLean.Autodiff.lowerScalarToTypedGraph
    (α := Scalar) (paramShapes := Core.paramShapes width) (inputShapes := [Core.xShape])
      (lossProg width)

  -- Stage 2: PGD on x to find violations, then train on them
  let mut seed : UInt64 := 1
  let mut foundViolations : Nat := 0
  for round in [0:stage2Rounds] do
    for _ci in [0:opts.candidates] do
      let (seed', x0) := sampleStateVector seed rad
      seed := seed'
      let lossBeforePgd := (←
        _root_.Runtime.Autograd.Torch.ScalarTrainer.lossPacked tr (.cons x0 .nil) .nil).item
      let params ← tr.getState
      let mut x := x0
      for _k in [0:pgdSteps] do
        x := LossAnalysis.projectedGradientStep
          width cLoss params x pgdStepSize rad
      let xs : _root_.TorchLean.TensorPack Scalar [Core.xShape] := TensorPack! x
      let lossFound := (←
        _root_.Runtime.Autograd.Torch.ScalarTrainer.lossPacked tr xs .nil).item
      if (0 : Scalar) < lossFound then
        foundViolations := foundViolations + 1
      _root_.Runtime.Autograd.Torch.ScalarTrainer.stepPacked tr lr xs .nil
      IO.println s!"[stage2] round {round}: lossBefore={lossBeforePgd} lossAfterPGD={lossFound}"

  let params ← tr.getState
  IO.println
    (s!"[stage2] PGD counterexample candidates={stage2Rounds * opts.candidates} " ++
      s!"(positive-loss={foundViolations})")
  LossAnalysis.checkLossBox width params epsCheck

/-- Default hidden width used by the hybrid workflow. -/
def defaultWidth : Nat := 500

/-- CLI entrypoint (hybrid pipeline, default width). -/
def main (args : List String) : IO Unit :=
  run (width := defaultWidth) args

end NN.MLTheory.CROWN.Lyapunov.TwoStage.PipelineII.Hybrid
