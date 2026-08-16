/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Neural
public import NN.API.Trainer.Reporting

/-!
# Training Summaries

Small report types returned by the high-level trainer API.
-/

@[expose] public section

namespace TorchLean

namespace Trainer

/--
Backend-independent before/after training summary.

Losses are rendered as strings because the runtime scalar type is chosen inside the training run.
-/
structure TrainSummary where
  /-- Metric name, usually `loss`. -/
  metric : String := "loss"
  /-- Number of optimizer steps requested by the configuration. -/
  steps : Nat
  /-- Metric before training. -/
  before : String
  /-- Metric after training. -/
  after : String

namespace TrainSummary

/-- One-line summary suitable for quickstarts and scripts. -/
def summary (report : TrainSummary) : String :=
  s!"steps={report.steps} {report.metric}0={report.before} {report.metric}1={report.after}"

/-- Print the one-line before/after summary. -/
def printSummary (report : TrainSummary) : IO Unit :=
  IO.println (summary report)

instance : ToString TrainSummary where
  toString := summary

/-- Parse a `ToString`-rendered scalar as a JSON number when possible. -/
def parseFloat? (s : String) : Option Float :=
  match _root_.Lean.Json.parse s with
  | .ok (.num n) => some n.toFloat
  | _ => none

/-- Convert a before/after summary into the standard two-point TrainLog when values are finite. -/
def toTrainLog? (title : String) (notes : Array String) (report : TrainSummary) :
    Option Training.TrainLog := do
  let before ← parseFloat? report.before
  let after ← parseFloat? report.after
  some <| _root_.Runtime.Training.TrainLog.beforeAfterLoss
    title report.steps before after notes

/--
Read the before/after metrics back as ordinary `Float`s.

Most scripts call `trained.printSummary`. Examples that write JSON logs can use this operation when they
need the same metrics as `Float`s.
-/
def requireFloatLosses (context : String) (report : TrainSummary) : IO (Float × Float) := do
  let before ←
    match parseFloat? report.before with
    | some value => pure value
    | none =>
        throw <| IO.userError
          s!"{context}: non-numeric initial {report.metric} {report.before}"
  let after ←
    match parseFloat? report.after with
    | some value => pure value
    | none =>
      throw <| IO.userError
          s!"{context}: non-numeric final {report.metric} {report.after}"
  pure (before, after)

/-- Print numeric before/after losses and return them for artifact writers, or report an error. -/
def printFloatLosses (context : String) (report : TrainSummary)
    (steps? : Option Nat := none) (lr? : Option Float := none) : IO (Float × Float) := do
  let (before, after) ← requireFloatLosses context report
  let stepsPart :=
    match steps? with
    | some steps => s!"steps={steps} "
    | none => ""
  let lrPart :=
    match lr? with
    | some lr => s!"lr={lr} "
    | none => ""
  IO.println s!"  {stepsPart}{lrPart}{report.metric}0={before} {report.metric}1={after}"
  pure (before, after)

/-- Write this summary to a log destination when logging is enabled. -/
def writeLog (dest : Training.LogDestination) (title : String) (notes : Array String)
    (report : TrainSummary) : IO Unit := do
  if dest.isEnabled then
    match report.toTrainLog? title notes with
    | some log => Training.writeLogTo dest log
    | none =>
        throw <| IO.userError
          s!"Trainer.TrainSummary.writeLog: cannot write TrainLog because {report.metric} values are not JSON numbers: {report.before}, {report.after}"

end TrainSummary

end Trainer

end TorchLean
