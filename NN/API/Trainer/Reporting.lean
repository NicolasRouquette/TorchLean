/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Training.Log
public import NN.Runtime.Autograd.Train.Dataset

@[expose] public section

/-!
# Training Reports

Training curves, metric histories, and logs produced by `Trainer` runs.
-/

namespace TorchLean.Training

export _root_.Runtime.Training
  (Curve TrainLog ExperimentLog LogDestination MetricHistory)
export _root_.Runtime.Autograd.Train (Dataset DataLoader)

namespace MetricHistory

export _root_.Runtime.Training.MetricHistory (empty)

end MetricHistory

/-- Return whether a completed step should emit a periodic report. -/
def shouldReport (every completed : Nat) : Bool :=
  every != 0 && completed % every = 0

/-- Write a training log as JSON and report the output path. -/
def writeLog (path : System.FilePath) (log : TrainLog) : IO Unit := do
  _root_.Runtime.Training.TrainLog.writeJson path log
  IO.println s!"  wrote TrainLog JSON: {path}"

/-- Write a training log to an enabled or disabled destination. -/
def writeLogTo (destination : LogDestination) (log : TrainLog) : IO Unit := do
  _root_.Runtime.Training.LogDestination.writeTrainLog destination log
  match destination.path? with
  | some path => IO.println s!"  wrote TrainLog JSON: {path}"
  | none => IO.println "  TrainLog JSON disabled"

/-- Write a log containing the loss before and after a fixed number of training steps. -/
def writeLossComparison (path : System.FilePath) (title : String) (steps : Nat)
    (beforeLoss afterLoss : Float) (notes : Array String := #[]) : IO Unit :=
  writeLog path (_root_.Runtime.Training.TrainLog.beforeAfterLoss
    title steps beforeLoss afterLoss notes)

/-- Write a before-and-after loss log to an enabled or disabled destination. -/
def writeLossComparisonTo (destination : LogDestination) (title : String) (steps : Nat)
    (beforeLoss afterLoss : Float) (notes : Array String := #[]) : IO Unit :=
  writeLogTo destination (_root_.Runtime.Training.TrainLog.beforeAfterLoss
    title steps beforeLoss afterLoss notes)

/-- First and last values of a nonempty scalar training curve. -/
structure CurveEndpoints where
  /-- Step associated with the final value. -/
  finalStep : Nat
  /-- First recorded value. -/
  first : Float
  /-- Last recorded value. -/
  last : Float
deriving Repr

/-- Return the endpoints of a scalar curve, or `none` when it contains no values. -/
def Curve.endpoints? (curve : Curve) : Option CurveEndpoints := do
  let first ← curve.values[0]?
  let last ← curve.values.back?
  let finalStep := curve.steps.back?.getD (curve.values.size - 1)
  pure { finalStep, first, last }

/-- Return the endpoints of a scalar curve or raise a contextual error when it is empty. -/
def Curve.endpoints (curve : Curve) (context : String := "training curve") : IO CurveEndpoints :=
  match TorchLean.Training.Curve.endpoints? curve with
  | some endpoints => pure endpoints
  | none => throw <| IO.userError s!"{context}: empty training curve"

/-- Print the first and last loss from a scalar training curve. -/
def Curve.printLossSummary (curve : Curve) (steps : Nat) : IO Unit := do
  let endpoints ← TorchLean.Training.Curve.endpoints curve "Curve.printLossSummary"
  let finalStep := if curve.steps.isEmpty then steps else endpoints.finalStep
  IO.println s!"  steps={finalStep} loss_before={endpoints.first} loss_after={endpoints.last}"

/-- Write a scalar curve as a one-series training log. -/
def Curve.writeLog (curve : Curve) (path : System.FilePath) (title : String)
    (seriesName : String := "loss") (notes : Array String := #[]) : IO Unit :=
  TorchLean.Training.writeLog path
    (_root_.Runtime.Training.Curve.toTrainLog curve title seriesName (notes := notes))

/-- Write a scalar curve to an enabled or disabled training-log destination. -/
def Curve.writeLogTo (curve : Curve) (destination : LogDestination) (title : String)
    (seriesName : String := "loss") (notes : Array String := #[]) : IO Unit :=
  TorchLean.Training.writeLogTo destination
    (_root_.Runtime.Training.Curve.toTrainLog curve title seriesName (notes := notes))

end TorchLean.Training
