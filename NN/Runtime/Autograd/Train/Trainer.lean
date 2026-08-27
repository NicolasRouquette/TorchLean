/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Train.Core

/-!
# Trainer API with metrics and logging

This module defines a small, higher-level training API on top of a step function.
It stays local (no global state), while making it easy to:

* return a structured report per step (loss + metrics)
* render reports into readable logs
* plug in a logger if you want to print during training
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Train

/-!
## Metrics and step reports
-/

/--
A named scalar metric for logging/monitoring.

Examples: `"acc"`, `"top5"`, `"grad_norm"`, `"lr"`.
-/
structure Metric (a : Type) where
  /-- Metric name (used as the key in logs). -/
  name : String
  /-- Metric value (typically the same scalar type as the loss). -/
  value : a

/-- Render a metric as `name=value`. -/
def Metric.render {a : Type} [ToString a] (m : Metric a) : String :=
  s!"{m.name}={m.value}"

/--
Per-step training report.

This is a compact record: a single scalar loss (to drive optimization) plus optional metrics
for logging/monitoring.
-/
structure StepReport (a : Type) where
  /-- Scalar objective value for this step or evaluation pass. -/
  loss : a
  /-- Additional named scalar metrics for logging/monitoring. -/
  metrics : Array (Metric a) := #[]

/-- Render an array of metrics as a comma-separated string. -/
def renderMetrics {a : Type} [ToString a] (metrics : Array (Metric a)) : String :=
  String.intercalate ", " (metrics.map Metric.render).toList

/-- Render a single step report (loss + metrics). -/
def renderReport {a : Type} [ToString a] (step : Nat) (r : StepReport a) : String :=
  let base := s!"step {step}: loss={r.loss}"
  if r.metrics.isEmpty then
    base
  else
    base ++ ", " ++ renderMetrics r.metrics

/-- Render reports for a full run, with step numbers starting at `0`. -/
def renderReports {a : Type} [ToString a] (reports : Array (StepReport a)) : Array String :=
  reports.mapIdx fun step report => renderReport step report

/-!
## Generic training loop

This is a light wrapper around a "step" function that returns a new state and an output.
It is still useful for very small tests that do not need full metrics.
-/
/--
Run a monadic step function for a fixed number of steps, collecting the per-step outputs.

This is a generic utility (not Torch-specific): it threads a `state` value and accumulates an
`out` value per step.
-/
def runStepsM {m : Type -> Type} [Monad m] {state out : Type}
  (steps : Nat) (init : state) (step : state -> m (Prod state out)) :
  m (Prod state (Array out)) := by
  let rec go : Nat -> state -> Array out -> m (Prod state (Array out))
    | 0, s, outputs => pure (s, outputs)
    | n + 1, s, outputs => do
        let (s', out) ← step s
        go n s' (outputs.push out)
  exact go steps init #[]

/-!
## Trainer structure

`Trainer` bundles the initial state, step function, and optional logger.
The logger runs *after* each step and can observe the updated state and report.
-/
/--
A small "trainer bundle": initial state, step function, and a per-step logger.

The logger runs after each step and can observe both the updated state and the report, which
matches how training scripts typically print "after-update" metrics.
-/
structure Trainer (m : Type -> Type) (state : Type) (a : Type) where
  /-- Initial training state. -/
  init : state
  /-- A single training step: update state and produce a report. -/
  step : state -> m (Prod state (StepReport a))
  /-- Optional logger hook called after each step. -/
  logger : Nat -> state -> StepReport a -> m Unit

namespace Trainer

/-- Construct a `Trainer` with a no-op logger and collected step reports. -/
def noLog {m : Type -> Type} [Monad m] {state a : Type}
  (init : state) (step : state -> m (Prod state (StepReport a))) :
  Trainer m state a :=
  { init := init
    step := step
    logger := fun _ _ _ => pure () }

/-- Run a trainer for `steps` steps, returning the final state and the collected reports. -/
def run {m : Type -> Type} [Monad m] {state a : Type}
  (steps : Nat) (t : Trainer m state a) :
  m (Prod state (Array (StepReport a))) := by
  let rec go : Nat -> Nat -> state -> Array (StepReport a) ->
      m (Prod state (Array (StepReport a)))
    | 0, _, s, reports => pure (s, reports)
    | n + 1, stepIdx, s, reports => do
        let (s', report) ← t.step s
        t.logger stepIdx s' report
        go n (stepIdx + 1) s' (reports.push report)
  exact go steps 0 t.init #[]

/-- Run a trainer and project the report stream to per-step losses. -/
def runLosses {m : Type -> Type} [Monad m] {state a : Type}
  (steps : Nat) (t : Trainer m state a) :
  m (Prod state (Array a)) := do
  let (s, reports) ← run (steps := steps) t
  pure (s, reports.map (fun r => r.loss))

end Trainer

end Train
end Autograd
end Runtime
