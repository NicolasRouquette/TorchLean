/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Train.Core
public import NN.Data.SampleStream
public import NN.Runtime.Autograd.Train.Trainer

/-!
# Evaluation helpers

These utilities aggregate per-sample or per-batch `Report`s into a single
mean report. Metrics are matched by name and position.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Train
namespace Eval

/-!
## Metric aggregation
-/
/--
Add two metric arrays pointwise.

Names must match, so unrelated quantities are not silently averaged.
-/
def addMetrics {a : Type} [Add a]
  (tag : String) (xs ys : Array (Metric a)) : Result (Array (Metric a)) := do
  if hsize : xs.size = ys.size then
    let mut result := Array.emptyWithCapacity xs.size
    for h : i in [:xs.size] do
      let left := xs[i]
      let right := ys[i]'(hsize ▸ h.2.1)
      if left.name != right.name then
        throw (tagError tag s!"metric name mismatch: {left.name} vs {right.name}")
      result := result.push { name := left.name, value := left.value + right.value }
    pure result
  else
    throw (tagError tag "metric length mismatch")

/-!
## Report sums (for weighted aggregation)
-/
/--
An accumulator for averaging `Report`s.

Instead of retaining every report and reducing at the end, we maintain:
- `count`: how many samples contributed,
- `lossSum`: the sum of losses (optionally weighted by batch size),
- `metricsSum`: a pointwise sum of named metrics.

This is the same idea as computing streaming averages in a typical PyTorch evaluation loop.
-/
structure ReportSum (a : Type) where
  /-- Number of samples represented by this accumulator. -/
  count : Nat
  /-- Sum of losses, already weighted by sample count for batch reports. -/
  lossSum : a
  /-- Pointwise sum of metrics; names must stay aligned across additions. -/
  metricsSum : Array (Metric a)

namespace ReportSum

/-- Start an accumulator from a single-sample report. -/
def ofReport {a : Type} (r : Report a) : ReportSum a :=
  { count := 1, lossSum := r.loss, metricsSum := r.metrics }

/-- Combine two accumulators (failing if metric names/lengths mismatch). -/
def add {a : Type} [Add a]
  (tag : String) (acc next : ReportSum a) : Result (ReportSum a) := do
  let metrics ← addMetrics (tag := tag) acc.metricsSum next.metricsSum
  pure { count := acc.count + next.count
         lossSum := acc.lossSum + next.lossSum
         metricsSum := metrics }

/-- Convert an accumulator to a mean `Report`. -/
def mean {a : Type} [Div a] [NatCast a] (s : ReportSum a) : Report a :=
  let denom : a := (s.count : a)
  { loss := s.lossSum / denom
    metrics := s.metricsSum.map (fun m => { name := m.name, value := m.value / denom }) }

end ReportSum

/-!
## Sample-stream evaluation
-/
/--
Evaluate an array of samples and average their reports.

Samples are evaluated in index order. Pairwise summation avoids losing every subsequent unit
contribution once a narrow floating-point accumulator can no longer represent `sum + 1`.
It does not guarantee an accurate mean for every input, or prevent overflow of the sum.
-/
def evalArray {sample a : Type}
  [Add a] [Div a] [NatCast a]
  (tag : String) (xs : Array sample) (evalSample : sample -> Result (Report a)) :
  Result (Report a) := do
  if xs.isEmpty then throw (tagError tag "empty dataset")
  let rec reduce (fuel start count : Nat) : Result (ReportSum a) := do
    match fuel with
    | 0 => throw (tagError tag "invalid evaluation range")
    | fuel + 1 =>
      if count = 1 then
        let some sample := xs[start]?
          | throw (tagError tag "evaluation index out of bounds")
        return ReportSum.ofReport (← evalSample sample)
      let leftCount := count / 2
      let left ← reduce fuel start leftCount
      let right ← reduce fuel (start + leftCount) (count - leftCount)
      ReportSum.add tag left right
  return ReportSum.mean (← reduce xs.size 0 xs.size)

/-- Evaluate a finite sample stream in index order. -/
def evalDataset {sample a : Type}
  [Add a] [Div a] [NatCast a]
  (tag : String) (ds : TorchLean.Data.SampleStream sample)
  (evalSample : sample -> Result (Report a)) :
  Result (Report a) :=
  evalArray (tag := tag) ds.toArray evalSample

end Eval
end Train
end Autograd
end Runtime
