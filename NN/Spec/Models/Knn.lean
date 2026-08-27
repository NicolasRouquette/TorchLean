/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Numerics
public import Std.Data.TreeMap.Basic

/-!
# k‑Nearest Neighbors (kNN) (spec model)

This file provides a small kNN classifier/regressor baseline:

- a dataset is an array of `(featureTensor, label)` pairs,
- prediction is based on the `k` closest points under a chosen distance.

We include deterministic tie‑breaking so results are stable (useful for regression tests and
formal reasoning).

References:

- Cover and Hart (1967), "Nearest Neighbor Pattern Classification":
  https://ieeexplore.ieee.org/document/1053964

PyTorch / sklearn analogies:

- In the Python ecosystem this is closest to `sklearn.neighbors.KNeighborsClassifier` /
  `sklearn.neighbors.KNeighborsRegressor`.
- kNN is not typically an `nn.Module` in PyTorch, but it is a common baseline for "classic ML"
  comparisons and reference checks.
-/

public section


namespace Spec
open Tensor

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-! ## Model container -/

/-- A small kNN model container (parameters + stored dataset).

This is a *lazy* model: inference consults the stored `dataset` at query time, rather than learning
weights.
-/
structure KNN (α : Type) (β : Type) (n : Nat) where
  /-- Number of neighbors to consult. -/
  k : Nat
  /-- Training data: feature vectors paired with labels/targets. -/
  dataset : Array (Tensor α [n] × β)

/-! ## Neighbor selection -/

/-!
The key technical detail here is deterministic tie-breaking: when two points are at exactly the
same distance, we prefer the earlier point in the dataset. This makes evaluation stable and keeps
formal reasoning about the classifier simpler.
-/

/-- Sort dataset points by distance to `input`, breaking ties by dataset order. -/
private def stableSortByDistance {β : Type} {n : Nat}
  (distanceFn : Tensor α [n] → Tensor α [n] → α)
  (input : Tensor α [n])
  (dataset : Array (Tensor α [n] × β)) :
  Array (Tensor α [n] × β) :=
  -- Deterministic tie-breaking: for equal distances, prefer earlier dataset points.
  let withDistances := dataset.mapIdx (fun idx point =>
      let (features, label) := point
      let dist := distanceFn input features
      ((features, label), dist, idx))
  let sorted :=
    withDistances.qsort (fun a b =>
      let da := a.2.1
      let db := b.2.1
      let ia := a.2.2
      let ib := b.2.2
      da < db ∨ (¬ (db < da) ∧ ia < ib))
  sorted.map (fun triple => triple.1)

/-- Find the `k` nearest neighbors under Euclidean distance.

PyTorch/sklearn analogy: Euclidean `L2` distance is the default for many baseline kNN examples.
-/
def findKNearest (α β : Type) (n : ℕ)
  [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  (knn : KNN α β n) (input : Tensor α [n]) :
  Array (Tensor α [n] × β) :=
  if knn.dataset.isEmpty then #[]
  else
    let sorted := stableSortByDistance (α := α) (β := β) (n := n)
      (distanceFn := euclideanDistanceSpec) input knn.dataset
    sorted.extract 0 (min knn.k sorted.size)

/-- Find the `k` nearest neighbors under a user-provided distance function.

Notes:
- The distance value is only used for *ranking* neighbors. It does not need to satisfy metric
  axioms, but it should be consistent with "smaller means closer".
- We keep deterministic tie-breaking via dataset order.
-/
def findKNearestWithDistance (α β : Type) (n : ℕ)
  [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  (distanceFn : Tensor α [n] → Tensor α [n] → α)
  (knn : KNN α β n) (input : Tensor α [n]) :
  Array (Tensor α [n] × β) :=
  if knn.dataset.isEmpty then #[]
  else
    let sorted := stableSortByDistance (α := α) (β := β) (n := n)
      (distanceFn := distanceFn) input knn.dataset
    sorted.extract 0 (min knn.k sorted.size)

/-! ## Classification -/

/-- Count label frequencies in an array (hash-map implementation). -/
private def voteCounts {β : Type} [BEq β] [Hashable β] (labels : Array β) : Std.HashMap β Nat :=
  labels.foldl (fun acc label =>
    let currentCount := acc[label]? |>.getD 0
    acc.insert label (currentCount + 1)
  ) Std.HashMap.emptyWithCapacity

/-- Majority vote among the neighbors.

Tie-breaking: if multiple labels have the same maximum count, we prefer the label of the closest
neighbor. This is deterministic and matches common "stable mode" implementations.
-/
def classify (α β : Type) (n : ℕ)
  [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [BEq β] [Hashable β] [Inhabited β]
  (knn : KNN α β n) (input : Tensor α [n]) : β :=
  let neighbors := findKNearest α β n knn input
  if neighbors.isEmpty then default
  else
    let labels := neighbors.map (fun (_, label) => label)
    let labelCounts : Std.HashMap β Nat := voteCounts (β := β) labels
    -- Choose the most frequent label; for ties, take the closest neighbor's label.
    let entries := labelCounts.toArray
    match entries[0]? with
    | none => default
    | some first =>
      let maxCount := entries.foldl (fun best cur => max best cur.snd) first.snd
      let nearestLabel := neighbors[0]!.2
      let tied := entries.filter (fun e => e.snd = maxCount) |>.map (·.fst)
      if tied.any (fun l => l == nearestLabel) then nearestLabel
      else tied[0]?.getD default

/-- Classification using an ordered tree map for label counts.

This variant avoids hashing, at the cost of requiring an `Ord β` instance.
-/
def classifyTreeMap (α β : Type) (n : ℕ)
  [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [Ord β] [Inhabited β]
  (knn : KNN α β n) (input : Tensor α [n]) : Option β :=
  let neighbors := findKNearest α β n knn input
  let labels := neighbors.map (fun (_, label) => label)
  if labels.isEmpty then none
  else
    let labelCounts := labels.foldl (fun (acc : Std.TreeMap β Nat compare) label =>
      acc.insert label (acc.getD label 0 + 1)
    ) Std.TreeMap.empty
    let grouped := labelCounts.toArray
    some (grouped.foldl
      (fun best cur => if cur.snd > best.snd then cur else best)
      (labels[0]!, 0)
    ).fst

/-! ## Regression -/

/-- Unweighted kNN regression: average of the neighbor targets. -/
def predict (α : Type) (n : ℕ)
  [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  (knn : KNN α α n) (input : Tensor α [n]) : α :=
  let neighbors := findKNearest α α n knn input
  if neighbors.isEmpty then 0
  else
    let values := neighbors.map (fun (_, value) => value)
    (values.foldl (· + ·) 0) / (values.size : α)

/-- Weighted kNN regression using inverse-distance weights.

We use weights `w_i = 1 / d(x, x_i)`. If a neighbor is exactly at distance `0` we give it a large
weight so it dominates the average.

This is a common heuristic and matches what many "classic ML" baselines do in practice.
-/
def predictWeighted (α : Type) (n : ℕ)
  [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  (knn : KNN α α n) (input : Tensor α [n]) : α :=
  if knn.dataset.isEmpty then 0
  else
    let neighborsWithDist := knn.dataset.map (fun (features, value) =>
      let dist := euclideanDistanceSpec input features
      (value, dist))
    let sorted := neighborsWithDist.qsort (fun a b => a.2 < b.2)
    let kNearest := sorted.extract 0 (min knn.k sorted.size)
    let thousand := Numbers.ten ^ (Numbers.three : α)
    if kNearest.isEmpty then 0
    else
      let weightedSum := kNearest.foldl (fun acc (value, dist) =>
        let weight := if dist == 0 then thousand else 1 / dist
        acc + weight * value
      ) 0
      let totalWeight := kNearest.foldl (fun acc (_, dist) =>
        let weight := if dist == 0 then thousand else 1 / dist
        acc + weight
      ) 0
      if totalWeight == 0 then 0 else weightedSum / totalWeight

/-! ## Helpers -/

/-- Constructor helper (explicit arguments keep elaboration simple in examples). -/
def KNN.fromData (α β : Type) (n : ℕ) (k : Nat)
    (data : Array (Tensor α [n] × β)) : KNN α β n :=
  { k := k, dataset := data }

/-- Batch regression: map `predict` over an array of inputs. -/
def batchPredict (α : Type) (n : ℕ)
  [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  (knn : KNN α α n) (inputs : Array (Tensor α [n])) : Array α :=
  inputs.map (predict α n knn)

/-- Batch classification: map `classify` over an array of inputs. -/
def batchClassify (α β : Type) (n : ℕ)
  [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [Hashable β] [Inhabited β] [BEq β]
  (knn : KNN α β n) (inputs : Array (Tensor α [n])) : Array β :=
  inputs.map (classify α β n knn)

/-- kNN classification using an explicit distance function (for non-Euclidean metrics). -/
def classifyWithDistance (α β : Type) (n : ℕ)
  [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [BEq β] [Hashable β] [Inhabited β]
  (distanceFn : Tensor α [n] → Tensor α [n] → α)
  (knn : KNN α β n) (input : Tensor α [n]) : β :=
  let neighbors := findKNearestWithDistance α β n distanceFn knn input
  if neighbors.isEmpty then default
  else
    let labels := neighbors.map (fun (_, label) => label)
    let labelCounts : Std.HashMap β Nat := voteCounts (β := β) labels
    let entries := labelCounts.toArray
    match entries[0]? with
    | none => default
    | some first =>
      let maxEntry := entries.foldl (fun best cur =>
        if cur.snd > best.snd then cur else best
      ) first
      maxEntry.fst

/-- kNN classification plus a simple confidence score (`maxVotes / k`).

This returns the winning label together with a scalar in `[0,1]` that measures what fraction of the
neighbors agreed with the winner. This is a heuristic, but it is useful for examples and baselines.
-/
def classifyWithConfidence (α β : Type) (n : ℕ)
  [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [BEq β] [Hashable β] [Inhabited β]
  (knn : KNN α β n) (input : Tensor α [n]) : (β × α) :=
  let neighbors := findKNearest α β n knn input
  if neighbors.isEmpty then (default, 0)
  else
    let labels := neighbors.map (fun (_, label) => label)
    let labelCounts : Std.HashMap β Nat := voteCounts (β := β) labels
    let entries := labelCounts.toArray
    match entries[0]? with
    | none => (default, 0)
    | some first =>
      let maxEntry := entries.foldl (fun best cur =>
        if cur.snd > best.snd then cur else best
      ) first
      let confidence := (maxEntry.snd : α) / (neighbors.size : α)
      (maxEntry.fst, confidence)

end Spec
