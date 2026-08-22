/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.List.Defs
public import NN.Spec.Core.TensorReductionShape.LinearAlgebra
public import NN.Spec.Core.TensorReductionShape.Reductions
public import NN.Spec.Core.Tensor.API

/-!
# Metrics

TorchLean metrics helpers.

These are non-differentiable evaluation helpers for classification and accuracy reports.
-/

@[expose] public section


namespace TorchLean

open Spec
open Tensor

namespace Metrics

/-- Index of the maximum entry in a length-`n` vector, if `n > 0`.

This small, nondifferentiable evaluation helper is written against `Tensor` so it can be used with
multiple scalar types.
-/
def argmaxVector? {α : Type} [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    {n : Nat} (y : Tensor α (.dim n .scalar)) : Option (Fin n) :=
  match y with
  | Tensor.dim f =>
      if h0 : 0 < n then
        let init : Fin n := ⟨0, h0⟩
        let (bestIdx, _bestVal) := (List.finRange n).foldl (fun (acc : Fin n × α) k =>
          let (_, bestVal) := acc
          let vk := match f k with | Tensor.scalar v => v
          if vk > bestVal then (k, vk) else acc
        ) (init, match f init with | Tensor.scalar v => v)
        some bestIdx
      else
        none

/-- Compare predicted `argmax` against a one-hot target; returns `none` when `n = 0`. -/
def correctOneHotVector? {α : Type} [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    {n : Nat} (logits : Tensor α (.dim n .scalar)) (targetOneHot : Tensor α (.dim n .scalar)) :
    Option Bool := do
  let p ← argmaxVector? (α := α) (n := n) logits
  let y ← argmaxVector? (α := α) (n := n) targetOneHot
  pure (p = y)

/--
Indices of the maxima along `axis`, in row-major order over all remaining dimensions.

The selected axis is moved to the innermost position before the tensor is flattened, so each
contiguous chunk is one class slice. An empty class axis contributes `none` for every slice of the
remaining shape. Ties are resolved in favor of the first index, matching `argmaxVector?`.
-/
def argmaxAxis? {α : Type} [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s] (values : Tensor α s) :
    List (Option Nat) :=
  let argmaxChunk (xs : List α) : Option Nat :=
    match xs with
    | [] => none
    | x :: xs =>
        let best := xs.zipIdx 1 |>.foldl (fun (bestIndex, bestValue) (value, index) =>
          if value > bestValue then (index, value) else (bestIndex, bestValue)) (0, x)
        some best.1
  let axisExtent := Shape.axisSize s axis
  if axisExtent = 0 then
    List.replicate (Shape.size (Tensor.shapeAfterSum s axis)) none
  else
    let swaps := Shape.moveAxisToLastSwaps (Shape.rank s) axis
    let moved := Tensor.permuteByAdjacentSwaps values swaps
    (Spec.toList moved).toChunks axisExtent |>.map argmaxChunk

/--
Compare logits with one-hot targets along `axis`, once for every slice orthogonal to that axis.

An entry is `none` exactly when the selected class axis is empty. Otherwise it records whether the
two first-maximum indices agree.
-/
def correctOneHotAxis? {α : Type} [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (logits targetOneHot : Tensor α s) : List (Option Bool) :=
  List.zipWith (fun predicted target => do
    let predicted ← predicted
    let target ← target
    pure (predicted = target))
    (argmaxAxis? axis logits) (argmaxAxis? axis targetOneHot)

/-- Count correct and total one-hot classifications along `axis`. -/
def accuracyOneHotAxis {α : Type} [LT α] [DecidableRel ((· > ·) : α → α → Prop)]
    {s : Shape} (axis : Nat) [Shape.AxisInBounds axis s]
    (logits targetOneHot : Tensor α s) : Nat × Nat :=
  correctOneHotAxis? axis logits targetOneHot |>.foldl (fun (correct, total) result =>
    (if result = some true then correct + 1 else correct, total + 1)) (0, 0)

end Metrics

end TorchLean
