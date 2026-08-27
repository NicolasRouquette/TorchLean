/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Sequence
public import NN.Tensor.Constructors

/-!
# Public Tensor Operations

Shape-polymorphic lookup, axis operations, and list-shaped mapping and flattening helpers for
`TorchLean.Tensor`. The implementations delegate to the canonical specification operations.
-/

@[expose] public section

namespace TorchLean.Tensor

export Spec (get pretty)

/-- Read one scalar by zero-based coordinates, returning `none` for an invalid coordinate. -/
def at? {α : Type} {shape : Spec.Shape} (tensor : Tensor α shape)
    (coordinates : Array Nat) : Option α :=
  Spec.getSpec tensor coordinates.toList

namespace Internal

/-- Recursive implementation of `Tensor.stack`. -/
def stack {α : Type} {count : Nat} :
    {shape : Spec.Shape} → (axis : Nat) → (hAxis : axis ≤ shape.rank) →
      (Fin count → Tensor α shape) → Tensor α (shape.insertAxis axis count)
  | shape, 0, _, tensors => by
      rw [Spec.Shape.insertAxis_zero]
      exact Spec.Tensor.dim tensors
  | .dim _ rest, axis + 1, hAxis, tensors =>
      Spec.Tensor.dim fun i =>
        stack axis (by simp only [Spec.Shape.rank] at hAxis; grind) fun j =>
          Spec.get (tensors j) i
  | .scalar, axis + 1, hAxis, _ => by
      simp only [Spec.Shape.rank] at hAxis
      grind

end Internal

/-- Stack equally shaped tensors along any valid insertion axis. -/
def stack {α : Type} {count : Nat} {shape : Spec.Shape}
    (axis : Nat) (tensors : Fin count → Tensor α shape)
    (hAxis : axis ≤ shape.rank := by grind) :
    Tensor α (shape.insertAxis axis count) :=
  Internal.stack axis hAxis tensors

/-- Repeat a tensor along any newly inserted axis. -/
def repeatAxis {α : Type} {shape : Spec.Shape} (axis count : Nat) (tensor : Tensor α shape)
    (hAxis : axis ≤ shape.rank := by grind) : Tensor α (shape.insertAxis axis count) :=
  stack axis (fun _ => tensor) hAxis

/-- Keep the first `count` entries of any valid axis. -/
def take {α : Type} {shape : Spec.Shape} (tensor : Tensor α shape) (axis count : Nat)
    [Spec.Shape.AxisInBounds axis shape]
    (hCount : count ≤ shape.axisSize axis := by grind) :
    Tensor α (shape.replaceAxis axis count) :=
  Spec.Tensor.sliceAxisRangeSpec axis tensor 0 count (by simpa using hCount)

/-- Apply a function independently at every coordinate of the listed leading dimensions. -/
def mapEach {α : Type} (leading : List Nat) {source target : List Nat}
    (f : Tensor α source → Tensor α target)
    (tensor : Tensor α (leading ++ source)) : Tensor α (leading ++ target) := by
  let tensor' : Tensor α
      ((Spec.Shape.ofList leading).concat (Spec.Shape.ofList source)) := by
    simpa only [Spec.Shape.ofList_append] using tensor
  simpa only [Spec.Shape.ofList_append] using
    Spec.Tensor.mapEach (Spec.Shape.ofList leading)
      (inShape := Spec.Shape.ofList source) (outShape := Spec.Shape.ofList target) f tensor'

/-- Preserve `leading` and flatten every remaining axis into one row-major vector. -/
def flattenAfter {α : Type} [Inhabited α] (leading : List Nat) {source : List Nat}
    (tensor : Tensor α (leading ++ source)) : Tensor α (leading ++ [source.prod]) :=
  Spec.Tensor.reshapeSpec tensor (by
    simp [Spec.Shape.size_concat, Spec.Shape.size_ofList, Spec.Shape.size])

/-- Flatten after `leading`, then keep a checked prefix of each resulting vector. -/
def flattenThenTake {α : Type} [Inhabited α]
    (leading : List Nat) (count : Nat) {source : List Nat}
    (hCount : count ≤ source.prod)
    (tensor : Tensor α (leading ++ source)) : Tensor α (leading ++ [count]) :=
  mapEach leading
    (fun suffix =>
      let flat : Tensor α [source.prod] := by simpa using flattenAfter [] suffix
      take flat 0 count (by simpa using hCount))
    tensor

end TorchLean.Tensor
