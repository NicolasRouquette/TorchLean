/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.Float32
public import NN.Spec.Core.Complex
public import NN.Spec.Core.Tensor
import Mathlib.Algebra.Order.Algebra

/-!
# Public Tensor Constructors

This module owns the `TorchLean.Tensor` construction surface. It re-exports the shape-indexed
`Spec.Tensor` type and provides checked array conversion, coordinate generation, constants, and
bounded-index constructors. Import `NN.Tensor` for the complete public tensor API.
-/

@[expose] public section

namespace TorchLean

export Spec (Shape Tensor)

namespace Shape

export Spec.Shape
  (AxisInBounds appendDim appendDim_appendDim_eq_concat appendDim_eq_concat concat
   concat_appendDim concat_assoc getDim insertAxis insertAxis_zero ofArray ofList ofList_append
   prependDim pretty rank replaceAxis size size_appendDim size_concat toList axisSize)

end Shape

namespace Tensor

open _root_.Spec

export Spec.Tensor (item map ofArrayExact ofFlatArrayExact)

/-- Construct an arbitrary-rank tensor from a coordinate function. -/
abbrev generate {α : Type} (dims : List Nat) (f : List Nat → α) : Tensor α dims :=
  Spec.Tensor.generate dims f

/-- One-hot vector of length `n`, with a single `1` at index `k`. -/
def oneHot {α : Type} [Zero α] [One α] (n : Nat) (k : Fin n) : Tensor α [n] :=
  Spec.Tensor.dim fun i => Spec.Tensor.scalar (if decide (i = k) then (1 : α) else 0)

/-- One-hot encode every bounded index along a new final axis. -/
def oneHotIndices {α : Type} [Zero α] [One α] (n : Nat) :
    {s : Shape} → Tensor (Fin n) s → Tensor α (s.appendDim n)
  | .scalar, .scalar k => oneHot (α := α) n k
  | .dim _ _, .dim values =>
      Spec.Tensor.dim fun i => oneHotIndices (α := α) n (values i)

/-- Convert a natural number to a bounded index, rejecting an out-of-range value. -/
def checkIndex (n k : Nat) : Except String (Fin n) :=
  if h : k < n then
    .ok ⟨k, h⟩
  else
    .error s!"index {k} is outside the valid range [0, {n})"

namespace Internal

/-- Decide whether every tensor entry is smaller than `n`. -/
def indicesInRangeDecidable (n : Nat) :
    {s : Shape} → (x : Tensor Nat s) →
      Decidable (Spec.Tensor.Forall (fun k => k < n) x)
  | .scalar, .scalar _ => by
      unfold Spec.Tensor.Forall
      infer_instance
  | .dim _ _, .dim values =>
      letI : DecidablePred (fun i => Spec.Tensor.Forall (fun k => k < n) (values i)) :=
        fun i => indicesInRangeDecidable n (values i)
      by
        unfold Spec.Tensor.Forall
        exact Fintype.decidableForallFintype

/-- Attach a proved scalar bound to every entry of a tensor. -/
def boundIndices (n : Nat) :
    {s : Shape} → (x : Tensor Nat s) →
      Spec.Tensor.Forall (fun k => k < n) x → Tensor (Fin n) s
  | .scalar, .scalar k, h => .scalar ⟨k, h⟩
  | .dim _ _, .dim values, h => .dim fun i => boundIndices n (values i) (h i)

/-- Build a tensor from a flat array whose length exactly matches its dimensions. -/
def fromArray {α : Type} (dims : List Nat) (xs : Array α)
    (h : xs.size = dims.prod) : Tensor α dims :=
  Spec.Tensor.ofFlatArrayExact (Shape.ofList dims) xs (by simpa using h)

end Internal

/-- Validate every natural-number entry and return a tensor of bounded indices. -/
def checkIndices (n : Nat) {s : Shape} (x : Tensor Nat s) :
    Except String (Tensor (Fin n) s) :=
  letI := Internal.indicesInRangeDecidable n x
  if h : Spec.Tensor.Forall (fun k => k < n) x then
    .ok (Internal.boundIndices n x h)
  else
    .error s!"tensor contains an index outside the valid range [0, {n})"

/-- Build a tensor from runtime dimensions and a flat array, checking the payload length. -/
def ofArray {α : Type} (dims : List Nat) (xs : Array α) : Except String (Tensor α dims) :=
  let expected := dims.prod
  if h : xs.size = expected then
    .ok (Spec.Tensor.ofFlatArrayExact (Shape.ofList dims) xs (by simpa using h))
  else
    .error s!"ofArray: expected {expected} elements for dims={dims}, got {xs.size}"

/-- Fill a tensor of arbitrary shape with one value. -/
def full {α : Type} (dims : List Nat) (value : α) : Tensor α dims :=
  Spec.fill value (Shape.ofList dims)

/-- Construct an all-zero tensor of arbitrary shape. -/
def zeros {α : Type} [Zero α] (dims : List Nat) : Tensor α dims :=
  full (α := α) dims 0

/-- Construct an all-one tensor of arbitrary shape. -/
def ones {α : Type} [One α] (dims : List Nat) : Tensor α dims :=
  full (α := α) dims 1

/-- Build from `Float` storage and convert each entry to the selected scalar type. -/
def fromFloatArray {α : Type} (cast : Float → α) (dims : List Nat) (xs : Array Float) :
    Except String (Tensor α dims) := do
  let tensor ← ofArray (α := Float) dims xs
  pure (map cast tensor)

/-- Generate entries from flat indices as `Float` values and convert them to another scalar type. -/
def generateFromFloat {α : Type} (cast : Float → α) (dims : List Nat) (f : Nat → Float) :
    Tensor α dims :=
  let xs := Array.ofFn (fun i : Fin dims.prod => f i)
  have hSize : xs.size = dims.prod := by simp [xs]
  map cast (Spec.Tensor.ofFlatArrayExact (Shape.ofList dims) xs (by simpa using hSize))

end Tensor
end TorchLean
