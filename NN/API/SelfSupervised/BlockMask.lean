/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Scalar
public import NN.API.Tensor
public import NN.API.TensorPack
public import NN.MLTheory.SelfSupervised.PredictiveView

/-!
# Arbitrary-Rank Block Masks

Masked prediction is not intrinsically an image operation. A model may hide intervals in a signal,
rectangles in an image, cuboids in a volume, or blocks in a higher-dimensional simulation field.
This module therefore describes a mask by two rank-indexed vectors:

* `shape : Vector Nat d` gives the tensor extents;
* `blocks : Vector (Option Nat) d` selects the axes that form a block grid.

`none` means that an axis does not participate in the block index. `some k` groups that axis into
consecutive blocks of width `k`. The selected block-grid coordinates are flattened in row-major
order, and one congruence class modulo `period` is hidden. A zero period, a zero block width, a rank
mismatch, or an out-of-bounds coordinate hides nothing.

The executable mask and its coordinate theorems use the same finite predicate. Consequently the
runtime training sample cannot silently use a different patch convention from the one stated in
Lean.
-/

@[expose] public section

namespace TorchLean
namespace ssl

open Spec Tensor

namespace BlockMask

namespace Internal

def index :
    List Nat → List (Option Nat) → List Nat → Nat → Bool → Option Nat
  | [], [], [], index, true => some index
  | [], [], [], _, false => none
  | extent :: extents, policy :: policies, coordinate :: coordinates, index, used =>
      if coordinate < extent then
        match policy with
        | none => Internal.index extents policies coordinates index used
        | some blockSize =>
            if blockSize = 0 then
              none
            else
              let blocksAlongAxis := (extent + blockSize - 1) / blockSize
              let blockCoordinate := coordinate / blockSize
              Internal.index extents policies coordinates
                (index * blocksAlongAxis + blockCoordinate) true
      else
        none
  | _, _, _, _, _ => none

end Internal

/-- Row-major block index, or `none` for an invalid/degenerate block description. -/
def index {d : Nat} (shape : Vector Nat d) (blocks : Vector (Option Nat) d)
    (coordinate : Vector Nat d) : Option Nat :=
  Internal.index shape.toList blocks.toList coordinate.toList 0 false

/-- Whether a coordinate belongs to the selected congruence class of blocks. -/
def hidden {d : Nat} (shape : Vector Nat d) (blocks : Vector (Option Nat) d)
    (period offset : Nat) (coordinate : Vector Nat d) : Bool :=
  if period = 0 then
    false
  else
    match index shape blocks coordinate with
    | some index => decide (index % period = offset % period)
    | none => false

/-- Read a scalar from a shape-indexed tensor using runtime coordinates. -/
def scalarAt {α : Type} :
    (dims : List Nat) → Spec.Tensor α (Spec.Shape.ofList dims) → List Nat → Option α
  | [], .scalar value, [] => some value
  | [], .scalar _, _ => none
  | extent :: extents, .dim values, coordinate :: coordinates =>
      if h : coordinate < extent then
        scalarAt extents (values ⟨coordinate, h⟩) coordinates
      else
        none
  | _ :: _, .dim _, [] => none

namespace Internal

def hidden (shape : List Nat) (blocks : List (Option Nat))
    (period offset : Nat) (coordinate : List Nat) : Bool :=
  if period = 0 then
    false
  else
    match index shape blocks coordinate 0 false with
    | some index => decide (index % period = offset % period)
    | none => false

def apply (shape : List Nat) (blocks : List (Option Nat))
    (period offset : Nat) (coordinatePrefix : List Nat) :
    (dims : List Nat) → Spec.Tensor Float (Spec.Shape.ofList dims) →
      Spec.Tensor Float (Spec.Shape.ofList dims)
  | [], .scalar value =>
      .scalar (if hidden shape blocks period offset coordinatePrefix then 0.0 else value)
  | _ :: extents, .dim values =>
      .dim fun coordinate =>
        apply shape blocks period offset (coordinatePrefix ++ [coordinate.val]) extents
          (values coordinate)

end Internal

/--
Set every scalar in a selected block to zero, preserving the tensor's arbitrary-rank shape.

For example, policies `[none, some 4, some 4]` repeat a 4-by-4 block mask across the first axis;
`[some 8]` masks intervals in a signal; and `[some 2, some 2, some 2]` masks volume blocks.
-/
def apply {d : Nat} (shape : Vector Nat d) (blocks : Vector (Option Nat) d)
    (period offset : Nat) (x : Spec.Tensor Float (Spec.Shape.ofList shape.toList)) :
    Spec.Tensor Float (Spec.Shape.ofList shape.toList) :=
  Internal.apply shape.toList blocks.toList period offset [] shape.toList x

@[simp] private theorem applyAux_dim
    (shape : List Nat) (blocks : List (Option Nat)) (period offset : Nat)
    (coordinatePrefix : List Nat) (extent : Nat) (extents : List Nat)
    (values : Fin extent → Spec.Tensor Float (Spec.Shape.ofList extents)) :
    Internal.apply shape blocks period offset coordinatePrefix (extent :: extents) (.dim values) =
      .dim (fun coordinate =>
        Internal.apply shape blocks period offset (coordinatePrefix ++ [coordinate.val]) extents
          (values coordinate)) := by
  rfl

@[simp] private theorem scalarAt_dim_cons {α : Type} (extent : Nat) (extents : List Nat)
    (values : Fin extent → Spec.Tensor α (Spec.Shape.ofList extents))
    (coordinate : Nat) (coordinates : List Nat) :
    scalarAt (extent :: extents) (.dim values) (coordinate :: coordinates) =
      if h : coordinate < extent then
        scalarAt extents (values ⟨coordinate, h⟩) coordinates
      else
        none := by
  rfl

private theorem scalar_at_apply_aux
    (shape : List Nat) (blocks : List (Option Nat)) (period offset : Nat)
    (coordinatePrefix : List Nat) :
    ∀ (dims : List Nat) (x : Spec.Tensor Float (Spec.Shape.ofList dims)) (coordinates : List Nat),
      scalarAt dims (Internal.apply shape blocks period offset coordinatePrefix dims x) coordinates =
        (scalarAt dims x coordinates).map (fun value =>
          if Internal.hidden shape blocks period offset (coordinatePrefix ++ coordinates) then
            0.0
          else
            value) := by
  intro dims
  induction dims generalizing coordinatePrefix with
  | nil =>
      intro x coordinates
      cases x with
      | scalar value =>
          cases coordinates <;> simp [scalarAt, Internal.apply]
  | cons extent extents ih =>
      intro x coordinates
      cases x with
      | dim values =>
          cases coordinates with
          | nil => rfl
          | cons coordinate coordinates =>
              rw [applyAux_dim, scalarAt_dim_cons, scalarAt_dim_cons]
              by_cases h : coordinate < extent
              · simp only [dif_pos h]
                simpa [List.append_assoc] using
                  ih (coordinatePrefix := coordinatePrefix ++ [coordinate])
                    (values ⟨coordinate, h⟩) coordinates
              · simp [h]

/-- Exact coordinate semantics of `apply`, including out-of-bounds coordinates. -/
theorem apply_scalar_at {d : Nat} (shape : Vector Nat d)
    (blocks : Vector (Option Nat) d) (period offset : Nat)
    (x : Spec.Tensor Float (Spec.Shape.ofList shape.toList)) (coordinate : Vector Nat d) :
    scalarAt shape.toList (apply shape blocks period offset x) coordinate.toList =
      (scalarAt shape.toList x coordinate.toList).map (fun value =>
        if hidden shape blocks period offset coordinate then 0.0 else value) := by
  simpa [apply, hidden, index, Internal.hidden] using
    scalar_at_apply_aux shape.toList blocks.toList period offset [] shape.toList x
      coordinate.toList

/-- A selected in-bounds coordinate is exactly zero after masking. -/
theorem hidden_scalar_eq_zero {d : Nat} (shape : Vector Nat d)
    (blocks : Vector (Option Nat) d) (period offset : Nat)
    (x : Spec.Tensor Float (Spec.Shape.ofList shape.toList)) (coordinate : Vector Nat d)
    (value : Float) (hValue : scalarAt shape.toList x coordinate.toList = some value)
    (hHidden : hidden shape blocks period offset coordinate = true) :
    scalarAt shape.toList (apply shape blocks period offset x) coordinate.toList =
      some 0.0 := by
  rw [apply_scalar_at, hValue, hHidden]
  rfl

/-- A visible in-bounds coordinate is copied unchanged by the mask. -/
theorem visible_scalar_eq_input {d : Nat} (shape : Vector Nat d)
    (blocks : Vector (Option Nat) d) (period offset : Nat)
    (x : Spec.Tensor Float (Spec.Shape.ofList shape.toList)) (coordinate : Vector Nat d)
    (value : Float) (hValue : scalarAt shape.toList x coordinate.toList = some value)
    (hVisible : hidden shape blocks period offset coordinate = false) :
    scalarAt shape.toList (apply shape blocks period offset x) coordinate.toList =
      some value := by
  rw [apply_scalar_at, hValue, hVisible]
  rfl

/-- Apply the same block mask independently at every index of an arbitrary leading shape. -/
def applyLeading {d : Nat} (leading : Spec.Shape) (shape : Vector Nat d)
    (blocks : Vector (Option Nat) d) (period offset : Nat)
    (x : Spec.Tensor Float (leading.concat (Spec.Shape.ofList shape.toList))) :
    Spec.Tensor Float (leading.concat (Spec.Shape.ofList shape.toList)) :=
  match leading, x with
  | .scalar, x => apply shape blocks period offset x
  | .dim _n rest, .dim rows =>
      .dim fun i => applyLeading rest shape blocks period offset (rows i)

/-- Coordinate semantics of one row when `applyLeading` is used with an ordinary batch axis. -/
theorem apply_leading_row_scalar_at {d : Nat} (batch : Nat) (shape : Vector Nat d)
    (blocks : Vector (Option Nat) d) (period offset : Nat)
    (x : Spec.Tensor Float (.dim batch (Spec.Shape.ofList shape.toList)))
    (row : Fin batch) (coordinate : Vector Nat d) :
    scalarAt shape.toList
        (Spec.get (applyLeading (.dim batch .scalar) shape blocks period offset x) row)
        coordinate.toList =
      (scalarAt shape.toList (Spec.get x row) coordinate.toList).map (fun value =>
        if hidden shape blocks period offset coordinate then 0.0 else value) := by
  cases x with
  | dim rows =>
      simpa [applyLeading, Spec.get, Spec.getAtSpec] using
        apply_scalar_at shape blocks period offset (rows row) coordinate

end BlockMask

namespace BlockMAE

/--
Create a masked-reconstruction sample with arbitrary leading and data shapes.

The model input retains its original shape. The target is a row-major prefix of the unmasked source
because TorchLean's compact decoder heads produce matrices; `reconDim` may be the entire sample or a
smaller prefix for an experiment.
-/
def sample {d : Nat} (leading : Spec.Shape) (reconDim : Nat) (shape : Vector Nat d)
    (blocks : Vector (Option Nat) d) (period offset : Nat)
    (hRecon : reconDim ≤ Spec.Shape.size (Spec.Shape.ofList shape.toList))
    (x : Spec.Tensor Float (leading.concat (Spec.Shape.ofList shape.toList))) :
    TorchLean.Sample.Supervised Float
      (leading.concat (Spec.Shape.ofList shape.toList))
      (leading.appendDim reconDim) :=
  TorchLean.Sample.mk
    (BlockMask.applyLeading leading shape blocks period offset x)
    (TorchLean.Tensor.flattenPrefix leading reconDim hRecon x)

/-- Flattened reconstruction coordinates hidden by the block mask. -/
def hiddenReconstructionIndices {d : Nat} (reconDim : Nat) (shape : Vector Nat d)
    (blocks : Vector (Option Nat) d) (period offset : Nat)
    (hRecon : reconDim ≤ Spec.Shape.size (Spec.Shape.ofList shape.toList)) :
    List (Fin reconDim) :=
  let ones : Spec.Tensor Float (Spec.Shape.ofList shape.toList) :=
    Spec.fill (α := Float) 1.0 (Spec.Shape.ofList shape.toList)
  let masked := BlockMask.apply shape blocks period offset ones
  let flatMask := TorchLean.Tensor.flattenPrefix .scalar reconDim hRecon masked
  (List.finRange reconDim).filter fun i =>
    Spec.Tensor.item (Spec.get flatMask i) == 0.0

/-- One batch row of block-MAE training as a finite predictive-view contract. -/
def rowPredictiveContract {d : Nat} (batch reconDim : Nat)
    (shape : Vector Nat d) (blocks : Vector (Option Nat) d) (period offset : Nat)
    (hRecon : reconDim ≤ Spec.Shape.size (Spec.Shape.ofList shape.toList))
    (x : Spec.Tensor Float (.dim batch (Spec.Shape.ofList shape.toList)))
    (prediction : Spec.Tensor Float (.dim batch (.dim reconDim .scalar)))
    (row : Fin batch) (loss : Float → Float → Nat) :
    NN.MLTheory.SelfSupervised.PredictiveViewContract reconDim Unit Float Float Float :=
  NN.MLTheory.SelfSupervised.maeAsPredictiveViewContract
    (hiddenReconstructionIndices reconDim shape blocks period offset hRecon)
    (fun j => Spec.Tensor.item <|
      Spec.get (Spec.get
        (TorchLean.Sample.y
          (sample (.dim batch .scalar) reconDim shape blocks period offset hRecon x)) row) j)
    (fun j => Spec.Tensor.item (Spec.get (Spec.get prediction row) j))
    loss

/-- The runnable block-MAE row objective is exactly the finite MAE objective. -/
theorem row_predictive_objective_eq_mae_loss {d : Nat} (batch reconDim : Nat)
    (shape : Vector Nat d) (blocks : Vector (Option Nat) d) (period offset : Nat)
    (hRecon : reconDim ≤ Spec.Shape.size (Spec.Shape.ofList shape.toList))
    (x : Spec.Tensor Float (.dim batch (Spec.Shape.ofList shape.toList)))
    (prediction : Spec.Tensor Float (.dim batch (.dim reconDim .scalar)))
    (row : Fin batch) (loss : Float → Float → Nat) :
    NN.MLTheory.SelfSupervised.predictiveViewObjective
        (rowPredictiveContract batch reconDim shape blocks period offset hRecon x prediction
          row loss) =
      NN.MLTheory.SelfSupervised.maeLoss
        (hiddenReconstructionIndices reconDim shape blocks period offset hRecon)
        (fun j => Spec.Tensor.item <|
          Spec.get (Spec.get
            (TorchLean.Sample.y
              (sample (.dim batch .scalar) reconDim shape blocks period offset hRecon x)) row) j)
        (fun j => Spec.Tensor.item (Spec.get (Spec.get prediction row) j))
        loss := by
  exact NN.MLTheory.SelfSupervised.mae_is_predictive_view_objective
    (hiddenReconstructionIndices reconDim shape blocks period offset hRecon)
    (fun j => Spec.Tensor.item <|
      Spec.get (Spec.get
        (TorchLean.Sample.y
          (sample (.dim batch .scalar) reconDim shape blocks period offset hRecon x)) row) j)
    (fun j => Spec.Tensor.item (Spec.get (Spec.get prediction row) j))
    loss

end BlockMAE

end ssl
end TorchLean
