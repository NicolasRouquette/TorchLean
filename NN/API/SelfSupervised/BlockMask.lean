/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.SelfSupervised.MaskedPrediction

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

/--
Compute the row-major index of the block containing `coordinate`.

Axes marked `none` are ignored, so the same spatial mask is repeated across batch, channel, token,
or feature axes. The final Boolean records whether at least one axis participates in the block grid.
-/
def indexAux :
    List Nat → List (Option Nat) → List Nat → Nat → Bool → Option Nat
  | [], [], [], index, true => some index
  | [], [], [], _, false => none
  | extent :: extents, policy :: policies, coordinate :: coordinates, index, used =>
      if coordinate < extent then
        match policy with
        | none => indexAux extents policies coordinates index used
        | some blockSize =>
            if blockSize = 0 then
              none
            else
              let blocksAlongAxis := (extent + blockSize - 1) / blockSize
              let blockCoordinate := coordinate / blockSize
              indexAux extents policies coordinates
                (index * blocksAlongAxis + blockCoordinate) true
      else
        none
  | _, _, _, _, _ => none

end Internal

/-- Row-major block index, or `none` for an invalid/degenerate block description. -/
def index {d : Nat} (shape : Vector Nat d) (blocks : Vector (Option Nat) d)
    (coordinate : Vector Nat d) : Option Nat :=
  Internal.indexAux shape.toList blocks.toList coordinate.toList 0 false

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

/-- List-level hidden-coordinate predicate used by the recursive tensor implementation. -/
def hiddenList (shape : List Nat) (blocks : List (Option Nat))
    (period offset : Nat) (coordinate : List Nat) : Bool :=
  if period = 0 then
    false
  else
    match indexAux shape blocks coordinate 0 false with
    | some index => decide (index % period = offset % period)
    | none => false

/-- Recursively apply a block mask while accumulating the current coordinate. -/
def applyAux (shape : List Nat) (blocks : List (Option Nat))
    (period offset : Nat) (coordinatePrefix : List Nat) :
    (dims : List Nat) → Spec.Tensor Float (Spec.Shape.ofList dims) →
      Spec.Tensor Float (Spec.Shape.ofList dims)
  | [], .scalar value =>
      .scalar (if hiddenList shape blocks period offset coordinatePrefix then 0.0 else value)
  | _ :: extents, .dim values =>
      .dim fun coordinate =>
        applyAux shape blocks period offset (coordinatePrefix ++ [coordinate.val]) extents
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
  Internal.applyAux shape.toList blocks.toList period offset [] shape.toList x

@[simp] private theorem applyAux_dim
    (shape : List Nat) (blocks : List (Option Nat)) (period offset : Nat)
    (coordinatePrefix : List Nat) (extent : Nat) (extents : List Nat)
    (values : Fin extent → Spec.Tensor Float (Spec.Shape.ofList extents)) :
    Internal.applyAux shape blocks period offset coordinatePrefix (extent :: extents) (.dim values) =
      .dim (fun coordinate =>
        Internal.applyAux shape blocks period offset (coordinatePrefix ++ [coordinate.val]) extents
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
      scalarAt dims (Internal.applyAux shape blocks period offset coordinatePrefix dims x) coordinates =
        (scalarAt dims x coordinates).map (fun value =>
          if Internal.hiddenList shape blocks period offset (coordinatePrefix ++ coordinates) then
            0.0
          else
            value) := by
  intro dims
  induction dims generalizing coordinatePrefix with
  | nil =>
      intro x coordinates
      cases x with
      | scalar value =>
          cases coordinates <;> simp [scalarAt, Internal.applyAux]
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
  simpa [apply, hidden, index, Internal.hiddenList] using
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

/-- Apply the same block mask independently to each row of a batch. -/
def applyBatch {d : Nat} (batch : Nat) (shape : Vector Nat d)
    (blocks : Vector (Option Nat) d) (period offset : Nat)
    (x : Spec.Tensor Float (.dim batch (Spec.Shape.ofList shape.toList))) :
    Spec.Tensor Float (.dim batch (Spec.Shape.ofList shape.toList)) :=
  .dim fun row => apply shape blocks period offset (Spec.get x row)

/-- Coordinate semantics of one row of `applyBatch`. -/
theorem apply_batch_scalar_at {d : Nat} (batch : Nat) (shape : Vector Nat d)
    (blocks : Vector (Option Nat) d) (period offset : Nat)
    (x : Spec.Tensor Float (.dim batch (Spec.Shape.ofList shape.toList)))
    (row : Fin batch) (coordinate : Vector Nat d) :
    scalarAt shape.toList (Spec.get (applyBatch batch shape blocks period offset x) row)
        coordinate.toList =
      (scalarAt shape.toList (Spec.get x row) coordinate.toList).map (fun value =>
        if hidden shape blocks period offset coordinate then 0.0 else value) := by
  cases x with
  | dim rows =>
      simpa [applyBatch, Spec.get, Spec.getAtSpec] using
        apply_scalar_at shape blocks period offset (rows row) coordinate

end BlockMask

namespace BlockMAE

/--
Create a masked-reconstruction sample from a batch of arbitrary-rank tensors.

The model input retains its original shape. The target is a row-major prefix of the unmasked source
because TorchLean's compact decoder heads produce matrices; `reconDim` may be the entire sample or a
smaller prefix for an experiment.
-/
def sample {d : Nat} (batch reconDim : Nat) (shape : Vector Nat d)
    (blocks : Vector (Option Nat) d) (period offset : Nat)
    (hRecon : reconDim ≤ Spec.Shape.size (Spec.Shape.ofList shape.toList))
    (x : Spec.Tensor Float (.dim batch (Spec.Shape.ofList shape.toList))) :
    TorchLean.Sample.Supervised Float
      (.dim batch (Spec.Shape.ofList shape.toList))
      (.dim batch (.dim reconDim .scalar)) :=
  TorchLean.Sample.mk
    (BlockMask.applyBatch batch shape blocks period offset x)
    (TorchLean.Tensor.flattenPrefix (.dim batch .scalar) reconDim hRecon x)

/-- The model input of a block-MAE sample is exactly the masked source batch. -/
theorem sample_input_eq_mask {d : Nat} (batch reconDim : Nat)
    (shape : Vector Nat d) (blocks : Vector (Option Nat) d) (period offset : Nat)
    (hRecon : reconDim ≤ Spec.Shape.size (Spec.Shape.ofList shape.toList))
    (x : Spec.Tensor Float (.dim batch (Spec.Shape.ofList shape.toList))) :
    TorchLean.Sample.x (sample batch reconDim shape blocks period offset hRecon x) =
      BlockMask.applyBatch batch shape blocks period offset x := by
  rfl

/-- The target of a block-MAE sample is the requested prefix of the unmasked source batch. -/
theorem sample_target_eq_source_prefix {d : Nat} (batch reconDim : Nat)
    (shape : Vector Nat d) (blocks : Vector (Option Nat) d) (period offset : Nat)
    (hRecon : reconDim ≤ Spec.Shape.size (Spec.Shape.ofList shape.toList))
    (x : Spec.Tensor Float (.dim batch (Spec.Shape.ofList shape.toList))) :
    TorchLean.Sample.y (sample batch reconDim shape blocks period offset hRecon x) =
      TorchLean.Tensor.flattenPrefix (.dim batch .scalar) reconDim hRecon x := by
  rfl

/-- Every decoder coordinate participates in the compact reconstruction objective. -/
def reconstructionIndices (reconDim : Nat) : List (Fin reconDim) :=
  List.finRange reconDim

/-- One batch row of block-MAE training as a finite predictive-view contract. -/
def rowPredictiveContract {d : Nat} (batch reconDim : Nat)
    (shape : Vector Nat d) (blocks : Vector (Option Nat) d) (period offset : Nat)
    (hRecon : reconDim ≤ Spec.Shape.size (Spec.Shape.ofList shape.toList))
    (x : Spec.Tensor Float (.dim batch (Spec.Shape.ofList shape.toList)))
    (prediction : Spec.Tensor Float (.dim batch (.dim reconDim .scalar)))
    (row : Fin batch) (loss : Float → Float → Nat) :
    NN.MLTheory.SelfSupervised.PredictiveViewContract reconDim Unit Float Float Float :=
  NN.MLTheory.SelfSupervised.maeAsPredictiveViewContract
    (reconstructionIndices reconDim)
    (VectorMAE.rowAsPatchBatch batch reconDim
      (TorchLean.Sample.y (sample batch reconDim shape blocks period offset hRecon x)) row)
    (VectorMAE.rowAsPrediction batch reconDim prediction row)
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
        (reconstructionIndices reconDim)
        (VectorMAE.rowAsPatchBatch batch reconDim
          (TorchLean.Sample.y
            (sample batch reconDim shape blocks period offset hRecon x)) row)
        (VectorMAE.rowAsPrediction batch reconDim prediction row)
        loss := by
  exact NN.MLTheory.SelfSupervised.mae_is_predictive_view_objective
    (reconstructionIndices reconDim)
    (VectorMAE.rowAsPatchBatch batch reconDim
      (TorchLean.Sample.y (sample batch reconDim shape blocks period offset hRecon x)) row)
    (VectorMAE.rowAsPrediction batch reconDim prediction row)
    loss

end BlockMAE

end ssl
end TorchLean
