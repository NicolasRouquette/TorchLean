/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Scalar
public import NN.Tensor
public import NN.API.Sample
public import NN.MLTheory.SelfSupervised.PredictiveView

/-!
# Arbitrary-Rank Block Masks

Masked prediction is not intrinsically an image operation. A model may hide intervals in a signal,
rectangles in an image, cuboids in a volume, or blocks in a higher-dimensional simulation field.
This module describes a mask by a rank-indexed policy tensor. The extents come from the input
tensor's type, while `blocks : Tensor (Option Nat) [d]` selects the axes that form a block
grid.

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
def index {dims : List Nat} (blocks : Tensor (Option Nat) [dims.length])
    (coordinate : Tensor Nat [dims.length]) : Option Nat :=
  Internal.index dims blocks.toList coordinate.toList 0 false

/-- Whether a coordinate belongs to the selected congruence class of blocks. -/
def hidden {dims : List Nat} (blocks : Tensor (Option Nat) [dims.length])
    (period offset : Nat) (coordinate : Tensor Nat [dims.length]) : Bool :=
  if period = 0 then
    false
  else
    match index blocks coordinate with
    | some index => decide (index % period = offset % period)
    | none => false

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
    (dims : List Nat) → Tensor Float (Shape.ofList dims) →
      Tensor Float (Shape.ofList dims)
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
def apply {dims : List Nat} (blocks : Tensor (Option Nat) [dims.length])
    (period offset : Nat) (x : Tensor Float dims) : Tensor Float dims :=
  Internal.apply dims blocks.toList period offset [] dims x

@[simp] private theorem applyAux_dim
    (shape : List Nat) (blocks : List (Option Nat)) (period offset : Nat)
    (coordinatePrefix : List Nat) (extent : Nat) (extents : List Nat)
    (values : Fin extent → Tensor Float (Shape.ofList extents)) :
    Internal.apply shape blocks period offset coordinatePrefix (extent :: extents) (.dim values) =
      .dim (fun coordinate =>
        Internal.apply shape blocks period offset (coordinatePrefix ++ [coordinate.val]) extents
          (values coordinate)) := by
  rfl

@[simp] private theorem at_dim_cons {α : Type} (extent : Nat) (extents : List Nat)
    (values : Fin extent → Tensor α (Shape.ofList extents))
    (coordinate : Nat) (coordinates : List Nat) :
    Spec.getSpec (.dim values) (coordinate :: coordinates) =
      if h : coordinate < extent then
        Spec.getSpec (values ⟨coordinate, h⟩) coordinates
      else
        none := by
  simp

private theorem at_apply_aux
    (shape : List Nat) (blocks : List (Option Nat)) (period offset : Nat)
    (coordinatePrefix : List Nat) :
    ∀ (dims : List Nat) (x : Tensor Float (Shape.ofList dims)) (coordinates : List Nat),
      Spec.getSpec
          (Internal.apply shape blocks period offset coordinatePrefix dims x) coordinates =
        (Spec.getSpec x coordinates).map (fun value =>
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
          cases coordinates <;>
            simp [Spec.getSpec, Internal.apply]
  | cons extent extents ih =>
      intro x coordinates
      cases x with
      | dim values =>
          cases coordinates with
          | nil => simp
          | cons coordinate coordinates =>
              rw [applyAux_dim, at_dim_cons, at_dim_cons]
              by_cases h : coordinate < extent
              · simp only [dif_pos h]
                simpa [List.append_assoc] using
                  ih (coordinatePrefix := coordinatePrefix ++ [coordinate])
                    (values ⟨coordinate, h⟩) coordinates
              · simp [h]

/-- Exact coordinate semantics of `apply`, including out-of-bounds coordinates. -/
theorem apply_scalar_at {dims : List Nat}
    (blocks : Tensor (Option Nat) [dims.length]) (period offset : Nat)
    (x : Tensor Float dims) (coordinate : Tensor Nat [dims.length]) :
    Spec.getSpec (apply blocks period offset x) coordinate.toList =
      (Spec.getSpec x coordinate.toList).map (fun value =>
        if hidden blocks period offset coordinate then
          0.0
        else value) := by
  simpa [apply, hidden, index, Internal.hidden] using
    at_apply_aux dims blocks.toList period offset [] dims x
      coordinate.toList

/-- A selected in-bounds coordinate is exactly zero after masking. -/
theorem hidden_scalar_eq_zero {dims : List Nat}
    (blocks : Tensor (Option Nat) [dims.length]) (period offset : Nat)
    (x : Tensor Float dims) (coordinate : Tensor Nat [dims.length])
    (value : Float) (hValue : Spec.getSpec x coordinate.toList = some value)
    (hHidden : hidden blocks period offset coordinate = true) :
    Spec.getSpec (apply blocks period offset x) coordinate.toList =
      some 0.0 := by
  rw [apply_scalar_at, hValue, hHidden]
  rfl

/-- A visible in-bounds coordinate is copied unchanged by the mask. -/
theorem visible_scalar_eq_input {dims : List Nat}
    (blocks : Tensor (Option Nat) [dims.length]) (period offset : Nat)
    (x : Tensor Float dims) (coordinate : Tensor Nat [dims.length])
    (value : Float) (hValue : Spec.getSpec x coordinate.toList = some value)
    (hVisible : hidden blocks period offset coordinate = false) :
    Spec.getSpec (apply blocks period offset x) coordinate.toList =
      some value := by
  rw [apply_scalar_at, hValue, hVisible]
  rfl

namespace Internal

def applyPrefix {d : Nat} (leadingShape : Shape) (shape : Tensor Nat [d])
    (blocks : Tensor (Option Nat) [d]) (period offset : Nat)
    (x : Tensor Float (leadingShape.concat (Shape.ofList shape.toList))) :
    Tensor Float (leadingShape.concat (Shape.ofList shape.toList)) :=
  match leadingShape, x with
  | .scalar, x =>
      apply shape.toList blocks.toList period offset [] shape.toList x
  | .dim _n rest, .dim rows =>
      .dim fun i => applyPrefix rest shape blocks period offset (rows i)

end Internal

end BlockMask

namespace BlockMAE

/--
Create a masked-reconstruction sample with arbitrary leading and data shapes.

The model input retains its original shape. The target is a row-major prefix of the unmasked source
because TorchLean's compact decoder heads produce matrices; `reconDim` may be the entire sample or a
smaller prefix for an experiment.
-/
def sample {d : Nat} (leading : List Nat) (reconDim : Nat) (shape : Tensor Nat [d])
    (blocks : Tensor (Option Nat) [d]) (period offset : Nat)
    (hRecon : reconDim ≤ shape.toList.prod)
    (x : Tensor Float (leading ++ shape.toList)) :
    TorchLean.Sample.Supervised Float
      (leading ++ shape.toList : List Nat) (leading ++ [reconDim] : List Nat) :=
  TorchLean.Sample.mk
    (by
      let x' : Tensor Float
          ((Shape.ofList leading).concat (Shape.ofList shape.toList)) := by
        simpa only [Shape.ofList_append] using x
      simpa only [Shape.ofList_append] using
        BlockMask.Internal.applyPrefix (Shape.ofList leading) shape blocks period offset x')
    (TorchLean.Tensor.flattenThenTake leading reconDim hRecon x)

/-- Flattened reconstruction coordinates hidden by the block mask. -/
def hiddenReconstructionIndices {d : Nat} (reconDim : Nat) (shape : Tensor Nat [d])
    (blocks : Tensor (Option Nat) [d]) (period offset : Nat)
    (hRecon : reconDim ≤ shape.toList.prod) :
    Array (Fin reconDim) :=
  let ones : Tensor Float shape.toList := Spec.fill (α := Float) 1.0 shape.toList
  let blocks' : Tensor (Option Nat) [shape.toList.length] := by
    have hLength : shape.toList.length = d := by
      rw [Spec.Tensor.toList_length]
      simp only [Spec.Shape.size, Nat.mul_one]
    rw [hLength]
    exact blocks
  let masked := BlockMask.apply blocks' period offset ones
  let flatMask := TorchLean.Tensor.flattenThenTake [] reconDim hRecon masked
  (Array.finRange reconDim).filter fun i =>
    Spec.Tensor.item (Spec.get flatMask i) == 0.0

/-- One batch row of block-MAE training as a finite predictive-view contract. -/
def rowPredictiveContract {d : Nat} (batch reconDim : Nat)
    (shape : Tensor Nat [d]) (blocks : Tensor (Option Nat) [d]) (period offset : Nat)
    (hRecon : reconDim ≤ shape.toList.prod)
    (x : Tensor Float (batch :: shape.toList))
    (prediction : Tensor Float [batch, reconDim])
    (row : Fin batch) (loss : Float → Float → Nat) :
    NN.MLTheory.SelfSupervised.PredictiveViewContract reconDim Unit Float Float Float :=
  NN.MLTheory.SelfSupervised.maeAsPredictiveViewContract
    (hiddenReconstructionIndices reconDim shape blocks period offset hRecon)
    (fun j => Spec.Tensor.item <|
      Spec.get (Spec.get
        (TorchLean.Sample.y
          (sample [batch] reconDim shape blocks period offset hRecon x)) row) j)
    (fun j => Spec.Tensor.item (Spec.get (Spec.get prediction row) j))
    loss

/-- The runnable block-MAE row objective is exactly the finite MAE objective. -/
theorem row_predictive_objective_eq_mae_loss {d : Nat} (batch reconDim : Nat)
    (shape : Tensor Nat [d]) (blocks : Tensor (Option Nat) [d]) (period offset : Nat)
    (hRecon : reconDim ≤ shape.toList.prod)
    (x : Tensor Float (batch :: shape.toList))
    (prediction : Tensor Float [batch, reconDim])
    (row : Fin batch) (loss : Float → Float → Nat) :
    NN.MLTheory.SelfSupervised.predictiveViewObjective
        (rowPredictiveContract batch reconDim shape blocks period offset hRecon x prediction
          row loss) =
      NN.MLTheory.SelfSupervised.maeLoss
        (hiddenReconstructionIndices reconDim shape blocks period offset hRecon)
        (fun j => Spec.Tensor.item <|
          Spec.get (Spec.get
            (TorchLean.Sample.y
              (sample [batch] reconDim shape blocks period offset hRecon x)) row) j)
        (fun j => Spec.Tensor.item (Spec.get (Spec.get prediction row) j))
        loss := by
  exact NN.MLTheory.SelfSupervised.mae_is_predictive_view_objective
    (hiddenReconstructionIndices reconDim shape blocks period offset hRecon)
    (fun j => Spec.Tensor.item <|
      Spec.get (Spec.get
        (TorchLean.Sample.y
          (sample [batch] reconDim shape blocks period offset hRecon x)) row) j)
    (fun j => Spec.Tensor.item (Spec.get (Spec.get prediction row) j))
    loss

end BlockMAE

end ssl
end TorchLean
