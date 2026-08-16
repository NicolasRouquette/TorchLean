/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Scalar
public import NN.API.TensorPack
public import NN.API.Tensor
public import NN.MLTheory.SelfSupervised.PredictiveView

/-!
# Masked-Prediction API

Self-supervised learning is primarily a **training objective and data-view** interface, not a
special kind of layer.

This module is the public, model-independent SSL surface:

- it turns ordinary typed tensors into supervised training samples whose targets are derived from
  the input itself;
- it exposes deterministic masks that line up with the finite-mask theory in
  `NN.MLTheory.API`;
- it stays independent of any particular encoder, so the same SSL sample/objective helpers can be
  used with an MLP, CNN, ViT, ResNet, or a custom `nn.Sequential`.

Architecture constructors, when useful, live under `NN.API.Models.*`. For example, a compact vector
autoencoder is convenient for CIFAR runs, but the MAE idea itself belongs here: create a masked view
of a tensor and reconstruct the original content.
-/

@[expose] public section

namespace TorchLean

open Spec Tensor

namespace ssl

/-! ## Compact MAE-style masked reconstruction -/

namespace VectorMAE

/--
The hidden-coordinate mask used by compact MAE training.

`true` means "this feature coordinate is hidden from the encoder."  The type is the finite-mask
type used in the ML-theory files, specialized to the feature axis of a `batch × dataDim` matrix.
-/
def hiddenMask (dataDim period offset : Nat) :
    NN.MLTheory.SelfSupervised.Mask dataDim :=
  fun j =>
    if period = 0 then
      false
    else
      decide (j.val % period = offset % period)

private theorem hidden_mask_selected_iff (dataDim period offset : Nat) (j : Fin dataDim) :
    NN.MLTheory.SelfSupervised.selected (hiddenMask dataDim period offset) j ↔
      period ≠ 0 ∧ j.val % period = offset % period := by
  by_cases hp : period = 0
  · simp [NN.MLTheory.SelfSupervised.selected, hiddenMask, hp]
  · simp [NN.MLTheory.SelfSupervised.selected, hiddenMask, hp]

/--
Feature-level deterministic mask for MAE samples over a `batch × dataDim` matrix.

Every coordinate whose index is congruent to `offset` modulo `period` is hidden by setting it to
zero. The mask is deterministic so examples and tests are reproducible.
-/
def mask (batch dataDim period offset : Nat)
    (x : Spec.Tensor Float (.dim batch (.dim dataDim .scalar))) :
    Spec.Tensor Float (.dim batch (.dim dataDim .scalar)) :=
  Spec.Tensor.dim (fun bi =>
    let row := Spec.getAtSpec x bi
    Spec.Tensor.dim (fun j =>
      let v := Spec.Tensor.toScalar (Spec.get row j)
      let keep :=
        if period = 0 then
          true
        else
          j.val % period ≠ offset % period
      Spec.Tensor.scalar (if keep then v else 0.0)))

/--
Coordinate-level bridge from the executable tensor mask to the finite mask used in the
self-supervised theory files.

For every batch row and feature coordinate, `mask` returns exactly zero on hidden
coordinates and the original tensor value on visible coordinates.
-/
theorem mask_get_eq_if_selected_hidden (batch dataDim period offset : Nat)
    (x : Spec.Tensor Float (.dim batch (.dim dataDim .scalar)))
    (bi : Fin batch) (j : Fin dataDim) :
    Spec.Tensor.toScalar (Spec.get (Spec.get (mask batch dataDim period offset x) bi) j) =
      if hiddenMask dataDim period offset j then
        0.0
      else
        Spec.Tensor.toScalar (Spec.get (Spec.get x bi) j) := by
  by_cases hp : period = 0
  · cases x with
    | dim rows =>
        simp [mask, hiddenMask, Spec.get, Spec.getAtSpec,
          Spec.Tensor.toScalar, hp]
  · by_cases hj : j.val % period = offset % period
    · cases x with
      | dim rows =>
          simp [mask, hiddenMask, Spec.get, Spec.getAtSpec, Spec.Tensor.toScalar,
            hp, hj]
    · cases x with
      | dim rows =>
          simp [mask, hiddenMask, Spec.get, Spec.getAtSpec, Spec.Tensor.toScalar,
            hp, hj]

/-- Hidden feature coordinates are exactly zero after applying `mask`. -/
theorem mask_hidden_get_eq_zero (batch dataDim period offset : Nat)
    (x : Spec.Tensor Float (.dim batch (.dim dataDim .scalar)))
    (bi : Fin batch) (j : Fin dataDim)
    (h : NN.MLTheory.SelfSupervised.selected (hiddenMask dataDim period offset) j) :
    Spec.Tensor.toScalar (Spec.get (Spec.get (mask batch dataDim period offset x) bi) j) =
      0.0 := by
  have hbool : hiddenMask dataDim period offset j = true := by
    simpa [NN.MLTheory.SelfSupervised.selected] using h
  simpa [hbool] using
    mask_get_eq_if_selected_hidden batch dataDim period offset x bi j

/-- Visible feature coordinates are preserved by `mask`. -/
theorem mask_visible_get_eq_input (batch dataDim period offset : Nat)
    (x : Spec.Tensor Float (.dim batch (.dim dataDim .scalar)))
    (bi : Fin batch) (j : Fin dataDim)
    (h : ¬ NN.MLTheory.SelfSupervised.selected (hiddenMask dataDim period offset) j) :
    Spec.Tensor.toScalar (Spec.get (Spec.get (mask batch dataDim period offset x) bi) j) =
      Spec.Tensor.toScalar (Spec.get (Spec.get x bi) j) := by
  have hbool : hiddenMask dataDim period offset j = false := by
    cases hm : hiddenMask dataDim period offset j <;>
      simp [NN.MLTheory.SelfSupervised.selected, hm] at h ⊢
  simpa [hbool] using
    mask_get_eq_if_selected_hidden batch dataDim period offset x bi j

/--
Build a compact MAE training sample from a vector batch.

The model sees the masked vector and reconstructs the original vector.  This is represented using
TorchLean's existing supervised sample type because the "label" is derived from the input.
-/
def sample (batch dataDim period offset : Nat)
    (x : Spec.Tensor Float (.dim batch (.dim dataDim .scalar))) :
    TorchLean.Sample.Supervised Float (.dim batch (.dim dataDim .scalar))
      (.dim batch (.dim dataDim .scalar)) :=
  TorchLean.Sample.mk (mask batch dataDim period offset x) x

/--
The executable vector MAE training input is exactly the masked tensor.

This is the whole-tensor statement behind the coordinate theorems below. When the runtime training
loop calls `TorchLean.Sample.x`, it receives this tensor and no other preprocessing is hidden in
the sample wrapper.
-/
theorem sample_input_eq_mask (batch dataDim period offset : Nat)
    (x : Spec.Tensor Float (.dim batch (.dim dataDim .scalar))) :
    TorchLean.Sample.x (sample batch dataDim period offset x) =
      mask batch dataDim period offset x := by
  simp [sample]

/--
The executable vector MAE training target is exactly the original tensor.

Together with `sample_input_eq_mask`, this says the fixed-sample training call compares a
model output against the unmasked source tensor.
-/
theorem sample_target_eq_source (batch dataDim period offset : Nat)
    (x : Spec.Tensor Float (.dim batch (.dim dataDim .scalar))) :
    TorchLean.Sample.y (sample batch dataDim period offset x) = x := by
  simp [sample]

/-! ## Tensor-to-theory bridge for predictive-view SSL -/

/--
The finite hidden-index list induced by the executable vector MAE mask.

This is the serialization of the masked coordinate set used by the finite MAE/predictive-view
objective. The tensor API uses the Boolean mask directly; the theory objective sums over a list.
-/
def selectedIndices (dataDim period offset : Nat) : List (Fin dataDim) :=
  (List.finRange dataDim).filter (fun j => hiddenMask dataDim period offset j)

/-- Extract one runtime tensor row as the finite patch batch used by the SSL theory layer. -/
def rowAsPatchBatch (batch n : Nat)
    (x : Spec.Tensor Float (.dim batch (.dim n .scalar))) (bi : Fin batch) :
    NN.MLTheory.SelfSupervised.PatchBatch n Float :=
  fun j => Spec.Tensor.toScalar (Spec.get (Spec.get x bi) j)

/-- Extract one runtime prediction row as a finite prediction function. -/
def rowAsPrediction (batch n : Nat)
    (yhat : Spec.Tensor Float (.dim batch (.dim n .scalar))) (bi : Fin batch) :
    Fin n → Float :=
  fun j => Spec.Tensor.toScalar (Spec.get (Spec.get yhat bi) j)

/--
The tensor MAE sample keeps the original row as the finite theory target.

The target row is exactly the patch batch appearing in the finite MAE/predictive-view objective.
-/
theorem sample_target_row_eq_source_row (batch dataDim period offset : Nat)
    (x : Spec.Tensor Float (.dim batch (.dim dataDim .scalar)))
    (bi : Fin batch) :
    rowAsPatchBatch batch dataDim
      (TorchLean.Sample.y (sample batch dataDim period offset x)) bi =
      rowAsPatchBatch batch dataDim x bi := by
  funext j
  simp [rowAsPatchBatch, sample]

/--
The tensor MAE sample input has zero at every finite hidden coordinate.

This is the executable masking invariant seen by the model before the theory objective asks it to
predict those original target coordinates back.
-/
theorem sample_input_hidden_get_eq_zero (batch dataDim period offset : Nat)
    (x : Spec.Tensor Float (.dim batch (.dim dataDim .scalar)))
    (bi : Fin batch) (j : Fin dataDim)
    (h : NN.MLTheory.SelfSupervised.selected (hiddenMask dataDim period offset) j) :
    Spec.Tensor.toScalar
        (Spec.get (Spec.get (TorchLean.Sample.x
          (sample batch dataDim period offset x)) bi) j) = 0.0 := by
  simpa [sample] using
    mask_hidden_get_eq_zero batch dataDim period offset x bi j h

/--
A single row of the executable vector MAE path instantiates the finite predictive-view contract.

`yhat` is the model output tensor. After extracting row `bi`, the finite objective is precisely the
MAE masked reconstruction loss over the selected hidden coordinates. This is the key bridge from
`Spec.Tensor` implementation data to the SSL objective algebra.
-/
def rowPredictiveContract (batch dataDim period offset : Nat)
    (x yhat : Spec.Tensor Float (.dim batch (.dim dataDim .scalar)))
    (bi : Fin batch) (patchLoss : Float → Float → Nat) :
    NN.MLTheory.SelfSupervised.PredictiveViewContract dataDim Unit Float Float Float :=
  NN.MLTheory.SelfSupervised.maeAsPredictiveViewContract
    (selectedIndices dataDim period offset)
    (rowAsPatchBatch batch dataDim x bi)
    (rowAsPrediction batch dataDim yhat bi)
    patchLoss

/--
The extracted tensor-row predictive-view objective is exactly the finite MAE loss.

This is the formal version of the implementation diagram:

`Spec.Tensor` batch row → hidden-coordinate mask → model prediction row → masked reconstruction
objective.
-/
theorem row_predictive_objective_eq_mae_loss
    (batch dataDim period offset : Nat)
    (x yhat : Spec.Tensor Float (.dim batch (.dim dataDim .scalar)))
    (bi : Fin batch) (patchLoss : Float → Float → Nat) :
    NN.MLTheory.SelfSupervised.predictiveViewObjective
        (rowPredictiveContract batch dataDim period offset x yhat bi patchLoss) =
      NN.MLTheory.SelfSupervised.maeLoss
        (selectedIndices dataDim period offset)
        (rowAsPatchBatch batch dataDim x bi)
        (rowAsPrediction batch dataDim yhat bi)
        patchLoss := by
  exact NN.MLTheory.SelfSupervised.mae_is_predictive_view_objective
    (selectedIndices dataDim period offset)
    (rowAsPatchBatch batch dataDim x bi)
    (rowAsPrediction batch dataDim yhat bi)
    patchLoss

/--
Build a compact MAE sample from any batched tensor source.

The source can be an image tensor, spectrogram tensor, token-feature tensor, etc.  This helper
chooses a flattened prefix of each row, masks that prefix, and reconstructs the original prefix.
A full ViT/patch MAE can replace this prefix projection with a patch embedding while keeping the
same training idea.
-/
def tensorPrefixSample {source : Spec.Shape} (batch dataDim : Nat)
    (hData : dataDim ≤ Spec.Shape.size source) (period offset : Nat)
    (x : Spec.Tensor Float (.dim batch source)) :
    TorchLean.Sample.Supervised Float (.dim batch (.dim dataDim .scalar))
      (.dim batch (.dim dataDim .scalar)) :=
  sample batch dataDim period offset
    (TorchLean.Tensor.flattenPrefix (.dim batch .scalar) dataDim hData x)

end VectorMAE

end ssl
end TorchLean
