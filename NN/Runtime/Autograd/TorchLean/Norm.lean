/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.Functional

import Mathlib.Algebra.Order.Algebra

/-!
# Norm

Execution-generic normalization operations built from the small `Ops` surface. Spatial dimensions
are represented by an arbitrary `Shape`; the same definitions therefore cover vectors, images,
volumes, and higher-rank data.

Note: we do **not** implement PyTorch-style running-stat updates as a backend-generic op.
You can still compute batch statistics (mean/var) and update running buffers in the imperative
session layer.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor

namespace Norm

namespace Internal

/-- Flatten an arbitrary spatial shape while preserving the batch and channel axes. -/
theorem reshapeBatchChannelFlatSize {batch channels : Nat} {spatial : Shape} :
    Shape.size (.dim batch (.dim channels spatial)) =
      Shape.size (.dim batch (.dim channels (.dim (Shape.size spatial) .scalar))) := by
  simp [Spec.Shape.size]

/-- Flatten an arbitrary spatial shape while preserving the channel axis. -/
theorem reshapeChannelFlatSize {channels : Nat} {spatial : Shape} :
    Shape.size (.dim channels spatial) =
      Shape.size (.dim channels (.dim (Shape.size spatial) .scalar)) := by
  simp [Spec.Shape.size]

/-- Repeat a channel vector over the batch and flattened spatial axes. -/
def broadcastChannelToBatchSpatial {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    (batch channels spatialSize : Nat)
    (x : RefTy (m := m) (α := α) (.dim channels .scalar)) :
    m (RefTy (m := m) (α := α)
      (.dim batch (.dim channels (.dim spatialSize .scalar)))) := do
  let acrossSpatial ← Runtime.Autograd.Torch.broadcastAfterSum
    (m := m) (α := α) (.dim channels (.dim spatialSize .scalar)) 1 x
  Runtime.Autograd.Torch.broadcastAfterSum
    (m := m) (α := α) (.dim batch (.dim channels (.dim spatialSize .scalar))) 0 acrossSpatial

/-- Repeat a channel vector over a flattened spatial axis. -/
def broadcastChannelToSpatial {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    (channels spatialSize : Nat)
    (x : RefTy (m := m) (α := α) (.dim channels .scalar)) :
    m (RefTy (m := m) (α := α) (.dim channels (.dim spatialSize .scalar))) :=
  Runtime.Autograd.Torch.broadcastAfterSum
    (m := m) (α := α) (.dim channels (.dim spatialSize .scalar)) 1 x

end Internal

/-- RMS normalization over the final axis of a tensor, including zero-sized leading axes. -/
def rmsNorm {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {leading : Shape} {width : Nat} (hWidth : width > 0)
    (x : RefTy (m := m) (α := α) (leading.appendDim width))
    (gamma : RefTy (m := m) (α := α) (.dim width .scalar))
    (ε : α := Numbers.normalizationEpsilon) :
    m (RefTy (m := m) (α := α) (leading.appendDim width)) := by
  by_cases hLeading : Shape.size leading = 0
  · exact const (m := m) (α := α) (s := leading.appendDim width)
      (Spec.fill (0 : α) (leading.appendDim width))
  · exact do
      let s := leading.appendDim width
      let hSize : 0 < Shape.size s := by
        simpa [s, Shape.size_appendDim] using
          Nat.mul_pos (Nat.pos_of_ne_zero hLeading) hWidth
      let _ : Shape.WellFormed s := ⟨Shape.wellFormed_of_size_pos hSize⟩
      let sq ← F.square (m := m) (α := α) (s := s) x
      let axis := Shape.rank s - 1
      let _ : Shape.HasNonemptyAxis axis s :=
        Shape.inferNonemptyAxis (by simp [axis, s])
      let meanSq ← reduceMean (m := m) (α := α) (s := s) axis sq
      let meanSqShape := shapeAfterSum s axis
      let zero ← const (m := m) (α := α) (s := meanSqShape) (Spec.fill (0 : α) meanSqShape)
      let meanSqClamped ← max (m := m) (α := α) (s := meanSqShape) meanSq zero
      let epsT ← const (m := m) (α := α) (s := meanSqShape) (Spec.fill ε meanSqShape)
      let denom ← sqrt (m := m) (α := α) (s := meanSqShape)
        (← add (m := m) (α := α) (s := meanSqShape) meanSqClamped epsT)
      let invDenom ← inv (m := m) (α := α) (s := meanSqShape) denom
      let invDenomB ← Runtime.Autograd.Torch.broadcastAfterSum
        (m := m) (α := α) s axis invDenom
      let normalized ← mul (m := m) (α := α) (s := s) x invDenomB
      let gammaB ← broadcastTo (m := m) (α := α) (s₁ := .dim width .scalar) (s₂ := s)
        (by
          simpa [s, Shape.appendDim_eq_concat] using
            Shape.CanBroadcastTo.prependTarget leading (.dim width .scalar)) gamma
      mul (m := m) (α := α) (s := s) normalized gammaB

/-- Normalize the final axis by `sqrt(sum (x * x) + epsilon)`. -/
def l2Normalize {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {leading : Shape} {width : Nat} (hWidth : width > 0)
    (x : RefTy (m := m) (α := α) (leading.appendDim width))
    (epsilon : RefTy (m := m) (α := α) .scalar) :
    m (RefTy (m := m) (α := α) (leading.appendDim width)) := by
  by_cases hLeading : Shape.size leading = 0
  · exact const (m := m) (α := α) (s := leading.appendDim width)
      (Spec.fill (0 : α) (leading.appendDim width))
  · exact do
      let s := leading.appendDim width
      let hSize : 0 < Shape.size s := by
        simpa [s, Shape.size_appendDim] using
          Nat.mul_pos (Nat.pos_of_ne_zero hLeading) hWidth
      let _ : Shape.WellFormed s := ⟨Shape.wellFormed_of_size_pos hSize⟩
      let squared ← mul (m := m) (α := α) (s := s) x x
      let axis := Shape.rank s - 1
      let _ : Shape.HasNonemptyAxis axis s :=
        Shape.inferNonemptyAxis (by simp [axis, s])
      let normSquared ← reduceSum (m := m) (α := α) (s := s) axis squared
      let reducedShape := shapeAfterSum s axis
      let epsilonB ← broadcastTo (m := m) (α := α) (s₁ := .scalar) (s₂ := reducedShape)
        (Shape.CanBroadcastTo.scalarTo reducedShape) epsilon
      let denominator ← sqrt (m := m) (α := α) (s := reducedShape)
        (← add (m := m) (α := α) (s := reducedShape) normSquared epsilonB)
      let inverse ← inv (m := m) (α := α) (s := reducedShape) denominator
      let inverseB ← Runtime.Autograd.Torch.broadcastAfterSum
        (m := m) (α := α) s axis inverse
      mul (m := m) (α := α) (s := s) x inverseB

/-- Normalize each sample and channel independently over every spatial axis. -/
def instanceNorm {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch channels : Nat} {spatial : Shape}
    (hWellFormed : (Shape.dim batch (Shape.dim channels spatial)).wellFormed)
    (x : RefTy (m := m) (α := α) (.dim batch (.dim channels spatial)))
    (gamma beta : RefTy (m := m) (α := α) (.dim channels .scalar))
    (ε : α := Numbers.normalizationEpsilon) :
    m (RefTy (m := m) (α := α) (.dim batch (.dim channels spatial))) := do
  let spatialSize := Shape.size spatial
  let inputShape : Shape := .dim batch (.dim channels spatial)
  let flatShape : Shape := .dim batch (.dim channels (.dim spatialSize .scalar))
  let hBatch := hWellFormed.1
  let hChannels := hWellFormed.2.1
  let hSpatial := Shape.size_pos_of_well_formed hWellFormed.2.2
  let _ : Shape.WellFormed inputShape := ⟨hWellFormed⟩
  let _ : Shape.WellFormed flatShape := ⟨⟨hBatch, ⟨hChannels, ⟨hSpatial, trivial⟩⟩⟩⟩
  let xFlat ← reshape (m := m) (α := α) (s₁ := inputShape) (s₂ := flatShape) x
    (Internal.reshapeBatchChannelFlatSize (batch := batch) (channels := channels)
      (spatial := spatial))
  let axis := Shape.rank flatShape - 1
  let _ : Shape.HasNonemptyAxis axis flatShape :=
    Shape.inferNonemptyAxis (by simp [axis, flatShape, Shape.rank])
  let mean ← reduceMean (m := m) (α := α) (s := flatShape) axis xFlat
  let meanShape := shapeAfterSum flatShape axis
  let meanB ← Runtime.Autograd.Torch.broadcastAfterSum
    (m := m) (α := α) flatShape axis mean
  let centered ← sub (m := m) (α := α) (s := flatShape) xFlat meanB
  let sq ← F.square (m := m) (α := α) (s := flatShape) centered
  let var ← reduceMean (m := m) (α := α) (s := flatShape) axis sq
  let zero ← const (m := m) (α := α) (s := meanShape) (Spec.fill (0 : α) meanShape)
  let varClamped ← max (m := m) (α := α) (s := meanShape) var zero
  let epsT ← const (m := m) (α := α) (s := meanShape) (Spec.fill ε meanShape)
  let denom ← sqrt (m := m) (α := α) (s := meanShape) (← add (m := m) (α := α) (s := meanShape)
    varClamped epsT)
  let invDenom ← inv (m := m) (α := α) (s := meanShape) denom
  let invDenomB ← Runtime.Autograd.Torch.broadcastAfterSum
    (m := m) (α := α) flatShape axis invDenom
  let normalized ← mul (m := m) (α := α) (s := flatShape) centered invDenomB
  let gammaB ← Internal.broadcastChannelToBatchSpatial
    (m := m) (α := α) batch channels spatialSize gamma
  let betaB ← Internal.broadcastChannelToBatchSpatial
    (m := m) (α := α) batch channels spatialSize beta
  let yFlat ← add (m := m) (α := α) (s := flatShape)
    (← mul (m := m) (α := α) (s := flatShape) normalized gammaB) betaB
  reshape (m := m) (α := α) (s₁ := flatShape) (s₂ := inputShape) yFlat
    (Internal.reshapeBatchChannelFlatSize (batch := batch) (channels := channels)
      (spatial := spatial)).symm

/--
Group normalization over an arbitrary spatial shape.
-/
def groupNorm {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch channels groups : Nat} {spatial : Shape}
    (hWellFormed : (Shape.dim batch (Shape.dim channels spatial)).wellFormed)
    (hGroups : groups > 0) (hGroupsLe : channels ≥ groups)
    (hDiv : channels % groups = 0)
    (x : RefTy (m := m) (α := α) (.dim batch (.dim channels spatial)))
    (gamma beta : RefTy (m := m) (α := α) (.dim channels .scalar))
    (ε : α := Numbers.normalizationEpsilon) :
    m (RefTy (m := m) (α := α) (.dim batch (.dim channels spatial))) := do
  let channelsPerGroup := channels / groups
  let spatialSize := Shape.size spatial
  let groupSize := channelsPerGroup * spatialSize
  let inputShape : Shape := .dim batch (.dim channels spatial)
  let groupedShape : Shape := .dim batch (.dim groups (.dim groupSize .scalar))
  let hBatch := hWellFormed.1
  let hSpatial := Shape.size_pos_of_well_formed hWellFormed.2.2
  have hChannelsPerGroup : channelsPerGroup > 0 := by
    exact Nat.div_pos hGroupsLe hGroups
  have hGroupSize : groupSize > 0 := Nat.mul_pos hChannelsPerGroup hSpatial
  have hChannelsEq : channels = groups * channelsPerGroup := by
    simpa [channelsPerGroup, hDiv] using (Nat.mod_add_div channels groups).symm
  let _ : Shape.WellFormed inputShape := ⟨hWellFormed⟩
  let _ : Shape.WellFormed groupedShape :=
    ⟨⟨hBatch, ⟨hGroups, ⟨hGroupSize, trivial⟩⟩⟩⟩
  have hReshape : Shape.size inputShape = Shape.size groupedShape := by
    rw [show inputShape = .dim batch (.dim channels spatial) by rfl,
      show groupedShape = .dim batch (.dim groups (.dim groupSize .scalar)) by rfl]
    simp only [Shape.size]
    rw [hChannelsEq]
    simp [groupSize, spatialSize, Nat.mul_assoc]
  let xGrouped ← reshape (m := m) (α := α) (s₁ := inputShape) (s₂ := groupedShape) x hReshape
  let axis := Shape.rank groupedShape - 1
  let _ : Shape.HasNonemptyAxis axis groupedShape :=
    Shape.inferNonemptyAxis (by simp [axis, groupedShape, Shape.rank])
  let mean ← reduceMean (m := m) (α := α) (s := groupedShape) axis xGrouped
  let meanShape := shapeAfterSum groupedShape axis
  let meanB ← Runtime.Autograd.Torch.broadcastAfterSum
    (m := m) (α := α) groupedShape axis mean
  let centered ← sub (m := m) (α := α) (s := groupedShape) xGrouped meanB
  let sq ← F.square (m := m) (α := α) (s := groupedShape) centered
  let var ← reduceMean (m := m) (α := α) (s := groupedShape) axis sq
  let zeroVar ← const (m := m) (α := α) (s := meanShape) (Spec.fill (0 : α) meanShape)
  let varClamped ← max (m := m) (α := α) (s := meanShape) var zeroVar
  let epsT ← const (m := m) (α := α) (s := meanShape) (Spec.fill ε meanShape)
  let denom ← sqrt (m := m) (α := α) (s := meanShape)
    (← add (m := m) (α := α) (s := meanShape) varClamped epsT)
  let invDenom ← inv (m := m) (α := α) (s := meanShape) denom
  let invDenomB ← Runtime.Autograd.Torch.broadcastAfterSum
    (m := m) (α := α) groupedShape axis invDenom
  let normalized ← mul (m := m) (α := α) (s := groupedShape) centered invDenomB
  let normalizedInput ← reshape (m := m) (α := α)
    (s₁ := groupedShape) (s₂ := inputShape) normalized hReshape.symm
  let flatShape : Shape := .dim batch (.dim channels (.dim spatialSize .scalar))
  let _ : Shape.WellFormed flatShape :=
    ⟨⟨hBatch, ⟨hWellFormed.2.1, ⟨hSpatial, trivial⟩⟩⟩⟩
  let yFlat ← reshape (m := m) (α := α) (s₁ := inputShape) (s₂ := flatShape)
    normalizedInput (Internal.reshapeBatchChannelFlatSize (batch := batch)
      (channels := channels) (spatial := spatial))
  let gammaB ← Internal.broadcastChannelToBatchSpatial
    (m := m) (α := α) batch channels spatialSize gamma
  let betaB ← Internal.broadcastChannelToBatchSpatial
    (m := m) (α := α) batch channels spatialSize beta
  let yFlat ← add (m := m) (α := α) (s := flatShape)
    (← mul (m := m) (α := α) (s := flatShape) yFlat gammaB) betaB
  reshape (m := m) (α := α) (s₁ := flatShape) (s₂ := inputShape) yFlat
    (Internal.reshapeBatchChannelFlatSize (batch := batch) (channels := channels)
      (spatial := spatial)).symm

/-- Batch normalization using batch and spatial statistics, returning per-channel statistics. -/
def batchNormTrainStats {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch channels : Nat} {spatial : Shape}
    (hWellFormed : (Shape.dim batch (Shape.dim channels spatial)).wellFormed)
    (x : RefTy (m := m) (α := α) (.dim batch (.dim channels spatial)))
    (gamma beta : RefTy (m := m) (α := α) (.dim channels .scalar))
    (ε : α := Numbers.normalizationEpsilon) :
    m (RefTy (m := m) (α := α) (.dim batch (.dim channels spatial)) ×
       RefTy (m := m) (α := α) (.dim channels .scalar) ×
       RefTy (m := m) (α := α) (.dim channels .scalar)) := do
  let spatialSize := Shape.size spatial
  let inputShape : Shape := .dim batch (.dim channels spatial)
  let flatShape : Shape := .dim batch (.dim channels (.dim spatialSize .scalar))
  let hBatch := hWellFormed.1
  let hChannels := hWellFormed.2.1
  let hSpatial := Shape.size_pos_of_well_formed hWellFormed.2.2
  let _ : Shape.WellFormed inputShape := ⟨hWellFormed⟩
  let _ : Shape.WellFormed flatShape := ⟨⟨hBatch, ⟨hChannels, ⟨hSpatial, trivial⟩⟩⟩⟩
  let xFlat ← reshape (m := m) (α := α) (s₁ := inputShape) (s₂ := flatShape) x
    (Internal.reshapeBatchChannelFlatSize (batch := batch) (channels := channels)
      (spatial := spatial))
  let spatialAxis := Shape.rank flatShape - 1
  let _ : Shape.HasNonemptyAxis spatialAxis flatShape :=
    Shape.inferNonemptyAxis (by simp [spatialAxis, flatShape, Shape.rank])
  let meanSpatial ← reduceMean (m := m) (α := α) (s := flatShape) spatialAxis xFlat
  let batchChannelShape := shapeAfterSum flatShape spatialAxis
  have hBatchChannelShape : batchChannelShape = .dim batch (.dim channels .scalar) := by
    simp [batchChannelShape, flatShape, spatialAxis, Shape.rank, shapeAfterSum]
  let _ : Shape.WellFormed batchChannelShape := by
    simpa [hBatchChannelShape] using
      (show Shape.WellFormed (.dim batch (.dim channels .scalar)) from
        ⟨⟨hBatch, ⟨hChannels, trivial⟩⟩⟩)
  let batchAxis := 0
  let _ : Shape.HasNonemptyAxis batchAxis batchChannelShape := by
    simpa [hBatchChannelShape] using
      Shape.hasNonemptyAxisZeroOfPos (n := batch) (s := .dim channels .scalar) hBatch
  let mean ← reduceMean (m := m) (α := α) (s := batchChannelShape) batchAxis meanSpatial
  let meanB ← Internal.broadcastChannelToBatchSpatial
    (m := m) (α := α) batch channels spatialSize mean
  let centered ← sub (m := m) (α := α) (s := flatShape) xFlat meanB
  let sq ← F.square (m := m) (α := α) (s := flatShape) centered
  let varSpatial ← reduceMean (m := m) (α := α) (s := flatShape) spatialAxis sq
  let var ← reduceMean (m := m) (α := α) (s := batchChannelShape) batchAxis varSpatial
  let channelShape : Shape := .dim channels .scalar
  let zero ← const (m := m) (α := α) (s := channelShape)
    (Spec.fill (0 : α) channelShape)
  let varClamped ← max (m := m) (α := α) (s := channelShape) var zero
  let epsT ← const (m := m) (α := α) (s := channelShape) (Spec.fill ε channelShape)
  let denom ← sqrt (m := m) (α := α) (s := channelShape)
    (← add (m := m) (α := α) (s := channelShape) varClamped epsT)
  let invDenom ← inv (m := m) (α := α) (s := channelShape) denom
  let invDenomB ← Internal.broadcastChannelToBatchSpatial
    (m := m) (α := α) batch channels spatialSize invDenom
  let normalized ← mul (m := m) (α := α) (s := flatShape) centered invDenomB
  let gammaB ← Internal.broadcastChannelToBatchSpatial
    (m := m) (α := α) batch channels spatialSize gamma
  let betaB ← Internal.broadcastChannelToBatchSpatial
    (m := m) (α := α) batch channels spatialSize beta
  let yFlat ← add (m := m) (α := α) (s := flatShape)
    (← mul (m := m) (α := α) (s := flatShape) normalized gammaB) betaB
  let y ← reshape (m := m) (α := α) (s₁ := flatShape) (s₂ := inputShape) yFlat
    (Internal.reshapeBatchChannelFlatSize (batch := batch) (channels := channels)
      (spatial := spatial)).symm
  pure (y, mean, varClamped)

/-- Batch normalization using statistics computed over the batch and every spatial axis. -/
def batchNormTrain {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch channels : Nat} {spatial : Shape}
    (hWellFormed : (Shape.dim batch (Shape.dim channels spatial)).wellFormed)
    (x : RefTy (m := m) (α := α) (.dim batch (.dim channels spatial)))
    (gamma beta : RefTy (m := m) (α := α) (.dim channels .scalar))
    (ε : α := Numbers.normalizationEpsilon) :
    m (RefTy (m := m) (α := α) (.dim batch (.dim channels spatial))) := do
  let (y, _mean, _var) ← batchNormTrainStats
    (α := α) (m := m) hWellFormed x gamma beta (ε := ε)
  pure y

/--
Update running BatchNorm statistics (EMA):

`running := (1 - momentum) * running + momentum * batch`.

This helper updates the tensors it is given. A PyTorch-compatible caller passes the biased batch
mean and an unbiased running-variance estimate; `NN.unbiasedRunningVariance` performs that
conversion for the stateful layer builders.
-/
def batchNormRunningUpdate {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {c : Nat} (h_c_pos : c > 0)
    (runningMean runningVar batchMean batchVar : RefTy (m := m) (α := α) (.dim c .scalar))
    (momentum : α) :
    m (RefTy (m := m) (α := α) (.dim c .scalar) × RefTy (m := m) (α := α) (.dim c .scalar)) := do
  let sC : Shape := .dim c .scalar
  let _ : Shape.WellFormed sC := ⟨⟨h_c_pos, trivial⟩⟩
  let momT ← const (m := m) (α := α) (s := sC) (Spec.fill momentum sC)
  let oneMinusMomT ← const (m := m) (α := α) (s := sC) (Spec.fill ((1 : α) - momentum) sC)
  let mean' ← add (m := m) (α := α) (s := sC)
    (← mul (m := m) (α := α) (s := sC) runningMean oneMinusMomT)
    (← mul (m := m) (α := α) (s := sC) batchMean momT)
  let var' ← add (m := m) (α := α) (s := sC)
    (← mul (m := m) (α := α) (s := sC) runningVar oneMinusMomT)
    (← mul (m := m) (α := α) (s := sC) batchVar momT)
  pure (mean', var')

/-- Batch normalization using supplied per-channel statistics. -/
def batchNormEval {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch channels : Nat} {spatial : Shape}
    (hWellFormed : (Shape.dim batch (Shape.dim channels spatial)).wellFormed)
    (x : RefTy (m := m) (α := α) (.dim batch (.dim channels spatial)))
    (gamma beta mean var : RefTy (m := m) (α := α) (.dim channels .scalar))
    (ε : α := Numbers.normalizationEpsilon) :
    m (RefTy (m := m) (α := α) (.dim batch (.dim channels spatial))) := do
  let spatialSize := Shape.size spatial
  let inputShape : Shape := .dim batch (.dim channels spatial)
  let flatShape : Shape := .dim batch (.dim channels (.dim spatialSize .scalar))
  let hSpatial := Shape.size_pos_of_well_formed hWellFormed.2.2
  let _ : Shape.WellFormed inputShape := ⟨hWellFormed⟩
  let _ : Shape.WellFormed flatShape :=
    ⟨⟨hWellFormed.1, ⟨hWellFormed.2.1, ⟨hSpatial, trivial⟩⟩⟩⟩
  let xFlat ← reshape (m := m) (α := α) (s₁ := inputShape) (s₂ := flatShape) x
    (Internal.reshapeBatchChannelFlatSize (batch := batch) (channels := channels)
      (spatial := spatial))
  let meanB ← Internal.broadcastChannelToBatchSpatial
    (m := m) (α := α) batch channels spatialSize mean
  let centered ← sub (m := m) (α := α) (s := flatShape) xFlat meanB
  let channelShape : Shape := .dim channels .scalar
  let zero ← const (m := m) (α := α) (s := channelShape)
    (Spec.fill (0 : α) channelShape)
  let varClamped ← max (m := m) (α := α) (s := channelShape) var zero
  let epsT ← const (m := m) (α := α) (s := channelShape) (Spec.fill ε channelShape)
  let denom ← sqrt (m := m) (α := α) (s := channelShape)
    (← add (m := m) (α := α) (s := channelShape) varClamped epsT)
  let invDenom ← inv (m := m) (α := α) (s := channelShape) denom
  let invDenomB ← Internal.broadcastChannelToBatchSpatial
    (m := m) (α := α) batch channels spatialSize invDenom
  let normalized ← mul (m := m) (α := α) (s := flatShape) centered invDenomB
  let gammaB ← Internal.broadcastChannelToBatchSpatial
    (m := m) (α := α) batch channels spatialSize gamma
  let betaB ← Internal.broadcastChannelToBatchSpatial
    (m := m) (α := α) batch channels spatialSize beta
  let yFlat ← add (m := m) (α := α) (s := flatShape)
    (← mul (m := m) (α := α) (s := flatShape) normalized gammaB) betaB
  reshape (m := m) (α := α) (s₁ := flatShape) (s₂ := inputShape) yFlat
    (Internal.reshapeBatchChannelFlatSize (batch := batch) (channels := channels)
      (spatial := spatial)).symm

/-- Batch normalization for one unbatched sample using supplied per-channel statistics. -/
def batchNormSampleEval {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {channels : Nat} {spatial : Shape}
    (hWellFormed : (Shape.dim channels spatial).wellFormed)
    (x : RefTy (m := m) (α := α) (.dim channels spatial))
    (gamma beta mean var : RefTy (m := m) (α := α) (.dim channels .scalar))
    (ε : α := Numbers.normalizationEpsilon) :
    m (RefTy (m := m) (α := α) (.dim channels spatial)) := do
  let spatialSize := Shape.size spatial
  let inputShape : Shape := .dim channels spatial
  let flatShape : Shape := .dim channels (.dim spatialSize .scalar)
  let hSpatial := Shape.size_pos_of_well_formed hWellFormed.2
  let _ : Shape.WellFormed inputShape := ⟨hWellFormed⟩
  let _ : Shape.WellFormed flatShape :=
    ⟨⟨hWellFormed.1, ⟨hSpatial, trivial⟩⟩⟩
  let xFlat ← reshape (m := m) (α := α) (s₁ := inputShape) (s₂ := flatShape) x
    (Internal.reshapeChannelFlatSize (channels := channels) (spatial := spatial))
  let meanB ← Internal.broadcastChannelToSpatial
    (m := m) (α := α) channels spatialSize mean
  let centered ← sub (m := m) (α := α) (s := flatShape) xFlat meanB
  let channelShape : Shape := .dim channels .scalar
  let zero ← const (m := m) (α := α) (s := channelShape)
    (Spec.fill (0 : α) channelShape)
  let varClamped ← max (m := m) (α := α) (s := channelShape) var zero
  let epsT ← const (m := m) (α := α) (s := channelShape) (Spec.fill ε channelShape)
  let denom ← sqrt (m := m) (α := α) (s := channelShape)
    (← add (m := m) (α := α) (s := channelShape) varClamped epsT)
  let invDenom ← inv (m := m) (α := α) (s := channelShape) denom
  let invDenomB ← Internal.broadcastChannelToSpatial
    (m := m) (α := α) channels spatialSize invDenom
  let normalized ← mul (m := m) (α := α) (s := flatShape) centered invDenomB
  let gammaB ← Internal.broadcastChannelToSpatial
    (m := m) (α := α) channels spatialSize gamma
  let betaB ← Internal.broadcastChannelToSpatial
    (m := m) (α := α) channels spatialSize beta
  let yFlat ← add (m := m) (α := α) (s := flatShape)
    (← mul (m := m) (α := α) (s := flatShape) normalized gammaB) betaB
  reshape (m := m) (α := α) (s₁ := flatShape) (s₂ := inputShape) yFlat
    (Internal.reshapeChannelFlatSize (channels := channels) (spatial := spatial)).symm

end Norm

end TorchLean
end Autograd
end Runtime
