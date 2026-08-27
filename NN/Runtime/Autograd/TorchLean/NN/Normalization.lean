/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.NN.Activations

/-!
# Normalization Layers

The channel normalization layers in this module accept an arbitrary spatial shape. A single
definition therefore covers sequence, image, volume, and higher-rank inputs.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor
open Proofs.Autograd.Algebra

namespace NN

/-- Layer normalization over the final axis of a tensor. -/
def layerNorm
    (leading : Shape) (width : Nat)
    {hWidth : width > 0}
    (seedGamma seedBeta : Nat := 0) :
    Layer (leading.appendDim width) (leading.appendDim width) :=
  let gammaShape : Shape := .dim width .scalar
  let betaShape : Shape := .dim width .scalar
  let gamma0 : Tensor Float gammaShape :=
    Torch.Init.tensor (s := gammaShape) (sch := .ones) (seed := seedGamma)
  let beta0 : Tensor Float betaShape :=
    Torch.Init.tensor (s := betaShape) (sch := .zeros) (seed := seedBeta)
  { kind := "LayerNorm"
    stateShapes := [gammaShape, betaShape]
    initState := .cons gamma0 (.cons beta0 .nil)
    runtimeInit := some (.cons .ones (.cons .zeros .nil))
    requiresGrad := #[true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun gamma beta x =>
          (TorchLean.layerNorm (m := m) (α := α)
            (leading := leading) (width := width) hWidth x gamma beta :
            m (RefTy (m := m) (α := α) (leading.appendDim width)))
  }

/-- Root-mean-square normalization over the final axis of a tensor. -/
def rmsNorm
    (leading : Shape) (width : Nat)
    {hWidth : width > 0}
    (seedGamma : Nat := 0) :
    Layer (leading.appendDim width) (leading.appendDim width) :=
  let gammaShape : Shape := .dim width .scalar
  let gamma0 : Tensor Float gammaShape :=
    Torch.Init.tensor (s := gammaShape) (sch := .ones) (seed := seedGamma)
  { kind := "RMSNorm"
    stateShapes := [gammaShape]
    initState := .cons gamma0 .nil
    runtimeInit := some (.cons .ones .nil)
    requiresGrad := #[true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun gamma x =>
          (TorchLean.Norm.rmsNorm (m := m) (α := α)
            (leading := leading) (width := width) hWidth x gamma :
            m (RefTy (m := m) (α := α) (leading.appendDim width)))
  }

/--
Batch normalization over a batch, channel axis, and arbitrary spatial shape.

Training computes statistics over the batch and every spatial position. Evaluation uses the
stored per-channel mean and variance.
-/
def batchNorm
    (batch channels : Nat) (spatial : Shape)
    (hWellFormed : (Shape.dim batch (Shape.dim channels spatial)).wellFormed)
    (seedGamma seedBeta seedMean seedVar : Nat := 0)
    (momentum : Float := 0.1) :
    Layer (.dim batch (.dim channels spatial)) (.dim batch (.dim channels spatial)) :=
  let gammaShape : Shape := .dim channels .scalar
  let betaShape : Shape := .dim channels .scalar
  let meanShape : Shape := .dim channels .scalar
  let varShape : Shape := .dim channels .scalar
  let momentumShape : Shape := Shape.scalar
  let gamma0 : Tensor Float gammaShape :=
    Torch.Init.tensor (s := gammaShape) (sch := .ones) (seed := seedGamma)
  let beta0 : Tensor Float betaShape :=
    Torch.Init.tensor (s := betaShape) (sch := .zeros) (seed := seedBeta)
  let mean0 : Tensor Float meanShape :=
    Torch.Init.tensor (s := meanShape) (sch := .zeros) (seed := seedMean)
  let var0 : Tensor Float varShape :=
    Torch.Init.tensor (s := varShape) (sch := .ones) (seed := seedVar)
  let momentum0 : Tensor Float momentumShape := Tensor.scalar momentum
  { kind := "BatchNorm"
    stateShapes := [gammaShape, betaShape, meanShape, varShape, momentumShape]
    initState := .cons gamma0 (.cons beta0 (.cons mean0 (.cons var0 (.cons momentum0 .nil))))
    runtimeInit := some (.cons .ones (.cons .zeros (.cons .zeros (.cons .ones
      (.cons (.flat (FloatArray.mk #[momentum])) .nil)))))
    requiresGrad := #[true, true, false, false, false]
    updateBuffers := some (fun mode {_α} _ _ ps x => do
      match mode, ps with
      | .eval, _ => pure ps
      | .train, .cons gamma (.cons beta (.cons runningMean (.cons runningVar
          (.cons momentumT .nil)))) =>
          let (batchMean, batchVar) := batchChannelStats x
          let sampleCount := batch * Shape.size spatial
          let runningBatchVar := unbiasedRunningVariance batchVar sampleCount
          let nextMean := updateRunning runningMean batchMean momentumT
          let nextVar := updateRunning runningVar runningBatchVar momentumT
          pure (.cons gamma (.cons beta (.cons nextMean (.cons nextVar (.cons momentumT .nil)))))
      | .train, _ => pure ps)
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        fun gamma beta mean var _momentum x =>
          match mode with
          | .train =>
              TorchLean.Norm.batchNormTrain (m := m) (α := α)
                hWellFormed x gamma beta
          | .eval =>
              TorchLean.Norm.batchNormEval (m := m) (α := α)
                hWellFormed x gamma beta mean var
  }

/-- Normalize every sample and channel independently over an arbitrary spatial shape. -/
def instanceNorm
    (batch channels : Nat) (spatial : Shape)
    (hWellFormed : (Shape.dim batch (Shape.dim channels spatial)).wellFormed)
    (seedGamma seedBeta : Nat := 0) :
    Layer (.dim batch (.dim channels spatial)) (.dim batch (.dim channels spatial)) :=
  let gammaShape : Shape := .dim channels .scalar
  let betaShape : Shape := .dim channels .scalar
  let gamma0 : Tensor Float gammaShape :=
    Torch.Init.tensor (s := gammaShape) (sch := .ones) (seed := seedGamma)
  let beta0 : Tensor Float betaShape :=
    Torch.Init.tensor (s := betaShape) (sch := .zeros) (seed := seedBeta)
  { kind := "InstanceNorm"
    stateShapes := [gammaShape, betaShape]
    initState := .cons gamma0 (.cons beta0 .nil)
    runtimeInit := some (.cons .ones (.cons .zeros .nil))
    requiresGrad := #[true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun gamma beta x =>
          TorchLean.Norm.instanceNorm (m := m) (α := α)
            hWellFormed x gamma beta
  }

/-- Normalize groups of channels over an arbitrary spatial shape. -/
def groupNorm
    (batch channels groups : Nat) (spatial : Shape)
    (hWellFormed : (Shape.dim batch (Shape.dim channels spatial)).wellFormed)
    (hGroups : groups > 0) (hGroupsLe : channels ≥ groups)
    (hDiv : channels % groups = 0)
    (seedGamma seedBeta : Nat := 0) :
    Layer (.dim batch (.dim channels spatial)) (.dim batch (.dim channels spatial)) :=
  let gammaShape : Shape := .dim channels .scalar
  let betaShape : Shape := .dim channels .scalar
  let gamma0 : Tensor Float gammaShape :=
    Torch.Init.tensor (s := gammaShape) (sch := .ones) (seed := seedGamma)
  let beta0 : Tensor Float betaShape :=
    Torch.Init.tensor (s := betaShape) (sch := .zeros) (seed := seedBeta)
  { kind := s!"GroupNorm(groups={groups})"
    stateShapes := [gammaShape, betaShape]
    initState := .cons gamma0 (.cons beta0 .nil)
    runtimeInit := some (.cons .ones (.cons .zeros .nil))
    requiresGrad := #[true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun gamma beta x =>
          TorchLean.Norm.groupNorm (m := m) (α := α)
            hWellFormed hGroups hGroupsLe hDiv x gamma beta
  }

end NN

end TorchLean
end Autograd
end Runtime
