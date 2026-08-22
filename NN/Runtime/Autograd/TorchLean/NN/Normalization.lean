/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.NN.Activations

/-!
# TorchLean NN: Normalization Layers
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
  let gamma0 : Tensor Float gammaShape := Torch.Init.tensor (s := gammaShape) (sch := .ones) (seed
    := seedGamma)
  let beta0 : Tensor Float betaShape := Torch.Init.tensor (s := betaShape) (sch := .zeros) (seed :=
    seedBeta)
  { kind := "LayerNorm"
    stateShapes := [gammaShape, betaShape]
    initState := .cons gamma0 (.cons beta0 .nil)
    runtimeInit := some (.cons .ones (.cons .zeros .nil))
    requiresGrad := [true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun gamma beta x =>
          (do
            let rows := Shape.size leading
            let xRows ← TorchLean.reshape (m := m) (α := α)
              (s₁ := leading.appendDim width)
              (s₂ := .dim rows (.dim width .scalar)) x (by
                simp [rows, Shape.size_appendDim, Shape.size])
            let yRows ← TorchLean.layerNorm (m := m) (α := α)
              (rows := rows) (width := width) hWidth xRows gamma beta
            TorchLean.reshape (m := m) (α := α)
              (s₁ := .dim rows (.dim width .scalar))
              (s₂ := leading.appendDim width) yRows (by
                simp [rows, Shape.size_appendDim, Shape.size]) :
            m (RefTy (m := m) (α := α) (leading.appendDim width)))
  }

/-- Root-mean-square normalization over the final axis of a tensor. -/
def rmsNorm
    (leading : Shape) (width : Nat)
    {hWidth : width > 0}
    (seedGamma : Nat := 0) :
    Layer (leading.appendDim width) (leading.appendDim width) :=
  let gammaShape : Shape := .dim width .scalar
  let gamma0 : Tensor Float gammaShape := Torch.Init.tensor (s := gammaShape) (sch := .ones) (seed
    := seedGamma)
  { kind := "RMSNorm"
    stateShapes := [gammaShape]
    initState := .cons gamma0 .nil
    runtimeInit := some (.cons .ones .nil)
    requiresGrad := [true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun gamma x =>
          (do
            let rows := Shape.size leading
            let xRows ← TorchLean.reshape (m := m) (α := α)
              (s₁ := leading.appendDim width)
              (s₂ := .dim rows (.dim width .scalar)) x (by
                simp [rows, Shape.size_appendDim, Shape.size])
            let yRows ← TorchLean.Norm.rmsNorm (m := m) (α := α)
              (rows := rows) (width := width) hWidth xRows gamma
            TorchLean.reshape (m := m) (α := α)
              (s₁ := .dim rows (.dim width .scalar))
              (s₂ := leading.appendDim width) yRows (by
                simp [rows, Shape.size_appendDim, Shape.size]) :
            m (RefTy (m := m) (α := α) (leading.appendDim width)))
  }

/--
Batch normalization for an unbatched `C×H×W` tensor (channel-first).

This uses the current activation’s per-channel statistics (over spatial axes) and applies learnable
affine parameters `gamma`/`beta`.

PyTorch analogy: `torch.nn.BatchNorm2d(channels)` in training mode (applied to a single sample).
-/
def batchNormChannelFirst
    (channels height width : Nat)
    {h_c : channels > 0} {h_h : height > 0} {h_w : width > 0}
    (seedGamma seedBeta : Nat := 0) :
    Layer (.dim channels (.dim height (.dim width .scalar))) (.dim channels (.dim height (.dim width .scalar)))
      :=
  let gammaShape : Shape := .dim channels .scalar
  let betaShape : Shape := .dim channels .scalar
  let gamma0 : Tensor Float gammaShape := Torch.Init.tensor (s := gammaShape) (sch := .ones) (seed
    := seedGamma)
  let beta0 : Tensor Float betaShape := Torch.Init.tensor (s := betaShape) (sch := .zeros) (seed :=
    seedBeta)
  { kind := "BatchNorm2d"
    stateShapes := [gammaShape, betaShape]
    initState := .cons gamma0 (.cons beta0 .nil)
    runtimeInit := some (.cons .ones (.cons .zeros .nil))
    requiresGrad := [true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun gamma beta x =>
          TorchLean.batchNormChannelFirst (m := m) (α := α)
            (channels := channels) (height := height) (width := width) (h_c := h_c) (h_h := h_h)
              (h_w := h_w)
            x gamma beta
  }

/--
Batch normalization for a `C×H×W` tensor in eval mode using provided running statistics.

Parameters include `gamma`/`beta` plus fixed `mean`/`var` buffers.

PyTorch analogy: `torch.nn.BatchNorm2d` in eval mode (uses `running_mean` / `running_var`).
-/
def batchNormChannelFirstEval
    (channels height width : Nat)
    {h_c : channels > 0} {h_h : height > 0} {h_w : width > 0}
    (seedGamma seedBeta seedMean seedVar : Nat := 0) :
    Layer (.dim channels (.dim height (.dim width .scalar))) (.dim channels (.dim height (.dim width .scalar)))
      :=
  let gammaShape : Shape := .dim channels .scalar
  let betaShape : Shape := .dim channels .scalar
  let meanShape : Shape := .dim channels .scalar
  let varShape : Shape := .dim channels .scalar
  let gamma0 : Tensor Float gammaShape := Torch.Init.tensor (s := gammaShape) (sch := .ones) (seed
    := seedGamma)
  let beta0 : Tensor Float betaShape := Torch.Init.tensor (s := betaShape) (sch := .zeros) (seed :=
    seedBeta)
  let mean0 : Tensor Float meanShape := Torch.Init.tensor (s := meanShape) (sch := .zeros) (seed :=
    seedMean)
  let var0 : Tensor Float varShape := Torch.Init.tensor (s := varShape) (sch := .ones) (seed :=
    seedVar)
  { kind := "BatchNorm2d(eval)"
    stateShapes := [gammaShape, betaShape, meanShape, varShape]
    initState := .cons gamma0 (.cons beta0 (.cons mean0 (.cons var0 .nil)))
    runtimeInit := some (.cons .ones (.cons .zeros (.cons .zeros (.cons .ones .nil))))
    requiresGrad := [true, true, false, false]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun gamma beta mean var x =>
          TorchLean.Norm.batchNorm2dChwEval (m := m) (α := α)
            (c := channels) (h := height) (w := width)
            h_c h_h h_w x gamma beta mean var
  }

/--
Batch normalization for `C×H×W` with explicit `Mode` and running-statistics buffers.

- In `Mode.train`, computes per-channel batch stats and updates `(runningMean, runningVar)` using
  `momentum`.
- In `Mode.eval`, normalizes using the stored running buffers.

PyTorch analogy: `torch.nn.BatchNorm2d(channels, momentum=...)` with `.train()` / `.eval()`
  behavior.
-/
def batchNormChannelFirstMode
    (channels height width : Nat)
    {h_c : channels > 0} {h_h : height > 0} {h_w : width > 0}
    (seedGamma seedBeta seedMean seedVar : Nat := 0)
    (momentum : Float := 0.1) :
    Layer (.dim channels (.dim height (.dim width .scalar))) (.dim channels (.dim height (.dim width .scalar)))
      :=
  let gammaShape : Shape := .dim channels .scalar
  let betaShape : Shape := .dim channels .scalar
  let meanShape : Shape := .dim channels .scalar
  let varShape : Shape := .dim channels .scalar
  let momentumShape : Shape := Shape.scalar
  let gamma0 : Tensor Float gammaShape := Torch.Init.tensor (s := gammaShape) (sch := .ones) (seed
    := seedGamma)
  let beta0 : Tensor Float betaShape := Torch.Init.tensor (s := betaShape) (sch := .zeros) (seed :=
    seedBeta)
  let mean0 : Tensor Float meanShape := Torch.Init.tensor (s := meanShape) (sch := .zeros) (seed :=
    seedMean)
  let var0 : Tensor Float varShape := Torch.Init.tensor (s := varShape) (sch := .ones) (seed :=
    seedVar)
  let momentum0 : Tensor Float momentumShape := Tensor.scalar momentum
  { kind := "BatchNorm2d"
    stateShapes := [gammaShape, betaShape, meanShape, varShape, momentumShape]
    initState := .cons gamma0 (.cons beta0 (.cons mean0 (.cons var0 (.cons momentum0 .nil))))
    runtimeInit := some (.cons .ones (.cons .zeros (.cons .zeros (.cons .ones
      (.cons (.flat (FloatArray.mk #[momentum])) .nil)))))
    requiresGrad := [true, true, false, false, false]
    updateBuffers := some (fun mode {_α} _ _ ps x => do
      match mode, ps with
      | .eval, _ => pure ps
      | .train, .cons gamma (.cons beta (.cons runningMean (.cons runningVar (.cons momentumT
        .nil)))) =>
          let (batchMean, batchVar) := chwBatchStats x
          let runningBatchVar := unbiasedRunningVariance batchVar (height * width)
          let nextMean := updateRunningVec runningMean batchMean momentumT
          let nextVar := updateRunningVec runningVar runningBatchVar momentumT
          pure (.cons gamma (.cons beta (.cons nextMean (.cons nextVar (.cons momentumT .nil)))))
      | .train, _ => pure ps
    )
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        fun gamma beta mean var _momentum x =>
          match mode with
          | .train =>
              TorchLean.batchNormChannelFirst (m := m) (α := α)
                (channels := channels) (height := height) (width := width)
                (h_c := h_c) (h_h := h_h) (h_w := h_w) x gamma beta
          | .eval =>
              TorchLean.Norm.batchNorm2dChwEval (m := m) (α := α)
                (c := channels) (h := height) (w := width)
                h_c h_h h_w x gamma beta mean var
  }

/--
Instance normalization for `N×C×H×W` tensors.

This normalizes each sample independently (per-channel), then applies learnable affine parameters
`gamma`/`beta`.

PyTorch analogy: `torch.nn.InstanceNorm2d(c, affine=True)` (with `NCHW` layout).
-/
def instanceNorm2dNchw
    (n c h w : Nat)
    {h_n_pos : n > 0} {h_c_pos : c > 0} {h_h_pos : h > 0} {h_w_pos : w > 0}
    (seedGamma seedBeta : Nat := 0) :
    Layer (.dim n (.dim c (.dim h (.dim w .scalar)))) (.dim n (.dim c (.dim h (.dim w .scalar)))) :=
  let gammaShape : Shape := .dim c .scalar
  let betaShape : Shape := .dim c .scalar
  let gamma0 : Tensor Float gammaShape := Torch.Init.tensor (s := gammaShape) (sch := .ones) (seed
    := seedGamma)
  let beta0 : Tensor Float betaShape := Torch.Init.tensor (s := betaShape) (sch := .zeros) (seed :=
    seedBeta)
  { kind := "InstanceNorm2d"
    stateShapes := [gammaShape, betaShape]
    initState := .cons gamma0 (.cons beta0 .nil)
    runtimeInit := some (.cons .ones (.cons .zeros .nil))
    requiresGrad := [true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun gamma beta x =>
          TorchLean.Norm.instanceNorm2dNchw (m := m) (α := α)
            (n := n) (c := c) (h := h) (w := w)
            h_n_pos h_c_pos h_h_pos h_w_pos
            x gamma beta
  }

/--
Group normalization for `N×C×H×W` tensors.

Channels are split into `groups` groups (requiring `c % groups = 0`), normalization is performed per
group, then learnable affine parameters `gamma`/`beta` are applied.

PyTorch analogy: `torch.nn.GroupNorm(groups, c)` (with `NCHW` layout).
-/
def groupNorm2dNchw
    (n c h w groups : Nat)
    {h_n_pos : n > 0} {h_c_pos : c > 0} {h_h_pos : h > 0} {h_w_pos : w > 0} {h_g_pos : groups > 0}
    (h_ge : c ≥ groups) (h_div : c % groups = 0)
    (seedGamma seedBeta : Nat := 0) :
    Layer (.dim n (.dim c (.dim h (.dim w .scalar)))) (.dim n (.dim c (.dim h (.dim w .scalar)))) :=
  let gammaShape : Shape := .dim c .scalar
  let betaShape : Shape := .dim c .scalar
  let gamma0 : Tensor Float gammaShape := Torch.Init.tensor (s := gammaShape) (sch := .ones) (seed
    := seedGamma)
  let beta0 : Tensor Float betaShape := Torch.Init.tensor (s := betaShape) (sch := .zeros) (seed :=
    seedBeta)
  { kind := s!"GroupNorm2d(groups={groups})"
    stateShapes := [gammaShape, betaShape]
    initState := .cons gamma0 (.cons beta0 .nil)
    runtimeInit := some (.cons .ones (.cons .zeros .nil))
    requiresGrad := [true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun gamma beta x =>
          TorchLean.Norm.groupNorm2dNchw (m := m) (α := α)
            (n := n) (c := c) (h := h) (w := w) (groups := groups)
            h_n_pos h_c_pos h_h_pos h_w_pos h_g_pos h_ge h_div
            x gamma beta
  }

/--
Batch normalization training behavior for `N×C×H×W` tensors (no running buffers).

This computes batch statistics across `(N, H, W)` and applies learnable affine parameters
`gamma`/`beta`.

PyTorch analogy: `torch.nn.BatchNorm2d(c)` in training mode (stat computation).
-/
def batchNorm2dNchw
    (n c h w : Nat)
    {h_n_pos : n > 0} {h_c_pos : c > 0} {h_h_pos : h > 0} {h_w_pos : w > 0}
    (seedGamma seedBeta : Nat := 0) :
    Layer (.dim n (.dim c (.dim h (.dim w .scalar)))) (.dim n (.dim c (.dim h (.dim w .scalar)))) :=
  let gammaShape : Shape := .dim c .scalar
  let betaShape : Shape := .dim c .scalar
  let gamma0 : Tensor Float gammaShape := Torch.Init.tensor (s := gammaShape) (sch := .ones) (seed
    := seedGamma)
  let beta0 : Tensor Float betaShape := Torch.Init.tensor (s := betaShape) (sch := .zeros) (seed :=
    seedBeta)
  { kind := "BatchNorm2d"
    stateShapes := [gammaShape, betaShape]
    initState := .cons gamma0 (.cons beta0 .nil)
    runtimeInit := some (.cons .ones (.cons .zeros .nil))
    requiresGrad := [true, true]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ =>
        fun gamma beta x =>
          TorchLean.Norm.batchNorm2dNchwTrain (m := m) (α := α)
            (n := n) (c := c) (h := h) (w := w)
            h_n_pos h_c_pos h_h_pos h_w_pos
            x gamma beta
  }

/--
Batch normalization for `N×C×H×W` with explicit `Mode` and running-statistics buffers.

Parameters include `gamma`, `beta`, running `mean`/`var` buffers, and a momentum scalar:
- in `Mode.train`, compute batch stats and update running buffers,
- in `Mode.eval`, normalize using the running buffers.

PyTorch analogy: `torch.nn.BatchNorm2d(c, momentum=...)` with `.train()` / `.eval()` behavior.
-/
def batchNorm2dNchwMode
    (n c h w : Nat)
    {h_n_pos : n > 0} {h_c_pos : c > 0} {h_h_pos : h > 0} {h_w_pos : w > 0}
    (seedGamma seedBeta seedMean seedVar : Nat := 0)
    (momentum : Float := 0.1) :
    Layer (.dim n (.dim c (.dim h (.dim w .scalar)))) (.dim n (.dim c (.dim h (.dim w .scalar)))) :=
  let gammaShape : Shape := .dim c .scalar
  let betaShape : Shape := .dim c .scalar
  let meanShape : Shape := .dim c .scalar
  let varShape : Shape := .dim c .scalar
  let momentumShape : Shape := Shape.scalar
  let gamma0 : Tensor Float gammaShape := Torch.Init.tensor (s := gammaShape) (sch := .ones) (seed
    := seedGamma)
  let beta0 : Tensor Float betaShape := Torch.Init.tensor (s := betaShape) (sch := .zeros) (seed :=
    seedBeta)
  let mean0 : Tensor Float meanShape := Torch.Init.tensor (s := meanShape) (sch := .zeros) (seed :=
    seedMean)
  let var0 : Tensor Float varShape := Torch.Init.tensor (s := varShape) (sch := .ones) (seed :=
    seedVar)
  let momentum0 : Tensor Float momentumShape := Tensor.scalar momentum
  { kind := "BatchNorm2d"
    stateShapes := [gammaShape, betaShape, meanShape, varShape, momentumShape]
    initState := .cons gamma0 (.cons beta0 (.cons mean0 (.cons var0 (.cons momentum0 .nil))))
    runtimeInit := some (.cons .ones (.cons .zeros (.cons .zeros (.cons .ones
      (.cons (.flat (FloatArray.mk #[momentum])) .nil)))))
    requiresGrad := [true, true, false, false, false]
    updateBuffers := some (fun mode {_α} _ _ ps x => do
      match mode, ps with
      | .eval, _ => pure ps
      | .train, .cons gamma (.cons beta (.cons runningMean (.cons runningVar (.cons momentumT
        .nil)))) =>
          let (batchMean, batchVar) := nchwBatchStats x
          let runningBatchVar := unbiasedRunningVariance batchVar (n * h * w)
          let nextMean := updateRunningVec runningMean batchMean momentumT
          let nextVar := updateRunningVec runningVar runningBatchVar momentumT
          pure (.cons gamma (.cons beta (.cons nextMean (.cons nextVar (.cons momentumT .nil)))))
      | .train, _ => pure ps
    )
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        fun gamma beta mean var _momentum x =>
          match mode with
          | .train =>
              TorchLean.Norm.batchNorm2dNchwTrain (m := m) (α := α)
                (n := n) (c := c) (h := h) (w := w)
                h_n_pos h_c_pos h_h_pos h_w_pos
                x gamma beta
          | .eval =>
              TorchLean.Norm.batchNorm2dNchwEval (m := m) (α := α)
                (n := n) (c := c) (h := h) (w := w)
                h_n_pos h_c_pos h_h_pos h_w_pos
                x gamma beta mean var
  }
end NN

end TorchLean
end Autograd
end Runtime
