/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Normalization.Core

/-!
# Batch Normalization

Generic channel-first BatchNorm semantics together with the JVP and VJP used by TorchLean's
concrete 2D graph operator. Training-time statistics and inference-time running statistics remain
separate mathematical operations.
-/

@[expose] public section

namespace Spec
open Tensor
open Numbers

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-- Repeat a channel vector over every spatial position of a channel-first tensor. -/
def broadcastChannelFirst {channels : Nat} (sSpatial : Shape)
    (x : Tensor α (.dim channels .scalar)) : Tensor α (.dim channels sSpatial) :=
  let spatialSize := Spec.Shape.size sSpatial
  let sFlat : Shape := .dim channels (.dim spatialSize .scalar)
  let expanded : Tensor α sFlat := broadcastAfterSum sFlat 1 x
  have hSize : Spec.Shape.size sFlat = Spec.Shape.size (.dim channels sSpatial) := by
    simp [sFlat, spatialSize, Spec.Shape.size]
  reshapeSpec expanded hSize

/-
  Batch Normalization (spec layer)

  TorchLean models *pure* (stateless) BatchNorm operators:

  - `batchNorm`: "training-mode" normalization using statistics computed from the current input
    (TorchLean does not model the running-statistics update),
  - `batchNorm_inference`: inference-time normalization using fixed running mean/variance.
-/

/--
Stateless BatchNorm for channel-first tensors of shape `.dim channels sSpatial`.

This computes per-channel mean/variance over the `sSpatial` axes and applies:

`y = ((x - mean) / sqrt(var + eps)) * gamma + beta`.

PyTorch analogy: `torch.nn.BatchNorm{1,2,3}d` in training mode on an input with batch size `N=1`.
TorchLean does **not** model the running-statistics update here.
-/
def batchNorm
  {channels : Nat} {sSpatial : Shape}
  (x : Tensor α (.dim channels sSpatial))
  (gamma : Tensor α (.dim channels .scalar))
  (beta : Tensor α (.dim channels .scalar))
  (epsilon : α := Numbers.normalizationEpsilon)
  [Shape.WellFormed (.dim channels sSpatial)] :
  Tensor α (.dim channels sSpatial) :=
  let spatialSize : Nat := Spec.Shape.size sSpatial
  let s_flat : Shape := .dim channels (.dim spatialSize .scalar)
  have h_reshape : Spec.Shape.size (.dim channels sSpatial) = Spec.Shape.size s_flat := by
    simp [s_flat, spatialSize, Spec.Shape.size]
  let x2 : Tensor α s_flat := reshapeSpec x h_reshape
  have hwf_x : (Shape.dim channels sSpatial).wellFormed := Shape.WellFormed.proof
  have h_channels : channels > 0 := hwf_x.1
  have h_spatial_wf : sSpatial.wellFormed := hwf_x.2
  have h_spatialSize : spatialSize > 0 := by
    simpa [spatialSize] using Shape.size_pos_of_well_formed (s := sSpatial) h_spatial_wf
  letI : Shape.WellFormed s_flat := ⟨⟨h_channels, ⟨h_spatialSize, trivial⟩⟩⟩
  let h_rank : Spec.Shape.rank s_flat > 0 := by simp [s_flat, Spec.Shape.rank]
  let h_valid : Shape.HasNonemptyAxis (Spec.Shape.rank s_flat - 1) s_flat :=
    Shape.inferNonemptyLastAxis h_rank
  let mean : Tensor α (.dim channels .scalar) := reduceMeanLast x2 h_valid
  let centered := subSpec x2 (broadcastAfterSum s_flat 1 mean)
  let centered_sq := mulSpec centered centered
  let varianceRaw : Tensor α (.dim channels .scalar) := reduceMeanLast centered_sq h_valid
  let variance := maxSpec varianceRaw (fill 0 (.dim channels .scalar))
  let meanB := broadcastChannelFirst sSpatial mean
  let varianceB := broadcastChannelFirst sSpatial variance
  let gammaB := broadcastChannelFirst sSpatial gamma
  let betaB := broadcastChannelFirst sSpatial beta
  let centered := subSpec x meanB
  let std := sqrtSpec (addSpec varianceB (fill epsilon (.dim channels sSpatial)))
  addSpec (mulSpec (divSpec centered std) gammaB) betaB

/-- `batchNorm` specialized to a single channel-first image `(C,H,W)`. -/
def batchNorm2d
  {channels height width : Nat}
  (x : Tensor α (.dim channels (.dim height (.dim width .scalar))))
  (gamma : Tensor α (.dim channels .scalar))
  (beta : Tensor α (.dim channels .scalar))
  (h_c : channels > 0 := by norm_num)
  (h_h : height > 0 := by norm_num)
  (h_w : width > 0 := by norm_num)
  (epsilon : α := Numbers.normalizationEpsilon) :
  Tensor α (.dim channels (.dim height (.dim width .scalar))) :=
  letI : Shape.WellFormed (.dim channels (.dim height (.dim width .scalar))) :=
    ⟨⟨h_c, ⟨h_h, ⟨h_w, trivial⟩⟩⟩⟩
  batchNorm (x := x) (gamma := gamma) (beta := beta) (epsilon := epsilon)

/--
Forward-mode JVP for `batchNorm2d`.

TorchLean's stateless BatchNorm2d computes one set of statistics per channel over the spatial
grid. The input tangent therefore uses the same closed-form normalization differential as
LayerNorm, but with the mean taken over `(height,width)` for each channel:

`dxhat = inv_std * (dx - mean(dx) - xhat * mean(dx*xhat))`.

Affine tangents contribute `xhat * dgamma + dbeta` channel-wise.
-/
def batchNorm2dJvp
  {channels height width : Nat}
  (x tangent : Tensor α (.dim channels (.dim height (.dim width .scalar))))
  (gamma dgamma _beta dbeta : Tensor α (.dim channels .scalar))
  (_h_c : channels > 0 := by norm_num)
  (_h_h : height > 0 := by norm_num)
  (_h_w : width > 0 := by norm_num)
  (epsilon : α := Numbers.normalizationEpsilon) :
  Tensor α (.dim channels (.dim height (.dim width .scalar))) :=

  let spatial_size := height * width
  let nScalar : Tensor α Shape.scalar := Tensor.scalar (spatial_size : α)

  Tensor.dim (fun c =>
    let channel_data := getAtSpec x c
    let channel_tangent := getAtSpec tangent c
    let channel_gamma := getAtSpec gamma c
    let channel_dgamma := getAtSpec dgamma c
    let channel_dbeta := getAtSpec dbeta c

    let channel_sum :=
      (List.finRange height).foldl (fun acc_h i =>
        (List.finRange width).foldl (fun acc_w j =>
          if h_i : i < height then
            if h_j : j < width then
              addSpec acc_w (getAtSpec (getAtSpec channel_data ⟨i, h_i⟩) ⟨j, h_j⟩)
            else acc_w
          else acc_w
        ) acc_h
      ) (Tensor.scalar 0)
    let mean := divSpec channel_sum nScalar

    let variance_sum :=
      (List.finRange height).foldl (fun acc_h i =>
        (List.finRange width).foldl (fun acc_w j =>
          if h_i : i < height then
            if h_j : j < width then
              let val := getAtSpec (getAtSpec channel_data ⟨i, h_i⟩) ⟨j, h_j⟩
              let diff := subSpec val mean
              addSpec acc_w (mulSpec diff diff)
            else acc_w
          else acc_w
        ) acc_h
      ) (Tensor.scalar 0)

    let varianceRaw := divSpec variance_sum nScalar
    let variance := maxSpec varianceRaw (Tensor.scalar 0)
    let std := sqrtSpec (addSpec variance (Tensor.scalar epsilon))
    let inv_std := divSpec (Tensor.scalar 1) std

    let tangent_sum :=
      (List.finRange height).foldl (fun acc_h i =>
        (List.finRange width).foldl (fun acc_w j =>
          if h_i : i < height then
            if h_j : j < width then
              addSpec acc_w (getAtSpec (getAtSpec channel_tangent ⟨i, h_i⟩) ⟨j, h_j⟩)
            else acc_w
          else acc_w
        ) acc_h
      ) (Tensor.scalar 0)
    let mean_tangent := divSpec tangent_sum nScalar

    let tangent_norm_sum :=
      (List.finRange height).foldl (fun acc_h i =>
        (List.finRange width).foldl (fun acc_w j =>
          if h_i : i < height then
            if h_j : j < width then
              let val := getAtSpec (getAtSpec channel_data ⟨i, h_i⟩) ⟨j, h_j⟩
              let dx := getAtSpec (getAtSpec channel_tangent ⟨i, h_i⟩) ⟨j, h_j⟩
              let xHat := mulSpec (subSpec val mean) inv_std
              addSpec acc_w (mulSpec dx xHat)
            else acc_w
          else acc_w
        ) acc_h
      ) (Tensor.scalar 0)
    let mean_tangent_norm := divSpec tangent_norm_sum nScalar

    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        let val := getAtSpec (getAtSpec channel_data i) j
        let dx := getAtSpec (getAtSpec channel_tangent i) j
        let xHat := mulSpec (subSpec val mean) inv_std
        let dnorm :=
          mulSpec inv_std
            (subSpec (subSpec dx mean_tangent) (mulSpec xHat mean_tangent_norm))
        addSpec (addSpec (mulSpec dnorm channel_gamma) (mulSpec xHat channel_dgamma))
          channel_dbeta))
  )

-- Batch normalization backward pass for channel-first format
/--
Backward/VJP for `batchNorm2d`.

Returns `(dx, dGamma, dBeta)`. This matches the shape of gradients you expect from a PyTorch-style
BatchNorm2d, but note that our forward is the per-image variant (no explicit batch dimension and no
running statistics).
-/
def batchNorm2dBackward
  {channels height width : Nat}
  (x : Tensor α (.dim channels (.dim height (.dim width .scalar))))
  (gamma : Tensor α (.dim channels .scalar))
  (grad_output : Tensor α (.dim channels (.dim height (.dim width .scalar))))
  (_h_c : channels > 0 := by norm_num)
  (_h_h : height > 0 := by norm_num)
  (_h_w : width > 0 := by norm_num)
  (epsilon : α := Numbers.normalizationEpsilon) :
  (Tensor α (.dim channels (.dim height (.dim width .scalar))) ×  -- ∂L/∂x
   Tensor α (.dim channels .scalar) ×           -- ∂L/∂gamma
   Tensor α (.dim channels .scalar)) :=         -- ∂L/∂beta

  let spatial_size := height * width

  -- Process each channel independently
  let grad_x := Tensor.dim (fun c =>
    let channel_data := getAtSpec x c
    let channel_grad := getAtSpec grad_output c
    let channel_gamma := getAtSpec gamma c

    -- Recompute forward pass statistics (in practice, these would be cached)
    let channel_sum :=
      (List.finRange height).foldl (fun acc_h i =>
        (List.finRange width).foldl (fun acc_w j =>
          if h_i : i < height then
            if h_j : j < width then
              addSpec acc_w (getAtSpec (getAtSpec channel_data ⟨i, h_i⟩) ⟨j, h_j⟩)
            else acc_w
          else acc_w
        ) acc_h
      ) (Tensor.scalar 0)

    let mean := divSpec channel_sum (Tensor.scalar spatial_size)

    let variance_sum :=
      (List.finRange height).foldl (fun acc_h i =>
        (List.finRange width).foldl (fun acc_w j =>
          if h_i : i < height then
            if h_j : j < width then
              let val := getAtSpec (getAtSpec channel_data ⟨i, h_i⟩) ⟨j, h_j⟩
              let diff := subSpec val mean
              addSpec acc_w (mulSpec diff diff)
            else acc_w
          else acc_w
        ) acc_h
      ) (Tensor.scalar 0)

    let varianceRaw := divSpec variance_sum (Tensor.scalar spatial_size)
    let variance := maxSpec varianceRaw (Tensor.scalar 0)
    let std := sqrtSpec (addSpec variance (Tensor.scalar epsilon))
    let inv_std := divSpec (Tensor.scalar 1) std

    -- Standard normalization gradient (equivalent to layer-norm over the spatial grid).
    let dXhat_sum :=
      (List.finRange height).foldl (fun acc_h i =>
        (List.finRange width).foldl (fun acc_w j =>
          if h_i : i < height then
            if h_j : j < width then
              let grad_val := getAtSpec (getAtSpec channel_grad ⟨i, h_i⟩) ⟨j, h_j⟩
              let dXhat := mulSpec grad_val channel_gamma
              addSpec acc_w dXhat
            else acc_w
          else acc_w
        ) acc_h
      ) (Tensor.scalar 0)

    let dXhatXhat_sum :=
      (List.finRange height).foldl (fun acc_h i =>
        (List.finRange width).foldl (fun acc_w j =>
          if h_i : i < height then
            if h_j : j < width then
              let val := getAtSpec (getAtSpec channel_data ⟨i, h_i⟩) ⟨j, h_j⟩
              let grad_val := getAtSpec (getAtSpec channel_grad ⟨i, h_i⟩) ⟨j, h_j⟩
              let centered := subSpec val mean
              let xHat := mulSpec centered inv_std
              let dXhat := mulSpec grad_val channel_gamma
              addSpec acc_w (mulSpec dXhat xHat)
            else acc_w
          else acc_w
        ) acc_h
      ) (Tensor.scalar 0)

    let nScalar : Tensor α Shape.scalar := Tensor.scalar (spatial_size : α)
    let inv_n : Tensor α Shape.scalar := divSpec (Tensor.scalar (1 : α)) nScalar

    Tensor.dim (fun i =>
      Tensor.dim (fun j =>
        let val := getAtSpec (getAtSpec channel_data i) j
        let grad_val := getAtSpec (getAtSpec channel_grad i) j
        let centered := subSpec val mean
        let xHat := mulSpec centered inv_std
        let dXhat := mulSpec grad_val channel_gamma
        let term :=
          subSpec (subSpec (mulSpec nScalar dXhat) dXhat_sum)
            (mulSpec xHat dXhatXhat_sum)
        mulSpec (mulSpec term inv_std) inv_n
      )
    )
  )

  -- Compute gamma gradients (sum over spatial dimensions for each channel)
  let grad_gamma := Tensor.dim (fun c =>
    let channel_data := getAtSpec x c
    let channel_grad := getAtSpec grad_output c

    -- Recompute mean and std for this channel
    let channel_sum :=
      (List.finRange height).foldl (fun acc_h i =>
        (List.finRange width).foldl (fun acc_w j =>
          if h_i : i < height then
            if h_j : j < width then
              addSpec acc_w (getAtSpec (getAtSpec channel_data ⟨i, h_i⟩) ⟨j, h_j⟩)
            else acc_w
          else acc_w
        ) acc_h
      ) (Tensor.scalar 0)

    let mean := divSpec channel_sum (Tensor.scalar spatial_size)

    let variance_sum :=
      (List.finRange height).foldl (fun acc_h i =>
        (List.finRange width).foldl (fun acc_w j =>
          if h_i : i < height then
            if h_j : j < width then
              let val := getAtSpec (getAtSpec channel_data ⟨i, h_i⟩) ⟨j, h_j⟩
              let diff := subSpec val mean
              addSpec acc_w (mulSpec diff diff)
            else acc_w
          else acc_w
        ) acc_h
      ) (Tensor.scalar 0)

    let varianceRaw := divSpec variance_sum (Tensor.scalar spatial_size)
    let variance := maxSpec varianceRaw (Tensor.scalar 0)
    let std := sqrtSpec (addSpec variance (Tensor.scalar epsilon))
    let inv_std := divSpec (Tensor.scalar 1) std

    -- Sum grad_output * normalized_input for this channel
    (List.finRange height).foldl (fun acc_h i =>
      (List.finRange width).foldl (fun acc_w j =>
        if h_i : i < height then
          if h_j : j < width then
            let val := getAtSpec (getAtSpec channel_data ⟨i, h_i⟩) ⟨j, h_j⟩
            let grad_val := getAtSpec (getAtSpec channel_grad ⟨i, h_i⟩) ⟨j, h_j⟩
            let normalized := mulSpec (subSpec val mean) inv_std
            addSpec acc_w (mulSpec grad_val normalized)
          else acc_w
        else acc_w
      ) acc_h
    ) (Tensor.scalar 0)
  )

  -- Compute beta gradients (sum of grad_output for each channel)
  let grad_beta := Tensor.dim (fun c =>
    let channel_grad := getAtSpec grad_output c

    (List.finRange height).foldl (fun acc_h i =>
      (List.finRange width).foldl (fun acc_w j =>
        if h_i : i < height then
          if h_j : j < width then
            addSpec acc_w (getAtSpec (getAtSpec channel_grad ⟨i, h_i⟩) ⟨j, h_j⟩)
          else acc_w
        else acc_w
      ) acc_h
    ) (Tensor.scalar 0)
  )

  (grad_x, grad_gamma, grad_beta)

/-!
## BatchNorm (inference-time, running statistics)

PyTorch distinction:

- *training*: normalize using batch statistics (and update running mean/variance);
- *inference*: normalize using the stored running mean/variance.

TorchLean keeps things pure and explicit: inference-time BatchNorm takes the running statistics
as arguments.
-/

/--
Inference-time BatchNorm for channel-first tensors of shape `.dim channels sSpatial`, using fixed
running statistics.

Formula (per channel `c`):

`y = ((x - μ) / sqrt(σ² + eps)) * γ + β`

This matches the standard evaluation-time behavior of `torch.nn.BatchNorm{1,2,3}d` (no
batch-statistics computation, no running-statistics update).

At inference time, `(μ, σ², γ, β)` are constants, so this is an **affine** map in `x`. See
`NN.Proofs.Analysis.Normalization.batchNorm_inference_eq_mul_add`.
-/
def batchNormInference
  {channels : Nat} {sSpatial : Shape}
  (x : Tensor α (.dim channels sSpatial))
  (runningMean : Tensor α (.dim channels .scalar))
  (runningVar : Tensor α (.dim channels .scalar))
  (gamma : Tensor α (.dim channels .scalar))
  (beta : Tensor α (.dim channels .scalar))
  (epsilon : α := Numbers.normalizationEpsilon) :
  Tensor α (.dim channels sSpatial) :=
  -- Clamp the variance to stay nonnegative in approximate numeric backends.
  let runningVar := maxSpec runningVar (fill 0 (.dim channels .scalar))
  let meanB := broadcastChannelFirst sSpatial runningMean
  let varianceB := broadcastChannelFirst sSpatial runningVar
  let gammaB := broadcastChannelFirst sSpatial gamma
  let betaB := broadcastChannelFirst sSpatial beta
  let centered := subSpec x meanB
  let std := sqrtSpec (addSpec varianceB (fill epsilon (.dim channels sSpatial)))
  addSpec (mulSpec (divSpec centered std) gammaB) betaB

end Spec
