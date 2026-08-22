/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.TensorReductionShape

/-!
# Normalization layers (spec layer)

This file collects a few normalization operators used throughout TorchLean's spec/model code.

The common pattern is:

- compute per-axis statistics (mean / variance or RMS),
- normalize with an `epsilon` for numerical stability,
- optionally apply an affine transform (`gamma`, `beta`) like PyTorch does.

## References (papers + PyTorch behavior)

- LayerNorm: Ba et al., "Layer Normalization" (2016): https://arxiv.org/abs/1607.06450
- BatchNorm: Ioffe, Szegedy, "Batch Normalization" (2015): https://arxiv.org/abs/1502.03167
- GroupNorm: Wu, He, "Group Normalization" (2018): https://arxiv.org/abs/1803.08494
- RMSNorm: Zhang, Sennrich, "Root Mean Square Layer Normalization" (2019):
  https://arxiv.org/abs/1910.07467
- WeightNorm: Salimans, Kingma, "Weight Normalization" (2016): https://arxiv.org/abs/1602.07868

- PyTorch LayerNorm: https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
- PyTorch BatchNorm2d: https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
-/

@[expose] public section


namespace Spec
open Tensor
open Numbers

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-- Core normalization routine with explicit broadcast proofs.

This is the shared “math step” behind normalization layers:

`y = ((x - mean) / sqrt(variance + ε)) * gamma + beta`.
-/
def normalizeCore
  (s s_mean s_var s_gamma s_beta : Shape)
  (epsilon : α)
  (x : Tensor α s)
  (mean : Tensor α s_mean)
  (variance : Tensor α s_var)
  (gamma : Tensor α s_gamma)
  (beta : Tensor α s_beta)
  (cb_mean : Shape.CanBroadcastTo s_mean s)
  (cb_var : Shape.CanBroadcastTo s_var s)
  (cb_gamma : Shape.CanBroadcastTo s_gamma s)
  (cb_beta : Shape.CanBroadcastTo s_beta s) : Tensor α s :=

  let mean_broadcast := broadcastTo cb_mean mean
  let variance_broadcast := broadcastTo cb_var variance
  let gamma_broadcast := broadcastTo cb_gamma gamma
  let beta_broadcast := broadcastTo cb_beta beta

  let centered := subSpec x mean_broadcast
  let std := sqrtSpec (addSpec variance_broadcast (fill epsilon s))
  let normalized := divSpec centered std
  addSpec (mulSpec normalized gamma_broadcast) beta_broadcast


/-
  Layer Normalization
  Normalizes along the last dimension
-/
/-- LayerNorm over the last dimension of a `(seqLen, embedDim)` tensor.

Uses `epsilon` (default `Numbers.normalizationEpsilon`) for numerical stability in the denominator.
-/
def layerNorm {seqLen embedDim : Nat}
  (x : Tensor α (.dim seqLen (.dim embedDim .scalar)))
  (gamma : Tensor α (.dim embedDim .scalar))
  (beta : Tensor α (.dim embedDim .scalar))
  (h_seq_pos : seqLen > 0 := by norm_num)
  (h_embed_pos : embedDim > 0 := by norm_num)
  (epsilon : α := Numbers.normalizationEpsilon) :
  Tensor α (.dim seqLen (.dim embedDim .scalar)) :=

  -- Compute mean along last dimension (dim = 1)
  let _ : Shape.WellFormed (.dim seqLen (.dim embedDim .scalar)) :=
  ⟨⟨h_seq_pos, ⟨h_embed_pos, trivial⟩⟩⟩

  let s := Shape.dim seqLen (Shape.dim embedDim Shape.scalar)
  let h_rank : Spec.Shape.rank s > 0 := by simp [s, Spec.Shape.rank]
  let h_valid : Shape.HasNonemptyAxis (Spec.Shape.rank s - 1) s := Shape.inferNonemptyLastAxis h_rank

  let mean := reduceMeanLast x h_valid

  have h₁ : (Shape.dim seqLen (Shape.dim embedDim Shape.scalar)).rank = 2 := by
    simp [Spec.Shape.rank]

  let mean_broadcast := broadcastAfterSum s (Spec.Shape.rank s - 1) mean
  let centered := subSpec x mean_broadcast

  have inst : Shape.HasNonemptyAxis (Spec.Shape.rank (.dim seqLen (.dim embedDim .scalar)) - 1) (.dim
    seqLen (.dim embedDim .scalar)) := by
    apply Shape.inferNonemptyLastAxis
    simp [h₁]

  let varianceRaw := reduceVarLast centered inst
  -- Clamp variance to be nonnegative so `std` is always defined/bounded away from 0 even for
  -- approximate numeric contexts (Float/NF) where small negative variance can occur.
  let variance := maxSpec varianceRaw (fill 0 (.dim seqLen .scalar))

  let std := sqrtSpec (addSpec variance (fill epsilon (.dim seqLen .scalar)))

  let std_broadcast := broadcastAfterSum s (Spec.Shape.rank s - 1) std
  let normalized := divSpec centered std_broadcast

  have h5 : Shape.CanBroadcastTo (.dim embedDim .scalar) (.dim seqLen (.dim embedDim .scalar)) := by
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.dim_eq
    exact Shape.CanBroadcastTo.scalar

  let gamma_broadcast := broadcastTo h5 gamma
  let beta_broadcast := broadcastTo h5 beta
  let scaled := mulSpec normalized gamma_broadcast
  addSpec scaled beta_broadcast

/-- Backward/VJP for `layerNorm` (returns `(dx, dGamma, dBeta)`). -/
def layerNormBackward
  {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (x : Tensor α (.dim seqLen (.dim embedDim .scalar)))
  (gamma : Tensor α (.dim embedDim .scalar))
  (_beta : Tensor α (.dim embedDim .scalar))
  (grad_output : Tensor α (.dim seqLen (.dim embedDim .scalar)))
  (epsilon : α := Numbers.normalizationEpsilon) :
  (Tensor α (.dim seqLen (.dim embedDim .scalar)) ×  -- ∂L/∂x
   Tensor α (.dim embedDim .scalar) ×                 -- ∂L/∂gamma
   Tensor α (.dim embedDim .scalar)) :=               -- ∂L/∂beta := sum of grad_output

  -- Forward recomputation
  let _ : Shape.WellFormed (.dim seqLen (.dim embedDim .scalar)) :=
  ⟨⟨h_seq_pos, ⟨h_embed_pos, trivial⟩⟩⟩

  let s := Shape.dim seqLen (Shape.dim embedDim Shape.scalar)
  let h_rank : Spec.Shape.rank s > 0 := by simp [s, Spec.Shape.rank]
  let h_valid : Shape.HasNonemptyAxis (Spec.Shape.rank s - 1) s := Shape.inferNonemptyLastAxis h_rank

  let mean := reduceMeanLast x h_valid

  have h₁ : (Shape.dim seqLen (Shape.dim embedDim Shape.scalar)).rank = 2 := by
    simp [Spec.Shape.rank]

  have h₂ : shapeAfterSum (Shape.dim seqLen (Shape.dim embedDim Shape.scalar)) 1
            = Shape.dim seqLen Shape.scalar := by
    rw [shape_after_sum_dim_1_alt seqLen embedDim]

  have h3 : shapeAfterSum (Shape.dim seqLen (Shape.dim embedDim Shape.scalar)) ((Shape.dim seqLen
    (Shape.dim embedDim Shape.scalar)).rank - 1)
          = Shape.dim seqLen Shape.scalar := by
    rw [h₁]
    rw [h₂]

  let mean_broadcast := broadcastAfterSum s (Spec.Shape.rank s - 1) mean
  let centered := subSpec x mean_broadcast

  have inst : Shape.HasNonemptyAxis (Spec.Shape.rank (.dim seqLen (.dim embedDim .scalar)) - 1) (.dim
    seqLen (.dim embedDim .scalar)) := by
    apply Shape.inferNonemptyLastAxis
    simp [h₁]

  let varianceRaw := reduceVarLast centered inst
  let variance := maxSpec varianceRaw (fill 0 (.dim seqLen .scalar))
  let std := sqrtSpec (addSpec variance (fill epsilon (.dim seqLen .scalar)))
  let inv_std := divSpec (fill 1 (.dim seqLen .scalar)) std

  let std_broadcast := broadcastAfterSum s (Spec.Shape.rank s - 1) std
  let norm := divSpec centered std_broadcast

  have h5 : Shape.CanBroadcastTo (.dim embedDim .scalar) (.dim seqLen (.dim embedDim .scalar)) := by
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.dim_eq
    exact Shape.CanBroadcastTo.scalar

  -- `gamma` and `beta` have shape `[embedDim]` and are shared across all `seqLen` positions, so
  -- their
  -- gradients sum over the sequence dimension (axis 0).
  let hSequenceAxis := Shape.hasNonemptyAxisZeroOfPos h_seq_pos
  let grad_beta := reduceSum 0 grad_output hSequenceAxis.proof
  let grad_gamma := reduceSum 0 (mulSpec grad_output norm) hSequenceAxis.proof

  -- ∂L/∂x: standard LayerNorm VJP, using per-position statistics over the feature dimension.
  --
  -- Let `N = embedDim`, `xhat = norm`, and `dy = grad_output`.
  -- With `dy_gamma = dy ⊙ gamma`, the closed form is:
  --
  --   dx = inv_std ⊙ ( dy_gamma
  --                    - mean(dy_gamma)
  --                    - xhat ⊙ mean(dy_gamma ⊙ xhat) )
  --
  -- where the `mean` is taken over the last dimension (features) for each sequence position.
  let gamma_broadcast := broadcastTo h5 gamma
  let inv_std_broadcast := broadcastAfterSum s (Spec.Shape.rank s - 1) inv_std
  let dy_gamma := mulSpec grad_output gamma_broadcast

  let sum_dy_gamma := reduceSumLast dy_gamma
  -- We interpret `embedDim` as the feature-count `N` in the closed-form LayerNorm VJP.
  --
  -- Note: this relies on the `Context`'s `Coe Nat α` behaving sensibly (in particular, that
  -- `(embedDim : α)` is nonzero when `embedDim > 0`). This holds for TorchLean's shipped backends
  -- (Float/ℝ/IEEE32Exec), but for exotic saturating casts a specialized scalar interface may be
  -- preferable.
  let N : α := (embedDim : α)
  let mean_dy_gamma := divSpec sum_dy_gamma (fill N (.dim seqLen .scalar))

  let sum_dy_gamma_xhat := reduceSumLast (mulSpec dy_gamma norm)
  let mean_dy_gamma_xhat := divSpec sum_dy_gamma_xhat (fill N (.dim seqLen .scalar))

  let mean_dy_gamma_broadcast :=
    broadcastAfterSum s (Spec.Shape.rank s - 1) mean_dy_gamma
  let mean_dy_gamma_xhat_broadcast :=
    broadcastAfterSum s (Spec.Shape.rank s - 1) mean_dy_gamma_xhat

  let grad_x :=
    mulSpec inv_std_broadcast
      (subSpec (subSpec dy_gamma mean_dy_gamma_broadcast) (mulSpec norm
        mean_dy_gamma_xhat_broadcast))

  (grad_x, grad_gamma, grad_beta)

/--
Forward-mode JVP for `layerNorm`.

For each sequence position, LayerNorm is the map
`y = gamma ⊙ xhat + beta` with `xhat = (x - mean(x)) / sqrt(var(x)+eps)`.
The input tangent is normalized by the standard closed form

`dxhat = inv_std ⊙ (dx - mean(dx) - xhat ⊙ mean(dx ⊙ xhat))`,

and affine-parameter tangents contribute `xhat ⊙ dgamma + dbeta`. This is the forward-mode
counterpart of the closed-form VJP above and follows the same clamped-variance convention as the
forward pass.
-/
def layerNormJvp
  {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (x tangent : Tensor α (.dim seqLen (.dim embedDim .scalar)))
  (gamma dgamma _beta dbeta : Tensor α (.dim embedDim .scalar))
  (epsilon : α := Numbers.normalizationEpsilon) :
  Tensor α (.dim seqLen (.dim embedDim .scalar)) :=

  let _ : Shape.WellFormed (.dim seqLen (.dim embedDim .scalar)) :=
  ⟨⟨h_seq_pos, ⟨h_embed_pos, trivial⟩⟩⟩

  let s := Shape.dim seqLen (Shape.dim embedDim Shape.scalar)
  let h_rank : Spec.Shape.rank s > 0 := by simp [s, Spec.Shape.rank]
  let h_valid : Shape.HasNonemptyAxis (Spec.Shape.rank s - 1) s := Shape.inferNonemptyLastAxis h_rank

  let mean := reduceMeanLast x h_valid

  have h₁ : (Shape.dim seqLen (Shape.dim embedDim Shape.scalar)).rank = 2 := by
    simp [Spec.Shape.rank]

  let mean_broadcast := broadcastAfterSum s (Spec.Shape.rank s - 1) mean
  let centered := subSpec x mean_broadcast

  have inst : Shape.HasNonemptyAxis (Spec.Shape.rank (.dim seqLen (.dim embedDim .scalar)) - 1) (.dim
    seqLen (.dim embedDim .scalar)) := by
    apply Shape.inferNonemptyLastAxis
    simp [h₁]

  let varianceRaw := reduceVarLast centered inst
  let variance := maxSpec varianceRaw (fill 0 (.dim seqLen .scalar))
  let std := sqrtSpec (addSpec variance (fill epsilon (.dim seqLen .scalar)))
  let inv_std := divSpec (fill 1 (.dim seqLen .scalar)) std
  let inv_std_broadcast := broadcastAfterSum s (Spec.Shape.rank s - 1) inv_std
  let norm := mulSpec centered inv_std_broadcast

  let sum_tangent := reduceSumLast tangent
  let N : α := (embedDim : α)
  let mean_tangent := divSpec sum_tangent (fill N (.dim seqLen .scalar))

  let sum_tangent_norm := reduceSumLast (mulSpec tangent norm)
  let mean_tangent_norm := divSpec sum_tangent_norm (fill N (.dim seqLen .scalar))

  let mean_tangent_broadcast := broadcastAfterSum s (Spec.Shape.rank s - 1) mean_tangent
  let mean_tangent_norm_broadcast :=
    broadcastAfterSum s (Spec.Shape.rank s - 1) mean_tangent_norm

  let dnorm :=
    mulSpec inv_std_broadcast
      (subSpec (subSpec tangent mean_tangent_broadcast)
        (mulSpec norm mean_tangent_norm_broadcast))

  have h5 : Shape.CanBroadcastTo (.dim embedDim .scalar) (.dim seqLen (.dim embedDim .scalar)) := by
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.dim_eq
    exact Shape.CanBroadcastTo.scalar

  let gamma_broadcast := broadcastTo h5 gamma
  let dgamma_broadcast := broadcastTo h5 dgamma
  let dbeta_broadcast := broadcastTo h5 dbeta
  addSpec (addSpec (mulSpec dnorm gamma_broadcast) (mulSpec norm dgamma_broadcast))
    dbeta_broadcast
/-
  Group Normalization
  Normalizes within groups of channels
-/
 /--
GroupNorm for channel-last tensors `(batch, height, width, channels)`.

PyTorch analogy: `torch.nn.GroupNorm(num_groups=groups, num_channels=channels)` applied per sample.
The mean/variance are computed over *both* the spatial dimensions and the channels within each
group, then an affine transform is applied per channel via `gamma` and `beta`.

This operator is useful in settings where BatchNorm's dependence on batch statistics is awkward
(small batches, verification, or when you want purely per-sample behavior).
-/
def groupNorm
  {batchSize height width channels groups : Nat}
  (x : Tensor α (.dim batchSize (.dim height (.dim width (.dim channels .scalar)))))
  (gamma : Tensor α (.dim channels .scalar))
  (beta : Tensor α (.dim channels .scalar))
  (h_b : batchSize > 0 := by norm_num)
  (h_h : height > 0 := by norm_num)
  (h_w : width > 0 := by norm_num)
  (h_c : channels > 0 := by norm_num)
  (h_g : groups > 0 := by norm_num)
  (h_ge : channels ≥ groups)
  (h_div : channels % groups = 0)
  (epsilon : α := Numbers.normalizationEpsilon) :
  Tensor α (.dim batchSize (.dim height (.dim width (.dim channels .scalar)))) :=
  let channelsPerGroup := channels / groups
  let hw : Nat := height * width

  have h_hw : hw > 0 := Nat.mul_pos h_h h_w
  have h_cpg_pos : channelsPerGroup > 0 := by
    unfold channelsPerGroup
    apply Nat.div_pos
    exact h_ge
    exact h_g

  have h_ch : channels = groups * channelsPerGroup := by
    rw [← Nat.mod_add_div channels groups, h_div]
    unfold channelsPerGroup
    simp

  -- Flatten the spatial axes, move channels before space, and only then split channels into groups.
  -- A direct NHWC-to-group reshape would mix spatial positions with channels in row-major order.
  let s_orig := Shape.dim batchSize (.dim height (.dim width (.dim channels .scalar)))
  let s_spatial := Shape.dim batchSize (.dim hw (.dim channels .scalar))
  let s_channel := Shape.dim batchSize (.dim channels (.dim hw .scalar))
  let s_reshaped := Shape.dim batchSize (.dim groups (.dim channelsPerGroup (.dim hw .scalar)))

  have _wf : Shape.WellFormed s_orig := ⟨⟨h_b, ⟨h_h, ⟨h_w, ⟨h_c, trivial⟩⟩⟩⟩⟩
  have wf_r : Shape.WellFormed s_reshaped := ⟨⟨h_b, ⟨h_g, ⟨h_cpg_pos, ⟨h_hw, trivial⟩⟩⟩⟩⟩

  have h_flatten_spatial : Spec.Shape.size s_orig = Spec.Shape.size s_spatial := by
    simp [s_orig, s_spatial, Spec.Shape.size, hw, Nat.mul_assoc]
  have h_group : Spec.Shape.size s_channel = Spec.Shape.size s_reshaped := by
    simp [s_channel, s_reshaped, Spec.Shape.size, h_ch, Nat.mul_comm,
      Nat.mul_left_comm]

  let xSpatial : Tensor α s_spatial := reshapeSpec x h_flatten_spatial
  let xChannel : Tensor α s_channel := swapAdjacentAxesWithinLeading xSpatial 0
  let x4 := reshapeSpec xChannel h_group

  -- Mean over spatial axis, then mean over channels within the group.
  letI : Shape.WellFormed s_reshaped := wf_r
  let rank_x4 : Spec.Shape.rank s_reshaped > 0 := by simp [s_reshaped, Spec.Shape.rank]
  let ax_last_x4 : Shape.HasNonemptyAxis (Spec.Shape.rank s_reshaped - 1) s_reshaped :=
    Shape.inferNonemptyLastAxis rank_x4
  let mean_hw := reduceMeanLast x4 ax_last_x4

  let s_mean_hw := Shape.dim batchSize (.dim groups (.dim channelsPerGroup .scalar))
  letI : Shape.WellFormed s_mean_hw := ⟨⟨h_b, ⟨h_g, ⟨h_cpg_pos, trivial⟩⟩⟩⟩

  -- Help typeclass search: `reduce_mean_last_general_wf` wants `WellFormed` on the *input* shape.
  -- Here the input is `shape_after_sum s_reshaped (rank-1)`, which simplifies to `s_mean_hw`.
  have h_after_sum :
      shapeAfterSum s_reshaped (Spec.Shape.rank s_reshaped - 1) = s_mean_hw := by
    simp [s_reshaped, s_mean_hw, Spec.Shape.rank]
  have wf_after_sum :
      Shape.WellFormed (shapeAfterSum s_reshaped (Spec.Shape.rank s_reshaped - 1)) := by
    simpa [h_after_sum] using (show Shape.WellFormed s_mean_hw from inferInstance)
  letI : Shape.WellFormed (shapeAfterSum s_reshaped (Spec.Shape.rank s_reshaped - 1)) := wf_after_sum

  let rank_mean_hw : Spec.Shape.rank s_mean_hw > 0 := by simp [s_mean_hw, Spec.Shape.rank]
  let ax_last_mean_hw : Shape.HasNonemptyAxis (Spec.Shape.rank s_mean_hw - 1) s_mean_hw :=
    Shape.inferNonemptyLastAxis rank_mean_hw
  let mean := reduceMeanLast mean_hw ax_last_mean_hw

  let mean_gc := broadcastAfterSum s_mean_hw (Spec.Shape.rank s_mean_hw - 1) mean
  let mean_b := broadcastAfterSum s_reshaped (Spec.Shape.rank s_reshaped - 1) mean_gc
  let centered := subSpec x4 mean_b

  -- Variance: mean of squared centered values across both axes.
  let centered_sq := mulSpec centered centered
  let var_hw := reduceMeanLast centered_sq ax_last_x4
  let variance := reduceMeanLast var_hw ax_last_mean_hw
  let variance := maxSpec variance (fill 0 (Shape.dim batchSize (.dim groups .scalar)))
  let std := sqrtSpec (addSpec variance (fill epsilon (Shape.dim batchSize (.dim groups
    .scalar))))
  let std_gc := broadcastAfterSum s_mean_hw (Spec.Shape.rank s_mean_hw - 1) std
  let std_b := broadcastAfterSum s_reshaped (Spec.Shape.rank s_reshaped - 1) std_gc

  let normalized := divSpec centered std_b

  -- Reshape `gamma`/`beta` to `[groups, channelsPerGroup]` and broadcast across batch and spatial.
  have h_param_reshape : Spec.Shape.size (Shape.dim channels .scalar) = Spec.Shape.size (Shape.dim groups
    (.dim channelsPerGroup .scalar)) := by
    rw [h_ch]
    simp [Spec.Shape.size]
  let gamma2 := reshapeSpec gamma h_param_reshape
  let beta2 := reshapeSpec beta h_param_reshape

  let s_params := Shape.dim 1 (.dim groups (.dim channelsPerGroup (.dim 1 .scalar)))
  have h_param_keep : Spec.Shape.size (Shape.dim groups (.dim channelsPerGroup .scalar)) =
      Spec.Shape.size s_params := by simp [s_params, Spec.Shape.size]
  let gamma4 := reshapeSpec gamma2 h_param_keep
  let beta4 := reshapeSpec beta2 h_param_keep
  have cb_params : Shape.CanBroadcastTo s_params s_reshaped := by
    apply Shape.CanBroadcastTo.dim_1_to_n
    apply Shape.CanBroadcastTo.dim_eq
    apply Shape.CanBroadcastTo.dim_eq
    apply Shape.CanBroadcastTo.dim_1_to_n
    exact Shape.CanBroadcastTo.scalar
  let gamma_b := broadcastTo cb_params gamma4
  let beta_b := broadcastTo cb_params beta4

  let y4 := addSpec (mulSpec normalized gamma_b) beta_b
  let yChannel : Tensor α s_channel := reshapeSpec y4 h_group.symm
  let ySpatial : Tensor α s_spatial := swapAdjacentAxesWithinLeading yChannel 0
  reshapeSpec ySpatial h_flatten_spatial.symm

/-
  Normalize along a specific dimension
-/
/--
Normalize along a chosen axis `dim` of a tensor `x`, using per-element affine parameters `gamma`
and `beta` of the same shape as `x`.

This is a "generic building block" that is handy in specs; it is closer to the raw math than to a
single PyTorch module. Most named normalizations (LayerNorm, GroupNorm, BatchNorm) are special
cases of this pattern with a specific choice of axis set and parameter shape.
-/
def normalizeAlongDim
  {s : Shape}
  (x : Tensor α s)
  (gamma : Tensor α s)
  (beta : Tensor α s)
  (dim : Nat)
  (h_valid : Shape.HasNonemptyAxis dim s)
  (_h_wf : Shape.WellFormed s)
  (epsilon : α := Numbers.normalizationEpsilon)
  : Tensor α s :=

  -- mean shape: shape_after_sum s dimension
  let mean := reduceMean dim x h_valid.proof

  let mean_broadcast := broadcastAfterSum s dim mean
  -- center x by subtracting mean (broadcasted)
  let centered := subSpec x mean_broadcast

  -- variance shape: shape_after_sum s dimension (same shape as mean)
  let variance := reduceVar dim centered h_valid.proof

  -- broadcast variance to s for addition of epsilon and sqrt
  let variance_broadcast := broadcastAfterSum s dim variance

  -- compute std = sqrt(variance + epsilon)
  let std := sqrtSpec (addSpec variance_broadcast (fill epsilon s))
  -- normalize centered by dividing by std (broadcasted)
  let normalized := divSpec centered std
  -- multiply by gamma (shape s) and add beta (shape s)
  let result := addSpec (mulSpec normalized gamma) beta
  result

/-
  RMS Normalization
  Normalizes using RMS instead of mean/variance
-/
/--
RMSNorm over the last dimension of a `(seqLen, embedDim)` tensor.

Compared to LayerNorm, RMSNorm skips subtracting the mean and normalizes by:

`rms(x) = sqrt(mean(x^2) + eps)`.

This shows up in many Transformer-style models as a cheaper alternative to LayerNorm.
-/
def rmsNorm {seqLen embedDim : Nat}
  (x : Tensor α (.dim seqLen (.dim embedDim .scalar)))
  (gamma : Tensor α (.dim embedDim .scalar))
  (h_seq_pos : seqLen > 0 := by norm_num)
  (h_embed_pos : embedDim > 0 := by norm_num)
  (epsilon : α := Numbers.normalizationEpsilon) :
  Tensor α (.dim seqLen (.dim embedDim .scalar)) :=
  -- Compute RMS along last dimension
  let squared := squareSpec x

  -- Proofs
  let _ : Shape.WellFormed (.dim seqLen (.dim embedDim .scalar)) :=
  ⟨⟨h_seq_pos, ⟨h_embed_pos, trivial⟩⟩⟩
  let s := Shape.dim seqLen (Shape.dim embedDim Shape.scalar)
  let h_rank : Spec.Shape.rank s > 0 := by simp [s, Spec.Shape.rank]
  let h_valid : Shape.HasNonemptyAxis (Spec.Shape.rank s - 1) s := Shape.inferNonemptyLastAxis h_rank

  -- Compute mean along last dimension (dim = 1)
  let mean_squared := reduceMeanLast squared h_valid
  let rms := sqrtSpec (addSpec mean_squared (fill epsilon (.dim seqLen .scalar)))
  -- shape: [seqLen]

  -- Normalize by RMS
  let rms_broadcast := broadcastAfterSum s (Spec.Shape.rank s - 1) rms
  let normalized := divSpec x rms_broadcast

  have h_gamma_broadcast : Shape.CanBroadcastTo (Shape.dim embedDim Shape.scalar) (Shape.dim seqLen
    (.dim embedDim .scalar)) := by
    apply Shape.CanBroadcastTo.expand_dims
    apply Shape.CanBroadcastTo.dim_eq
    exact Shape.CanBroadcastTo.scalar

  -- Scale
  let gamma_broadcast := broadcastTo h_gamma_broadcast gamma
  let result := mulSpec normalized gamma_broadcast
  result

/-
  Weight Normalization
  Normalizes the weight matrix
-/
/--
WeightNorm for a dense weight matrix `(outDim, inDim)`.

This implements the "normalize weight vectors then scale" idea:

- normalize each output row by its L2 norm,
- then rescale by `gamma` (one scalar per output row).

PyTorch analogy: weight normalization is typically applied as a parametrization of a module's
weights rather than as a standalone tensor operator.
-/
def weightNorm {inDim outDim : Nat}
  (weight : Tensor α (.dim outDim (.dim inDim .scalar)))
  (gamma : Tensor α (.dim outDim .scalar))
  (h_out_pos : outDim > 0 := by norm_num)
  (h_in_pos : inDim > 0 := by norm_num)
  (epsilon : α := Numbers.normalizationEpsilon) :
  Tensor α (.dim outDim (.dim inDim .scalar)) :=

  -- Compute L2 norm of each row
  let squared := squareSpec weight

  -- Register well-formedness and axis

  let s := Shape.dim outDim (Shape.dim inDim Shape.scalar)
  let _ : Shape.WellFormed s := ⟨⟨h_out_pos, ⟨h_in_pos, trivial⟩⟩⟩

  -- The selected innermost axis is statically known to be nonempty.
  let h_rank : Spec.Shape.rank s > 0 := by simp [s, Spec.Shape.rank]
  letI : Shape.HasNonemptyAxis (Spec.Shape.rank s - 1) s := Shape.inferNonemptyLastAxis h_rank

  -- Sum each row along its `inDim` axis.
  let rowSums := reduceSumLast squared
  let rowNorms := sqrtSpec (addSpec rowSums (fill epsilon (.dim outDim .scalar)))
  -- shape: [outDim]

  -- Normalize weights
  let rowNorms_broadcast := broadcastAfterSum s (Spec.Shape.rank s - 1) rowNorms
  let normalized := divSpec weight rowNorms_broadcast

  -- Scale
  let gamma_broadcast := broadcastAfterSum s (Spec.Shape.rank s - 1) gamma
  let result := mulSpec normalized gamma_broadcast
  result

end Spec
