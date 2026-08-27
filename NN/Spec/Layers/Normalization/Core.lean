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
- PyTorch BatchNorm modules: https://docs.pytorch.org/docs/stable/nn.html#normalization-layers
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
  (x : Tensor α [seqLen, embedDim])
  (gamma : Tensor α [embedDim])
  (beta : Tensor α [embedDim])
  (h_seq_pos : seqLen > 0 := by norm_num)
  (h_embed_pos : embedDim > 0 := by norm_num)
  (epsilon : α := Numbers.normalizationEpsilon) :
  Tensor α [seqLen, embedDim] :=

  -- Compute mean along last dimension (dim = 1)
  let _ : Shape.WellFormed (.dim seqLen (.dim embedDim .scalar)) :=
  ⟨⟨h_seq_pos, ⟨h_embed_pos, trivial⟩⟩⟩

  let s := Shape.dim seqLen (Shape.dim embedDim Shape.scalar)
  let h_rank : Spec.Shape.rank s > 0 := by simp [s, Spec.Shape.rank]
  let h_valid : Shape.HasNonemptyAxis (Spec.Shape.rank s - 1) s :=
    Shape.inferNonemptyAxis (Nat.sub_lt h_rank Nat.zero_lt_one)

  let mean := reduceMean (Spec.Shape.rank s - 1) x h_valid.proof

  have h₁ : (Shape.dim seqLen (Shape.dim embedDim Shape.scalar)).rank = 2 := by
    simp [Spec.Shape.rank]

  let mean_broadcast := broadcastAfterSum s (Spec.Shape.rank s - 1) mean
  let centered := subSpec x mean_broadcast

  have inst : Shape.HasNonemptyAxis (Spec.Shape.rank (.dim seqLen (.dim embedDim .scalar)) - 1) (.dim
    seqLen (.dim embedDim .scalar)) := by
    apply Shape.inferNonemptyAxis
    simp [h₁]

  let varianceRaw := reduceVar (Spec.Shape.rank s - 1) centered inst.proof
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
  (x : Tensor α [seqLen, embedDim])
  (gamma : Tensor α [embedDim])
  (_beta : Tensor α [embedDim])
  (grad_output : Tensor α [seqLen, embedDim])
  (epsilon : α := Numbers.normalizationEpsilon) :
  (Tensor α [seqLen, embedDim] ×  -- ∂L/∂x
   Tensor α [embedDim] ×                 -- ∂L/∂gamma
   Tensor α [embedDim]) :=               -- ∂L/∂beta := sum of grad_output

  -- Forward recomputation
  let _ : Shape.WellFormed (.dim seqLen (.dim embedDim .scalar)) :=
  ⟨⟨h_seq_pos, ⟨h_embed_pos, trivial⟩⟩⟩

  let s := Shape.dim seqLen (Shape.dim embedDim Shape.scalar)
  let h_rank : Spec.Shape.rank s > 0 := by simp [s, Spec.Shape.rank]
  let h_valid : Shape.HasNonemptyAxis (Spec.Shape.rank s - 1) s :=
    Shape.inferNonemptyAxis (Nat.sub_lt h_rank Nat.zero_lt_one)

  let mean := reduceMean (Spec.Shape.rank s - 1) x h_valid.proof

  have h₁ : (Shape.dim seqLen (Shape.dim embedDim Shape.scalar)).rank = 2 := by
    simp [Spec.Shape.rank]

  have h₂ : shapeAfterSum (Shape.dim seqLen (Shape.dim embedDim Shape.scalar)) 1
            = Shape.dim seqLen Shape.scalar := by
    simp

  have h3 : shapeAfterSum (Shape.dim seqLen (Shape.dim embedDim Shape.scalar)) ((Shape.dim seqLen
    (Shape.dim embedDim Shape.scalar)).rank - 1)
          = Shape.dim seqLen Shape.scalar := by
    rw [h₁]
    rw [h₂]

  let mean_broadcast := broadcastAfterSum s (Spec.Shape.rank s - 1) mean
  let centered := subSpec x mean_broadcast

  have inst : Shape.HasNonemptyAxis (Spec.Shape.rank (.dim seqLen (.dim embedDim .scalar)) - 1) (.dim
    seqLen (.dim embedDim .scalar)) := by
    apply Shape.inferNonemptyAxis
    simp [h₁]

  let varianceRaw := reduceVar (Spec.Shape.rank s - 1) centered inst.proof
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

  let sum_dy_gamma := reduceSum (Spec.Shape.rank s - 1) dy_gamma inst.proof
  -- We interpret `embedDim` as the feature-count `N` in the closed-form LayerNorm VJP.
  --
  -- Note: this relies on the `Context`'s `Coe Nat α` behaving sensibly (in particular, that
  -- `(embedDim : α)` is nonzero when `embedDim > 0`). This holds for TorchLean's shipped backends
  -- (Float/ℝ/IEEE32Exec), but for exotic saturating casts a specialized scalar interface may be
  -- preferable.
  let N : α := (embedDim : α)
  let mean_dy_gamma := divSpec sum_dy_gamma (fill N (.dim seqLen .scalar))

  let sum_dy_gamma_xhat :=
    reduceSum (Spec.Shape.rank s - 1) (mulSpec dy_gamma norm) inst.proof
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
  (x tangent : Tensor α [seqLen, embedDim])
  (gamma dgamma _beta dbeta : Tensor α [embedDim])
  (epsilon : α := Numbers.normalizationEpsilon) :
  Tensor α [seqLen, embedDim] :=

  let _ : Shape.WellFormed (.dim seqLen (.dim embedDim .scalar)) :=
  ⟨⟨h_seq_pos, ⟨h_embed_pos, trivial⟩⟩⟩

  let s := Shape.dim seqLen (Shape.dim embedDim Shape.scalar)
  let h_rank : Spec.Shape.rank s > 0 := by simp [s, Spec.Shape.rank]
  let h_valid : Shape.HasNonemptyAxis (Spec.Shape.rank s - 1) s :=
    Shape.inferNonemptyAxis (Nat.sub_lt h_rank Nat.zero_lt_one)

  let mean := reduceMean (Spec.Shape.rank s - 1) x h_valid.proof

  have h₁ : (Shape.dim seqLen (Shape.dim embedDim Shape.scalar)).rank = 2 := by
    simp [Spec.Shape.rank]

  let mean_broadcast := broadcastAfterSum s (Spec.Shape.rank s - 1) mean
  let centered := subSpec x mean_broadcast

  have inst : Shape.HasNonemptyAxis (Spec.Shape.rank (.dim seqLen (.dim embedDim .scalar)) - 1) (.dim
    seqLen (.dim embedDim .scalar)) := by
    apply Shape.inferNonemptyAxis
    simp [h₁]

  let varianceRaw := reduceVar (Spec.Shape.rank s - 1) centered inst.proof
  let variance := maxSpec varianceRaw (fill 0 (.dim seqLen .scalar))
  let std := sqrtSpec (addSpec variance (fill epsilon (.dim seqLen .scalar)))
  let inv_std := divSpec (fill 1 (.dim seqLen .scalar)) std
  let inv_std_broadcast := broadcastAfterSum s (Spec.Shape.rank s - 1) inv_std
  let norm := mulSpec centered inv_std_broadcast

  let sum_tangent := reduceSum (Spec.Shape.rank s - 1) tangent inst.proof
  let N : α := (embedDim : α)
  let mean_tangent := divSpec sum_tangent (fill N (.dim seqLen .scalar))

  let sum_tangent_norm :=
    reduceSum (Spec.Shape.rank s - 1) (mulSpec tangent norm) inst.proof
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
/-! ## Group normalization -/

/--
Normalize each sample over groups of channels and every spatial position.

The spatial domain is an arbitrary `Shape`. Channels are split into `groups` contiguous groups;
each group is flattened together with the spatial axes, normalized, and then transformed by the
per-channel `gamma` and `beta` parameters.
-/
def groupNorm
    {batch channels groups : Nat} {spatial : Shape}
    (x : Tensor α (([batch, channels] : Shape).concat spatial))
    (gamma beta : Tensor α [channels])
    (hGroups : groups > 0 := by norm_num)
    (hGroupsLe : channels ≥ groups)
    (hDiv : channels % groups = 0)
    (epsilon : α := Numbers.normalizationEpsilon)
    [Shape.WellFormed (([batch, channels] : Shape).concat spatial)] :
    Tensor α (([batch, channels] : Shape).concat spatial) :=
  let channelsPerGroup := channels / groups
  let spatialSize := Shape.size spatial
  let groupSize := channelsPerGroup * spatialSize
  let inputShape : Shape := ([batch, channels] : Shape).concat spatial
  let groupedShape : Shape := [batch, groups, groupSize]
  let flatShape : Shape := [batch, channels, spatialSize]
  have hInput := Shape.WellFormed.proof (s := inputShape)
  have hBatch : 0 < batch := hInput.1
  have hChannels : 0 < channels := hInput.2.1
  have hSpatial : 0 < spatialSize := by
    simpa [spatialSize] using Shape.size_pos_of_well_formed hInput.2.2
  have hChannelsPerGroup : 0 < channelsPerGroup :=
    Nat.div_pos hGroupsLe hGroups
  have hGroupSize : 0 < groupSize :=
    Nat.mul_pos hChannelsPerGroup hSpatial
  have hChannelsEq : channels = groups * channelsPerGroup := by
    simpa [channelsPerGroup, hDiv] using (Nat.mod_add_div channels groups).symm
  letI : Shape.WellFormed groupedShape :=
    ⟨⟨hBatch, ⟨hGroups, ⟨hGroupSize, trivial⟩⟩⟩⟩
  letI : Shape.WellFormed flatShape :=
    ⟨⟨hBatch, ⟨hChannels, ⟨hSpatial, trivial⟩⟩⟩⟩
  have hGroupedSize : Shape.size inputShape = Shape.size groupedShape := by
    simp only [inputShape, groupedShape, Shape.size]
    rw [hChannelsEq]
    simp [groupSize, spatialSize, Nat.mul_assoc]
  have hFlatSize : Shape.size inputShape = Shape.size flatShape := by
    simp [inputShape, flatShape, spatialSize, Shape.size]
  let grouped : Tensor α groupedShape := reshapeSpec x hGroupedSize
  let axis := Shape.rank groupedShape - 1
  let hAxis : Shape.HasNonemptyAxis axis groupedShape :=
    Shape.inferNonemptyAxis (by simp [axis, groupedShape, Shape.rank])
  let mean := reduceMean axis grouped hAxis.proof
  let meanBroadcast := broadcastAfterSum groupedShape axis mean
  let centered := subSpec grouped meanBroadcast
  let variance := reduceMean axis (mulSpec centered centered) hAxis.proof
  let variance := maxSpec variance (fill 0 (shapeAfterSum groupedShape axis))
  let denominator :=
    sqrtSpec (addSpec variance (fill epsilon (shapeAfterSum groupedShape axis)))
  let normalized :=
    divSpec centered (broadcastAfterSum groupedShape axis denominator)
  let normalizedInput : Tensor α inputShape :=
    reshapeSpec normalized hGroupedSize.symm
  let normalizedFlat : Tensor α flatShape :=
    reshapeSpec normalizedInput hFlatSize
  let channelSpatialShape : Shape := [channels, spatialSize]
  letI : Shape.WellFormed channelSpatialShape :=
    ⟨⟨hChannels, ⟨hSpatial, trivial⟩⟩⟩
  let gammaSpatial : Tensor α channelSpatialShape :=
    broadcastAfterSum channelSpatialShape 1 gamma
  let betaSpatial : Tensor α channelSpatialShape :=
    broadcastAfterSum channelSpatialShape 1 beta
  let gammaFlat : Tensor α flatShape :=
    broadcastAfterSum flatShape 0 gammaSpatial
  let betaFlat : Tensor α flatShape :=
    broadcastAfterSum flatShape 0 betaSpatial
  let outputFlat := addSpec (mulSpec normalizedFlat gammaFlat) betaFlat
  reshapeSpec outputFlat hFlatSize.symm

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
  (x : Tensor α [seqLen, embedDim])
  (gamma : Tensor α [embedDim])
  (h_seq_pos : seqLen > 0 := by norm_num)
  (h_embed_pos : embedDim > 0 := by norm_num)
  (epsilon : α := Numbers.normalizationEpsilon) :
  Tensor α [seqLen, embedDim] :=
  -- Compute RMS along last dimension
  let squared := squareSpec x

  -- Proofs
  let _ : Shape.WellFormed (.dim seqLen (.dim embedDim .scalar)) :=
  ⟨⟨h_seq_pos, ⟨h_embed_pos, trivial⟩⟩⟩
  let s := Shape.dim seqLen (Shape.dim embedDim Shape.scalar)
  let h_rank : Spec.Shape.rank s > 0 := by simp [s, Spec.Shape.rank]
  let h_valid : Shape.HasNonemptyAxis (Spec.Shape.rank s - 1) s :=
    Shape.inferNonemptyAxis (Nat.sub_lt h_rank Nat.zero_lt_one)

  -- Compute mean along last dimension (dim = 1)
  let mean_squared := reduceMean (Spec.Shape.rank s - 1) squared h_valid.proof
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
  (weight : Tensor α [outDim, inDim])
  (gamma : Tensor α [outDim])
  (h_out_pos : outDim > 0 := by norm_num)
  (h_in_pos : inDim > 0 := by norm_num)
  (epsilon : α := Numbers.normalizationEpsilon) :
  Tensor α [outDim, inDim] :=

  -- Compute L2 norm of each row
  let squared := squareSpec weight

  -- Register well-formedness and axis

  let s := Shape.dim outDim (Shape.dim inDim Shape.scalar)
  let _ : Shape.WellFormed s := ⟨⟨h_out_pos, ⟨h_in_pos, trivial⟩⟩⟩

  -- The selected innermost axis is statically known to be nonempty.
  let h_rank : Spec.Shape.rank s > 0 := by simp [s, Spec.Shape.rank]
  let hAxis : Shape.HasNonemptyAxis (Spec.Shape.rank s - 1) s :=
    Shape.inferNonemptyAxis (Nat.sub_lt h_rank Nat.zero_lt_one)

  -- Sum each row along its `inDim` axis.
  let rowSums := reduceSum (Spec.Shape.rank s - 1) squared hAxis.proof
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
