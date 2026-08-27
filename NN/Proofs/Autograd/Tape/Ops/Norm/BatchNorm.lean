/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Layers.Normalization.BatchNorm
public import NN.Proofs.Tensor.Basic
public import NN.Proofs.Autograd.Tape.Core.FDeriv

/-!
# Batch Normalization Backward Correctness

The training-time BatchNorm differential couples all positions in a channel through the channel
mean and variance. This file proves the reverse-mode rule used by TorchLean for an arbitrary
spatial shape. The proof is stated as JVP/VJP adjointness: pairing an input and parameter tangent
with the forward differential gives the same scalar as pairing the upstream gradient with
`Spec.batchNormBackward`.

The theorem is independent of a memory layout. The spatial shape is flattened only inside the
BatchNorm definition, so the result covers vectors, images, volumes, and higher-rank tensors with
one statement.
-/

@[expose] public section

namespace Proofs
namespace Autograd
namespace BatchNorm

open Spec
open Spec.Tensor

noncomputable section

open scoped BigOperators

private abbrev Matrix (channels positions : Nat) :=
  Tensor ℝ [channels, positions]

private abbrev ChannelTensor (channels : Nat) := Tensor ℝ [channels]

private abbrev entry {channels positions : Nat} (x : Matrix channels positions)
    (channel : Fin channels) (position : Fin positions) : ℝ :=
  get2 x channel position

private abbrev channelEntry {channels : Nat} (x : ChannelTensor channels)
    (channel : Fin channels) : ℝ :=
  Spec.Tensor.getScalar x channel

private theorem inner_getScalarE_cast {n m : Nat} (h : n = m)
    (a b : Tensor ℝ [n]) :
    inner ℝ (getScalarE (h ▸ a)) (getScalarE (h ▸ b)) =
      inner ℝ (getScalarE a) (getScalarE b) := by
  subst m
  rfl

private theorem dot_reshapeSpec {s₁ s₂ : Shape} (a b : Tensor ℝ s₁)
    (h : s₁.size = s₂.size) :
    dot (reshapeSpec a h) (reshapeSpec b h) = dot a b := by
  rw [dot_eq_inner_tensorToVec, dot_eq_inner_tensorToVec]
  unfold tensorToVec
  rw [flatten_reshapeSpec, flatten_reshapeSpec]
  exact inner_getScalarE_cast h _ _

private theorem dot_reshapeSpec_left {s₁ s₂ : Shape} (a : Tensor ℝ s₁)
    (b : Tensor ℝ s₂) (h : s₁.size = s₂.size) :
    dot (reshapeSpec a h) b = dot a (reshapeSpec b h.symm) := by
  have hDot := dot_reshapeSpec a (reshapeSpec b h.symm) h
  rw [reshapeSpec_roundtrip] at hDot
  exact hDot

private lemma entry_add {channels positions : Nat} (x y : Matrix channels positions)
    (channel : Fin channels) (position : Fin positions) :
    entry (addSpec x y) channel position =
      entry x channel position + entry y channel position := by
  simp [entry, addSpec]

private lemma entry_sub {channels positions : Nat} (x y : Matrix channels positions)
    (channel : Fin channels) (position : Fin positions) :
    entry (subSpec x y) channel position =
      entry x channel position - entry y channel position := by
  simp [entry, subSpec]

private lemma entry_mul {channels positions : Nat} (x y : Matrix channels positions)
    (channel : Fin channels) (position : Fin positions) :
    entry (mulSpec x y) channel position =
      entry x channel position * entry y channel position := by
  simp [entry, mulSpec]

private lemma entry_broadcastChannel {channels positions : Nat} (x : ChannelTensor channels)
    (channel : Fin channels) (position : Fin positions) :
    entry (broadcastAfterSum (.dim channels (.dim positions .scalar)) 1 x) channel position =
      channelEntry x channel := by
  cases x with
  | dim values =>
      cases hValue : values channel with
      | scalar value =>
          rfl

private lemma channelEntry_reduceSum_axis_one {channels positions : Nat}
    (hPositions : 0 < positions)
    (x : Matrix channels positions) (channel : Fin channels) :
    channelEntry
        (reduceSum 1 x
          (Shape.NonemptyAxis.succ
            (Shape.hasNonemptyAxisZeroOfPos hPositions).proof))
        channel =
      ∑ position : Fin positions, entry x channel position := by
  cases positions with
  | zero => grind
  | succ positions =>
      cases x with
      | dim rows =>
          simp only [reduceSum, reduceDim, Internal.reduceDimCore_dim_succ,
            Internal.reduceDimCore_dim_zero, Internal.reduceOuterAxis_vector, shapeAfterSum,
            channelEntry, Spec.Tensor.getScalar_dim]
          rw [sum_spec_vec]
          apply Finset.sum_congr rfl
          intro position _
          cases hRow : rows channel with
          | dim values =>
              cases hValue : values position with
              | scalar value =>
                  simp [Spec.Tensor.getScalar, entry, get2, hRow, hValue]

private lemma channelEntry_reduceMean_axis_one {channels positions : Nat}
    (hPositions : 0 < positions)
    (x : Matrix channels positions) (channel : Fin channels) :
    channelEntry
        (reduceMean 1 x
          (Shape.NonemptyAxis.succ
            (Shape.hasNonemptyAxisZeroOfPos hPositions).proof))
        channel =
      (∑ position : Fin positions, entry x channel position) / positions := by
  cases positions with
  | zero => grind
  | succ positions =>
      cases x with
      | dim rows =>
          simp only [reduceMean, reduceSum, reduceDim, Internal.reduceDimCore_dim_succ,
            Internal.reduceDimCore_dim_zero, Internal.reduceOuterAxis_vector, shapeAfterSum, mapSpec,
            channelEntry, Spec.Tensor.getScalar_map, Spec.Tensor.getScalar_dim,
            Shape.axisSize_succ, Shape.axisSize_zero]
          rw [sum_spec_vec]
          congr 1
          apply Finset.sum_congr rfl
          intro position _
          cases hRow : rows channel with
          | dim values =>
              cases hValue : values position with
              | scalar value =>
                  simp [Spec.Tensor.getScalar, entry, get2, hRow, hValue]

private theorem channel_normalization_adjoint
    (positions : Nat) (hPositions : 0 < positions)
    (tangent gradOutput xHat : Fin positions → ℝ)
    (invStd gamma dgamma dbeta : ℝ) :
    (∑ position,
        (invStd *
              (tangent position - (∑ k, tangent k) / positions -
                xHat position * ((∑ k, tangent k * xHat k) / positions)) * gamma +
            xHat position * dgamma + dbeta) * gradOutput position) =
      (∑ position,
          tangent position *
            (invStd *
              (gradOutput position * gamma - (∑ k, gradOutput k * gamma) / positions -
                xHat position *
                  ((∑ k, gradOutput k * gamma * xHat k) / positions)))) +
        dgamma * (∑ position, gradOutput position * xHat position) +
        dbeta * (∑ position, gradOutput position) := by
  let meanTangent := (∑ k, tangent k) / (positions : ℝ)
  let meanTangentXHat := (∑ k, tangent k * xHat k) / (positions : ℝ)
  let meanGradGamma := (∑ k, gradOutput k * gamma) / (positions : ℝ)
  let meanGradGammaXHat :=
    (∑ k, gradOutput k * gamma * xHat k) / (positions : ℝ)
  let scale := invStd * gamma
  have hForward :
      (∑ position,
          (invStd *
                (tangent position - meanTangent - xHat position * meanTangentXHat) * gamma +
              xHat position * dgamma + dbeta) * gradOutput position) =
        scale * (∑ position, tangent position * gradOutput position) -
          scale * meanTangent * (∑ position, gradOutput position) -
          scale * meanTangentXHat * (∑ position, xHat position * gradOutput position) +
          dgamma * (∑ position, xHat position * gradOutput position) +
          dbeta * (∑ position, gradOutput position) := by
    calc
      _ = ∑ position,
          (scale * (tangent position * gradOutput position) -
            (scale * meanTangent) * gradOutput position -
            (scale * meanTangentXHat) * (xHat position * gradOutput position) +
            dgamma * (xHat position * gradOutput position) +
            dbeta * gradOutput position) := by
              apply Finset.sum_congr rfl
              intro position _
              dsimp [scale]
              ring
      _ = _ := by
        simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum]
  have hBackward :
      (∑ position,
          tangent position *
            (invStd *
              (gradOutput position * gamma - meanGradGamma -
                xHat position * meanGradGammaXHat))) =
        scale * (∑ position, tangent position * gradOutput position) -
          invStd * meanGradGamma * (∑ position, tangent position) -
          invStd * meanGradGammaXHat *
            (∑ position, tangent position * xHat position) := by
    calc
      _ = ∑ position,
          (scale * (tangent position * gradOutput position) -
            (invStd * meanGradGamma) * tangent position -
            (invStd * meanGradGammaXHat) *
              (tangent position * xHat position)) := by
              apply Finset.sum_congr rfl
              intro position _
              dsimp [scale]
              ring
      _ = _ := by
        simp only [Finset.sum_sub_distrib, ← Finset.mul_sum]
  change
    (∑ position,
        (invStd *
              (tangent position - meanTangent - xHat position * meanTangentXHat) * gamma +
            xHat position * dgamma + dbeta) * gradOutput position) = _
  rw [hForward, hBackward]
  dsimp [meanTangent, meanTangentXHat, meanGradGamma, meanGradGammaXHat]
  have hGradGamma :
      (∑ k, gradOutput k * gamma) = (∑ k, gradOutput k) * gamma := by
    simpa using (Finset.sum_mul Finset.univ gradOutput gamma).symm
  have hGradGammaXHat :
      (∑ k, gradOutput k * gamma * xHat k) =
        gamma * (∑ k, gradOutput k * xHat k) := by
    calc
      _ = ∑ k, gamma * (gradOutput k * xHat k) := by
        apply Finset.sum_congr rfl
        intro k _
        ring
      _ = _ := by
        simpa using
          (Finset.mul_sum Finset.univ (fun k => gradOutput k * xHat k) gamma).symm
  have hXHatGrad :
      (∑ k, xHat k * gradOutput k) = ∑ k, gradOutput k * xHat k := by
    apply Finset.sum_congr rfl
    intro k _
    ring
  rw [hGradGamma, hGradGammaXHat, hXHatGrad]
  dsimp [scale]
  have hPositionsNe : (positions : ℝ) ≠ 0 := by
    exact_mod_cast (Nat.ne_of_gt hPositions)
  field_simp

/-- The normalized BatchNorm reverse rule is adjoint to its forward differential. -/
theorem normalizedJvp_normalizedBackward_adjoint
    {channels positions : Nat} (hPositions : 0 < positions)
    (tangent gradOutput xHat : Tensor ℝ [channels, positions])
    (invStd gamma dgamma dbeta : Tensor ℝ [channels]) :
    dot (Spec.BatchNorm.normalizedJvp hPositions tangent xHat invStd gamma dgamma dbeta)
        gradOutput =
      let backward := Spec.BatchNorm.normalizedBackward hPositions gradOutput xHat invStd gamma
      dot tangent backward.1 + dot dgamma backward.2.1 + dot dbeta backward.2.2 := by
  rw [dot_mat_eq_sum]
  simp only [Spec.BatchNorm.normalizedBackward]
  rw [dot_mat_eq_sum, dot_vec_eq_sum, dot_vec_eq_sum]
  simp only [Spec.BatchNorm.normalizedJvp]
  simp only [entry_add, entry_sub, entry_mul, entry_broadcastChannel,
    channelEntry_reduceMean_axis_one hPositions, channelEntry_reduceSum_axis_one hPositions]
  rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro channel _
  exact channel_normalization_adjoint positions hPositions
    (fun position => entry tangent channel position)
    (fun position => entry gradOutput channel position)
    (fun position => entry xHat channel position)
    (channelEntry invStd channel) (channelEntry gamma channel)
    (channelEntry dgamma channel) (channelEntry dbeta channel)

private theorem normalizedJvp_normalizedBackward_spatial_adjoint
    {channels positions : Nat} {sSpatial : Shape}
    (hPositions : 0 < positions)
    (hReshape : Shape.size (.dim channels sSpatial) =
      Shape.size (.dim channels (.dim positions .scalar)))
    (tangent gradOutput : Tensor ℝ (.dim channels sSpatial))
    (xHat : Tensor ℝ [channels, positions])
    (invStd gamma dgamma dbeta : Tensor ℝ [channels]) :
    dot
        (reshapeSpec
          (Spec.BatchNorm.normalizedJvp hPositions (reshapeSpec tangent hReshape)
            xHat invStd gamma dgamma dbeta)
          hReshape.symm)
        gradOutput =
      let backward := Spec.BatchNorm.normalizedBackward hPositions
        (reshapeSpec gradOutput hReshape) xHat invStd gamma
      dot tangent (reshapeSpec backward.1 hReshape.symm) +
        dot dgamma backward.2.1 + dot dbeta backward.2.2 := by
  rw [dot_reshapeSpec_left]
  let backward := Spec.BatchNorm.normalizedBackward hPositions
    (reshapeSpec gradOutput hReshape) xHat invStd gamma
  change
    dot
        (Spec.BatchNorm.normalizedJvp hPositions (reshapeSpec tangent hReshape)
          xHat invStd gamma dgamma dbeta)
        (reshapeSpec gradOutput hReshape) =
      dot tangent (reshapeSpec backward.1 hReshape.symm) +
        dot dgamma backward.2.1 + dot dbeta backward.2.2
  rw [← dot_reshapeSpec_left tangent backward.1 hReshape]
  exact normalizedJvp_normalizedBackward_adjoint hPositions
    (reshapeSpec tangent hReshape) (reshapeSpec gradOutput hReshape)
    xHat invStd gamma dgamma dbeta

/--
The rank-general BatchNorm reverse rule is adjoint to its forward differential.

This is the correctness theorem for the public channel-first operator. The spatial shape is
arbitrary; flattening is an implementation detail used only to compute each channel's statistics.
The conclusion accounts for the input tangent and both affine parameters.
-/
theorem batchNormJvp_batchNormBackward_adjoint
    {channels : Nat} {sSpatial : Shape}
    (x tangent gradOutput : Tensor ℝ (.dim channels sSpatial))
    (gamma dgamma beta dbeta : Tensor ℝ [channels])
    (epsilon : ℝ := Numbers.normalizationEpsilon)
    [Shape.WellFormed (.dim channels sSpatial)] :
    dot (Spec.batchNormJvp x tangent gamma dgamma beta dbeta epsilon) gradOutput =
      let backward := Spec.batchNormBackward x gamma gradOutput epsilon
      dot tangent backward.1 + dot dgamma backward.2.1 + dot dbeta backward.2.2 := by
  unfold Spec.batchNormJvp Spec.batchNormBackward
  exact normalizedJvp_normalizedBackward_spatial_adjoint _ _ _ _ _ _ _ _ _

end
end BatchNorm
end Autograd
end Proofs
