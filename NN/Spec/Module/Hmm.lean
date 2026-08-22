/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Models.Hmm
public import NN.Spec.Module.Core

/-!
# HMM adapters as `Spec.Module`s

The HMM spec model (`NN/Spec/Models/Hmm.lean`) uses discrete observations (`Fin nObservations`).

For composition and examples, it is sometimes convenient to accept a tensor of scores/probabilities
over the observation alphabet and decode each timestep via `argmax`. The wrappers in this file
provide that bridge and package the resulting behavior as `Spec.Module`s.
-/

@[expose] public section


namespace Spec.Module

open Tensor

variable {α : Type} [Context α]

/-- Decode a single observation vector into a discrete symbol by taking `argmax`. -/
def decodeObservation
  {nObservations : Nat} (hObservations : nObservations > 0)
  (scores : Tensor α (.dim nObservations .scalar)) : Fin nObservations :=
  argmaxVector hObservations scores

/-- Convert a tensor of per-symbol scores/probabilities into a discrete observation sequence by
decoding each timestep with `argmax`. -/
def decodeObservations
  {seqLen nObservations : Nat} (hObservations : nObservations > 0)
  (scores : Tensor α (.dim seqLen (.dim nObservations .scalar))) :
  Vector (Fin nObservations) seqLen :=
  Vector.ofFn fun t => decodeObservation hObservations (get scores t)

/-- A one-step HMM module: map an observation distribution to a filtered state distribution. -/
def hmm {nStates nObservations : Nat}
  (hObservations : nObservations > 0) (m : HMMSpec α nStates nObservations) :
  Spec.Module α (.dim nObservations .scalar) (.dim nStates .scalar) :=
{
  forward := fun scores =>
    let observation := decodeObservation hObservations scores
    let unnormalized := mulSpec m.initial (emissionVec (α := α) m observation)
    (normalizeVec (α := α) unnormalized).1,
  kind := "HMM",
  pythonExpr := "UnsupportedLayer(\"HMM\", \"torch.distributions.Categorical\")"
}

/-- Forward messages `α_t` for each timestep (scaled). -/
def forwardMessages
  {nStates nObservations : Nat}
  (m : HMMSpec α nStates nObservations)
  (observations : ObservationSeq nObservations) :
  Vector (Tensor α (.dim nStates .scalar)) observations.length :=
  (hmmForwardScaled (α := α) m observations).map (·.message)

/-- Sequence module: compute forward messages `α_t` for each timestep. -/
def hmmSequence {seqLen nStates nObservations : Nat}
  (hObservations : nObservations > 0) (m : HMMSpec α nStates nObservations) :
  Spec.Module α (.dim seqLen (.dim nObservations .scalar)) (.dim seqLen (.dim nStates .scalar)) :=
{
  forward := fun scores =>
    let observations := decodeObservations hObservations scores
    let messages := forwardMessages (α := α) m observations.toList
    Tensor.dim fun t => messages.get ⟨t.val, by simp [t.isLt]⟩,
  kind := "HMMSequence",
  pythonExpr := "UnsupportedLayer(\"HMMSequence\", \"torch.distributions.Categorical\")"
}

/-- Sequence module: compute prefix likelihoods `p(o₀:t)` for each timestep `t`. -/
def hmmPrefixLikelihoods {seqLen nStates nObservations : Nat}
  (hObservations : nObservations > 0) (m : HMMSpec α nStates nObservations) :
  Spec.Module α (.dim seqLen (.dim nObservations .scalar)) (.dim seqLen .scalar) :=
{
  forward := fun scores =>
    let observations := decodeObservations hObservations scores
    Tensor.dim (fun t =>
      let prefixObservations := observations.toList.take (t.val + 1)
      Tensor.scalar (hmmForwardSpec (α := α) m prefixObservations)
    ),
  kind := "HMMPrefixLikelihoods",
  pythonExpr := "UnsupportedLayer(\"HMMPrefixLikelihoods\", \"torch.distributions.Categorical\")"
}

/-- Sequence module: normalized state probabilities at each timestep. -/
def hmmStateProbabilities {seqLen nStates nObservations : Nat}
  (hObservations : nObservations > 0) (m : HMMSpec α nStates nObservations) :
  Spec.Module α (.dim seqLen (.dim nObservations .scalar)) (.dim seqLen (.dim nStates .scalar)) :=
{
  forward := fun scores =>
    let observations := decodeObservations hObservations scores
    let messages := forwardMessages (α := α) m observations.toList
    Tensor.dim (fun t =>
      let message := messages.get ⟨t.val, by simp [t.isLt]⟩
      let total := sumSpec message
      if total > 0 then
        Tensor.dim (fun s =>
          match get message s with
          | Tensor.scalar val => Tensor.scalar (val / total)
        )
      else
        Tensor.dim (fun _ => Tensor.scalar (1 / nStates))
    ),
  kind := "HMMStateProbabilities",
  pythonExpr := "UnsupportedLayer(\"HMMStateProbabilities\", \"torch.distributions.Categorical\")"
}

/-- Sequence module: apply the one-step update independently at each timestep. -/
def hmmIndependent {seqLen nStates nObservations : Nat}
  (hObservations : nObservations > 0) (m : HMMSpec α nStates nObservations) :
  Spec.Module α (.dim seqLen (.dim nObservations .scalar)) (.dim seqLen (.dim nStates .scalar)) :=
{
  forward := fun scores =>
    Tensor.dim (fun t =>
      let observation := decodeObservation hObservations (get scores t)
      let likelihood := hmmForwardSpec (α := α) m [observation]
      Tensor.dim (fun _s => Tensor.scalar likelihood)
    ),
  kind := "HMMIndependent",
  pythonExpr := "UnsupportedLayer(\"HMMIndependent\", \"torch.distributions.Categorical\")"
}

end Spec.Module
