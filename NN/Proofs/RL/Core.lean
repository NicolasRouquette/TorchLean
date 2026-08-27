/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.RL.Core

/-!
# RL Core Proofs

Small structural theorems about TorchLean's pure RL helper functions in `NN.Spec.RL.Core`.

The emphasis here is on *shape/structure* properties (mostly array sizes and truncation
behavior). These facts are often used to justify that derived quantities (discounted returns,
GAE advantages, etc.) align with the rollout data they came from.

These are kept modest but useful:

- discounted-return helpers preserve array size,
- GAE preserves array size,
- `returnsFromAdvantages` truncates to the shorter array.

References:

- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018),
  Sections 3-6 (returns, discounted backups, TD ideas):
  http://incompleteideas.net/book/the-book-2nd.html
- Schulman et al., “High-Dimensional Continuous Control Using Generalized Advantage Estimation”
  (2016): https://arxiv.org/abs/1506.02438
-/

@[expose] public section

namespace Proofs
namespace RL
namespace Core

/-- `discountedReturnsFrom` produces exactly one return per input reward. -/
theorem discountedReturnsFrom_size {α : Type} [Zero α] [Add α] [Mul α]
    (gamma : α) (rewards : Array α) (bootstrap : α) :
    (Spec.RL.discountedReturnsFrom (α := α) gamma rewards bootstrap).size = rewards.size := by
  simp [Spec.RL.discountedReturnsFrom]

/-- `discountedReturns` preserves list length. -/
theorem discountedReturns_size {α : Type} [Zero α] [Add α] [Mul α]
    (gamma : α) (rewards : Array α) :
    (Spec.RL.discountedReturns (α := α) gamma rewards).size = rewards.size := by
  simpa [Spec.RL.discountedReturns] using
    discountedReturnsFrom_size (α := α) gamma rewards 0

/-- `discountedReturnsDone` returns one value per paired reward/done entry. -/
theorem discountedReturnsDone_size_eq_min {α : Type} [Zero α] [One α] [Add α] [Mul α]
    (gamma : α) (rewards : Array α) (dones : Array Bool) (bootstrap : α) :
    (Spec.RL.discountedReturnsDone (α := α) gamma rewards dones bootstrap).size =
      min rewards.size dones.size := by
  unfold Spec.RL.discountedReturnsDone
  rw [Array.size_pop, Array.size_scanr, Array.size_zip]
  grind

/-- When reward and done arrays have the same size, `discountedReturnsDone` preserves that size. -/
theorem discountedReturnsDone_size_of_eqSize {α : Type} [Zero α] [One α] [Add α] [Mul α]
    (gamma : α) (rewards : Array α) (dones : Array Bool) (bootstrap : α)
    (h : rewards.size = dones.size) :
    (Spec.RL.discountedReturnsDone (α := α) gamma rewards dones bootstrap).size =
      rewards.size := by
  rw [discountedReturnsDone_size_eq_min (α := α) (gamma := gamma) (rewards := rewards)
    (dones := dones) (bootstrap := bootstrap)]
  simp [h]

/-- Generalized-advantage-estimation returns one advantage per input step. -/
theorem generalizedAdvantageEstimation_size {α : Type}
    [Zero α] [One α] [Add α] [Mul α] [Sub α]
    (gamma lam : α) (steps : Array (Spec.RL.AdvantageStep α)) :
    (Spec.RL.generalizedAdvantageEstimation (α := α) gamma lam steps).size = steps.size := by
  simp [Spec.RL.generalizedAdvantageEstimation]

/-- `returnsFromAdvantages` truncates to the shorter of the two input arrays. -/
theorem returnsFromAdvantages_size {α : Type} [Add α]
    (advantages values : Array α) :
    (Spec.RL.returnsFromAdvantages (α := α) advantages values).size =
      min advantages.size values.size := by
  simp [Spec.RL.returnsFromAdvantages]

end Core
end RL
end Proofs
