/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Batteries.Data.Array.Scan

/-!
# Core Reinforcement-Learning Definitions

This module collects the small mathematical definitions that sit underneath TorchLean's RL
development.

These definitions are intentionally spec-level rather than runtime-level:

- Bellman-style backups,
- discounted returns,
- generalized advantage estimation (GAE),
- and simple typed rollout records.

That keeps the actual RL mathematics in a proof-friendly namespace and avoids duplicating it inside
runtime/trainer code.

## Numerical Containers

Dynamic trajectories use `Array α`; fixed-horizon trajectories use vectors.

- A trajectory length is usually *data-dependent* (episode termination, truncation, variable rollout
  horizon), so its length may not be available in the tensor type.
- TorchLean uses typed tensors heavily for *fixed-shape* objects (value tables, Q-tables, logits,
  and fixed rollout windows). Arrays are the homogeneous runtime-sized counterpart.

This keeps numerical payloads in the same two representations used elsewhere in TorchLean:
`Tensor α shape` when the shape is known and `Array α` when it is not.

Primary references:

- Sutton, "Learning to Predict by the Methods of Temporal Differences" (1988):
  https://doi.org/10.1023/A:1022633531479
- Watkins and Dayan, "Q-learning" (1992): https://doi.org/10.1007/BF00992698
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed.):
  http://incompleteideas.net/book/the-book-2nd.html
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  (2015): https://arxiv.org/abs/1506.02438
- TorchRL documentation (rollouts, tensordicts, and GAE-style objectives):
  https://pytorch.org/rl/
-/

@[expose] public section

namespace Spec
namespace RL

variable {α : Type}

/-- Small record used by generalized-advantage-estimation helpers. -/
structure AdvantageStep (α : Type) where
  /-- Immediate reward $r_t$. -/
  reward : α
  /-- Baseline / critic value estimate $V(s_t)$. -/
  value : α
  /-- Bootstrap value $V(s_{t+1})$. -/
  nextValue : α
  /-- Episode termination flag. -/
  done : Bool

/-- Convert a terminal flag into a multiplicative continuation mask (`1` for continue, `0` for
stop). -/
def continueMask [Zero α] [One α] (done : Bool) : α :=
  if done then 0 else 1

/-- Bellman-style one-step backup:
$r+\gamma(1-\mathtt{done})\mathtt{bootstrap}$. -/
def discountedBackup [Zero α] [One α] [Add α] [Mul α]
    (reward gamma bootstrap : α) (done : Bool) : α :=
  reward + gamma * continueMask (α := α) done * bootstrap

/-- One-step TD target for state-value or action-value updates. -/
def tdTarget [Zero α] [One α] [Add α] [Mul α]
    (reward gamma nextValue : α) (done : Bool) : α :=
  discountedBackup (α := α) reward gamma nextValue done

/-- TD residual / Bellman error:
$r+\gamma(1-d)\mathtt{nextValue}-\mathtt{value}$. -/
def tdResidual [Zero α] [One α] [Add α] [Mul α] [Sub α]
    (value reward gamma nextValue : α) (done : Bool) : α :=
  tdTarget (α := α) reward gamma nextValue done - value

/-- Discounted returns with a bootstrap value on the far right:
$G_t=r_t+\gamma G_{t+1}$. -/
def discountedReturnsFrom [Zero α] [Add α] [Mul α]
    (gamma : α) (rewards : Array α) (bootstrap : α := 0) : Array α :=
  (rewards.scanr (fun reward future => reward + gamma * future) bootstrap).pop

/-- Discounted returns for a terminal trajectory (bootstrap defaults to `0`). -/
def discountedReturns [Zero α] [Add α] [Mul α] (gamma : α) (rewards : Array α) : Array α :=
  discountedReturnsFrom (α := α) gamma rewards 0

/-- Discounted returns with explicit termination markers.

When `done = true`, the future return is reset before bootstrapping the current reward.
-/
def discountedReturnsDone [Zero α] [One α] [Add α] [Mul α]
    (gamma : α) (rewards : Array α) (dones : Array Bool) (bootstrap : α := 0) :
    Array α :=
  ((rewards.zip dones).scanr
    (fun step future => discountedBackup (α := α) step.1 gamma future step.2)
    bootstrap).pop

/-- Generalized Advantage Estimation (GAE).

Each input step provides $r_t$, $V(s_t)$, $V(s_{t+1})$, and $\mathtt{done}_t$. The resulting array contains
advantages in forward time order.
-/
def generalizedAdvantageEstimation [Zero α] [One α] [Add α] [Mul α] [Sub α]
    (gamma lam : α) (steps : Array (AdvantageStep α)) : Array α :=
  (steps.scanr
    (fun step nextAdvantage =>
      let mask := continueMask (α := α) step.done
      let delta := step.reward + gamma * mask * step.nextValue - step.value
      delta + gamma * lam * mask * nextAdvantage)
    0).pop

/-- Recover lambda-returns from advantages and baseline values via $R_t=A_t+V(s_t)$. -/
def returnsFromAdvantages [Add α] (advantages values : Array α) : Array α :=
  Array.zipWith (fun advantage value => advantage + value) advantages values

end RL
end Spec
