/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Optim.Schedulers.Core

/-!
# Native Learning-Rate Schedulers

TorchLean-native schedules with explicit state and total formulas. Zero-length warmup or cycle
phases have defined fallback behavior, which makes the schedules convenient for direct execution
and theorem statements. `currentStep` is zero-indexed and `step` advances it once.

`Schedulers.Core` documents the shared arithmetic, state convention, and literature. Use the
separate `PyTorch` module when exact PyTorch phase and step-count behavior is required.
-/

@[expose] public section


namespace Optim

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

open MathFunctions

/-! ## Native Schedulers -/

/-- Constant scheduler (no learning rate changes). -/
structure ConstantScheduler (α : Type) where
  /-- Fixed learning rate. -/
  lr : α

/--
Get the learning rate for a constant schedule.

The step argument is ignored (the LR never changes).

PyTorch analogy: no scheduler (or a scheduler that keeps LR fixed).
-/
def ConstantScheduler.getLr (scheduler : ConstantScheduler α) (_ : Nat) : α :=
  scheduler.lr

/--
Advance a constant scheduler by one step.

This is the identity since there is no state to update.

PyTorch analogy: `scheduler.step()` for a scheduler that does nothing.
-/
def ConstantScheduler.step (scheduler : ConstantScheduler α) : ConstantScheduler α :=
  scheduler

/--
Create a constant learning-rate scheduler.

PyTorch analogy: constructing training code with a fixed `lr` and no `lr_scheduler`.
-/
def constantScheduler (lr : α) : ConstantScheduler α :=
  { lr := lr }

/-! ## Exponential decay -/

/--
Exponential decay scheduler: `lr(step) = initial_lr * decay_rate^step`.

PyTorch analogy: similar spirit to `ExponentialLR`, but we keep state as a simple counter.
-/
structure ExponentialDecayScheduler (α : Type) where
  /-- Learning rate at step `0`. -/
  initialLr : α
  /-- Multiplicative decay factor per step (`gamma` in PyTorch terminology). -/
  decayRate : α
  /-- Current step counter (0-indexed). -/
  currentStep : Nat := 0

/--
Get the learning rate for an exponential decay schedule at the current step.

Formula: `initial_lr * decay_rate ^ current_step`.

PyTorch analogy: `torch.optim.lr_scheduler.ExponentialLR` (but here kept as a pure counter-based
  record).
-/
def ExponentialDecayScheduler.getLr (scheduler : ExponentialDecayScheduler α) : α :=
  scheduler.initialLr * (scheduler.decayRate ^ (scheduler.currentStep : α))

/--
Advance the exponential decay scheduler by one step.

PyTorch analogy: `scheduler.step()`.
-/
def ExponentialDecayScheduler.step (scheduler : ExponentialDecayScheduler α) :
  ExponentialDecayScheduler α :=
  { scheduler with currentStep := scheduler.currentStep + 1 }

/--
Create an exponential decay scheduler starting at step `0`.

PyTorch analogy: `torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)`.
-/
def exponentialDecayScheduler (initialLr : α) (decay_rate : α) : ExponentialDecayScheduler α :=
  { initialLr := initialLr, decayRate := decay_rate }

/-! ## Step decay -/

/--
Piecewise-constant decay: every `step_size` steps, multiply the learning rate by `decay_factor`.
-/
structure StepDecayScheduler (α : Type) where
  /-- Learning rate at step `0`. -/
  initialLr : α
  /-- Multiplicative decay factor applied every `step_size` steps. -/
  decayFactor : α
  /-- Number of steps between decays. -/
  stepSize : Nat
  /-- Current step counter (0-indexed). -/
  currentStep : Nat := 0

/--
Get the learning rate for step decay at the current step.

Every `step_size` steps, the LR is multiplied by `decay_factor`. When `step_size = 0`, this falls
back to a constant LR.

PyTorch analogy: `torch.optim.lr_scheduler.StepLR`.
-/
def StepDecayScheduler.getLr (scheduler : StepDecayScheduler α) : α :=
  if scheduler.stepSize = 0 then
    scheduler.initialLr
  else
    let decayCount := scheduler.currentStep / scheduler.stepSize
    scheduler.initialLr * (scheduler.decayFactor ^ (decayCount : α))

omit [DecidableRel ((· > ·) : α → α → Prop)] in
/--
The totalized `step_size = 0` case is constant.

PyTorch would reject this configuration; TorchLean keeps scheduler evaluation total so configs can
be validated separately from pure schedule semantics.
-/
theorem StepDecayScheduler.getLr_zero_stepSize
    (initialLr decayFactor : α) (currentStep : Nat) :
    StepDecayScheduler.getLr
      { initialLr := initialLr
        decayFactor := decayFactor
        stepSize := 0
        currentStep := currentStep } = initialLr := by
  simp [StepDecayScheduler.getLr]

/--
Advance the step-decay scheduler by one step.

PyTorch analogy: `scheduler.step()`.
-/
def StepDecayScheduler.step (scheduler : StepDecayScheduler α) : StepDecayScheduler α :=
  { scheduler with currentStep := scheduler.currentStep + 1 }

/--
Create a step-decay scheduler starting at step `0`.

PyTorch analogy: `torch.optim.lr_scheduler.StepLR(optimizer, step_size=..., gamma=decay_factor)`.
-/
def stepDecayScheduler (initialLr : α) (decay_factor : α) (stepSize : Nat) : StepDecayScheduler α
  :=
  { initialLr := initialLr, decayFactor := decay_factor, stepSize := stepSize }

/-! ## Cosine annealing -/

/--
Cosine annealing down to `min_lr` over `max_steps` steps.

PyTorch analogy: `CosineAnnealingLR` (without restarts).
-/
structure CosineAnnealingScheduler (α : Type) where
  /-- Learning rate at step `0`. -/
  initialLr : α
  /-- Minimum learning rate after annealing completes. -/
  minLr : α
  /-- Number of steps over which to anneal. -/
  maxSteps : Nat
  /-- Current step counter (0-indexed). -/
  currentStep : Nat := 0

/--
Get the learning rate for cosine annealing at the current step.

We anneal from `initial_lr` to `min_lr` over `max_steps` steps (clamping once we pass `max_steps`).

PyTorch analogy: `torch.optim.lr_scheduler.CosineAnnealingLR` (without restarts).
-/
def CosineAnnealingScheduler.getLr (scheduler : CosineAnnealingScheduler α) : α :=
  if scheduler.maxSteps = 0 then
    scheduler.initialLr
  else
    let step := if scheduler.currentStep < scheduler.maxSteps then scheduler.currentStep else
      scheduler.maxSteps
    let factor := SchedulerUtils.ratioNat step scheduler.maxSteps
    SchedulerUtils.cosineInterpolation scheduler.initialLr scheduler.minLr factor

/--
Advance the cosine annealing scheduler by one step.

PyTorch analogy: `scheduler.step()`.
-/
def CosineAnnealingScheduler.step (scheduler : CosineAnnealingScheduler α) :
  CosineAnnealingScheduler α :=
  { scheduler with currentStep := scheduler.currentStep + 1 }

/--
Create a cosine annealing scheduler starting at step `0`.

PyTorch analogy: `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps,
  eta_min=min_lr)`.
-/
def cosineAnnealingScheduler (initialLr : α) (maxSteps : Nat) (minLr : α := 0) :
    CosineAnnealingScheduler α :=
  { initialLr := initialLr, minLr := minLr, maxSteps := maxSteps }

/-! ## Linear warmup -/

/--
Linear warmup from `start_lr` to `initial_lr` over `warmup_steps`, then constant.

Warmup is a practical trick commonly used when training large models (e.g. Transformers) to avoid
instability at the start of training.
-/
structure LinearWarmupScheduler (α : Type) where
  /-- Target learning rate after warmup. -/
  initialLr : α
  /-- Number of warmup steps. -/
  warmupSteps : Nat
  /-- Starting learning rate during warmup. -/
  startLr : α
  /-- Current step counter (0-indexed). -/
  currentStep : Nat := 0

/--
Get the learning rate for linear warmup (then constant).

Before `warmup_steps`, linearly interpolate from `start_lr` to `initial_lr`. Afterwards, keep
`initial_lr` fixed.

PyTorch analogy: warmup logic commonly implemented in training scripts (and in some scheduler
  helpers).
-/
def LinearWarmupScheduler.getLr (scheduler : LinearWarmupScheduler α) : α :=
  if scheduler.warmupSteps = 0 then
    scheduler.initialLr
  else if scheduler.currentStep < scheduler.warmupSteps then
    let factor := SchedulerUtils.ratioNat scheduler.currentStep scheduler.warmupSteps
    SchedulerUtils.linearInterpolation scheduler.startLr scheduler.initialLr factor
  else
    scheduler.initialLr

/--
Advance the linear warmup scheduler by one step.

PyTorch analogy: `scheduler.step()`.
-/
def LinearWarmupScheduler.step (scheduler : LinearWarmupScheduler α) : LinearWarmupScheduler α :=
  { scheduler with currentStep := scheduler.currentStep + 1 }

/--
Create a linear warmup scheduler starting at step `0`.

PyTorch analogy: a warmup wrapper around an optimizer or a base scheduler.
-/
def linearWarmupScheduler (initialLr : α) (warmupSteps : Nat) (startLr : α := 0) :
    LinearWarmupScheduler α :=
  { initialLr := initialLr, warmupSteps := warmupSteps, startLr := startLr }

/-! ## Warmup + cosine -/

/--
Warmup followed by cosine annealing.

This is a common “default” schedule for Transformer-style training: warm up for a few thousand
steps, then gradually anneal.
-/
structure WarmupCosineScheduler (α : Type) where
  /-- Peak learning rate (reached at the end of warmup). -/
  initialLr : α
  /-- Number of warmup steps. -/
  warmupSteps : Nat
  /-- Total number of steps for the whole schedule (warmup + anneal). -/
  totalSteps : Nat
  /-- Current step counter (0-indexed). -/
  currentStep : Nat := 0

/--
Get the learning rate for the warmup-then-cosine schedule at the current step.

- During warmup, LR increases linearly from `0` to `initial_lr`.
- After warmup, LR follows a cosine anneal over the remaining steps.
- At and after `total_steps`, LR remains at `0` instead of beginning another cosine period.

PyTorch analogy: a common Transformer schedule, often implemented by composing warmup with cosine
  decay.
-/
def WarmupCosineScheduler.getLr (scheduler : WarmupCosineScheduler α) : α :=
  if scheduler.totalSteps = 0 then
    scheduler.initialLr
  else if scheduler.currentStep >= scheduler.totalSteps then
    0
  else if scheduler.currentStep < scheduler.warmupSteps then
    if scheduler.warmupSteps = 0 then
      scheduler.initialLr
    else
      scheduler.initialLr * SchedulerUtils.ratioNat scheduler.currentStep scheduler.warmupSteps
  else
    let remaining_steps := scheduler.totalSteps - scheduler.warmupSteps
    if remaining_steps = 0 then
      scheduler.initialLr
    else
      let current_remaining := scheduler.currentStep - scheduler.warmupSteps
      let progress := SchedulerUtils.ratioNat current_remaining remaining_steps
      let cosine_factor := (1 + cos ((pi : α) * progress)) / (1 + 1)
      scheduler.initialLr * cosine_factor

/--
Advance the warmup+cosine scheduler by one step.

PyTorch analogy: `scheduler.step()`.
-/
def WarmupCosineScheduler.step (scheduler : WarmupCosineScheduler α) : WarmupCosineScheduler α :=
  { scheduler with currentStep := scheduler.currentStep + 1 }

/--
Create a warmup+cosine scheduler starting at step `0`.

PyTorch analogy: composing a warmup schedule with cosine annealing in a training script.
-/
def warmupCosineScheduler (initialLr : α) (warmupSteps : Nat) (totalSteps : Nat) :
    WarmupCosineScheduler α :=
  { initialLr := initialLr, warmupSteps := warmupSteps, totalSteps := totalSteps }

/-! ## Cyclic LR -/

/--
Cyclic learning rate schedule.

This corresponds to the “triangular” family of schedules where the LR increases linearly from
`base_lr` to `max_lr` and then decreases back, repeating in cycles.

We keep `mode` as a `String` so this runtime layer can be configured from simple config files or
CLI arguments (mirroring how training scripts are usually written).
-/
structure CyclicScheduler (α : Type) where
  /-- Minimum learning rate within the cycle. -/
  baseLr : α
  /-- Maximum learning rate within the cycle (before any mode-specific adjustment). -/
  maxLr : α
  /-- Half-cycle size (in steps). -/
  stepSize : Nat
  /-- `"triangular"`, `"triangular2"`, or `"exp_range"`. -/
  mode : String
  /-- Decay factor used by `"exp_range"`. -/
  gamma : α
  /-- Current step counter (0-indexed). -/
  currentStep : Nat := 0

/--
Get the learning rate for the cyclic schedule at the current step.

Supports the common `"triangular"`, `"triangular2"`, and `"exp_range"` variants (matching the
flavor of PyTorch's `CyclicLR`).

PyTorch analogy: `torch.optim.lr_scheduler.CyclicLR`.
-/
def CyclicScheduler.getLr (scheduler : CyclicScheduler α) : α :=
  if scheduler.stepSize = 0 then
    scheduler.baseLr
  else
    let cycleStep := scheduler.currentStep % (2 * scheduler.stepSize)
    let x := SchedulerUtils.ratioNat cycleStep scheduler.stepSize

    let cycle := scheduler.currentStep / (2 * scheduler.stepSize)
    let adjustedMaxLR :=
      if scheduler.mode == "triangular" then scheduler.maxLr
      else if scheduler.mode == "triangular2" then
        scheduler.maxLr - (scheduler.maxLr - scheduler.baseLr) * (1 - 1 / ((1 + 1) ^ (cycle :
          α)))
      else if scheduler.mode == "exp_range" then
        scheduler.baseLr + (scheduler.maxLr - scheduler.baseLr) * scheduler.gamma ^
          (scheduler.currentStep : α)
      else
        scheduler.maxLr

    if cycleStep < scheduler.stepSize then
      scheduler.baseLr + (adjustedMaxLR - scheduler.baseLr) * x
    else
      adjustedMaxLR - (adjustedMaxLR - scheduler.baseLr) * (x - 1)

/--
Advance the cyclic scheduler by one step.

PyTorch analogy: `scheduler.step()`.
-/
def CyclicScheduler.step (scheduler : CyclicScheduler α) : CyclicScheduler α :=
  { scheduler with currentStep := scheduler.currentStep + 1 }

/--
Create a cyclic learning-rate scheduler starting at step `0`.

PyTorch analogy: `torch.optim.lr_scheduler.CyclicLR(base_lr=..., max_lr=..., step_size_up=...)`.
-/
def cyclicScheduler (baseLr : α) (maxLr : α) (stepSize : Nat)
    (mode : String := "triangular") (gamma : α := 1) : CyclicScheduler α :=
  { baseLr := baseLr, maxLr := maxLr, stepSize := stepSize, mode := mode, gamma := gamma }

/-! ## Triangular cycle (special case) -/

/--
A specialized cyclic schedule with fixed amplitude.

This is essentially `CyclicScheduler` in `"triangular"` mode, but we provide it as a separate type
so callers don't have to thread mode strings around.
-/
structure TriangularCycleScheduler (α : Type) where
  /-- Minimum learning rate within the cycle. -/
  baseLr : α
  /-- Maximum learning rate within the cycle. -/
  maxLr : α
  /-- Half-cycle size (in steps). -/
  stepSize : Nat
  /-- Current step counter (0-indexed). -/
  currentStep : Nat := 0

/--
Get the learning rate for the triangular cycle schedule at the current step.

This is the canonical "triangle up then down" schedule with fixed amplitude.

PyTorch analogy: `CyclicLR` in `"triangular"` mode.
-/
def TriangularCycleScheduler.getLr (scheduler : TriangularCycleScheduler α) : α :=
  if scheduler.stepSize = 0 then
    scheduler.baseLr
  else
    let cycle_step := scheduler.currentStep % (2 * scheduler.stepSize)
    if cycle_step < scheduler.stepSize then
      scheduler.baseLr + (scheduler.maxLr - scheduler.baseLr) * SchedulerUtils.ratioNat
        cycle_step scheduler.stepSize
    else
      let decStep := cycle_step - scheduler.stepSize
      scheduler.maxLr - (scheduler.maxLr - scheduler.baseLr) * SchedulerUtils.ratioNat decStep
        scheduler.stepSize

/--
Advance the triangular cycle scheduler by one step.

PyTorch analogy: `scheduler.step()`.
-/
def TriangularCycleScheduler.step (scheduler : TriangularCycleScheduler α) :
  TriangularCycleScheduler α :=
  { scheduler with currentStep := scheduler.currentStep + 1 }

/--
Create a triangular cycle scheduler starting at step `0`.

PyTorch analogy: `CyclicLR(base_lr=..., max_lr=..., mode=\"triangular\")`.
-/
def triangularCycleScheduler (baseLr : α) (maxLr : α) (stepSize : Nat) :
    TriangularCycleScheduler α :=
  { baseLr := baseLr, maxLr := maxLr, stepSize := stepSize }

/-! ## 1cycle Learning Rate Schedule -/

/--
One-cycle learning-rate schedule.

- increase LR from `initial_lr` to `max_lr` over the first `pct_start` fraction of steps,
- then decrease to `final_lr` over the rest.

In the original 1cycle policy, momentum is also scheduled; we keep this runtime version LR-only.
-/
structure OneCycleScheduler (α : Type) where
  /-- Peak learning rate (reached at `pct_start` of the schedule). -/
  maxLr : α
  /-- Total number of steps in the schedule. -/
  totalSteps : Nat
  /-- Learning rate at step `0`. -/
  initialLr : α
  /-- Learning rate after the full schedule finishes. -/
  finalLr : α
  /-- Divides `max_lr` to get `initial_lr` in the factory constructor. -/
  divFactor : α
  /-- Fraction of the schedule spent increasing LR (0..1). -/
  pctStart : α
  /-- Current step counter (0-indexed). -/
  currentStep : Nat := 0

/--
Get the learning rate for the one-cycle schedule at the current step.

This ramps up to `max_lr` over the `pct_start` fraction of the schedule, then anneals down to
`final_lr`.

PyTorch analogy: `torch.optim.lr_scheduler.OneCycleLR`, restricted here to the learning-rate curve.
-/
def OneCycleScheduler.getLr (scheduler : OneCycleScheduler α) : α :=
  if scheduler.totalSteps = 0 then
    scheduler.initialLr
  else if scheduler.currentStep >= scheduler.totalSteps then
    scheduler.finalLr
  else
    let stepInCycle := scheduler.currentStep
    let cycleStep := SchedulerUtils.ratioNat stepInCycle scheduler.totalSteps
    if cycleStep < scheduler.pctStart then
      let factor := cycleStep / scheduler.pctStart
      SchedulerUtils.linearInterpolation scheduler.initialLr scheduler.maxLr factor
    else
      let factor := (cycleStep - scheduler.pctStart) / (1 - scheduler.pctStart)
      SchedulerUtils.linearInterpolation scheduler.maxLr scheduler.finalLr factor

/--
Advance the 1cycle scheduler by one step.

PyTorch analogy: `scheduler.step()`.
-/
def OneCycleScheduler.step (scheduler : OneCycleScheduler α) : OneCycleScheduler α :=
  { scheduler with currentStep := scheduler.currentStep + 1 }

/--
Create a simplified 1cycle schedule starting at step `0`.

We derive `initial_lr := max_lr / div_factor` and `final_lr := max_lr / final_div_factor`.

PyTorch analogy: `torch.optim.lr_scheduler.OneCycleLR(max_lr=..., total_steps=...)`.
-/
def oneCycleScheduler (maxLr : α) (totalSteps : Nat) (divFactor : α) (pctStart : α)
    (finalDivFactor : α) :
    OneCycleScheduler α :=
  let initialLr := maxLr / divFactor
  let finalLr := maxLr / finalDivFactor
  { maxLr := maxLr, totalSteps := totalSteps, initialLr := initialLr, finalLr := finalLr
    divFactor := divFactor, pctStart := pctStart }

/-! ## LR finder -/

/--
Learning-rate finder schedule: exponential sweep from `initial_lr` to `final_lr` over `num_steps`.
-/
structure LRFinder (α : Type) where
  /-- Learning rate at step `0`. -/
  initialLr : α
  /-- Target learning rate at the end of the sweep. -/
  finalLr : α
  /-- Number of steps in the sweep. -/
  numSteps : Nat
  /-- Current step counter (0-indexed). -/
  currentStep : Nat := 0

/--
Get the learning rate for the LR-finder exponential sweep at the current step.

This increases LR exponentially from `initial_lr` toward `final_lr` across `num_steps`.

PyTorch analogy: LR finder utilities used by libraries like fastai, often implemented as a custom
  schedule.
-/
def LRFinder.getLr (finder : LRFinder α) : α :=
  if finder.numSteps = 0 then
    finder.initialLr
  else
    let progress := SchedulerUtils.ratioNat finder.currentStep finder.numSteps
    finder.initialLr * (finder.finalLr / finder.initialLr) ^ (progress : α)

/--
Advance the LR finder by one step.

PyTorch analogy: stepping a custom LR schedule inside a training loop.
-/
def LRFinder.step (finder : LRFinder α) : LRFinder α :=
  { finder with currentStep := finder.currentStep + 1 }

/--
Create an LR finder schedule starting at step `0`.

PyTorch analogy: setting up an LR finder run to sweep learning rates.
-/
def lrFinder (initialLr : α) (finalLr : α) (numSteps : Nat) : LRFinder α :=
  { initialLr := initialLr, finalLr := finalLr, numSteps := numSteps }

end Optim
