/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# Learning-Rate Schedules

Pure learning-rate schedules used by TorchLean training loops and examples. A `Config` describes a
schedule, and `lrAt` evaluates it at an optimizer step.

## References

- PyTorch schedulers: `torch.optim.lr_scheduler.*`
  (`https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate`)
-/

@[expose] public section


namespace TorchLean
namespace Trainer
namespace Scheduler

/--
Small learning-rate scheduler surface for higher-level training code.

This file keeps the interface compact: a `Config` is just a description of a schedule,
and `lrAt cfg t` computes the learning rate at step/epoch index `t`.

### PyTorch mapping

`Config.step` and `Config.exponential` correspond to the schedule math of:
- `torch.optim.lr_scheduler.StepLR`
- `torch.optim.lr_scheduler.ExponentialLR`

`Config.warmupCosine` is the schedule commonly used for Transformer pretraining: a short linear
warm-up followed by cosine decay to a nonzero floor.
-/
inductive Config where
  | constant (lr : Float)
  | step (base : Float) (stepSize : Nat) (gamma : Float := 0.1)
  | exponential (base : Float) (gamma : Float)
  | warmupCosine
      (peak min : Float) (warmupSteps totalSteps : Nat)
  deriving Repr

/-- Constant learning-rate schedule. -/
def constant (lr : Float) : Config := .constant lr

/-- Step decay learning-rate schedule. -/
def step (base : Float) (stepSize : Nat) (gamma : Float := 0.1) : Config :=
  .step base stepSize gamma

/-- Exponential learning-rate schedule. -/
def exponential (base : Float) (gamma : Float) : Config :=
  .exponential base gamma

/--
Linearly warm up to `peak`, then follow a cosine curve down to `min`.

`warmupSteps` counts optimizer updates. The first update uses
`peak / warmupSteps`, and the last warm-up update reaches `peak`. Once `totalSteps` updates have
been scheduled, the learning rate remains at `min`. A warm-up longer than the run is clamped to
`totalSteps`. When `totalSteps = 0`, no update belongs to the schedule and `lrAt` returns `min`.
-/
def warmupCosine
    (peak min : Float) (warmupSteps totalSteps : Nat) : Config :=
  .warmupCosine peak min warmupSteps totalSteps

/-- Learning rate at a given step or epoch index. -/
def lrAt : Config → Nat → Float
  | .constant lr, _ => lr
  | .step base stepSize gamma, t =>
      if stepSize = 0 then
        base
      else
        let k := t / stepSize
        base * (Float.pow gamma (Float.ofNat k))
  | .exponential base gamma, t =>
      base * (Float.pow gamma (Float.ofNat t))
  | .warmupCosine peak min warmupSteps totalSteps, t =>
      if totalSteps = 0 then
        min
      else if t >= totalSteps then
        min
      else
        let warmupSteps := Nat.min warmupSteps totalSteps
        if t < warmupSteps then
          peak * Float.ofNat (t + 1) / Float.ofNat warmupSteps
        else
          let decaySteps := totalSteps - warmupSteps
          if decaySteps = 0 then
            min
          else
            let progress := Float.ofNat (t - warmupSteps) / Float.ofNat decaySteps
            let cosine := (1.0 + Float.cos (3.141592653589793 * progress)) / 2.0
            min + (peak - min) * cosine

end Scheduler
end Trainer
end TorchLean
