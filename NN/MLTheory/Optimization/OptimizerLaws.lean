/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Optim.Optimizers

/-!
# Optimizer Law Interface

This module gives TorchLean optimizers a small proof layer interface.

Runtime optimizers live in `NN.Runtime.Optim.Optimizers` as executable tensor equations.  The
definitions below package those equations as shape-polymorphic optimizers and provide a common
interface for independent update specifications.

The pattern for adding an optimizer is:

1. define a pure per-tensor `init` and `update` equation;
2. package it as a `TensorOptimizer`;
3. state an independent `StepSpec` when a proof-facing recurrence is needed;
4. prove optimizer-specific algebraic facts as consequences of that generic interface.

TorchLean does not register a second, definitionally identical copy of every runtime update. Such a
copy would add a theorem name without adding an independent claim. Higher-level trainer proofs can
instead quantify over any `TensorOptimizer`, reason about whole gradient streams via `runSteps`,
and introduce a `StepSpec` only when its equations come from a separate mathematical description.
-/

@[expose] public section

namespace Optim

open Spec
open Tensor

variable {α : Type} [Context α]

/-- A shape-polymorphic per-tensor optimizer. -/
structure TensorOptimizer (α : Type) [Context α] where
  /-- Per-parameter optimizer state for a tensor of shape `s`. -/
  State : Shape → Type
  /-- Initialize optimizer state from the current parameter tensor. -/
  init : {s : Shape} → Tensor α s → State s
  /-- One update from state, parameters, and gradients. -/
  update : {s : Shape} → State s → Tensor α s → Tensor α s → State s × Tensor α s

namespace TensorOptimizer

section ConcreteOptimizers

variable [DecidableRel ((· > ·) : α → α → Prop)]

/-- Package plain SGD as a `TensorOptimizer`. -/
def sgd (lr : α) : TensorOptimizer α :=
  { State := SGD.State α
    init := fun {s} p => SGD.init (α := α) (s := s) lr p
    update := fun {_s} st p g => (st, SGD.update (α := α) st p g) }

/-- Package momentum SGD as a `TensorOptimizer`. -/
def momentumSGD (lr momentum : α) : TensorOptimizer α :=
  { State := MomentumSGD.State α
    init := fun {s} p => MomentumSGD.init (α := α) (s := s) lr momentum p
    update := fun {_s} st p g => MomentumSGD.update (α := α) st p g }

/-- Package AdaGrad as a `TensorOptimizer`. -/
def adagrad (lr epsilon : α) : TensorOptimizer α :=
  { State := AdaGrad.State α
    init := fun {s} p => AdaGrad.init (α := α) (s := s) lr epsilon p
    update := fun {_s} st p g => AdaGrad.update (α := α) st p g }

/-- Package RMSProp as a `TensorOptimizer`. -/
def rmsprop (lr decay epsilon : α) : TensorOptimizer α :=
  { State := RMSProp.State α
    init := fun {s} p => RMSProp.init (α := α) (s := s) lr decay epsilon p
    update := fun {_s} st p g => RMSProp.update (α := α) st p g }

/-- Package Adam as a `TensorOptimizer`. -/
def adam (lr beta1 beta2 epsilon : α) : TensorOptimizer α :=
  { State := Adam.State α
    init := fun {s} p => Adam.init (α := α) (s := s) lr beta1 beta2 epsilon p
    update := fun {_s} st p g => Adam.update (α := α) st p g }

/-- Package AdamW as a `TensorOptimizer`. -/
def adamw (lr weightDecay beta1 beta2 epsilon : α) : TensorOptimizer α :=
  { State := AdamW.State α
    init := fun {s} p => AdamW.init (α := α) (s := s) lr weightDecay beta1 beta2 epsilon p
    update := fun {_s} st p g => AdamW.update (α := α) st p g }

/-- Package Adadelta as a `TensorOptimizer`. -/
def adadelta (lr rho epsilon : α) : TensorOptimizer α :=
  { State := Adadelta.State α
    init := fun {s} p => Adadelta.init (α := α) (s := s) lr rho epsilon p
    update := fun {_s} st p g => Adadelta.update (α := α) st p g }

/-- Package Muon-style orthogonalized momentum as a `TensorOptimizer`. -/
def muon (lr momentum : α)
    (orthogonalizer : {s : Shape} → Muon.Orthogonalizer α s :=
      fun {s} => Muon.identityOrthogonalizer (α := α) (s := s)) :
    TensorOptimizer α :=
  { State := Muon.State α
    init := fun {s} p => Muon.init (α := α) (s := s) lr momentum (orthogonalizer (s := s)) p
    update := fun {_s} st p g => Muon.update (α := α) st p g }

end ConcreteOptimizers

/-- State/parameter pair threaded by an optimizer for one fixed tensor shape. -/
abbrev Step (opt : TensorOptimizer α) (s : Shape) :=
  opt.State s × Tensor α s

/-- Run one optimizer step on a state/parameter pair. -/
def step (opt : TensorOptimizer α) {s : Shape}
    (current : Step opt s) (grads : Tensor α s) : Step opt s :=
  opt.update current.1 current.2 grads

/-- Run a finite stream of gradients through an optimizer. -/
def runSteps (opt : TensorOptimizer α) {s : Shape}
    (current : Step opt s) (grads : Array (Tensor α s)) : Step opt s :=
  grads.foldl (fun step grad => opt.step step grad) current

/--
Splitting a gradient stream and running the two pieces sequentially gives the same state and
parameters as running the concatenated stream.
-/
theorem runSteps_append (opt : TensorOptimizer α) {s : Shape}
    (current : Step opt s) (left right : Array (Tensor α s)) :
    opt.runSteps current (left ++ right) = opt.runSteps (opt.runSteps current left) right := by
  simp [runSteps]

/-- Optimizer state after a finite gradient stream. -/
def stateAfter (opt : TensorOptimizer α) {s : Shape}
    (current : Step opt s) (grads : Array (Tensor α s)) : opt.State s :=
  (opt.runSteps current grads).1

/-- Optimizer parameters after a finite gradient stream. -/
def paramsAfter (opt : TensorOptimizer α) {s : Shape}
    (current : Step opt s) (grads : Array (Tensor α s)) : Tensor α s :=
  (opt.runSteps current grads).2

/-- State projection of `runSteps_append`. -/
theorem stateAfter_append (opt : TensorOptimizer α) {s : Shape}
    (current : Step opt s) (left right : Array (Tensor α s)) :
    opt.stateAfter current (left ++ right) =
      opt.stateAfter (opt.runSteps current left) right := by
  exact congrArg Prod.fst (opt.runSteps_append current left right)

/-- Parameter projection of `runSteps_append`. -/
theorem paramsAfter_append (opt : TensorOptimizer α) {s : Shape}
    (current : Step opt s) (left right : Array (Tensor α s)) :
    opt.paramsAfter current (left ++ right) =
      opt.paramsAfter (opt.runSteps current left) right := by
  exact congrArg Prod.snd (opt.runSteps_append current left right)

end TensorOptimizer

/-! ## Generic step specifications -/

/--
Proof-facing specification of one optimizer step.

An optimizer-specific file only has to identify the next-state and next-parameter equations once.
The generic theorems below then lift that one-step fact to whole finite gradient streams.
-/
structure StepSpec (opt : TensorOptimizer α) where
  /-- Spec equation for the next optimizer state. -/
  nextState : {s : Shape} → opt.State s → Tensor α s → Tensor α s → opt.State s
  /-- Spec equation for the next parameter tensor. -/
  nextParams : {s : Shape} → opt.State s → Tensor α s → Tensor α s → Tensor α s
  /-- The executable optimizer update agrees with the stated step equations. -/
  update_eq : ∀ {s : Shape} (state : opt.State s) (params grads : Tensor α s),
    opt.update state params grads = (nextState state params grads, nextParams state params grads)

namespace StepSpec

variable {opt : TensorOptimizer α}

/-- Run one step through the proof layer equations. -/
def step (law : StepSpec opt) {s : Shape}
    (current : TensorOptimizer.Step opt s) (grads : Tensor α s) :
    TensorOptimizer.Step opt s :=
  (law.nextState current.1 current.2 grads, law.nextParams current.1 current.2 grads)

/-- Run a finite stream of gradients through the proof layer equations. -/
def runSteps (law : StepSpec opt) {s : Shape}
    (current : TensorOptimizer.Step opt s) (grads : Array (Tensor α s)) :
    TensorOptimizer.Step opt s :=
  grads.foldl (fun step grad => law.step step grad) current

/-- A registered step spec agrees with the executable optimizer for one step. -/
theorem step_eq_optimizer_step (law : StepSpec opt) {s : Shape}
    (current : TensorOptimizer.Step opt s) (grads : Tensor α s) :
    law.step current grads = opt.step current grads := by
  cases current with
  | mk state params =>
      simp [step, TensorOptimizer.step, law.update_eq]

/--
A registered one-step optimizer spec agrees with the executable optimizer over any finite gradient
stream.  This is the general theorem optimizer-specific registrations feed into.
-/
theorem runSteps_eq_optimizer_runSteps (law : StepSpec opt) {s : Shape}
    (current : TensorOptimizer.Step opt s) (grads : Array (Tensor α s)) :
    law.runSteps current grads = opt.runSteps current grads := by
  have hstep :
      (fun (step : TensorOptimizer.Step opt s) (grad : Tensor α s) => law.step step grad) =
        (fun step grad => opt.step step grad) := by
    funext step grad
    exact law.step_eq_optimizer_step step grad
  unfold runSteps TensorOptimizer.runSteps
  rw [hstep]

/--
The proof layer equations compose over concatenated gradient streams just like the executable
optimizer.
-/
theorem runSteps_append (law : StepSpec opt) {s : Shape}
    (current : TensorOptimizer.Step opt s)
    (left right : Array (Tensor α s)) :
    law.runSteps current (left ++ right) = law.runSteps (law.runSteps current left) right := by
  simp [runSteps]

end StepSpec

/-! ## Muon comparison laws -/

variable [DecidableRel ((· > ·) : α → α → Prop)]

namespace Muon

/--
If a Muon backend returns the fresh momentum buffer unchanged on this step, then the parameter
update agrees with momentum SGD for this step.
-/
theorem update_params_eq_momentumSGD_of_apply_eq {s : Shape}
    (state : State α s) (params grads : Tensor α s)
    (happly :
      state.orthogonalizer.apply
        (OptimizerUtils.updateMomentumBuf state.buf state.momentum grads) =
        OptimizerUtils.updateMomentumBuf state.buf state.momentum grads) :
    (update state params grads).2 =
      (MomentumSGD.update
        ({ lr := state.lr, momentum := state.momentum, buf := state.buf } :
          MomentumSGD.State α s)
        params grads).2 := by
  simp [update, MomentumSGD.update, happly]

/--
Initialized version of `update_params_eq_momentumSGD_of_apply_eq`.
-/
theorem init_update_params_eq_momentumSGD_of_apply_eq {s : Shape}
    (lr momentum : α) (orthogonalizer : Orthogonalizer α s)
    (params grads : Tensor α s)
    (happly :
      orthogonalizer.apply (OptimizerUtils.updateMomentumBuf (fill 0 s) momentum grads) =
        OptimizerUtils.updateMomentumBuf (fill 0 s) momentum grads) :
    (update (init lr momentum orthogonalizer params) params grads).2 =
      (MomentumSGD.update (MomentumSGD.init lr momentum params) params grads).2 := by
  exact update_params_eq_momentumSGD_of_apply_eq
    (state := init lr momentum orthogonalizer params)
    (params := params) (grads := grads) happly

end Muon

end Optim
