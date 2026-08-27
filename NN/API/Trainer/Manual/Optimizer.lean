/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Optim.Config
public import NN.API.Trainer.Manual.Core
public import NN.API.Trainer.Scheduler

/-!
# Manual Optimizer Scheduling

Learning-rate scheduling helpers for shape-indexed optimizer state used by manual training loops.
-/

@[expose] public section

namespace TorchLean
namespace Trainer
namespace Manual

open SeqTask

namespace Internal

/-- Bind a configured optimizer, its initialized state, and its scheduled state updater. -/
def withBoundOptimizer {σ τ : Spec.Shape} {task : SeqTask σ τ} {α β : Type}
    [_root_.Context α] [DecidableEq Spec.Shape] [_root_.TorchLean.Runtime.FromFloat α]
    (runner : Runner α task) (config : _root_.TorchLean.optim.Optimizer)
    (scheduler : Option _root_.TorchLean.Trainer.Scheduler.Config)
    (k :
      (optimizer : _root_.Runtime.Autograd.TorchLean.Optim.Optimizer α (stateShapes task)) →
      optimizer.State → (Nat → optimizer.State → optimizer.State) → IO β) : IO β := do
  let stepLr (step : Nat) : Float :=
    scheduler.map (fun cfg => _root_.TorchLean.Trainer.Scheduler.lrAt cfg step)
      |>.getD config.learningRate
  let rec mapStateList {State : Type → Spec.Shape → Type} :
      {ss : List Spec.Shape} →
      ({s : Spec.Shape} → State α s → State α s) →
      _root_.Runtime.Autograd.TorchLean.Optim.StateList State α ss →
      _root_.Runtime.Autograd.TorchLean.Optim.StateList State α ss
    | [], _, .nil => .nil
    | _ :: ss, f, .cons st rest => .cons (f st) (mapStateList (ss := ss) f rest)
  match config with
  | .sgd lr momentum =>
      if momentum == 0.0 then
        let optimizer := _root_.Runtime.Autograd.TorchLean.Optim.sgd
          (α := α) (paramShapes := stateShapes task) (_root_.TorchLean.Runtime.ofFloat lr)
        let state ← TorchLean.Module.initOptimizer runner.module optimizer
        k optimizer state fun step state =>
          mapStateList (fun st =>
            { st with lr := _root_.TorchLean.Runtime.ofFloat (stepLr step) }) state
      else
        let optimizer := _root_.Runtime.Autograd.TorchLean.Optim.momentumSGD
          (α := α) (paramShapes := stateShapes task)
          (_root_.TorchLean.Runtime.ofFloat lr) (_root_.TorchLean.Runtime.ofFloat momentum)
        let state ← TorchLean.Module.initOptimizer runner.module optimizer
        k optimizer state fun step state =>
          mapStateList (fun st =>
            { st with lr := _root_.TorchLean.Runtime.ofFloat (stepLr step) }) state
  | .adagrad lr epsilon =>
      let optimizer := _root_.Runtime.Autograd.TorchLean.Optim.adagrad
        (α := α) (paramShapes := stateShapes task)
        (_root_.TorchLean.Runtime.ofFloat lr) (_root_.TorchLean.Runtime.ofFloat epsilon)
      let state ← TorchLean.Module.initOptimizer runner.module optimizer
      k optimizer state fun step state =>
        mapStateList (fun st =>
          { st with lr := _root_.TorchLean.Runtime.ofFloat (stepLr step) }) state
  | .rmsprop lr decay epsilon =>
      let optimizer := _root_.Runtime.Autograd.TorchLean.Optim.rmsprop
        (α := α) (paramShapes := stateShapes task)
        (_root_.TorchLean.Runtime.ofFloat lr) (_root_.TorchLean.Runtime.ofFloat decay)
        (_root_.TorchLean.Runtime.ofFloat epsilon)
      let state ← TorchLean.Module.initOptimizer runner.module optimizer
      k optimizer state fun step state =>
        mapStateList (fun st =>
          { st with lr := _root_.TorchLean.Runtime.ofFloat (stepLr step) }) state
  | .adam lr beta1 beta2 epsilon =>
      let optimizer := _root_.Runtime.Autograd.TorchLean.Optim.adam
        (α := α) (paramShapes := stateShapes task)
        (_root_.TorchLean.Runtime.ofFloat lr) (_root_.TorchLean.Runtime.ofFloat beta1)
        (_root_.TorchLean.Runtime.ofFloat beta2) (_root_.TorchLean.Runtime.ofFloat epsilon)
      let state ← TorchLean.Module.initOptimizer runner.module optimizer
      k optimizer state fun step state =>
        mapStateList (fun st =>
          { st with lr := _root_.TorchLean.Runtime.ofFloat (stepLr step) }) state
  | .adamw lr weightDecay beta1 beta2 epsilon =>
      let optimizer := _root_.Runtime.Autograd.TorchLean.Optim.adamw
        (α := α) (paramShapes := stateShapes task)
        (_root_.TorchLean.Runtime.ofFloat lr) (_root_.TorchLean.Runtime.ofFloat weightDecay)
        (_root_.TorchLean.Runtime.ofFloat beta1) (_root_.TorchLean.Runtime.ofFloat beta2)
        (_root_.TorchLean.Runtime.ofFloat epsilon)
      let state ← TorchLean.Module.initOptimizer runner.module optimizer
      k optimizer state fun step state =>
        mapStateList (fun st =>
          { st with lr := _root_.TorchLean.Runtime.ofFloat (stepLr step) }) state
  | .adadelta lr rho epsilon =>
      let optimizer := _root_.Runtime.Autograd.TorchLean.Optim.adadelta
        (α := α) (paramShapes := stateShapes task)
        (_root_.TorchLean.Runtime.ofFloat lr) (_root_.TorchLean.Runtime.ofFloat rho)
        (_root_.TorchLean.Runtime.ofFloat epsilon)
      let state ← TorchLean.Module.initOptimizer runner.module optimizer
      k optimizer state fun step state =>
        mapStateList (fun st =>
          { st with lr := _root_.TorchLean.Runtime.ofFloat (stepLr step) }) state

end Internal
end Manual
end Trainer
end TorchLean
