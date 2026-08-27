/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.Graph.ForwardApprox

/-!
# Numerical Contracts for Optimizer Steps

An optimizer proof has two kinds of state: the mathematical recurrence and the rounded runtime
state. `NumericalStepContract` records their relation once. A concrete optimizer supplies its exact
and runtime update equations, a transformer for state/parameter error bounds, and a proof that one
step preserves the relation. `run_approx` then composes that local proof over any finite gradient
stream.

This interface is deliberately independent of SGD, Adam, or a particular scalar backend. It avoids
duplicating an induction theorem for every optimizer and, unlike an equality theorem obtained by
unfolding two identical definitions, states the numerical refinement claim needed by training.

For the distinction between local rounding errors and their propagation through an iterative
algorithm, see N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., 2002.
-/

@[expose] public section

namespace Proofs.RuntimeApprox.Optimizer

open Spec

noncomputable section

/-- Per-step state and parameter error information computed by a numerical optimizer contract. -/
structure StepBound (StateBound : Shape → Type) (s : Shape) where
  /-- Bound object for the optimizer's private state after the step. -/
  state : StateBound s
  /-- Infinity-norm error budget for the parameter tensor after the step. -/
  params : ℝ

/-- A numerical refinement contract for one shape-polymorphic optimizer update.

`StepData` carries numerical information required only for the current update. It is `Unit` for
unconditional rules such as SGD, while adaptive optimizers use it for denominator margins and
rounded scalar-expression bounds. This lets one finite-run theorem cover both cases.
-/
structure NumericalStepContract (R : Type) (toSpec : R → ℝ) where
  /-- Stable optimizer name used in numerical reports. -/
  name : String
  /-- Mathematical optimizer state. -/
  StateSpec : Shape → Type
  /-- Rounded runtime optimizer state. -/
  StateRuntime : Shape → Type
  /-- Error information relating mathematical and runtime state. -/
  StateBound : Shape → Type
  /-- Numerical data and domain margins supplied for one update. -/
  StepData : Shape → Type
  /-- Relation certified between mathematical and runtime state. -/
  stateApprox : {s : Shape} → StateSpec s → StateRuntime s → StateBound s → Prop
  /-- Conditions under which one step's numerical data is valid. -/
  stepDataValid : {s : Shape} →
    StateSpec s → StateRuntime s → StateBound s →
    Tensor ℝ s → Tensor R s → ℝ →
    Tensor ℝ s → Tensor R s → ℝ → StepData s → Prop
  /-- One exact-real optimizer update. -/
  updateSpec : {s : Shape} → StateSpec s → Tensor ℝ s → Tensor ℝ s → StateSpec s × Tensor ℝ s
  /-- One rounded runtime optimizer update. -/
  updateRuntime : {s : Shape} →
    StateRuntime s → Tensor R s → Tensor R s → StateRuntime s × Tensor R s
  /-- Compute the next state/parameter bounds from current errors and runtime values. -/
  updateBound : {s : Shape} → StateBound s → ℝ → ℝ →
    StateRuntime s → Tensor R s → Tensor R s → StepData s → StepBound StateBound s
  /-- Proof-free scalar components of a state bound for reports and UI consumers. -/
  stateBoundReport : {s : Shape} → StateBound s → Array (String × ℝ)
  /-- Proof-free scalar components of one step's side data. -/
  stepDataReport : {s : Shape} → StepData s → Array (String × ℝ)
  /-- One-step numerical soundness. -/
  updateSound : ∀ {s : Shape}
      (stateS : StateSpec s) (stateR : StateRuntime s) (stateBound : StateBound s)
      (paramsS : Tensor ℝ s) (paramsR : Tensor R s) (paramsError : ℝ)
      (gradsS : Tensor ℝ s) (gradsR : Tensor R s) (gradsError : ℝ)
      (stepData : StepData s),
    stateApprox stateS stateR stateBound →
    approxTensor (α := R) (toSpec := toSpec) paramsS paramsR paramsError →
    approxTensor (α := R) (toSpec := toSpec) gradsS gradsR gradsError →
    stepDataValid stateS stateR stateBound paramsS paramsR paramsError
      gradsS gradsR gradsError stepData →
      let nextBound := updateBound stateBound paramsError gradsError stateR paramsR gradsR stepData
      stateApprox (updateSpec stateS paramsS gradsS).1
          (updateRuntime stateR paramsR gradsR).1 nextBound.state ∧
        approxTensor (α := R) (toSpec := toSpec)
          (updateSpec stateS paramsS gradsS).2
          (updateRuntime stateR paramsR gradsR).2 nextBound.params

namespace NumericalStepContract

variable {R : Type} {toSpec : R → ℝ}

/-- Exact state and parameters threaded through an optimizer run. -/
abbrev SpecStep (contract : NumericalStepContract R toSpec) (s : Shape) :=
  contract.StateSpec s × Tensor ℝ s

/-- Runtime state and parameters threaded through an optimizer run. -/
abbrev RuntimeStep (contract : NumericalStepContract R toSpec) (s : Shape) :=
  contract.StateRuntime s × Tensor R s

/-- Error information threaded through an optimizer run. -/
abbrev RunBound (contract : NumericalStepContract R toSpec) (s : Shape) :=
  StepBound contract.StateBound s

/-- Exact, rounded, and error information for one optimizer update. -/
structure StepInput (contract : NumericalStepContract R toSpec) (s : Shape) where
  /-- Exact-real gradient. -/
  gradSpec : Tensor ℝ s
  /-- Rounded runtime gradient. -/
  gradRuntime : Tensor R s
  /-- Infinity-norm error relating the exact and runtime gradients. -/
  gradError : ℝ
  /-- Optimizer-specific side data and domain margins. -/
  data : contract.StepData s

/-- Execute a finite step stream using the exact-real recurrence. -/
def runSpec (contract : NumericalStepContract R toSpec) {s : Shape}
    (initial : SpecStep contract s) (steps : Array (StepInput contract s)) :
    SpecStep contract s :=
  steps.foldl
    (fun current step => contract.updateSpec current.1 current.2 step.gradSpec) initial

/-- Execute the same finite step stream using the rounded runtime recurrence. -/
def runRuntime (contract : NumericalStepContract R toSpec) {s : Shape}
    (initial : RuntimeStep contract s) (steps : Array (StepInput contract s)) :
    RuntimeStep contract s :=
  steps.foldl
    (fun current step => contract.updateRuntime current.1 current.2 step.gradRuntime) initial

/-- Propagate state and parameter errors over a bundled optimizer step stream. -/
def runBounds (contract : NumericalStepContract R toSpec) {s : Shape}
    (initialBound : RunBound contract s) (initialRuntime : RuntimeStep contract s)
    (steps : Array (StepInput contract s)) : RunBound contract s :=
  (steps.foldl
    (fun (bound, runtime) step =>
      let nextBound := contract.updateBound bound.state bound.params step.gradError
        runtime.1 runtime.2 step.gradRuntime step.data
      let nextRuntime := contract.updateRuntime runtime.1 runtime.2 step.gradRuntime
      (nextBound, nextRuntime))
    (initialBound, initialRuntime)).1

/-- Approximation and side-condition evidence for a complete optimizer run.

The indices thread exact state, runtime state, and error bounds through the same recurrence used by
`runSpec`, `runRuntime`, and `runBounds`. Adaptive-domain conditions are therefore checked at the
step where they are needed rather than asserted once for an entire run.
-/
inductive StepStreamApprox (contract : NumericalStepContract R toSpec) {s : Shape} :
    SpecStep contract s → RuntimeStep contract s → RunBound contract s →
    Array (StepInput contract s) → Prop
  | empty {spec runtime bound} : StepStreamApprox contract spec runtime bound #[]
  | cons {spec runtime bound step steps} :
      approxTensor (α := R) (toSpec := toSpec)
        step.gradSpec step.gradRuntime step.gradError →
      contract.stepDataValid spec.1 runtime.1 bound.state spec.2 runtime.2 bound.params
        step.gradSpec step.gradRuntime step.gradError step.data →
      StepStreamApprox contract
        (contract.updateSpec spec.1 spec.2 step.gradSpec)
        (contract.updateRuntime runtime.1 runtime.2 step.gradRuntime)
        (contract.updateBound bound.state bound.params step.gradError
          runtime.1 runtime.2 step.gradRuntime step.data)
        steps →
      StepStreamApprox contract spec runtime bound
        (#[step] ++ steps)

/-- Final soundness statement associated with one finite optimizer run. -/
def RunSound (contract : NumericalStepContract R toSpec) {s : Shape}
    (spec : SpecStep contract s) (runtime : RuntimeStep contract s) (bound : RunBound contract s)
    (steps : Array (StepInput contract s)) : Prop :=
  contract.stateApprox
      (contract.runSpec spec steps).1
      (contract.runRuntime runtime steps).1
      (contract.runBounds bound runtime steps).state ∧
    approxTensor (α := R) (toSpec := toSpec)
      (contract.runSpec spec steps).2
      (contract.runRuntime runtime steps).2
      (contract.runBounds bound runtime steps).params

/-- A local optimizer contract composes over any finite validated gradient stream. -/
theorem run_approx (contract : NumericalStepContract R toSpec) {s : Shape}
    {spec : SpecStep contract s} {runtime : RuntimeStep contract s}
    {bound : RunBound contract s}
    {steps : Array (StepInput contract s)}
    (hsteps : StepStreamApprox contract spec runtime bound steps) :
    contract.stateApprox spec.1 runtime.1 bound.state →
    approxTensor (α := R) (toSpec := toSpec) spec.2 runtime.2 bound.params →
    RunSound contract spec runtime bound steps := by
  cases hsteps with
  | empty =>
      intro hstate hparams
      exact ⟨hstate, hparams⟩
  | @cons spec runtime bound step steps hgrad hvalid tail =>
      intro hstate hparams
      have hstep := contract.updateSound spec.1 runtime.1 bound.state
        spec.2 runtime.2 bound.params step.gradSpec step.gradRuntime step.gradError step.data
        hstate hparams hgrad hvalid
      simpa [RunSound, runSpec, runRuntime, runBounds] using run_approx contract
        (spec := contract.updateSpec spec.1 spec.2 step.gradSpec)
        (runtime := contract.updateRuntime runtime.1 runtime.2 step.gradRuntime)
        (bound := contract.updateBound bound.state bound.params step.gradError
          runtime.1 runtime.2 step.gradRuntime step.data)
        (steps := steps)
        tail hstep.1 hstep.2
termination_by steps.size
decreasing_by simp

end NumericalStepContract

end
end Proofs.RuntimeApprox.Optimizer
