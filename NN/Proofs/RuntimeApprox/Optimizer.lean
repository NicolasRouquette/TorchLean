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
  stateBoundReport : {s : Shape} → StateBound s → List (String × ℝ)
  /-- Proof-free scalar components of one step's side data. -/
  stepDataReport : {s : Shape} → StepData s → List (String × ℝ)
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

/-- Execute a finite gradient stream using the exact-real recurrence. -/
def runSpec (contract : NumericalStepContract R toSpec) {s : Shape} :
    SpecStep contract s → List (Tensor ℝ s) → SpecStep contract s
  | current, [] => current
  | current, grad :: rest =>
      runSpec contract (contract.updateSpec current.1 current.2 grad) rest

/-- Execute the same finite gradient stream using the rounded runtime recurrence. -/
def runRuntime (contract : NumericalStepContract R toSpec) {s : Shape} :
    RuntimeStep contract s → List (Tensor R s) → RuntimeStep contract s
  | current, [] => current
  | current, grad :: rest =>
      runRuntime contract (contract.updateRuntime current.1 current.2 grad) rest

/-- Propagate state and parameter errors over a runtime gradient stream. -/
def runBounds (contract : NumericalStepContract R toSpec) {s : Shape} :
    RunBound contract s → RuntimeStep contract s → List (Tensor R s) → List ℝ →
      List (contract.StepData s) →
      Option (RunBound contract s)
  | bound, _, [], [], [] => some bound
  | bound, current, grad :: grads, gradError :: gradErrors, stepData :: steps =>
      let nextBound := contract.updateBound bound.state bound.params gradError
        current.1 current.2 grad stepData
      let nextRuntime := contract.updateRuntime current.1 current.2 grad
      runBounds contract nextBound nextRuntime grads gradErrors steps
  | _, _, _, _, _ => none

/-- Approximation and side-condition evidence for a complete optimizer run.

The indices thread exact state, runtime state, and error bounds through the same recurrence used by
`runSpec`, `runRuntime`, and `runBounds`. Adaptive-domain conditions are therefore checked at the
step where they are needed rather than asserted once for an entire run.
-/
inductive StepStreamApprox (contract : NumericalStepContract R toSpec) {s : Shape} :
    SpecStep contract s → RuntimeStep contract s → RunBound contract s →
    List (Tensor ℝ s) → List (Tensor R s) → List ℝ → List (contract.StepData s) → Prop
  | nil {spec runtime bound} : StepStreamApprox contract spec runtime bound [] [] [] []
  | cons {spec runtime bound gradS gradR error stepData gradsS gradsR errors steps} :
      approxTensor (α := R) (toSpec := toSpec) gradS gradR error →
      contract.stepDataValid spec.1 runtime.1 bound.state spec.2 runtime.2 bound.params
        gradS gradR error stepData →
      StepStreamApprox contract
        (contract.updateSpec spec.1 spec.2 gradS)
        (contract.updateRuntime runtime.1 runtime.2 gradR)
        (contract.updateBound bound.state bound.params error runtime.1 runtime.2 gradR stepData)
        gradsS gradsR errors steps →
      StepStreamApprox contract spec runtime bound
        (gradS :: gradsS) (gradR :: gradsR) (error :: errors) (stepData :: steps)

/-- Final soundness statement associated with one finite optimizer run. -/
def RunSound (contract : NumericalStepContract R toSpec) {s : Shape}
    (spec : SpecStep contract s) (runtime : RuntimeStep contract s) (bound : RunBound contract s)
    (gradsS : List (Tensor ℝ s)) (gradsR : List (Tensor R s))
    (gradErrors : List ℝ) (steps : List (contract.StepData s)) : Prop :=
    ∃ finalBound,
      contract.runBounds bound runtime gradsR gradErrors steps = some finalBound ∧
        contract.stateApprox
          (contract.runSpec spec gradsS).1
          (contract.runRuntime runtime gradsR).1 finalBound.state ∧
        approxTensor (α := R) (toSpec := toSpec)
          (contract.runSpec spec gradsS).2
          (contract.runRuntime runtime gradsR).2 finalBound.params

/-- A local optimizer contract composes over any finite validated gradient stream. -/
theorem run_approx (contract : NumericalStepContract R toSpec) {s : Shape}
    {spec : SpecStep contract s} {runtime : RuntimeStep contract s}
    {bound : RunBound contract s}
    {gradsS : List (Tensor ℝ s)} {gradsR : List (Tensor R s)} {gradErrors : List ℝ}
    {steps : List (contract.StepData s)}
    (hsteps : StepStreamApprox contract spec runtime bound gradsS gradsR gradErrors steps) :
    contract.stateApprox spec.1 runtime.1 bound.state →
    approxTensor (α := R) (toSpec := toSpec) spec.2 runtime.2 bound.params →
    RunSound contract spec runtime bound gradsS gradsR gradErrors steps := by
  cases hsteps with
  | nil =>
      intro hstate hparams
      exact ⟨bound, rfl, hstate, hparams⟩
  | @cons spec runtime bound gradS gradR gradError stepData gradsS gradsR gradErrors steps
      hgrad hvalid tail =>
      intro hstate hparams
      have hstep := contract.updateSound spec.1 runtime.1 bound.state
        spec.2 runtime.2 bound.params gradS gradR gradError stepData
        hstate hparams hgrad hvalid
      exact run_approx contract
        (spec := contract.updateSpec spec.1 spec.2 gradS)
        (runtime := contract.updateRuntime runtime.1 runtime.2 gradR)
        (bound := contract.updateBound bound.state bound.params gradError
          runtime.1 runtime.2 gradR stepData)
        (gradsS := gradsS) (gradsR := gradsR) (gradErrors := gradErrors) (steps := steps)
        tail hstep.1 hstep.2
termination_by steps.length

end NumericalStepContract

end
end Proofs.RuntimeApprox.Optimizer
