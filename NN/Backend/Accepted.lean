/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Gate

/-!
# Accepted Kernel Plans

One entry point for the backend-contract pipeline.

This module ties together target availability, graph planning, conservative grouping, recheck, and
acceptance gates. It is intentionally still data-level: it produces a grouped plan that has passed
a policy gate, or it returns the gate failures. Grouping is audit and scheduling metadata; it does
not claim semantic lowering, kernel fusion, or execution.
-/

@[expose] public section

namespace NN
namespace Backend

/-- A graph kernel plan after grouping and assurance-gate checking. -/
structure AcceptedGraphKernelPlan where
  graphPlan : IR.GraphKernelPlan
  groupedPlan : GroupedKernelPlan
  policy : AssurancePolicy
  gateProof : groupedPlan.gate policy = .accepted

instance : Repr AcceptedGraphKernelPlan where
  reprPrec p _ := Std.Format.text s!"AcceptedGraphKernelPlan({repr p.groupedPlan})"

namespace AcceptedGraphKernelPlan

/-- Source IR node ids covered by the accepted grouped plan. -/
def nodeIds (p : AcceptedGraphKernelPlan) : Array Nat :=
  p.groupedPlan.nodeIds

/-- Selected capsule names in accepted group order. -/
def capsuleNames (p : AcceptedGraphKernelPlan) : Array String :=
  p.groupedPlan.capsuleNames

/-- Audit for the accepted grouped plan. -/
def audit (p : AcceptedGraphKernelPlan) : KernelPlanAudit :=
  p.groupedPlan.audit

/-- Recheck reports for the accepted grouped plan. -/
def obligationReports (p : AcceptedGraphKernelPlan) : Array ObligationReport :=
  p.audit.obligationReports

end AcceptedGraphKernelPlan

/-- Result of planning, grouping, and gating a graph. -/
inductive GraphKernelPlanResult where
  | accepted (plan : AcceptedGraphKernelPlan)
  | rejected (groupedPlan : GroupedKernelPlan) (failures : Array GateFailure)
  deriving Repr

namespace GraphKernelPlanResult

/-- Whether the pipeline returned an accepted plan. -/
def isAccepted : GraphKernelPlanResult → Bool
  | .accepted _ => true
  | .rejected .. => false

/-- Gate failures when the pipeline rejected the plan. -/
def failures : GraphKernelPlanResult → Array GateFailure
  | .accepted _ => #[]
  | .rejected _ failures => failures

end GraphKernelPlanResult

/-- Gate a grouped graph plan and expose it only when every obligation passes policy. -/
def acceptGraphKernelPlan (graphPlan : IR.GraphKernelPlan) (groupedPlan : GroupedKernelPlan)
    (policy : AssurancePolicy) : GraphKernelPlanResult :=
  match h : groupedPlan.gate policy with
  | .accepted => .accepted { graphPlan, groupedPlan, policy, gateProof := h }
  | .rejected failures => .rejected groupedPlan failures

end Backend
end NN
