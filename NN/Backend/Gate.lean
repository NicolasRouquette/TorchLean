/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Recheck
public import NN.Backend.Lowering

/-!
# Backend Acceptance Gates

Policy gates for contract-carrying backend plans.

The planner can select a backend capsule and the audit/recheck layers can report its obligations.
The gate layer turns those reports into an explicit yes/no decision before a plan is accepted for a
particular run mode.
-/

@[expose] public section

namespace NN
namespace Backend

/-- Whether an obligation disposition is admitted by an assurance policy. -/
def AssurancePolicy.acceptsDisposition (p : AssurancePolicy) (d : EvidenceDisposition) : Bool :=
  match d with
  | .missing => !p.requireEvidence
  | .trusted => p.allowTrustedExternal
  | .fuzzed => p.allowFuzzed
  | .guarded => p.allowRuntimeGuards
  | .tested => p.allowTestEvidence
  | .notApplicable => true

/-- Why a candidate plan was rejected by an acceptance gate. -/
inductive GateFailure where
  | missingEvidence (reports : List ObligationReport)
  | runtimeGuardEvidence (reports : List ObligationReport)
  | testEvidence (reports : List ObligationReport)
  | trustedBoundary (reports : List ObligationReport)
  | fuzzEvidence (reports : List ObligationReport)
  deriving Repr

/-- Result of applying an acceptance policy to an execution audit. -/
inductive GateResult where
  | accepted
  | rejected (failures : List GateFailure)
  deriving Repr

namespace ObligationReport

/-- Whether this obligation report is admitted by an acceptance policy. -/
def acceptedBy (policy : AssurancePolicy) (r : ObligationReport) : Bool :=
  policy.acceptsDisposition r.disposition

/-- Whether this obligation is fuzz-backed. -/
def isFuzzed (r : ObligationReport) : Bool :=
  r.disposition == .fuzzed

def isGuarded (r : ObligationReport) : Bool :=
  r.disposition == .guarded

def isTested (r : ObligationReport) : Bool :=
  r.disposition == .tested

end ObligationReport

namespace ExecutionAudit

def guardedReports (a : ExecutionAudit) : List ObligationReport :=
  a.obligationReports.filter ObligationReport.isGuarded

def testedReports (a : ExecutionAudit) : List ObligationReport :=
  a.obligationReports.filter ObligationReport.isTested

/-- Fuzz-backed recheck obligations. -/
def fuzzReports (a : ExecutionAudit) : List ObligationReport :=
  a.obligationReports.filter ObligationReport.isFuzzed

/-- Gate failures induced by an acceptance policy. -/
def gateFailures (policy : AssurancePolicy) (a : ExecutionAudit) : List GateFailure :=
  let missing :=
    if policy.requireEvidence then
      match a.missingReports with
      | [] => []
      | reports => [GateFailure.missingEvidence reports]
    else
      []
  let trusted :=
    if policy.allowTrustedExternal then
      []
    else
      match a.trustedBoundaryReports with
      | [] => []
      | reports => [GateFailure.trustedBoundary reports]
  let guarded :=
    if policy.allowRuntimeGuards then [] else
      match a.guardedReports with
      | [] => []
      | reports => [GateFailure.runtimeGuardEvidence reports]
  let tested :=
    if policy.allowTestEvidence then [] else
      match a.testedReports with
      | [] => []
      | reports => [GateFailure.testEvidence reports]
  let fuzzed :=
    if policy.allowFuzzed then
      []
    else
      match a.fuzzReports with
      | [] => []
      | reports => [GateFailure.fuzzEvidence reports]
  missing ++ guarded ++ tested ++ trusted ++ fuzzed

/-- Apply an acceptance policy to an execution audit. -/
def gate (policy : AssurancePolicy) (a : ExecutionAudit) : GateResult :=
  match a.gateFailures policy with
  | [] => .accepted
  | failures => .rejected failures

/-- An audit is accepted by a policy exactly when the policy reports no gate failures. -/
theorem gate_eq_accepted_iff_gateFailures_eq_nil
    (policy : AssurancePolicy) (a : ExecutionAudit) :
    a.gate policy = .accepted ↔ a.gateFailures policy = [] := by
  unfold gate
  cases a.gateFailures policy <;> simp

end ExecutionAudit

namespace ExecutionPlan

/-- Apply an acceptance policy to a selected execution plan. -/
def gate (policy : AssurancePolicy) (p : ExecutionPlan) : GateResult :=
  p.audit.gate policy

/-- Whether a selected execution plan is accepted by a policy. -/
def acceptedBy (policy : AssurancePolicy) (p : ExecutionPlan) : Bool :=
  match p.gate policy with
  | .accepted => true
  | .rejected _ => false

end ExecutionPlan

/-- One planned operation whose capsule has passed an acceptance policy. -/
structure AcceptedKernel where
  op : BackendOp
  capsule : KernelCapsule
  policy : AssurancePolicy
  gateProof : ({ kernels := [{ op, capsule }] } : ExecutionPlan).gate policy = .accepted

instance : Repr AcceptedKernel where
  reprPrec k _ := Std.Format.text s!"AcceptedKernel({k.op.name}, {k.capsule.name})"

/-- Gate a planned kernel and return a value that an executor can consume only on success. -/
def PlannedKernel.accept (policy : AssurancePolicy) (k : PlannedKernel) :
    Except (List GateFailure) AcceptedKernel :=
  let plan : ExecutionPlan := { kernels := [k] }
  match h : plan.gate policy with
  | .accepted => .ok { op := k.op, capsule := k.capsule, policy, gateProof := h }
  | .rejected failures => .error failures

namespace GraphLoweringPlan

/-- Apply an acceptance policy to a lowered backend plan. -/
def gate (policy : AssurancePolicy) (p : GraphLoweringPlan) : GateResult :=
  p.toExecutionPlan.gate policy

/-- Whether a lowered backend plan is accepted by a policy. -/
def acceptedBy (policy : AssurancePolicy) (p : GraphLoweringPlan) : Bool :=
  p.toExecutionPlan.acceptedBy policy

end GraphLoweringPlan

end Backend
end NN
