/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Recheck
public import NN.Backend.Grouping

/-!
# Backend Acceptance Gates

Policy gates for contract-carrying kernel plans.

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
  | missingEvidence (reports : Array ObligationReport)
  | runtimeGuardEvidence (reports : Array ObligationReport)
  | testEvidence (reports : Array ObligationReport)
  | trustedBoundary (reports : Array ObligationReport)
  | fuzzEvidence (reports : Array ObligationReport)
  deriving Repr

/-- Result of applying an acceptance policy to an execution audit. -/
inductive GateResult where
  | accepted
  | rejected (failures : Array GateFailure)
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

namespace KernelPlanAudit

def guardedReports (a : KernelPlanAudit) : Array ObligationReport :=
  a.obligationReports.filter ObligationReport.isGuarded

def testedReports (a : KernelPlanAudit) : Array ObligationReport :=
  a.obligationReports.filter ObligationReport.isTested

/-- Fuzz-backed recheck obligations. -/
def fuzzReports (a : KernelPlanAudit) : Array ObligationReport :=
  a.obligationReports.filter ObligationReport.isFuzzed

/-- Gate failures induced by an acceptance policy. -/
def gateFailures (policy : AssurancePolicy) (a : KernelPlanAudit) : Array GateFailure :=
  let missing :=
    if policy.requireEvidence then
      let reports := a.missingReports
      if reports.isEmpty then #[] else #[GateFailure.missingEvidence reports]
    else
      #[]
  let trusted :=
    if policy.allowTrustedExternal then
      #[]
    else
      let reports := a.trustedBoundaryReports
      if reports.isEmpty then #[] else #[GateFailure.trustedBoundary reports]
  let guarded :=
    if policy.allowRuntimeGuards then #[] else
      let reports := a.guardedReports
      if reports.isEmpty then #[] else #[GateFailure.runtimeGuardEvidence reports]
  let tested :=
    if policy.allowTestEvidence then #[] else
      let reports := a.testedReports
      if reports.isEmpty then #[] else #[GateFailure.testEvidence reports]
  let fuzzed :=
    if policy.allowFuzzed then
      #[]
    else
      let reports := a.fuzzReports
      if reports.isEmpty then #[] else #[GateFailure.fuzzEvidence reports]
  missing ++ guarded ++ tested ++ trusted ++ fuzzed

/-- Apply an acceptance policy to an execution audit. -/
def gate (policy : AssurancePolicy) (a : KernelPlanAudit) : GateResult :=
  let failures := a.gateFailures policy
  if failures.isEmpty then .accepted else .rejected failures

/-- An audit is accepted by a policy exactly when the policy reports no gate failures. -/
theorem gate_eq_accepted_iff_gateFailures_eq_empty
    (policy : AssurancePolicy) (a : KernelPlanAudit) :
    a.gate policy = .accepted ↔ a.gateFailures policy = #[] := by
  simp [gate, Array.isEmpty_iff]

end KernelPlanAudit

namespace KernelPlan

/-- Apply an acceptance policy to a selected execution plan. -/
def gate (policy : AssurancePolicy) (p : KernelPlan) : GateResult :=
  p.audit.gate policy

/-- Whether a selected execution plan is accepted by a policy. -/
def acceptedBy (policy : AssurancePolicy) (p : KernelPlan) : Bool :=
  match p.gate policy with
  | .accepted => true
  | .rejected _ => false

end KernelPlan

/-- One planned operation whose capsule has passed an acceptance policy. -/
structure AcceptedKernel where
  op : BackendOp
  capsule : KernelCapsule
  policy : AssurancePolicy
  gateProof : ({ kernels := #[{ op, capsule }] } : KernelPlan).gate policy = .accepted

instance : Repr AcceptedKernel where
  reprPrec k _ := Std.Format.text s!"AcceptedKernel({k.op.name}, {k.capsule.name})"

/-- Gate a planned kernel and return a value that an executor can consume only on success. -/
def PlannedKernel.accept (policy : AssurancePolicy) (k : PlannedKernel) :
    Except (Array GateFailure) AcceptedKernel :=
  let plan : KernelPlan := { kernels := #[k] }
  match h : plan.gate policy with
  | .accepted => .ok { op := k.op, capsule := k.capsule, policy, gateProof := h }
  | .rejected failures => .error failures

namespace GroupedKernelPlan

/-- Apply an acceptance policy to a grouped kernel plan. -/
def gate (policy : AssurancePolicy) (p : GroupedKernelPlan) : GateResult :=
  p.toKernelPlan.gate policy

/-- Whether a grouped kernel plan is accepted by a policy. -/
def acceptedBy (policy : AssurancePolicy) (p : GroupedKernelPlan) : Bool :=
  p.toKernelPlan.acceptedBy policy

end GroupedKernelPlan

end Backend
end NN
