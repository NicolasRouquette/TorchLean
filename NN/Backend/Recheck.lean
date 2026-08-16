/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Audit

/-!
# Backend Recheck Obligations

Plan audits say which backend capsules were selected. Recheck obligations unpack the contract fields
inside those capsules: shape, layout, value, and VJP evidence.

This layer is intentionally data-level. It does not claim that a foreign kernel is proved. It
records whether each obligation is guarded, tested, fuzzed, delegated to a trusted boundary,
inapplicable, or missing evidence.
-/

@[expose] public section

namespace NN
namespace Backend

/-- The contract obligation attached to a backend capsule. -/
inductive ContractObligation where
  | shape
  | layout
  | value
  | vjp
  deriving DecidableEq, Repr

/-- Normalized evidence class used by reports and policy filters. -/
inductive EvidenceDisposition where
  | guarded
  | tested
  | fuzzed
  | trusted
  | notApplicable
  | missing
  deriving DecidableEq, Repr

namespace ContractEvidence

/-- Convert detailed evidence into the coarse class used by recheck reports. -/
def disposition (e : ContractEvidence) : EvidenceDisposition :=
  match e with
  | .runtimeGuard _ => .guarded
  | .testSuite _ => .tested
  | .fuzzOracle _ => .fuzzed
  | .trustedBoundary _ => .trusted
  | .notApplicable => .notApplicable
  | .notProvided => .missing

/-- Whether this evidence gives any non-missing justification. -/
def isProvided (e : ContractEvidence) : Bool :=
  e.disposition != .missing

/-- Whether this evidence crosses a trusted boundary. -/
def isTrustedBoundary (e : ContractEvidence) : Bool :=
  e.disposition == .trusted

end ContractEvidence

/-- Recheck row for one obligation of one selected backend kernel. -/
structure ObligationReport where
  op : BackendOp
  capsuleName : String
  obligation : ContractObligation
  claim : ContractClaim
  evidence : ContractEvidence
  disposition : EvidenceDisposition
  deriving Repr

namespace ObligationReport

/-- Whether this obligation has no recorded evidence. -/
def isMissing (r : ObligationReport) : Bool :=
  r.disposition == .missing

/-- Whether this obligation is discharged only by a trusted boundary. -/
def isTrusted (r : ObligationReport) : Bool :=
  r.disposition == .trusted

end ObligationReport

namespace KernelAudit

/-- All contract obligations associated with a selected kernel. -/
def obligationReports (a : KernelAudit) : List ObligationReport :=
  [ { op := a.op
      capsuleName := a.capsuleName
      obligation := .shape
      claim := a.shapeContract.claim
      evidence := a.shapeContract.evidence
      disposition := a.shapeContract.evidence.disposition }
  , { op := a.op
      capsuleName := a.capsuleName
      obligation := .layout
      claim := a.layoutContract.claim
      evidence := a.layoutContract.evidence
      disposition := a.layoutContract.evidence.disposition }
  , { op := a.op
      capsuleName := a.capsuleName
      obligation := .value
      claim := a.valueContract.claim
      evidence := a.valueContract.evidence
      disposition := a.valueContract.evidence.disposition }
  , { op := a.op
      capsuleName := a.capsuleName
      obligation := .vjp
      claim := a.vjpContract.claim
      evidence := a.vjpContract.evidence
      disposition := a.vjpContract.evidence.disposition } ]

/-- Obligations without any recorded evidence. -/
def missingObligations (a : KernelAudit) : List ContractObligation :=
  (a.obligationReports.filter ObligationReport.isMissing).map (·.obligation)

/-- Obligations discharged by a trusted boundary. -/
def trustedBoundaryObligations (a : KernelAudit) : List ContractObligation :=
  (a.obligationReports.filter ObligationReport.isTrusted).map (·.obligation)

end KernelAudit

namespace KernelPlanAudit

/-- All recheck obligations for all selected kernels. -/
def obligationReports (a : KernelPlanAudit) : List ObligationReport :=
  a.kernels.foldr (fun k acc => k.obligationReports ++ acc) []

/-- Recheck obligations with no recorded evidence. -/
def missingReports (a : KernelPlanAudit) : List ObligationReport :=
  a.obligationReports.filter ObligationReport.isMissing

/-- Recheck obligations discharged by trusted external boundaries. -/
def trustedBoundaryReports (a : KernelPlanAudit) : List ObligationReport :=
  a.obligationReports.filter ObligationReport.isTrusted

/-- Whether every obligation has a non-missing audit classification. -/
def hasNoMissingEvidence (a : KernelPlanAudit) : Bool :=
  a.missingReports.isEmpty

end KernelPlanAudit

namespace KernelPlan

/-- Whether every selected backend contract obligation has a non-missing audit classification. -/
def hasNoMissingEvidence (p : KernelPlan) : Bool :=
  p.audit.hasNoMissingEvidence

/-- Missing recheck obligations for a selected plan. -/
def missingReports (p : KernelPlan) : List ObligationReport :=
  p.audit.missingReports

/-- Trusted-boundary recheck obligations for a selected plan. -/
def trustedBoundaryReports (p : KernelPlan) : List ObligationReport :=
  p.audit.trustedBoundaryReports

end KernelPlan

end Backend
end NN
