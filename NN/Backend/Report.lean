/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Profile

/-!
# Kernel Selection Reports

Small human-readable reports for contract-carrying kernel plans.

The planner data is intentionally precise; these helpers are the user-facing explanation layer. They
are useful in examples, command-line choosers, docs, and debugging output.
-/

@[expose] public section

namespace NN
namespace Backend

namespace Provider

/-- Short stable spelling for a backend provider. -/
def label : Provider → String
  | .reference => "reference"
  | .torchLean => "torchlean"
  | .nativeCuda => "native-cuda"
  | .libTorch => "libtorch"
  | .aten => "aten"
  | .mps => "mps"
  | .webGpu => "webgpu"
  | .cuBLAS => "cublas"
  | .cuDNN => "cudnn"
  | .cuFFT => "cufft"
  | .xla => "xla"
  | .neuron => "neuron"
  | .customChip => "custom-chip"
  | .external => "external"

end Provider

namespace TrustLevel

/-- Short stable spelling for a capsule trust level. -/
def label : TrustLevel → String
  | .verified => "verified"
  | .checked => "checked"
  | .fuzzed => "fuzzed"
  | .trustedExternal => "trusted-external"

end TrustLevel

namespace AssurancePolicy

/-- Short stable spelling for a profile assurance policy. -/
def label (policy : AssurancePolicy) : String :=
  if policy == .verified then "verified"
  else if policy == .checked then "checked"
  else if policy == .external then "external"
  else "custom"

end AssurancePolicy

namespace VJPMode

/-- Short stable spelling for how a capsule handles backward/VJP. -/
def label : VJPMode → String
  | .none => "none"
  | .torchLeanTape => "torchlean-tape"
  | .backendVJP => "backend-vjp"

end VJPMode

namespace TensorLayout

/-- Short stable spelling for a tensor-layout contract. -/
def label : TensorLayout → String
  | .canonicalTensor => "canonical-tensor"
  | .flatRowMajor => "flat-row-major"
  | .libTorchCudaView => "libtorch-cuda-view"

end TensorLayout

namespace ContractClaim

/-- Human-readable statement of a structured backend obligation. -/
def label : ContractClaim → String
  | .shapeSafety op => s!"shape safety for {op.name}"
  | .layoutCompatibility op layout =>
      s!"{layout.label} layout compatibility for {op.name}"
  | .valueRefinement op => s!"{op.name} forward refines its TorchLean semantics"
  | .vjpRefinement op mode =>
      s!"{op.name} {mode.label} VJP refines its TorchLean semantics"
  | .vjpUnavailable op => s!"{op.name} has no VJP in this capsule"

end ContractClaim

namespace ContractEvidence

/-- Concise description of the evidence attached to a contract claim. -/
def label : ContractEvidence → String
  | .runtimeGuard name => s!"guarded at runtime by {name}"
  | .testSuite name => s!"covered by test suite {name}"
  | .fuzzOracle name => s!"compared by fuzz oracle {name}"
  | .trustedBoundary reason => s!"trusted boundary: {reason}"
  | .notApplicable => "not applicable"
  | .notProvided => "no evidence recorded"

end ContractEvidence

namespace ContractDescriptor

/-- One report line for a named contract field. -/
def reportLine (field : String) (d : ContractDescriptor) : String :=
  s!"    {field}: {d.claim.label}; {d.evidence.label}"

end ContractDescriptor

namespace KernelAudit

/-- One-line summary for a selected backend capsule. -/
def reportLine (a : KernelAudit) : String :=
  s!"  {a.op.name}: {a.capsuleName} " ++
  s!"provider={a.provider.label} trust={a.trustLevel.label} vjp={a.vjpMode.label} " ++
  s!"numeric=[{a.numericalPolicy.reportLabel}]"

/-- Full contract report for a selected backend capsule. -/
def detailedReportLines (a : KernelAudit) : Array String :=
  #[ a.reportLine
  , a.shapeContract.reportLine "shape"
  , a.layoutContract.reportLine "layout"
  , a.valueContract.reportLine "value"
  , a.vjpContract.reportLine "vjp"
  , s!"    numerical: {a.numericalPolicy.reportLabel}" ]

end KernelAudit

namespace KernelPlanAudit

/-- Human-readable lines for all selected capsules. -/
def reportLines (a : KernelPlanAudit) : Array String :=
  a.kernels.map KernelAudit.reportLine

/-- Human-readable contract details for all selected capsules. -/
def detailedReportLines (a : KernelPlanAudit) : Array String :=
  a.kernels.flatMap KernelAudit.detailedReportLines

end KernelPlanAudit

namespace KernelPlan

/-- Human-readable lines for a selected kernel plan. -/
def reportLines (p : KernelPlan) : Array String :=
  p.audit.reportLines

/-- Human-readable contract details for a selected kernel plan. -/
def detailedReportLines (p : KernelPlan) : Array String :=
  p.audit.detailedReportLines

end KernelPlan

namespace BackendProfile

/-- One-line profile description for logs and interactive choosers. -/
def summary (p : BackendProfile) : String :=
  s!"profile={p.name} device={p.policy.device.cliName} assurance={p.policy.assurance.label} " ++
  s!"vjp={p.policy.vjpMode.label}"

/-- Plan a list of backend ops and format the selected capsules. -/
def planReport (p : BackendProfile) (ops : Array BackendOp) : Except String String := do
  let plan ← p.planOps ops
  let boundary :=
    if plan.hasTrustedExternal then
      "trusted external boundary: " ++ String.intercalate ", " plan.trustedExternalOps.toList
    else
      "trusted external boundary: none"
  pure <| String.intercalate "\n" <|
    (#[p.summary, boundary] ++ plan.detailedReportLines).toList

end BackendProfile

end Backend
end NN
