/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Types

/-!
# Backend Capsules

A backend capsule is TorchLean's unit of delegation to fast code.

The capsule records the contract TorchLean expects from a foreign implementation: which operation
it implements, which spec it refines, what layout and shape
conventions are assumed, how the value/VJP claims are justified, and what trust level the planner
must account for. The contract does not prove the implementation.
-/

@[expose] public section

namespace NN
namespace Backend

/-- A source file inside the TorchLean checkout that supports a backend contract. -/
structure SourceRef where
  path : String
  deriving DecidableEq, BEq, Repr

/--
Reference to a native/FFI symbol used by a backend capsule.

The linter checks that `path` exists, that `symbol` occurs in that source file, and that
`buildTarget?`, when present, names a Lake target in `lakefile.lean`.
-/
structure NativeSymbolRef where
  path : String
  symbol : String
  buildTarget? : Option String := none
  deriving DecidableEq, BEq, Repr

/-- Source-level provenance for a contract descriptor. Provenance is not correctness evidence. -/
inductive ContractProvenance where
  | sourceFile (ref : SourceRef)
  | nativeSymbol (ref : NativeSymbolRef)
  | note (text : String)
  deriving DecidableEq, BEq, Repr

/-- Concrete tensor-layout convention named by a backend contract. -/
inductive TensorLayout where
  /-- TorchLean's ordinary typed tensor representation. -/
  | canonicalTensor
  /-- Contiguous flat storage with the last axis varying fastest. -/
  | flatRowMajor
  /-- A contiguous CUDA tensor view owned by LibTorch. -/
  | libTorchCudaView
  deriving DecidableEq, Repr

/-- A structured backend obligation, independent of how evidence for it is obtained. -/
inductive ContractClaim where
  /-- Inputs and outputs satisfy the shape rule associated with an operation. -/
  | shapeSafety (op : BackendOp)
  /-- Runtime buffers use the declared layout for an operation. -/
  | layoutCompatibility (op : BackendOp) (layout : TensorLayout)
  /-- Forward execution refines the named mathematical specification. -/
  | valueRefinement (op : BackendOp) (specName : String)
  /-- Local backward execution refines the named VJP specification. -/
  | vjpRefinement (op : BackendOp) (specName : String) (mode : VJPMode)
  deriving DecidableEq, Repr

/-- How a capsule justifies one part of its contract.

`runtimeGuard` and `testSuite` record useful engineering assurance but do not discharge a theorem.
Only `theorem` and `checker` contain proof terms.
-/
inductive ContractEvidence where
  | theorem (theoremName : String) (statement : Prop) (proof : statement)
  | checker (checkerName : String) (statement : Prop) (accepted : Bool)
      (sound : accepted = true -> statement) (acceptanceProof : accepted = true)
  | runtimeGuard (name : String)
  | testSuite (name : String)
  | fuzzOracle (name : String)
  | trustedBoundary (reason : String)
  | notProvided

instance : Repr ContractEvidence where
  reprPrec evidence _ := Std.Format.text <| match evidence with
    | .theorem name .. => s!"theorem({name})"
    | .checker name .. => s!"checker({name})"
    | .runtimeGuard name => s!"runtimeGuard({name})"
    | .testSuite name => s!"testSuite({name})"
    | .fuzzOracle name => s!"fuzzOracle({name})"
    | .trustedBoundary reason => s!"trustedBoundary({reason})"
    | .notProvided => "notProvided"

/-- A structured contract claim together with its evidence and human-readable explanation. -/
structure ContractDescriptor where
  claim : ContractClaim
  summary : String
  evidence : ContractEvidence
  provenance : List ContractProvenance := []

instance : Repr ContractDescriptor where
  reprPrec d _ := Std.Format.text s!"ContractDescriptor({repr d.claim}, {repr d.evidence})"

namespace ContractDescriptor

/-- A contract claim enforced by a named runtime guard. -/
def guarded (claim : ContractClaim) (summary guard : String)
    (provenance : List ContractProvenance := []) : ContractDescriptor :=
  { claim, summary, evidence := .runtimeGuard guard, provenance }

/-- A contract claim covered by a named regression suite. -/
def tested (claim : ContractClaim) (summary suite : String)
    (provenance : List ContractProvenance := []) : ContractDescriptor :=
  { claim, summary, evidence := .testSuite suite, provenance }

/-- A contract claim delegated to an explicitly named trusted boundary. -/
def trusted (claim : ContractClaim) (summary reason : String)
    (provenance : List ContractProvenance := []) : ContractDescriptor :=
  { claim, summary, evidence := .trustedBoundary reason, provenance }

end ContractDescriptor

/-! ## Numerical execution policy

These fields describe floating-point choices that are invisible at the tensor-shape level but
matter to numerical certificates. They are metadata, not correctness evidence: the ordinary value
and VJP contracts still state what a capsule refines and how that claim is justified.
-/

/-- Rounding behavior advertised by a backend capsule. -/
inductive RoundingPolicy where
  | scalarContext
  | nearestEven
  | directed
  | implementationDefined
  | unspecified
  deriving DecidableEq, Repr

/-- Treatment of subnormal values at the backend boundary. -/
inductive SubnormalPolicy where
  | gradualUnderflow
  | flushToZero
  | implementationDefined
  | unspecified
  deriving DecidableEq, Repr

/-- Whether multiplication and addition may be contracted into one fused operation. -/
inductive ContractionPolicy where
  | separate
  | fused
  | implementationDefined
  | notApplicable
  | unspecified
  deriving DecidableEq, Repr

/-- Ordering contract for reductions. Different valid orders need not be bitwise equal. -/
inductive ReductionPolicy where
  | fixedLeft
  | fixedTree
  | implementationDefined
  | notApplicable
  | unspecified
  deriving DecidableEq, Repr

/-- Floating-point choices attached to one kernel capsule. -/
structure NumericalPolicy where
  rounding : RoundingPolicy := .unspecified
  subnormals : SubnormalPolicy := .unspecified
  contraction : ContractionPolicy := .unspecified
  reduction : ReductionPolicy := .unspecified
  deriving DecidableEq, Repr

namespace RoundingPolicy

/-- Stable report label for rounding behavior. -/
def label : RoundingPolicy -> String
  | .scalarContext => "scalar-context"
  | .nearestEven => "nearest-even"
  | .directed => "directed"
  | .implementationDefined => "implementation-defined"
  | .unspecified => "unspecified"

end RoundingPolicy

namespace SubnormalPolicy

/-- Stable report label for subnormal handling. -/
def label : SubnormalPolicy -> String
  | .gradualUnderflow => "gradual-underflow"
  | .flushToZero => "flush-to-zero"
  | .implementationDefined => "implementation-defined"
  | .unspecified => "unspecified"

end SubnormalPolicy

namespace ContractionPolicy

/-- Stable report label for multiply-add contraction. -/
def label : ContractionPolicy -> String
  | .separate => "separate"
  | .fused => "fused"
  | .implementationDefined => "implementation-defined"
  | .notApplicable => "n/a"
  | .unspecified => "unspecified"

end ContractionPolicy

namespace ReductionPolicy

/-- Stable report label for reduction order. -/
def label : ReductionPolicy -> String
  | .fixedLeft => "fixed-left"
  | .fixedTree => "fixed-tree"
  | .implementationDefined => "implementation-defined"
  | .notApplicable => "n/a"
  | .unspecified => "unspecified"

end ReductionPolicy

namespace NumericalPolicy

/-- Compact representation used in execution audits. -/
def reportLabel (policy : NumericalPolicy) : String :=
  s!"round={policy.rounding.label},subnormal={policy.subnormals.label}," ++
    s!"contract={policy.contraction.label},reduce={policy.reduction.label}"

end NumericalPolicy

/-- The four contract fields carried by every kernel capsule. -/
inductive ContractObligationKind where
  | shape
  | layout
  | value
  | vjp
  deriving DecidableEq, Repr

namespace ContractClaim

/-- Whether a claim has the expected kind and operation for a capsule contract field. -/
def matchesObligation (op : BackendOp) (vjpMode : VJPMode) :
    ContractObligationKind → ContractClaim → Bool
  | .shape, .shapeSafety claimOp => claimOp == op
  | .layout, .layoutCompatibility claimOp _ => claimOp == op
  | .value, .valueRefinement claimOp _ => claimOp == op
  | .vjp, .vjpRefinement claimOp _ claimMode => claimOp == op && claimMode == vjpMode
  | _, _ => false

end ContractClaim

/-- A contract-carrying fast kernel or reference implementation. -/
structure KernelCapsule where
  name : String
  op : BackendOp
  provider : Provider
  device : Device
  trustLevel : TrustLevel
  supportsForward : Bool := true
  vjpMode : VJPMode := .none
  shapeContract : ContractDescriptor
  layoutContract : ContractDescriptor
  valueContract : ContractDescriptor
  vjpContract : ContractDescriptor
  numericalPolicy : NumericalPolicy := {}
  notes : String := ""
  deriving Repr

namespace KernelCapsule

/-- Whether each descriptor states the obligation advertised by its field.

Evidence is useful only when it proves or checks the right claim. This guard prevents, for example,
a value-refinement theorem from being placed in the shape field and then accepted as shape evidence.
-/
def contractsAligned (c : KernelCapsule) : Bool :=
  c.shapeContract.claim.matchesObligation c.op c.vjpMode .shape &&
  c.layoutContract.claim.matchesObligation c.op c.vjpMode .layout &&
  c.valueContract.claim.matchesObligation c.op c.vjpMode .value &&
  c.vjpContract.claim.matchesObligation c.op c.vjpMode .vjp

/-- Stable identity used when adjacent graph nodes select the same registered capsule. -/
def sameIdentity (a b : KernelCapsule) : Bool :=
  a.name == b.name && a.op == b.op && a.provider == b.provider && a.device == b.device

/-- Validate the common part of an eager executor request.

This does not decide how an operation invokes a provider. It prevents every runtime operation from
reimplementing the op, device, and wiring checks before interpreting the capsule locally.
-/
def validateEagerRequest (c : KernelCapsule) (op : BackendOp) (device : Device) :
    Except String Unit := do
  unless c.op == op do
    throw s!"capsule `{c.name}` implements `{c.op.name}`, not `{op.name}`"
  unless c.device == device do
    throw s!"capsule `{c.name}` targets `{c.device.cliName}`, not `{device.cliName}`"

/-- Whether the assurance policy admits this capsule. -/
def allowedBy (cfg : ExecutionConfig) (c : KernelCapsule) : Bool :=
  cfg.assurance.acceptsTrust c.trustLevel

/-- Whether the backend preference admits this capsule's provider. -/
def matchesPreference (cfg : ExecutionConfig) (c : KernelCapsule) : Bool :=
  match cfg.backend with
  | .auto => true
  | .prefer _ => true
  | .only p => p = c.provider

/-- Whether this capsule is available on the selected device. -/
def matchesDevice (cfg : ExecutionConfig) (c : KernelCapsule) : Bool :=
  cfg.device = c.device

/--
Whether the capsule's gradient boundary is compatible with the requested execution config.

`none` is inference mode, so any forward-capable capsule is suitable even when it also advertises a
VJP. A backend VJP is compatible with normal TorchLean tape mode: TorchLean still owns the node and
calls the selected backend only for that local derivative computation.
-/
def matchesVJP (cfg : ExecutionConfig) (c : KernelCapsule) : Bool :=
  if cfg.vjpMode == .none || !c.op.requiresVJP then
    true
  else
    match cfg.vjpMode with
    | .none => true
    | .torchLeanTape => c.vjpMode == .torchLeanTape || c.vjpMode == .backendVJP
    | .backendVJP => c.vjpMode == .backendVJP

/-- Planner-side admissibility predicate for a single capsule. -/
def admissible (cfg : ExecutionConfig) (c : KernelCapsule) : Bool :=
  c.supportsForward && c.contractsAligned && c.allowedBy cfg && c.matchesPreference cfg &&
    c.matchesDevice cfg && c.matchesVJP cfg

end KernelCapsule

/--
Pick an admissible capsule for a typed operation.

An `.only` preference filters the catalog through `admissible`. An `.auto` preference preserves
catalog order. A `.prefer provider` request first searches that provider and then falls back to the
ordinary catalog, so preference does not depend on module registration order.
-/
def chooseCapsuleFor? (cfg : ExecutionConfig) (op : BackendOp)
    (capsules : List KernelCapsule) : Option KernelCapsule :=
  let eligible := fun c => c.op == op && c.admissible cfg
  match cfg.backend with
  | .prefer provider =>
      (capsules.find? fun c => eligible c && c.provider == provider).orElse fun _ =>
        capsules.find? eligible
  | .auto | .only _ =>
      capsules.find? eligible

end Backend
end NN
