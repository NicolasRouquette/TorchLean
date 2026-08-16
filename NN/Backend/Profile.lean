/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Accepted

/-!
# Backend Profiles

Named backend profiles bundle the choices that should move together:

- target/build availability,
- kernel-selection policy,
- ordered capsule modules,
- graph-plan grouping mode,
- and assurance policy.

This gives downstream APIs one object to pass around instead of separately threading flags such as
"CUDA", "LibTorch enabled", "trusted external allowed", and "coalesce adjacent graph nodes".
-/

@[expose] public section

namespace NN
namespace Backend

/-- How adjacent graph-node plans are grouped for scheduling and audit. -/
inductive GroupingMode where
  | singleton
  | coalesced
  deriving DecidableEq, Repr

/-- A named kernel-selection profile for one target. -/
structure BackendProfile where
  /-- Human-readable profile name used in diagnostics and reports. -/
  name : String
  /-- Device, provider preference, assurance policy, and VJP ownership. -/
  policy : KernelPolicy
  /-- Operating-system, architecture, and accelerator capabilities available to planning. -/
  target : Target
  /-- Capsule modules used to construct and validate the profile's planning registry. -/
  capsuleModules : List Registry.CapsuleModule := Registry.maintainedModules
  /-- Whether accepted nodes remain separate or share compatible scheduling/audit groups. -/
  groupingMode : GroupingMode := .coalesced
  deriving Repr

namespace BackendProfile

/--
Extend a profile with operation/provider capsule modules without changing model semantics, target
selection, grouping, or assurance policy.

A new contribution replaces an existing module with the same name. Duplicate names within `modules`
are still rejected when the profile is planned.
-/
def withCapsuleModules (p : BackendProfile) (modules : List Registry.CapsuleModule) :
    BackendProfile :=
  let names := modules.map (·.name)
  { p with
    capsuleModules :=
      modules ++ p.capsuleModules.filter (fun old => !names.contains old.name) }

/-- Planner capabilities declared by the profile target; runtime execution probes separately. -/
def availability (p : BackendProfile) : Availability :=
  p.target.declaredAvailability

/-- Capsule registry selected by the profile. -/
def registry (p : BackendProfile) : List KernelCapsule :=
  Registry.flatten p.capsuleModules

/-- Select capsules using the profile registry, target, and kernel policy. -/
def planOps (p : BackendProfile) (ops : List BackendOp) : Except String KernelPlan := do
  Registry.validateModules p.capsuleModules
  NN.Backend.planOpsAvailable p.policy p.availability p.registry ops

/-- Gate a planned operation sequence under the profile's assurance policy. -/
def gateOps (p : BackendProfile) (ops : List BackendOp) : Except String GateResult := do
  let plan ← p.planOps ops
  pure <| plan.gate p.policy.assurance

/-- Select a capsule for every runtime-relevant IR node. -/
def planGraphNodes (p : BackendProfile) (g : NN.IR.Graph) :
    Except String NN.Backend.IR.GraphKernelPlan := do
  Registry.validateModules p.capsuleModules
  NN.Backend.IR.checkedPlanGraph p.policy p.availability p.registry g

/-- Group a graph kernel plan according to the profile's audit grouping mode. -/
def groupGraphPlan (p : BackendProfile) (plan : NN.Backend.IR.GraphKernelPlan) :
    GroupedKernelPlan :=
  match p.groupingMode with
  | .singleton => plan.toSingletonGroups
  | .coalesced => plan.toCoalescedGroups

/-- Plan, group, and gate a graph under the profile. -/
def acceptGraph (p : BackendProfile) (g : NN.IR.Graph) :
    Except String GraphKernelPlanResult := do
  let graphPlan ← p.planGraphNodes g
  let groupedPlan := p.groupGraphPlan graphPlan
  pure <| acceptGraphKernelPlan graphPlan groupedPlan p.policy.assurance

/-- Maintained portable CPU/reference profile with runtime guards and regression evidence. -/
def checkedCpu : BackendProfile :=
  { name := "checked_cpu"
    policy :=
      { device := .cpu
        provider := .auto
        assurance := .checked
        vjpMode := .torchLeanTape }
    target := Target.portableCpu
    capsuleModules := Registry.maintainedModules
    groupingMode := .coalesced }

/-- Checked CPU/reference profile for a named operating system and architecture. -/
def checkedCpuTarget (os : OperatingSystem) (arch : Architecture := .unknown) : BackendProfile :=
  { checkedCpu with
    name := s!"checked_{reprStr os}_{reprStr arch}_cpu"
    target := Target.portableCpu os arch }

/-- Checked native CUDA profile. External trusted providers are not admitted. -/
def checkedCuda : BackendProfile :=
  { name := "checked_cuda"
    policy :=
      { device := .cuda
        provider := .prefer .torchLean
        assurance := .checked
        vjpMode := .torchLeanTape }
    target := Target.linuxCuda false
    capsuleModules := Registry.maintainedModules
    groupingMode := .coalesced }

/--
LibTorch forward scaling profile.

LibTorch is allowed to provide selected forward values, but TorchLean still records the graph/tape
boundary and does not hand local backward ownership to LibTorch autograd.
-/
def libTorchForwardCuda : BackendProfile :=
  { name := "libtorch_forward_cuda"
    policy :=
      { device := .cuda
        provider := .prefer .libTorch
        assurance := .external
        vjpMode := .torchLeanTape }
    target := Target.linuxCuda true
    capsuleModules := Registry.maintainedModules ++ [Registry.libTorchModule]
    groupingMode := .coalesced }

/-- Checked macOS CPU/reference profile. -/
def macOSCpu : BackendProfile :=
  checkedCpuTarget .macOS .aarch64

/-- Checked Windows CPU/reference profile. Used for CI bring-up before native Windows kernels exist. -/
def windowsCpu : BackendProfile :=
  checkedCpuTarget .windows .x86_64

/--
Maintained execution profile for a device, when TorchLean currently provides one.

Targets such as Metal, ROCm, TPU, and custom accelerators remain expressible through `Target` and
caller-supplied profiles. They are not silently converted into profiles with empty registries.
-/
def maintainedForDevice? : Device → Option BackendProfile
  | .cpu => some checkedCpu
  | .cuda => some checkedCuda
  | .rocm | .metal | .wasm | .tpu | .trainium | .custom | .external => none

/-- Whether this profile registers at least one capsule for its selected device. -/
def hasDeviceCapsule (profile : BackendProfile) : Bool :=
  profile.registry.any fun capsule =>
    capsule.device == profile.policy.device

end BackendProfile

end Backend
end NN
