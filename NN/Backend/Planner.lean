/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Availability

/-!
# Backend Planner

The planner is the bridge between a semantic graph and backend capsules.

At this stage it is deliberately small: given an execution config, an operation tag, and a capsule
registry, choose an admissible capsule or explain why none is available. Graph-aware layers can
recover those operation tags from `NN.IR.OpKind`, lower adjacent nodes, and eventually produce
executable command buffers without changing the contract story.
-/

@[expose] public section

namespace NN
namespace Backend

universe u v

/-- A backend choice for one graph operation or fused operation. -/
structure PlannedKernel where
  op : BackendOp
  capsule : KernelCapsule
  deriving Repr

/-- A simple execution plan: one selected capsule per requested operation. -/
structure ExecutionPlan where
  kernels : List PlannedKernel
  deriving Repr

namespace ExecutionPlan

/-- Names of the selected backend capsules, useful for audits and logs. -/
def capsuleNames (p : ExecutionPlan) : List String :=
  p.kernels.map fun k => k.capsule.name

/-- Whether every selected capsule is admitted by the config's trust/device/backend policy. -/
def admissible (cfg : ExecutionConfig) (p : ExecutionPlan) : Bool :=
  p.kernels.all fun k => k.capsule.admissible cfg

end ExecutionPlan

/-- Choose a backend capsule for one operation. -/
def planOp (cfg : ExecutionConfig) (registry : List KernelCapsule)
    (op : BackendOp) : Except String PlannedKernel := do
  match chooseCapsuleFor? cfg op registry with
  | some capsule => pure { op, capsule }
  | none =>
      throw s!"no admissible backend capsule for op {op.name} on device {cfg.device.cliName}"

/-- Choose backend capsules for a sequence of operations. -/
def planOps (cfg : ExecutionConfig) (registry : List KernelCapsule)
    (ops : List BackendOp) : Except String ExecutionPlan := do
  let kernels ← ops.mapM (planOp cfg registry)
  pure { kernels }

/-- Choose backend capsules after filtering the registry by machine/build availability. -/
def planOpsAvailable (cfg : ExecutionConfig) (availability : Availability)
    (registry : List KernelCapsule) (ops : List BackendOp) : Except String ExecutionPlan :=
  planOps cfg (availability.filterCapsules registry) ops

/-! ## Typed verified planning -/

/--
A proof-bearing kernel selected without erasing its implementation or refinement theorem.

Unlike `PlannedKernel`, this type is indexed by the exact Lean specification implemented by the
kernel. The `selectable` field records device, provider, VJP, contract-shape, and proof-oriented
policy checks. Execution can therefore recover the refinement theorem directly.
-/
structure VerifiedPlannedKernel (ι : Type u) (ο : Type v) (op : BackendOp)
    (specification : ι → ο) (cfg : ExecutionConfig) where
  kernel : ProofCarryingKernel ι ο op specification
  selectable : kernel.selectable cfg = true

namespace VerifiedPlannedKernel

/-- Execute the selected typed implementation. -/
def run {ι : Type u} {ο : Type v} {op : BackendOp} {specification : ι → ο}
    {cfg : ExecutionConfig} (planned : VerifiedPlannedKernel ι ο op specification cfg)
    (input : ι) : ο :=
  planned.kernel.run input

/-- Verified planning preserves the kernel's exact Lean specification. -/
theorem run_eq_specification {ι : Type u} {ο : Type v} {op : BackendOp}
    {specification : ι → ο} {cfg : ExecutionConfig}
    (planned : VerifiedPlannedKernel ι ο op specification cfg) (input : ι) :
    planned.run input = specification input :=
  planned.kernel.run_eq_specification input

end VerifiedPlannedKernel

/--
Select a proof-bearing implementation while retaining its refinement theorem.

This is the only planner entrypoint for `AssurancePolicy.verified`. The ordinary planner works with
erased capsule metadata and therefore rejects capsules whose trust level is merely labelled
`verified`.
-/
def planVerifiedKernel {ι : Type u} {ο : Type v} {op : BackendOp}
    {specification : ι → ο} (cfg : ExecutionConfig)
    (kernels : List (ProofCarryingKernel ι ο op specification)) :
    Except String (VerifiedPlannedKernel ι ο op specification cfg) :=
  match kernels with
  | [] =>
      throw <| s!"no selectable proof-carrying kernel for op {op.name} on device " ++
        s!"{cfg.device.cliName}"
  | kernel :: rest =>
      if h : kernel.selectable cfg = true then
        pure { kernel, selectable := h }
      else
        planVerifiedKernel cfg rest

end Backend
end NN
