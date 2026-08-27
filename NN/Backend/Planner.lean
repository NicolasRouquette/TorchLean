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

At this stage it is deliberately small: given a kernel policy, an operation tag, and a capsule
registry, choose an admissible capsule or explain why none is available. Graph-aware layers can
recover those operation tags from `NN.IR.OpKind`, group adjacent nodes, and eventually produce
executable command buffers while preserving the declared contracts.
-/

@[expose] public section

namespace NN
namespace Backend

universe u v

/-- A selected kernel capsule for one graph operation or fused operation. -/
structure PlannedKernel where
  op : BackendOp
  capsule : KernelCapsule
  deriving Repr

/-- One selected kernel capsule per requested operation. This record does not execute them. -/
structure KernelPlan where
  kernels : Array PlannedKernel
  deriving Repr

namespace KernelPlan

/-- Names of the selected backend capsules, useful for audits and logs. -/
def capsuleNames (p : KernelPlan) : Array String :=
  p.kernels.map fun k => k.capsule.name

/-- Whether every selected capsule is admitted by the provider, device, and assurance policy. -/
def admissible (policy : KernelPolicy) (p : KernelPlan) : Bool :=
  p.kernels.all fun k => k.capsule.admissible policy

end KernelPlan

/-- Choose a backend capsule for one operation. -/
def planOp (policy : KernelPolicy) (registry : Array KernelCapsule)
    (op : BackendOp) : Except String PlannedKernel := do
  match chooseCapsuleFor? policy op registry with
  | some capsule => pure { op, capsule }
  | none =>
      throw s!"no admissible kernel capsule for op {op.name} on device {policy.device.cliName}"

/-- Choose backend capsules for a sequence of operations. -/
def planOps (policy : KernelPolicy) (registry : Array KernelCapsule)
    (ops : Array BackendOp) : Except String KernelPlan := do
  let kernels ← ops.mapM (planOp policy registry)
  pure { kernels }

/-- Choose backend capsules after filtering the registry by machine/build availability. -/
def planOpsAvailable (policy : KernelPolicy) (availability : Availability)
    (registry : Array KernelCapsule) (ops : Array BackendOp) : Except String KernelPlan :=
  planOps policy (availability.filterCapsules registry) ops

/-! ## Typed verified planning -/

/--
A proof-bearing kernel selected without erasing its implementation or refinement theorem.

Unlike `PlannedKernel`, this type is indexed by the exact Lean specification implemented by the
kernel. The `selectable` field records device, provider, VJP, contract-shape, and proof-oriented
policy checks. Execution can therefore recover the refinement theorem directly.
-/
structure VerifiedPlannedKernel (ι : Type u) (ο : Type v) (op : BackendOp)
    (specification : ι → ο) (policy : KernelPolicy) where
  kernel : ProofCarryingKernel ι ο op specification
  selectable : kernel.selectable policy = true

namespace VerifiedPlannedKernel

/-- Execute the selected typed implementation. -/
def run {ι : Type u} {ο : Type v} {op : BackendOp} {specification : ι → ο}
    {policy : KernelPolicy} (planned : VerifiedPlannedKernel ι ο op specification policy)
    (input : ι) : ο :=
  planned.kernel.run input

/-- Verified planning preserves the kernel's exact Lean specification. -/
theorem run_eq_specification {ι : Type u} {ο : Type v} {op : BackendOp}
    {specification : ι → ο} {policy : KernelPolicy}
    (planned : VerifiedPlannedKernel ι ο op specification policy) (input : ι) :
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
    {specification : ι → ο} (policy : KernelPolicy)
    (kernels : Array (ProofCarryingKernel ι ο op specification)) :
    Except String (VerifiedPlannedKernel ι ο op specification policy) := do
  let mut selected : Option (VerifiedPlannedKernel ι ο op specification policy) := none
  for kernel in kernels do
    if selected.isNone then
      if h : kernel.selectable policy = true then
        selected := some { kernel, selectable := h }
  match selected with
  | some kernel => pure kernel
  | none =>
      throw <| s!"no selectable proof-carrying kernel for op {op.name} on device " ++
        s!"{policy.device.cliName}"

end Backend
end NN
