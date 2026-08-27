/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Audit
public import NN.Backend.IR

/-!
# Backend Kernel Groups

Graph-aware grouping data for kernel selection and audit.

`GraphKernelPlan` chooses a capsule for each runtime-relevant IR node. A
`GroupedKernelPlan` places adjacent choices with the same operation and capsule contract in one
scheduling and audit group. Grouping does not translate graph semantics, fuse operations, or claim
that a group executes as one kernel launch. A future fused capsule must carry its own multi-node
contract.
-/

@[expose] public section

namespace NN
namespace Backend

/-- Why adjacent planned nodes share one audit/scheduling group. This does not assert kernel fusion. -/
inductive GroupKind where
  | singleton
  | sameCapsuleBoundary
  deriving Repr

/-- One scheduling and audit group, with source IR nodes retained for diagnostics. -/
structure KernelGroup where
  nodeIds : Array Nat
  kinds : Array NN.IR.OpKind
  op : BackendOp
  capsule : KernelCapsule
  kind : GroupKind := .singleton
  deriving Repr

/-- A node-level graph plan grouped for scheduling and trust-boundary audit. -/
structure GroupedKernelPlan where
  groups : Array KernelGroup
  deriving Repr

namespace KernelGroup

/-- Project the operation and capsule selected for this group. -/
def toPlannedKernel (g : KernelGroup) : PlannedKernel :=
  { op := g.op, capsule := g.capsule }

/-- Whether this group crosses a trusted-external boundary. -/
def hasTrustedExternal (g : KernelGroup) : Bool :=
  g.capsule.isTrustedExternal

/-- Whether a planned node can share this scheduling and audit group. -/
def canAppend (g : KernelGroup) (k : IR.PlannedNodeKernel) : Bool :=
  g.op == k.op && g.capsule.sameIdentity k.capsule

/-- Append a node to an existing backend group, preserving source-node provenance. -/
def appendNode (g : KernelGroup) (k : IR.PlannedNodeKernel) : KernelGroup :=
  { g with
    nodeIds := g.nodeIds.push k.nodeId
    kinds := g.kinds.push k.kind
    kind := .sameCapsuleBoundary }

end KernelGroup

namespace GroupedKernelPlan

/-- Source IR node ids covered by the grouped plan, in graph order. -/
def nodeIds (p : GroupedKernelPlan) : Array Nat :=
  p.groups.flatMap (·.nodeIds)

/-- Selected capsule names, in group order. -/
def capsuleNames (p : GroupedKernelPlan) : Array String :=
  p.groups.map fun g => g.capsule.name

/-- Project groups to the capsule rows consumed by the trust-boundary gate. -/
def toKernelPlan (p : GroupedKernelPlan) : KernelPlan :=
  { kernels := p.groups.map KernelGroup.toPlannedKernel }

/-- Audit the selected backend boundaries of the grouped plan. -/
def audit (p : GroupedKernelPlan) : KernelPlanAudit :=
  p.toKernelPlan.audit

/-- Whether any group crosses a trusted-external boundary. -/
def hasTrustedExternal (p : GroupedKernelPlan) : Bool :=
  p.groups.any KernelGroup.hasTrustedExternal

end GroupedKernelPlan

namespace IR

namespace PlannedNodeKernel

/-- Place one graph-planned node in a singleton scheduling group. -/
def toSingletonGroup (k : PlannedNodeKernel) : KernelGroup :=
  { nodeIds := #[k.nodeId]
    kinds := #[k.kind]
    op := k.op
    capsule := k.capsule
    kind := .singleton }

end PlannedNodeKernel

namespace GraphKernelPlan

/-- Keep one scheduling group per runtime-relevant IR node. -/
def toSingletonGroups (p : GraphKernelPlan) : GroupedKernelPlan :=
  { groups := p.kernels.map PlannedNodeKernel.toSingletonGroup }

/-- Fold state for conservative same-boundary grouping. -/
structure GroupingState where
  groups : Array KernelGroup
  deriving Repr

namespace GroupingState

/-- Add one planned node, sharing the preceding group when the boundary is identical. -/
def pushKernel (s : GroupingState) (k : PlannedNodeKernel) : GroupingState :=
  match s.groups.back? with
  | none =>
      { groups := #[k.toSingletonGroup] }
  | some g =>
      if g.canAppend k then
        { groups := s.groups.pop.push (g.appendNode k) }
      else
        { groups := s.groups.push k.toSingletonGroup }

end GroupingState

/--
Conservative grouping: adjacent nodes with the same backend boundary share one audit/scheduling
group.

This does not claim that one kernel invocation implements the group. A future fused capsule must
declare its multi-node pattern and execution contract explicitly.
-/
def toCoalescedGroups (p : GraphKernelPlan) : GroupedKernelPlan :=
  let s := p.kernels.foldl GroupingState.pushKernel { groups := #[] }
  { groups := s.groups }

end GraphKernelPlan

end IR

end Backend
end NN
