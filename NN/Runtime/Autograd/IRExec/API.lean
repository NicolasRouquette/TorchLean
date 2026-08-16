/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Lowering

/-!
# IR To Executable Graph Lowering

Public entrypoint for validating and lowering a complete shared IR graph.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace IRExec

open Spec
open Tensor
open Proofs.Autograd.Algebra
open NN.IR

/--
Lower an op-tagged IR graph into an executable `ForwardGraph`.

Requirements:
- Node id 0 must be `.input`.
- The graph must satisfy `Graph.checkWellFormed`.
- The external payload must contain entries for every `.const`/`.linear`/`.conv2d` node id.

This returns a `ForwardGraph` whose `eval` computes all node values in topological order. The
artifact is intentionally forward-only; it is distinct from the differentiable `Torch.TypedGraph`
used by the autograd lowering path.

This is the main API consumed by runtime callers that want executable evaluation while remaining
aligned with the shared `NN.IR.Graph` semantics.
-/
def lowerToForwardGraph
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) : Except String (ForwardGraph α) := do
  g.checkWellFormed
  let n0 ← g.getNode 0
  match n0.kind with
  | .input =>
      let inShape := n0.outShape
      let stFinal ← Internal.buildFrom (α := α) (g := g) (payload := payload)
        (inShape := inShape) (i := 1) (st := (⟨[], .nil⟩ : Internal.State α inShape))
      let ⟨ss, gd⟩ := stFinal
      pure { inShape := inShape, ss := ss, body := gd }
  | _ =>
      throw s!"IRExec: node 0 is not `.input` (got {n0.kind.tag})"
