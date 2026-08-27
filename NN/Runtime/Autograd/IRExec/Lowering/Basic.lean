/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Lowering.Common

/-!
# Basic and Random IR Lowering

Checked lowering for graph inputs, constants, detachment, and random operations.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace IRExec

open Spec
open Tensor
open Proofs.Autograd.Algebra
open NN.IR

namespace Internal

/-- Checked lowering for graph inputs, constants, detachment, and random operations. -/
@[simp] def lowerBasic {α : Type} [Context α] [shapeDecidable : DecidableEq Shape]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (kind : OpKind) :
    NodeLoweringResult ctx := do
  let g := ctx.graph
  let payload := ctx.payload
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : _root_.TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match kind with
  | .input =>
      throw s!"IRExec: internal error (handled above)"
  | .const s =>
      let t ← NN.IR.Graph.evalConst (α := α) (payload := payload) (id := n.id) (s := s)
      if hOut : s = τ then
        -- Cast so the node is typed at the declared outShape.
        pure <| fwd (fun _ctx => hOut ▸ t)
      else
        throw s!"IRExec: const node {i}: outShape mismatch: kind={repr s}, declared={repr τ}"
  | .detach =>
      match unaryParent? n.parents with
      | some pId =>
          let pNode ← g.getNode pId
          let s := pNode.outShape
          let ip ← parentIdx pId s
          if hOut : s = τ then
            let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
              hOut ▸ (getIdx (α := α) (xs := ctx) ip)
            pure <| fwd forward
          else
            throw s!"IRExec: node {i}: detach expects outShape=parent.outShape ({n.summary})"
      | _ => throw s!"IRExec: node {i}: detach expects 1 parent ({n.summary})"
  | .randUniform seed =>
      match n.parents.isEmpty with
      | true =>
          let key := Runtime.Autograd.TorchLean.Random.keyOf seed i
          let t : Tensor α τ := Runtime.Autograd.TorchLean.Random.uniform (α := α) key (s := τ)
          pure <| fwd (fun _ctx => t)
      | _ => throw s!"IRExec: node {i}: rand_uniform expects 0 parents ({n.summary})"
  | .bernoulliMask seed =>
      match unaryParent? n.parents with
      | some pId =>
          let ip ← parentIdx pId Shape.scalar
          let key := Runtime.Autograd.TorchLean.Random.keyOf seed i
          let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
            let kpT := getIdx (α := α) (xs := ctx) ip
            let kp : α :=
              match kpT with
              | Tensor.scalar v => v
            Runtime.Autograd.TorchLean.Random.mask (α := α) key kp (s := τ)
          pure <| fwd forward
      | _ => throw s!"IRExec: node {i}: bernoulli_mask expects 1 parent ({n.summary})"
  | _ => throw s!"IRExec: internal error: operation routed to lowerBasic"


end Internal
end IRExec
end Autograd
end Runtime
