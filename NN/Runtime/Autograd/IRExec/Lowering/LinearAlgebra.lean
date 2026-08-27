/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Lowering.Common

/-!
# Linear Algebra IR Lowering

Checked lowering for matrix multiplication and payload-backed linear layers.
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

/-- Checked lowering for matrix multiplication and payload-backed linear layers. -/
@[simp] def lowerLinearAlgebra {α : Type} [Context α] [shapeDecidable : DecidableEq Shape]
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
  | .matmul =>
      match binaryParents? n.parents with
      | some (aId, bId) =>
          let aNode ← g.getNode aId
          let bNode ← g.getNode bId
          match aNode.outShape, bNode.outShape with
          | .dim m (.dim nDim Shape.scalar), .dim n' (.dim p Shape.scalar) =>
              if _hn : nDim = n' then
                let ia ← parentIdx aId (.dim m (.dim nDim .scalar))
                let ib ← parentIdx bId (.dim nDim (.dim p .scalar))
                let expected : Shape := .dim m (.dim p .scalar)
                if hOut : expected = τ then
                  let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
                    let aT := getIdx (α := α) (xs := ctx) ia
                    let bT := getIdx (α := α) (xs := ctx) ib
                    let y : Tensor α expected := Spec.matMulSpec (α := α) (m := m) (n :=
                      nDim) (p := p) aT bT
                    hOut ▸ y
                  pure <| fwd forward
                else
                  throw <|
                    s!"IRExec: node {i}: matmul outShape mismatch: " ++
                      s!"expected={repr expected}, declared={repr τ} ({n.summary})"
              else
                throw s!"IRExec: node {i}: matmul inner dims mismatch: {nDim} vs {n'}"
          | .dim batch (.dim m (.dim nDim Shape.scalar)), .dim batch' (.dim n' (.dim p
            Shape.scalar)) =>
              if _hb : batch = batch' then
                if _hn : nDim = n' then
                  let ia ← parentIdx aId (.dim batch (.dim m (.dim nDim .scalar)))
                  let ib ← parentIdx bId (.dim batch (.dim nDim (.dim p .scalar)))
                  let expected : Shape := .dim batch (.dim m (.dim p .scalar))
                  if hOut : expected = τ then
                    let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
                      let aT := getIdx (α := α) (xs := ctx) ia
                      let bT := getIdx (α := α) (xs := ctx) ib
                      let y : Tensor α expected :=
                        Tensor.Internal.bmmLikeSpec (α := α) (batch := batch) (m := m)
                          (n := nDim) (p := p) aT bT
                      hOut ▸ y
                    pure <| fwd forward
                  else
                    throw <|
                      s!"IRExec: node {i}: bmm outShape mismatch: " ++
                        s!"expected={repr expected}, declared={repr τ} ({n.summary})"
                else
                  throw s!"IRExec: node {i}: matmul inner dims mismatch: {nDim} vs {n'}"
              else
                throw s!"IRExec: node {i}: matmul batch dims mismatch: {batch} vs {batch'}"
          | _, _ =>
              throw <|
                s!"IRExec: node {i}: unsupported matmul shapes: {repr aNode.outShape} · " ++
                  s!"{repr bNode.outShape}"
      | _ => throw s!"IRExec: node {i}: matmul expects 2 parents ({n.summary})"
  | .linear =>
      match unaryParent? n.parents with
      | some xId =>
          match payload.linear? n.id with
          | none => throw s!"IRExec: missing linear payload for node {n.id}"
          | some p =>
              let expectedIn : Shape := .dim p.inDim .scalar
              let expectedOut : Shape := .dim p.outDim .scalar
              let ix ← parentIdx xId expectedIn
              if hOut : expectedOut = τ then
                let W := p.W
                let b := p.b
                let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
                  let x := getIdx (α := α) (xs := ctx) ix
                  let y : Tensor α expectedOut :=
                    Tensor.addSpec (α := α)
                      (Spec.matVecMulSpec (α := α) (m := p.outDim) (n := p.inDim) W x) b
                  hOut ▸ y
                pure <| fwd forward
              else
                throw <|
                  s!"IRExec: linear {n.id}: declared outShape mismatch: {repr τ} vs " ++
                    s!"expected {repr expectedOut}"
      | _ => throw s!"IRExec: node {i}: linear expects 1 parent ({n.summary})"
  | _ => throw s!"IRExec: internal error: operation routed to lowerLinearAlgebra"


end Internal
end IRExec
end Autograd
end Runtime
