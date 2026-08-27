/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Lowering.Common

/-!
# Reduction IR Lowering

Checked lowering for broadcasts, axis reductions, full reductions, and scalar losses.
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

/-- Checked lowering for broadcasts, axis reductions, full reductions, and scalar losses. -/
@[simp] def lowerReduction {α : Type} [Context α] [shapeDecidable : DecidableEq Shape]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (kind : OpKind) :
    NodeLoweringResult ctx := do
  let g := ctx.graph
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : _root_.TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match kind with
  | .broadcastTo s₁ s₂ =>
      match unaryParent? n.parents with
      | some pId =>
          let ip ← parentIdx pId s₁
          match NN.IR.OpContracts.mkCanBroadcastTo? s₁ s₂ with
          | none =>
              throw s!"IRExec: node {i}: broadcastTo invalid: {repr s₁} → {repr s₂}"
          | some cb =>
              if hOut : s₂ = τ then
                let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
                  let x := getIdx (α := α) (xs := ctx) ip
                  hOut ▸ Tensor.broadcastTo (α := α) (s₁ := s₁) (s₂ := s₂) cb x
                pure <| fwd forward
              else
                throw <|
                  s!"IRExec: node {i}: broadcastTo outShape mismatch: kind={repr s₂}, " ++
                    s!"declared={repr τ}"
      | _ => throw s!"IRExec: node {i}: broadcastTo expects 1 parent ({n.summary})"
  | .reduceSum axis =>
      match unaryParent? n.parents with
      | some pId =>
          let pNode ← g.getNode pId
          let s := pNode.outShape
          let ip ← parentIdx pId s
          match Spec.Shape.nonemptyAxis? (axis := axis) s with
          | none =>
              throw s!"IRExec: node {i}: reduce_sum invalid axis={axis} for shape {repr s}"
          | some hAxis =>
              let hRed := hAxis.down
              let expected : Shape := Spec.Tensor.shapeAfterSum s axis
              if hOut : expected = τ then
                let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
                  let x := getIdx (α := α) (xs := ctx) ip
                  let y : Tensor α expected := Tensor.reduceSum (α := α) (s := s) axis x hRed
                  hOut ▸ y
                pure <| fwd forward
              else
                throw <|
                  s!"IRExec: node {i}: reduce_sum outShape mismatch: " ++
                    s!"expected={repr expected}, declared={repr τ} ({n.summary})"
      | _ => throw s!"IRExec: node {i}: reduce_sum expects 1 parent ({n.summary})"
  | .reduceMean axis =>
      match unaryParent? n.parents with
      | some pId =>
          let pNode ← g.getNode pId
          let s := pNode.outShape
          let ip ← parentIdx pId s
          match Spec.Shape.nonemptyAxis? (axis := axis) s with
          | none =>
              throw s!"IRExec: node {i}: reduce_mean invalid axis={axis} for shape {repr s}"
          | some hAxis =>
              let hRed := hAxis.down
              let expected : Shape := Spec.Tensor.shapeAfterSum s axis
              if hOut : expected = τ then
                let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
                  let x := getIdx (α := α) (xs := ctx) ip
                  let y : Tensor α expected := Tensor.reduceMean (α := α) (s := s) axis x hRed
                  hOut ▸ y
                pure <| fwd forward
              else
                throw <|
                  s!"IRExec: node {i}: reduce_mean outShape mismatch: " ++
                    s!"expected={repr expected}, declared={repr τ} ({n.summary})"
      | _ => throw s!"IRExec: node {i}: reduce_mean expects 1 parent ({n.summary})"
  | .sum =>
      match unaryParent? n.parents with
      | some pId =>
          let pNode ← g.getNode pId
          let s := pNode.outShape
          let ip ← parentIdx pId s
          if hOut : Shape.scalar = τ then
            let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
              let x := getIdx (α := α) (xs := ctx) ip
              hOut ▸ Tensor.scalar (Tensor.sumSpec (α := α) x)
            pure <| fwd forward
          else
            throw s!"IRExec: node {i}: sum expects scalar outShape ({n.summary})"
      | _ => throw s!"IRExec: node {i}: sum expects 1 parent ({n.summary})"
  | .mseLoss =>
      match binaryParents? n.parents with
      | some (yId, tId) =>
          let yNode ← g.getNode yId
          let tNode ← g.getNode tId
          if _hShape : yNode.outShape = tNode.outShape then
            if hOut : Shape.scalar = τ then
              let s := yNode.outShape
              let iy ← parentIdx yId s
              let it ← parentIdx tId s
              let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
                let yhat := getIdx (α := α) (xs := ctx) iy
                let target := getIdx (α := α) (xs := ctx) it
                let diff := Tensor.subSpec (α := α) yhat target
                let sq := Tensor.mulSpec (α := α) diff diff
                let total : α := Tensor.sumSpec (α := α) sq
                let y0 : Tensor α .scalar :=
                  Tensor.scalar (total / (↑(NN.IR.Graph.meanDenom s) : α))
                Tensor.castShape y0 hOut
              pure <| fwd forward
            else
              throw s!"IRExec: node {i}: mse_loss expects scalar outShape ({n.summary})"
          else
            throw <|
              s!"IRExec: node {i}: mse_loss expects equal shapes, got " ++
                s!"{repr yNode.outShape} vs {repr tNode.outShape}"
      | _ => throw s!"IRExec: node {i}: mse_loss expects 2 parents ({n.summary})"
  | _ => throw s!"IRExec: internal error: operation routed to lowerReduction"


end Internal
end IRExec
end Autograd
end Runtime
