/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Lowering.Common

/-!
# Elementwise and Activation IR Lowering

Checked lowering for pointwise arithmetic, unary functions, and activation operations.
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

/-- Checked lowering for pointwise arithmetic, unary functions, and activation operations. -/
@[simp] def lowerElementwise {α : Type} [Context α]
    {Γ : List Shape} (ctx : NodeLoweringContext α Γ) (kind : OpKind) :
    NodeLoweringResult ctx := do
  let i := ctx.index
  let n := ctx.node
  let τ : Shape := n.outShape
  let parentIdx := ctx.parentIdx
  let fwd (forward : _root_.TorchLean.TensorPack α Γ → Tensor α τ) :
      ForwardNode α Γ τ :=
    mkForwardNode (α := α) (Γ := Γ) (τ := τ) forward
  match kind with
  | .add =>
      match binaryParents? n.parents with
      | some (aId, bId) =>
          let ia ← parentIdx aId τ
          let ib ← parentIdx bId τ
          let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
            Tensor.addSpec (α := α) (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs :=
              ctx) ib)
          pure <| fwd forward
      | _ => throw s!"IRExec: node {i}: add expects 2 parents ({n.summary})"
  | .sub =>
      match binaryParents? n.parents with
      | some (aId, bId) =>
          let ia ← parentIdx aId τ
          let ib ← parentIdx bId τ
          let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
            Tensor.subSpec (α := α) (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs :=
              ctx) ib)
          pure <| fwd forward
      | _ => throw s!"IRExec: node {i}: sub expects 2 parents ({n.summary})"
  | .mul_elem =>
      match binaryParents? n.parents with
      | some (aId, bId) =>
          let ia ← parentIdx aId τ
          let ib ← parentIdx bId τ
          let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
            Tensor.mulSpec (α := α) (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs :=
              ctx) ib)
          pure <| fwd forward
      | _ => throw s!"IRExec: node {i}: mul_elem expects 2 parents ({n.summary})"
  | .abs =>
      match unaryParent? n.parents with
      | some pId =>
          let ip ← parentIdx pId τ
          let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
            Tensor.absSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
          pure <| fwd forward
      | _ => throw s!"IRExec: node {i}: abs expects 1 parent ({n.summary})"
  | .sqrt =>
      match unaryParent? n.parents with
      | some pId =>
          let ip ← parentIdx pId τ
          let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
            Tensor.sqrtSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
          pure <| fwd forward
      | _ => throw s!"IRExec: node {i}: sqrt expects 1 parent ({n.summary})"
  | .inv =>
      match unaryParent? n.parents with
      | some pId =>
          let ip ← parentIdx pId τ
          let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
            Tensor.invSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
          pure <| fwd forward
      | _ => throw s!"IRExec: node {i}: inv expects 1 parent ({n.summary})"
  | .maxElem =>
      match binaryParents? n.parents with
      | some (aId, bId) =>
          let ia ← parentIdx aId τ
          let ib ← parentIdx bId τ
          let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
            Tensor.maxSpec (α := α) (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs :=
              ctx) ib)
          pure <| fwd forward
      | _ => throw s!"IRExec: node {i}: max_elem expects 2 parents ({n.summary})"
  | .minElem =>
      match binaryParents? n.parents with
      | some (aId, bId) =>
          let ia ← parentIdx aId τ
          let ib ← parentIdx bId τ
          let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
            Tensor.minSpec (α := α) (getIdx (α := α) (xs := ctx) ia) (getIdx (α := α) (xs :=
              ctx) ib)
          pure <| fwd forward
      | _ => throw s!"IRExec: node {i}: min_elem expects 2 parents ({n.summary})"
  | .relu =>
      match unaryParent? n.parents with
      | some pId =>
          let ip ← parentIdx pId τ
          let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
            Activation.reluSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
          pure <| fwd forward
      | _ => throw s!"IRExec: node {i}: relu expects 1 parent ({n.summary})"
  | .tanh =>
      match unaryParent? n.parents with
      | some pId =>
          let ip ← parentIdx pId τ
          let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
            Activation.tanhSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
          pure <| fwd forward
      | _ => throw s!"IRExec: node {i}: tanh expects 1 parent ({n.summary})"
  | .sigmoid =>
      match unaryParent? n.parents with
      | some pId =>
          let ip ← parentIdx pId τ
          let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
            Activation.sigmoidSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
          pure <| fwd forward
      | _ => throw s!"IRExec: node {i}: sigmoid expects 1 parent ({n.summary})"
  | .exp =>
      match unaryParent? n.parents with
      | some pId =>
          let ip ← parentIdx pId τ
          let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
            Tensor.expSpec (α := α) (getIdx (α := α) (xs := ctx) ip)
          pure <| fwd forward
      | _ => throw s!"IRExec: node {i}: exp expects 1 parent ({n.summary})"
  | .log =>
      match unaryParent? n.parents with
      | some pId =>
          let ip ← parentIdx pId τ
          let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
            let x := getIdx (α := α) (xs := ctx) ip
            -- This forward-graph closure is pure, so it cannot return the eager engine's
            -- `Except` error. A bad raw-log domain reaches a runtime panic; use `safe_log` for
            -- total epsilon protection.
            if Tensor.allSpec (α := α) (s := τ) (fun v => decide (v > (0 : α))) x then
              Tensor.logSpec (α := α) x
            else
              panic! "IRExec: log: input contains values <= 0 (or NaN); use `safe_log` if you want epsilon protection"
          pure <| fwd forward
      | _ => throw s!"IRExec: node {i}: log expects 1 parent ({n.summary})"
  | .sin =>
      match unaryParent? n.parents with
      | some pId =>
          let ip ← parentIdx pId τ
          let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
            Tensor.mapSpec (α := α) (s := τ) (fun x => MathFunctions.sin x) (getIdx (α := α)
              (xs := ctx) ip)
          pure <| fwd forward
      | _ => throw s!"IRExec: node {i}: sin expects 1 parent ({n.summary})"
  | .cos =>
      match unaryParent? n.parents with
      | some pId =>
          let ip ← parentIdx pId τ
          let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
            Tensor.mapSpec (α := α) (s := τ) (fun x => MathFunctions.cos x) (getIdx (α := α)
              (xs := ctx) ip)
          pure <| fwd forward
      | _ => throw s!"IRExec: node {i}: cos expects 1 parent ({n.summary})"
  | .softmax axis =>
      match unaryParent? n.parents with
      | some pId => do
          match Spec.Shape.axisInBounds? axis τ with
          | none =>
              throw s!"softmax: invalid axis {axis} for rank {Spec.Shape.rank τ}"
          | some h =>
              parentIdx pId τ >>= fun ip =>
                let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
                  @Activation.softmaxSpec α _ τ axis h.down
                    (getIdx (α := α) (xs := ctx) ip)
                pure <| fwd forward
      | _ => throw s!"IRExec: node {i}: softmax expects 1 parent ({n.summary})"
  | .hardMaskedSoftmax mask =>
      match unaryParent? n.parents with
      | some pId => do
          let ip ← parentIdx pId τ
          let allowed ←
            match NN.IR.HardMask.toTensorAs? mask τ with
            | .ok value => pure value
            | .error msg =>
                throw s!"IRExec: node {i}: hard_masked_softmax: {msg} ({n.summary})"
          let forward := fun ctx : _root_.TorchLean.TensorPack α Γ =>
            Spec.hardMaskedSoftmaxSpec
              (getIdx (α := α) (xs := ctx) ip) allowed
          pure <| fwd forward
      | _ =>
          throw s!"IRExec: node {i}: hard_masked_softmax expects 1 parent ({n.summary})"
  | _ => throw s!"IRExec: internal error: operation routed to lowerElementwise"


end Internal
end IRExec
end Autograd
end Runtime
