/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.Runtime.Autograd.Engine.Core.Elementwise

/-!
Linear-algebra operations for the eager engine.

The definitions here cover matrix products, batched products, affine layers, and the corresponding
runtime graph nodes shared by CPU and CUDA-backed execution.
-/

@[expose] public section

namespace Runtime
namespace Autograd

open Spec
open Tensor

namespace Tape

/--
Fully-connected linear layer `y = W x + b` (matvec).

Type-level shapes enforce `W : (outDim, inDim)`, `x : (inDim,)`, `b : (outDim,)`.
PyTorch comparison: `torch.nn.functional.linear`.
-/
def linear {α : Type} [Add α] [Mul α] [Zero α] [DecidableEq Shape]
  {inDim outDim : Nat}
  (t : Tape α) (wId bId xId : Nat) : Result (Tape α × Nat) := do
  let W ← requireValue (α:=α) (t:=t) (s:=.dim outDim (.dim inDim .scalar)) wId
  let b ← requireValue (α:=α) (t:=t) (s:=.dim outDim .scalar) bId
  let x ← requireValue (α:=α) (t:=t) (s:=.dim inDim .scalar) xId
  let layer : Spec.LinearSpec α inDim outDim := { weights := W, bias := b }
  let y := Spec.linearSpec (α:=α) layer x
  let node : Node α :=
    { name := some "linear"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [wId, bId, xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := .dim outDim .scalar) dLdyAny
        let dW := Spec.linearWeightsDerivSpec (α:=α) x dLdy
        let db := Spec.linearBiasDerivSpec (α:=α) (dW) dLdy x
        let dx := Spec.linearInputDerivSpec (α:=α) W dLdy
        pure [
          (wId, AnyTensor.mk dW),
          (bId, AnyTensor.mk db),
          (xId, AnyTensor.mk dx)
        ]
    }
  pure (t.addNode node)

/--
2D matrix multiplication.

PyTorch comparison: `torch.mm(a, b)`.
-/
def matmul {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {m n p : Nat} (t : Tape α) (aId bId : Nat) : Result (Tape α × Nat) := do
  let a ← requireValue (α:=α) (t:=t) (s:=.dim m (.dim n .scalar)) aId
  let b ← requireValue (α:=α) (t:=t) (s:=.dim n (.dim p .scalar)) bId
  let y := Spec.matMulSpec a b
  let node : Node α :=
    { name := some "matmul"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := .dim m (.dim p .scalar)) dLdyAny
        let (dA, dB) := Spec.Tensor.matMulBackwardSpec a b dLdy
        pure [(aId, AnyTensor.mk dA), (bId, AnyTensor.mk dB)]
    }
  pure (t.addNode node)

/--
Batched matrix multiplication.

PyTorch comparison: `torch.bmm(a, b)`.
-/
def bmm {α : Type} [Add α] [Mul α] [Zero α] [DecidableEq Shape]
  {batch m n p : Nat} (t : Tape α) (aId bId : Nat) : Result (Tape α × Nat) := do
  let a ← requireValue (α:=α) (t:=t) (s:=.dim batch (.dim m (.dim n .scalar))) aId
  let b ← requireValue (α:=α) (t:=t) (s:=.dim batch (.dim n (.dim p .scalar))) bId
  let y := Spec.Tensor.bmmSpec (α := α) (batch := batch) (m := m) (n := n) (p := p) a b
  let node : Node α :=
    { name := some "bmm"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := .dim batch (.dim m (.dim p .scalar))) dLdyAny
        let (dA, dB) := Spec.Tensor.bmmBackwardSpec (α := α) (batch := batch) (m := m) (n := n)
          (p := p) a b dLdy
        pure [(aId, AnyTensor.mk dA), (bId, AnyTensor.mk dB)]
    }
  pure (t.addNode node)

/--
Concatenate two 1D vectors along dimension 0.

PyTorch comparison: `torch.cat([a, b], dim=0)` for vectors.
-/
def concatVectors {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq
  Shape]
  {n m : Nat} (t : Tape α) (aId bId : Nat) : Result (Tape α × Nat) := do
  let a ← requireValue (α:=α) (t:=t) (s:=.dim n .scalar) aId
  let b ← requireValue (α:=α) (t:=t) (s:=.dim m .scalar) bId
  let y := Spec.Tensor.concatVectorsSpec a b
  let node : Node α :=
    { name := some "concat_vectors"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := .dim (n + m) .scalar) dLdyAny
        let dA := Spec.Tensor.sliceVectorSpec dLdy 0 n (by simp)
        let dB := Spec.Tensor.sliceVectorSpec dLdy n m (by exact Nat.le_refl _)
        pure [(aId, AnyTensor.mk dA), (bId, AnyTensor.mk dB)]
    }
  pure (t.addNode node)

/--
Concatenate two tensors along dimension 0.

PyTorch comparison: `torch.cat([a, b], dim=0)`.
-/
def concatLeadingAxis {α : Type} [DecidableEq Shape]
  {n m : Nat} {s : Shape} (t : Tape α) (aId bId : Nat) : Result (Tape α × Nat) := do
  let a ← requireValue (α := α) (t := t) (s := .dim n s) aId
  let b ← requireValue (α := α) (t := t) (s := .dim m s) bId
  let y := Spec.Tensor.concatLeadingAxisSpec (α := α) (n := n) (m := m) (s := s) a b
  let node : Node α :=
    { name := some "concat_leading_axis"
      value := AnyTensor.mk y
      requires_grad := true
      parents := [aId, bId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad (α := α) (τ := .dim (n + m) s) dLdyAny
        let dA := Spec.sliceRangeSpec (α := α) (n := n + m) (s := s) dLdy 0 n
          (by simp)
        let dB := Spec.sliceRangeSpec (α := α) (n := n + m) (s := s) dLdy n m
          (by simp [Nat.add_comm])
        pure [(aId, AnyTensor.mk dA), (bId, AnyTensor.mk dB)]
    }
  pure (t.addNode node)

/--
Slice along dimension 0: `x[start : start+len]`.

The proof argument `h` enforces bounds.
PyTorch comparison: `x[start:start+len]` on tensors with a leading dimension.
-/
def sliceLeadingAxisRange {α : Type} [Zero α] [DecidableEq Shape]
  {n : Nat} {s : Shape} (t : Tape α) (xId : Nat) (start len : Nat) (h : len + start ≤ n) :
  Result (Tape α × Nat) :=
  unary (α := α) (t := t) (σ := .dim n s) (τ := .dim len s)
    "slice_leading_axis_range" xId
    (forward := fun x => Spec.sliceRangeSpec (α := α) (n := n) (s := s) x start len h)
    (backward := fun _x dLdz =>
      Spec.Tensor.sliceLeadingAxisRangeBackwardSpec (α := α) (n := n) (s := s) start len h dLdz)
