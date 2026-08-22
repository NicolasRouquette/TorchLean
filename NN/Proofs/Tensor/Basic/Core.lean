/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.BigOperators.GroupWithZero.Action
public import Mathlib.Algebra.BigOperators.Ring.Finset
public import Mathlib.Algebra.BigOperators.Ring.List
public import Mathlib.Algebra.BigOperators.Ring.Multiset
public import Mathlib.Algebra.BigOperators.Ring.Nat
public import Mathlib.Algebra.Module.TransferInstance
public import Mathlib.Data.Fin.Basic
public import Mathlib.Data.List.FinRange
public import Mathlib.Data.Real.Basic
public import NN.Proofs.Tensor.Algebra
public import NN.Spec.Core.Context
public import NN.Spec.Core.Shape
public import NN.Spec.Core.Tensor
public import NN.Spec.Core.TensorOps
public import NN.Spec.Core.TensorReductionShape
public import NN.Spec.Core.Tensor.API

/-!
# Real Tensor Proof Toolkit

This file is the `ℝ`-specialized proof layer companion to the spec tensor layer.

The tensor proof folder has two layers:

- `NN.Proofs.Tensor.Algebra` is backend-generic and proves semiring facts about recursive tensor
  dot products and executable folds.
- this file works in `Spec` over `ℝ`, where calculus, norms, Frobenius products, and model-analysis
  lemmas live.

The statements use PyTorch-shaped names where that helps readers:

- `flattenR` / `unflattenR` give a `Fin (Spec.Shape.size s) → ℝ` view of `Tensor ℝ s`.
- lemmas relate `toVec` views to `add_spec`, `scale_spec`, etc.

We re-export tensor-specific helpers from `NN.Proofs.Tensor.Algebra` into the `Spec` namespace.
General list-fold lemmas retain their canonical `List` names.

## PyTorch correspondence / citations

- Flatten / reshape: `torch.flatten`, `torch.reshape`, and `Tensor.view`.
  https://pytorch.org/docs/stable/generated/torch.flatten.html
  https://pytorch.org/docs/stable/generated/torch.reshape.html
  https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
- “numel”: `tensor.numel()` corresponds to `Spec.Shape.size`.
  https://pytorch.org/docs/stable/generated/torch.Tensor.numel.html
-/

@[expose] public section


namespace Spec

open Tensor
open scoped BigOperators

-- Re-export generic helpers (defined once in `Proofs.TensorAlgebra`) into `Spec.*`.
export Proofs.TensorAlgebra (toVec ofVec toVec_ofVec ofVec_toVec)
export Proofs.TensorAlgebra
  (add_finRange_foldl_add_zero foldl_tensorScalar_mulAdd foldl_matvec_scalar)

/-! ## Algebraic instances for small tensor shapes -/

/-- Additive commutative monoid structure on scalar-shaped real tensors, transported by equivalence.
-/
instance : AddCommMonoid (Tensor ℝ .scalar) :=
  Equiv.addCommMonoid (Tensor.scalarEquiv ℝ)

/-- Additive commutative monoid structure on 1D real tensors (transported via an equiv). -/
instance {n : Nat} : AddCommMonoid (Tensor ℝ (.dim n .scalar)) :=
  Equiv.addCommMonoid (Tensor.dimScalarEquiv n)

/-- Scalar tensors inherit an `ℝ`-module structure when their entries do, transported by equivalence.
-/
instance {α : Type} [AddCommMonoid α] [Module ℝ α] : Module ℝ (Tensor α .scalar) :=
  Equiv.module ℝ (Tensor.scalarEquiv α)

/-- `Tensor α (dim n scalar)` inherits an `ℝ`-module structure when `α` is an `ℝ`-module (via an
  equiv). -/
instance {α : Type} [AddCommMonoid α] [Module ℝ α] {n : Nat} : Module ℝ (Tensor α (.dim n .scalar))
  :=
  Equiv.module ℝ (Tensor.dimScalarEquiv n)

/-- Noncomputable `ℝ`-module instance on scalar real tensors (for calculus proofs). -/
noncomputable instance : Module ℝ (Tensor ℝ .scalar) :=
  Equiv.module ℝ (Tensor.scalarEquiv ℝ)

/-- Noncomputable `ℝ`-module instance on 1D real tensors (for calculus proofs). -/
noncomputable instance {n : Nat} : Module ℝ (Tensor ℝ (.dim n .scalar)) :=
  Equiv.module ℝ (Tensor.dimScalarEquiv n)

/-! ## 1D helpers -/

/-- Mapping a scalar tensor and then extracting it is the same as mapping its scalar value. -/
@[simp] lemma toScalar_mapTensor {α β : Type} (f : α -> β) (x : Tensor α .scalar) :
    Tensor.item (Spec.mapTensor f x) = f (Tensor.item x) := by
  cases x
  rfl

/-- Coordinate extraction commutes with a tensor map on vectors. -/
@[simp] lemma toVec_mapTensor {α β : Type} {n : Nat}
    (f : α -> β) (x : Tensor α (.dim n .scalar)) (i : Fin n) :
    Proofs.TensorAlgebra.toVec (Spec.mapTensor f x) i =
      f (Proofs.TensorAlgebra.toVec x i) := by
  cases x with
  | dim values =>
      cases hvalue : values i
      simp [Proofs.TensorAlgebra.toVec, Spec.mapTensor, hvalue]

/-- `toVec` distributes over pointwise addition (`add_spec`). -/
lemma toVec_add_spec {n : Nat} (x y : Tensor ℝ (.dim n .scalar)) :
    toVec (addSpec x y) = fun i => toVec x i + toVec y i := by
  cases x with
  | dim vx =>
    cases y with
    | dim vy =>
      funext i
      cases hx : vx i
      cases hy : vy i
      simp [toVec, addSpec, map2Spec, hx, hy]

/-- `toVec` distributes over pointwise scaling (`scale_spec`). -/
lemma toVec_scale_spec {n : Nat} (x : Tensor ℝ (.dim n .scalar)) (c : ℝ) :
    toVec (scaleSpec x c) = fun i => toVec x i * c := by
  cases x with
  | dim vx =>
    funext i
    cases hx : vx i
    simp [toVec, scaleSpec, mapSpec, hx]

/--
Flatten a tensor of shape `s` into a 1D view `Fin (Spec.Shape.size s) → ℝ`.

This is the proof layer counterpart of `Spec.Tensor.flatten_spec` specialized to `ℝ`. In PyTorch
terms it is the functional analogue of flattening a tensor and then indexing it linearly
(`torch.flatten`, `tensor.view(-1)`). See the spec file `NN/Spec/Core/TensorReductionShape.lean`
for the definitional flatten/unflatten interface.

Citations:
https://pytorch.org/docs/stable/generated/torch.flatten.html
https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
-/
def flattenR {s : Shape} (x : Tensor ℝ s) : Fin (Spec.Shape.size s) → ℝ :=
  toVec (flattenSpec (α:=ℝ) x)

/--
Unflatten a 1D view `Fin (Spec.Shape.size s) → ℝ` back into a tensor of shape `s`.

This is the proof layer counterpart of `Spec.Tensor.unflatten_spec` specialized to `ℝ`, and is
intended to round-trip with `flattenR` under the spec lemmas in
`NN/Spec/Core/TensorReductionShape.lean`.
-/
def unflattenR {s : Shape} (v : Fin (Spec.Shape.size s) → ℝ) : Tensor ℝ s :=
  unflattenSpec (α:=ℝ) s (ofVec v)

/-! ## Pointwise tensor algebra -/


end Spec
