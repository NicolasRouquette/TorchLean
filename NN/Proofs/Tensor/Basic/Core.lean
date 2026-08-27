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
- lemmas relate `getScalar` views to `add_spec`, `scale_spec`, etc.

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
export Proofs.TensorAlgebra
  (add_finRange_foldl_add_zero foldl_tensorScalar_mulAdd foldl_matvec_scalar get2_eq get_eq)

/-! ## Algebraic instances -/

/-- Tensor scalar multiplication is pointwise at every rank. -/
@[instance_reducible] noncomputable def Tensor.moduleReal
    {α : Type} [AddCommMonoid α] [Module ℝ α] :
    (s : Shape) → Module ℝ (Tensor α s)
  | .scalar => Equiv.module ℝ (Tensor.scalarEquiv α)
  | .dim n s =>
      letI : Module ℝ (Tensor α s) := Tensor.moduleReal s
      Equiv.module ℝ (Tensor.dimEquiv n s)

/-- Every tensor shape inherits the pointwise real-module structure of its scalar type. -/
noncomputable instance {α : Type} [AddCommMonoid α] [Module ℝ α] {s : Shape} :
    Module ℝ (Tensor α s) :=
  Tensor.moduleReal s

/-! ## 1D helpers -/

/-- Mapping a scalar tensor and then extracting it is the same as mapping its scalar value. -/
@[simp] lemma toScalar_mapTensor {α β : Type} (f : α -> β) (x : Tensor α .scalar) :
    Tensor.item (Spec.Tensor.map f x) = f (Tensor.item x) := by
  cases x
  rfl

/-- Coordinate extraction commutes with a tensor map on vectors. -/
@[simp] lemma getScalar_mapTensor {α β : Type} {n : Nat}
    (f : α -> β) (x : Tensor α [n]) (i : Fin n) :
    Spec.Tensor.getScalar (Spec.Tensor.map f x) i =
      f (Spec.Tensor.getScalar x i) := by
  exact Spec.Tensor.getScalar_map f x i

/-- `getScalar` distributes over pointwise addition (`add_spec`). -/
lemma getScalar_add_spec {n : Nat} (x y : Tensor ℝ [n]) :
    getScalar (addSpec x y) = fun i => getScalar x i + getScalar y i := by
  cases x with
  | dim vx =>
    cases y with
    | dim vy =>
      funext i
      cases hx : vx i
      cases hy : vy i
      simp [getScalar, addSpec, map2Spec, hx, hy]

/-- `getScalar` distributes over pointwise scaling (`scale_spec`). -/
lemma getScalar_scale_spec {n : Nat} (x : Tensor ℝ [n]) (c : ℝ) :
    getScalar (scaleSpec x c) = fun i => getScalar x i * c := by
  cases x with
  | dim vx =>
    funext i
    cases hx : vx i
    simp [getScalar, scaleSpec, mapSpec, hx]

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
  getScalar (flattenSpec (α:=ℝ) x)

/--
Unflatten a 1D view `Fin (Spec.Shape.size s) → ℝ` back into a tensor of shape `s`.

This is the proof layer counterpart of `Spec.Tensor.unflatten_spec` specialized to `ℝ`, and is
intended to round-trip with `flattenR` under the spec lemmas in
`NN/Spec/Core/TensorReductionShape.lean`.
-/
def unflattenR {s : Shape} (v : Fin (Spec.Shape.size s) → ℝ) : Tensor ℝ s :=
  unflattenSpec (α:=ℝ) s (ofFn v)

/-! ## Pointwise tensor algebra -/


end Spec
