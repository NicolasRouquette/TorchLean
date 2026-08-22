/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core
public import NN.Runtime.Autograd.Train.Dataset

/-!
# Tensor Packs

Shape-indexed tensor packs and supervised samples.

Main declarations:
- `TorchLean.TensorPack`: a typed tuple of tensors.
- `TorchLean.TensorPack.*`: indexed access, mapping, zipping, append, and split.
- `TorchLean.Sample.*`: supervised `(x, y)` samples and minibatch wrappers.

These operations preserve the shapes carried by the underlying tensor-pack representation.
-/

@[expose] public section

namespace TorchLean

/-- A heterogeneous tensor tuple whose list of shapes is tracked in its type. -/
abbrev TensorPack (α : Type) (shapes : List Spec.Shape) :=
  _root_.Runtime.Autograd.Torch.TList α shapes

namespace TensorPack

/-- Map each tensor entry (shape-preserving). -/
def map {α β : Type} (f : ∀ {s : Spec.Shape}, Spec.Tensor α s → Spec.Tensor β s) :
    {ss : List Spec.Shape} → TensorPack α ss → TensorPack β ss
  | [], .nil => .nil
  | _s :: ss, .cons x xs => .cons (f x) (map (f := f) (ss := ss) xs)

/-- Zip two tensor packs pointwise (shape-preserving). -/
def zipWith {α β γ : Type}
    (f : ∀ {s : Spec.Shape}, Spec.Tensor α s → Spec.Tensor β s → Spec.Tensor γ s) :
    {ss : List Spec.Shape} →
      TensorPack α ss → TensorPack β ss → TensorPack γ ss
  | [], .nil, .nil => .nil
  | _s :: ss, .cons x xs, .cons y ys =>
      .cons (f x y) (zipWith (f := f) (ss := ss) xs ys)

/-- Append two tensor packs. -/
def append {α : Type} :
    {ss₁ ss₂ : List Spec.Shape} →
      TensorPack α ss₁ → TensorPack α ss₂ → TensorPack α (ss₁ ++ ss₂)
  | [], _ss₂, .nil, ys => ys
  | _s :: ss₁, ss₂, .cons x xs, ys => .cons x (append (ss₁ := ss₁) (ss₂ := ss₂) xs ys)

/-- Split a tensor pack into its prefix and suffix. -/
def split {α : Type} :
    {ss₁ ss₂ : List Spec.Shape} →
      TensorPack α (ss₁ ++ ss₂) → TensorPack α ss₁ × TensorPack α ss₂
  | [], _ss₂, xs => (.nil, xs)
  | _s :: ss₁, ss₂, .cons x xs =>
      let (xs₁, xs₂) := split (α := α) (ss₁ := ss₁) (ss₂ := ss₂) xs
      (.cons x xs₁, xs₂)

/-- Return the tensor at position `i`; its result shape is obtained from the pack's shape list. -/
def get {α : Type} {ss : List Spec.Shape} (xs : TensorPack α ss) (i : Fin ss.length) :
    Spec.Tensor α (ss.get i) :=
  _root_.Proofs.Autograd.Algebra.TList.get xs i

end TensorPack
end TorchLean

namespace TorchLean.Sample

/-- A supervised sample `(x, y)` with input shape `σ` and target shape `τ`. -/
abbrev Supervised (α : Type) (σ τ : Spec.Shape) :=
  TorchLean.TensorPack α [σ, τ]

/-- A fixed-size minibatch of supervised samples. -/
abbrev Batch (α : Type) (n : Nat) (σ τ : Spec.Shape) :=
  Supervised α (.dim n σ) (.dim n τ)

/-- Build a supervised sample `(x, y)` as a two-tensor pack. -/
def mk {α : Type} {σ τ : Spec.Shape} (x : Spec.Tensor α σ) (y : Spec.Tensor α τ) :
    Supervised α σ τ :=
  .cons x (.cons y .nil)

/-- Build a batched supervised sample `(xBatch, yBatch)`. -/
def batch {α : Type} {n : Nat} {σ τ : Spec.Shape}
    (x : Spec.Tensor α (.dim n σ)) (y : Spec.Tensor α (.dim n τ)) :
    Batch α n σ τ :=
  mk x y

/-- Extract the input tensor `x` from a supervised sample. -/
def x {α : Type} {σ τ : Spec.Shape} (s : Supervised α σ τ) : Spec.Tensor α σ :=
  TorchLean.TensorPack.get s ⟨0, by simp⟩

/-- Extract the target tensor `y` from a supervised sample. -/
def y {α : Type} {σ τ : Spec.Shape} (s : Supervised α σ τ) : Spec.Tensor α τ :=
  TorchLean.TensorPack.get s ⟨1, by simp⟩

/-- Unpack a supervised sample as the ordinary pair `(x, y)`. -/
def toPair {α : Type} {σ τ : Spec.Shape} (s : Supervised α σ τ) :
    Spec.Tensor α σ × Spec.Tensor α τ :=
  (x s, y s)

/-- `x` of a constructed supervised sample `mk x y` is `x`. -/
@[simp] theorem x_mk {α : Type} {σ τ : Spec.Shape}
    (xT : Spec.Tensor α σ) (yT : Spec.Tensor α τ) :
    x (mk (α := α) (σ := σ) (τ := τ) xT yT) = xT := by
  rfl

/-- `y` of a constructed supervised sample `mk x y` is `y`. -/
@[simp] theorem y_mk {α : Type} {σ τ : Spec.Shape}
    (xT : Spec.Tensor α σ) (yT : Spec.Tensor α τ) :
    y (mk (α := α) (σ := σ) (τ := τ) xT yT) = yT := by
  rfl

@[simp] theorem toPair_mk {α : Type} {σ τ : Spec.Shape}
    (xT : Spec.Tensor α σ) (yT : Spec.Tensor α τ) :
    toPair (mk (α := α) (σ := σ) (τ := τ) xT yT) = (xT, yT) := by
  rfl

/-- Map a function over the input tensor `x`, leaving the target `y` unchanged. -/
def mapX {α : Type} {σ τ : Spec.Shape}
    (f : Spec.Tensor α σ → Spec.Tensor α σ) (s : Supervised α σ τ) :
    Supervised α σ τ :=
  mk (f (x s)) (y s)

/-- Map a function over the target tensor `y`, leaving the input `x` unchanged. -/
def mapY {α : Type} {σ τ : Spec.Shape}
    (f : Spec.Tensor α τ → Spec.Tensor α τ) (s : Supervised α σ τ) :
    Supervised α σ τ :=
  mk (x s) (f (y s))

/-- Map functions over both `x` and `y` in a supervised sample. -/
def mapXY {α : Type} {σ τ : Spec.Shape}
    (fx : Spec.Tensor α σ → Spec.Tensor α σ)
    (fy : Spec.Tensor α τ → Spec.Tensor α τ)
    (s : Supervised α σ τ) :
    Supervised α σ τ :=
  mk (fx (x s)) (fy (y s))

end TorchLean.Sample
