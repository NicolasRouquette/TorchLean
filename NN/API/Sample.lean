/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor
public import NN.Tensor.Pack

/-!
# Supervised Samples

Typed input-target samples built from the canonical tensor-pack representation.

Main declarations:
- `TorchLean.Sample.*`: supervised `(x, y)` samples and minibatch wrappers.

Input and target dimensions are ordinary `List Nat` values and remain visible in every operation.
The implementation uses `TensorPack`, the heterogeneous tensor collection defined in
`NN.Tensor.Pack`.
-/

@[expose] public section

namespace TorchLean.Sample

/-- A supervised sample containing an input tensor and its target tensor. -/
abbrev Supervised (α : Type) (inputShape targetShape : List Nat) :=
  TorchLean.TensorPack α [inputShape, targetShape]

/-- A fixed-size minibatch whose input and target tensors have leading dimension `n`. -/
abbrev Batch (α : Type) (n : Nat) (inputShape targetShape : List Nat) :=
  Supervised α (n :: inputShape) (n :: targetShape)

/-- Build a supervised sample `(x, y)` as a two-tensor pack. -/
def mk {α : Type} {inputShape targetShape : List Nat}
    (x : Tensor α inputShape) (y : Tensor α targetShape) :
    Supervised α inputShape targetShape :=
  .cons x (.cons y .nil)

/-- Build a minibatch from input and target tensors sharing leading dimension `n`. -/
def batch {α : Type} {n : Nat} {inputShape targetShape : List Nat}
    (x : Tensor α (n :: inputShape)) (y : Tensor α (n :: targetShape)) :
    Batch α n inputShape targetShape :=
  mk x y

/-- Extract the input tensor `x` from a supervised sample. -/
def x {α : Type} {inputShape targetShape : List Nat}
    (s : Supervised α inputShape targetShape) : Tensor α inputShape :=
  TorchLean.TensorPack.get s ⟨0, by simp⟩

/-- Extract the target tensor `y` from a supervised sample. -/
def y {α : Type} {inputShape targetShape : List Nat}
    (s : Supervised α inputShape targetShape) : Tensor α targetShape :=
  TorchLean.TensorPack.get s ⟨1, by simp⟩

/-- Unpack a supervised sample as the ordinary pair `(x, y)`. -/
def toPair {α : Type} {inputShape targetShape : List Nat}
    (s : Supervised α inputShape targetShape) :
    Tensor α inputShape × Tensor α targetShape :=
  (x s, y s)

/-- `x` of a constructed supervised sample `mk x y` is `x`. -/
@[simp] theorem x_mk {α : Type} {inputShape targetShape : List Nat}
    (xT : Tensor α inputShape) (yT : Tensor α targetShape) :
    x (mk (α := α) (inputShape := inputShape) (targetShape := targetShape) xT yT) = xT := by
  rfl

/-- `y` of a constructed supervised sample `mk x y` is `y`. -/
@[simp] theorem y_mk {α : Type} {inputShape targetShape : List Nat}
    (xT : Tensor α inputShape) (yT : Tensor α targetShape) :
    y (mk (α := α) (inputShape := inputShape) (targetShape := targetShape) xT yT) = yT := by
  rfl

/-- Converting `mk x y` to a pair returns `(x, y)`. -/
@[simp] theorem toPair_mk {α : Type} {inputShape targetShape : List Nat}
    (xT : Tensor α inputShape) (yT : Tensor α targetShape) :
    toPair (mk (α := α) (inputShape := inputShape) (targetShape := targetShape) xT yT) =
      (xT, yT) := by
  rfl

/-- Map a function over the input tensor `x`, leaving the target `y` unchanged. -/
def mapX {α : Type} {inputShape targetShape : List Nat}
    (f : Tensor α inputShape → Tensor α inputShape)
    (s : Supervised α inputShape targetShape) :
    Supervised α inputShape targetShape :=
  mk (f (x s)) (y s)

/-- Map a function over the target tensor `y`, leaving the input `x` unchanged. -/
def mapY {α : Type} {inputShape targetShape : List Nat}
    (f : Tensor α targetShape → Tensor α targetShape)
    (s : Supervised α inputShape targetShape) :
    Supervised α inputShape targetShape :=
  mk (x s) (f (y s))

/-- Map functions over both `x` and `y` in a supervised sample. -/
def mapXY {α : Type} {inputShape targetShape : List Nat}
    (fx : Tensor α inputShape → Tensor α inputShape)
    (fy : Tensor α targetShape → Tensor α targetShape)
    (s : Supervised α inputShape targetShape) :
    Supervised α inputShape targetShape :=
  mk (fx (x s)) (fy (y s))

end TorchLean.Sample
