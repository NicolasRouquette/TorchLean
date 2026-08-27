/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Constructors
public import NN.Spec.Core.TensorOps

/-!
# Heterogeneous Tensor Packs

`TorchLean.TensorPack α shapes` stores one tensor for every shape in `shapes`. Unlike a tensor,
whose entries all have one scalar type and one rectangular shape, a tensor pack may contain tensors
of different ranks and extents. The shape of every entry is nevertheless known statically.

Tensor packs are the common representation for model parameters, gradients, supervised samples,
and typed graph contexts. This module owns the datatype and its representation-independent
operations so those layers do not depend on autograd proofs or a particular runtime.
-/

@[expose] public section

namespace TorchLean

open Spec

/-- A heterogeneous sequence containing one tensor for every shape in `shapes`. -/
inductive TensorPack (α : Type) : List Shape → Type where
  /-- The empty tensor pack. -/
  | nil : TensorPack α []
  /-- Add a tensor whose shape becomes the head of the pack's shape list. -/
  | cons {s : Shape} {ss : List Shape} :
      Spec.Tensor α s → TensorPack α ss → TensorPack α (s :: ss)

namespace TensorPack

variable {α β γ : Type}

/-- Return the tensor at position `i`; its shape is determined by the pack's shape list. -/
def get : {ss : List Shape} →
    TensorPack α ss → (i : Fin ss.length) → Spec.Tensor α (ss.get i)
  | [], .nil, i => nomatch i
  | _ :: _, .cons x _, ⟨0, _⟩ => x
  | _ :: _ss, .cons _ xs, ⟨Nat.succ i, hi⟩ =>
      get xs ⟨i, Nat.lt_of_succ_lt_succ hi⟩

@[simp] theorem get_cons_zero {s : Shape} {ss : List Shape}
    (x : Spec.Tensor α s) (xs : TensorPack α ss) (h : 0 < (s :: ss).length) :
    get (.cons x xs) ⟨0, h⟩ = x := by
  rfl

@[simp] theorem get_cons_succ {s : Shape} {ss : List Shape}
    (x : Spec.Tensor α s) (xs : TensorPack α ss)
    (i : Nat) (h : Nat.succ i < (s :: ss).length) :
    get (.cons x xs) ⟨Nat.succ i, h⟩ =
      get xs ⟨i, Nat.lt_of_succ_lt_succ h⟩ := by
  rfl

/-- Apply a shape-preserving function to every tensor in a pack. -/
def map (f : ∀ {shape : Shape}, Spec.Tensor α shape → Spec.Tensor β shape) :
    {ss : List Shape} → TensorPack α ss → TensorPack β ss
  | [], .nil => .nil
  | _ :: ss, .cons x xs => .cons (f x) (map (f := f) (ss := ss) xs)

/-- Combine two packs pointwise with a shape-preserving binary function. -/
def zipWith
    (f : ∀ {shape : Shape},
      Spec.Tensor α shape → Spec.Tensor β shape → Spec.Tensor γ shape) :
    {ss : List Shape} →
      TensorPack α ss → TensorPack β ss → TensorPack γ ss
  | [], .nil, .nil => .nil
  | _ :: ss, .cons x xs, .cons y ys =>
      .cons (f x y) (zipWith (f := f) (ss := ss) xs ys)

/-- Concatenate two tensor packs. -/
def append : {ss₁ ss₂ : List Shape} →
    TensorPack α ss₁ → TensorPack α ss₂ → TensorPack α (ss₁ ++ ss₂)
  | [], _, .nil, ys => ys
  | _ :: ss₁, ss₂, .cons x xs, ys =>
      .cons x (append (ss₁ := ss₁) (ss₂ := ss₂) xs ys)

/-- Split a tensor pack at a statically known shape-list boundary. -/
def split : {ss₁ ss₂ : List Shape} →
    TensorPack α (ss₁ ++ ss₂) → TensorPack α ss₁ × TensorPack α ss₂
  | [], _, xs => (.nil, xs)
  | _ :: ss₁, ss₂, .cons x xs =>
      let (xsLeft, xsRight) := split (ss₁ := ss₁) (ss₂ := ss₂) xs
      (.cons x xsLeft, xsRight)

/-- Splitting a concatenated pair of packs recovers both inputs. -/
@[simp] theorem split_append {ss₁ ss₂ : List Shape}
    (xs : TensorPack α ss₁) (ys : TensorPack α ss₂) :
    split (append xs ys) = (xs, ys) := by
  induction ss₁ with
  | nil => cases xs; rfl
  | cons _ ss₁ ih =>
      cases xs with
      | cons x xs =>
          simp only [append, split]
          rw [ih xs]

/-- Construct the all-zero tensor pack. -/
def zero [Zero α] : {ss : List Shape} → TensorPack α ss
  | [] => .nil
  | shape :: ss => .cons (Spec.fill (0 : α) shape) (zero (ss := ss))

/-- Add two tensor packs pointwise. -/
def add [Add α] : {ss : List Shape} →
    TensorPack α ss → TensorPack α ss → TensorPack α ss
  | [], .nil, .nil => .nil
  | _ :: ss, .cons x xs, .cons y ys =>
      .cons (Spec.Tensor.addSpec x y) (add (ss := ss) xs ys)

/-- Multiply every tensor entry by the same scalar. -/
def scale [Mul α] (c : α) : {ss : List Shape} →
    TensorPack α ss → TensorPack α ss
  | [], .nil => .nil
  | _ :: ss, .cons x xs =>
      .cons (Spec.Tensor.scaleSpec x c) (scale c (ss := ss) xs)

/-- Subtract two tensor packs pointwise. -/
def sub [Sub α] : {ss : List Shape} →
    TensorPack α ss → TensorPack α ss → TensorPack α ss
  | [], .nil, .nil => .nil
  | _ :: ss, .cons x xs, .cons y ys =>
      .cons (Spec.Tensor.subSpec x y) (sub (ss := ss) xs ys)

/-- Append one tensor to the end of a pack. -/
def snoc {τ : Shape} : {ss : List Shape} →
    TensorPack α ss → Spec.Tensor α τ → TensorPack α (ss ++ [τ])
  | [], .nil, x => .cons x .nil
  | _ :: ss, .cons x xs, last => .cons x (snoc (ss := ss) xs last)

/-- Separate a nonempty pack into its prefix and final tensor. -/
def unsnoc {τ : Shape} : {ss : List Shape} →
    TensorPack α (ss ++ [τ]) → TensorPack α ss × Spec.Tensor α τ
  | [], .cons x .nil => (.nil, x)
  | _ :: ss, .cons x xs =>
      let (init, last) := unsnoc (ss := ss) xs
      (.cons x init, last)

/-- Transport a tensor pack along an equality between its shape lists. -/
def cast {ss₁ ss₂ : List Shape} (h : ss₁ = ss₂)
    (xs : TensorPack α ss₁) : TensorPack α ss₂ :=
  Eq.mp (congrArg (TensorPack α) h) xs

@[simp] theorem cast_rfl {ss : List Shape} (xs : TensorPack α ss) :
    cast rfl xs = xs := by
  rfl

@[simp] theorem cast_cast {ss₁ ss₂ ss₃ : List Shape}
    (h₁ : ss₁ = ss₂) (h₂ : ss₂ = ss₃) (xs : TensorPack α ss₁) :
    cast h₂ (cast h₁ xs) = cast (h₁.trans h₂) xs := by
  cases h₁
  cases h₂
  rfl

@[simp] theorem cast_symm {ss₁ ss₂ : List Shape} (h : ss₁ = ss₂)
    (xs : TensorPack α ss₁) :
    cast h.symm (cast h xs) = xs := by
  cases h
  rfl

/-- `unsnoc` recovers the two arguments supplied to `snoc`. -/
@[simp] theorem unsnoc_snoc {ss : List Shape} {τ : Shape}
    (xs : TensorPack α ss) (x : Spec.Tensor α τ) :
    unsnoc (snoc xs x) = (xs, x) := by
  induction ss with
  | nil =>
      cases xs
      simp [snoc, unsnoc]
  | cons _ ss ih =>
      cases xs with
      | cons head tail => simp [snoc, unsnoc, ih]

/-- Re-appending the final tensor obtained by `unsnoc` reconstructs the original pack. -/
@[simp] theorem snoc_unsnoc {ss : List Shape} {τ : Shape}
    (xs : TensorPack α (ss ++ [τ])) :
    snoc (unsnoc xs).1 (unsnoc xs).2 = xs := by
  induction ss with
  | nil =>
      cases xs with
      | cons x xs => cases xs; simp [snoc, unsnoc]
  | cons _ ss ih =>
      cases xs with
      | cons head tail => simp [snoc, unsnoc, ih]

end TensorPack
end TorchLean
