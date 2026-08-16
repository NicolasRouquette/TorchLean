/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Data.Dataset

/-!
# Tensor Datasets

Conversion between leading-axis tensors, tensor packs, and runtime datasets.
-/

@[expose] public section

namespace TorchLean
namespace Data
namespace TensorDataset

/-!
## TensorDataset (leading-axis batching)

PyTorch's `TensorDataset` concept is: given one or more tensors that share the same `size(0)`,
build a dataset of samples by slicing each tensor along its leading batch axis.

In TorchLean we do the same thing, but with shapes tracked in the type:

- a batched tensor has shape `.dim n σ`,
- slicing at `i : Fin n` yields a sample of shape `σ`,
- and a batch of multiple tensors is represented as a `TensorPack`.
-/

/--
Slice a batched `TensorPack` along its leading batch axis.

If a sample is represented as a shape-indexed tuple `TensorPack β ss`, then a minibatch of size `n`
is `TensorPack β (ss.map (fun s => .dim n s))`. This function picks a batch index `i : Fin n` and returns
the corresponding single sample.
-/
def unbatch {β : Type} {n : Nat} :
    {ss : List Spec.Shape} →
      _root_.TorchLean.TensorPack β (ss.map (fun s => Spec.Shape.dim n s)) →
      Fin n →
      _root_.TorchLean.TensorPack β ss
  | [], .nil, _i => .nil
  | _s :: ss, .cons x xs, i =>
      .cons (Spec.getAtSpec x i) (unbatch (β := β) (ss := ss) xs i)

/-- Convert a shape-indexed `TensorPack` of `Float` tensors to the runtime scalar type `α`. -/
def castFloat {α : Type} [_root_.TorchLean.Runtime.FromFloat α] :
    {ss : List Spec.Shape} →
      _root_.TorchLean.TensorPack Float ss →
      _root_.TorchLean.TensorPack α ss
  | [], .nil => .nil
  | _s :: ss, .cons x xs =>
      .cons (Spec.mapTensor (_root_.TorchLean.Runtime.ofFloat (α := α)) x)
        (castFloat (ss := ss) xs)

/--
Build a dataset by slicing a *batched* `TensorPack` along the leading batch axis. This gives the
typed counterpart of a tensor dataset built from several aligned arrays.
-/
def ofBatched {β : Type} {n : Nat} {ss : List Spec.Shape}
    (xs : _root_.TorchLean.TensorPack β (ss.map (fun s => Spec.Shape.dim n s))) :
    Dataset (_root_.TorchLean.TensorPack β ss) :=
  fromList <| (List.finRange n).map fun i => unbatch (β := β) (n := n) (ss := ss) xs i

/--
Float-to-`α` variant of `ofBatched`, for data loaded from disk.
-/
def ofBatchedFloat {α : Type} [_root_.TorchLean.Runtime.FromFloat α]
    {n : Nat} {ss : List Spec.Shape}
    (xs : _root_.TorchLean.TensorPack Float (ss.map (fun s => Spec.Shape.dim n s))) :
    Dataset (_root_.TorchLean.TensorPack α ss) :=
  let samples : List (_root_.TorchLean.TensorPack α ss) :=
    (List.finRange n).map (fun i =>
      castFloat (α := α) (unbatch (β := Float) (n := n) (ss := ss) xs i))
  fromList samples

/--
Supervised dataset from two batched tensors `X : (n, σ)` and `Y : (n, τ)` by slicing the leading batch axis.

This is the common regression/supervised-learning case: the TorchLean analogue of
`TensorDataset(X, Y)` in PyTorch.
-/
def supervised {α : Type}
    {n : Nat} {σ τ : Spec.Shape}
    (X : Spec.Tensor α (.dim n σ))
    (Y : Spec.Tensor α (.dim n τ)) :
    Dataset (_root_.TorchLean.TensorPack α [σ, τ]) :=
  ofBatched (β := α) (n := n) (ss := [σ, τ])
    (tensorpack! X, Y)

/-- Float-to-`α` variant of `supervised`, for data loaded from disk. -/
def supervisedFloat {α : Type} [_root_.TorchLean.Runtime.FromFloat α]
    {n : Nat} {σ τ : Spec.Shape}
    (X : Spec.Tensor Float (.dim n σ))
    (Y : Spec.Tensor Float (.dim n τ)) :
    Dataset (_root_.TorchLean.TensorPack α [σ, τ]) :=
  ofBatchedFloat (α := α) (n := n) (ss := [σ, τ])
    (tensorpack! X, Y)

end TensorDataset
end Data
end TorchLean
