/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Scalar
public import NN.API.Macros
public import NN.Tensor
public import NN.API.Sample
public import NN.Data.SampleStream

/-!
# Tensor-Backed Sample Streams

These constructors expose the samples along the leading axis of one or more aligned tensors. They
return lazy `SampleStream`s: selecting a sample performs the corresponding tensor slice, while
constructing the stream does not materialize every slice.
-/

@[expose] public section

namespace TorchLean
namespace Data
namespace TensorDataset

/-- Convert `(input, target)` float tensors into supervised samples for scalar type `α`. -/
def ofSupervisedFloatPairs
    {α : Type} [_root_.Context α] [_root_.TorchLean.Runtime.FromFloat α]
    {inputShape targetShape : List Nat}
    (xs : Array (Tensor Float inputShape × Tensor Float targetShape)) :
    SampleStream (_root_.TorchLean.TensorPack α [inputShape, targetShape]) :=
  (SampleStream.ofArray xs).map fun (xF, yF) =>
    let x : Tensor α inputShape :=
      Spec.Tensor.map (_root_.TorchLean.Runtime.ofFloat (α := α)) xF
    let y : Tensor α targetShape :=
      Spec.Tensor.map (_root_.TorchLean.Runtime.ofFloat (α := α)) yF
    _root_.TorchLean.TensorPack! x, y

/-- Convert `(input, label)` pairs into samples with one-hot classification targets. -/
def ofLabeledFloatPairs
    {α : Type} [_root_.Context α] [_root_.TorchLean.Runtime.FromFloat α]
    {inputShape : List Nat}
    (classes : Nat) (xs : Array (Tensor Float inputShape × Fin classes)) :
    SampleStream (_root_.TorchLean.TensorPack α [inputShape, [classes]]) :=
  (SampleStream.ofArray xs).map fun (xF, label) =>
    let x : Tensor α inputShape :=
      Spec.Tensor.map (_root_.TorchLean.Runtime.ofFloat (α := α)) xF
    let yF : Tensor Float [classes] :=
      TorchLean.Tensor.oneHot (α := Float) classes label
    let y : Tensor α [classes] :=
      Spec.Tensor.map (_root_.TorchLean.Runtime.ofFloat (α := α)) yF
    _root_.TorchLean.TensorPack! x, y

/-- Slice every tensor in a pack at the same leading-axis index. -/
def unbatch {β : Type} {n : Nat} :
    {shapes : List Shape} →
      _root_.TorchLean.TensorPack β (shapes.map (fun shape => shape.prependDim n)) →
      Fin n →
      _root_.TorchLean.TensorPack β shapes
  | [], .nil, _ => .nil
  | _ :: shapes, .cons x xs, i =>
      .cons (Spec.get x i) (unbatch (β := β) (shapes := shapes) xs i)

/-- Lazily expose the leading-axis slices of an aligned tensor pack. -/
def ofBatched {β : Type} {n : Nat} {shapes : List Shape}
    (xs : _root_.TorchLean.TensorPack β
      (shapes.map (fun shape => shape.prependDim n))) :
    SampleStream (_root_.TorchLean.TensorPack β shapes) :=
  SampleStream.ofFn n fun i => unbatch (β := β) (shapes := shapes) xs i

/-- Lazily slice a float tensor pack and convert each requested sample to `α`. -/
def ofBatchedFloat {α : Type} [_root_.TorchLean.Runtime.FromFloat α]
    {n : Nat} {shapes : List Shape}
    (xs : _root_.TorchLean.TensorPack Float
      (shapes.map (fun shape => shape.prependDim n))) :
    SampleStream (_root_.TorchLean.TensorPack α shapes) :=
  (ofBatched (β := Float) (n := n) (shapes := shapes) xs).map fun sample =>
    _root_.TorchLean.TensorPack.map
      (fun x => Spec.Tensor.map (_root_.TorchLean.Runtime.ofFloat (α := α)) x) sample

end TensorDataset
end Data
end TorchLean
