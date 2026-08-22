/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Runtime
public import NN.API.Rand
public import NN.Runtime.Autograd.TorchLean.Functional.ShapeOps
public import NN.Spec.Core.Sequence
public import NN.Spec.Core.TensorReductionShape

/-!
# Shape-Indexed Tensors

Tensor construction, shape operations, runtime conversion, and executable tensor operators.
-/

@[expose] public section

namespace TorchLean

export Spec (Shape)

/--
TorchLean's shape-indexed tensor type.

The scalar type is the first parameter and the statically known shape is the second.
-/
abbrev Tensor := Spec.Tensor

namespace Ops

/-!
Low-level executable ops for verification and graph-lowering examples.

Most model code should use `nn.*` and `Trainer.*`. Use `Ops.*` when writing an explicit
TorchLean executable program directly, for example before lowering a hand-built fragment to
`NN.IR.Graph`.
-/

export _root_.Runtime.Autograd.TorchLean (RefTy)
export _root_.Runtime.Autograd.Torch
  (const add sub mul scale abs sqrt clamp max min
   broadcastTo reshape reduceSum reduceMean
   gatherScalar gatherRow gatherScalarNatOrZero gatherVecNatOrZero gatherRowsNatOrZero scatterAddVec scatterAddRow
   mm
   maxPool avgPool smoothMaxPool
   relu silu gelu sigmoid tanh softplus exp log inv safeLog
   sum flatten linear mseLoss layerNorm multiHeadAttention conv convTranspose)

/--
Permute arbitrary tensor dimensions, checking the requested output shape.

The list gives each output dimension's source dimension, as in `torch.permute`. The result is
`none` when the list is not a permutation of all input dimensions or when its computed shape is not
the expected result shape.
-/
def permute {α : Type} [Context α] [DecidableEq Spec.Shape]
    {m : Type → Type} [Monad m]
    [_root_.Runtime.Autograd.Torch.Ops (m := m) (α := α)]
    {s sOut : Spec.Shape} (axes : List Nat)
    (x : _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) s) :
    m (Option (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) sOut)) :=
  _root_.Runtime.Autograd.TorchLean.F.permute (m := m) (α := α) axes x

/--
Permute arbitrary tensor dimensions and return the statically unknown result shape.

Use this form at dynamic data boundaries. Statically shaped model code should normally use
`permute`, whose expected result shape remains in the type.
-/
def permute? {α : Type} [Context α] [DecidableEq Spec.Shape]
    {m : Type → Type} [Monad m]
    [_root_.Runtime.Autograd.Torch.Ops (m := m) (α := α)]
    {s : Spec.Shape} (axes : List Nat)
    (x : _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) s) :
    m (Option (Σ s' : Spec.Shape,
      _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) s')) :=
  _root_.Runtime.Autograd.TorchLean.F.permute? (m := m) (α := α) axes x

/-- Apply softmax along any valid tensor dimension. -/
def softmax {α : Type} [Context α] [DecidableEq Spec.Shape]
    {m : Type → Type} [Monad m]
    [_root_.Runtime.Autograd.Torch.Ops (m := m) (α := α)]
    {s : Spec.Shape} (axis : Nat) [Spec.Shape.AxisInBounds axis s]
    (x : _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) s) :
    m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) s) :=
  _root_.Runtime.Autograd.TorchLean.F.softmax (m := m) (α := α) axis x

/-- Apply stable log-softmax along any valid tensor dimension. -/
def logSoftmax {α : Type} [Context α] [DecidableEq Spec.Shape]
    {m : Type → Type} [Monad m]
    [_root_.Runtime.Autograd.Torch.Ops (m := m) (α := α)]
    {s : Spec.Shape} (axis : Nat) [Spec.Shape.AxisInBounds axis s]
    (x : _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) s) :
    m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) s) :=
  _root_.Runtime.Autograd.TorchLean.F.logSoftmax (m := m) (α := α) axis x

end Ops

namespace Shape

export Spec.Shape (scalar size toList ofList)

end Shape

namespace Tensor

export Spec.Tensor
  (scalar dim vecGet item mapLeading zipWithLeading sumLeadingAxis reverseLeadingAxis)
export NN.Tensor
  (vector oneHot oneHotNatOrZero oneHotIndicesOrZero matrix? matrix matrixResize matrixPadRight
   ofListOfLength ofList someTensorOfList fillOfDims zerosOfDims onesOfDims print)
export Spec (vectorFromList matrixFromRows? fill pretty)

/--
Preserve `leading`, flatten the remaining source shape, and retain its first `takeDim` entries.

The proof `hTake` rules out truncation beyond the flattened source shape. A conventional batch is
the special case `leading = shape![batch]`; several leading axes are handled by the same operation.
-/
def flattenPrefix {α : Type} [Inhabited α]
    (leading : Shape) (takeDim : Nat) {source : Shape}
    (hTake : takeDim ≤ Shape.size source)
    (x : Spec.Tensor α (leading.concat source)) :
    Spec.Tensor α (leading.appendDim takeDim) :=
  match leading with
  | .scalar =>
    let flat := Spec.Tensor.flattenSpec x
    Spec.Tensor.dim (fun j =>
      let h : j.val < Shape.size source := Nat.lt_of_lt_of_le j.isLt hTake
      Spec.Tensor.scalar (Spec.Tensor.item (Spec.get flat ⟨j.val, h⟩)))
  | .dim _ rest =>
      Spec.Tensor.dim (fun i => flattenPrefix rest takeDim hTake (Spec.getAtSpec x i))

/-- Apply a scalar function pointwise while preserving the tensor shape. -/
abbrev map {α β : Type} {s : Shape} (f : α → β) (x : Tensor α s) : Tensor β s :=
  Spec.mapTensor f x

/--
Construct a tensor from a flat list of `Float` values and convert each entry to the selected scalar
type. The list length must equal the product of `dims`.
-/
def fromFloatList {α : Type} [Context α]
    (cast : Float → α) (dims : List Nat) (xs : List Float) :
    Except String (Tensor α (Shape.ofList dims)) := do
  let tensor ← NN.Tensor.ofList (α := Float) dims xs
  pure (map cast tensor)

/--
Generate a tensor from its flat element index and convert each generated `Float` to the selected
scalar type.
-/
def generateFromFloat {α : Type} [Context α]
    (cast : Float → α) (dims : List Nat) (f : Nat → Float) :
    Tensor α (Shape.ofList dims) :=
  let xs := (List.range dims.prod).map f
  have hLength : xs.length = dims.prod := by
    simp [xs]
  map cast (NN.Tensor.ofListOfLength (α := Float) (dims := dims) (xs := xs) hLength)

/-- Generate a tensor of a statically known shape from its flat element index. -/
def generateFromFloatShape {α : Type} [Context α]
    (cast : Float → α) (shape : Shape) (f : Nat → Float) : Tensor α shape := by
  simpa using (generateFromFloat (α := α) cast shape.toList f)

/--
Convert a runtime tensor back to a `Float` tensor inside `IO`.

Trainer prediction results use this so examples can train under executable IEEE32 or another scalar
backend, then inspect ordinary `Float` tensors afterward.
-/
def toFloatIO {α : Type} [_root_.Context α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α] :
    ∀ {s : Shape}, Tensor α s → IO (Tensor Float s)
  | .scalar, .scalar x => do
      pure <| .scalar (← _root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv.toFloat
        (α := α) x)
  | .dim n s, .dim f => do
      let xs ← Array.mapM' (fun t => toFloatIO (s := s) t) (Array.ofFn f)
      have hSize : xs.val.size = n := by
        simpa using xs.property
      pure <| .dim (fun i =>
        xs.val[i.val]'(by
          rw [hSize]
          exact i.isLt))

end Tensor

/-!
## Related Namespaces

Model construction, training, runtime selection, and verification live in their corresponding
`TorchLean` namespaces.
-/

end TorchLean
