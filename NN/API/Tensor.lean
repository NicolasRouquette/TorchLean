/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Runtime
public import NN.API.Rand
public import NN.Spec.Core.TensorReductionShape

/-!
# Shape-Indexed Tensors

Tensor construction, shape operations, runtime conversion, and executable tensor operators.
-/

@[expose] public section

namespace TorchLean

export Spec (Shape)

namespace Ops

/-!
Low-level executable ops for verification and compiler-facing examples.

Most model code should use `nn.*` and `Trainer.*`. Use `Ops.*` when writing an explicit
TorchLean executable program directly, for example before compiling a hand-built fragment to
`NN.IR.Graph`.
-/

export _root_.Runtime.Autograd.TorchLean (RefTy)
export _root_.Runtime.Autograd.Torch
  (const add sub mul scale abs sqrt clamp max min
   broadcastTo reshape swapAdjacentAtDepth reduceSum reduceMean
   gatherScalar gatherRow gatherScalarNat gatherVecNat gatherRowsNat scatterAddVec scatterAddRow
   matmul concatVectors
   maxPool avgPool smoothMaxPool
   relu silu gelu sigmoid tanh softmax softplus exp log inv safeLog logSoftmax
   sum flatten linear mseLoss layerNorm multiHeadAttention conv convTranspose)

end Ops

namespace Shape

export Spec.Shape (scalar size toList)

/-- Construct a shape from its outermost-to-innermost dimensions. -/
abbrev ofDims := NN.Tensor.shapeOfDims

end Shape

namespace Tensor

export Spec.Tensor (scalar dim vecGet toScalar)
export NN.Tensor
  (shapeOfDims numelDims vector oneHot oneHotNat matrix? matrix matrixPadTo matrixPadRight
   ofListOfLength ofList dynamicOfList fillOfDims zerosOfDims onesOfDims
   float32Vector float32Matrix print)
export Spec (vectorFromList matrixFromRows fill pretty)

/--
Flatten each sample in a batch and retain its first `takeDim` entries.

The proof `hTake` rules out truncation beyond the flattened source shape, so the result has the
statically known shape `batch × takeDim`.
-/
def flattenBatchPrefix {α : Type} [Inhabited α]
    (batch takeDim : Nat) {source : Shape}
    (hTake : takeDim ≤ Shape.size source)
    (x : Spec.Tensor α (.dim batch source)) :
    Spec.Tensor α (.dim batch (.dim takeDim .scalar)) :=
  Spec.Tensor.dim (fun bi =>
    let flat := Spec.Tensor.flattenSpec (Spec.getAtSpec x bi)
    Spec.Tensor.dim (fun j =>
      let h : j.val < Shape.size source := Nat.lt_of_lt_of_le j.isLt hTake
      Spec.Tensor.scalar (Spec.Tensor.toScalar (Spec.get flat ⟨j.val, h⟩))))

/--
Construct a fixed-length vector from an array.

Entries beyond the array use `fallback`; entries beyond length `n` are ignored. This constructor is
useful at runtime data boundaries where the tensor length is fixed by a model type while the source
array is checked or padded by the caller.
-/
def vectorFromArray {α : Type} (n : Nat) (xs : Array α) (fallback : α) :
    Spec.Tensor α (.dim n .scalar) :=
  Spec.Tensor.dim fun i => Spec.Tensor.scalar (xs.getD i.val fallback)

/-- Public shorthand for TorchLean's shape-indexed tensor family. -/
abbrev T := Spec.Tensor

/-- Apply a scalar function pointwise while preserving the tensor shape. -/
abbrev map {α β : Type} {s : Shape} (f : α → β) (x : Tensor.T α s) : Tensor.T β s :=
  Spec.mapTensor f x

/-- Convert a `Float` tensor pointwise with an explicitly supplied scalar cast. -/
def castFloat {α : Type} (cast : Float → α) {s : Shape} (t : Tensor.T Float s) :
    Tensor.T α s :=
  Spec.mapTensor cast t

/--
Construct a tensor from a flat list of `Float` values and convert each entry to the selected scalar
type. The list length must equal the product of `dims`.
-/
def fromFloatList {α : Type} [Context α]
    (cast : Float → α) (dims : List Nat) (xs : List Float) :
    Except String (Tensor.T α (Shape.ofDims dims)) := do
  let tensor ← NN.Tensor.ofList (α := Float) dims xs
  pure (castFloat cast tensor)

/--
Generate a tensor from its flat element index and convert each generated `Float` to the selected
scalar type.
-/
def generateFromFloat {α : Type} [Context α]
    (cast : Float → α) (dims : List Nat) (f : Nat → Float) :
    Tensor.T α (Shape.ofDims dims) :=
  let xs := (List.range (NN.Tensor.numelDims dims)).map f
  have hLength : xs.length = NN.Tensor.numelDims dims := by
    simp [xs]
  castFloat cast (NN.Tensor.ofListOfLength (α := Float) (dims := dims) (xs := xs) hLength)

/-- Generate a tensor of a statically known shape from its flat element index. -/
def generateFromFloatShape {α : Type} [Context α]
    (cast : Float → α) (shape : Shape) (f : Nat → Float) : Tensor.T α shape := by
  simpa [NN.Tensor.shapeOfDims_toList] using
    (generateFromFloat (α := α) cast shape.toList f)

/--
Repeat one tensor across a fixed batch axis.

Use this for classifier demos whose checked model consumes a whole batch, while the example wants to
inspect one ordinary input.
-/
def repeatBatch {α : Type} {s : Shape} (batch : Nat) (x : Tensor.T α s) :
    Tensor.T α (.dim batch s) :=
  Spec.Tensor.dim (fun _ => x)

/--
Convert a runtime tensor back to a `Float` tensor inside `IO`.

Trainer prediction handles use this so examples can train under executable IEEE32 or another scalar
backend, then inspect ordinary `Float` tensors afterward.
-/
def toFloatIO {α : Type} [_root_.Context α]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α] :
    ∀ {s : Shape}, Tensor.T α s → IO (Tensor.T Float s)
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
