/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Public.Facade.Base.Runtime

/-!
# TorchLean Tensor Names

Ops, random, shape, tensor, and semantic names exposed by the `NN` umbrella.
-/

@[expose] public section

namespace TorchLean

namespace Ops

/-!
Low-level executable ops for verification and compiler-facing examples.

Most model code should use `nn.*` and `Trainer.*`. Use `Ops.*` when writing an explicit
TorchLean executable program directly, for example before compiling a hand-built fragment to
`NN.IR.Graph`.
-/

export NN.API.TorchLean
  (RefTy const add sub mul scale abs sqrt clamp max min
   broadcastTo reshape swapAdjacentAtDepth reduceSum reduceMean
   gatherScalar gatherRow gatherScalarNat gatherVecNat gatherRowsNat scatterAddVec scatterAddRow
   matmul concatVectors
   maxPool avgPool smoothMaxPool
   relu silu gelu sigmoid tanh softmax softplus exp log inv safeLog logSoftmax
   sum flatten linear mseLoss layerNorm multiHeadAttention conv convTranspose)

end Ops

namespace rand

export NN.API.rand
  (keyOf uniform mask uniformND maskND randND
   nextSeed nextSeedGlobal nextSeedsGlobal)

end rand

namespace Shape

/-- Scalar shape with no tensor dimensions. -/
abbrev scalar := Spec.Shape.scalar

/-- Number of scalar entries represented by a shape. -/
abbrev size := Spec.Shape.size

/-- Dimensions of a shape in outermost-to-innermost order. -/
abbrev toList := Spec.Shape.toList

/-- Construct a shape from its outermost-to-innermost dimensions. -/
abbrev ofDims := NN.Tensor.shapeOfDims

end Shape

namespace Tensor

/-- Construct a length-indexed vector tensor from a function on indices. -/
abbrev vector := NN.Tensor.vector

/-- Check a flat list against `dims` and reshape it into the corresponding tensor. -/
abbrev ofList {α : Type} (dims : List Nat) (xs : List α) :
    Except String (Spec.Tensor α (Shape.ofDims dims)) :=
  NN.Tensor.ofList dims xs

/-- Construct an executable IEEE32 vector from ordinary `Float` values. -/
abbrev float32Vector := NN.Tensor.float32Vector

/-- Print a tensor using the scalar type's registered display implementation. -/
abbrev print {α : Type} [NN.Tensor.DTypeName α] [NN.Tensor.TensorPrintable α]
    {s : Shape} (t : Spec.Tensor α s) : IO Unit :=
  NN.Tensor.print t

/-- Convert a list to a vector whose shape records the list length. -/
abbrev vectorFromList {α : Type} (xs : List α) :
    Spec.Tensor α (.dim xs.length .scalar) :=
  Spec.vectorFromList xs

/-- Convert rectangular list rows into a shape-indexed matrix tensor. -/
abbrev matrixFromRows {α : Type} [Inhabited α] (xss : List (List α)) :=
  Spec.matrixFromRows xss

/-- Fill every scalar position of a shape with one value. -/
abbrev fill {α : Type} (value : α) (s : Shape) : Spec.Tensor α s :=
  Spec.fill value s

/-- Render a shape-indexed tensor as a compact human-readable string. -/
abbrev pretty {α : Type} [ToString α] {s : Shape} (t : Spec.Tensor α s) : String :=
  Spec.pretty t

/-- Read one element from a vector tensor at a bounded index. -/
abbrev vecGet {α : Type} {n : Nat} (x : Spec.Tensor α (.dim n .scalar)) (i : Fin n) : α :=
  Spec.Tensor.vecGet x i

/-- Extract the value stored in a scalar tensor. -/
abbrev toScalar {α : Type} (t : Spec.Tensor α .scalar) : α :=
  Spec.Tensor.toScalar t

/-- Public shorthand for TorchLean's shape-indexed tensor family. -/
abbrev T := Spec.Tensor

/-- Apply a scalar function pointwise while preserving the tensor shape. -/
abbrev map {α β : Type} {s : Shape} (f : α → β) (x : Tensor.T α s) : Tensor.T β s :=
  Spec.mapTensor f x

/-- Convert a `Float` tensor pointwise with an explicitly supplied scalar cast. -/
abbrev castFloat {α : Type} (cast : Float → α) {s : Shape} (t : Tensor.T Float s) :
    Tensor.T α s :=
  NN.API.Common.castTensor cast t

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
def toFloatIO {α : Type} [Runtime.TensorScalar α]
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

export NN.API.Common
  (tensorF tensorFGen tensorFGen! tensorFGenShape!)

end Tensor

namespace Semantics

/-- Scalar rectified-linear activation used by public model definitions. -/
abbrev relu {α : Type} [Zero α] [Max α] (x : α) : α :=
  NN.API.Semantics.relu x

end Semantics

/-!
## Public Namespaces

These are the names users usually type for model construction, tensor utilities, training, runtime
selection, and verification examples. The definitions below forward to the checked implementation;
the semantics are not copied or forked.
-/

end TorchLean
