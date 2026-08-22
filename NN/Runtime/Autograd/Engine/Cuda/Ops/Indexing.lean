/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Engine.Cuda.Ops.Linear

/-!
# CUDA Tape Operations: Concatenation, Slicing, and Indexing
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Cuda

open Spec
open Tensor

namespace Tape

/-!
## Vector slice
-/

/-- Slice `len` entries from a one-dimensional CUDA buffer starting at `start`. -/
def sliceVectorBuffer {n start len : Nat} (t : Tape) (xId : Nat) : Result (Tape × Nat) := do
  if start + len ≤ n then
    let n32 ← AnyBuffer.natToU32Checked n
    let start32 ← AnyBuffer.natToU32Checked start
    let len32 ← AnyBuffer.natToU32Checked len
    let outShape : Shape := .dim len .scalar
    let x ← requireValue (t := t) xId (.dim n .scalar)
    let y := Buffer.sliceVectorBuffer x n32 start32 len32
    let node : Node :=
      { name := some s!"slice_vector_buffer[{start}:{start+len}]"
        value := { s := outShape, buf := y }
        requiresGrad := true
        parents := [xId]
        backward := fun dLdyAny => do
          let dLdy ← requireGrad dLdyAny outShape
          let pre := Buffer.zeros start32
          let postLen : Nat := n - start - len
          let post32 ← AnyBuffer.natToU32Checked postLen
          let post := Buffer.zeros post32
          let tmp := Buffer.concatVectorBuffers pre dLdy.buf start32 len32
          let startLen32 ← AnyBuffer.natToU32Checked (start + len)
          let dx := Buffer.releaseThen pre <|
            Buffer.releaseThen post <|
              Buffer.releaseThen tmp <|
                Buffer.concatVectorBuffers tmp post startLen32 post32
          pure [(xId, { s := .dim n .scalar, buf := dx })] }
    pure (t.addNode node)
  else
    throw "autograd: slice_vector_buffer: start+len out of bounds"

/-!
## Concat / slice along dim 0
-/

/-- Concatenate along dim 0 for tensors with leading dimension (CPU tape name). -/
def concatLeadingAxis {n m : Nat} {s : Shape} (t : Tape) (aId bId : Nat) : Result (Tape × Nat) := do
  let inner : Nat := Spec.Shape.size s
  let nLen : Nat := n * inner
  let mLen : Nat := m * inner
  let nLen32 ← AnyBuffer.natToU32Checked nLen
  let mLen32 ← AnyBuffer.natToU32Checked mLen
  let nmLen32 ← AnyBuffer.natToU32Checked (nLen + mLen)
  binary (t := t) "concat_leading_axis" aId bId (.dim n s) (.dim m s) (.dim (n + m) s)
    (forward := fun a b => Buffer.concatVectorBuffers a b nLen32 mLen32)
    (backward := fun _a _b dLdy =>
      let dA := Buffer.sliceVectorBuffer dLdy nmLen32 0 nLen32
      let dB := Buffer.sliceVectorBuffer dLdy nmLen32 nLen32 mLen32
      (dA, dB))

/-- Slice along dim 0: `x[start:start+len]` (CPU tape name). -/
def sliceLeadingAxisRange {n : Nat} {s : Shape} (t : Tape) (xId : Nat) (start len : Nat)
    (_h : start + len ≤ n) : Result (Tape × Nat) := do
  let inner : Nat := Spec.Shape.size s
  let nTot : Nat := n * inner
  let startOff : Nat := start * inner
  let lenTot : Nat := len * inner
  let rightTot : Nat := nTot - (startOff + lenTot)
  let nTot32 ← AnyBuffer.natToU32Checked nTot
  let start32 ← AnyBuffer.natToU32Checked startOff
  let len32 ← AnyBuffer.natToU32Checked lenTot
  let right32 ← AnyBuffer.natToU32Checked rightTot
  let midLen32 ← AnyBuffer.natToU32Checked (startOff + lenTot)
  unary (t := t) "slice_leading_axis_range" xId (.dim n s) (.dim len s)
    (forward := fun x => Buffer.sliceVectorBuffer x nTot32 start32 len32)
    (backward := fun _x dLdy =>
      let left := Buffer.zeros start32
      let right := Buffer.zeros right32
      let tmp := Buffer.concatVectorBuffers left dLdy start32 len32
      Buffer.releaseThen left <|
        Buffer.releaseThen right <|
          Buffer.releaseThen tmp <|
            Buffer.concatVectorBuffers tmp right midLen32 right32)

/-!
## Gather / scatter (host Nat indices)

Indices are non-differentiable and remain on the host. Kernels totalize representable out-of-bounds
indices as documented in `NN.Runtime.Autograd.Engine.Cuda.Kernels`; the tape wrappers reject
natural numbers outside the `UInt32` ABI before entering native code.
-/

/-- Gather a scalar from a 1D vector using a compile-time index. -/
def gatherScalar {n : Nat} (t : Tape) (xId : Nat) (i : Fin n) : Result (Tape × Nat) := do
  let n32 ← AnyBuffer.natToU32Checked n
  let one32 : UInt32 := 1
  let indices : Array Nat := #[i.val]
  let x ← requireValue (t := t) xId (.dim n .scalar)
  let y := Buffer.gatherVec x n32 indices one32
  let node : Node :=
    { name := some s!"gather_scalar[{i.val}]"
      value := { s := Shape.scalar, buf := y }
      requiresGrad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny Shape.scalar
        let zeros := Buffer.zeros n32
        let dx := Buffer.releaseThen zeros <|
          Buffer.scatterAdd zeros dLdy.buf n32 indices one32
        pure [(xId, { s := .dim n .scalar, buf := dx })] }
  pure (t.addNode node)

/-- Gather a row from a 2D matrix using a compile-time index. -/
def gatherRow {rows cols : Nat} (t : Tape) (xId : Nat) (i : Fin rows) : Result (Tape × Nat) := do
  let rows32 ← AnyBuffer.natToU32Checked rows
  let cols32 ← AnyBuffer.natToU32Checked cols
  let one32 : UInt32 := 1
  let i32 ← AnyBuffer.natToU32Checked i.val
  let indices : Array Nat := #[i.val]
  let x ← requireValue (t := t) xId (.dim rows (.dim cols .scalar))
  let y := Buffer.gatherRows x rows32 cols32 indices one32
  let node : Node :=
    { name := some s!"gather_row[{i.val}]"
      value := { s := .dim cols .scalar, buf := y }
      requiresGrad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny (.dim cols .scalar)
        let zerosLen ← AnyBuffer.natToU32Checked (rows * cols)
        let zeros := Buffer.zeros zerosLen
        let dx := Buffer.releaseThen zeros <|
          Buffer.scatterAddRow zeros dLdy.buf rows32 cols32 i32
        pure [(xId, { s := .dim rows (.dim cols .scalar), buf := dx })] }
  pure (t.addNode node)

/-- Gather a scalar from a 1D vector using a runtime `Nat` index (totalized by the kernel). -/
def gatherScalarNatOrZero {n : Nat} (t : Tape) (xId : Nat) (i : Nat) : Result (Tape × Nat) := do
  let n32 ← AnyBuffer.natToU32Checked n
  if (UInt32.ofNat i).toNat != i then
    throw "autograd: cuda: gather_scalar_nat_or_zero: index does not fit in UInt32"
  let one32 : UInt32 := 1
  let indices : Array Nat := #[i]
  let x ← requireValue (t := t) xId (.dim n .scalar)
  let y := Buffer.gatherVec x n32 indices one32
  let node : Node :=
    { name := some s!"gather_scalar_nat_or_zero[{i}]"
      value := { s := Shape.scalar, buf := y }
      requiresGrad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny Shape.scalar
        let zeros := Buffer.zeros n32
        let dx := Buffer.releaseThen zeros <|
          Buffer.scatterAdd zeros dLdy.buf n32 indices one32
        pure [(xId, { s := .dim n .scalar, buf := dx })] }
  pure (t.addNode node)

/-- Convert a length-`k` natural-number tensor into the index array expected by CUDA gather/scatter kernels. -/
def natTensorToIndexArray {k : Nat} (idx : Tensor Nat (.dim k .scalar)) : Array Nat :=
  match idx with
  | .dim f =>
      Array.ofFn (fun i : Fin k =>
        match f i with
        | .scalar n => n)

/-- Gather `k` scalars from a length-`n` vector. -/
def gatherVecNatOrZero {n k : Nat} (t : Tape) (xId : Nat) (idx : Tensor Nat (.dim k .scalar)) :
    Result (Tape × Nat) := do
  let n32 ← AnyBuffer.natToU32Checked n
  let k32 ← AnyBuffer.natToU32Checked k
  let indices := natTensorToIndexArray (k := k) idx
  for i in indices do
    if (UInt32.ofNat i).toNat != i then
      throw "autograd: cuda: gather_vec_nat_or_zero: index does not fit in UInt32"
  let x ← requireValue (t := t) xId (.dim n .scalar)
  let y := Buffer.gatherVec x n32 indices k32
  let node : Node :=
    { name := some "gather_vec_nat_or_zero"
      value := { s := .dim k .scalar, buf := y }
      requiresGrad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny (.dim k .scalar)
        -- Scatter-add the gathered gradient back into the length-`n` input.
        let zeros := Buffer.zeros n32
        let dx := Buffer.releaseThen zeros <|
          Buffer.scatterAdd zeros dLdy.buf n32 indices k32
        pure [(xId, { s := .dim n .scalar, buf := dx })] }
  pure (t.addNode node)

/-- Gather `k` rows from a `(rows, cols)` matrix (row-major). -/
def gatherRowsNatOrZero {rows cols k : Nat} (t : Tape) (xId : Nat)
    (idx : Tensor Nat (.dim k .scalar)) :
    Result (Tape × Nat) := do
  let rows32 ← AnyBuffer.natToU32Checked rows
  let cols32 ← AnyBuffer.natToU32Checked cols
  let k32 ← AnyBuffer.natToU32Checked k
  let indices := natTensorToIndexArray (k := k) idx
  for i in indices do
    if (UInt32.ofNat i).toNat != i then
      throw "autograd: cuda: gather_rows_nat_or_zero: index does not fit in UInt32"
  let x ← requireValue (t := t) xId (.dim rows (.dim cols .scalar))
  let y := Buffer.gatherRows x rows32 cols32 indices k32
  let node : Node :=
    { name := some "gather_rows_nat_or_zero"
      value := { s := .dim k (.dim cols .scalar), buf := y }
      requiresGrad := true
      parents := [xId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny (.dim k (.dim cols .scalar))
        -- Scatter-add the gathered row gradients back into the `(rows, cols)` input.
        let zerosLen ← AnyBuffer.natToU32Checked (rows * cols)
        let zeros := Buffer.zeros zerosLen
        let dx := Buffer.releaseThen zeros <|
          Buffer.scatterAddRows zeros dLdy.buf rows32 cols32 indices k32
        pure [(xId, { s := .dim rows (.dim cols .scalar), buf := dx })] }
  pure (t.addNode node)

/-- Scatter-add into a vector: `out = x` with `out[i] += v`. -/
def scatterAddVec {n : Nat} (t : Tape) (xId vId : Nat) (i : Fin n) : Result (Tape × Nat) := do
  let n32 ← AnyBuffer.natToU32Checked n
  let one32 : UInt32 := 1
  let x ← requireValue (t := t) xId (.dim n .scalar)
  let v ← requireValue (t := t) vId Shape.scalar
  let indices : Array Nat := #[i.val]
  let y := Buffer.scatterAdd x v n32 indices one32
  let node : Node :=
    { name := some s!"scatter_add_vec[{i.val}]"
      value := { s := .dim n .scalar, buf := y }
      requiresGrad := true
      parents := [xId, vId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny (.dim n .scalar)
        let dv1 := Buffer.gatherVec dLdy.buf n32 indices one32
        let dx := Buffer.copy dLdy.buf
        -- `gatherVec` returns length-1; reinterpret as scalar (same numel).
        pure [
          (xId, { s := .dim n .scalar, buf := dx }),
          (vId, { s := Shape.scalar, buf := dv1 })
        ] }
  pure (t.addNode node)

/-- Scatter-add into a matrix row: `out = x` with `out[i,:] += v`. -/
def scatterAddRow {rows cols : Nat} (t : Tape) (xId vId : Nat) (i : Fin rows) :
    Result (Tape × Nat) := do
  let rows32 ← AnyBuffer.natToU32Checked rows
  let cols32 ← AnyBuffer.natToU32Checked cols
  let one32 : UInt32 := 1
  let i32 ← AnyBuffer.natToU32Checked i.val
  let x ← requireValue (t := t) xId (.dim rows (.dim cols .scalar))
  let v ← requireValue (t := t) vId (.dim cols .scalar)
  let y := Buffer.scatterAddRow x v rows32 cols32 i32
  let node : Node :=
    { name := some s!"scatter_add_row[{i.val}]"
      value := { s := .dim rows (.dim cols .scalar), buf := y }
      requiresGrad := true
      parents := [xId, vId]
      backward := fun dLdyAny => do
        let dLdy ← requireGrad dLdyAny (.dim rows (.dim cols .scalar))
        let indices : Array Nat := #[i.val]
        let dv1 := Buffer.gatherRows dLdy.buf rows32 cols32 indices one32
        let dx := Buffer.copy dLdy.buf
        -- `gatherRows` returns (1,cols) laid out as length `cols`; reinterpret as vector.
        pure [
          (xId, { s := .dim rows (.dim cols .scalar), buf := dx }),
          (vId, { s := .dim cols .scalar, buf := dv1 })
        ] }
  pure (t.addNode node)
end Tape

end Cuda
end Autograd
end Runtime
