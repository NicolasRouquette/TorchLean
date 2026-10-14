/-
Copyright (c) 2026 TorchLean contributors
Released under MIT license as described in the file LICENSE.
Authors: TorchLean contributors
-/
module

public import NN.Kernel.Cuda
public import NN.Tensor.Constructors
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Value.CoreProof

/-!
# Exact scalar encodings for custom computations

Native floats use hardware arithmetic. Configured binary values use their complete interchange
words; the device selects compatible native operations or software arithmetic from the format.
Wide values are never narrowed through binary64.
The packing law concerns Lean values, not the foreign transfer or arithmetic implementation.
-/

@[expose] public section

namespace NN.Kernel

open FloatLib.Floats FloatLib.Floats.Formats.BinaryInterchange

/-- Scalar operations in their original evaluation order, without algebraic reassociation. -/
def Arithmetic.ofOps {α : Type} [Add α] [Sub α] [Mul α] [Div α] [Neg α]
    [BEq α] [LT α] [LE α] [DecidableLT α] [DecidableLE α] : Arithmetic α where
  binary
    | .add, x, y => x + y
    | .sub, x, y => x - y
    | .mul, x, y => x * y
    | .div, x, y => x / y
  neg := fun x => -x
  compare
    | .eq, x, y => x == y
    | .lt, x, y => x < y
    | .le, x, y => x ≤ y

/-- Arithmetic representation selected by a scalar type, independently of the execution device. -/
inductive Precision where
  | native (format : Cuda.Format)
  | binary (format : FloatFormat)

/-- Complete unsigned encodings and their lossless reconstruction. -/
class Scalar (α : Type) where
  precision : Precision
  /-- Operations implemented by the selected GPU representation. -/
  arithmetic : Arithmetic α
  encode : α → Nat
  decode : Nat → α
  decode_encode : ∀ x, decode (encode x) = x

instance instScalarFloat32 : Scalar Float32 where
  precision := .native .binary32
  arithmetic := Arithmetic.ofOps
  encode x := x.toBits.toNat
  decode n := Float32.ofBits n.toUInt32
  decode_encode x := by
    simpa only [Nat.toUInt32, UInt32.ofNat_toNat, Cuda.Literal.ofValue, Cuda.Literal.value] using
      Cuda.Literal.value_ofValue .binary32 x

instance instScalarFloat : Scalar Float where
  precision := .native .binary64
  arithmetic := Arithmetic.ofOps
  encode x := x.toBits.toNat
  decode n := Float.ofBits n.toUInt64
  decode_encode x := by
    simpa only [Nat.toUInt64, UInt64.ofNat_toNat, Cuda.Literal.ofValue, Cuda.Literal.value] using
      Cuda.Literal.value_ofValue .binary64 x

instance instScalarConfigured {format : FloatFormat} {plan : Configured.StoragePlan format}
    {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code] :
    Scalar (ExecFloat (Configured.Family format code plan)) where
  precision := .binary format
  arithmetic := Arithmetic.ofOps
  encode := ExecFloat.Binary.toNatBits
  decode := ExecFloat.Binary.ofNatBits
  decode_encode := ExecFloat.Binary.ofNatBits_toNatBits

/-- Byte width of one complete value, padded only to a whole number of bytes. -/
def Precision.bytes : Precision → Nat
  | .native .binary32 => 4
  | .native .binary64 => 8
  | .binary format => (format.bitWidth + 7) / 8

/-- Whether the custom GPU arithmetic implementation accepts this complete scalar format.
This is a type/format capability check, not a claim that a GPU or enough memory is available. -/
def Precision.supportsGpu : Precision → Bool
  | .native _ => true
  | .binary format => format.expWidth ≤ 30 && format.fracWidth ≤ 4096

/-- Serialize complete scalar words in row-major little-endian order. -/
def Scalar.pack {α : Type} [TorchLean.Storage α] {shape : Spec.Shape} (scalar : Scalar α)
    (tensor : TorchLean.Tensor α shape) : Except String ByteArray := do
  let mut bytes := ByteArray.empty
  for h : i in [:shape.size] do
    let bits := scalar.encode (TorchLean.Tensor.Internal.Rep.getFlat tensor
      ⟨i, by simpa only [Spec.Shape.internalSize_eq] using h.upper⟩)
    unless bits < 2 ^ (8 * scalar.precision.bytes) do
      throw "custom computation: input encoding exceeds the scalar byte width"
    for j in [:scalar.precision.bytes] do
      bytes := bytes.push ((bits >>> (8 * j)) % 256).toUInt8
  return bytes

/-- Decode complete words only after checking the requested shape and byte extent. -/
def Scalar.unpack {α : Type} [TorchLean.Storage α] (scalar : Scalar α) (shape : Spec.Shape)
    (bytes : ByteArray) : Except String (TorchLean.Tensor α shape) := do
  let width := scalar.precision.bytes
  unless bytes.size == shape.size * width do
    throw s!"custom computation: expected {shape.size * width} bytes, received {bytes.size}"
  return TorchLean.Tensor.generateFlat shape fun i => scalar.decode <| Id.run do
    let mut bits := 0
    for j in [:width] do
      bits := bits + (bytes.get! (i * width + j)).toNat * 2 ^ (8 * j)
    return bits

end NN.Kernel
