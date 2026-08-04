/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.CheckpointIO
public import NN.Runtime.Autograd.TorchLean.NN
public import NN.Spec.Core.TensorBridge
public import Lean

/-!
# Parameter IO

TorchLean examples often want a small "train once, save weights, reload later" workflow.

This module provides two explicit formats for saving and loading `Float` parameter packs.

## Host Float Format

We encode each `Float` value by its IEEE-754 bit pattern (`Float.toBits : Float → UInt64`) and
store those bits as JSON natural numbers.

This is:
- exact (round-trips every NaN payload and subnormal),
- stable across locales, and
- easy to validate (length = `Spec.Shape.size`).

The file layout is:

```json
{
  "format": "torchlean_paramlist_bits_v1",
  "params": [
    { "shape": [d1, d2, ...], "values": [u64bits, u64bits, ...] },
    ...
  ]
}
```

## Runtime Float32 Format

Device-backed modules use a versioned binary stream. Each parameter records its rank, dimensions,
and element count before its little-endian float32 payload. Loading checks all metadata against the
expected shape-indexed parameter list and rejects unsupported versions, truncated payloads, and
trailing data. The stream is written one tensor at a time, so saving a large CUDA model does not
construct a second host-side copy of the whole checkpoint.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace TorchLean
namespace ParamIO

open Spec

/-- Format tag stored in Float parameter-pack JSON files. -/
def formatTag : String := "torchlean_paramlist_bits_v1"

/-- Versioned header for streamed runtime float32 parameter checkpoints. -/
def float32StreamFormat : Torch.Internal.CheckpointIO.Format where
  name := "TorchLean float32 parameter checkpoint"
  magic := "TLPF32B".toUTF8
  version := 2

/-- Encode a natural number as a JSON number. -/
def jsonNat (n : Nat) : Lean.Json :=
  Lean.Json.num (Lean.JsonNumber.fromInt (Int.ofNat n))

/-- Encode a Float by writing its exact IEEE bit pattern as a JSON natural number. -/
def floatToJsonBits (x : Float) : Lean.Json :=
  jsonNat x.toBits.toNat

/-- Decode a JSON natural number as the exact IEEE bit pattern of a Float. -/
def jsonBitsToFloat (j : Lean.Json) : Except String Float := do
  let n ← Lean.Json.getNat? j
  let limit : Nat := (2 : Nat) ^ 64
  if n >= limit then
    throw s!"ParamIO: float bits out of range (expected < 2^64, got {n})"
  let bits : UInt64 := UInt64.ofNat n
  pure (Float.ofBits bits)

/-- The runtime-list product of a shape agrees with its type-level element count. -/
private theorem shapeProd_toList (s : Shape) :
    TensorArray.shapeProd (Shape.toList s) = Shape.size s := by
  induction s with
  | scalar => simp [Shape.toList, Shape.size]
  | dim n rest ih => simp [Shape.toList, Shape.size, ih]

/-- Rebuild a tensor from a flat list, rejecting length mismatches instead of changing the data. -/
def tensorOfFlatListExact {α : Type} (tag : String) (s : Shape) (xs : List α) :
    Except String (Tensor α s) := do
  if hLength : xs.length = Shape.size s then
    let dims := Shape.toList s
    have hProduct : xs.length = TensorArray.shapeProd dims := by
      simpa [dims, shapeProd_toList] using hLength
    have hShape : TensorBridge.listToShape dims = s := by
      change Shape.ofList (Shape.toList s) = s
      exact Shape.ofList_toList s
    pure <| hShape ▸ TensorBridge.unflatten dims xs hProduct
  else
    throw <|
      s!"{tag}: expected {Shape.size s} scalars for shape {Shape.toList s}, " ++
        s!"got {xs.length}"

/-- Encode one Float tensor as shape metadata plus exact IEEE bit-pattern values. -/
def tensorToJsonBits (s : Shape) (t : Tensor Float s) : Lean.Json :=
  let dims : Lean.Json := Lean.Json.arr (Shape.toList s |>.toArray |>.map jsonNat)
  let values : Lean.Json :=
    Lean.Json.arr ((Spec.toList t).toArray.map floatToJsonBits)
  Lean.Json.mkObj [("shape", dims), ("values", values)]

/-- Decode one shape-checked Float tensor from the bit-pattern parameter format. -/
def tensorFromJsonBits (tag : String) (s : Shape) (j : Lean.Json) :
    Except String (Tensor Float s) := do
  let o ← Lean.Json.getObj? j
  let shapeJ := (o.get? "shape").getD Lean.Json.null
  let valuesJ := (o.get? "values").getD (Lean.Json.arr #[])
  let dimsArr ← Lean.Json.getArr? shapeJ
  let dims : List Nat ←
    dimsArr.toList.mapM (fun d => Lean.Json.getNat? d)
  if Shape.ofList dims != s then
    throw s!"{tag}: shape mismatch (file={dims}, expected={Shape.toList s})"
  let valsArr ← Lean.Json.getArr? valuesJ
  let vals : List Float ←
    valsArr.toList.mapM (fun v => jsonBitsToFloat v)
  tensorOfFlatListExact (tag := tag) s vals

/-- Encode a shape-indexed parameter list as the JSON array stored under `params`. -/
def tListToJsonBits {ss : List Shape} : Torch.TList Float ss → Lean.Json
  | .nil => Lean.Json.arr #[]
  | .cons (s := s) t ts =>
      match tListToJsonBits (ss := _) ts with
      | Lean.Json.arr xs =>
          Lean.Json.arr (#[tensorToJsonBits s t] ++ xs)
      | _ => Lean.Json.arr #[]

/-- Decode an expected shape list from a parameter array, starting at `offset`. -/
def tListFromJsonBitsArray (tag : String) (xs : Array Lean.Json) :
    {ss : List Shape} → (offset : Nat) → Except String (Torch.TList Float ss)
  | [], offset => do
      unless offset = xs.size do
        throw s!"{tag}: unexpected {xs.size - offset} trailing parameter record(s)"
      pure .nil
  | s :: ss, offset => do
      let headJson ← match xs[offset]? with
        | some value => pure value
        | none => throw s!"{tag}: missing parameter {offset + 1}/{offset + 1 + ss.length}"
      let head ← tensorFromJsonBits (tag := tag) s headJson
      let tail ← tListFromJsonBitsArray (tag := tag) xs (ss := ss) (offset + 1)
      pure (.cons head tail)

/-- Decode the `params` JSON array into the expected shape-indexed parameter list. -/
def tListFromJsonBits (tag : String) {ss : List Shape} (json : Lean.Json) :
    Except String (Torch.TList Float ss) := do
  let xs ← Lean.Json.getArr? json
  tListFromJsonBitsArray (tag := tag) xs (ss := ss) 0

/-- Write Float parameters using exact IEEE bit patterns rather than decimal floats. -/
def writeParamBits (path : System.FilePath) {ss : List Shape}
    (ps : Torch.TList Float ss) (pretty : Bool := true) : IO Unit := do
  let top : Lean.Json :=
    Lean.Json.mkObj [("format", Lean.Json.str formatTag), ("params", tListToJsonBits ps)]
  let s := if pretty then top.pretty else top.compress
  Torch.Internal.CheckpointIO.writeAtomically path fun handle =>
    handle.write s.toUTF8

/-- Read Float parameters previously written by `writeParamBits`. -/
def readParamBits (path : System.FilePath) {ss : List Shape} :
    IO (Except String (Torch.TList Float ss)) := do
  let s ← IO.FS.readFile path
  match Lean.Json.parse s with
  | Except.error e =>
      pure (Except.error s!"ParamIO: JSON parse error: {e}")
  | Except.ok j =>
      match Lean.Json.getObj? j with
      | Except.error e =>
          pure (Except.error s!"ParamIO: expected object: {e}")
      | Except.ok o =>
          let fmt := (o.get? "format").getD (Lean.Json.str "")
          match Lean.Json.getStr? fmt with
          | Except.error _ =>
              pure (Except.error "ParamIO: missing `format` string")
          | Except.ok t =>
              if t != formatTag then
                pure (Except.error s!"ParamIO: unsupported format: {t}")
              else
                let paramsJ := (o.get? "params").getD (Lean.Json.arr #[])
                pure (tListFromJsonBits (tag := "ParamIO") (ss := ss) paramsJ)

/-! ## Streaming float32 module checkpoints -/

/-- Write one expected tensor shape to a streaming checkpoint. -/
def writeShape (handle : IO.FS.Handle) (shape : Shape) : IO Unit := do
  let dims := Shape.toList shape
  Torch.Internal.CheckpointIO.writeNat64 "ParamIO" handle dims.length
  for dim in dims do
    Torch.Internal.CheckpointIO.writeNat64 "ParamIO" handle dim
  Torch.Internal.CheckpointIO.writeNat64 "ParamIO" handle (Shape.size shape)

/-- Read and validate one tensor shape from a streaming checkpoint. -/
def readShape (handle : IO.FS.Handle) (expected : Shape) : IO Unit := do
  let rank ← Torch.Internal.CheckpointIO.readNat64 "ParamIO" handle
  let mut reversedDims := []
  for _ in [0:rank] do
    reversedDims := (← Torch.Internal.CheckpointIO.readNat64 "ParamIO" handle) :: reversedDims
  let dims := reversedDims.reverse
  let count ← Torch.Internal.CheckpointIO.readNat64 "ParamIO" handle
  if Shape.ofList dims != expected then
    throw <| IO.userError
      s!"ParamIO: checkpoint shape mismatch (file={dims}, expected={Shape.toList expected})"
  if count != Shape.size expected then
    throw <| IO.userError <|
      s!"ParamIO: checkpoint element count mismatch for {Shape.pretty expected} "
        ++ s!"(file={count}, expected={Shape.size expected})"

/-- Obtain one parameter as raw float32 bytes without materializing the whole parameter pack. -/
def paramFloat32Bytes {shape : Shape}
    (param : Torch.Param Float shape) : IO ByteArray := do
  match ← param.cudaValue.get with
  | some value =>
      if value.s != shape then
        throw <| IO.userError <|
          s!"ParamIO: CUDA parameter shape mismatch "
            ++ s!"(buffer={Shape.pretty value.s}, expected={Shape.pretty shape})"
      Runtime.Autograd.Cuda.Buffer.toFloat32BytesIO value.buf
  | none =>
      let tensor ← param.value.get
      let values := Runtime.Autograd.Cuda.Convert.flattenFloat (s := shape) tensor
      pure <| Runtime.Autograd.Cuda.Buffer.floatArrayToFloat32Bytes values

/-- Stream a shape-indexed runtime parameter list to an open checkpoint handle. -/
def writeParamListFloat32 (handle : IO.FS.Handle) :
    {shapes : List Shape} → Torch.ParamList Float shapes → IO Unit
  | [], .nil => pure ()
  | shape :: _, .cons param params => do
      writeShape handle shape
      let bytes ← paramFloat32Bytes param
      let expectedBytes := Shape.size shape * 4
      if bytes.size != expectedBytes then
        throw <| IO.userError <|
          s!"ParamIO: float32 payload size mismatch for {Shape.pretty shape} "
            ++ s!"(got={bytes.size}, expected={expectedBytes})"
      handle.write bytes
      writeParamListFloat32 handle params

/--
Write a runtime parameter list as a streamed float32 checkpoint.

Only one parameter payload is resident on the host at a time. This is the appropriate format for
large CUDA models; it records the exact values used by the float32 runtime instead of expanding
them into one in-memory JSON tree.
-/
def writeModuleParamFloat32
    (path : System.FilePath) {shapes : List Shape}
    (params : Torch.ParamList Float shapes) : IO Unit := do
  Torch.Internal.CheckpointIO.writeAtomically path fun handle => do
    Torch.Internal.CheckpointIO.writeFormat float32StreamFormat handle
    Torch.Internal.CheckpointIO.writeNat64 "ParamIO" handle shapes.length
    writeParamListFloat32 handle params

/-- Check whether a file begins with the streaming float32 checkpoint header. -/
def isModuleParamFloat32 (path : System.FilePath) : IO Bool := do
  Torch.Internal.CheckpointIO.hasFormatMagic float32StreamFormat path

/-- Read one float32 payload directly into an existing runtime parameter. -/
def readParamFloat32Into
    (useCuda : Bool) (handle : IO.FS.Handle) {shape : Shape}
    (param : Torch.Param Float shape) : IO Unit := do
  readShape handle shape
  let bytes ← Torch.Internal.CheckpointIO.readExact
    "ParamIO" handle (Shape.size shape * 4)
  if useCuda then
    let buffer ← Runtime.Autograd.Cuda.Buffer.ofFloat32BytesIO bytes
    Torch.Internal.setParamCudaValue param { s := shape, buf := buffer }
  else
    let values := Runtime.Autograd.Cuda.Buffer.float32BytesToFloatArray bytes
    let tensor := Runtime.Autograd.Cuda.Convert.unflattenFloatUnsafe (s := shape) values
    Torch.Internal.setParamHostValue param tensor

/-- Stream a checkpoint into a shape-indexed runtime parameter list. -/
def readParamListFloat32Into
    (useCuda : Bool) (handle : IO.FS.Handle) :
    {shapes : List Shape} → Torch.ParamList Float shapes → IO Unit
  | [], .nil => pure ()
  | _ :: _, .cons param params => do
      readParamFloat32Into useCuda handle param
      readParamListFloat32Into useCuda handle params

/-- Validate every tensor record and payload without changing the destination parameters. -/
def validateParamListFloat32 (handle : IO.FS.Handle) : (shapes : List Shape) → IO Unit
  | [] => pure ()
  | shape :: shapes => do
      readShape handle shape
      let _ ← Torch.Internal.CheckpointIO.readExact
        "ParamIO" handle (Shape.size shape * 4)
      validateParamListFloat32 handle shapes

/-- Read and validate the common header of a streamed float32 checkpoint. -/
def readAndCheckFloat32Header
    (handle : IO.FS.Handle) (expectedParameterCount : Nat) : IO Unit := do
  Torch.Internal.CheckpointIO.readFormat float32StreamFormat handle
  let parameterCount ← Torch.Internal.CheckpointIO.readNat64 "ParamIO" handle
  if parameterCount != expectedParameterCount then
    throw <| IO.userError <|
      s!"ParamIO: checkpoint parameter count mismatch "
        ++ s!"(file={parameterCount}, expected={expectedParameterCount})"

/-- Reject extra bytes after the last expected tensor payload. -/
def checkFloat32EndOfFile (handle : IO.FS.Handle) : IO Unit := do
  let trailing ← handle.read 1
  if trailing.size != 0 then
    throw <| IO.userError "ParamIO: trailing bytes after final parameter"

/-- Load a streamed float32 checkpoint into an existing runtime parameter list. -/
def readModuleParamFloat32Into
    (path : System.FilePath) (useCuda : Bool) {shapes : List Shape}
    (params : Torch.ParamList Float shapes) : IO Unit := do
  -- Check the complete stream first. Under ordinary file ownership this prevents a malformed or
  -- truncated checkpoint from leaving the live module with only a prefix of its parameters
  -- replaced, while retaining one-tensor-at-a-time memory use for large CUDA models.
  IO.FS.withFile path IO.FS.Mode.read fun handle => do
    readAndCheckFloat32Header handle shapes.length
    validateParamListFloat32 handle shapes
    checkFloat32EndOfFile handle
  IO.FS.withFile path IO.FS.Mode.read fun handle => do
    readAndCheckFloat32Header handle shapes.length
    readParamListFloat32Into useCuda handle params
    checkFloat32EndOfFile handle

end ParamIO
end TorchLean
end Autograd
end Runtime
