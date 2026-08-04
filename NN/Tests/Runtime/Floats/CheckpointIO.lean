/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.OptimizerCheckpoint
public import NN.Runtime.Autograd.TorchLean.ParamIO

/-!
# Checkpoint IO Tests

Regression checks for the native checkpoint boundary: explicit little-endian float32 bytes,
optimizer-state round trips, and rejection of incompatible or truncated files.
-/

@[expose] public section

namespace Tests
namespace Floats
namespace CheckpointIO

open Runtime.Autograd.Torch.Internal.EagerSession

def expectRejection {α : Type} (message : String) (action : IO α) : IO Unit := do
  let mut rejected := false
  try
    let _ ← action
  catch _ =>
    rejected := true
  unless rejected do
    throw <| IO.userError message

def temporaryCheckpoint : IO System.FilePath := do
  let timestamp ← IO.monoNanosNow
  pure <| System.FilePath.mk s!".lake/build/tmp/checkpoint_io_{timestamp}.bin"

def removeIfPresent (path : System.FilePath) : IO Unit := do
  try
    IO.FS.removeFile path
  catch _ =>
    pure ()

/-- Replace the explicit version field while retaining a checkpoint's family and payload. -/
def withFormatVersion
    (format : Runtime.Autograd.Torch.Internal.CheckpointIO.Format)
    (version : Nat) (bytes : ByteArray) : ByteArray :=
  format.magic ++
    Runtime.Autograd.Torch.Internal.CheckpointIO.nat64LE version ++
    bytes.extract (format.magic.size + 8) bytes.size

/-- Atomic replacement must overwrite a complete file and preserve it if the writer fails. -/
def checkAtomicReplacement : IO Unit := do
  let path ← temporaryCheckpoint
  try
    IO.FS.writeBinFile path "old checkpoint".toUTF8
    Runtime.Autograd.Torch.Internal.CheckpointIO.writeAtomically path fun handle =>
      handle.write "new checkpoint".toUTF8
    unless (← IO.FS.readBinFile path) = "new checkpoint".toUTF8 do
      throw <| IO.userError "atomic checkpoint replacement did not install the new file"
    expectRejection "atomic checkpoint writer failure was not reported" <|
      Runtime.Autograd.Torch.Internal.CheckpointIO.writeAtomically path fun handle => do
        handle.write "partial checkpoint".toUTF8
        throw <| IO.userError "injected checkpoint writer failure"
    unless (← IO.FS.readBinFile path) = "new checkpoint".toUTF8 do
      throw <| IO.userError "failed atomic checkpoint replacement changed the destination"
  finally
    removeIfPresent path

/-- Check that the native bridge uses the documented little-endian float32 representation. -/
def checkLittleEndianFloat32 : IO Unit := do
  let values : FloatArray := FloatArray.mk #[1.0, -2.5]
  let expected : ByteArray :=
    ByteArray.mk #[0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x20, 0xc0]
  let encoded := Runtime.Autograd.Cuda.Buffer.floatArrayToFloat32Bytes values
  if encoded != expected then
    throw <| IO.userError s!"float32 byte encoding mismatch: got {encoded.data}"
  let decoded := Runtime.Autograd.Cuda.Buffer.float32BytesToFloatArray expected
  if decoded.size != 2 || decoded[0]! != 1.0 || decoded[1]! != -2.5 then
    throw <| IO.userError "float32 byte decoding did not recover the source values"

/-- The JSON parameter format must reject records beyond the expected shape list. -/
def checkJsonParameterCount : IO Unit := do
  let tensor : Spec.Tensor Float (.dim 1 .scalar) :=
    Spec.Tensor.dim fun _ => Spec.Tensor.scalar 1.0
  let record := Runtime.Autograd.TorchLean.ParamIO.tensorToJsonBits (.dim 1 .scalar) tensor
  let payload := Lean.Json.arr #[record, record]
  match Runtime.Autograd.TorchLean.ParamIO.tListFromJsonBits
      (tag := "checkpoint test") (ss := [.dim 1 .scalar]) payload with
  | .error _ => pure ()
  | .ok _ => throw <| IO.userError "JSON checkpoint accepted a trailing parameter record"

/-- Round-trip one Adam state and reject incompatible metadata and truncated payloads. -/
def checkCudaAdamCheckpoint : IO Unit := do
  let path ← temporaryCheckpoint
  let truncatedPath := System.FilePath.mk s!"{path}.truncated"
  let wrongVersionPath := System.FilePath.mk s!"{path}.wrong-version"
  let mValues : FloatArray := FloatArray.mk #[0.25, -0.5]
  let vValues : FloatArray := FloatArray.mk #[1.5, 2.0]
  let m ← Runtime.Autograd.Cuda.Buffer.ofFloatArrayIO mValues
  let v ← Runtime.Autograd.Cuda.Buffer.ofFloatArrayIO vValues
  let sourceState : CudaAdamState :=
    Std.HashMap.emptyWithCapacity |>.insert 0 { m, v, t := 7 }
  let sourceStateRef ← IO.mkRef sourceState
  let config : CudaAdamConfig :=
    { kind := .adam, beta1 := 0.9, beta2 := 0.999, epsilon := 1.0e-8, weightDecay := 0.0 }
  let sourceConfigRef ← IO.mkRef (some config)
  let schema : Runtime.Autograd.Torch.Internal.OptimizerCheckpoint.ParameterSchema :=
    { shapes := [.dim 2 .scalar], requiresGrad := [true] }
  let loadedStateRef ← IO.mkRef (Std.HashMap.emptyWithCapacity : CudaAdamState)
  let loadedConfigRef ← IO.mkRef (none : Option CudaAdamConfig)
  let incompatibleConfig := { config with beta1 := 0.8 }
  let incompatibleConfigRef ← IO.mkRef (some incompatibleConfig)
  let incompatibleStateRef ← IO.mkRef (Std.HashMap.emptyWithCapacity : CudaAdamState)
  try
    writeCudaAdamStateFloat32 path schema sourceConfigRef sourceStateRef
    readCudaAdamStateFloat32 path schema loadedConfigRef loadedStateRef
    let loadedConfig ← match ← loadedConfigRef.get with
      | some loaded => pure loaded
      | none => throw <| IO.userError "optimizer checkpoint lost its configuration"
    unless loadedConfig.sameBits config do
      throw <| IO.userError "optimizer checkpoint changed its configuration"
    let loadedState ← loadedStateRef.get
    let loaded ← match loadedState.get? 0 with
      | some entry => pure entry
      | none => throw <| IO.userError "optimizer checkpoint lost parameter state 0"
    if loaded.t != 7 then
      throw <| IO.userError s!"optimizer checkpoint changed the step counter to {loaded.t}"
    let loadedM ← Runtime.Autograd.Cuda.Buffer.toFloatArrayIO loaded.m
    let loadedV ← Runtime.Autograd.Cuda.Buffer.toFloatArrayIO loaded.v
    if loadedM.size != 2 || loadedM[0]! != 0.25 || loadedM[1]! != -0.5 ||
        loadedV.size != 2 || loadedV[0]! != 1.5 || loadedV[1]! != 2.0 then
      throw <| IO.userError "optimizer checkpoint changed moment values"
    expectRejection "optimizer checkpoint accepted a different parameter shape" <|
      readCudaAdamStateFloat32 path
        { shapes := [.dim 3 .scalar], requiresGrad := [true] }
        loadedConfigRef loadedStateRef
    let bytes ← IO.FS.readBinFile path
    IO.FS.writeBinFile truncatedPath (bytes.extract 0 (bytes.size - 1))
    expectRejection "optimizer checkpoint accepted a truncated payload" <|
      readCudaAdamStateFloat32 truncatedPath schema loadedConfigRef loadedStateRef
    IO.FS.writeBinFile wrongVersionPath <|
      withFormatVersion cudaAdamCheckpointFormat
        (cudaAdamCheckpointFormat.version + 1) bytes
    expectRejection "optimizer checkpoint accepted an unsupported format version" <|
      readCudaAdamStateFloat32 wrongVersionPath schema loadedConfigRef loadedStateRef
    expectRejection "optimizer checkpoint replaced a bound, incompatible configuration" <|
      readCudaAdamStateFloat32 path schema incompatibleConfigRef incompatibleStateRef
    unless (← incompatibleStateRef.get).isEmpty do
      throw <| IO.userError "rejected optimizer checkpoint installed replacement state"
    let retainedConfig ← match ← incompatibleConfigRef.get with
      | some retained => pure retained
      | none => throw <| IO.userError "rejected optimizer checkpoint cleared its configuration"
    unless retainedConfig.sameBits incompatibleConfig do
      throw <| IO.userError "rejected optimizer checkpoint changed its configuration"
    expectRejection "optimizer state accepted different hyperparameters" <|
      ensureCudaAdamConfig loadedConfigRef incompatibleConfig
    let invalidConfigRef ← IO.mkRef (none : Option CudaAdamConfig)
    expectRejection "optimizer state accepted beta1 = 1" <|
      ensureCudaAdamConfig invalidConfigRef { config with beta1 := 1.0 }
    unless (← invalidConfigRef.get).isNone do
      throw <| IO.userError "invalid optimizer configuration changed the live configuration"
  finally
    releaseCudaAdamState (← sourceStateRef.get)
    releaseCudaAdamState (← loadedStateRef.get)
    releaseCudaAdamState (← incompatibleStateRef.get)
    removeIfPresent path
    removeIfPresent truncatedPath
    removeIfPresent wrongVersionPath

/--
Round-trip a streamed module checkpoint and check that preflight rejection leaves parameters
unchanged.
-/
def checkModuleFloat32Checkpoint : IO Unit := do
  let path ← temporaryCheckpoint
  let truncatedPath := System.FilePath.mk s!"{path}.truncated"
  let wrongVersionPath := System.FilePath.mk s!"{path}.wrong-version"
  let sourceTensor : Spec.Tensor Float (.dim 2 .scalar) :=
    Spec.Tensor.dim fun i => Spec.Tensor.scalar (if i.val = 0 then 1.0 else -2.5)
  let zeroTensor : Spec.Tensor Float (.dim 2 .scalar) :=
    Spec.Tensor.dim fun _ => Spec.Tensor.scalar 0.0
  let sourceValues : Proofs.Autograd.Algebra.TList Float [.dim 2 .scalar] :=
    .cons sourceTensor .nil
  let targetValues : Proofs.Autograd.Algebra.TList Float [.dim 2 .scalar] :=
    .cons zeroTensor .nil
  let sourceParams ← Runtime.Autograd.Torch.ParamList.ofTList sourceValues
  let targetParams ← Runtime.Autograd.Torch.ParamList.ofTList targetValues
  try
    Runtime.Autograd.TorchLean.ParamIO.writeModuleParamFloat32 path sourceParams
    let bytes ← IO.FS.readBinFile path
    IO.FS.writeBinFile truncatedPath (bytes.extract 0 (bytes.size - 1))
    expectRejection "module checkpoint accepted a truncated payload" <|
      Runtime.Autograd.TorchLean.ParamIO.readModuleParamFloat32Into
        truncatedPath false targetParams
    IO.FS.writeBinFile wrongVersionPath <|
      withFormatVersion Runtime.Autograd.TorchLean.ParamIO.float32StreamFormat
        (Runtime.Autograd.TorchLean.ParamIO.float32StreamFormat.version + 1) bytes
    expectRejection "module checkpoint accepted an unsupported format version" <|
      Runtime.Autograd.TorchLean.ParamIO.readModuleParamFloat32Into
        wrongVersionPath false targetParams
    let retainedValues : Proofs.Autograd.Algebra.TList Float [.dim 2 .scalar] ←
      Runtime.Autograd.Torch.ParamList.values targetParams
    match retainedValues with
    | .cons retained .nil =>
        let values := Runtime.Autograd.Cuda.Convert.flattenFloat retained
        if values.size != 2 || values[0]! != 0.0 || values[1]! != 0.0 then
          throw <| IO.userError "rejected module checkpoint changed live parameters"
    Runtime.Autograd.TorchLean.ParamIO.readModuleParamFloat32Into path false targetParams
    let loadedValues : Proofs.Autograd.Algebra.TList Float [.dim 2 .scalar] ←
      Runtime.Autograd.Torch.ParamList.values targetParams
    match loadedValues with
    | .cons loaded .nil =>
        let values := Runtime.Autograd.Cuda.Convert.flattenFloat loaded
        if values.size != 2 || values[0]! != 1.0 || values[1]! != -2.5 then
          throw <| IO.userError "module checkpoint did not recover the source parameters"
  finally
    removeIfPresent path
    removeIfPresent truncatedPath
    removeIfPresent wrongVersionPath

/-- Run checkpoint boundary regressions. -/
def run : IO Unit := do
  checkLittleEndianFloat32
  checkAtomicReplacement
  checkJsonParameterCount
  checkCudaAdamCheckpoint
  checkModuleFloat32Checkpoint

end CheckpointIO
end Floats
end Tests
