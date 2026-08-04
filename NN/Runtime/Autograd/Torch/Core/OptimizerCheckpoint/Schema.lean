/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.CheckpointIO
public import NN.Spec.Core.Shape

/-!
# Optimizer Checkpoint Schemas

An optimizer checkpoint is meaningful only for the parameter layout that produced it. This module
stores that shared layout: parameter shapes in module order and the corresponding trainability
flags. Optimizer-specific codecs validate this schema before installing backend state.
-/

@[expose] public section

namespace Runtime.Autograd.Torch.Internal.OptimizerCheckpoint

/-- Ordered parameter metadata shared by backend-owned optimizer checkpoints. -/
structure ParameterSchema where
  /-- Parameter shapes in module order. -/
  shapes : List Spec.Shape
  /-- Whether each corresponding parameter is trainable. -/
  requiresGrad : List Bool

namespace ParameterSchema

/-- Whether the shape list and mutability mask describe the same number of parameters. -/
def isWellFormed (schema : ParameterSchema) : Bool :=
  schema.shapes.length == schema.requiresGrad.length

/-- Number of parameters for which an optimizer state entry is expected. -/
def trainableCount (schema : ParameterSchema) : Nat :=
  schema.requiresGrad.count true

/-- Write the ordered parameter schema for an optimizer-specific checkpoint format. -/
def write
    (format : CheckpointIO.Format) (handle : IO.FS.Handle) (schema : ParameterSchema) : IO Unit := do
  unless schema.isWellFormed do
    throw <| IO.userError s!"{format.name}: malformed in-memory parameter schema"
  CheckpointIO.writeNat64 format.name handle schema.shapes.length
  for (shape, requiresGrad) in List.zip schema.shapes schema.requiresGrad do
    let dims := Spec.Shape.toList shape
    CheckpointIO.writeNat64 format.name handle dims.length
    for dim in dims do
      CheckpointIO.writeNat64 format.name handle dim
    CheckpointIO.writeNat64 format.name handle (Spec.Shape.size shape)
    CheckpointIO.writeNat64 format.name handle (if requiresGrad then 1 else 0)

/-- Read a parameter schema and reject any difference from the expected module layout. -/
def readAndCheck
    (format : CheckpointIO.Format) (handle : IO.FS.Handle) (expected : ParameterSchema) : IO Unit := do
  unless expected.isWellFormed do
    throw <| IO.userError s!"{format.name}: malformed expected parameter schema"
  let parameterCount ← CheckpointIO.readNat64 format.name handle
  if parameterCount != expected.shapes.length then
    throw <| IO.userError <|
      s!"{format.name}: parameter count mismatch " ++
        s!"(file={parameterCount}, expected={expected.shapes.length})"
  for index in [0:parameterCount] do
    let shape ← match expected.shapes[index]? with
      | some shape => pure shape
      | none => throw <| IO.userError s!"{format.name}: internal shape index error"
    let expectedRequiresGrad ← match expected.requiresGrad[index]? with
      | some flag => pure flag
      | none => throw <| IO.userError s!"{format.name}: internal mutability index error"
    let rank ← CheckpointIO.readNat64 format.name handle
    let expectedDims := Spec.Shape.toList shape
    if rank != expectedDims.length then
      throw <| IO.userError <|
        s!"{format.name}: rank mismatch for parameter {index} " ++
          s!"(file={rank}, expected={expectedDims.length})"
    let mut reversedDims := []
    for _ in [0:rank] do
      reversedDims := (← CheckpointIO.readNat64 format.name handle) :: reversedDims
    let dims := reversedDims.reverse
    if dims != expectedDims then
      throw <| IO.userError <|
        s!"{format.name}: shape mismatch for parameter {index} " ++
          s!"(file={dims}, expected={expectedDims})"
    let count ← CheckpointIO.readNat64 format.name handle
    if count != Spec.Shape.size shape then
      throw <| IO.userError <|
        s!"{format.name}: element count mismatch for parameter {index} " ++
          s!"(file={count}, expected={Spec.Shape.size shape})"
    let requiresGrad ← match ← CheckpointIO.readNat64 format.name handle with
      | 0 => pure false
      | 1 => pure true
      | flag => throw <| IO.userError s!"{format.name}: invalid `requiresGrad` flag {flag}"
    if requiresGrad != expectedRequiresGrad then
      throw <| IO.userError <|
        s!"{format.name}: `requiresGrad` mismatch for parameter {index}"

end ParameterSchema
end Runtime.Autograd.Torch.Internal.OptimizerCheckpoint
