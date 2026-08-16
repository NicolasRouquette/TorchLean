/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.StateIO
public import NN.Runtime.Autograd.TorchLean.Module
public import NN.API.Neural

/-!
# Runtime Checkpoints

TorchLean examples often want the same simple workflow:

1. train a model for a few steps,
2. save its state, and
3. restore that state before inference or further training.

The implementation delegates binary encoding and device transfers to
`Runtime.Autograd.TorchLean.StateIO`, while this module provides the checked runtime API used by
trainers and data loaders.

## What Is Supported

Both checkpoint formats preserve the model's shape-indexed state layout:

- Native `Float32` modules use exact little-endian binary32 payloads on CPU and CUDA.
- Compatibility `Float` modules use exact binary64 JSON on CPU and binary32 payloads on CUDA.
- Streamed checkpoints write one tensor at a time instead of materializing a second host copy of a
  large device-resident model.

This is enough to checkpoint any TorchLean runtime model implemented as a
`TorchLean.Module.Objective` over native `Float32` or compatibility `Float`, independent of
architecture.
-/

@[expose] public section

namespace TorchLean
namespace Checkpoint

open Spec

/--
Write a model-state pack to a JSON bits checkpoint.

Use this when the model's shape-indexed parameters and buffers are already available and only need
to be persisted.
-/
def saveStateBits
    {stateShapes : List Spec.Shape}
    (path : System.FilePath)
    (state : _root_.Runtime.Autograd.Torch.TList Float stateShapes)
    (pretty : Bool := true) : IO Unit := do
  _root_.Runtime.Autograd.TorchLean.StateIO.writeStateBits (ss := stateShapes) path state pretty

/-- Scalar-dependent persistence for a shape-indexed runtime parameter list. -/
class CheckpointScalar (α : Type) where
  /-- Write every parameter and persistent buffer in order. -/
  write : {shapes : List Spec.Shape} →
    Bool → System.FilePath → _root_.Runtime.Autograd.Torch.ParamList α shapes → IO Unit
  /-- Validate and restore every parameter and persistent buffer in order. -/
  read : {shapes : List Spec.Shape} →
    Bool → System.FilePath → _root_.Runtime.Autograd.Torch.ParamList α shapes → IO Unit

/-- Compatibility `Float` checkpoints keep exact binary64 values on CPU and binary32 values on CUDA. -/
instance : CheckpointScalar Float where
  write := fun {shapes} useCuda path state => do
    if useCuda then
      _root_.Runtime.Autograd.TorchLean.StateIO.writeModuleStateFloat32
        Float.toFloat32 (shapes := shapes) path state
    else
      let values ← _root_.Runtime.Autograd.Torch.ParamList.valuesSynced (α := Float)
        (ss := shapes) state
      _root_.Runtime.Autograd.TorchLean.StateIO.writeStateBits (ss := shapes) path values
  read := fun {shapes} useCuda path state => do
    if ← _root_.Runtime.Autograd.TorchLean.StateIO.isModuleStateFloat32 path then
      _root_.Runtime.Autograd.TorchLean.StateIO.readModuleStateFloat32Into
        Float32.toFloat path useCuda (shapes := shapes) state
    else
      let result ← _root_.Runtime.Autograd.TorchLean.StateIO.readStateBits (ss := shapes) path
      match result with
      | .error message => throw <| IO.userError s!"Checkpoint: load failed for {path}: {message}"
      | .ok values =>
          _root_.Runtime.Autograd.Torch.ParamList.setValues (α := Float) (ss := shapes) state values

/-- Native `Float32` checkpoints preserve exact binary32 payloads on both CPU and CUDA. -/
instance : CheckpointScalar Float32 where
  write := fun {shapes} _ path state =>
    _root_.Runtime.Autograd.TorchLean.StateIO.writeModuleStateFloat32
      id (shapes := shapes) path state
  read := fun {shapes} useCuda path state => do
    unless ← _root_.Runtime.Autograd.TorchLean.StateIO.isModuleStateFloat32 path do
      throw <| IO.userError
        s!"Checkpoint: {path} is not a native Float32 module checkpoint"
    _root_.Runtime.Autograd.TorchLean.StateIO.readModuleStateFloat32Into
      id path useCuda (shapes := shapes) state

/--
Save the current values of a TorchLean runtime module.

The scalar type selects the checkpoint encoding through `CheckpointScalar`. Native `Float32`
modules preserve exact binary32 payloads on CPU and CUDA. Compatibility `Float` modules preserve
binary64 values on CPU and the runtime's binary32 values on CUDA.
-/
def saveModule {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [CheckpointScalar α]
    {stateShapes inputShapes natInputShapes : List Spec.Shape}
    (m : _root_.Runtime.Autograd.TorchLean.Module.Objective α stateShapes inputShapes
      natInputShapes)
    (path : System.FilePath) : IO Unit :=
  CheckpointScalar.write m.opts.usesCuda path m.trainer.state

/--
Load a scalar-compatible checkpoint into a module.

The reader checks the format header, state-tensor count, every shape, and every payload length before
accepting the checkpoint.
-/
def loadModule {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [CheckpointScalar α]
    {stateShapes inputShapes natInputShapes : List Spec.Shape}
    (m : _root_.Runtime.Autograd.TorchLean.Module.Objective α stateShapes inputShapes
      natInputShapes)
    (path : System.FilePath) : IO Unit :=
  CheckpointScalar.read m.opts.usesCuda path m.trainer.state

/--
Save optimizer state retained by the module's runtime backend.

This currently applies to eager CUDA Adam and AdamW, whose moment buffers live on the device. The
operation fails explicitly when the selected trainer has no backend-owned optimizer state instead
of writing an incomplete resume checkpoint.
-/
def saveOptimizerState
    {stateShapes inputShapes natInputShapes : List Spec.Shape}
    (m : _root_.Runtime.Autograd.TorchLean.Module.Objective Float stateShapes inputShapes
      natInputShapes)
    (path : System.FilePath) : IO Unit := do
  match m.trainer.optimizerStateCheckpoint? with
  | some checkpoint => checkpoint.save path
  | none =>
      throw <| IO.userError
        "Checkpoint: selected trainer does not expose backend-owned optimizer state"

/-- Restore optimizer state retained by the module's runtime backend. -/
def loadOptimizerState
    {stateShapes inputShapes natInputShapes : List Spec.Shape}
    (m : _root_.Runtime.Autograd.TorchLean.Module.Objective Float stateShapes inputShapes
      natInputShapes)
    (path : System.FilePath) : IO Unit := do
  match m.trainer.optimizerStateCheckpoint? with
  | some checkpoint => checkpoint.load path
  | none =>
      throw <| IO.userError
        "Checkpoint: selected trainer does not expose backend-owned optimizer state"

/--
Load a JSON bits checkpoint as model state without mutating a module.

This is useful when you want to run typed graph inference directly and never instantiate a trainer.
-/
def loadStateBits
    {stateShapes : List Spec.Shape}
    (path : System.FilePath) : IO (_root_.Runtime.Autograd.Torch.TList Float stateShapes) := do
  let stateResult ← _root_.Runtime.Autograd.TorchLean.StateIO.readStateBits
    (ss := stateShapes) path
  match stateResult with
  | Except.error e =>
      throw <| IO.userError s!"Checkpoint: load failed for {path}: {e}"
  | Except.ok ps =>
      pure ps

/--
Read a JSON bits checkpoint, returning an error string instead of throwing an exception.

This is useful in batch tools or CI-style runs where you want to keep going and report failures.
-/
def readStateBits
    {stateShapes : List Spec.Shape}
    (path : System.FilePath) :
    IO (Except String (_root_.Runtime.Autograd.Torch.TList Float stateShapes)) :=
  _root_.Runtime.Autograd.TorchLean.StateIO.readStateBits (ss := stateShapes) path

/-- Load a bits checkpoint whose state layout is determined by `model`. -/
def loadModelState {σ τ : Shape}
    (model : nn.Sequential σ τ) (path : System.FilePath) :
    IO (TensorPack Float (nn.stateShapes model)) :=
  loadStateBits (stateShapes := nn.stateShapes model) path

/-- Allocate runtime modules from checked model state. -/
def toRuntimeState {stateShapes : List Shape}
    (state : TensorPack Float stateShapes) :
    IO (_root_.Runtime.Autograd.Torch.ParamList Float stateShapes) :=
  _root_.Runtime.Autograd.Torch.ParamList.ofTList (α := Float) (ss := stateShapes) state

/-- Restore a checkpoint into a runtime module attached to `model`. -/
def restoreModule {α : Type} [_root_.Context α] [DecidableEq Shape] [CheckpointScalar α]
    {σ τ : Shape}
    (model : nn.Sequential σ τ)
    (m : _root_.Runtime.Autograd.TorchLean.Module.Objective α
      (nn.stateShapes model) [σ, τ])
    (path : System.FilePath) : IO Unit :=
  loadModule (stateShapes := nn.stateShapes model) (inputShapes := [σ, τ]) m path

/-- Restore a model checkpoint when the caller supplied a path. -/
def restoreModuleIfSome {α : Type} [_root_.Context α] [DecidableEq Shape] [CheckpointScalar α]
    {σ τ : Shape}
    (model : nn.Sequential σ τ)
    (m : _root_.Runtime.Autograd.TorchLean.Module.Objective α
      (nn.stateShapes model) [σ, τ])
    (path? : Option System.FilePath) : IO Unit :=
  match path? with
  | none => pure ()
  | some path => restoreModule model m path

/-- Save a model's runtime state when the caller supplied a path. -/
def saveModuleIfSome {α : Type} [_root_.Context α] [DecidableEq Shape] [CheckpointScalar α]
    {σ τ : Shape}
    (model : nn.Sequential σ τ)
    (m : _root_.Runtime.Autograd.TorchLean.Module.Objective α
      (nn.stateShapes model) [σ, τ])
    (path? : Option System.FilePath) : IO Unit := do
  match path? with
  | none => pure ()
  | some path =>
      saveModule (stateShapes := nn.stateShapes model) (inputShapes := [σ, τ]) m path
      IO.println s!"  wrote checkpoint: {path}"

end Checkpoint
end TorchLean
