/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.ParamIO
public import NN.Runtime.Autograd.TorchLean.Module
public import NN.API.Neural

/-!
# Runtime Checkpoints

TorchLean examples often want the same simple workflow:

1. train a model for a few steps,
2. save its parameters, and
3. reload those parameters later to sample / run inference.

The implementation delegates binary encoding and device transfers to
`Runtime.Autograd.TorchLean.ParamIO`, while this module provides the checked runtime API used by
trainers and data loaders.

## What Is Supported

Both checkpoint paths preserve the model's shape-indexed parameter layout:

- CPU modules use exact `Float.toBits` (`UInt64`) values stored in JSON.
- CUDA modules stream the float32 values used by the runtime, four bytes per scalar, without first
  materializing the full model in host tensors.

This is enough to checkpoint *any* TorchLean runtime model implemented as a
`TorchLean.Module.ScalarModule` over `Float`, independent of architecture.
-/

@[expose] public section

namespace TorchLean
namespace Checkpoint

open Spec

/--
Write a parameter pack to a JSON bits checkpoint.

Use this when the model's shape-indexed parameter tensors are already available and only need to
be persisted.
-/
def saveParamBits
    {paramShapes : List Spec.Shape}
    (path : System.FilePath)
    (ps : _root_.Runtime.Autograd.Torch.TList Float paramShapes)
    (pretty : Bool := true) : IO Unit := do
  _root_.Runtime.Autograd.TorchLean.ParamIO.writeParamBits (ss := paramShapes) path ps pretty

/--
Save the current values of a TorchLean runtime module.

CPU modules use the exact `Float.toBits` JSON format. CUDA modules use the streamed float32 format
so large models remain on the device while their parameters are written. This is
architecture-agnostic: it works for any `ScalarModule Float …`.
-/
def saveModuleParams
    {paramShapes inputShapes natInputShapes : List Spec.Shape}
    (m : _root_.Runtime.Autograd.TorchLean.Module.ScalarModule Float paramShapes inputShapes
      natInputShapes)
    (path : System.FilePath) : IO Unit := do
  if m.opts.usesCuda then
    _root_.Runtime.Autograd.TorchLean.ParamIO.writeModuleParamFloat32
      (shapes := paramShapes) path m.trainer.params
  else
    let ps ← _root_.Runtime.Autograd.Torch.ParamList.valuesSynced (α := Float) (ss := paramShapes)
      m.trainer.params
    _root_.Runtime.Autograd.TorchLean.ParamIO.writeParamBits (ss := paramShapes) path ps

/--
Load a JSON-bits or streamed-float32 checkpoint into a module.

The format is detected from its header. Both readers check the parameter count, every tensor shape,
and every payload length before accepting the checkpoint.
-/
def loadModuleParams
    {paramShapes inputShapes natInputShapes : List Spec.Shape}
    (m : _root_.Runtime.Autograd.TorchLean.Module.ScalarModule Float paramShapes inputShapes
      natInputShapes)
    (path : System.FilePath) : IO Unit := do
  if ← _root_.Runtime.Autograd.TorchLean.ParamIO.isModuleParamFloat32 path then
    _root_.Runtime.Autograd.TorchLean.ParamIO.readModuleParamFloat32Into
      path m.opts.usesCuda m.trainer.params
  else
    let psRes ← _root_.Runtime.Autograd.TorchLean.ParamIO.readParamBits (ss := paramShapes) path
    match psRes with
    | Except.error e =>
        throw <| IO.userError s!"Checkpoint: load failed for {path}: {e}"
    | Except.ok ps =>
        _root_.Runtime.Autograd.Torch.ParamList.setValues (α := Float) (ss := paramShapes)
          m.trainer.params ps

/--
Save optimizer state retained by the module's runtime backend.

This currently applies to eager CUDA Adam and AdamW, whose moment buffers live on the device. The
operation fails explicitly when the selected trainer has no backend-owned optimizer state instead
of writing an incomplete resume checkpoint.
-/
def saveOptimizerState
    {paramShapes inputShapes natInputShapes : List Spec.Shape}
    (m : _root_.Runtime.Autograd.TorchLean.Module.ScalarModule Float paramShapes inputShapes
      natInputShapes)
    (path : System.FilePath) : IO Unit := do
  match m.trainer.optimizerStateCheckpoint? with
  | some checkpoint => checkpoint.save path
  | none =>
      throw <| IO.userError
        "Checkpoint: selected trainer does not expose backend-owned optimizer state"

/-- Restore optimizer state retained by the module's runtime backend. -/
def loadOptimizerState
    {paramShapes inputShapes natInputShapes : List Spec.Shape}
    (m : _root_.Runtime.Autograd.TorchLean.Module.ScalarModule Float paramShapes inputShapes
      natInputShapes)
    (path : System.FilePath) : IO Unit := do
  match m.trainer.optimizerStateCheckpoint? with
  | some checkpoint => checkpoint.load path
  | none =>
      throw <| IO.userError
        "Checkpoint: selected trainer does not expose backend-owned optimizer state"

/--
Load a JSON bits checkpoint as a parameter list (without mutating a module).

This is useful when you want to run compiled inference directly and never instantiate a trainer.
-/
def loadParamBits
    {paramShapes : List Spec.Shape}
    (path : System.FilePath) : IO (_root_.Runtime.Autograd.Torch.TList Float paramShapes) := do
  let psRes ← _root_.Runtime.Autograd.TorchLean.ParamIO.readParamBits (ss := paramShapes) path
  match psRes with
  | Except.error e =>
      throw <| IO.userError s!"Checkpoint: load failed for {path}: {e}"
  | Except.ok ps =>
      pure ps

/--
Read a JSON bits checkpoint, returning an error string instead of throwing an exception.

This is useful in batch tools or CI-style runs where you want to keep going and report failures.
-/
def readParamBits
    {paramShapes : List Spec.Shape}
    (path : System.FilePath) : IO (Except String (_root_.Runtime.Autograd.Torch.TList Float paramShapes)) :=
  _root_.Runtime.Autograd.TorchLean.ParamIO.readParamBits (ss := paramShapes) path

/-- Load a bits checkpoint whose parameter layout is determined by `model`. -/
def loadModelParamBits {σ τ : Shape}
    (model : nn.Sequential σ τ) (path : System.FilePath) :
    IO (TensorPack Float (nn.paramShapes model)) :=
  loadParamBits (paramShapes := nn.paramShapes model) path

/-- Allocate runtime parameter handles from a checked parameter pack. -/
def toRuntimeParams {paramShapes : List Shape}
    (ps : TensorPack Float paramShapes) :
    IO (_root_.Runtime.Autograd.Torch.ParamList Float paramShapes) :=
  _root_.Runtime.Autograd.Torch.ParamList.ofTList (α := Float) (ss := paramShapes) ps

/-- Load a checkpoint into a runtime module attached to `model`. -/
def loadModelIntoModule {σ τ : Shape}
    (model : nn.Sequential σ τ)
    (m : _root_.Runtime.Autograd.TorchLean.Module.ScalarModule Float
      (nn.paramShapes model) [σ, τ])
    (path : System.FilePath) : IO Unit :=
  loadModuleParams (paramShapes := nn.paramShapes model) (inputShapes := [σ, τ]) m path

/-- Load a model checkpoint when the caller supplied a path. -/
def loadModelIntoModuleIfSome {σ τ : Shape}
    (model : nn.Sequential σ τ)
    (m : _root_.Runtime.Autograd.TorchLean.Module.ScalarModule Float
      (nn.paramShapes model) [σ, τ])
    (path? : Option System.FilePath) : IO Unit :=
  match path? with
  | none => pure ()
  | some path => loadModelIntoModule model m path

/-- Save a model's runtime parameters when the caller supplied a path. -/
def saveModelIntoPathIfSome {σ τ : Shape}
    (model : nn.Sequential σ τ)
    (m : _root_.Runtime.Autograd.TorchLean.Module.ScalarModule Float
      (nn.paramShapes model) [σ, τ])
    (path? : Option System.FilePath) : IO Unit := do
  match path? with
  | none => pure ()
  | some path =>
      saveModuleParams (paramShapes := nn.paramShapes model) (inputShapes := [σ, τ]) m path
      IO.println s!"  wrote params: {path}"

end Checkpoint
end TorchLean
