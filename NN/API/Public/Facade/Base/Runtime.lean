/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Report
public import NN.API.Public.Facade.Base.Root

/-!
# TorchLean Runtime Names

Dtype, backend, device, and runtime-selection helpers.
-/

@[expose] public section

namespace TorchLean

namespace Runtime

/-- Public runtime scalar-format selection used by command-line options. -/
abbrev DType := NN.API.DType

namespace DType

/-- Native Lean `Float` execution. -/
abbrev float := NN.API.DType.float

/-- Exact real-number semantics intended for specifications and proofs. -/
abbrev real := NN.API.DType.real

/-- Executable IEEE32 semantics configured by rounding and transcendental options. -/
abbrev float32 (cfg : NN.API.Float32Config := {}) : DType :=
  NN.API.DType.float32 cfg

/-- Complex execution whose real components use the selected IEEE32 configuration. -/
abbrev complex (cfg : NN.API.Float32Config := {}) : DType :=
  NN.API.DType.complex cfg

end DType

/-!
Scalar classes used after a runtime backend has been selected.

Most TorchLean code should let `Trainer` provide these instances implicitly. A few demos use
`Runtime.withOptions` / `Runtime.withOptionsScalar` to run the same code under a CLI-selected scalar
backend. These classes stay under `TorchLean.Runtime` so the root namespace can stay focused on
models, data, and training.
-/

/--
Executable scalar support for TorchLean examples.

Use this when an example chooses a scalar backend or writes code that is polymorphic over the
selected runtime.
-/
abbrev Scalar := NN.API.Runtime.Scalar

/--
Scalar math used by TorchLean model and loss definitions.

Runtime-selected examples and verification demos use this when they need the same model code to run
under more than one scalar backend.
-/
abbrev SemanticScalar := NN.API.Semantics.Scalar

/-- Scalar operations needed to build and manipulate shape-indexed TorchLean tensors. -/
abbrev TensorScalar := _root_.Context

export NN.API.TorchLean (Ops RefTy Program)

/-- Runtime execution strategy: eager evaluation or compiled graph execution. -/
abbrev Backend := _root_.Runtime.Autograd.Torch.Backend

namespace Backend

/-- Execute operations immediately while constructing the autograd tape. -/
abbrev eager : Backend := _root_.Runtime.Autograd.Torch.Backend.eager

/-- Compile the captured program before executing it. -/
abbrev compiled : Backend := _root_.Runtime.Autograd.Torch.Backend.compiled

end Backend

/-- Physical or logical device selected for runtime execution. -/
abbrev Device := NN.Backend.Device

namespace Device

/-- Portable CPU execution. -/
abbrev cpu : Device := NN.Backend.Device.cpu

/-- Native NVIDIA CUDA execution. -/
abbrev cuda : Device := NN.Backend.Device.cuda

/-- AMD ROCm device selector. -/
abbrev rocm : Device := NN.Backend.Device.rocm

/-- Apple Metal device selector. -/
abbrev metal : Device := NN.Backend.Device.metal

/-- WebAssembly runtime selector. -/
abbrev wasm : Device := NN.Backend.Device.wasm

/-- Google TPU device selector. -/
abbrev tpu : Device := NN.Backend.Device.tpu

/-- AWS Trainium device selector. -/
abbrev trainium : Device := NN.Backend.Device.trainium

/-- User-registered device provider. -/
abbrev custom : Device := NN.Backend.Device.custom

/-- Execution delegated to an external runtime provider. -/
abbrev external : Device := NN.Backend.Device.external

/--
Parse a public device selector. `auto` chooses the portable CPU runtime; every other value is
validated against the devices known to the backend registry.
-/
def parse (value : String) : Except String Device :=
  if value == "auto" then pure .cpu else NN.Backend.Device.parse value

end Device

/-- Embed a Lean `Float` into the runtime-selected scalar representation. -/
def ofFloat {α : Type} [TorchLean.Runtime.Scalar α] (x : Float) : α :=
  NN.API.Runtime.ofFloat x

/--
Parse the usual TorchLean runtime flags and run a `Float` callback.

Examples should use this instead of calling `TorchLean.Module.run` directly; that lower-level
dispatcher is what backs this wrapper.
-/
def runFloat
    (exeName : String) (args : List String)
    (banner : Options → String)
    (k : (opts : Options) → (rest : List String) → IO Unit)
    (printOk : Bool := true) : IO UInt32 :=
  NN.API.Common.runFloat exeName args banner k printOk

/-- Parse runtime flags and execute a callback on the native CUDA `Float` path. -/
abbrev runCudaFloat := NN.API.Common.runCudaFloat

/-- Parse runtime flags and execute a callback on the eager CUDA `Float` path. -/
abbrev runCudaEagerFloat := NN.API.Common.runCudaEagerFloat

/--
Parse the standard TorchLean runtime flags and return the resulting `Options`.

Non-polymorphic sibling of `Runtime.withOptions`: examples that always run at `Float` can still
parse `--device`, `--backend`, and `--dtype` without exposing a polymorphic callback.
-/
def parseArgs (args : List String) (defaultDType : DType := .float) :
    Except String (Options × List String) := do
  let (cfg, rest) ←
    NN.API.TorchLean.Module.ExecConfig.parseAndStripWithDefaultDType args defaultDType
  let opts ← NN.API.TorchLean.Module.ExecConfig.toOptions cfg
  pure (opts, rest)

namespace BackendContracts

/-- Backend-contract profile corresponding to the selected runtime options. -/
def profileForOptions (opts : Options) : NN.Backend.BackendProfile :=
  opts.backendProfile

/-- Plan operations under the runtime-selected backend-contract profile. -/
def planReport (opts : Options) (ops : List NN.Backend.BackendOp) : Except String String :=
  (profileForOptions opts).planReport ops

/-- Print the selected backend capsules for operations. -/
def printPlan (opts : Options) (ops : List NN.Backend.BackendOp) : IO Unit := do
  match planReport opts ops with
  | .ok report => IO.println report
  | .error msg => IO.println s!"backend plan unavailable: {msg}"

end BackendContracts

/--
Run an example under the selected runtime and pass through the parsed runtime options.

Use this when an example needs to inspect `--backend`, `--device`, or similar flags after TorchLean
has selected the scalar backend.
-/
def withOptions
    (args : List String)
    (k :
      ∀ {α : Type}, [SemanticScalar α] → [DecidableEq Shape] → [ToString α] →
        [Scalar α] → (cast : Float → α) → Options → (rest : List String) → IO Unit) :
    IO Unit :=
  NN.API.TorchLean.Module.withRuntime args
    (fun {α} _ _ _ _ cast opts rest => k (α := α) cast opts rest)

/--
Run an example under the selected runtime and pass through runtime options when the callback does
not need an explicit Float-cast function.
-/
def withOptionsScalar
    (args : List String)
    (k :
      ∀ {α : Type}, [SemanticScalar α] → [DecidableEq Shape] → [ToString α] →
        [Scalar α] → Options → (rest : List String) → IO Unit) :
    IO Unit :=
  withOptions args (fun {α} _ _ _ _ _cast opts rest => k (α := α) opts rest)

/--
Run a verification or demo command under the selected runtime dtype.

Banner-printing runtime dispatcher matching the convention used by
`lake exe verify -- ...` commands.
-/
def runWithDType
    (title : String) (args : List String)
    (k : ∀ {α : Type}, [SemanticScalar α] → [DecidableEq Shape] → [ToString α] → [Scalar α] →
      IO Unit) :
    IO Unit :=
  NN.API.Common.runWithRuntimeDType title args
    (fun {α} _ _ _ _ => k (α := α))

end Runtime


end TorchLean
