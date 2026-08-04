/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Backend.Report
public import NN.API.Scalar
public import NN.API.Module.Command

/-!
# Runtime Selection

Dtype, backend, device, and runtime-selection helpers.
-/

@[expose] public section

namespace TorchLean

export _root_.Runtime.Autograd.Torch (Options)

namespace Runtime

export _root_.Runtime.Autograd.Torch (Ops)
export _root_.Runtime.Autograd.TorchLean (RefTy Program)

/-- Runtime execution strategy: eager evaluation or compiled graph execution. -/
abbrev Backend := _root_.Runtime.Autograd.Torch.Backend

namespace Backend

export _root_.Runtime.Autograd.Torch.Backend (eager compiled)

end Backend

/-- Physical or logical device selected for runtime execution. -/
abbrev Device := NN.Backend.Device

namespace Device

export NN.Backend.Device (cpu cuda rocm metal wasm tpu trainium custom external)

/--
Parse a public device selector. `auto` chooses the portable CPU runtime; every other value is
validated against the devices known to the backend registry.
-/
def parse (value : String) : Except String Device :=
  if value == "auto" then pure .cpu else NN.Backend.Device.parse value

end Device

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
  TorchLean.Module.run exeName args (.float k)
    { banner? := some banner, printOk := printOk }

/-- Run a Float command on CUDA, adding `--device cuda` when no device was supplied. -/
def runCudaFloat
    (exeName : String)
    (args : List String)
    (banner : Options → String)
    (k : (opts : Options) → (rest : List String) → IO Unit)
    (printOk : Bool := true) : IO UInt32 := do
  let hasDeviceFlag :=
    args.any fun arg => arg == "--device" || arg.startsWith "--device="
  let cudaArgs : Except String (List String) := do
    if hasDeviceFlag then
      let (cfg, _) ←
        TorchLean.Module.ExecConfig.parseAndStripWithDefaultDType args .float
      unless cfg.device == .cuda do
        throw s!"this command requires --device cuda, not --device {cfg.device.cliName}"
      pure args
    else
      pure ("--device" :: "cuda" :: args)
  let args ← match cudaArgs with
    | .ok parsed => pure parsed
    | .error msg =>
        IO.eprintln s!"{exeName}: {msg}"
        return 1
  runFloat exeName args banner k printOk

/-- Run a Float command on the eager CUDA runtime. -/
def runCudaEagerFloat
    (exeName : String)
    (args : List String)
    (banner : Options → String)
    (k : (opts : Options) → (rest : List String) → IO Unit)
    (printOk : Bool := true) : IO UInt32 :=
  runCudaFloat exeName args banner
    (fun opts rest => do
      unless opts.backend == .eager do
        throw <| IO.userError <|
          s!"{exeName}: --backend eager is required for CUDA execution"
      k opts rest)
    printOk

/--
Parse the standard TorchLean runtime flags and return the resulting `Options`.

Non-polymorphic sibling of `Runtime.withOptions`: examples that always run at `Float` can still
parse `--device`, `--backend`, and `--dtype` without exposing a polymorphic callback.
-/
def parseArgs (args : List String) (defaultDType : DType := .float) :
    Except String (Options × List String) := do
  let (cfg, rest) ←
    TorchLean.Module.ExecConfig.parseAndStripWithDefaultDType args defaultDType
  let opts ← TorchLean.Module.ExecConfig.toOptions cfg
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
      ∀ {α : Type}, [Context α] → [DecidableEq Spec.Shape] → [ToString α] →
        [FromFloat α] → (cast : Float → α) → Options → (rest : List String) → IO Unit) :
    IO Unit :=
  TorchLean.Module.withRuntime args
    (fun {α} _ _ _ _ cast opts rest => k (α := α) cast opts rest)

/--
Run an example under the selected runtime and pass through runtime options when the callback does
not need an explicit Float-cast function.
-/
def withOptionsScalar
    (args : List String)
    (k :
      ∀ {α : Type}, [Context α] → [DecidableEq Spec.Shape] → [ToString α] →
        [FromFloat α] → Options → (rest : List String) → IO Unit) :
    IO Unit :=
  withOptions args (fun {α} _ _ _ _ _cast opts rest => k (α := α) opts rest)

/--
Run a verification or demo command under the selected runtime dtype.

Banner-printing runtime dispatcher matching the convention used by
`lake exe verify -- ...` commands.
-/
def runWithDType
    (title : String) (args : List String)
    (k : ∀ {α : Type}, [Context α] → [DecidableEq Spec.Shape] → [ToString α] → [FromFloat α] →
      IO Unit) :
    IO Unit :=
  do
    IO.println s!"=== {title} ==="
    let (dtype, _rest) ←
      match DType.parseAndStrip args with
      | .ok parsed => pure parsed
      | .error msg => throw <| IO.userError msg
    DType.log dtype
    match (← DType.withRuntime dtype (fun {α} _ _ _ _ => k (α := α))) with
    | .ok () => pure ()
    | .error msg => throw <| IO.userError msg

end Runtime


end TorchLean
