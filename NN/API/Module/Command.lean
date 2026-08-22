/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.CLI
public import NN.API.Rand
public import NN.API.Module.Execution

/-!
# Executable Module Commands

Command-line support for executable TorchLean programs, including scalar and device selection,
help output, seed parsing, banners, and exit codes.
-/

@[expose] public section

namespace TorchLean
namespace Module

/-- Runtime choices parsed from the shared command-line flags. -/
structure ExecConfig where
  /-- Scalar semantics for this execution. -/
  scalar : _root_.TorchLean.Runtime.ScalarMode := .float32
  /-- Immediate or typed-graph execution. -/
  execution : ExecutionMode := .eager
  /-- Requested execution device. -/
  device : NN.Backend.Device := .cpu
  /-- Whether to print each backend capsule when first used. -/
  showBackend : Bool := false
  deriving Repr, DecidableEq

namespace ExecConfig

/--
Parse the shared scalar, execution, device, and backend-reporting flags.

Named devices without an installed runtime remain parseable so diagnostics can report the intended
target. `Options.validateForExecution` rejects such a configuration before execution.
-/
def parseWithScalar
    (args : List String) (defaultScalar : _root_.TorchLean.Runtime.ScalarMode) :
    Except String (ExecConfig × List String) := do
  let (scalar, args) ←
    _root_.TorchLean.Runtime.ScalarMode.parseAndStripWithDefault args defaultScalar
  let (execution, args) ←
    _root_.TorchLean.CLI.takeParsedFlagDefault args "execution" "eager"
      TorchLean.Runtime.ExecutionMode.parse
  let rec go (device : NN.Backend.Device) (showBackend : Bool) (acc : List String) :
      List String → Except String (NN.Backend.Device × Bool × List String)
    | [] => pure (device, showBackend, acc.reverse)
    | "--device" :: value :: rest => do
        go (← TorchLean.Runtime.Device.parse value) showBackend acc rest
    | ["--device"] =>
        throw "missing value after --device (supported: auto | cpu | cuda | rocm | metal | wasm | tpu | trainium | custom | external)"
    | arg :: rest =>
        if arg.startsWith "--device=" then do
          let device ← TorchLean.Runtime.Device.parse ((arg.drop "--device=".length).toString)
          go device showBackend acc rest
        else if arg == "--show-backend" then
          go device true acc rest
        else
          go device showBackend (arg :: acc) rest
  let (device, showBackend, rest) ← go .cpu false [] args
  pure ({ scalar, execution, device, showBackend }, rest)

/-- Convert parsed command-line choices to explicit runtime options. -/
def toOptions (cfg : ExecConfig) (seed : Nat := 0) : Except String Options := do
  if (NN.Backend.BackendProfile.maintainedForDevice? cfg.device).isNone then
    throw s!"device `{cfg.device.cliName}` has no maintained runtime profile; use a programmatic backend profile"
  pure
    { execution := cfg.execution
      device := cfg.device
      seed
      backendProfile? := none
      showBackend := cfg.showBackend }

/-- Parse the shared runtime flags with native binary32 as the default scalar. -/
def parseAndStrip (args : List String) : Except String (ExecConfig × List String) :=
  parseWithScalar args .float32

/-- Print the selected scalar, execution strategy, and device. -/
def log (cfg : ExecConfig) : IO Unit := do
  _root_.TorchLean.Runtime.ScalarMode.log cfg.scalar
  IO.println s!"[TorchLean] execution: {reprStr cfg.execution}"
  IO.println s!"[TorchLean] device: {cfg.device.cliName}"

end ExecConfig

/--
Parse the shared runtime flags, select an executable scalar, and call `k` with the corresponding
literal conversion and explicit runtime options.
-/
def withRuntime
    (args : List String)
    (k :
      ∀ {α : Type}, [_root_.Context α] → [DecidableEq Spec.Shape] → [ToString α] →
        [_root_.TorchLean.Runtime.FromFloat α] →
        (cast : Float → α) → (opts : Options) → (rest : List String) → IO Unit) :
    IO Unit := do
  let (cfg, rest) ← match ExecConfig.parseAndStrip args with
    | .ok result => pure result
    | .error message => throw <| IO.userError message
  ExecConfig.log cfg
  let opts ← match ExecConfig.toOptions cfg with
    | .ok result => pure result
    | .error message => throw <| IO.userError message
  opts.validateForExecution
  match (← _root_.TorchLean.Runtime.ScalarMode.withRuntime cfg.scalar (fun {α} _ _ _ _ =>
      k (α := α) (_root_.TorchLean.Runtime.ofFloat (α := α)) opts rest)) with
  | .ok () => pure ()
  | .error message => throw <| IO.userError message

/--
Parse the shared runtime flags, instantiate `defn`, and pass the resulting scalar module to `k`.
-/
def withModule
    {stateShapes inputShapes : List Spec.Shape}
    (defn : ObjectiveDef stateShapes inputShapes)
    (args : List String)
    (k :
      ∀ {α : Type}, [_root_.Context α] → [DecidableEq Spec.Shape] → [ToString α] →
        [_root_.TorchLean.Runtime.FromFloat α] →
        (cast : Float → α) → Objective α stateShapes inputShapes →
        (rest : List String) → IO Unit) :
    IO Unit := do
  let (cfg, rest) ← match ExecConfig.parseAndStrip args with
    | .ok result => pure result
    | .error message => throw <| IO.userError message
  ExecConfig.log cfg
  let opts ← match ExecConfig.toOptions cfg with
    | .ok result => pure result
    | .error message => throw <| IO.userError message
  opts.validateForExecution
  match cfg.scalar with
  | .float32 =>
      let module ← _root_.Runtime.Autograd.TorchLean.Module.ObjectiveDef.instantiateWith
        (α := Float32) (stateShapes := stateShapes) (inputShapes := inputShapes)
        defn Float.toFloat32 opts
      k (α := Float32) Float.toFloat32 module rest
  | _ =>
      if cfg.device == .cuda then
        throw <| IO.userError "torch: CUDA module execution currently requires --scalar float32"
      match (← _root_.TorchLean.Runtime.ScalarMode.withRuntime cfg.scalar (fun {α} _ _ _ _ => do
          let cast := _root_.TorchLean.Runtime.ofFloat (α := α)
          let module ← _root_.Runtime.Autograd.TorchLean.Module.ObjectiveDef.instantiateWith
            (α := α) (stateShapes := stateShapes) (inputShapes := inputShapes)
            defn cast opts
          k (α := α) cast module rest)) with
      | .ok () => pure ()
      | .error message => throw <| IO.userError message

end Module
end TorchLean

namespace TorchLean.Module.Command

/-- Banner, success-message, and flushing configuration for an executable command. -/
structure Config where
  /-- Optional banner to print before executing the program. -/
  banner? : Option (Options → String) := none
  /-- Flush stdout after printing the banner, when present. -/
  flush : Bool := true
  /-- Print `"{exeName}: ok"` on success. -/
  printOk : Bool := false
deriving Inhabited

namespace Config

/-- Print the configured executable banner, if one was supplied. -/
def printBanner (config : Config) (opts : Options) : IO Unit := do
  match config.banner? with
  | none => pure ()
  | some banner =>
      IO.println (banner opts)
      if config.flush then
        (← IO.getStdout).flush

end Config

/-- How an executable command chooses its scalar semantics. -/
inductive Action where
  /-- Allow scalar selection; the continuation must work for every executable scalar backend. -/
  | selectedScalar
      (k :
        ∀ {α : Type}, [_root_.Context α] → [DecidableEq Spec.Shape] → [ToString α] →
          [_root_.TorchLean.Runtime.FromFloat α] →
          (cast : Float → α) → (opts : Options) → (rest : List String) → IO Unit)
  /-- Run a command fixed to native binary32 rather than exposing scalar selection. -/
  | float32 (k : (opts : Options) → (rest : List String) → IO Unit)

/-- Generic help text for executables built on `TorchLean.Module.Command.run`. -/
def usage (exeName : String) : String :=
  String.intercalate "\n"
    [ s!"Usage: {exeName} [runtime flags] [command flags]"
    , ""
    , "Quick examples:"
    , s!"  {exeName} --device cpu --steps 10"
    , s!"  {exeName} --device cuda --steps 10"
    , ""
    , "Runtime flags:"
    , "  -h, --help"
    , "  --device auto|cpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external"
    , "      cpu and cuda are implemented by the current eager runtime;"
    , "      other names are planning targets and fail until a runtime is registered."
    , "  --scalar float32|ieee32-exec|complex64"
    , "      choose native binary32 or TorchLean's bit-level binary32 reference."
    , "  --execution eager|typed-graph"
    , "      eager executes immediately; typed-graph records and reuses a shape-indexed SSA graph."
    , "  --seed N"
    , "  --show-backend"
    , "      print the backend capsules selected by the current device profile."
    , ""
    , "Verification commands:"
    , "  lake exe verify -- list"
    , "  lake exe verify -- margin-cert"
    , "  lake exe verify -- abcrown-leaf"
    , "  lake exe verify -- torchlean-robustness"
    , "  lake exe verify -- torchlean-mlp-workflow"
    , ""
    , "Use `lake exe torchlean --help` for the full example list."
    ]

/--
Run a TorchLean executable after parsing the shared seed and runtime flags.

The selected seed initializes TorchLean's global random stream and is also stored in `Options`, so
model initialization and either execution mode observe the same seed.
-/
def run
    (exeName : String)
    (args : List String)
    (action : Action)
    (config : Config := {}) :
    IO UInt32 := do
  let args := TorchLean.CLI.dropDashDash args
  if args.contains "--help" || args.contains "-h" then
    IO.println (usage exeName)
    return 0
  let (seed, args) ←
    match TorchLean.CLI.takeSeed args 0 with
    | .ok v => pure v
    | .error msg => throw <| IO.userError s!"{exeName}: {msg}"

  _root_.TorchLean.rand.manualSeed seed

  let printOk : IO Unit := do
    if config.printOk then
      IO.println s!"{exeName}: ok"

  match action with
  | .selectedScalar k =>
      withRuntime args (fun {α} _ _ _ _ cast opts rest => do
        let opts : Options := { opts with seed := seed }
        config.printBanner opts
        k (α := α) cast opts rest
        printOk)
      pure 0
  | .float32 k =>
      let (cfg, rest) ←
        match ExecConfig.parseWithScalar args .float32 with
        | .ok v => pure v
        | .error msg => throw <| IO.userError msg
      if cfg.scalar != .float32 then
        throw <| IO.userError s!"{exeName}: this program only supports `--scalar float32`"
      ExecConfig.log cfg
      let opts ← match ExecConfig.toOptions cfg seed with
        | .ok opts => pure opts
        | .error msg => throw <| IO.userError msg
      opts.validateForExecution
      config.printBanner opts
      k opts rest
      printOk
      pure 0

/--
Run a command fixed to TorchLean's native `Float32` runtime.

The callback may still author datasets or reports with host `Float` values; trainer and module
construction convert those values to native binary32 before execution.
-/
def runFloat32
    (exeName : String) (args : List String)
    (banner : Options → String)
    (k : (opts : Options) → (rest : List String) → IO Unit)
    (printOk : Bool := true) : IO UInt32 :=
  run exeName args (.float32 k)
    { banner? := some banner, printOk := printOk }

/-- Run a native-`Float32` command on CUDA, adding `--device cuda` when needed. -/
def runCudaFloat32
    (exeName : String)
    (args : List String)
    (banner : Options → String)
    (k : (opts : Options) → (rest : List String) → IO Unit)
    (printOk : Bool := true) : IO UInt32 := do
  let hasDeviceFlag :=
    args.any fun arg => arg == "--device" || arg.startsWith "--device="
  let cudaArgs : Except String (List String) := do
    if hasDeviceFlag then
      let (cfg, _) ← ExecConfig.parseWithScalar args .float32
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
  runFloat32 exeName args banner k printOk

/-- Run a native-`Float32` command on the eager CUDA runtime. -/
def runCudaEagerFloat32
    (exeName : String)
    (args : List String)
    (banner : Options → String)
    (k : (opts : Options) → (rest : List String) → IO Unit)
    (printOk : Bool := true) : IO UInt32 :=
  runCudaFloat32 exeName args banner
    (fun opts rest => do
      unless opts.execution == .eager do
        throw <| IO.userError <|
          s!"{exeName}: --execution eager is required for CUDA execution"
      k opts rest)
    printOk

end TorchLean.Module.Command
