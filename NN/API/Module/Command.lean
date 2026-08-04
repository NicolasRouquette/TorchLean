/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Rand
public import NN.API.Module.Execution

/-!
# Executable Module Commands

Command-line support for executable TorchLean programs, including help output, seed parsing,
banners, and exit codes. Module construction and device selection are defined in
`NN.API.Module.Execution`.
-/

@[expose] public section

namespace TorchLean.Module

/-- Options for `TorchLean.Module.run` (banner printing, trailing success message, and flushing). -/
structure RunOptions where
  /-- Optional banner to print before executing the program. -/
  banner? : Option (Options → String) := none
  /-- Flush stdout after printing the banner, when present. -/
  flush : Bool := true
  /-- Print `"{exeName}: ok"` on success. -/
  printOk : Bool := false
deriving Inhabited

namespace RunOptions

/-- Print the configured executable banner, if one was supplied. -/
def printBanner (o : RunOptions) (opts : Options) : IO Unit := do
  match o.banner? with
  | none => pure ()
  | some banner =>
      IO.println (banner opts)
      if o.flush then
        (← IO.getStdout).flush

end RunOptions

/-- How `run` should select the scalar backend for an executable. -/
inductive RunAction where
  /-- Allow dtype selection; the continuation must work for every executable scalar backend. -/
  | scalarPolymorphic
      (k :
        ∀ {α : Type}, [_root_.Context α] → [DecidableEq Spec.Shape] → [ToString α] →
          [_root_.TorchLean.Runtime.FromFloat α] →
          (cast : Float → α) → (opts : Options) → (rest : List String) → IO Unit)
  /-- Force builtin `Float`, as required by Float-only IO bridges and CUDA upload paths. -/
  | float (k : (opts : Options) → (rest : List String) → IO Unit)

/-- Generic help text for executables built on `TorchLean.Module.run`. -/
def runUsage (exeName : String) : String :=
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
    , "  --dtype float|ieee754exec"
    , "      float is native runtime arithmetic; ieee754exec is bit-level binary32 where supported."
    , "  --backend eager|compiled"
    , "      eager executes directly; compiled records and runs the proof-linked graph where supported."
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
model initialization and the eager or compiled session observe the same seed.
-/
def run
    (exeName : String)
    (args : List String)
    (action : RunAction)
    (runOpts : RunOptions := {}) :
    IO UInt32 := do
  let args := TorchLean.CLI.dropDashDash args
  if args.contains "--help" || args.contains "-h" then
    IO.println (runUsage exeName)
    return 0
  let (seed, args) ←
    match TorchLean.CLI.takeSeed args 0 with
    | .ok v => pure v
    | .error msg => throw <| IO.userError s!"{exeName}: {msg}"

  _root_.TorchLean.rand.manualSeed seed

  let printOk : IO Unit := do
    if runOpts.printOk then
      IO.println s!"{exeName}: ok"

  match action with
  | .scalarPolymorphic k =>
      withRuntime args (fun {α} _ _ _ _ cast opts rest => do
        let opts : Options := { opts with seed := seed }
        runOpts.printBanner opts
        k (α := α) cast opts rest
        printOk)
      pure 0
  | .float k =>
      let (cfg, rest) ←
        match ExecConfig.parseAndStripWithDefaultDType args .float with
        | .ok v => pure v
        | .error msg => throw <| IO.userError msg
      if cfg.dtype != .float then
        throw <| IO.userError s!"{exeName}: this program only supports `--dtype float`"
      ExecConfig.log cfg
      let opts ← match ExecConfig.toOptions cfg seed with
        | .ok opts => pure opts
        | .error msg => throw <| IO.userError msg
      opts.validateForExecution
      runOpts.printBanner opts
      k opts rest
      printOk
      pure 0

end TorchLean.Module
