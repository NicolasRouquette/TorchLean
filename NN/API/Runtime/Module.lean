/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Core
public import NN.API.Rand
public import NN.API.TorchLean.ParamIO
public import NN.API.TorchLean.Schedulers
public import NN.GraphSpec.Models.TorchLean
public import NN.Runtime.Autograd.TorchLean
public import NN.Runtime.Autograd.Train.Dataset
public import NN.Runtime.RL
public import NN.Spec.RL.MDP
public import NN.Spec.RL.MarkovMDP
public import NN.Spec.RL.FiniteStochasticMDP
public import NN.API.Runtime.Core
public import NN.API.Runtime.Layers
public import NN.API.Runtime.Autograd
public import NN.Backend.Report

import Mathlib.Algebra.Order.Algebra
import NN.Spec.Autograd.AutogradSpec

@[expose] public section

namespace NN
namespace API
namespace TorchLean

/-!
# Runtime Modules and Executable Entrypoints

Scalar-module instantiation, runtime initialization, CLI execution config parsing, and executable
`main` helpers.
-/

namespace Models

/-!
## Model constructors (re-export)

This namespace re-exports ready-made model constructors for runnable
examples and integration checks.

For compositional building blocks, prefer `API.TorchLean.LayerCore` and `API.TorchLean.LayerCore`.
-/

export _root_.NN.GraphSpec.Models.TorchLean
  (mlp autoencoder twoConvCnn softmaxRegression mlpClassifier transformerBlock)
end Models

namespace Module

/-!
### ScalarModule API (Session-Like Interface)

The `ScalarModule` interface is the TorchLean equivalent of "instantiate a model, then do forward,
backward, and optimizer steps" in an imperative runtime.

The module mostly re-exports `Runtime.Autograd.TorchLean.Module.*` and adds small CLI-friendly
helpers (`Module.withModule` / `Module.withModuleRuntime`) that select dtype/backend from flags.
-/

export _root_.Runtime.Autograd.TorchLean.Module (ScalarModuleDef ScalarModule)
namespace RuntimeInit
export _root_.Runtime.Autograd.TorchLean.Module.RuntimeInit
  (FloatInit Plan xavierUniformForShape kaimingUniformForShape xavierLinearWeight
   kaimingLinearWeight)
end RuntimeInit
export _root_.Runtime.Autograd.TorchLean.Module.ScalarModule
  (create forward lossAndBackward backward stepWithLoss step initOptim stepWith
   stepWithOptimizerAndLoss params setParams trainSGD trainWith meanLoss)
export _root_.Runtime.Autograd.TorchLean.Module.ScalarModuleDef
  (forwardWithParams instantiate instantiateFloat instantiateFloatWithRuntimePlan
   instantiateFloatWithRuntimeInit)

/--
Instantiate a `ScalarModuleDef` under explicit Torch options such as `backend` and `device`.

This is the most direct "runtime" entrypoint (used by the CPU/CUDA example binaries), since it
threads the same options record all the way down to the eager tape / CUDA tape selection.

The shorter `instantiate` entrypoint selects the backend and then uses default runtime options.
-/
def instantiateConfigured
    {α : Type} [API.Semantics.Scalar α] [DecidableEq Spec.Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    {paramShapes inputShapes : List Spec.Shape}
    (defn : ScalarModuleDef paramShapes inputShapes)
    (cast : Float → α) (opts : Options) :
    IO (ScalarModule α paramShapes inputShapes) :=
  _root_.Runtime.Autograd.TorchLean.Module.ScalarModuleDef.instantiateWith
    (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes) defn cast opts

/--
Instantiate a Float module with runtime layer parameter initializers.

This is the public runtime-initialization entrypoint. The initializer plan is indexed by the same
`paramShapes` list as the module, so Lean checks that every parameter has exactly one initializer.
In CUDA mode, supported initializers allocate device buffers directly instead of first constructing
every parameter as a large nested Lean tensor.
-/
def instantiateFloatWithPlan
    {paramShapes inputShapes : List Spec.Shape}
    (defn : ScalarModuleDef paramShapes inputShapes)
    (opts : Options)
    (plan : RuntimeInit.Plan paramShapes) :
    IO (ScalarModule Float paramShapes inputShapes) :=
  _root_.Runtime.Autograd.TorchLean.Module.ScalarModuleDef.instantiateFloatWithRuntimePlan
    (paramShapes := paramShapes) (inputShapes := inputShapes) defn opts plan

/--
List-based wrapper for checkpoint/JSON boundaries.

If the caller has a statically known parameter list, prefer
`instantiateFloatWithPlan`; this wrapper checks the list length before applying it.
-/
def instantiateFloatWithInit
    {paramShapes inputShapes : List Spec.Shape}
    (defn : ScalarModuleDef paramShapes inputShapes)
    (opts : Options)
    (inits : List RuntimeInit.FloatInit) :
    IO (ScalarModule Float paramShapes inputShapes) :=
  _root_.Runtime.Autograd.TorchLean.Module.ScalarModuleDef.instantiateFloatWithRuntimeInit
    (paramShapes := paramShapes) (inputShapes := inputShapes) defn opts inits

/--
Execution configuration parsed from CLI flags.

Supported flags (parsed by `ExecConfig.parseAndStrip`):
- `--dtype ...` / `--float32-mode ...` (see `NN.API.DType`)
- `--backend eager|compiled`
- `--device auto|cpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external`
- `--show-backend` (print backend capsules when the eager runtime first executes them)
-/
structure ExecConfig where
  /-- Scalar dtype selection. -/
  dtype : DType := .float
  /-- Execution backend selection. -/
  backend : Backend := .eager
  /-- Explicit eager execution device. -/
  device : NN.Backend.Device := .cpu
  /-- Print each backend capsule when the eager runtime first executes it. -/
  showBackend : Bool := false
  deriving Repr, DecidableEq

namespace ExecConfig

/-- Parse a backend selector string into a runtime `Backend`. -/
def parseBackend (v : String) : Except String Backend := do
  if v == "eager" then
    pure .eager
  else if v == "compiled" then
    pure .compiled
  else
    throw s!"unknown --backend {v} (supported: eager | compiled)"

/-- Parse a CLI device selector. `auto` currently resolves to the portable CPU runtime. -/
def parseDevice (value : String) : Except String NN.Backend.Device :=
  if value == "auto" then pure .cpu else NN.Backend.Device.parse value

/-- Whether a raw CLI argument list explicitly requests CUDA. -/
def requestsCuda : List String → Bool
  | [] => false
  | "--device=cuda" :: _ => true
  | "--device" :: "cuda" :: _ => true
  | _ :: rest => requestsCuda rest

/--
Parse CLI flags handled by `ExecConfig` and return `(cfg, rest)`.

Consumed flags:
- `--backend eager|compiled` (at most once),
- `--device auto|cpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external`,
- `--show-backend` (boolean flag; removed from `rest`).

All dtype/Float32 selection flags are delegated to `DType.parseAndStripWithDefault`.

Default dtype policy:
- If the user does not specify `--dtype` / `--float32-mode` and CUDA is selected, default to
  `dtype=float` (CUDA eager supports `Float` upload/download).
- Otherwise default to `dtype=float32` (executable IEEE-754 float32 semantics).

Named future devices are accepted at parse time so `--show-backend` and planning diagnostics can
explain them. Runtime session creation still rejects devices that this build cannot execute.

The selected device chooses its normal registered kernels. Users do not need a second performance
flag after selecting CUDA or another accelerator.
-/
def parseAndStripWithDefaultDType (args : List String) (defaultDType : DType) :
    Except String (ExecConfig × List String) := do
  let (dtype, args1) ← DType.parseAndStripWithDefault args defaultDType
  let (backend, args2) ←
    _root_.TorchLean.CLI.takeParsedFlagDefault args1 "backend" "eager" parseBackend
  let rec go (device : NN.Backend.Device) (showBackend : Bool) (acc : List String) :
      List String → Except String (NN.Backend.Device × Bool × List String)
    | [] => pure (device, showBackend, acc.reverse)
    | "--device" :: v :: as => do
        let d ← parseDevice v
        go d showBackend acc as
    | "--device" :: [] =>
        throw "missing value after --device (supported: auto | cpu | cuda | rocm | metal | wasm | tpu | trainium | custom | external)"
    | a :: as =>
        if a.startsWith "--device=" then do
          let d ← parseDevice ((a.drop "--device=".length).toString)
          go d showBackend acc as
        else if a == "--show-backend" then
          go device true acc as
        else
          go device showBackend (a :: acc) as
  let (device, showBackend, rest) ← go .cpu false [] args2
  pure ({
    dtype := dtype,
    backend := backend,
    device := device,
    showBackend := showBackend
  }, rest)

/-- Convert a parsed CLI execution config to runtime `Options`. -/
def toOptions (cfg : ExecConfig) (seed : Nat := 0) : Except String Options := do
  let profile ← match NN.Backend.BackendProfile.maintainedForDevice? cfg.device with
    | some profile => pure profile
    | none =>
        throw s!"device `{cfg.device.cliName}` has no maintained runtime profile; use a programmatic backend profile"
  pure
    { backend := cfg.backend
      seed := seed
      executionProfile := profile
      showBackend := cfg.showBackend }

/-- Parse CLI flags with the standard TorchLean default dtype policy. -/
def parseAndStrip (args : List String) : Except String (ExecConfig × List String) := do
  let defaultDType : DType := if requestsCuda args then .float else .float32 {}
  parseAndStripWithDefaultDType args defaultDType

/-- Log the chosen execution config to stdout for reproducible runs. -/
def log (cfg : ExecConfig) : IO Unit := do
  DType.log cfg.dtype
  IO.println s!"[TorchLean] backend: {reprStr cfg.backend}"
  IO.println s!"[TorchLean] device: {cfg.device.cliName}"

end ExecConfig

/--
Parse runtime flags (`--dtype`, `--backend`, `--device`, `--show-backend`) and choose an executable
scalar `α`, then call `k` with:
- `cast : Float → α` for building inputs from literals
- `opts : Options` selecting the backend/kernel mode
- `rest : List String` containing the remaining CLI arguments

This is useful for scripts that need to build a dataset/loader (and maybe determine shapes/batch
sizes) before instantiating a concrete `ScalarModuleDef`.
-/
def withRuntime
    (args : List String)
    (k :
      ∀ {α : Type}, [API.Semantics.Scalar α] → [DecidableEq Spec.Shape] → [ToString α] →
        [API.Runtime.Scalar α] →
        (cast : Float → α) → (opts : Options) → (rest : List String) → IO Unit) :
    IO Unit := do
  let (cfg, rest) ←
    match ExecConfig.parseAndStrip args with
    | .ok v => pure v
    | .error msg => throw <| IO.userError msg
  ExecConfig.log cfg
  let opts ← match ExecConfig.toOptions cfg with
    | .ok opts => pure opts
    | .error msg => throw <| IO.userError msg
  opts.validateForExecution
  match (← DType.withRuntime cfg.dtype (fun {α} _ _ _ _ => do
        k (α := α) (API.Runtime.ofFloat (α := α)) opts rest
      )) with
  | .ok () => pure ()
  | .error msg => throw <| IO.userError msg

/--
Instantiate a `ScalarModuleDef` under CLI runtime flags (`--dtype`, `--backend`, `--device`,
`--show-backend`), then call a continuation.

This provides the cast function `Float → α` so call sites can build inputs from float literals.
-/
def withModule
    {paramShapes inputShapes : List Spec.Shape}
    (defn : ScalarModuleDef paramShapes inputShapes)
    (args : List String)
    (k :
      ∀ {α : Type}, [API.Semantics.Scalar α] → [DecidableEq Spec.Shape] → [ToString α] →
        (cast : Float → α) → ScalarModule α paramShapes inputShapes → (rest : List String) →
        IO Unit) :
    IO Unit := do
  let (cfg, rest) ←
    match ExecConfig.parseAndStrip args with
    | .ok v => pure v
    | .error msg => throw <| IO.userError msg
  ExecConfig.log cfg
  let opts ← match ExecConfig.toOptions cfg with
    | .ok opts => pure opts
    | .error msg => throw <| IO.userError msg
  opts.validateForExecution
  match cfg.dtype with
  | .float =>
      -- Keep the Float branch explicit. If this path is hidden behind the scalar-polymorphic
      -- `DType.withExec` continuation, Lean can elaborate module construction with the generic
      -- fallback CUDA converter instead of the real Float upload bridge. That still compiles, but
      -- a CUDA training step later fails when it tries to upload a Float tensor.
      let m ← _root_.Runtime.Autograd.TorchLean.Module.ScalarModuleDef.instantiateFloat
        (paramShapes := paramShapes) (inputShapes := inputShapes) defn opts
      k (α := Float) id m rest
  | _ =>
      if (cfg.device == .cuda) then
        throw <| IO.userError "torch: eager CUDA module execution currently requires --dtype float"
      match (← DType.withExec cfg.dtype (fun {α} _ _ _ cast => do
            let m ← _root_.Runtime.Autograd.TorchLean.Module.ScalarModuleDef.instantiateWith
              (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes)
              defn cast opts
            k (α := α) cast m rest
          )) with
      | .ok () => pure ()
      | .error msg => throw <| IO.userError msg

/--
Like `withModule`, but also provides an `API.Runtime.Scalar α` instance (for numeric literals).
-/
def withModuleRuntime
    {paramShapes inputShapes : List Spec.Shape}
    (defn : ScalarModuleDef paramShapes inputShapes)
    (args : List String)
    (k :
      ∀ {α : Type}, [API.Semantics.Scalar α] → [DecidableEq Spec.Shape] → [ToString α] →
        [API.Runtime.Scalar α] →
        ScalarModule α paramShapes inputShapes → (rest : List String) → IO Unit) :
    IO Unit := do
  let (cfg, rest) ←
    match ExecConfig.parseAndStrip args with
    | .ok v => pure v
    | .error msg => throw <| IO.userError msg
  ExecConfig.log cfg
  let opts ← match ExecConfig.toOptions cfg with
    | .ok opts => pure opts
    | .error msg => throw <| IO.userError msg
  opts.validateForExecution
  match cfg.dtype with
  | .float =>
      -- Same reason as `withModule`: CUDA module construction should see `α = Float` directly, so
      -- the Float-specific `TensorConv` instance is selected before the runner is handed to user
      -- code.
      let m ← _root_.Runtime.Autograd.TorchLean.Module.ScalarModuleDef.instantiateFloat
        (paramShapes := paramShapes) (inputShapes := inputShapes) defn opts
      k (α := Float) m rest
  | _ =>
      if (cfg.device == .cuda) then
        throw <| IO.userError "torch: eager CUDA module execution currently requires --dtype float"
      match (← DType.withRuntime cfg.dtype (fun {α} _ _ _ _ => do
            let m ← _root_.Runtime.Autograd.TorchLean.Module.ScalarModuleDef.instantiateWith
              (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes)
              defn (API.Runtime.ofFloat (α := α)) opts
            k (α := α) m rest
          )) with
      | .ok () => pure ()
      | .error msg => throw <| IO.userError msg

/-!
## Executable `main` Helpers

TorchLean has a lot of pure, type-indexed code (models live in `Type 2`), but runnable scripts still
want a "single entrypoint" that handles:
- parsing `--seed`,
- selecting an executable dtype/backend/device from flags,
- seeding TorchLean's global RNG stream (`API.rand`) so `nn.freshSeed`/`nn.withModel` are deterministic.
-/

/-- Options for `TorchLean.Module.run` (banner printing, trailing ok, etc.). -/
structure RunOptions where
  /-- Optional banner to print before executing the program. -/
  banner? : Option (Options → String) := none
  /-- Flush stdout after printing the banner (if present). -/
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
  /--
  Allow `--dtype` selection (the continuation must work for all executable scalar backends).
  -/
  | any
      (k :
        ∀ {α : Type}, [API.Semantics.Scalar α] → [DecidableEq Spec.Shape] → [ToString α] →
          [API.Runtime.Scalar α] →
          (cast : Float → α) → (opts : Options) → (rest : List String) → IO Unit)
  /--
  Force the scalar backend to builtin `Float` (useful for Float-only IO bridges / CUDA upload paths).
  -/
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
    , "      rocm, metal, wasm, tpu, trainium, custom, and external are named targets"
    , "      that fail until their runtimes are implemented and registered."
    , "  --dtype float|ieee754exec"
    , "      float is the native runtime scalar; ieee754exec uses executable Float32 semantics where supported."
    , "  --backend eager|compiled"
    , "      eager runs the ordinary runtime path; compiled records/runs the proof-linked graph path where supported."
    , "  --seed N"
    , "  --show-backend"
    , "      print the backend-contract capsules selected by the current device/profile."
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
CLI entrypoint helper for executable `main` functions.

This parses:
- `--seed N` (via `TorchLean.CLI.takeSeed`), and
- runtime execution flags (`--dtype`, `--float32-mode`, `--backend`, `--device`,
  `--show-backend`),
then executes the chosen `RunAction`.

It also seeds TorchLean's global RNG stream (`API.rand`) so code that draws init seeds via
`API.nn.freshSeed`/`API.nn.withModel` is deterministic by default, matching the PyTorch pattern of
calling `torch.manual_seed` once in `main`.
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

  _root_.NN.API.rand.manualSeed seed

  let printOk : IO Unit := do
    if runOpts.printOk then
      IO.println s!"{exeName}: ok"

  match action with
  | .any k =>
      withRuntime args (fun {α} _ _ _ _ cast opts rest => do
        -- Keep seed in the same `Options` record used by the Torch eager/compiled sessions so scripts
        -- can still follow the familiar pattern `nn.manualSeed opts.seed` when desired.
        let opts : Options := { opts with seed := seed }
        runOpts.printBanner opts
        k (α := α) cast opts rest
        printOk
      )
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

end Module

end TorchLean
end API
end NN
