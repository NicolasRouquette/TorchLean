/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.CLI
public import NN.API.Scalar
public import NN.Backend.Report
public import NN.GraphSpec.Models.TorchLean
public import NN.Runtime.Autograd.TorchLean

import Mathlib.Algebra.Order.Algebra
import NN.Spec.Autograd.AutogradSpec

@[expose] public section

namespace TorchLean
namespace Module

export _root_.Runtime.Autograd.Torch (Backend Options)

/-!
# Module Execution

This file connects typed module definitions to executable scalar modules. It provides parameter
initialization, execution settings, and helpers that select a scalar type and device from command
line arguments.

`ScalarModuleDef` describes the forward and loss programs together with their parameter and input
shapes. Instantiating it produces a mutable `ScalarModule` that can evaluate inputs, run backward,
and update parameters. The shape lists remain part of both types, so construction and execution use
the same parameter ordering.
-/

export _root_.Runtime.Autograd.TorchLean.Module
  (Evaluator ScalarEvaluator ScalarModuleDef ScalarModule)
namespace RuntimeInit
export _root_.Runtime.Autograd.TorchLean.Module.RuntimeInit
  (FloatInit Plan xavierUniformForShape kaimingUniformForShape xavierLinearWeight
   kaimingLinearWeight)
end RuntimeInit
export _root_.Runtime.Autograd.TorchLean.Module.ScalarModule
  (create forward lossAndBackward backward stepWithLoss step initOptim stepWith
   stepWithOptimizerAndLoss params setParams trainSGD trainWith meanLoss)
export _root_.Runtime.Autograd.TorchLean.Module.Evaluator (evaluateT withParams)
export _root_.Runtime.Autograd.TorchLean.Module.ScalarModuleDef
  (evaluatorWithParams forwardWithParams instantiate instantiateFloat
   instantiateFloatWithRuntimePlan instantiateFloatWithRuntimeInit)

/--
Instantiate a `ScalarModuleDef` under explicit Torch options such as `backend` and `device`.

The supplied options are passed unchanged to module construction, including the selected device and
execution strategy.
-/
def instantiateConfigured
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    {paramShapes inputShapes natInputShapes : List Spec.Shape}
    (defn : ScalarModuleDef paramShapes inputShapes natInputShapes)
    (cast : Float → α) (opts : Options) :
    IO (ScalarModule α paramShapes inputShapes natInputShapes) :=
  _root_.Runtime.Autograd.TorchLean.Module.ScalarModuleDef.instantiateWith
    (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes)
    (natInputShapes := natInputShapes) defn cast opts

/--
Instantiate a Float module with runtime layer parameter initializers.

The initializer plan is indexed by the same `paramShapes` list as the module, so Lean checks that
every parameter has exactly one initializer.
In CUDA mode, supported initializers allocate device buffers directly instead of first constructing
every parameter as a large nested Lean tensor.
-/
def instantiateFloatWithPlan
    {paramShapes inputShapes natInputShapes : List Spec.Shape}
    (defn : ScalarModuleDef paramShapes inputShapes natInputShapes)
    (opts : Options)
    (plan : RuntimeInit.Plan paramShapes) :
    IO (ScalarModule Float paramShapes inputShapes natInputShapes) :=
  _root_.Runtime.Autograd.TorchLean.Module.ScalarModuleDef.instantiateFloatWithRuntimePlan
    (paramShapes := paramShapes) (inputShapes := inputShapes)
    (natInputShapes := natInputShapes) defn opts plan

/--
List-based wrapper for checkpoint/JSON boundaries.

If the caller has a statically known parameter list, prefer
`instantiateFloatWithPlan`; this wrapper checks the list length before applying it.
-/
def instantiateFloatWithInit
    {paramShapes inputShapes natInputShapes : List Spec.Shape}
    (defn : ScalarModuleDef paramShapes inputShapes natInputShapes)
    (opts : Options)
    (inits : List RuntimeInit.FloatInit) :
    IO (ScalarModule Float paramShapes inputShapes natInputShapes) :=
  _root_.Runtime.Autograd.TorchLean.Module.ScalarModuleDef.instantiateFloatWithRuntimeInit
    (paramShapes := paramShapes) (inputShapes := inputShapes)
    (natInputShapes := natInputShapes) defn opts inits

/--
Execution configuration parsed from CLI flags.

Supported flags (parsed by `ExecConfig.parseAndStrip`):
- `--dtype ...` / `--float32-mode ...` (see `TorchLean.Runtime.DType`)
- `--backend eager|compiled`
- `--device auto|cpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external`
- `--show-backend` (print backend capsules when the eager runtime first executes them)
-/
structure ExecConfig where
  /-- Scalar dtype selection. -/
  dtype : _root_.TorchLean.Runtime.DType := .float
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

All dtype/Float32 selection flags are delegated to
`TorchLean.Runtime.DType.parseAndStripWithDefault`.

Default dtype policy:
- If the user does not specify `--dtype` / `--float32-mode` and CUDA is selected, default to
  `dtype=float` (CUDA eager supports `Float` upload/download).
- Otherwise default to `dtype=float32` (executable IEEE-754 float32 semantics).

Named future devices are accepted at parse time so `--show-backend` and planning diagnostics can
explain them. Runtime session creation still rejects devices that this build cannot execute.

The selected device chooses its normal registered kernels. Users do not need a second performance
flag after selecting CUDA or another accelerator.
-/
def parseAndStripWithDefaultDType
    (args : List String) (defaultDType : _root_.TorchLean.Runtime.DType) :
    Except String (ExecConfig × List String) := do
  let (dtype, args1) ←
    _root_.TorchLean.Runtime.DType.parseAndStripWithDefault args defaultDType
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
  let defaultDType : _root_.TorchLean.Runtime.DType :=
    if requestsCuda args then .float else .float32 {}
  parseAndStripWithDefaultDType args defaultDType

/-- Log the chosen execution config to stdout for reproducible runs. -/
def log (cfg : ExecConfig) : IO Unit := do
  _root_.TorchLean.Runtime.DType.log cfg.dtype
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
      ∀ {α : Type}, [_root_.Context α] → [DecidableEq Spec.Shape] → [ToString α] →
        [_root_.TorchLean.Runtime.FromFloat α] →
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
  match (← _root_.TorchLean.Runtime.DType.withRuntime cfg.dtype (fun {α} _ _ _ _ => do
        k (α := α) (_root_.TorchLean.Runtime.ofFloat (α := α)) opts rest
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
      ∀ {α : Type}, [_root_.Context α] → [DecidableEq Spec.Shape] → [ToString α] →
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
      match (← _root_.TorchLean.Runtime.DType.withExec cfg.dtype (fun {α} _ _ _ cast => do
            let m ← _root_.Runtime.Autograd.TorchLean.Module.ScalarModuleDef.instantiateWith
              (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes)
              defn cast opts
            k (α := α) cast m rest
          )) with
      | .ok () => pure ()
      | .error msg => throw <| IO.userError msg

/--
Like `withModule`, but also provides an `_root_.TorchLean.Runtime.FromFloat α` instance (for numeric literals).
-/
def withModuleRuntime
    {paramShapes inputShapes : List Spec.Shape}
    (defn : ScalarModuleDef paramShapes inputShapes)
    (args : List String)
    (k :
      ∀ {α : Type}, [_root_.Context α] → [DecidableEq Spec.Shape] → [ToString α] →
        [_root_.TorchLean.Runtime.FromFloat α] →
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
      match (← _root_.TorchLean.Runtime.DType.withRuntime cfg.dtype (fun {α} _ _ _ _ => do
            let m ← _root_.Runtime.Autograd.TorchLean.Module.ScalarModuleDef.instantiateWith
              (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes)
              defn (_root_.TorchLean.Runtime.ofFloat (α := α)) opts
            k (α := α) m rest
          )) with
      | .ok () => pure ()
      | .error msg => throw <| IO.userError msg

end Module
end TorchLean
