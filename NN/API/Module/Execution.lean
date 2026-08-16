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

export _root_.Runtime.Autograd.Torch (ExecutionMode Options)

/-!
# Module Execution

This file connects typed module definitions to executable scalar modules. It provides state
initialization, execution settings, and helpers that select a scalar type and device from command
line arguments.

`ObjectiveDef` describes a scalar objective together with model state and input shapes.
Instantiating it produces a mutable `Objective` that can evaluate the objective, return explicit
state gradients, and update trainable entries. The shape lists remain part of both types, so
construction and execution use the same state ordering.
-/

export _root_.Runtime.Autograd.TorchLean.Module
  (Evaluator ObjectiveEvaluator ObjectiveDef Objective)
namespace RuntimeInit
export _root_.Runtime.Autograd.TorchLean.Module.RuntimeInit
  (FloatInit Plan xavierUniformForShape kaimingUniformForShape xavierLinearWeight
   kaimingLinearWeight)
end RuntimeInit
export _root_.Runtime.Autograd.TorchLean.Module.Objective
  (create loss lossAndGradState gradState sgdStepWithLoss sgdStep initOptimizer optimizerStep
   optimizerStepWithLoss state loadState trainSGD trainWithOptimizer meanLoss)
export _root_.Runtime.Autograd.TorchLean.Module.Evaluator (evaluatePacked withState)
export _root_.Runtime.Autograd.TorchLean.Module.ObjectiveDef
  (evaluatorWithState lossWithState instantiate instantiateFloat64 instantiateWithPlan
   instantiateWithInit)

/--
Instantiate an `ObjectiveDef` under explicit Torch options such as `execution` and `device`.

The supplied options are passed unchanged to module construction, including the selected device and
execution strategy.
-/
def instantiateAs
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    {stateShapes inputShapes natInputShapes : List Spec.Shape}
    (defn : ObjectiveDef stateShapes inputShapes natInputShapes)
    (cast : Float → α) (opts : Options) :
    IO (Objective α stateShapes inputShapes natInputShapes) :=
  _root_.Runtime.Autograd.TorchLean.Module.ObjectiveDef.instantiateWith
    (α := α) (stateShapes := stateShapes) (inputShapes := inputShapes)
    (natInputShapes := natInputShapes) defn cast opts

/--
Execution configuration parsed from CLI flags.

Supported flags (parsed by `ExecConfig.parseAndStrip`):
- `--scalar float32|ieee32-exec|complex64` (see `TorchLean.Runtime.ScalarMode`)
- `--execution eager|typed-graph`
- `--device auto|cpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external`
- `--show-backend` (print backend capsules when the eager runtime first executes them)
-/
structure ExecConfig where
  /-- Scalar semantics for this execution. -/
  scalar : _root_.TorchLean.Runtime.ScalarMode := .float32
  /-- Execution mode selection. -/
  execution : ExecutionMode := .eager
  /-- Requested execution device. -/
  device : NN.Backend.Device := .cpu
  /-- Print each backend capsule when the eager runtime first executes it. -/
  showBackend : Bool := false
  deriving Repr, DecidableEq

namespace ExecConfig

/-- Parse an execution-mode selector. -/
def parseExecutionMode (v : String) : Except String ExecutionMode := do
  if v == "eager" then
    pure .eager
  else if v == "typed-graph" then
    pure .typedGraph
  else
    throw s!"unknown --execution {v} (supported: eager | typed-graph)"

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
- `--execution eager|typed-graph` (at most once),
- `--device auto|cpu|cuda|rocm|metal|wasm|tpu|trainium|custom|external`,
- `--show-backend` (boolean flag; removed from `rest`).

Scalar selection is delegated to
`TorchLean.Runtime.ScalarMode.parseAndStripWithDefault`.

The default is native `.float32` on both CPU and CUDA. The independent bit-level implementation is
available explicitly as `.ieee32Exec` for reference execution and proof-oriented checks.

Named future devices are accepted at parse time so `--show-backend` and planning diagnostics can
explain them. Runtime session creation still rejects devices that this build cannot execute.

The selected device chooses its normal registered kernels. Users do not need a second performance
flag after selecting CUDA or another accelerator.
-/
def parseWithScalar
    (args : List String) (defaultScalar : _root_.TorchLean.Runtime.ScalarMode) :
    Except String (ExecConfig × List String) := do
  let (scalar, args1) ←
    _root_.TorchLean.Runtime.ScalarMode.parseAndStripWithDefault args defaultScalar
  let (execution, args2) ←
    _root_.TorchLean.CLI.takeParsedFlagDefault args1 "execution" "eager" parseExecutionMode
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
    scalar := scalar,
    execution := execution,
    device := device,
    showBackend := showBackend
  }, rest)

/-- Convert a parsed CLI execution config to runtime `Options`. -/
def toOptions (cfg : ExecConfig) (seed : Nat := 0) : Except String Options := do
  let _ ← match NN.Backend.BackendProfile.maintainedForDevice? cfg.device with
    | some profile => pure profile
    | none =>
        throw s!"device `{cfg.device.cliName}` has no maintained runtime profile; use a programmatic backend profile"
  pure
    { execution := cfg.execution
      device := cfg.device
      seed := seed
      backendProfile? := none
      showBackend := cfg.showBackend }

/-- Parse CLI flags with the standard TorchLean default scalar policy. -/
def parseAndStrip (args : List String) : Except String (ExecConfig × List String) := do
  parseWithScalar args .float32

/-- Log the chosen execution config to stdout for reproducible runs. -/
def log (cfg : ExecConfig) : IO Unit := do
  _root_.TorchLean.Runtime.ScalarMode.log cfg.scalar
  IO.println s!"[TorchLean] execution: {reprStr cfg.execution}"
  IO.println s!"[TorchLean] device: {cfg.device.cliName}"

end ExecConfig

/--
Parse runtime flags (`--scalar`, `--execution`, `--device`, `--show-backend`) and choose an executable
scalar `α`, then call `k` with:
- `cast : Float → α` for building inputs from literals
- `opts : Options` selecting the execution mode and kernel profile
- `rest : List String` containing the remaining CLI arguments

This is useful for scripts that need to build a dataset/loader (and maybe determine shapes/batch
sizes) before instantiating a concrete `ObjectiveDef`.
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
  match (← _root_.TorchLean.Runtime.ScalarMode.withRuntime cfg.scalar (fun {α} _ _ _ _ => do
        k (α := α) (_root_.TorchLean.Runtime.ofFloat (α := α)) opts rest
      )) with
  | .ok () => pure ()
  | .error msg => throw <| IO.userError msg

/--
Instantiate an `ObjectiveDef` under CLI runtime flags, then call a continuation.

The continuation receives both the selected scalar instance and an explicit `Float → α` conversion
for input data authored with host `Float` values.
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
  let (cfg, rest) ←
    match ExecConfig.parseAndStrip args with
    | .ok v => pure v
    | .error msg => throw <| IO.userError msg
  ExecConfig.log cfg
  let opts ← match ExecConfig.toOptions cfg with
    | .ok opts => pure opts
    | .error msg => throw <| IO.userError msg
  opts.validateForExecution
  match cfg.scalar with
  | .float32 =>
      let m ← _root_.Runtime.Autograd.TorchLean.Module.ObjectiveDef.instantiateWith
        (α := Float32) (stateShapes := stateShapes) (inputShapes := inputShapes)
        defn Float.toFloat32 opts
      k (α := Float32) Float.toFloat32 m rest
  | _ =>
      if (cfg.device == .cuda) then
        throw <| IO.userError "torch: CUDA module execution currently requires --scalar float32"
      match (← _root_.TorchLean.Runtime.ScalarMode.withRuntime cfg.scalar (fun {α} _ _ _ _ => do
            let cast := _root_.TorchLean.Runtime.ofFloat (α := α)
            let m ← _root_.Runtime.Autograd.TorchLean.Module.ObjectiveDef.instantiateWith
              (α := α) (stateShapes := stateShapes) (inputShapes := inputShapes)
              defn cast opts
            k (α := α) cast m rest
          )) with
      | .ok () => pure ()
      | .error msg => throw <| IO.userError msg

end Module
end TorchLean
