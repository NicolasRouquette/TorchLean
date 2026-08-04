/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

-- shake: keep-all

public import NN.Floats.Float32
public import NN.Runtime.Autograd.TorchLean.Dual
public import NN.Runtime.Scalar
public import NN.Spec.Core.Complex
public import NN.Spec.Core.FloatInstances
public import NN.Spec.Core.Scalar
public import NN.API.CLI

import Mathlib.Algebra.Order.Algebra

/-!
# Runtime Scalar Selection

Runtime commands may choose Lean `Float`, executable IEEE binary32, complex binary32, or exact
real-number semantics. `DType` records that choice. `FromFloat` supplies the one additional
operation needed by commands that construct values from host `Float` literals.

Model definitions remain polymorphic over the existing `Context α` interface. The runtime dtype
does not replace that mathematical interface; it selects a concrete scalar type that satisfies it.
-/

@[expose] public section


namespace TorchLean
namespace Runtime

/--
Conversion from host `Float` constants into a selected runtime scalar type.
-/
class FromFloat (α : Type) where
  /-- Convert a host `Float` literal into this runtime scalar backend. -/
  ofFloat : Float → α

/-- Generic host-float injection for TorchLean scalar backends. -/
def ofFloat {α : Type} [FromFloat α] (x : Float) : α :=
  FromFloat.ofFloat x

/-- `Float` is already the host literal type, so injection is identity. -/
instance : FromFloat Float where
  ofFloat := id

/-- Inject host `Float` literals into the executable IEEE-754 binary32 backend. -/
instance : FromFloat TorchLean.Floats.IEEE754.IEEE32Exec where
  ofFloat := TorchLean.Floats.IEEE754.IEEE32Exec.ofFloat

/--
Inject host literals into the dual-number backend used by the runtime autograd engine.

We interpret a literal as a primal value with zero tangent/adjoint component.
-/
instance {α : Type} [FromFloat α] [Zero α] :
    FromFloat (_root_.Runtime.Autograd.TorchLean.Dual α) where
  ofFloat x := _root_.Runtime.Autograd.TorchLean.Dual.ofPrimal (ofFloat x)

/-- Inject host literals into TorchLean's parametric complex scalar (imaginary part defaults to 0). -/
instance {α : Type} [FromFloat α] [Zero α] : FromFloat (TorchLean.Complex α) where
  ofFloat x := ⟨ofFloat (α := α) x, 0⟩

/-- Allow numeric literals like `0.1` to elaborate to any TorchLean runtime scalar backend. -/
@[default_instance low]
instance {α : Type} [FromFloat α] : OfScientific α where
  ofScientific m s e := ofFloat (Float.ofScientific m s e)

/--
Configuration for the `float32` dtype option.

We support both:
- proof-only float32 semantics (`mode = .fp32`, noncomputable), and
- executable IEEE-754 float32 semantics (`mode = .ieee754Exec`).
-/
structure Float32Config where
  /-- Which float32 semantics backend to use. -/
  mode : TorchLean.Floats.Float32Mode := .ieee754Exec
  deriving Repr, DecidableEq

/--
Scalar type choice for runnable executables.

This is a runtime selection mechanism used by example programs; the core library itself is
parametric in the scalar type `α`.

### PyTorch Mapping

This corresponds loosely to choosing `dtype=` in PyTorch, but with additional "proof-only"
variants:
- `.float` uses Lean's builtin `Float` (executable, but its IEEE-754 behavior is trusted),
- `.float32` uses TorchLean's float32 model (either proof-only or executable),
- `.complex` uses TorchLean's parametric complex scalar over a float32 backend,
- `.real` uses `ℝ` (proof-only; not executable).
-/
inductive DType where
  | float
  | float32 (cfg : Float32Config := {})
  | complex (cfg : Float32Config)
  | real
  deriving Repr, DecidableEq

namespace DType

/-- Whether this dtype can be used in an executable (`IO`/`#eval`) context. -/
def isExecutable : DType → Bool
  | .float => true
  | .real => false
  | .float32 { mode := .fp32 } => false
  | .float32 _ => true
  | .complex { mode := .fp32 } => false
  | .complex _ => true

/-- Log a short description of the chosen dtype to stdout. -/
def log : DType → IO Unit
  | .float =>
      IO.println "[TorchLean] dtype: Float (Lean `Float`, trusted runtime semantics)"
  | .real =>
      IO.println "[TorchLean] dtype: ℝ (proof-only; not executable)"
  | .float32 cfg =>
      TorchLean.Floats.logFloat32Mode cfg.mode
  | .complex cfg => do
      IO.println "[TorchLean] dtype: Complex Float32 (TorchLean.Complex over Float32)"
      TorchLean.Floats.logFloat32Mode cfg.mode

namespace Internal

/-- Parse a `float32` mode selector string into a `Float32Mode`. -/
def parseFloat32Mode (v : String) : Except String TorchLean.Floats.Float32Mode := do
  if v == "fp32" then
    pure .fp32
  else if v == "ieee32" || v == "ieee754" || v == "ieee32exec" || v == "ieee754exec" then
    pure .ieee754Exec
  else
    throw s!"unknown float32 mode {v} (supported: fp32 | ieee754exec)"

/-- Parse a dtype selector string into a `DType`. -/
def parseDTypeValue (v : String) : Except String DType := do
  if v == "float" then
    pure .float
  else if v == "float32" || v == "f32" then
    pure (.float32 {})
  else if v == "complex" || v == "complex32" || v == "cfloat32" || v == "c32" then
    pure (.complex {})
  else if v == "real" || v == "reals" || v == "ℝ" then
    pure .real
  else if v == "fp32" || v == "ieee32" || v == "ieee754" || v == "ieee32exec" || v == "ieee754exec"
    then
    pure (.float32 { mode := (← parseFloat32Mode v) })
  else if v.startsWith "float32:" then
    let m ← parseFloat32Mode ((v.drop 8).toString)
    pure (.float32 { mode := m })
  else if v.startsWith "f32:" then
    let m ← parseFloat32Mode ((v.drop 4).toString)
    pure (.float32 { mode := m })
  else if v.startsWith "complex:" then
    let m ← parseFloat32Mode ((v.drop 8).toString)
    pure (.complex { mode := m })
  else if v.startsWith "c32:" then
    let m ← parseFloat32Mode ((v.drop 4).toString)
    pure (.complex { mode := m })
  else
    throw s!"unknown --dtype {v} (supported: float | real | complex | fp32 | ieee754exec | f32:<mode> | c32:<mode>)"

end Internal

open Internal

/--
Parse and remove dtype flags from CLI arguments, using `default` when no dtype flags are provided.

This is the same parsing logic as `parseAndStrip`, but it lets higher-level runners choose a
different default dtype depending on context (e.g. CUDA eager requires `Float`).
-/
def parseAndStripWithDefault (args : List String) (default : DType) :
    Except String (DType × List String) := do
  let (dtypeV?, args1) ← TorchLean.CLI.takeFlagValueOnce args "dtype"
  let (modeV?, args2) ← TorchLean.CLI.takeFlagValueOnce args1 "float32-mode"
  match dtypeV?, modeV? with
  | some _, some _ =>
      throw "--dtype and --float32-mode are mutually exclusive (pick one)"
  | none, none =>
      pure (default, args2)
  | some dv, none =>
      pure (← parseDTypeValue dv, args2)
  | none, some mv =>
      let m ← parseFloat32Mode mv
      pure (.float32 { mode := m }, args2)

/--
Parse and remove dtype flags from CLI arguments.

Supported flags:
- `--dtype <value>` or `--dtype=<value>`
- `--float32-mode <mode>` or `--float32-mode=<mode>`

The two flags are mutually exclusive.
-/
def parseAndStrip (args : List String) : Except String (DType × List String) := do
  parseAndStripWithDefault args (.float32 {})

/--
Run `k` under the scalar type selected by `dt`.

If `dt` is proof-only, this returns an error rather than trying to execute noncomputable code.
-/
def withRuntime
    (dt : DType)
    (k : ∀ {α : Type}, [Context α] → [DecidableEq Spec.Shape] → [ToString α] →
      [FromFloat α] → IO Unit) :
    IO (Except String Unit) := do
  match dt with
  | .float =>
      k (α := Float)
      pure (.ok ())
  | .real =>
      pure (.error
        "dtype=real is proof-only (noncomputable); use it in theorems, not in executables")
  | .complex { mode := .fp32 } =>
      pure (.error
        "dtype=complex:fp32 is proof-only (noncomputable); use it in theorems/verification proofs")
  | .complex { mode := .ieee754Exec } =>
      k (α := TorchLean.Complex (TorchLean.Floats.F32 .ieee754Exec))
      pure (.ok ())
  | .float32 { mode := .fp32 } =>
      pure (.error
        "float32-mode=fp32 is proof-only (noncomputable); use it in theorems/verification proofs")
  | .float32 { mode := .ieee754Exec } =>
      k (α := TorchLean.Floats.F32 .ieee754Exec)
      pure (.ok ())

/--
Run `k` under the scalar type selected by `dt`, passing an explicit cast function `Float → α`.

This is a convenient shape for executables that construct tensors from `Float` lists.
-/
def withExec
    (dt : DType)
    (k : ∀ {α : Type}, [Context α] → [DecidableEq Spec.Shape] → [ToString α] → (Float →
      α) → IO Unit) :
    IO (Except String Unit) :=
  withRuntime dt (fun {α} _ _ _ _ => k (α := α) (ofFloat (α := α)))

end DType

end Runtime
end TorchLean
