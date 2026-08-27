/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

-- shake: keep-all

public import NN.Floats.Float32
public import NN.Runtime.Autograd.TorchLean.Dual
public import NN.Spec.Core.Complex
public import NN.Spec.Core.FloatInstances
public import NN.Spec.Core.Scalar
public import NN.API.CLI

import Mathlib.Algebra.Order.Algebra

/-!
# Runtime Scalar Selection

Runtime commands may choose native binary32, reference IEEE binary32, or complex64.
`ScalarMode` records that semantic choice. `FromFloat` supplies the one additional
operation needed by commands that construct values from Lean binary64 `Float` literals.

The generic runtime dispatcher supports all three modes. The public supervised trainer accepts the
two real modes; complex training needs a real-valued objective, conjugate-aware reverse mode, and a
complex result boundary rather than treating a complex scalar objective as an ordinary real loss.

Model definitions remain polymorphic over the existing `Context α` interface. The runtime mode
does not replace that mathematical interface; it selects a concrete executable scalar type that
satisfies it. Proof-only models such as `ℝ` and rounded-real binary32 are selected directly in
theorems rather than through a command-line flag.
-/

@[expose] public section


namespace TorchLean
namespace Runtime

/--
Conversion from Lean `Float` constants into a selected runtime scalar type.
-/
class FromFloat (α : Type) where
  /-- Convert a Lean binary64 `Float` literal into this runtime scalar backend. -/
  ofFloat : Float → α

/-- Convert a Lean `Float` literal into a TorchLean scalar backend. -/
def ofFloat {α : Type} [FromFloat α] (x : Float) : α :=
  FromFloat.ofFloat x

/-- `Float` values inject into the same type by identity. -/
instance : FromFloat Float where
  ofFloat := id

/-- Round binary64 literals to Lean's native binary32 scalar. -/
instance : FromFloat Float32 where
  ofFloat := Float.toFloat32

/-- Inject binary64 literals into the executable IEEE-754 binary32 backend. -/
instance : FromFloat TorchLean.Floats.IEEE754.IEEE32Exec where
  ofFloat := TorchLean.Floats.IEEE754.IEEE32Exec.ofFloat

/--
Inject binary64 literals into the dual-number backend used by the runtime autograd engine.

We interpret a literal as a primal value with zero tangent/adjoint component.
-/
instance {α : Type} [FromFloat α] [Zero α] :
    FromFloat (_root_.Runtime.Autograd.TorchLean.Dual α) where
  ofFloat x := _root_.Runtime.Autograd.TorchLean.Dual.ofPrimal (ofFloat x)

/-- Inject binary64 literals into TorchLean's complex scalar with zero imaginary part. -/
instance {α : Type} [FromFloat α] [Zero α] : FromFloat (TorchLean.Complex α) where
  ofFloat x := ⟨ofFloat (α := α) x, 0⟩

/-- Allow numeric literals like `0.1` to elaborate to any TorchLean runtime scalar backend. -/
@[default_instance low]
instance {α : Type} [FromFloat α] : OfScientific α where
  ofScientific m s e := ofFloat (Float.ofScientific m s e)

/--
Scalar semantics for runnable executables.

This is a runtime selection mechanism used by example programs; the core library itself is
parametric in the scalar type `α`.

Unlike a PyTorch per-tensor dtype, this choice fixes one scalar semantics for the complete run:

- `.float32` uses Lean's native `Float32` operations on CPU and binary32 CUDA storage on GPU,
- `.ieee32Exec` uses TorchLean's independent bit-level IEEE-754 binary32 reference,
- `.complex64` uses TorchLean's complex scalar with binary32 real and imaginary components. The
  name follows the usual complex-format convention: two 32-bit components form complex64.
-/
inductive ScalarMode where
  | float32
  | ieee32Exec
  | complex64
  deriving Repr, DecidableEq

namespace ScalarMode

/-- Log a short description of the selected scalar semantics. -/
def log : ScalarMode → IO Unit
  | .float32 =>
      IO.println "[TorchLean] scalar: native IEEE-754 binary32"
  | .ieee32Exec =>
      IO.println "[TorchLean] scalar: bit-level IEEE-754 binary32 reference"
  | .complex64 =>
      IO.println "[TorchLean] scalar: complex over executable IEEE-754 binary32"

/-- Parse the value of `--scalar`. -/
def parse (value : String) : Except String ScalarMode :=
  match value with
  | "float32" => pure .float32
  | "ieee32-exec" => pure .ieee32Exec
  | "complex64" => pure .complex64
  | _ => throw s!"unknown --scalar {value} (supported: float32 | ieee32-exec | complex64)"

/-- Parse and remove `--scalar`, using `default` when the flag is absent. -/
def parseAndStripWithDefault (args : List String) (default : ScalarMode) :
    Except String (ScalarMode × List String) := do
  let (value?, rest) ← TorchLean.CLI.takeFlagValueOnce args "scalar"
  match value? with
  | none => pure (default, rest)
  | some value => pure (← parse value, rest)

/-- Parse and remove `--scalar`; native binary32 is the default. -/
def parseAndStrip (args : List String) : Except String (ScalarMode × List String) :=
  parseAndStripWithDefault args .float32

/--
Run `k` under the scalar type selected by `mode`.
-/
def withRuntime
    (mode : ScalarMode)
    (k : ∀ {α : Type}, [Context α] → [DecidableEq Spec.Shape] → [ToString α] →
      [FromFloat α] → IO Unit) :
    IO (Except String Unit) := do
  match mode with
  | .float32 =>
      k (α := Float32)
      pure (.ok ())
  | .ieee32Exec =>
      k (α := TorchLean.Floats.IEEE32Exec)
      pure (.ok ())
  | .complex64 =>
      k (α := TorchLean.Complex TorchLean.Floats.IEEE32Exec)
      pure (.ok ())

end ScalarMode

end Runtime
end TorchLean
