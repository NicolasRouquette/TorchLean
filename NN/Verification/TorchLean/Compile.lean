/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Scalar
public import NN.Verification.TorchLean.Compile.API

/-!
# TorchLean Verification Compiler

Umbrella import for the graph builder and the public compiled-verification API.
-/

@[expose] public section

namespace NN.Verification.TorchLean

open NN.MLTheory.CROWN

/--
Dispatch an executable bound-propagation computation under a dtype with explicit outward endpoint
operations.

Only host `Float` and the bit-level `IEEE32Exec` backend currently meet that interface. Other
runtime and proof-only dtypes are rejected instead of receiving ordinary round-to-nearest
arithmetic through an implicit fallback.
-/
def withBoundDType
    (dtype : TorchLean.Runtime.DType)
    (k : ∀ {α : Type}, [_root_.Context α] → [DecidableEq Spec.Shape] →
      [ToString α] → [_root_.TorchLean.Runtime.FromFloat α] → [BoundOps α] → IO Unit) :
    IO Unit := do
  match dtype with
  | .float =>
      k (α := Float)
  | .float32 { mode := .ieee754Exec } =>
      k (α := TorchLean.Floats.F32 .ieee754Exec)
  | .real | .float32 { mode := .fp32 } =>
      throw <| IO.userError
        "the selected dtype is proof-only and cannot run executable bound propagation"
  | .complex _ =>
      throw <| IO.userError
        "bound propagation currently supports real-valued scalar backends only"

/-- Parse and log a dtype, then run `withBoundDType`. -/
def runWithBoundDType
    (title : String) (args : List String)
    (k : ∀ {α : Type}, [_root_.Context α] → [DecidableEq Spec.Shape] →
      [ToString α] → [_root_.TorchLean.Runtime.FromFloat α] → [BoundOps α] → IO Unit) :
    IO Unit := do
  IO.println s!"=== {title} workflow ==="
  let (dtype, rest) ←
    match TorchLean.Runtime.DType.parseAndStrip args with
    | .ok parsed => pure parsed
    | .error msg => throw <| IO.userError msg
  unless rest.isEmpty do
    throw <| IO.userError s!"unexpected arguments: {rest}"
  TorchLean.Runtime.DType.log dtype
  withBoundDType dtype (fun {α} => k (α := α))

end NN.Verification.TorchLean
