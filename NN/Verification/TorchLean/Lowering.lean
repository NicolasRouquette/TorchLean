/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Scalar
public import NN.Verification.TorchLean.Lowering.API

/-!
# TorchLean Verification Lowering

Umbrella import for the graph builder and the public TorchLean-to-verifier-IR lowering API.
-/

@[expose] public section

namespace NN.Verification.TorchLean

open NN.MLTheory.CROWN

/--
Dispatch an executable bound-propagation computation under scalar semantics with explicit outward endpoint
operations.

Native `Float32` and the bit-level `IEEE32Exec` backend meet that interface. Complex
arithmetic is rejected instead of receiving an invalid real-valued bound implementation.
-/
def withBoundScalar
    (scalar : TorchLean.Runtime.ScalarMode)
    (k : ∀ {α : Type}, [_root_.Context α] → [DecidableEq Spec.Shape] →
      [ToString α] → [_root_.TorchLean.Runtime.FromFloat α] → [BoundOps α] → IO Unit) :
    IO Unit := do
  match scalar with
  | .float32 =>
      k (α := Float32)
  | .ieee32Exec =>
      k (α := TorchLean.Floats.IEEE32Exec)
  | .complex64 =>
      throw <| IO.userError
        "bound propagation currently supports real-valued scalar backends only"

/-- Parse and log scalar semantics, then run `withBoundScalar`. -/
def runWithBoundScalar
    (title : String) (args : List String)
    (k : ∀ {α : Type}, [_root_.Context α] → [DecidableEq Spec.Shape] →
      [ToString α] → [_root_.TorchLean.Runtime.FromFloat α] → [BoundOps α] → IO Unit) :
    IO Unit := do
  IO.println s!"=== {title} workflow ==="
  let (scalar, rest) ←
    match TorchLean.Runtime.ScalarMode.parseAndStrip args with
    | .ok parsed => pure parsed
    | .error msg => throw <| IO.userError msg
  unless rest.isEmpty do
    throw <| IO.userError s!"unexpected arguments: {rest}"
  TorchLean.Runtime.ScalarMode.log scalar
  withBoundScalar scalar (fun {α} => k (α := α))

end NN.Verification.TorchLean
