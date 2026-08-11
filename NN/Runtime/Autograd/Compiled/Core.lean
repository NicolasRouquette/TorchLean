/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Runtime.Link

/-!
# Compiled Core

Proof-compiled execution path.

This module exposes the "approach (a)" workflow:
1) Build an executable SSA/DAG graph (`Proofs.Autograd.Algebra.GraphData`).
2) Compile it to a runtime `Tape` with `compileAuxData`.
3) Run `Tape.backwardDenseFrom` / `Tape.backwardDenseAll`.

Notes / trust boundaries:
- If you instantiate `α := Float` or `α := TorchLean.Floats.IEEE754.IEEE32Exec`, you get an
  executable engine,
  but connecting those runs to real hardware semantics is treated as a trusted interface.
- The proof-carrying graph (`Proofs.Autograd.Algebra.Graph`) remains available for backends
  where you can actually discharge algebraic/calc correctness assumptions (e.g. `ℝ`, `ℚ`).

## Main declarations

- `NN.Runtime.Autograd.Compiled.GraphM` is the small authoring DSL for compiled graphs.
- `NN.Runtime.Autograd.Compiled.IRExec` bridges `NN.IR.Graph` to executable graph data.
- `NN.Runtime.Autograd.Compiled.IRExec.Correctness` proves the forward-correctness lemmas.

See also:
- Proof link between compiled tapes and proved graph backprop:
  `NN/Proofs/Autograd/Runtime/Link.lean`
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Compiled

open Spec

open Proofs.Autograd.Algebra

/--
Executable SSA/DAG graph for the proof-compiled pipeline.

This is `Proofs.Autograd.Algebra.GraphData` specialized to:
- `Δ := Unit` (no extra opaque environment threaded through evaluation), and
- the `Runtime.Autograd.Compiled` namespace.
-/
abbrev GraphData (α : Type) (Γ : List Shape) (ss : List Shape) :=
  Proofs.Autograd.Algebra.GraphData α Unit Γ ss

/--
Typed list of tensors whose shapes are tracked in a type-level `List Shape`.

This is the primary "context" representation for the compiled path: graph evaluation produces a
`TList α (Γ ++ ss)` containing all intermediate values.
-/
abbrev TList (α : Type) (ss : List Shape) :=
  Proofs.Autograd.Algebra.TList α ss

/--
Compile an executable `GraphData` into a runtime eager tape.

This is the bridge from the proof-compiled SSA representation to the runtime tape engine:
`Graph.compileAuxData` emits a `Runtime.Autograd.Tape` whose nodes replay the graph and whose
backward closures implement the graph's VJP rules.

PyTorch comparison: conceptually similar to the front half of `torch.compile` / TorchDynamo
(tracing a computation to an IR), except our target is an explicit autograd tape for which we
also maintain proof links.
-/
def compile {α : Type} [DecidableEq Shape]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Γ ss) (x : TList α Γ) :
    Runtime.Autograd.Tape α × TList α (Γ ++ ss) :=
  Proofs.Autograd.Algebra.Graph.compileAuxData (α := α) (Δ := Unit) (Γ := Γ) (ss := ss) g x ()

/--
Convention for the output node id in a compiled tape.

`compile` places the original inputs first (`Γ.length` nodes), then appends one node per element
of `ss`. The final output is therefore at index `Γ.length + ss.length - 1`.

Invariant: callers should only use this when `ss` is nonempty. For defensive code, prefer
`checkedOutId`, which reports an error instead of relying on `Nat`'s saturating subtraction.
-/
def outId {Γ ss : List Shape} : Nat :=
  Γ.length + ss.length - 1

/--
Checked version of `outId`.

Compiled scalar/output programs should always have at least one produced node. This helper makes
that precondition executable, which is friendlier for user-facing compiled APIs and tests.
-/
def checkedOutId {Γ ss : List Shape} : Runtime.Autograd.Result Nat :=
  match ss with
  | [] => .error "compiled graph has no output node"
  | _ :: _ => .ok (outId (Γ := Γ) (ss := ss))

/--
Run reverse-mode backprop on a compiled tape, returning gradients for *all* node ids.

This uses the "total/dense" variant `Tape.backwardDenseAll`, seeding the output gradient with 1
(the usual scalar-loss convention).
-/
def backwardDenseAllFromOutput {α : Type} [Add α] [Zero α] [One α] [DecidableEq Shape]
    {Γ : List Shape} {ss : List Shape}
    (t : Runtime.Autograd.Tape α) :
    Runtime.Autograd.Result (Array (Runtime.AnyTensor α)) := do
  -- Convention: output is the last node id of the compiled tape. For scalar losses, seed with 1.
  let oid ← checkedOutId (Γ := Γ) (ss := ss)
  Runtime.Autograd.Tape.backwardDenseAll (t := t) (outId := oid)
    (seed := Runtime.Autograd.AnyTensor.mk (Tensor.scalar (1 : α)))

/--
Run reverse-mode backprop starting from an explicit seed gradient context.

This is the most general entry point: callers provide a `TList` of initial gradients for every
value in the compiled context `(Γ ++ ss)`, and we run the proof-friendly dense loop
`Tape.backwardDenseFrom`.
-/
def backwardDenseFromSeedCtx {α : Type} [Add α] [DecidableEq Shape]
    {Γ : List Shape} {ss : List Shape}
    (t : Runtime.Autograd.Tape α) (seed : TList α (Γ ++ ss)) :
    Runtime.Autograd.Result (Array (Runtime.AnyTensor α)) :=
  Runtime.Autograd.Tape.backwardDenseFrom (t := t)
    (grads0 := Proofs.Autograd.Algebra.TList.toAnyArray (α := α) (ss := Γ ++ ss) seed)

end Compiled
end Autograd
end Runtime
