/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Runtime.Link

/-!
# Typed Graph Core

Shape-indexed graph execution and its runtime-tape lowering.

This module exposes the "approach (a)" workflow:
1) Build an executable SSA/DAG graph (`Proofs.Autograd.Algebra.GraphData`).
2) Lower it to a runtime `Tape` with `Graph.lowerGraphDataToTape`.
3) Run `Tape.backwardDenseFrom` / `Tape.backwardDenseAll`.

`GraphData` is executable data, not a derivative-correctness certificate. The lowering theorem
shows that the tape implements `GraphData.backpropAllCtx`. To prove that this operation is the
adjoint of the graph JVP, construct the proof-carrying `Proofs.Autograd.Algebra.Graph`, whose nodes
include their local adjointness laws.

Notes / trust boundaries:
- If you instantiate `α := Float` or `α := TorchLean.Floats.IEEE754.IEEE32Exec`, you get an
  executable engine,
  but connecting those runs to real hardware semantics is treated as a trusted interface.
- The proof-carrying graph (`Proofs.Autograd.Algebra.Graph`) is available for backends
  where you can actually discharge algebraic/calc correctness assumptions (e.g. `ℝ`, `ℚ`).

## Main declarations

- `NN.Runtime.Autograd.TypedGraph.GraphM` is the small authoring DSL for typed graphs.
- `NN.Runtime.Autograd.IRExec` bridges `NN.IR.Graph` to executable graph data.
- `NN.Runtime.Autograd.IRExec.Correctness` proves the forward-correctness lemmas.

See also:
- Equality between lowered-tape backprop and executable graph backprop:
  `NN/Proofs/Autograd/Runtime/Link.lean`
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace TypedGraph

open Spec

open Proofs.Autograd.Algebra

/--
Executable SSA/DAG graph for typed graph execution.

This is `Proofs.Autograd.Algebra.GraphData` specialized to:
- `Δ := Unit` (no extra opaque environment threaded through evaluation), and
- the `Runtime.Autograd.TypedGraph` namespace.
-/
abbrev GraphData (α : Type) (Γ : List Shape) (ss : List Shape) :=
  Proofs.Autograd.Algebra.GraphData α Unit Γ ss

/--
Typed list of tensors whose shapes are tracked in a type-level `List Shape`.

This is the primary context representation for typed graph execution: graph evaluation produces a
`TList α (Γ ++ ss)` containing all intermediate values.
-/
abbrev TList (α : Type) (ss : List Shape) :=
  Proofs.Autograd.Algebra.TList α ss

/--
Lower an executable `GraphData` into a runtime tape.

This is the bridge from the shape-indexed SSA representation to the runtime tape engine:
`Graph.lowerGraphDataToTape` emits a `Runtime.Autograd.Tape` whose nodes replay the graph and whose
backward closures implement the graph's VJP rules.

The graph remains the persistent artifact; the tape contains the runtime closures needed for one
execution and reverse pass.
-/
def lowerToTape {α : Type} [DecidableEq Shape]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Γ ss) (x : TList α Γ) :
    Runtime.Autograd.Tape α × TList α (Γ ++ ss) :=
  Proofs.Autograd.Algebra.Graph.lowerGraphDataToTape (α := α) (Δ := Unit) (Γ := Γ) (ss := ss) g x ()

/--
Run reverse-mode backpropagation from a typed output reference and cotangent seed.

The output may be an original graph input or any recorded node; it need not be the final node.
The full typed context is seeded explicitly, matching `GraphData.backpropCtx` and the tape-lowering
theorem. In particular, this path does not rely on reachability pruning or an unstated assumption
that every stored VJP maps a zero cotangent to zero.
-/
def backwardDenseAllFrom {α : Type} [Add α] [Zero α] [DecidableEq Shape]
    {Γ : List Shape} {ss : List Shape} {τ : Shape}
    (t : Runtime.Autograd.Tape α) (output : Idx (Γ ++ ss) τ) (seed : Tensor α τ) :
    Runtime.Autograd.Result (Array (Spec.PackedTensor α)) :=
  Runtime.Autograd.Tape.backwardDenseFrom (t := t)
    (grads0 := Proofs.Autograd.Algebra.TList.toPackedArray
      (Proofs.Autograd.Algebra.TList.single output seed))

/--
Run reverse-mode backprop starting from an explicit seed gradient context.

This is the most general entry point: callers provide a `TList` of initial gradients for every
value in the typed graph context `(Γ ++ ss)`, and we run the dense loop
`Tape.backwardDenseFrom`.
-/
def backwardDenseFromSeedCtx {α : Type} [Add α] [DecidableEq Shape]
    {Γ : List Shape} {ss : List Shape}
    (t : Runtime.Autograd.Tape α) (seed : TList α (Γ ++ ss)) :
    Runtime.Autograd.Result (Array (Spec.PackedTensor α)) :=
  Runtime.Autograd.Tape.backwardDenseFrom (t := t)
    (grads0 := Proofs.Autograd.Algebra.TList.toPackedArray (α := α) (ss := Γ ++ ss) seed)

/--
Lowering a typed graph to the runtime tape preserves reverse mode from any typed output reference.

The result covers outputs that are inputs or intermediate nodes, not only the final recorded node.
It states fidelity to the executable VJP stored in `GraphData`; derivative correctness requires the
separate local laws carried by `Proofs.Autograd.Algebra.Node`.
-/
theorem backwardDenseAllFrom_lowerToTape_eq_backpropAllCtx
    {α : Type} [CommSemiring α] [DecidableEq Shape]
    {Γ : List Shape} {ss : List Shape} {τ : Shape}
    (g : GraphData α Γ ss) (x : TList α Γ) (output : Idx (Γ ++ ss) τ)
    (seed : Tensor α τ) :
    backwardDenseAllFrom (lowerToTape g x).1 output seed =
      .ok
        (Proofs.Autograd.Algebra.TList.toPackedArray
          (Proofs.Autograd.Algebra.GraphData.backpropAllCtx
            g x () (Proofs.Autograd.Algebra.TList.single output seed))) := by
  simpa [backwardDenseAllFrom, lowerToTape] using
    (Proofs.Autograd.Algebra.Graph.backwardDenseFrom_lowerGraphDataToTape_eq_backpropAllCtx
      (α := α) (Δ := Unit) (Γ := Γ) (ss := ss) g x ()
      (Proofs.Autograd.Algebra.TList.single output seed))

end TypedGraph
end Autograd
end Runtime
