/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Tape.Algebra.Soundness
public import NN.Runtime.Autograd.Engine.Core
public import NN.Runtime.Autograd.TorchLean.Random

/-!
# GraphM Core

Typed variables, builder state, input binding, constants, random nodes, and detach for the
executable typed graph authoring API.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace TypedGraph
namespace GraphM

open Spec
open Tensor
open Proofs.Autograd.Algebra
open Runtime.Autograd.TorchLean

/--
A typed handle to a value in the growing graph context.

`Var s` carries its expected `Shape` at the type level, while `id` is the runtime index into the
concatenated context `Γ ++ ss`.
-/
structure Var (s : Shape) where
  /--
  Runtime id of the value inside the concatenated context `Γ ++ ss`.

  The shape index on `Var s` is the static guarantee; this numeric id is the executable handle used
  when constructing `Idx` proofs for `GraphData` nodes.
  -/
  id : Nat
deriving Repr

/-!
`GraphM.arg` is correct but a little noisy for examples (you must repeat the index and shape).

`VarList` + `args` give a typed variable layer: `args` returns one `Var` per entry in `Γ`,
in order, without spelling indices.
-/

/--
Dependent list of typed variables, aligned with a list of shapes.

`VarList Γ` contains exactly one `Var s` for each `s ∈ Γ`, in order.
-/
inductive VarList : List Shape → Type where
  | nil : VarList []
  | cons {s : Shape} {ss : List Shape} : Var s → VarList ss → VarList (s :: ss)

namespace VarList

/-- First variable in a nonempty `VarList`. -/
def head {s : Shape} {ss : List Shape} : VarList (s :: ss) → Var s
  | cons v _ => v

/-- Tail variables in a nonempty `VarList`. -/
def tail {s : Shape} {ss : List Shape} : VarList (s :: ss) → VarList ss
  | cons _ vs => vs

end VarList

/--
State for the `GraphM` builder.

It is a sigma pair of:
- the list of intermediate shapes `ss` produced so far, and
- the corresponding executable SSA graph payload `GraphData α Δ Γ ss`.
-/
abbrev StateWith (α : Type) (Δ : Type) (Γ : List Shape) : Type :=
  Σ ss : List Shape, GraphData α Δ Γ ss

/-- Default `GraphM` state with no extra environment (`Δ := Unit`). -/
abbrev State (α : Type) (Γ : List Shape) : Type :=
  StateWith α Unit Γ

/-- `StateT` builder monad for authoring a `GraphData` program, with explicit environment `Δ`. -/
abbrev MWith (α : Type) (Δ : Type) (Γ : List Shape) : Type → Type :=
  StateT (StateWith α Δ Γ) (Runtime.Autograd.Result)

/-- Default `GraphM` builder monad with `Δ := Unit`. -/
abbrev M (α : Type) (Γ : List Shape) : Type → Type :=
  MWith α Unit Γ

/-- Empty builder state (no intermediate nodes yet). -/
def empty {α : Type} {Γ : List Shape} : State α Γ :=
  ⟨[], .nil⟩

/-- Empty builder state for an explicit environment type `Δ`. -/
def emptyWith {α : Type} {Δ : Type} {Γ : List Shape} : StateWith α Δ Γ :=
  ⟨[], .nil⟩

/-- Run a `GraphM` program from an empty state. -/
def run {α : Type} {Γ : List Shape} {β : Type} (m : M α Γ β) :
    Runtime.Autograd.Result (β × State α Γ) :=
  StateT.run m empty

/-- Length of the current context `Γ ++ ss` (inputs + intermediates). -/
def ctxLen {Γ : List Shape} (ss : List Shape) : Nat :=
  (Γ ++ ss).length

/--
Convert a `Var s` into a dependent `Idx (Γ ++ ss) s`.

This performs bounds checking and a runtime shape check, returning a structured error if the
variable points outside the current context or has the wrong shape.
-/
def mkIdx {_α : Type} [DecidableEq Shape] {Γ : List Shape} (ss : List Shape) {s : Shape}
    (v : Var s) : Runtime.Autograd.Result (Idx (Γ ++ ss) s) := by
  let n := v.id
  if h : n < ctxLen (Γ := Γ) ss then
    let i : Fin (ctxLen (Γ := Γ) ss) := ⟨n, h⟩
    let got : Shape := (Γ ++ ss).get i
    if hg : got = s then
      exact .ok ⟨i, hg⟩
    else
      exact .error <|
        s!"typed GraphM: shape mismatch at id={n}: expected {Shape.pretty s}, " ++
          s!"got {Shape.pretty got}"
  else
      exact .error s!"typed GraphM: invalid id={n} for ctxLen={ctxLen (Γ := Γ) ss}"

/--
Append a node to the graph state and return a fresh `Var` pointing to its output.

The returned variable id is `Γ.length + ss.length`, i.e. it points at the newly appended entry.
-/
def push {α : Type} {Δ : Type} {Γ : List Shape} {ss : List Shape} {s : Shape}
    (g : GraphData α Δ Γ ss) (node : NodeData α Δ (Γ ++ ss) s) : MWith α Δ Γ (Var s) := do
  set (σ := StateWith α Δ Γ) ⟨ss ++ [s], .snoc g node⟩
  pure { id := Γ.length + ss.length }

/--
Reference an input variable from the initial context `Γ`.

This checks that the provided index is within bounds and that the requested shape matches the
shape at that position in `Γ`.

PyTorch comparison: this is like naming a graph input tensor in a traced graph.
-/
def arg {α : Type} {Δ : Type} [DecidableEq Shape] {Γ : List Shape} (i : Nat) (s : Shape) :
    MWith α Δ Γ (Var s) := do
  if h : i < Γ.length then
    let fin : Fin Γ.length := ⟨i, h⟩
    let got : Shape := Γ.get fin
    if _hg : got = s then
      pure { id := i }
    else
      throw <|
        s!"typed GraphM: input shape mismatch at i={i}: expected " ++
          s!"{Shape.pretty s}, got {Shape.pretty got}"
  else
    throw s!"typed GraphM: input index out of bounds i={i} (Γ.length={Γ.length})"

/-- Pure helper to build `VarList Γ` starting at a given id offset. -/
def argsAux : (Γ : List Shape) → Nat → VarList Γ
  | [], _i => .nil
  | _s :: ss, i => .cons { id := i } (argsAux ss (i + 1))

/--
Return one `Var` per entry of `Γ`, in order.

This is the canonical argument environment for a graph with input context `Γ`.
-/
def args {α : Type} {Δ : Type} {Γ : List Shape} : MWith α Δ Γ (VarList Γ) := do
  pure (argsAux Γ 0)

/--
Embed a constant tensor as a node in the typed graph.

This node has no input dependencies (`vjp = 0`, `jvp = 0`), i.e. it is treated as a constant
with respect to the graph inputs.

PyTorch comparison: a constant literal captured into a traced/typed graph.
-/
def const {α : Type} {Δ : Type} [Zero α] {Γ : List Shape} {s : Shape} (t : Tensor α s) :
    MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun _ctx _d => t
      jvp := fun _ctx _dctx _d => fill (0 : α) s
      vjp := fun _ctx _d _δ => TList.zero (α := α) (ss := Γ ++ ss) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/-- Deterministic `U[0,1)` tensor generator (seeded, pure). -/
def randUniform {α : Type} [Context α] {Δ : Type} {Γ : List Shape} {s : Shape} (seed : Nat) :
    MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let counter := ss.length
  let key := TorchLean.Random.keyOf seed counter
  let t : Tensor α s := TorchLean.Random.uniform (α := α) key (s := s)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun _ctx _d => t
      jvp := fun _ctx _dctx _d => fill (0 : α) s
      vjp := fun _ctx _d _δ => TList.zero (α := α) (ss := Γ ++ ss) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Deterministic `{0,1}` mask generator (seeded, pure).

Note: for differentiation purposes, this node is treated as a **stop-gradient** op:
`jvp = 0` and `vjp = 0` for all inputs (including `keepProb`). This matches the intended use in
dropout where the probability is a hyperparameter (not differentiated), while keeping execution
deterministic in `.typedGraph` execution.
-/
def bernoulliMask {α : Type} [Context α] [DecidableEq Shape]
    {Δ : Type} {Γ : List Shape} {s : Shape}
    (keepProb : Var Shape.scalar) (seed : Nat) :
    MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let counter := ss.length
  let key := TorchLean.Random.keyOf seed counter
  let ikp ← liftM (mkIdx (_α := α) (Γ := Γ) ss keepProb)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d =>
        let kpT := getIdx (α := α) (xs := ctx) ikp
        let kp : α :=
          match kpT with
          | Tensor.scalar v => v
        TorchLean.Random.mask (α := α) key kp (s := s)
      jvp := fun _ctx _dctx _d => fill (0 : α) s
      vjp := fun _ctx _d _δ => TList.zero (α := α) (ss := Γ ++ ss) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

/--
Stop-gradient boundary.

Forward semantics: identity (`detach(x) = x`).
Backward semantics: no gradient flows to `x` (treated as constant w.r.t. the graph inputs).
-/
def detach {α : Type} [Context α] [DecidableEq Shape]
    {Δ : Type} {Γ : List Shape} {s : Shape}
    (x : Var s) : MWith α Δ Γ (Var s) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) s :=
    { forward := fun ctx _d => getIdx (α := α) (xs := ctx) ix
      jvp := fun _ctx _dctx _d => fill (0 : α) s
      vjp := fun _ctx _d _δ => TList.zero (α := α) (ss := Γ ++ ss) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := s) g node

end GraphM
end TypedGraph
end Autograd
end Runtime
