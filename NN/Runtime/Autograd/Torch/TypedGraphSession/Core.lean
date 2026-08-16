/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core
import Mathlib.Algebra.Order.Algebra

/-!
# TypedGraphSession

Imperative session for shape-indexed graph recording.

Background:
- `Runtime.Autograd.TorchLean.Session` provides a unified imperative API for eager and typed graph
  execution.
- `Proofs.Autograd.Algebra.GraphData` is the executable, shape-indexed SSA/DAG representation, and
  `NN/Proofs/Autograd/Runtime/Link.lean` proves that running the runtime reverse-mode loop on the
  lowered tape matches `GraphData.backpropAllCtx`.

This file provides a session-style API that records `GraphData` as operations are called, then
runs the standard runtime tape loop on the lowered graph.

Key guarantee (pure theorem, no `IO` reasoning needed):
- If the session snapshot is `(g, x)`, then `Tape.backwardDenseFrom (lowerGraphDataToTape g x)` equals
  `GraphData.backpropAllCtx g x` (via `backwardDenseFrom_lowerGraphDataToTape_eq_backpropAllCtx`).

Practical note:
- This session enforces a simple invariant: **all leaf tensors are created before any op node**.
  This matches the standard training pattern (reset → add leaves → forward → backward).
- `const` is available as a graph node, so you can still introduce literal constants mid-graph.
- This is the typed graph variant used by `TorchLean.Session` when `opts.execution := .typedGraph`.
  Derivative correctness requires the separate local laws stored by
  `Proofs.Autograd.Algebra.Node`; recording `GraphData` alone does not establish those laws.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Torch

open Spec
open Tensor

namespace Internal

/--
Convenience: turn a `Result α` into `IO α` by throwing `IO.userError` on `.error`.

This mirrors the common pattern in the eager runtime front-end (`Torch.Core`).
-/
abbrev okOrThrow {α : Type} : Runtime.Autograd.Result α → IO α :=
  Runtime.Autograd.okOrThrow

/-- Non-differentiable external environment for the graph: a small array of `Nat` inputs. -/
abbrev NatEnv : Type := Array Nat

/-- Runtime metadata attached to one typed-graph leaf. -/
structure LeafMetadata where
  /-- Optional name used by tape diagnostics. -/
  name : Option String
  /-- Whether reverse mode accumulates a gradient for this leaf. -/
  requiresGrad : Bool

/-- Internal typed graph state: executable `GraphData` together with its leaf values. -/
structure TypedGraphSessionState (α : Type) where
  /-- Leaf shapes (inputs/parameters), in creation order. -/
  Γ : List Shape
  /-- Leaf values, aligned with `Γ`. -/
  x : _root_.Proofs.Autograd.Algebra.TList α Γ
  /-- Runtime metadata aligned with the leaves in `Γ`. -/
  leafMetadata : Array LeafMetadata := #[]
  /-- Non-differentiable external inputs (e.g. class labels/indices). -/
  nat : NatEnv
  /-- Internal node shapes, in creation order. -/
  ss : List Shape
  /-- SSA/DAG graph nodes (one per entry in `ss`). -/
  g : _root_.Proofs.Autograd.Algebra.GraphData α NatEnv Γ ss

namespace TypedGraphSessionState

/-- Empty session state: no leaves, no nodes, empty nat-environment. -/
def empty {α : Type} : TypedGraphSessionState α :=
  { Γ := []
    x := .nil
    leafMetadata := #[]
    nat := #[]
    ss := []
    g := .nil }

end TypedGraphSessionState

/--
`TypedGraphSession` is an imperative session that records executable `GraphData` as it runs.

Operations are called imperatively, but the resulting graph is explicit and shape-indexed. After
lowering, the runtime tape backward loop is provably equal to `GraphData.backpropAllCtx`; this is an
implementation-equivalence result, distinct from proving each stored VJP correct.
-/
structure TypedGraphSession (α : Type) where
  /-- Session options shared with the eager front-end. -/
  opts : Options
  /-- Mutable executable graph snapshot. -/
  st : IO.Ref (TypedGraphSessionState α)
  /-- Map from graph leaf ids to mutable parameter objects. -/
  paramsByLeaf : IO.Ref (Std.HashMap Nat (AnyParam α))

namespace TypedGraphSession

/--
Create a new typed graph session.

This allocates `IO.Ref`s for the session snapshot (`TypedGraphSessionState`) and the map from leaf
identifiers to parameters. Call `resetTape` to begin a new graph recording phase.
-/
def new {α : Type} (opts : Options := {}) : IO (TypedGraphSession α) := do
  let st ← IO.mkRef (TypedGraphSessionState.empty (α := α))
  let paramsByLeaf ← IO.mkRef (Std.HashMap.emptyWithCapacity)
  pure { opts := opts, st := st, paramsByLeaf := paramsByLeaf }

/--
Reset the session to an empty snapshot.

Important invariant: this session requires that **all leaves are created before any op node**.
`resetTape` is the intended boundary between training steps/forwards.
-/
def resetTape {α : Type} (s : TypedGraphSession α) : IO Unit := do
  s.st.set (TypedGraphSessionState.empty (α := α))
  s.paramsByLeaf.set (Std.HashMap.emptyWithCapacity)

/--
Create a mutable parameter object (not yet part of the recorded graph).

To use the parameter in the recorded graph, call `use`, which reads its current value and records
it as a *leaf* in `Γ`.
PyTorch comparison: analogous to creating a `torch.nn.Parameter` and then using it in a forward.
-/
def param {α : Type} (s : TypedGraphSession α) {sh : Shape}
  (init : Tensor α sh) (name : Option String := none) (requiresGrad : Option Bool := none) :
  IO (Param α sh) := do
  let r ← IO.mkRef init
  let cudaValue ← IO.mkRef (none : Option Runtime.Autograd.Cuda.AnyBuffer)
  let hostCurrent ← IO.mkRef true
  pure { name := name
         value := r
         cudaValue := cudaValue
         hostCurrent := hostCurrent
         requiresGrad := requiresGrad.getD s.opts.requiresGradByDefault }

/--
Enforce the session invariant: leaves must be created before any op node.

This matches the usual training pattern: `resetTape → add leaves → forward ops → backward`.
-/
def ensureNoNodes {α : Type} (st : TypedGraphSessionState α) : IO Unit := do
  match st.ss with
  | [] => pure ()
  | _ :: _ =>
      throw <| IO.userError
        ("torch(TypedGraphSession): cannot add a new leaf after graph nodes have been " ++
          "created (resetTape first)")

/--
Record a new differentiable leaf tensor in the session context `Γ`.

This is the primitive used by `use` (parameters) and `input` (external inputs).
-/
def addLeaf {α : Type} (s : TypedGraphSession α) {sh : Shape} (v : Tensor α sh)
    (name : Option String) (requiresGrad : Bool) :
    IO (TensorRef α sh) := do
  let st0 ← s.st.get
  ensureNoNodes st0
  let id := st0.Γ.length
  let Γ' := st0.Γ ++ [sh]
  let x' : _root_.Proofs.Autograd.Algebra.TList α Γ' :=
    _root_.Proofs.Autograd.Algebra.TList.snoc (α := α) (ss := st0.Γ) (τ := sh) st0.x v
  -- No nodes yet, so the graph stays `nil`.
  let st1 : TypedGraphSessionState α :=
    { Γ := Γ'
      x := x'
      leafMetadata := st0.leafMetadata.push { name, requiresGrad }
      nat := st0.nat
      ss := []
      g := .nil }
  s.st.set st1
  pure { id := id }

/--
Use a `Param` in the recorded graph by reading its current value and recording it as a leaf.

The returned `TensorRef` is the graph handle you pass to subsequent ops. The session also remembers
which leaf-id corresponds to which parameter, so `sgdStepAll` can update parameters after backward.
PyTorch comparison: like referencing a `torch.nn.Parameter` in the forward; the parameter's value
is treated as a leaf for autograd.
-/
def use {α : Type} (s : TypedGraphSession α) {sh : Shape} [DecidableEq Shape]
  (p : Param α sh) : IO (TensorRef α sh) := do
  let v ← p.value.get
  let leaf ← addLeaf (α := α) s (sh := sh) v p.name
    (s.opts.gradEnabled && p.requiresGrad)
  s.paramsByLeaf.modify (fun m => m.insert leaf.id (AnyParam.ofParam p))
  pure leaf

/--
Record an external input tensor as a leaf.

The input remains part of the typed context whether or not it is differentiable. The
`requiresGrad` flag controls gradient accumulation when the graph is lowered to a runtime tape;
`gradEnabled := false` overrides it for the whole session.
-/
def input {α : Type} (s : TypedGraphSession α) {sh : Shape} [DecidableEq Shape]
  (v : Tensor α sh) (name : Option String := none) (requiresGrad : Bool := false) :
  IO (TensorRef α sh) :=
  addLeaf (α := α) s (sh := sh) v name (s.opts.gradEnabled && requiresGrad)

/--
Record a non-differentiable `Nat` input in the external environment.

This is used for "index-like" inputs (labels, gather indices, etc.) that should not receive
gradients.
PyTorch comparison: like passing an integer tensor / index to an op; indices are not differentiable.
-/
def inputNat {α : Type} (s : TypedGraphSession α) (v : Nat) : IO NatRef := do
  let st0 ← s.st.get
  ensureNoNodes st0
  let id := st0.nat.size
  s.st.set { st0 with nat := st0.nat.push v }
  pure { id := id }

/-- Read a previously recorded `NatRef`. -/
def getNat {α : Type} (s : TypedGraphSession α) (r : NatRef) : IO Nat := do
  let st0 ← s.st.get
  if h : r.id < st0.nat.size then
    pure <| st0.nat[r.id]'h
  else
    throw <| IO.userError "torch(TypedGraphSession): invalid nat id"

/-- Overwrite a previously recorded `NatRef`. -/
def setNat {α : Type} (s : TypedGraphSession α) (r : NatRef) (v : Nat) : IO Unit := do
  let st0 ← s.st.get
  if h : r.id < st0.nat.size then
    let i : Fin st0.nat.size := ⟨r.id, h⟩
    s.st.set { st0 with nat := st0.nat.set i v }
  else
    throw <| IO.userError "torch(TypedGraphSession): invalid nat id"

/--
Convert a small `Tensor Nat (.dim k .scalar)` into an `Array Nat`.

This is used to stage `NatVecRef` inputs into the session nat-environment.
-/
def natVecToArray {k : Nat} (v : Tensor Nat (.dim k .scalar)) : Array Nat :=
  Array.ofFn (fun i : Fin k =>
    match getAtSpec v i with
    | .scalar n => n)

/--
Record a non-differentiable vector of `Nat` inputs.

Returns a `NatVecRef k` which points into the nat-environment. This is useful for "runtime gather"
style ops where indices are supplied externally (and are not differentiable).
-/
def inputNatVec {α : Type} {k : Nat} (s : TypedGraphSession α) (v : Tensor Nat (.dim k .scalar)) : IO
  (NatVecRef k) := do
  let st0 ← s.st.get
  ensureNoNodes st0
  let start := st0.nat.size
  let xsNew := (natVecToArray (k := k) v).foldl (fun acc x => acc.push x) st0.nat
  s.st.set { st0 with nat := xsNew }
  pure { start := start }

/-- Read back the `k`-vector stored at a `NatVecRef k`. -/
def getNatVec {α : Type} {k : Nat} (s : TypedGraphSession α) (r : NatVecRef k) : IO (Tensor Nat (.dim k
  .scalar)) := do
  let st0 ← s.st.get
  if h : r.start + k ≤ st0.nat.size then
    pure <|
      Tensor.dim (fun i =>
        have hi : r.start + i.val < r.start + k := Nat.add_lt_add_left i.is_lt r.start
        have hi' : r.start + i.val < st0.nat.size := lt_of_lt_of_le hi h
        Tensor.scalar (st0.nat[r.start + i.val]'hi'))
  else
    throw <| IO.userError "torch(TypedGraphSession): invalid nat vec ref (out of bounds)"

/-- Overwrite the nat-environment segment referenced by `NatVecRef k`. -/
def setNatVec {α : Type} {k : Nat} (s : TypedGraphSession α) (r : NatVecRef k) (v : Tensor Nat (.dim k
  .scalar)) : IO Unit := do
  let st0 ← s.st.get
  if h : r.start + k ≤ st0.nat.size then
    let xs' :=
      (List.finRange k).foldl (fun acc (i : Fin k) =>
        have hi : r.start + i.val < st0.nat.size := by
          have hlt : r.start + i.val < r.start + k := Nat.add_lt_add_left i.is_lt r.start
          exact lt_of_lt_of_le hlt h
        let vi : Nat :=
          match getAtSpec v i with
          | .scalar n => n
        acc.set! (r.start + i.val) vi
      ) st0.nat
    s.st.set { st0 with nat := xs' }
  else
    throw <| IO.userError "torch(TypedGraphSession): invalid nat vec ref (out of bounds)"

/--
Build a typed index into the current context `Γ ++ ss` from a raw numeric id and expected shape.

This is the main "dynamic check" used by `getValue` (and by a few index-driven nodes): it ensures
that the `Nat` id points to an existing tensor in the session context and that the shape matches.
-/
def mkIdxOrThrow {_α : Type} {Γ ss : List Shape} (id : Nat) (s : Shape) :
    Runtime.Autograd.Result (_root_.Proofs.Autograd.Algebra.Idx (Γ ++ ss) s) := by
    if h : id < (Γ ++ ss).length then
      let fin : Fin (Γ ++ ss).length := ⟨id, h⟩
      let got : Shape := (Γ ++ ss).get fin
      if hg : got = s then
        exact .ok ⟨fin, hg⟩
      else
        exact .error <|
          s!"torch(TypedGraphSession): shape mismatch at id={id}: expected {Shape.pretty s}, got "
            ++ s!"{Shape.pretty got}"
  else
    exact .error s!"torch(TypedGraphSession): invalid id={id} for ctxLen={(Γ ++ ss).length}"

/--
Evaluate the recorded graph and return the value of a `TensorRef`.

This is a pure graph evaluation (`GraphData.eval`) using the recorded leaf values and
nat-environment. It does **not** run the runtime tape or mutate session state.
-/
def getValue {α : Type} (s : TypedGraphSession α) {sh : Shape} [DecidableEq Shape]
  (x : TensorRef α sh) : IO (Tensor α sh) := do
  let st0 ← s.st.get
  -- Evaluate the recorded graph at the recorded leaf values.
  let ctx : _root_.Proofs.Autograd.Algebra.TList α (st0.Γ ++ st0.ss) :=
    _root_.Proofs.Autograd.Algebra.GraphData.eval (α := α) (Δ := NatEnv) (Γ := st0.Γ) (ss := st0.ss)
      st0.g st0.x st0.nat
  let idx ← okOrThrow (mkIdxOrThrow (_α := α) (Γ := st0.Γ) (ss := st0.ss) x.id sh)
  pure (_root_.Proofs.Autograd.Algebra.getIdx (α := α) (xs := ctx) idx)
end TypedGraphSession

end Internal

end Torch
end Autograd
end Runtime
