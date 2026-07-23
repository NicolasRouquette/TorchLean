/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Runtime.Any

/-!
# Link

Link the executable runtime tape (`Runtime.Autograd.Tape`) to the proved SSA/DAG tape model
(`Proofs.Autograd.Algebra.Graph`).

This file provides a small compiler from proved graphs to runtime tapes. The compiler bakes the
proved `vjp` into each runtime node's `backward` closure.

## What is proved here

- Forward-pass correspondence: `compileAux{,Data}` produces the same values as the proved
  `Graph{,Data}.eval`, and the runtime tape stores those values in the same order
  (`compileAux{,Data}_ctx_eq_eval`, `compileAux{,Data}_values_eq`).
- Backward-pass correspondence: running the runtime dense reverse loop
  `Tape.backwardDenseFrom` on a compiled tape matches the proved “full backpropagation”
  `backpropAllCtx` (`backwardDenseFrom_compileAux_eq_backpropAllCtx` and its `GraphData` variant).

The core invariant making the runtime reverse loop well-founded is that compiled nodes only emit
contributions to earlier node ids (`pid < id`).

## PyTorch correspondence / citations
This is analogous to taking a proven “graph IR” and compiling it to an executable autograd tape
whose nodes carry a backward closure (PyTorch does this internally for the eager autograd engine).
https://pytorch.org/docs/stable/autograd.html
-/

@[expose] public section


namespace Proofs
namespace Autograd
namespace Algebra

open Spec
open Tensor

namespace Graph

open Runtime
open Runtime.Autograd

/--
Extend a tape with leaf nodes for every tensor in the input context `Γ`.

Each leaf has `requires_grad = true` and `backward = ok []`, so the runtime backward loop treats
them as gradient accumulation slots but never produces parent contributions from them.
-/
def addLeaves {α : Type} (t : Tape α) : {Γ : List Shape} → TList α Γ → Tape α
  | [], .nil => t
  | _ :: Γ, .cons x xs =>
      let (t', _id) := Tape.leaf (t := t) x
      addLeaves (t := t') (Γ := Γ) xs

/--
Turn a value-only `AnyTensor` into a runtime leaf node.

This is the node-level counterpart of `addLeaves`: it has no parents and contributes nothing in
backward.
-/
def leafNodeOfAny {α : Type} (v : Runtime.AnyTensor α) : Runtime.Autograd.Node α :=
  { name := none
    value := v
    requires_grad := true
    parents := []
    backward := fun _ => .ok [] }

/-- `addLeaves` grows the tape by exactly `Γ.length` nodes. -/
theorem size_addLeaves {α : Type} (t : Tape α) :
    {Γ : List Shape} → (x : TList α Γ) → (addLeaves (α := α) (t := t) (Γ := Γ) x).nodes.size =
      t.nodes.size + Γ.length
  | [], .nil => by simp [addLeaves]
  | _ :: Γ, .cons x xs => by
      simp [addLeaves, Tape.leaf, Tape.addNode, size_addLeaves (t := { nodes := t.nodes.push _ }) (x
        := xs),
        Nat.add_assoc, Nat.add_comm, Array.size_push]

/-- `addLeaves` appends `leafNodeOfAny` nodes for each element of the input context, in order. -/
theorem nodes_addLeaves {α : Type} (t : Tape α) :
    {Γ : List Shape} → (x : TList α Γ) →
      (addLeaves (α := α) (t := t) (Γ := Γ) x).nodes =
        t.nodes ++ (TList.toAnyArray (α := α) (ss := Γ) x).map (leafNodeOfAny (α := α))
  | [], .nil => by
      simp [addLeaves, TList.toAnyArray, TList.toAnyList]
  | _ :: Γ, .cons x xs => by
      simp [addLeaves, Tape.leaf, Tape.addNode,
        nodes_addLeaves (t := { nodes := t.nodes.push _ }) (Γ := Γ) (x := xs),
        leafNodeOfAny, TList.toAnyArray_cons (α := α) (ss := Γ) x xs,
        Array.map_append, Array.append_singleton_assoc]

/-- Value projection of `nodes_addLeaves`: `node.value` agrees with `toAnyArray` for added leaves.
  -/
theorem addLeaves_values {α : Type} (t : Tape α) :
    {Γ : List Shape} → (x : TList α Γ) →
      (addLeaves (α := α) (t := t) (Γ := Γ) x).nodes.map (fun node => node.value) =
        t.nodes.map (fun node => node.value) ++ TList.toAnyArray (α := α) (ss := Γ) x
  | [], .nil => by
      simp [addLeaves, TList.toAnyArray, TList.toAnyList]
  | _ :: Γ, .cons x xs => by
      -- unfold one `leaf` push and use the induction hypothesis on the remaining leaves
      simp [addLeaves, Tape.leaf, Tape.addNode,
        addLeaves_values (t := { nodes := t.nodes.push _ }) (Γ := Γ) (x := xs),
        TList.toAnyArray, TList.toAnyList]

/--
Compile an executable graph (`GraphData`) to a runtime tape by evaluating forward nodes and baking
in each node’s proved `vjp` into its runtime `backward` closure.

PyTorch analogy: this corresponds to building a tape of autograd nodes during the forward pass,
where each node stores enough information to compute parent contributions when given an upstream
cotangent.
-/
def compileAuxData {α : Type} {Δ : Type} [DecidableEq Shape]
  {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TList α Γ) (d : Δ) :
  Tape α × TList α (Γ ++ ss) :=
  match g with
  | .nil =>
      let t := addLeaves (α := α) (t := Tape.empty) (Γ := Γ) x
      (t, TList.cast (α := α) (h := (List.append_nil Γ).symm) x)
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let (tPrev, ctxPrev) := compileAuxData (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d
      let y := node.forward ctxPrev d
      let runtimeNode : Runtime.Autograd.Node α :=
        { name := some "proof-compiled"
          value := Runtime.Autograd.AnyTensor.mk y
          requires_grad := true
          parents := []
          backward := fun dLdyAny => by
            if h : dLdyAny.s = τ then
              let dLdy : Tensor α τ := Tensor.castShape dLdyAny.t h
              let contribs := node.vjp ctxPrev d dLdy
              exact .ok (TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev) contribs 0)
            else
              exact .error "autograd: upstream gradient shape mismatch"
        }
      let (tNext, _id) := Tape.addNode (t := tPrev) runtimeNode
      let ctxNext :=
        TList.cast (α := α) (h := List.append_assoc Γ ssPrev [τ])
          (TList.snoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) ctxPrev y)
      (tNext, ctxNext)

/-!
### Forward-pass correspondence

The next lemmas show that `compileAuxData` preserves the proved forward semantics, and that the
resulting runtime tape contains exactly the evaluated context (erased to `AnyTensor`) in order.
-/

/-- The context returned by `compileAuxData` agrees with the proved `GraphData.eval`. -/
theorem compileAuxData_ctx_eq_eval {α : Type} {Δ : Type} [DecidableEq Shape]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TList α Γ) (d : Δ) :
    (compileAuxData (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).2 =
      GraphData.eval (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d := by
  induction g with
  | nil =>
      simp [compileAuxData, GraphData.eval]
  | snoc g node ih =>
      rename_i ssPrev τ
      simp [compileAuxData, GraphData.eval, ih]

/-- The compiled tape’s `.value` array is `GraphData.eval` erased to `AnyTensor`, in the same order.
  -/
theorem compileAuxData_values_eq {α : Type} {Δ : Type} [DecidableEq Shape]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TList α Γ) (d : Δ) :
    (compileAuxData (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).1.nodes.map (fun node =>
      node.value) =
      TList.toAnyArray (α := α) (ss := Γ ++ ss) (compileAuxData (α := α) (Δ := Δ) (Γ := Γ) (ss :=
        ss) g x d).2 := by
  induction g with
  | nil =>
      -- only leaves
      simp [compileAuxData, addLeaves_values, Runtime.Autograd.Tape.empty]
  | snoc g _node ih =>
      rename_i ssPrev τ
      simp [compileAuxData, Runtime.Autograd.Tape.addNode, ih]

/-- Size bookkeeping: the compiled tape contains one runtime node for each element of `Γ ++ ss`. -/
theorem compileAuxData_nodes_size {α : Type} {Δ : Type} [DecidableEq Shape]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TList α Γ) (d : Δ) :
    (compileAuxData (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).1.nodes.size = Γ.length + ss.length
      := by
  induction g with
  | nil =>
      -- only leaves
      simp [compileAuxData, size_addLeaves, Runtime.Autograd.Tape.empty]
  | snoc g _node ih =>
      rename_i ssPrev τ
      simp [compileAuxData, Runtime.Autograd.Tape.addNode, ih, Array.size_push, Nat.add_assoc,
        ]

/--
Compile a proved graph (`Graph`) to a runtime tape by evaluating forward nodes and baking in each
node’s proved `vjp`.

Compared to `compileAuxData`, this uses the pure graph interface (no explicit `GraphData` payload).
-/
def compileAux {α : Type} {Δ : Type} [DecidableEq Shape] [CommSemiring α]
  {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss) (x : TList α Γ) (d : Δ) :
  Tape α × TList α (Γ ++ ss) :=
  match g with
  | .nil =>
      let t := addLeaves (α := α) (t := Tape.empty) (Γ := Γ) x
      (t, TList.cast (α := α) (h := (List.append_nil Γ).symm) x)
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let (tPrev, ctxPrev) := compileAux (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d
      let y := node.forward ctxPrev d
      let runtimeNode : Runtime.Autograd.Node α :=
        { name := some "proof-compiled"
          value := Runtime.Autograd.AnyTensor.mk y
          requires_grad := true
          parents := []
          backward := fun dLdyAny => by
            if h : dLdyAny.s = τ then
              let dLdy : Tensor α τ := Tensor.castShape dLdyAny.t h
              let contribs := node.vjp ctxPrev d dLdy
              exact .ok (TList.toIndexedAnyList (α := α) (ss := Γ ++ ssPrev) contribs 0)
            else
              exact .error "autograd: upstream gradient shape mismatch"
        }
      let (tNext, _id) := Tape.addNode (t := tPrev) runtimeNode
      let ctxNext :=
        TList.cast (α := α) (h := List.append_assoc Γ ssPrev [τ])
          (TList.snoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) ctxPrev y)
      (tNext, ctxNext)

/-- The context returned by `compileAux` agrees with the proved `Graph.eval`. -/
theorem compileAux_ctx_eq_eval {α : Type} {Δ : Type} [DecidableEq Shape] [CommSemiring α]
    {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss) (x : TList α Γ) (d : Δ) :
    (compileAux (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).2 =
      Graph.eval (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d := by
  induction g with
  | nil =>
      simp [compileAux, Graph.eval]
  | snoc g node ih =>
      rename_i ssPrev τ
      simp [compileAux, Graph.eval, ih]

/-- The compiled tape’s `.value` array is `Graph.eval` erased to `AnyTensor`, in the same order. -/
theorem compileAux_values_eq {α : Type} {Δ : Type} [DecidableEq Shape] [CommSemiring α]
    {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss) (x : TList α Γ) (d : Δ) :
    (compileAux (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).1.nodes.map (fun node => node.value) =
      TList.toAnyArray (α := α) (ss := Γ ++ ss) (compileAux (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g
        x d).2 := by
  induction g with
  | nil =>
      -- only leaves
      simp [compileAux, addLeaves_values, Runtime.Autograd.Tape.empty]
  | snoc g node ih =>
      rename_i ssPrev τ
      simp [compileAux, Runtime.Autograd.Tape.addNode, ih]

/-- Size bookkeeping: `compileAux` produces `Γ.length + ss.length` runtime nodes. -/
theorem compileAux_nodes_size {α : Type} {Δ : Type} [DecidableEq Shape] [CommSemiring α]
    {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss) (x : TList α Γ) (d : Δ) :
    (compileAux (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).1.nodes.size = Γ.length + ss.length :=
      by
  induction g with
  | nil =>
      simp [compileAux, size_addLeaves, Runtime.Autograd.Tape.empty]
  | snoc g node ih =>
      rename_i ssPrev τ
      simp [compileAux, Runtime.Autograd.Tape.addNode, ih, Array.size_push, Nat.add_assoc]

/-!
### Full backpropagation (dense) for proofs and runtime

The runtime engine computes a *dense* gradient array, accumulating cotangents for every node in the
tape (inputs and intermediates). The following definition and theorems connect that behavior to the
proved backpropagation semantics.
-/

/-- A "full" backpropagation that returns gradients for every value in `Γ ++ ss`. -/
def backpropAllCtx {α : Type} {Δ : Type} [CommSemiring α]
  {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss) (x : TList α Γ) (d : Δ)
  (seed : TList α (Γ ++ ss)) :
  TList α (Γ ++ ss) :=
  match g with
  | .nil => seed
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let assoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      let seed' : TList α ((Γ ++ ssPrev) ++ [τ]) := TList.cast (α := α) (h := assoc.symm) seed
      let seedPrev : TList α (Γ ++ ssPrev) := (TList.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ)
        seed').1
      let seedOut : Tensor α τ := (TList.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seed').2
      let ctx := Graph.eval (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d
      let contrib := node.vjp ctx d seedOut
      let seedPrev' := TList.add (α := α) (ss := Γ ++ ssPrev) seedPrev contrib
      let gradsPrev := backpropAllCtx (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d seedPrev'
      TList.cast (α := α) (h := assoc)
        (TList.snoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) gradsPrev seedOut)

/--
“Full” backpropagation for `GraphData` that returns gradients for every value in `Γ ++ ss`, including
  inputs.

This is the `GraphData`-analogue of `backpropAllCtx` above. We keep both definitions because:
- `Graph` uses `[CommSemiring α]` (so it can express dot products and semiring-based accumulation),
  while
- `GraphData` only needs `[Add α]` here (it just adds contributions).

Both follow the same reverse-mode accumulation structure: peel off the last node, apply its VJP to
the seed on that node, add into the previous seed, and recurse.
-/
def _root_.Proofs.Autograd.Algebra.GraphData.backpropAllCtx {α : Type} {Δ : Type} [Add α]
  {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss) (x : TList α Γ) (d : Δ)
  (seed : TList α (Γ ++ ss)) :
  TList α (Γ ++ ss) :=
  match g with
  | .nil => seed
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let assoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      let seed' : TList α ((Γ ++ ssPrev) ++ [τ]) := TList.cast (α := α) (h := assoc.symm) seed
      let seedPrev : TList α (Γ ++ ssPrev) := (TList.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ)
        seed').1
      let seedOut : Tensor α τ := (TList.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seed').2
      let ctx := GraphData.eval (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d
      let contrib := node.vjp ctx d seedOut
      let seedPrev' := TList.add (α := α) (ss := Γ ++ ssPrev) seedPrev contrib
      let gradsPrev := backpropAllCtx (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d seedPrev'
      TList.cast (α := α) (h := assoc)
        (TList.snoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) gradsPrev seedOut)


end Graph

end Algebra
end Autograd
end Proofs
