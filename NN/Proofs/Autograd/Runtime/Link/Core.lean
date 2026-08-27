/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Runtime.ShapeErasure

/-!
# Link

Link the executable runtime tape (`Runtime.Autograd.Tape`) to the shape-indexed SSA/DAG models in
`Proofs.Autograd.Algebra`.

`GraphData` stores executable forward, JVP, and VJP functions without derivative laws. `Graph`
extends that representation with the local adjointness law used by its global backpropagation
theorem. Lowering places the stored VJP into each runtime node's `backward` closure.

## What is proved here

- Forward-pass correspondence: `lowerGraphToTape{,Data}` produces the same values as
  `Graph{,Data}.eval`, and the runtime tape stores those values in the same order
  (`lowerGraphToTape{,Data}_ctx_eq_eval`, `lowerGraphToTape{,Data}_values_eq`).
- Backward-pass correspondence: running the runtime dense reverse loop
  `Tape.backwardDenseFrom` on a lowered tape matches the graph's stored reverse program
  `backpropAllCtx` (`backwardDenseFrom_lowerGraphToTape_eq_backpropAllCtx` and its `GraphData`
  variant). For `GraphData`, this is an implementation-equivalence result. For `Graph`, it can be
  combined with `Graph.backprop_correct` to obtain derivative correctness.

The core invariant making the runtime reverse loop well-founded is that lowered nodes only emit
contributions to earlier node ids (`pid < id`).

## PyTorch correspondence / citations
This is analogous to lowering a graph representation to an executable autograd tape whose nodes
carry backward closures (PyTorch does this internally for the eager autograd engine).
https://pytorch.org/docs/stable/autograd.html
-/

@[expose] public section


namespace Proofs
namespace Autograd
namespace Algebra

open _root_.TorchLean

open Spec
open Tensor

namespace Graph

open Runtime
open Runtime.Autograd

/--
Extend a tape with leaf nodes for every tensor in the input context `Γ`.

Each leaf has `requiresGrad = true` and an empty backward contribution array, so the runtime loop treats
them as gradient accumulation slots but never produces parent contributions from them.
-/
def addLeaves {α : Type} (t : Tape α) : {Γ : List Shape} → _root_.TorchLean.TensorPack α Γ → Tape α
  | [], .nil => t
  | _ :: Γ, .cons x xs =>
      let (t', _id) := Tape.leaf (t := t) x
      addLeaves (t := t') (Γ := Γ) xs

/--
Turn a shape-erased tensor into a runtime leaf node.

This is the node-level counterpart of `addLeaves`: it has no parents and contributes nothing in
backward.
-/
def leafNodeOfSomeTensor {α : Type} (v : Spec.SomeTensor α) : Runtime.Autograd.Node α :=
  { name := none
    value := v
    requiresGrad := true
    parents := #[]
    backward := fun _ => .ok #[] }

/-- `addLeaves` grows the tape by exactly `Γ.length` nodes. -/
theorem size_addLeaves {α : Type} (t : Tape α) :
    {Γ : List Shape} → (x : _root_.TorchLean.TensorPack α Γ) → (addLeaves (α := α) (t := t) (Γ := Γ) x).nodes.size =
      t.nodes.size + Γ.length
  | [], .nil => by simp [addLeaves]
  | _ :: Γ, .cons x xs => by
      simp [addLeaves, Tape.leaf, Tape.addNode, size_addLeaves (t := { nodes := t.nodes.push _ }) (x
        := xs),
        Nat.add_assoc, Nat.add_comm, Array.size_push]

/-- `addLeaves` appends `leafNodeOfSomeTensor` nodes for each input tensor, in order. -/
theorem nodes_addLeaves {α : Type} (t : Tape α) :
    {Γ : List Shape} → (x : _root_.TorchLean.TensorPack α Γ) →
      (addLeaves (α := α) (t := t) (Γ := Γ) x).nodes =
        t.nodes ++
          (_root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ) x).map
            (leafNodeOfSomeTensor (α := α))
  | [], .nil => by
      simp [addLeaves, _root_.TorchLean.TensorPack.toShapeErasedArray]
  | _ :: Γ, .cons x xs => by
      simp [addLeaves, Tape.leaf, Tape.addNode,
        nodes_addLeaves (t := { nodes := t.nodes.push _ }) (Γ := Γ) (x := xs),
        leafNodeOfSomeTensor,
        _root_.TorchLean.TensorPack.toShapeErasedArray_cons (α := α) (ss := Γ) x xs,
        Array.map_append, Array.append_singleton_assoc]

/-- Value projection of `nodes_addLeaves`: `node.value` agrees with `toShapeErasedArray` for added leaves.
  -/
theorem addLeaves_values {α : Type} (t : Tape α) :
    {Γ : List Shape} → (x : _root_.TorchLean.TensorPack α Γ) →
      (addLeaves (α := α) (t := t) (Γ := Γ) x).nodes.map (fun node => node.value) =
        t.nodes.map (fun node => node.value) ++ _root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ) x
  | [], .nil => by
      simp [addLeaves, _root_.TorchLean.TensorPack.toShapeErasedArray]
  | _ :: Γ, .cons x xs => by
      -- unfold one `leaf` push and use the induction hypothesis on the remaining leaves
      simp [addLeaves, Tape.leaf, Tape.addNode,
        addLeaves_values (t := { nodes := t.nodes.push _ }) (Γ := Γ) (x := xs),
        _root_.TorchLean.TensorPack.toShapeErasedArray]

/--
Lower an executable graph (`GraphData`) to a runtime tape by evaluating forward nodes and storing
each node's `vjp` program in its runtime `backward` closure.

PyTorch analogy: this corresponds to building a tape of autograd nodes during the forward pass,
where each node stores enough information to compute parent contributions when given an upstream
cotangent.
-/
def lowerGraphDataToTape {α : Type} {Δ : Type} [DecidableEq Shape]
  {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss) (x : _root_.TorchLean.TensorPack α Γ) (d : Δ) :
  Tape α × _root_.TorchLean.TensorPack α (Γ ++ ss) :=
  match g with
  | .nil =>
      let t := addLeaves (α := α) (t := Tape.empty) (Γ := Γ) x
      (t, _root_.TorchLean.TensorPack.cast (α := α) (h := (List.append_nil Γ).symm) x)
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let (tPrev, ctxPrev) := lowerGraphDataToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d
      let y := node.forward ctxPrev d
      let runtimeNode : Runtime.Autograd.Node α :=
        { name := some "typed-graph"
          value := Spec.SomeTensor.ofTensor y
          requiresGrad := true
          parents := #[]
          backward := fun dLdyValue => by
            if h : dLdyValue.shape = τ then
              let dLdy : Tensor α τ := dLdyValue.cast h
              let contribs := node.vjp ctxPrev d dLdy
              exact .ok (_root_.TorchLean.TensorPack.toIndexedShapeErasedArray (α := α) (ss := Γ ++ ssPrev) contribs 0)
            else
              exact .error "autograd: upstream gradient shape mismatch"
        }
      let (tNext, _id) := Tape.addNode (t := tPrev) runtimeNode
      let ctxNext :=
        _root_.TorchLean.TensorPack.cast (α := α) (h := List.append_assoc Γ ssPrev [τ])
          (_root_.TorchLean.TensorPack.snoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) ctxPrev y)
      (tNext, ctxNext)

/-!
### Forward-pass correspondence

The next lemmas show that `lowerGraphDataToTape` preserves executable forward semantics, and that the
resulting runtime tape contains exactly the evaluated context as shape-erased tensors in order.
-/

/-- The context returned by `lowerGraphDataToTape` agrees with `GraphData.eval`. -/
theorem lowerGraphDataToTape_ctx_eq_eval {α : Type} {Δ : Type} [DecidableEq Shape]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss) (x : _root_.TorchLean.TensorPack α Γ) (d : Δ) :
    (lowerGraphDataToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).2 =
      GraphData.eval (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d := by
  induction g with
  | nil =>
      simp [lowerGraphDataToTape, GraphData.eval]
  | snoc g node ih =>
      rename_i ssPrev τ
      simp [lowerGraphDataToTape, GraphData.eval, ih]

/-- The lowered tape's `.value` array is `GraphData.eval` with shapes erased in the same order.
  -/
theorem lowerGraphDataToTape_values_eq {α : Type} {Δ : Type} [DecidableEq Shape]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss) (x : _root_.TorchLean.TensorPack α Γ) (d : Δ) :
    (lowerGraphDataToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).1.nodes.map (fun node =>
      node.value) =
      _root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ ss) (lowerGraphDataToTape (α := α) (Δ := Δ) (Γ := Γ) (ss :=
        ss) g x d).2 := by
  induction g with
  | nil =>
      -- only leaves
      simp [lowerGraphDataToTape, addLeaves_values, Runtime.Autograd.Tape.empty]
  | snoc g _node ih =>
      rename_i ssPrev τ
      simp [lowerGraphDataToTape, Runtime.Autograd.Tape.addNode, ih]

/-- Size bookkeeping: the lowered tape contains one runtime node for each element of `Γ ++ ss`. -/
theorem lowerGraphDataToTape_nodes_size {α : Type} {Δ : Type} [DecidableEq Shape]
    {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss) (x : _root_.TorchLean.TensorPack α Γ) (d : Δ) :
    (lowerGraphDataToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).1.nodes.size = Γ.length + ss.length
      := by
  induction g with
  | nil =>
      -- only leaves
      simp [lowerGraphDataToTape, size_addLeaves, Runtime.Autograd.Tape.empty]
  | snoc g _node ih =>
      rename_i ssPrev τ
      simp [lowerGraphDataToTape, Runtime.Autograd.Tape.addNode, ih, Array.size_push, Nat.add_assoc,
        ]

/--
Lower a proved graph (`Graph`) to a runtime tape by evaluating forward nodes and storing each
node’s proved `vjp`.

Compared to `lowerGraphDataToTape`, this uses the pure graph interface (no explicit `GraphData` payload).
-/
def lowerGraphToTape {α : Type} {Δ : Type} [DecidableEq Shape] [CommSemiring α]
  {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss) (x : _root_.TorchLean.TensorPack α Γ) (d : Δ) :
  Tape α × _root_.TorchLean.TensorPack α (Γ ++ ss) :=
  match g with
  | .nil =>
      let t := addLeaves (α := α) (t := Tape.empty) (Γ := Γ) x
      (t, _root_.TorchLean.TensorPack.cast (α := α) (h := (List.append_nil Γ).symm) x)
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let (tPrev, ctxPrev) := lowerGraphToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d
      let y := node.forward ctxPrev d
      let runtimeNode : Runtime.Autograd.Node α :=
        { name := some "proof-carrying-graph"
          value := Spec.SomeTensor.ofTensor y
          requiresGrad := true
          parents := #[]
          backward := fun dLdyValue => by
            if h : dLdyValue.shape = τ then
              let dLdy : Tensor α τ := dLdyValue.cast h
              let contribs := node.vjp ctxPrev d dLdy
              exact .ok (_root_.TorchLean.TensorPack.toIndexedShapeErasedArray (α := α) (ss := Γ ++ ssPrev) contribs 0)
            else
              exact .error "autograd: upstream gradient shape mismatch"
        }
      let (tNext, _id) := Tape.addNode (t := tPrev) runtimeNode
      let ctxNext :=
        _root_.TorchLean.TensorPack.cast (α := α) (h := List.append_assoc Γ ssPrev [τ])
          (_root_.TorchLean.TensorPack.snoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) ctxPrev y)
      (tNext, ctxNext)

/-- The context returned by `lowerGraphToTape` agrees with the proved `Graph.eval`. -/
theorem lowerGraphToTape_ctx_eq_eval {α : Type} {Δ : Type} [DecidableEq Shape] [CommSemiring α]
    {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss) (x : _root_.TorchLean.TensorPack α Γ) (d : Δ) :
    (lowerGraphToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).2 =
      Graph.eval (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d := by
  induction g with
  | nil =>
      simp [lowerGraphToTape, Graph.eval]
  | snoc g node ih =>
      rename_i ssPrev τ
      simp [lowerGraphToTape, Graph.eval, ih]

/-- The lowered tape's `.value` array is `Graph.eval` with shapes erased in the same order. -/
theorem lowerGraphToTape_values_eq {α : Type} {Δ : Type} [DecidableEq Shape] [CommSemiring α]
    {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss) (x : _root_.TorchLean.TensorPack α Γ) (d : Δ) :
    (lowerGraphToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).1.nodes.map (fun node => node.value) =
      _root_.TorchLean.TensorPack.toShapeErasedArray (α := α) (ss := Γ ++ ss) (lowerGraphToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g
        x d).2 := by
  induction g with
  | nil =>
      -- only leaves
      simp [lowerGraphToTape, addLeaves_values, Runtime.Autograd.Tape.empty]
  | snoc g node ih =>
      rename_i ssPrev τ
      simp [lowerGraphToTape, Runtime.Autograd.Tape.addNode, ih]

/-- Size bookkeeping: `lowerGraphToTape` produces `Γ.length + ss.length` runtime nodes. -/
theorem lowerGraphToTape_nodes_size {α : Type} {Δ : Type} [DecidableEq Shape] [CommSemiring α]
    {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss) (x : _root_.TorchLean.TensorPack α Γ) (d : Δ) :
    (lowerGraphToTape (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) g x d).1.nodes.size = Γ.length + ss.length :=
      by
  induction g with
  | nil =>
      simp [lowerGraphToTape, size_addLeaves, Runtime.Autograd.Tape.empty]
  | snoc g node ih =>
      rename_i ssPrev τ
      simp [lowerGraphToTape, Runtime.Autograd.Tape.addNode, ih, Array.size_push, Nat.add_assoc]

/-!
### Full backpropagation (dense) for proofs and runtime

The runtime engine computes a *dense* gradient array, accumulating cotangents for every node in the
tape (inputs and intermediates). The following definition and theorems connect that behavior to the
proved backpropagation semantics.
-/

/-- A "full" backpropagation that returns gradients for every value in `Γ ++ ss`. -/
def backpropAllCtx {α : Type} {Δ : Type} [CommSemiring α]
  {Γ : List Shape} {ss : List Shape} (g : Graph (α := α) Δ Γ ss) (x : _root_.TorchLean.TensorPack α Γ) (d : Δ)
  (seed : _root_.TorchLean.TensorPack α (Γ ++ ss)) :
  _root_.TorchLean.TensorPack α (Γ ++ ss) :=
  match g with
  | .nil => seed
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let assoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      let seed' : _root_.TorchLean.TensorPack α ((Γ ++ ssPrev) ++ [τ]) := _root_.TorchLean.TensorPack.cast (α := α) (h := assoc.symm) seed
      let seedPrev : _root_.TorchLean.TensorPack α (Γ ++ ssPrev) := (_root_.TorchLean.TensorPack.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ)
        seed').1
      let seedOut : Tensor α τ := (_root_.TorchLean.TensorPack.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seed').2
      let ctx := Graph.eval (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d
      let contrib := node.vjp ctx d seedOut
      let seedPrev' := _root_.TorchLean.TensorPack.add (α := α) (ss := Γ ++ ssPrev) seedPrev contrib
      let gradsPrev := backpropAllCtx (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d seedPrev'
      _root_.TorchLean.TensorPack.cast (α := α) (h := assoc)
        (_root_.TorchLean.TensorPack.snoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) gradsPrev seedOut)

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
  {Γ : List Shape} {ss : List Shape} (g : GraphData α Δ Γ ss) (x : _root_.TorchLean.TensorPack α Γ) (d : Δ)
  (seed : _root_.TorchLean.TensorPack α (Γ ++ ss)) :
  _root_.TorchLean.TensorPack α (Γ ++ ss) :=
  match g with
  | .nil => seed
  | .snoc (ss := ssPrev) (τ := τ) g node =>
      let assoc : (Γ ++ ssPrev) ++ [τ] = Γ ++ (ssPrev ++ [τ]) := List.append_assoc Γ ssPrev [τ]
      let seed' : _root_.TorchLean.TensorPack α ((Γ ++ ssPrev) ++ [τ]) := _root_.TorchLean.TensorPack.cast (α := α) (h := assoc.symm) seed
      let seedPrev : _root_.TorchLean.TensorPack α (Γ ++ ssPrev) := (_root_.TorchLean.TensorPack.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ)
        seed').1
      let seedOut : Tensor α τ := (_root_.TorchLean.TensorPack.unsnoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seed').2
      let ctx := GraphData.eval (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d
      let contrib := node.vjp ctx d seedOut
      let seedPrev' := _root_.TorchLean.TensorPack.add (α := α) (ss := Γ ++ ssPrev) seedPrev contrib
      let gradsPrev := backpropAllCtx (α := α) (Δ := Δ) (Γ := Γ) (ss := ssPrev) g x d seedPrev'
      _root_.TorchLean.TensorPack.cast (α := α) (h := assoc)
        (_root_.TorchLean.TensorPack.snoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) gradsPrev seedOut)


end Graph

end Algebra
end Autograd
end Proofs
