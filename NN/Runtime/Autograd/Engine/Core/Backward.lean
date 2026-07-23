/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.Runtime.Autograd.Engine.Core.ActivationsLoss

@[expose] public section

namespace Runtime
namespace Autograd

open Spec
open Tensor

namespace Tape

/-!
## Backpropagation

Reverse-mode is implemented by traversing node ids in reverse order. Each node’s `backward`
closure produces parent-gradient contributions, which we accumulate by elementwise summation.
-/

/--
 Internal helper: add a single parent gradient contribution into the dense optional gradient array.

 This is where we implement PyTorch-style accumulation for DAGs: if multiple children contribute
 to the same parent id, we sum the contributions.

 The dense array entry is `none` until we first reach a node during reverse traversal.
 -/
def addGradDense
  {α : Type} [Add α] [DecidableEq Shape]
  (t : Tape α) (grads : Array (Option (Runtime.AnyTensor α)))
  (id : Nat) (g : Runtime.AnyTensor α) : Result (Array (Option (Runtime.AnyTensor α))) := do
  let node ← match t.getNode? id with
    | some n => pure n
    | none => throw "autograd: invalid parent id during backward"
  if node.requires_grad = false then
    pure grads
  else if h : g.s = node.value.s then
    let g' : Runtime.AnyTensor α := AnyTensor.materialize
      { s := node.value.s, t := Tensor.castShape g.t h }
    if hid : id < grads.size then
      match grads[id]'hid with
      | none =>
          pure (grads.set id (some g') (h := hid))
      | some existing =>
          let summed ← AnyTensor.add existing g'
          pure (grads.set id (some summed) (h := hid))
    else
      throw "autograd: internal error (gradient array out of bounds)"
  else
    throw "autograd: gradient contribution has wrong shape for parent"

/--
Reverse-mode backpropagation producing a dense array of optional gradients.

- The result array has length `t.nodes.size`.
- Entry `id` is `some g` if the node was reached from `outId` during reverse traversal, otherwise
  `none`.
- When multiple paths contribute to the same node, we sum gradients via `AnyTensor.add`.

This is loosely analogous to PyTorch's autograd engine walking the dynamic graph and accumulating
`.grad` for leaf tensors, but we keep gradients for every node id rather than leaves alone. That makes the
runtime easier to debug and gives proof-bridge code direct access to intermediate cotangents.

Reference (PyTorch): https://pytorch.org/docs/stable/notes/autograd.html
-/
def backwardDense {α : Type} [Add α] [DecidableEq Shape]
  (t : Tape α) (outId : Nat) (seed : Runtime.AnyTensor α) :
  Result (Array (Option (Runtime.AnyTensor α))) := do
  let outNode ← match t.getNode? outId with
    | some n => pure n
    | none => throw "autograd: invalid output id"
  if h : seed.s = outNode.value.s then
    let seed' : Runtime.AnyTensor α := AnyTensor.materialize
      { s := outNode.value.s, t := Tensor.castShape seed.t h }
    let mut grads : Array (Option (Runtime.AnyTensor α)) := Array.replicate t.nodes.size none
    if hout : outId < grads.size then
      grads := grads.set outId (some seed') (h := hout)
    else
      throw "autograd: invalid output id"
    let ids := (List.range t.nodes.size).reverse
    ids.foldlM (fun acc id => do
      match acc[id]? with
      | none => throw "autograd: internal error (gradient array out of bounds)"
      | some none => pure acc
      | some (some dLdy) =>
        let node ← match t.getNode? id with
          | some n => pure n
          | none => throw "autograd: internal error (node missing)"
        if node.requires_grad = false then
          pure acc
        else
          let contribs ← node.backward dLdy
          contribs.foldlM (fun acc2 (pid, pg) => addGradDense (t:=t) acc2 pid pg) acc
    ) grads
  else
    throw "autograd: seed gradient shape mismatch for output"

/--
Internal helper: like `addGradDense`, but assumes the gradient array is total (no `Option`).

This is used by the proof-friendly `backwardDenseFrom*` variants, which start from an explicit
gradient tensor for every node.
-/
def addGradAll
  {α : Type} [Add α] [DecidableEq Shape]
  (t : Tape α) (grads : Array (Runtime.AnyTensor α))
  (id : Nat) (g : Runtime.AnyTensor α) : Result (Array (Runtime.AnyTensor α)) := do
  let node ← match t.getNode? id with
    | some n => pure n
    | none => throw "autograd: invalid parent id during backward"
  if node.requires_grad = false then
    pure grads
  else if h : g.s = node.value.s then
    let g' : Runtime.AnyTensor α := AnyTensor.materialize
      { s := node.value.s, t := Tensor.castShape g.t h }
    match grads[id]? with
    | none => throw "autograd: internal error (gradient array out of bounds)"
      | some existing =>
          if hex : existing.s = node.value.s then
            let existing' : Runtime.AnyTensor α :=
              { s := node.value.s, t := Tensor.castShape existing.t hex }
            let summed ← AnyTensor.add existing' g'
            if hid : id < grads.size then
              pure (grads.set id summed (h := hid))
            else
              throw "autograd: internal error (gradient array out of bounds)"
          else
            throw "autograd: gradient array has wrong shape for node"
  else
    throw "autograd: gradient contribution has wrong shape for parent"

/--
One reverse-mode backprop step at a single node id, updating a total dense gradient array.

Precondition by convention: `acc` has one entry per tape node, and every entry has the matching
node shape. The function checks those conditions dynamically and returns an error if a caller
violates them. This makes it suitable as the small proof-friendly step used by
`backwardDenseFromLoop`.
-/
def backwardDenseFromStep {α : Type} [Add α] [DecidableEq Shape]
  (t : Tape α) (acc : Array (Runtime.AnyTensor α)) (id : Nat) :
  Result (Array (Runtime.AnyTensor α)) := do
  let node ← match t.getNode? id with
    | some n => pure n
    | none => throw "autograd: internal error (node missing)"
  if node.requires_grad = false then
    pure acc
  else
    let dLdyAny ← match acc[id]? with
      | some g => pure g
      | none => throw "autograd: internal error (gradient array out of bounds)"
    if hshape : dLdyAny.s = node.value.s then
      let dLdy : Runtime.AnyTensor α := { s := node.value.s, t := Tensor.castShape dLdyAny.t hshape
        }
      let contribs ← node.backward dLdy
      contribs.foldlM (fun acc2 (pid, pg) => addGradAll (t := t) acc2 pid pg) acc
    else
      throw "autograd: gradient array has wrong shape for node"

/--
Reverse-mode accumulation over the first `n` nodes in reverse order.

The recursion visits node ids `n-1, n-2, ..., 0`. Passing `n = t.nodes.size` therefore traverses the
entire tape. This structurally recursive loop is the one used by proof-linked compiled sessions.
-/
def backwardDenseFromLoop {α : Type} [Add α] [DecidableEq Shape]
  (t : Tape α) : Nat → Array (Runtime.AnyTensor α) → Result (Array (Runtime.AnyTensor α))
  | 0, acc => pure acc
  | n + 1, acc => do
      let acc' ← backwardDenseFromStep (t := t) acc n
      backwardDenseFromLoop (t := t) n acc'

/--
Reverse-mode accumulation starting from an explicit dense gradient array.

This is a proof-friendly variant: it always runs every node (in reverse order) and keeps a
gradient tensor for every node id.
-/
def backwardDenseFrom {α : Type} [Add α] [DecidableEq Shape]
  (t : Tape α) (grads0 : Array (Runtime.AnyTensor α)) :
  Result (Array (Runtime.AnyTensor α)) := do
  if grads0.size = t.nodes.size then
    backwardDenseFromLoop (t := t) t.nodes.size grads0
  else
    throw "autograd: initial dense gradient array has wrong length"

/-- Reverse-mode accumulation that returns a dense gradient array for every node id.

Propagation uses `backwardDense`, so local VJP closures run only for nodes reached from `outId`.
The optional result is then totalized with explicit zero tensors for disconnected nodes. Keeping
that distinction matters at singular forward values: applying a disconnected VJP to a synthetic
zero cotangent can manufacture `NaN` through expressions such as `0 * (1 / 0)`, even though the
mathematical gradient of the selected output with respect to that node is zero.
-/
def backwardDenseAll {α : Type} [Add α] [Zero α] [DecidableEq Shape]
  (t : Tape α) (outId : Nat) (seed : Runtime.AnyTensor α) :
  Result (Array (Runtime.AnyTensor α)) := do
  let reached ← backwardDense (t := t) outId seed
  pure <| t.nodes.mapIdx fun id node =>
    match reached[id]? with
    | some (some grad) => grad
    | _ => AnyTensor.mk (fill (0 : α) node.value.s)

/--
Convert the optional dense gradient array returned by `backwardDense` into a sparse `HashMap`.

Only entries that are present (`some (some g)`) are kept. The result records exactly the nodes
reached by reverse-mode propagation.
-/
def denseToHashMap {α : Type}
  (grads : Array (Option (Runtime.AnyTensor α))) :
  Std.HashMap Nat (Runtime.AnyTensor α) :=
  (List.range grads.size).foldl (fun acc id =>
    match grads[id]? with
    | some (some g) => acc.insert id g
    | _ => acc
  ) (Std.HashMap.emptyWithCapacity)

/--
Reverse-mode backpropagation returning a `HashMap` of only the nodes that received gradients.

This is the sparse public form of `backwardDense`: it computes dense gradients first, then drops
nodes that did not receive a gradient.
-/
def backward {α : Type} [Add α] [DecidableEq Shape]
  (t : Tape α) (outId : Nat) (seed : Runtime.AnyTensor α) :
  Result (Std.HashMap Nat (Runtime.AnyTensor α)) := do
  let dense ← backwardDense (t := t) outId seed
  pure (denseToHashMap dense)

/--
Backpropagate from a scalar output with seed gradient `1`.

PyTorch analogy: `loss.backward()` when `loss` is a scalar.
-/
def backwardScalar {α : Type} [Add α] [One α] [DecidableEq Shape]
  (t : Tape α) (outId : Nat) : Result (Std.HashMap Nat (Runtime.AnyTensor α)) :=
  backward (t:=t) outId (AnyTensor.mk (Tensor.scalar (1 : α)))

end Tape
end Autograd
end Runtime
