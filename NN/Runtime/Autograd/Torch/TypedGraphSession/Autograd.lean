/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.TypedGraphSession.ConvAttention

/-!
# Typed Graph Session: Differentiation and Backpropagation
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec
open Tensor

namespace Internal

namespace TypedGraphSession

/-! ## Backward + SGD over the lowered runtime tape -/

/-- Apply the names and gradient mask recorded for typed-graph leaves to a lowered tape. -/
def applyLeafMetadata {α : Type} (metadata : Array LeafMetadata)
    (tape : Runtime.Autograd.Tape α) : Runtime.Autograd.Tape α :=
  { nodes := tape.nodes.mapIdx fun id node =>
      match metadata[id]? with
      | some leaf => { node with name := leaf.name, requiresGrad := leaf.requiresGrad }
      | none => node }

/--
Lower the recorded executable graph into a runtime tape and restore its leaf metadata.

The shape-indexed graph deliberately contains only mathematical leaf values. Names and
`requiresGrad` are runtime concerns, so the session stores them alongside the graph and attaches
them here. A malformed internal snapshot is rejected instead of silently changing gradient
behavior.
-/
def lowerTape {α : Type} [DecidableEq Shape]
    (st : TypedGraphSessionState α) : Runtime.Autograd.Result (Runtime.Autograd.Tape α) := do
  if st.leafMetadata.size != st.Γ.length then
    throw "typed graph session: leaf metadata is not aligned with the typed leaf context"
  let tape :=
    (Proofs.Autograd.Algebra.Graph.lowerGraphDataToTape (α := α) (Δ := NatEnv)
      (Γ := st.Γ) (ss := st.ss) st.g st.x st.nat).1
  pure (applyLeafMetadata st.leafMetadata tape)

/--
Run reverse-mode backprop for the whole recorded context and return a dense gradient array.

`seed` is the upstream gradient for `out` (same convention as PyTorch's
  `loss.backward(gradient=...)`).
-/
def backwardDenseAll {α : Type} (s : TypedGraphSession α) [Add α] [Zero α] [DecidableEq Shape]
  {sh : Shape} (out : TensorRef α sh) (seed : Tensor α sh) :
  IO (Array (Spec.PackedTensor α)) := do
  let st0 ← s.st.get
  let output ← okOrThrow (mkIdxOrThrow (_α := α) (Γ := st0.Γ) (ss := st0.ss) out.id sh)
  let t ← okOrThrow (lowerTape (α := α) (st := st0))
  okOrThrow (Runtime.Autograd.TypedGraph.backwardDenseAllFrom
    (α := α) (Γ := st0.Γ) (ss := st0.ss) t output seed)

/--
Run backward from a scalar loss with seed `1`.

PyTorch comparison: `loss.backward()` for a scalar loss.
-/
def backwardScalarDenseAll {α : Type} (s : TypedGraphSession α) [Add α] [Zero α] [One α] [DecidableEq Shape]
  (loss : TensorRef α Shape.scalar) : IO (Array (Spec.PackedTensor α)) :=
  backwardDenseAll (α := α) s (sh := Shape.scalar) loss (Tensor.scalar (1 : α))

/--
Extract the gradient tensor for a particular `TensorRef` from a dense gradient array.

This is the typed analogue of looking up `grads[x.id]` and casting it to the expected shape.
-/
def grad {α : Type} {sh : Shape} [DecidableEq Shape]
  (grads : Array (Spec.PackedTensor α)) (x : TensorRef α sh) : IO (Tensor α sh) := do
  let gAny ← match grads[x.id]? with
    | some g => pure g
    | none => throw <| IO.userError "torch(TypedGraphSession): gradient array out of bounds"
    if h : gAny.shape = sh then
      pure (gAny.cast h)
    else
      throw <| IO.userError <|
        s!"torch(TypedGraphSession): grad shape mismatch (expected {Shape.pretty sh}, got "
          ++ s!"{Shape.pretty gAny.shape})"

/-! ## Forward-mode JVP over the typed graph -/

/-- Like `mkIdxOrThrow`, but restricted to leaves `Γ` only. -/
def mkLeafIdxOrThrow {_α : Type} {Γ : List Shape} (id : Nat) (s : Shape) :
    Runtime.Autograd.Result (_root_.Proofs.Autograd.Algebra.Idx Γ s) := by
    if h : id < Γ.length then
      let fin : Fin Γ.length := ⟨id, h⟩
      let got : Shape := Γ.get fin
      if hg : got = s then
        exact .ok ⟨fin, hg⟩
      else
        exact .error <|
          s!"torch(TypedGraphSession): leaf shape mismatch at id={id}: expected {Shape.pretty s}, got "
            ++ s!"{Shape.pretty got}"
  else
    exact .error s!"torch(TypedGraphSession): invalid leaf id={id} for leafLen={Γ.length}"

/--
Convert a dense tangent array (aligned with leaf creation order) into a typed `TList α Γ`.

This is the main adapter needed to call the proved `GraphData.jvpCtx` forward-mode routine.
-/
def dxTListFromAnyArray {α : Type} [Zero α] [DecidableEq Shape]
    (Γ : List Shape) (dxs : Array (Spec.PackedTensor α)) :
    IO (_root_.Proofs.Autograd.Algebra.TList α Γ) := do
  if _hlen : dxs.size = Γ.length then
    let rec go : (Γ' : List Shape) → (off : Nat) → IO (_root_.Proofs.Autograd.Algebra.TList α Γ')
      | [], _ => pure .nil
      | s :: ss, off => do
          let any ← match dxs[off]? with
            | some v => pure v
            | none => throw <| IO.userError "torch(TypedGraphSession): dx array out of bounds"
            if hs : any.shape = s then
              let t : Tensor α s := any.cast hs
              pure (.cons t (← go ss (off + 1)))
            else
              throw <| IO.userError <|
                s!"torch(TypedGraphSession): dx shape mismatch at idx={off} (expected "
                  ++ s!"{Shape.pretty s}, got "
                  ++ s!"{Shape.pretty any.shape})"
    go Γ 0
  else
    throw <| IO.userError
      s!"torch(TypedGraphSession): dx array size mismatch (expected {Γ.length}, got {dxs.size})"

/--
Jacobian-vector product for the current session snapshot.

`dxs` is a dense array of tangents for leaf tensors, aligned with leaf creation order.
-/
def jvpDenseAll {α : Type} (s : TypedGraphSession α) [Zero α] [DecidableEq Shape]
    {sh : Shape} (out : TensorRef α sh) (dxs : Array (Spec.PackedTensor α)) :
    IO (Tensor α sh) := do
  let st0 ← s.st.get
  let dx ← dxTListFromAnyArray (α := α) (Γ := st0.Γ) dxs
  let dctx : _root_.Proofs.Autograd.Algebra.TList α (st0.Γ ++ st0.ss) :=
    _root_.Proofs.Autograd.Algebra.GraphData.jvpCtx (α := α) (Δ := NatEnv) (Γ := st0.Γ) (ss :=
      st0.ss)
      st0.g st0.x dx st0.nat
  let idx ← okOrThrow (mkIdxOrThrow (_α := α) (Γ := st0.Γ) (ss := st0.ss) out.id sh)
  pure (_root_.Proofs.Autograd.Algebra.getIdx (α := α) (xs := dctx) idx)

/-- JVP for a single leaf: tangent is nonzero only at `x`. -/
def jvpLeaf {α : Type} (s : TypedGraphSession α) [Zero α] [DecidableEq Shape]
    {shOut shX : Shape}
    (out : TensorRef α shOut) (x : TensorRef α shX) (dx : Tensor α shX) :
    IO (Tensor α shOut) := do
  let st0 ← s.st.get
  let idxX ← okOrThrow (mkLeafIdxOrThrow (_α := α) (Γ := st0.Γ) x.id shX)
  let dxAll : _root_.Proofs.Autograd.Algebra.TList α st0.Γ :=
    _root_.Proofs.Autograd.Algebra.TList.single (α := α) (Γ := st0.Γ) (s := shX) idxX dx
  let dxs : Array (Spec.PackedTensor α) :=
    _root_.Proofs.Autograd.Algebra.TList.toPackedArray (α := α) (ss := st0.Γ) dxAll
  jvpDenseAll (α := α) (sh := shOut) s out dxs

/-- Scalar-loss JVP for a single leaf. -/
def jvpScalarLeaf {α : Type} (s : TypedGraphSession α) [Zero α] [DecidableEq Shape]
    (loss : TensorRef α Shape.scalar) {shX : Shape} (x : TensorRef α shX) (dx : Tensor α shX) :
    IO α := do
  let dl ← jvpLeaf (α := α) s (shOut := Shape.scalar) (shX := shX) loss x dx
  match dl with
  | .scalar a => pure a

/--
Apply an SGD update to all parameters recorded via `use`.

`grads` is expected to be the dense gradient array returned by `backwardDenseAll` /
`backwardScalarDenseAll`. Only entries corresponding to parameters (leaves that were produced by
`use`) are used to update `Param.value`.
PyTorch comparison: like iterating `params` and doing `p.data -= lr * p.grad`.
-/
def sgdStepAll {α : Type} (s : TypedGraphSession α)
  [Sub α] [Mul α] [Add α] [Zero α] [DecidableEq Shape]
  (lr : α) (grads : Array (Spec.PackedTensor α)) : IO Unit := do
  let m ← s.paramsByLeaf.get
  for (id, p) in m.toList.filter (fun entry => entry.2.requiresGrad) do
    let gAny ← match grads[id]? with
      | some g => pure g
      | none => throw <| IO.userError "torch(TypedGraphSession): gradient array out of bounds during SGD"
    if hs : gAny.shape = p.s then
      let pv ← p.get
      if hp : pv.shape = p.s then
        let pvT : Tensor α p.s := pv.cast hp
        let gT : Tensor α p.s := gAny.cast hs
        let updated : Tensor α p.s :=
          Tensor.materialize <| subSpec pvT (scaleSpec (α := α) (s := p.s) gT lr)
        p.set (Spec.PackedTensor.ofTensor updated)
      else
        throw <| IO.userError "torch(TypedGraphSession): internal param shape mismatch"
    else
      throw <| IO.userError "torch(TypedGraphSession): internal grad shape mismatch during SGD"

/-! ## Lowered-tape equivalence -/

/--
Running the runtime reverse-mode loop on the lowered tape equals `GraphData` backpropagation.

`lowerGraphDataToTape` produces a tape, and `Tape.backwardDenseFrom` is equal to
`GraphData.backpropAllCtx` up to the `TList.toPackedArray` representation change. This theorem proves
the lowering faithful to the stored VJP program. It does not prove that the VJP is the derivative
of the stored forward function; that stronger statement requires proof-carrying `Node`s.
-/
theorem backwardDenseFrom_lowerGraphDataToTape_eq_backpropAllCtx
    {α : Type} [DecidableEq Shape] [CommSemiring α]
    (st : TypedGraphSessionState α) (seed : _root_.Proofs.Autograd.Algebra.TList α (st.Γ ++ st.ss)) :
    Runtime.Autograd.Tape.backwardDenseFrom
        (t := (Proofs.Autograd.Algebra.Graph.lowerGraphDataToTape (α := α) (Δ := NatEnv) (Γ := st.Γ) (ss
          := st.ss) st.g st.x st.nat).1)
        (grads0 := _root_.Proofs.Autograd.Algebra.TList.toPackedArray (α := α) (ss := st.Γ ++ st.ss)
          seed)
      =
      .ok
        (_root_.Proofs.Autograd.Algebra.TList.toPackedArray (α := α) (ss := st.Γ ++ st.ss)
          (_root_.Proofs.Autograd.Algebra.GraphData.backpropAllCtx (α := α) (Δ := NatEnv) (Γ :=
            st.Γ) (ss := st.ss) st.g st.x st.nat seed)) := by
  simpa using
    (Proofs.Autograd.Algebra.Graph.backwardDenseFrom_lowerGraphDataToTape_eq_backpropAllCtx
      (α := α) (Δ := NatEnv) (Γ := st.Γ) (ss := st.ss) st.g st.x st.nat seed)

end TypedGraphSession

end Internal
end Torch
end Autograd
end Runtime
