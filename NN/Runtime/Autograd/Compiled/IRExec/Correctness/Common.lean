/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Compiled.IRExec

/-!
# Common

Internal helper lemmas for `NN.Runtime.Autograd.Compiled.IRExec.Correctness`.

These lemmas relate the typed runtime context (`TList`) to the untyped IR value table (`Array
  DVal`),
and provide small “building block” correctness steps that are reused across the per-op proofs.

Reading map:

* `dValsOfCtx*` lemmas: relate the typed context produced by `GraphData.eval` to an untyped
  `Array (NN.IR.DVal α)` (this is what the IR evaluator uses).
* `denoteAllState*` lemmas: package the compiled evaluator (`ExecGraphData.denoteAll`) in the form
  expected by IR-style semantic equivalence proofs.

These lemmas are infrastructure: they should not encode op-specific logic. Per-op correctness files
(Matmul/Pool2d/LayerNorm/MSELoss) should depend on this module and not re-prove these bridges.

## Main definitions

- `throw_bind_ne_ok`: eliminates impossible success branches after `throw`.
- `NoMSELoss`: side condition for semantic equivalence theorems over fragments that exclude `.mse_loss`.
- `dValsOfCtx_*`: typed-context to IR-array bridge lemmas.
- `denoteAllState_*` helpers: semantic equivalence bridges between compiled state and IR denotation tables.

## Implementation notes

- This module is shared infrastructure: predictable proof contracts
  matter more than clever proof tricks.
- Many lemmas here are proof-irrelevance/indexing bridges; these are repetitive but they remove a
  lot of friction from op-specific proofs.
- Collecting these utilities in one place gives op-specific correctness modules shared rewrite and
  indexing lemmas instead of repeated local proof scripts.
- These files can build slowly because they connect two representations at once: typed `TList`
  contexts on the compiled side and dynamically shaped `DVal` arrays on the IR side. Most of the
  cost is not arithmetic; it is Lean checking that shape casts, array indices, and proof-irrelevant
  casts line up exactly.
- When the same proof pattern appears in multiple operator files, prefer a named lemma with a clear
  contract over another local `simp` script.

## Tags

correctness, infrastructure, tlist, dval, bridge-lemmas
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Compiled

open Spec
open Tensor
open Proofs.Autograd.Algebra
/-- `throw msg` in the `Except` monad is `.error msg`. -/
theorem throw_eq_error {β : Type} (msg : String) :
    (throw msg : Except String β) = .error msg := by
  simp [throw, throwThe, MonadExceptOf.throw]

open NN.IR
open IRExec

/-!
## Shared side conditions

These predicates describe the exact fragment covered by a theorem. Keeping them in `Common` lets
the per-op lemmas, the semantic equivalence proof, and the chapter index refer to the same public contract
without import cycles.
-/

/--
Core semantic equivalence side condition: the IR graph contains no `.mse_loss` nodes.

The compiled runtime has a correct `.mse_loss` step lemma in `Correctness.Ops.Loss`; the existing
end-to-end semantic equivalence theorem keeps this condition so its branch proof stays small and predictable.
-/
def NoMSELoss (g : NN.IR.Graph) : Prop :=
  ∀ i n, g.getNode i = .ok n → n.kind ≠ .mseLoss

/--
If a `do`-chain begins with `throw`, it cannot produce an `.ok` result.

This lemma is used throughout the compiled-correctness proofs to close
impossible branches where compilation would have thrown an error message.
-/
theorem throw_bind_ne_ok {β γ : Type} {msg : String} {k : β → Except String γ} {v : γ}
    (h : (do
      let y ← (throw msg : Except String β)
      k y) = Except.ok v) : False := by
  simp [throw_eq_error] at h

/-- If two unit guards and a tail computation return `.ok`, then the first guard succeeded. -/
theorem exceptUnit_two_bind_first_ok
    {β : Type} {e₁ e₂ : Except String Unit} {next : Except String β} {v : β}
    (h : (do let _ ← e₁; let _ ← e₂; next) = Except.ok v) :
    e₁ = Except.ok () := by
  cases h₁ : e₁ <;> simp [h₁] at h
  rename_i u
  cases u
  rfl

/-- If two unit guards and a tail computation return `.ok`, then the second guard succeeded. -/
theorem exceptUnit_two_bind_second_ok
    {β : Type} {e₁ e₂ : Except String Unit} {next : Except String β} {v : β}
    (h : (do let _ ← e₁; let _ ← e₂; next) = Except.ok v) :
    e₂ = Except.ok () := by
  cases h₁ : e₁ <;> simp [h₁] at h
  rename_i u₁
  cases u₁
  cases h₂ : e₂ <;> simp [h₂] at h
  rename_i u₂
  cases u₂
  rfl

/-- If two unit guards and a tail computation return `.ok`, then the tail returned `.ok`. -/
theorem exceptUnit_two_bind_tail_ok
    {β : Type} {e₁ e₂ : Except String Unit} {next : Except String β} {v : β}
    (h : (do let _ ← e₁; let _ ← e₂; next) = Except.ok v) :
    next = Except.ok v := by
  cases h₁ : e₁ <;> simp [h₁] at h
  rename_i u₁
  cases u₁
  cases h₂ : e₂ <;> simp [h₂] at h
  rename_i u₂
  cases u₂
  exact h

/--
Array indexing is proof-irrelevant.

This is a small technical lemma: in Lean, `xs[i]'h` carries a proof `h : i < xs.size`. Different
proofs should not change the value returned by indexing.
-/
theorem array_getElem_proof_irrel {β : Type}
    (xs : Array β) (i : Nat) (h₁ h₂ : i < xs.size) : xs[i]'h₁ = xs[i]'h₂ := by
  -- `Array.getElem` is implemented via `Array.get` on a `Fin` index, and `Fin` is proof-irrelevant.
  have hFin : (⟨i, h₁⟩ : Fin xs.size) = ⟨i, h₂⟩ := by
    ext
    rfl
  -- Use `Fin` indexing (`xs[j]`) since `Array.get` is not a named constant in Lean 4.
  exact congrArg (fun j : Fin xs.size => xs[j]) hFin

/--
Relate `xs[i]!` (defaulting lookup) and `xs[i]'h` (bounded lookup) when the index is in-bounds.

This is a small bridge lemma used throughout the IR/runtime context comparison proofs.
-/
theorem array_getElem!_eq_getElem {β : Type} [Inhabited β]
    (xs : Array β) (i : Nat) (h : i < xs.size) : xs[i]! = xs[i]'h := by
  have h1 : xs[i]! = xs.getD i default := by
    simp [Array.getElem!_eq_getD]
  have h2 : xs[i]'h = xs.getD i default := by
    simpa using (Array.getElem_eq_getD (xs := xs) (i := i) (h := h) (fallback := (default : β)))
  simpa [h2] using h1

/--
`dValsOfCtx` ignores type-level casts of the underlying `TList`.

`GraphData.eval` introduces a definitional cast when extending contexts; this lemma lets us erase it
before reasoning about the corresponding `Array` of `DVal`s.
-/
@[simp]
theorem dValsOfCtx_cast {α : Type} [Context α] {ss₁ ss₂ : List Shape}
    (h : ss₁ = ss₂) (ctx : Proofs.Autograd.Algebra.TList α ss₁) :
    dValsOfCtx (α := α) (ss := ss₂) (Proofs.Autograd.Algebra.TList.cast (α := α) h ctx) =
      dValsOfCtx (α := α) (ss := ss₁) ctx := by
  cases h
  simp [dValsOfCtx]

/-- `dValsOfCtx` for a snoc’d context corresponds to `Array.push` of the appended tensor. -/
@[simp]
theorem dValsOfCtx_snoc {α : Type} [Context α] {ss : List Shape} {τ : Shape}
    (ctx : Proofs.Autograd.Algebra.TList α ss) (t : Tensor α τ) :
    dValsOfCtx (α := α) (ss := ss ++ [τ])
        (Proofs.Autograd.Algebra.TList.snoc (α := α) (ss := ss) (τ := τ) ctx t) =
      (dValsOfCtx (α := α) (ss := ss) ctx).push (NN.IR.DVal.mk (α := α) τ t) := by
  simp [dValsOfCtx, dValOfAny, NN.IR.DVal.mk, AnyTensor.mk]

/--
Indexing `dValsOfCtx` agrees with indexing the underlying `TList` context.

This is the main bridge between the typed runtime context and the untyped IR value table.
-/
theorem dValsOfCtx_getElem!
    {α : Type} [Context α] [Inhabited α] {ss : List Shape}
    (ctx : Proofs.Autograd.Algebra.TList α ss) (i : Fin ss.length) :
    (dValsOfCtx (α := α) (ss := ss) ctx)[i.1]! =
      NN.IR.DVal.mk (α := α) (ss.get i)
        (Proofs.Autograd.Algebra.TList.get (α := α) (ss := ss) ctx i) := by
  let vals := dValsOfCtx (α := α) (ss := ss) ctx
  have hSize : vals.size = ss.length := by
    simp [vals, dValsOfCtx, Proofs.Autograd.Algebra.TList.size_toAnyArray]
  have hi : i.1 < vals.size := Nat.lt_of_lt_of_eq i.2 hSize.symm
  have hGet : vals[i.1]! = vals[i.1]'hi := by
    simpa [vals] using (array_getElem!_eq_getElem (xs := vals) (i := i.1) (h := hi))
  -- Reduce to a bounded lookup through the mapped `toAnyArray`.
  -- Then use the existing `get_toAnyArray` bridge lemma.
  let arr := Proofs.Autograd.Algebra.TList.toAnyArray (α := α) (ss := ss) ctx
  have hArr : i.1 < arr.size := by
    -- `arr.size = ss.length` by `size_toAnyArray`.
    have : arr.size = ss.length := by
      simp [arr, Proofs.Autograd.Algebra.TList.size_toAnyArray]
    exact Nat.lt_of_lt_of_eq i.2 this.symm
  have hMapped :
      vals[i.1]'hi =
        dValOfAny (α := α)
          (arr[i.1]'hArr) := by
    -- `vals = arr.map dValOfAny`, so this is `Array.getElem_map` plus proof irrelevance.
    have hiArr : i.1 < arr.size := Nat.lt_of_lt_of_eq hi (by
      simp [vals, dValsOfCtx, arr])
    have : vals[i.1]'hi = dValOfAny (α := α) (arr[i.1]'hiArr) := by
      simp [vals, dValsOfCtx, arr]
    -- Normalize the inner proof to `hArr`.
    simpa [array_getElem_proof_irrel (xs := arr) (i := i.1) (h₁ := hiArr) (h₂ := hArr)] using this
  -- Rewrite `arr[i]` via `get_toAnyArray`, then finish by `simp`.
  have hToAny :
      arr[i.1]'hArr =
        Runtime.Autograd.AnyTensor.mk
          (Proofs.Autograd.Algebra.TList.get (α := α) (ss := ss) ctx i) := by
    -- `get_toAnyArray` uses a particular bound proof; adjust it by proof irrelevance.
    have hStd :
        arr[i.1]'(by
          -- the proof used by `get_toAnyArray`
          dsimp [arr]
          exact Nat.lt_of_lt_of_eq i.2
            (Proofs.Autograd.Algebra.TList.size_toAnyArray (α := α) (ss := ss) ctx).symm) =
          Runtime.Autograd.AnyTensor.mk
            (Proofs.Autograd.Algebra.TList.get (α := α) (ss := ss) ctx i) := by
      simpa [arr] using
        (Proofs.Autograd.Algebra.TList.get_toAnyArray (α := α) (ss := ss) ctx i)
    -- Normalize the proof.
    simpa [array_getElem_proof_irrel (xs := arr) (i := i.1)
        (h₁ := hArr)
        (h₂ := (by
          dsimp [arr]
          exact Nat.lt_of_lt_of_eq i.2
            (Proofs.Autograd.Algebra.TList.size_toAnyArray (α := α) (ss := ss) ctx).symm))] using
              hStd
  -- Assemble.
  calc
    vals[i.1]! = vals[i.1]'hi := by
      simpa [vals] using hGet
    _ = dValOfAny (α := α) (arr[i.1]'hArr) := by
      simp [hMapped]
    _ = dValOfAny (α := α)
        (Runtime.Autograd.AnyTensor.mk
          (Proofs.Autograd.Algebra.TList.get (α := α) (ss := ss) ctx i)) := by
      simp [hToAny]
    _ = NN.IR.DVal.mk (α := α) (ss.get i)
        (Proofs.Autograd.Algebra.TList.get (α := α) (ss := ss) ctx i) := by
      rfl

/--
Indexing `dValsOfCtx` by a typed `Idx` agrees with `getIdx` on the underlying `TList`.

This packages `dValsOfCtx_getElem!` into the repository’s `Idx` wrapper.
-/
theorem dValsOfCtx_getIdx
    {α : Type} [Context α] [Inhabited α] {ss : List Shape} {s : Shape}
    (ctx : Proofs.Autograd.Algebra.TList α ss) (idx : Idx ss s) :
    (dValsOfCtx (α := α) (ss := ss) ctx)[idx.i.1]! =
      NN.IR.DVal.mk (α := α) s (getIdx (α := α) (xs := ctx) idx) := by
  cases idx with
  | mk i h =>
      -- Reduce to the `Fin`-indexed lemma and then specialize with the stored shape equality.
      cases h
      simpa [getIdx, Tensor.castShape, NN.IR.DVal.mk] using
        (dValsOfCtx_getElem! (α := α) (ss := ss) ctx i)

/-- `Graph.expectShape` succeeds on a `DVal` built with the same shape. -/
@[simp] theorem Graph.expectShape_mk {α : Type} [Context α] [DecidableEq Shape] {s : Shape}
    (t : Tensor α s) :
    NN.IR.Graph.expectShape (α := α) (expected := s) (NN.IR.DVal.mk (α := α) s t) = .ok t := by
  simp [NN.IR.Graph.expectShape, NN.IR.DVal.shape, NN.IR.DVal.tensor, NN.IR.DVal.mk]
  rfl

/-- Same as `Graph.expectShape_mk`, but for the sigma-style constructor `⟨s, t⟩`. -/
@[simp] theorem Graph.expectShape_sigma {α : Type} [Context α] [DecidableEq Shape] {s : Shape}
    (t : Tensor α s) :
    NN.IR.Graph.expectShape (α := α) (expected := s) (⟨s, t⟩ : NN.IR.DVal α) = .ok t := by
  simp [NN.IR.Graph.expectShape, NN.IR.DVal.shape, NN.IR.DVal.tensor]
  rfl

attribute [grind =] dValsOfCtx_cast dValsOfCtx_snoc Graph.expectShape_mk Graph.expectShape_sigma
  throw_eq_error array_getElem_proof_irrel array_getElem!_eq_getElem
  dValsOfCtx_getElem! dValsOfCtx_getIdx

/--
`NN.IR.Graph.evalAt` for a `.matmul` node specialized to 2D matrix multiply.

This is a proof-only helper that records the exact `Spec.mat_mul_spec` term produced by `evalAt`
in the well-typed success case.
-/
theorem evalAt_matmul_mm_ok
    {α : Type} [Context α] [Inhabited α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) (input : NN.IR.DVal α) (vals : Array (NN.IR.DVal α))
    (i : Nat) (n : NN.IR.Node) (aId bId : Nat) (m nDim p : Nat)
    (aT : Tensor α (.dim m (.dim nDim .scalar)))
    (bT : Tensor α (.dim nDim (.dim p .scalar)))
    (hN : g.getNode i = .ok n) (hk : n.kind = .matmul) (hp : n.parents = [aId, bId])
    (hGetA : vals[aId]! = NN.IR.DVal.mk (α := α) (.dim m (.dim nDim .scalar)) aT)
    (hGetB : vals[bId]! = NN.IR.DVal.mk (α := α) (.dim nDim (.dim p .scalar)) bT)
    (hOut : (.dim m (.dim p .scalar)) = n.outShape) :
    NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload) (input := input) (vals := vals) (i :=
      i) =
      .ok (NN.IR.DVal.mk (α := α) n.outShape
        (hOut ▸ Spec.matMulSpec (α := α) (m := m) (n := nDim) (p := p) aT bT)) := by
    simp [NN.IR.Graph.evalAt, hN, hk, hp, hGetA, hGetB, hOut, throw_eq_error]

/--
`NN.IR.Graph.evalAt` for a `.matmul` node specialized to batched matmul (`bmm`).

Like `evalAt_matmul_mm_ok`, this is used to relate the IR evaluator’s result to the compiled node’s
`forward` closure during the semantic equivalence correctness proof.
-/
theorem evalAt_matmul_bmm_ok
    {α : Type} [Context α] [Inhabited α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) (input : NN.IR.DVal α) (vals : Array (NN.IR.DVal α))
    (i : Nat) (n : NN.IR.Node) (aId bId : Nat) (batch m nDim p : Nat)
    (aT : Tensor α (.dim batch (.dim m (.dim nDim .scalar))))
    (bT : Tensor α (.dim batch (.dim nDim (.dim p .scalar))))
    (hN : g.getNode i = .ok n) (hk : n.kind = .matmul) (hp : n.parents = [aId, bId])
    (hGetA : vals[aId]! =
      NN.IR.DVal.mk (α := α) (.dim batch (.dim m (.dim nDim .scalar)))
        aT)
    (hGetB : vals[bId]! =
      NN.IR.DVal.mk (α := α) (.dim batch (.dim nDim (.dim p .scalar)))
        bT)
    (hOut : (.dim batch (.dim m (.dim p .scalar))) = n.outShape) :
    NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload) (input := input) (vals := vals) (i :=
      i) =
      .ok (NN.IR.DVal.mk (α := α) n.outShape
        (hOut ▸ Tensor.bmmSpec (α := α) (batch := batch) (m := m) (n := nDim) (p := p) aT bT)) :=
          by
    simp [NN.IR.Graph.evalAt, hN, hk, hp, hGetA, hGetB, hOut, throw_eq_error]

/--
`NN.IR.Graph.evalAt` for a `.reduceSum axis` node, specialized to a well-typed success case.

This helper records the exact `Tensor.reduceSum` term produced by the IR evaluator once:
- the parent has the expected shape `s`,
- the axis validity check succeeds, and
- the node's declared `outShape` matches `shapeAfterSum s axis`.

The final cast to `n.outShape` comes from the `evalAt` "shape-tag normalization" step.
-/
theorem evalAt_reduceSum_ok
    {α : Type} [Context α] [Inhabited α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) (input : NN.IR.DVal α) (vals : Array (NN.IR.DVal α))
    (i : Nat) (n : NN.IR.Node) (pId : Nat) (axis : Nat)
    (s : Shape) (pT : Tensor α s) (hAxisPf : PLift (Shape.valid_axis axis s))
    (hN : g.getNode i = .ok n) (hk : n.kind = .reduceSum axis) (hp : n.parents = [pId])
    (hGet : vals[pId]! = NN.IR.DVal.mk (α := α) s pT)
    (hAxis : NN.IR.Graph.mkValidAxis? (axis := axis) s = some hAxisPf)
    (hOut : Spec.Tensor.shapeAfterSum s axis = n.outShape) :
    NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload) (input := input) (vals := vals) (i :=
      i) =
      .ok (NN.IR.DVal.mk (α := α) n.outShape
        (hOut ▸ Tensor.reduceSum (α := α) (s := s) axis pT
          (Shape.proveReducibleAlong axis s hAxisPf.down))) := by
  have hGetSigma : vals[pId]! = (⟨s, pT⟩ : NN.IR.DVal α) := by
    simpa [NN.IR.DVal.mk] using hGet
  have hFst : vals[pId]!.fst = s := by
    simpa using congrArg Sigma.fst hGetSigma
  have hSndHeq : HEq vals[pId]!.snd pT := by
    rw [hGetSigma]
  cases hFst
  have hSnd : vals[pId]!.snd = pT := by
    exact eq_of_heq hSndHeq
  -- Unfold to the `.reduceSum` branch and discharge the runtime checks (axis validity + outShape).
  simp [NN.IR.Graph.evalAt, hN, hk, hp, throw_eq_error, hAxis, hOut, hSnd, Pure.pure, Except.pure]

/--
`NN.IR.Graph.evalAt` for a `.reduceMean axis` node, specialized to a well-typed success case.

This is the mean analogue of `evalAt_reduceSum_ok`.
-/
theorem evalAt_reduceMean_ok
    {α : Type} [Context α] [Inhabited α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) (input : NN.IR.DVal α) (vals : Array (NN.IR.DVal α))
    (i : Nat) (n : NN.IR.Node) (pId : Nat) (axis : Nat)
    (s : Shape) (pT : Tensor α s) (hAxisPf : PLift (Shape.valid_axis axis s))
    (hN : g.getNode i = .ok n) (hk : n.kind = .reduceMean axis) (hp : n.parents = [pId])
    (hGet : vals[pId]! = NN.IR.DVal.mk (α := α) s pT)
    (hAxis : NN.IR.Graph.mkValidAxis? (axis := axis) s = some hAxisPf)
    (hOut : Spec.Tensor.shapeAfterSum s axis = n.outShape) :
    NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload) (input := input) (vals := vals) (i :=
      i) =
      .ok (NN.IR.DVal.mk (α := α) n.outShape
        (hOut ▸ Tensor.reduceMean (α := α) (s := s) axis pT
          (Shape.proveReducibleAlong axis s hAxisPf.down))) := by
  have hGetSigma : vals[pId]! = (⟨s, pT⟩ : NN.IR.DVal α) := by
    simpa [NN.IR.DVal.mk] using hGet
  have hFst : vals[pId]!.fst = s := by
    simpa using congrArg Sigma.fst hGetSigma
  have hSndHeq : HEq vals[pId]!.snd pT := by
    rw [hGetSigma]
  cases hFst
  have hSnd : vals[pId]!.snd = pT := by
    exact eq_of_heq hSndHeq
  simp [NN.IR.Graph.evalAt, hN, hk, hp, throw_eq_error, hAxis, hOut, hSnd, Pure.pure, Except.pure]

/-- Repackage a compiled `State` as an `ExecGraphData` so we can call its evaluator helpers. -/
def execOfState {α : Type} (inShape : Shape) (st : State α inShape) : ExecGraphData α :=
  { inShape := inShape, ss := st.1, g := st.2 }

/-- Evaluate the compiled prefix state and convert its typed runtime context into an IR-style table.
  -/
def denoteAllState {α : Type} [Context α] (inShape : Shape) (st : State α inShape)
    (x : Tensor α inShape) : Array (NN.IR.DVal α) :=
  ExecGraphData.denoteAll (α := α) (e := execOfState (α := α) inShape st) x

/--
`denoteAllState` commutes with extending the SSA graph by one node (`GraphData.snoc`).

This is the key step for proving that the compiler’s prefix-building loop stays in semantic equivalence with the
IR denotation table.
-/
theorem denoteAllState_snoc {α : Type} [Context α]
    {inShape : Shape} {ss : List Shape} {τ : Shape}
    (gd : GraphData α Unit [inShape] ss)
    (nodeData : NodeData α Unit ([inShape] ++ ss) τ)
    (x : Tensor α inShape) :
    let st : State α inShape := ⟨ss, gd⟩
    let st' : State α inShape := ⟨ss ++ [τ], .snoc (ss := ss) gd nodeData⟩
    denoteAllState (α := α) inShape st' x =
      (denoteAllState (α := α) inShape st x).push
        (NN.IR.DVal.mk (α := α) τ (nodeData.forward (GraphData.eval (ss := ss) gd (.cons x .nil) ())
          ())) := by
  -- Expand `st`/`st'`.
  simp only
  -- Reduce both sides to `dValsOfCtx` of `GraphData.eval`.
  simp [denoteAllState, execOfState, ExecGraphData.denoteAll, ExecGraphData.eval]
  -- Now unfold `GraphData.eval` for the snoc graph.
  simp [GraphData.eval]
  -- At this point the goal is exactly the `dValsOfCtx_snoc` lemma, up to unfolding `DVal.mk`.
  simpa [NN.IR.DVal.mk] using
    (dValsOfCtx_snoc (α := α) (ss := ([inShape] ++ ss)) (τ := τ)
      (ctx := GraphData.eval (ss := ss) gd (.cons x .nil) ())
      (t := nodeData.forward (GraphData.eval (ss := ss) gd (.cons x .nil) ()) ()))

/--
Build a typed runtime index (`Idx`) for a numeric IR parent id.

The compiled runtime context is typed by a list of shapes `[inShape] ++ ss`. `mkIdx` checks that:
- `id` is in bounds, and
- the context shape at that position matches the expected shape `s`.
-/

theorem mkIdx_ok_i_eq
    [DecidableEq Shape] {inShape : Shape} {ss : List Shape} {id : Nat} {s : Shape}
    {idx : Idx ([inShape] ++ ss) s}
    (h : mkIdx (inShape := inShape) (ss := ss) id s = .ok idx) :
    idx.i.1 = id := by
  classical
  unfold mkIdx at h
  -- After unfolding, the bound check is expressed via `id ≤ ss.length` (since the ctx is `inShape
  -- :: ss`).
  by_cases hBound : id ≤ ss.length
  · have hLt : id < (inShape :: ss).length := by
      simpa using Nat.lt_succ_of_le hBound
    simp [hBound] at h
    by_cases hShape : (inShape :: ss)[id]'hLt = s
    · simp [hShape] at h
      cases h
      rfl
    · simp [hShape] at h
  · simp [hBound] at h

/--
Lookup lemma: `denoteAllState[..][pid]!` agrees with `getIdx` when `mkIdx pid s` succeeds.

This is used when proving correctness of the per-node compiler step: we translate parent ids in the
IR into typed indices into the compiled context.
-/
  theorem denoteAllState_get_mkIdx
    {α : Type} [Context α] [DecidableEq Shape] {inShape : Shape} {ss : List Shape}
    (gd : GraphData α Unit [inShape] ss) (x : Tensor α inShape)
    {pid : Nat} {s : Shape} {idx : Idx ([inShape] ++ ss) s}
    (hIdx : mkIdx (inShape := inShape) (ss := ss) pid s = .ok idx) :
    (denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x)[pid]! =
      NN.IR.DVal.mk (α := α) s
        (getIdx (α := α)
          (xs := GraphData.eval (α := α) (Δ := Unit) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)
            ()) idx) := by
  -- Unfold `denoteAllState` to `dValsOfCtx (GraphData.eval ...)`, then use `dValsOfCtx_getIdx`.
  have hPid : pid = idx.i.1 :=
    (mkIdx_ok_i_eq (inShape := inShape) (ss := ss) (id := pid) (s := s) (idx := idx) hIdx).symm
  simp [denoteAllState, execOfState, ExecGraphData.denoteAll, ExecGraphData.eval, dValsOfCtx, hPid]
  simpa using (dValsOfCtx_getIdx (α := α)
    (ctx := GraphData.eval (α := α) (Δ := Unit) (Γ := [inShape]) (ss := ss) gd (.cons x .nil) ())
      idx)

/--
One-step finishing lemma for the `buildFrom`/`denoteAllFrom` semantic equivalence proof.

If we know:
- the tail recursion `i+1` is correct (`hTail`),
- the IR evaluator step at `i` matches the compiled node’s `forward` (`hEval`), and
- the compiled table at `i` is the previous table plus the pushed node value (`hStep`),
then `denoteAllFrom` at `i` returns the final compiled table.
-/
theorem buildFrom_denoteAllFrom_finish
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (i : Nat) (x : Tensor α inShape)
    (hi : i < g.nodes.size)
    (τ : Shape) (nodeData : NodeData α Unit ([inShape] ++ ss) τ)
    (st1 st' : State α inShape)
    (ctx : TList α ([inShape] ++ ss))
    (vals0 : Array (NN.IR.DVal α))
    (input : NN.IR.DVal α)
    (hTail :
      NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := input) (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
        .ok (denoteAllState (α := α) inShape st' x))
    (hEval :
      NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
          (input := input) (vals := vals0) (i := i) =
        .ok (NN.IR.DVal.mk (α := α) τ (nodeData.forward ctx ())))
    (hStep :
      denoteAllState (α := α) inShape st1 x =
        vals0.push (NN.IR.DVal.mk (α := α) τ (nodeData.forward ctx ()))) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
        (input := input) (i := i) (vals := vals0) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  unfold NN.IR.Graph.denoteAllFrom
  simp [hi, hEval]
  simpa [hStep] using hTail
end Compiled
end Autograd
end Runtime
