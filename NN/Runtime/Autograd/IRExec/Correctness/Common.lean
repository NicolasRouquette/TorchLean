/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec

/-!
# Common

Internal helper lemmas for `NN.Runtime.Autograd.IRExec.Correctness`.

These lemmas relate the typed runtime context (`TList`) to the untyped IR value table (`Array
  Spec.PackedTensor`),
and provide small “building block” correctness steps that are reused across the per-op proofs.

The lemmas are grouped as follows:

* `packedTensorsOfContext*` lemmas: relate the typed context produced by `ForwardData.eval` to an untyped
  `Array (Spec.PackedTensor α)` (this is what the IR evaluator uses).
* `denoteAllState*` lemmas: package the IR forward evaluator (`ForwardGraph.denoteAll`) in the form
  expected by IR-style semantic equivalence proofs.

These lemmas are infrastructure: they should not encode op-specific logic. Per-op correctness files
(Matmul/Pool2d/LayerNorm/MSELoss) should depend on this module and not re-prove these bridges.

## Main definitions

- `throw_bind_ne_ok`: eliminates impossible success branches after `throw`.
- `NoMSELoss`: side condition for semantic equivalence theorems over fragments that exclude `.mse_loss`.
- `NoRawLog`: side condition for theorem statements that do not yet carry the positivity
  precondition required by raw real `log`.
- `NoConcat`: side condition for the concat branch not yet connected to end-to-end preservation.
- `packedTensorsOfContext_*`: typed-context to IR-array bridge lemmas.
- `denoteAllState_*` helpers: semantic equivalence bridges between lowered state and IR denotation tables.

## Implementation notes

- This module is shared infrastructure: predictable proof contracts
  matter more than clever proof tricks.
- Many lemmas here are proof-irrelevance/indexing bridges; these are repetitive but they remove a
  lot of friction from op-specific proofs.
- Collecting these utilities in one place gives op-specific correctness modules shared rewrite and
  indexing lemmas instead of repeated local proof scripts.
- These files can build slowly because they connect two representations at once: typed `TList`
  contexts on the forward-graph side and dynamically shaped `Spec.PackedTensor` arrays on the IR side. Most of the
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
namespace IRExec

open Spec
open Tensor
open Proofs.Autograd.Algebra
/-- `throw msg` in the `Except` monad is `.error msg`. -/
theorem throw_eq_error {β : Type} (msg : String) :
    (throw msg : Except String β) = .error msg := by
  simp [throw, throwThe, MonadExceptOf.throw]

open NN.IR
open Internal

/-!
## Shared side conditions

These predicates describe the exact fragment covered by a theorem. Keeping them in `Common` lets
the per-op lemmas, the semantic equivalence proof, and the chapter index refer to the same public contract
without import cycles.
-/

/--
Core semantic equivalence side condition: the IR graph contains no `.mse_loss` nodes.

The forward executor has a correct `.mse_loss` step lemma in `Correctness.Ops.Loss`; the existing
end-to-end semantic equivalence theorem keeps this condition so its branch proof stays small and predictable.
-/
def NoMSELoss (g : NN.IR.Graph) : Prop :=
  ∀ i n, g.getNode i = .ok n → n.kind ≠ .mseLoss

/--
Core semantic equivalence side condition: the IR graph contains no raw `.log` nodes.

Raw real logarithm is only mathematical on positive inputs. The executable closure produced by
`IRExec.Internal.buildFrom` is intentionally pure, while the IR denotation can report an
`Except.error` for
nonpositive inputs. Until the lowering theorem carries a per-node positivity/domain-validity
predicate, the end-to-end semantic-equivalence theorem excludes raw `.log`. Use the total
epsilon-protected `safe_log` operator when a graph needs unconditional execution.
-/
def NoRawLog (g : NN.IR.Graph) : Prop :=
  ∀ i n, g.getNode i = .ok n → n.kind ≠ .log

/--
Core semantic equivalence side condition: the IR graph contains no `.concat` nodes.

`IRExec.Internal.buildFrom` executes concat, including nonzero-axis permutation lowering, but the current
end-to-end proof does not yet relate that branch to `NN.IR.Graph.evalAt`. Keeping this limitation as
an explicit theorem hypothesis avoids presenting runtime support as a proved lowering pass fragment.
-/
def NoConcat (g : NN.IR.Graph) : Prop :=
  ∀ i n, g.getNode i = .ok n → ∀ axis, n.kind ≠ .concat axis

/--
If a `do`-chain begins with `throw`, it cannot produce an `.ok` result.

This lemma is used throughout the lowered-correctness proofs to close
impossible branches where lowering would have thrown an error message.
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
`packedTensorsOfContext` ignores type-level casts of the underlying `TList`.

`ForwardData.eval` introduces a definitional cast when extending contexts; this lemma lets us erase it
before reasoning about the corresponding `Array` of `Spec.PackedTensor`s.
-/
@[simp]
theorem packedTensorsOfContext_cast {α : Type} {ss₁ ss₂ : List Shape}
    (h : ss₁ = ss₂) (ctx : Proofs.Autograd.Algebra.TList α ss₁) :
    packedTensorsOfContext (α := α) (ss := ss₂) (Proofs.Autograd.Algebra.TList.cast (α := α) h ctx) =
      packedTensorsOfContext (α := α) (ss := ss₁) ctx := by
  cases h
  simp [packedTensorsOfContext]

/-- `packedTensorsOfContext` for a snoc’d context corresponds to `Array.push` of the appended tensor. -/
@[simp]
theorem packedTensorsOfContext_snoc {α : Type} {ss : List Shape} {τ : Shape}
    (ctx : Proofs.Autograd.Algebra.TList α ss) (t : Tensor α τ) :
    packedTensorsOfContext (α := α) (ss := ss ++ [τ])
        (Proofs.Autograd.Algebra.TList.snoc (α := α) (ss := ss) (τ := τ) ctx t) =
      (packedTensorsOfContext (α := α) (ss := ss) ctx).push
        (Spec.PackedTensor.ofTensor t) := by
  simp [packedTensorsOfContext, Spec.PackedTensor.ofTensor]

/--
Optional lookup in `packedTensorsOfContext` agrees with indexing the underlying typed context.

This is the main bridge between the typed runtime context and the untyped IR value table.
-/
theorem packedTensorsOfContext_getElem?
    {α : Type} {ss : List Shape}
    (ctx : Proofs.Autograd.Algebra.TList α ss) (i : Fin ss.length) :
    (packedTensorsOfContext (α := α) (ss := ss) ctx)[i.1]? =
      some (Spec.PackedTensor.ofTensor
        (Proofs.Autograd.Algebra.TList.get (α := α) (ss := ss) ctx i)) := by
  let arr := Proofs.Autograd.Algebra.TList.toPackedArray (α := α) (ss := ss) ctx
  have hi : i.1 < arr.size := by
    exact Nat.lt_of_lt_of_eq i.2
      (Proofs.Autograd.Algebra.TList.size_toPackedArray (α := α) (ss := ss) ctx).symm
  rw [show (packedTensorsOfContext (α := α) (ss := ss) ctx)[i.1]? = some arr[i.1] by
    simp [arr, packedTensorsOfContext]]
  congr 1
  simpa [arr] using
    (Proofs.Autograd.Algebra.TList.get_toPackedArray (α := α) (ss := ss) ctx i)

/--
Optional lookup in `packedTensorsOfContext` by a typed `Idx` agrees with `getIdx` on the
underlying `TList`.

This packages `packedTensorsOfContext_getElem?` into the repository’s `Idx` wrapper.
-/
theorem packedTensorsOfContext_getIdx?
    {α : Type} {ss : List Shape} {s : Shape}
    (ctx : Proofs.Autograd.Algebra.TList α ss) (idx : Idx ss s) :
    (packedTensorsOfContext (α := α) (ss := ss) ctx)[idx.i.1]? =
      some (Spec.PackedTensor.ofTensor (getIdx (α := α) (xs := ctx) idx)) := by
  cases idx with
  | mk i h =>
      -- Reduce to the `Fin`-indexed lemma and then specialize with the stored shape equality.
      cases h
      simpa [getIdx, Tensor.castShape] using
        (packedTensorsOfContext_getElem? (α := α) (ss := ss) ctx i)

/-- `Graph.expectShape` succeeds on a `Spec.PackedTensor` built with the same shape. -/
@[simp] theorem Graph.expectShape_mk {α : Type} [Context α] [DecidableEq Shape] {s : Shape}
    (t : Tensor α s) :
    NN.IR.Graph.expectShape (α := α) (expected := s) (Spec.PackedTensor.mk (α := α) s t) = .ok t := by
  simp [NN.IR.Graph.expectShape]
  rfl

attribute [grind =] packedTensorsOfContext_cast packedTensorsOfContext_snoc Graph.expectShape_mk
  throw_eq_error array_getElem_proof_irrel
  packedTensorsOfContext_getElem? packedTensorsOfContext_getIdx?

/-- Well-typed operands for either matrix-matrix or batched matrix multiplication. -/
inductive MatmulOperands (α : Type) [Context α] where
  /-- Two rank-two matrices with matching inner dimensions. -/
  | matrices {m n p : Nat}
      (left : Tensor α (.dim m (.dim n .scalar)))
      (right : Tensor α (.dim n (.dim p .scalar)))
  /-- Two rank-three matrix batches with matching batch and inner dimensions. -/
  | batches {batch m n p : Nat}
      (left : Tensor α (.dim batch (.dim m (.dim n .scalar))))
      (right : Tensor α (.dim batch (.dim n (.dim p .scalar))))

namespace MatmulOperands

/-- Package the left typed operand as a dynamic IR value. -/
def leftValue {α : Type} [Context α] : MatmulOperands α → Spec.PackedTensor α
  | .matrices left _ => Spec.PackedTensor.mk _ left
  | .batches left _ => Spec.PackedTensor.mk _ left

/-- Package the right typed operand as a dynamic IR value. -/
def rightValue {α : Type} [Context α] : MatmulOperands α → Spec.PackedTensor α
  | .matrices _ right => Spec.PackedTensor.mk _ right
  | .batches _ right => Spec.PackedTensor.mk _ right

/-- Evaluate a well-typed operand pair with the matching tensor specification. -/
def resultValue {α : Type} [Context α] : MatmulOperands α → Spec.PackedTensor α
  | .matrices left right => Spec.PackedTensor.mk _ (Spec.matMulSpec left right)
  | .batches left right => Spec.PackedTensor.mk _ (Tensor.bmmSpec left right)

/-- Retag the typed multiplication result with an equal graph-declared output shape. -/
def resultAtShape {α : Type} [Context α] (operands : MatmulOperands α)
    (outShape : Shape) (shapeEq : operands.resultValue.shape = outShape) : Spec.PackedTensor α :=
  Spec.PackedTensor.mk outShape (shapeEq ▸ operands.resultValue.tensor)

end MatmulOperands

/--
`NN.IR.Graph.evalAt` agrees with the typed matrix-multiplication specification for both supported
operand ranks. Rank compatibility is carried by `MatmulOperands`, so the evaluator proof is stated
once instead of duplicating its graph and parent-value reasoning for `mm` and `bmm`.
-/
theorem evalAt_matmul_ok
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) (input : Spec.PackedTensor α) (vals : Array (Spec.PackedTensor α))
    (i : Nat) (n : NN.IR.Node) (aId bId : Nat) (operands : MatmulOperands α)
    (hN : g.getNode i = .ok n) (hk : n.kind = .matmul) (hp : n.parents = [aId, bId])
    (hGetA : vals[aId]? = some operands.leftValue)
    (hGetB : vals[bId]? = some operands.rightValue)
    (hOut : operands.resultValue.shape = n.outShape) :
    NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload) (input := input)
        (vals := vals) (i := i) =
      .ok (operands.resultAtShape n.outShape hOut) := by
  cases operands <;>
    simp only [MatmulOperands.leftValue, MatmulOperands.rightValue,
      MatmulOperands.resultValue, MatmulOperands.resultAtShape] at hGetA hGetB hOut ⊢ <;>
    simp [NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput,
      hN, hk, hp, hGetA, hGetB, hOut, throw_eq_error]

/--
`NN.IR.Graph.evalAt` for a `.matmul` node specialized to 2D matrix multiply.

This is a proof-only helper that records the exact `Spec.mat_mul_spec` term produced by `evalAt`
in the well-typed success case.
-/
theorem evalAt_matmul_mm_ok
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) (input : Spec.PackedTensor α) (vals : Array (Spec.PackedTensor α))
    (i : Nat) (n : NN.IR.Node) (aId bId : Nat) (m nDim p : Nat)
    (aT : Tensor α (.dim m (.dim nDim .scalar)))
    (bT : Tensor α (.dim nDim (.dim p .scalar)))
    (hN : g.getNode i = .ok n) (hk : n.kind = .matmul) (hp : n.parents = [aId, bId])
    (hGetA : vals[aId]? = some (Spec.PackedTensor.mk (α := α) (.dim m (.dim nDim .scalar)) aT))
    (hGetB : vals[bId]? = some (Spec.PackedTensor.mk (α := α) (.dim nDim (.dim p .scalar)) bT))
    (hOut : (.dim m (.dim p .scalar)) = n.outShape) :
    NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload) (input := input) (vals := vals) (i :=
      i) =
      .ok (Spec.PackedTensor.mk (α := α) n.outShape
        (hOut ▸ Spec.matMulSpec (α := α) (m := m) (n := nDim) (p := p) aT bT)) := by
  simpa [MatmulOperands.leftValue, MatmulOperands.rightValue,
    MatmulOperands.resultValue, MatmulOperands.resultAtShape, Spec.PackedTensor.shape,
    Spec.PackedTensor.tensor] using
    (evalAt_matmul_ok (α := α) g payload input vals i n aId bId
      (.matrices aT bT) hN hk hp hGetA hGetB hOut)

/--
`NN.IR.Graph.evalAt` for a `.matmul` node specialized to batched matmul (`bmm`).

Like `evalAt_matmul_mm_ok`, this is used to relate the IR evaluator’s result to the forward-graph node’s
`forward` closure during the semantic equivalence correctness proof.
-/
theorem evalAt_matmul_bmm_ok
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) (input : Spec.PackedTensor α) (vals : Array (Spec.PackedTensor α))
    (i : Nat) (n : NN.IR.Node) (aId bId : Nat) (batch m nDim p : Nat)
    (aT : Tensor α (.dim batch (.dim m (.dim nDim .scalar))))
    (bT : Tensor α (.dim batch (.dim nDim (.dim p .scalar))))
    (hN : g.getNode i = .ok n) (hk : n.kind = .matmul) (hp : n.parents = [aId, bId])
    (hGetA : vals[aId]? = some
      (Spec.PackedTensor.mk (α := α) (.dim batch (.dim m (.dim nDim .scalar))) aT))
    (hGetB : vals[bId]? = some
      (Spec.PackedTensor.mk (α := α) (.dim batch (.dim nDim (.dim p .scalar))) bT))
    (hOut : (.dim batch (.dim m (.dim p .scalar))) = n.outShape) :
    NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload) (input := input) (vals := vals) (i :=
      i) =
      .ok (Spec.PackedTensor.mk (α := α) n.outShape
        (hOut ▸ Tensor.bmmSpec (α := α) (batch := batch) (m := m) (n := nDim) (p := p) aT bT)) :=
          by
    simpa [MatmulOperands.leftValue, MatmulOperands.rightValue,
      MatmulOperands.resultValue, MatmulOperands.resultAtShape, Spec.PackedTensor.shape,
      Spec.PackedTensor.tensor] using
      (evalAt_matmul_ok (α := α) g payload input vals i n aId bId
        (.batches aT bT) hN hk hp hGetA hGetB hOut)

/-- The two axis reductions supported by lowered IR semantic equivalence. -/
inductive AxisReductionKind where
  | sum
  | mean

/-- Convert a lowered axis-reduction case to its IR operation kind. -/
def AxisReductionKind.toOpKind (operation : AxisReductionKind) (axis : Nat) : NN.IR.OpKind :=
  match operation with
  | .sum => .reduceSum axis
  | .mean => .reduceMean axis

/-- Typed denotation of a lowered axis-reduction case. -/
def AxisReductionKind.denote
    {β : Type} [Context β] {shape : Shape}
    (operation : AxisReductionKind) (axis : Nat) (tensor : Tensor β shape)
    (axisValid : Shape.NonemptyAxis axis shape) : Tensor β (Tensor.shapeAfterSum shape axis) :=
  match operation with
  | .sum => Tensor.reduceSum (α := β) (s := shape) axis tensor
      (axisValid)
  | .mean => Tensor.reduceMean (α := β) (s := shape) axis tensor
      (axisValid)

/--
`NN.IR.Graph.evalAt` for either axis-reduction node in a well-typed success case.

This helper records the exact `Tensor.reduceSum` term produced by the IR evaluator once:
- the parent has the expected shape `s`,
- the axis validity check succeeds, and
- the node's declared `outShape` matches `shapeAfterSum s axis`.

The final cast to `n.outShape` comes from the `evalAt` "shape-tag normalization" step.
-/
theorem evalAt_axisReduction_ok
    {α : Type} [Context α] [DecidableEq Shape]
    (operation : AxisReductionKind)
    (g : NN.IR.Graph) (payload : Payload α) (input : Spec.PackedTensor α) (vals : Array (Spec.PackedTensor α))
    (i : Nat) (n : NN.IR.Node) (pId : Nat) (axis : Nat)
    (s : Shape) (pT : Tensor α s) (hAxisPf : PLift (Shape.NonemptyAxis axis s))
    (hN : g.getNode i = .ok n) (hk : n.kind = operation.toOpKind axis)
    (hp : n.parents = [pId])
    (hGet : vals[pId]? = some (Spec.PackedTensor.mk (α := α) s pT))
    (hAxis : Spec.Shape.nonemptyAxis? (axis := axis) s = some hAxisPf)
    (hOut : Spec.Tensor.shapeAfterSum s axis = n.outShape) :
    NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload) (input := input) (vals := vals) (i :=
      i) =
      .ok (Spec.PackedTensor.mk (α := α) n.outShape
        (hOut ▸ operation.denote axis pT hAxisPf.down)) := by
  cases operation <;>
    simp [AxisReductionKind.toOpKind, AxisReductionKind.denote, NN.IR.Graph.evalAt, NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput,
      NN.IR.Graph.evalNode, NN.IR.Graph.normalizeNodeOutput,
      hN, hk, hp, hGet, throw_eq_error, hAxis, hOut, Pure.pure, Except.pure]

/-- `evalAt_axisReduction_ok` specialized to summation. -/
theorem evalAt_reduceSum_ok
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) (input : Spec.PackedTensor α) (vals : Array (Spec.PackedTensor α))
    (i : Nat) (n : NN.IR.Node) (pId : Nat) (axis : Nat)
    (s : Shape) (pT : Tensor α s) (hAxisPf : PLift (Shape.NonemptyAxis axis s))
    (hN : g.getNode i = .ok n) (hk : n.kind = .reduceSum axis) (hp : n.parents = [pId])
    (hGet : vals[pId]? = some (Spec.PackedTensor.mk (α := α) s pT))
    (hAxis : Spec.Shape.nonemptyAxis? (axis := axis) s = some hAxisPf)
    (hOut : Spec.Tensor.shapeAfterSum s axis = n.outShape) :
    NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload) (input := input) (vals := vals) (i :=
      i) =
      .ok (Spec.PackedTensor.mk (α := α) n.outShape
        (hOut ▸ Tensor.reduceSum (α := α) (s := s) axis pT
          (hAxisPf.down))) := by
  exact evalAt_axisReduction_ok .sum g payload input vals i n pId axis s pT hAxisPf
    hN hk hp hGet hAxis hOut

/--
`NN.IR.Graph.evalAt` for a `.reduceMean axis` node, specialized to a well-typed success case.

This is the mean analogue of `evalAt_reduceSum_ok`.
-/
theorem evalAt_reduceMean_ok
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) (input : Spec.PackedTensor α) (vals : Array (Spec.PackedTensor α))
    (i : Nat) (n : NN.IR.Node) (pId : Nat) (axis : Nat)
    (s : Shape) (pT : Tensor α s) (hAxisPf : PLift (Shape.NonemptyAxis axis s))
    (hN : g.getNode i = .ok n) (hk : n.kind = .reduceMean axis) (hp : n.parents = [pId])
    (hGet : vals[pId]? = some (Spec.PackedTensor.mk (α := α) s pT))
    (hAxis : Spec.Shape.nonemptyAxis? (axis := axis) s = some hAxisPf)
    (hOut : Spec.Tensor.shapeAfterSum s axis = n.outShape) :
    NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload) (input := input) (vals := vals) (i :=
      i) =
      .ok (Spec.PackedTensor.mk (α := α) n.outShape
        (hOut ▸ Tensor.reduceMean (α := α) (s := s) axis pT
          (hAxisPf.down))) := by
  exact evalAt_axisReduction_ok .mean g payload input vals i n pId axis s pT hAxisPf
    hN hk hp hGet hAxis hOut

/-- Repackage a lowered `State` as an `ForwardGraph` so we can call its evaluator helpers. -/
def execOfState {α : Type} (inShape : Shape) (st : State α inShape) : ForwardGraph α :=
  { inShape := inShape, ss := st.1, body := st.2 }

/-- Evaluate the lowered prefix state and convert its typed runtime context into an IR-style table.
  -/
def denoteAllState {α : Type} [Context α] (inShape : Shape) (st : State α inShape)
    (x : Tensor α inShape) : Array (Spec.PackedTensor α) :=
  ForwardGraph.denoteAll (α := α) (e := execOfState (α := α) inShape st) x

/--
`denoteAllState` commutes with extending the SSA graph by one node (`ForwardData.snoc`).

This is the key step for proving that the lowering pass’s prefix-building loop stays in semantic equivalence with the
IR denotation table.
-/
theorem denoteAllState_snoc {α : Type} [Context α]
    {inShape : Shape} {ss : List Shape} {τ : Shape}
    (gd : ForwardData α [inShape] ss)
    (nodeData : ForwardNode α ([inShape] ++ ss) τ)
    (x : Tensor α inShape) :
    let st : State α inShape := ⟨ss, gd⟩
    let st' : State α inShape := ⟨ss ++ [τ], .snoc (ss := ss) gd nodeData⟩
    denoteAllState (α := α) inShape st' x =
      (denoteAllState (α := α) inShape st x).push
        (Spec.PackedTensor.mk (α := α) τ
          (nodeData.eval (ForwardData.eval (ss := ss) gd (.cons x .nil)))) := by
  -- Expand `st`/`st'`.
  simp only
  -- Reduce both sides to `packedTensorsOfContext` of `ForwardData.eval`.
  simp [denoteAllState, execOfState, ForwardGraph.denoteAll, ForwardGraph.eval]
  -- Now unfold `ForwardData.eval` for the snoc graph.
  simp [ForwardData.eval]

/--
Build a typed runtime index (`Idx`) for a numeric IR parent id.

The forward executor's context is typed by a list of shapes `[inShape] ++ ss`. `mkIdx` checks that:
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
Lookup in `denoteAllState` agrees with `getIdx` when `mkIdx pid s` succeeds.

This is used when proving correctness of the per-node lowering pass step: we translate parent ids in the
IR into typed indices into the forward-graph context.
-/
theorem denoteAllState_get_mkIdx?
    {α : Type} [Context α] [DecidableEq Shape] {inShape : Shape} {ss : List Shape}
    (gd : ForwardData α [inShape] ss) (x : Tensor α inShape)
    {pid : Nat} {s : Shape} {idx : Idx ([inShape] ++ ss) s}
    (hIdx : mkIdx (inShape := inShape) (ss := ss) pid s = .ok idx) :
    (denoteAllState (α := α) inShape (st := (⟨ss, gd⟩ : State α inShape)) x)[pid]? =
      some (Spec.PackedTensor.mk (α := α) s
        (getIdx (α := α)
          (xs := ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)) idx)) := by
  -- Unfold `denoteAllState` to the packed context and use the checked lookup theorem.
  have hPid : pid = idx.i.1 :=
    (mkIdx_ok_i_eq (inShape := inShape) (ss := ss) (id := pid) (s := s) (idx := idx) hIdx).symm
  rw [hPid]
  change
    (packedTensorsOfContext (α := α) (ss := [inShape] ++ ss)
      (ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil)))[idx.i.1]? =
      some (Spec.PackedTensor.mk (α := α) s
        (getIdx (α := α)
          (xs := ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd
            (.cons x .nil)) idx))
  exact packedTensorsOfContext_getIdx? (α := α)
    (ctx := ForwardData.eval (α := α) (Γ := [inShape]) (ss := ss) gd (.cons x .nil))
      idx

/--
One-step finishing lemma for the `buildFrom`/`denoteAllFrom` semantic equivalence proof.

If we know:
- the tail recursion `i+1` is correct (`hTail`),
- the IR evaluator step at `i` matches the forward-graph node’s `forward` (`hEval`), and
- the forward-graph table at `i` is the previous table plus the pushed node value (`hStep`),
then `denoteAllFrom` at `i` returns the final forward-graph table.
-/
theorem buildFrom_denoteAllFrom_finish
    {α : Type} [Context α] [DecidableEq Shape]
    (g : NN.IR.Graph) (payload : Payload α) {inShape : Shape} {ss : List Shape}
    (i : Nat) (x : Tensor α inShape)
    (hi : i < g.nodes.size)
    (τ : Shape) (nodeData : ForwardNode α ([inShape] ++ ss) τ)
    (st1 st' : State α inShape)
    (ctx : TList α ([inShape] ++ ss))
    (vals0 : Array (Spec.PackedTensor α))
    (input : Spec.PackedTensor α)
    (hTail :
      NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
          (input := input) (i := i + 1) (vals := denoteAllState (α := α) inShape st1 x) =
        .ok (denoteAllState (α := α) inShape st' x))
    (hEval :
      NN.IR.Graph.evalAt (α := α) (g := g) (payload := payload)
          (input := input) (vals := vals0) (i := i) =
        .ok (Spec.PackedTensor.mk (α := α) τ (nodeData.eval ctx)))
    (hStep :
      denoteAllState (α := α) inShape st1 x =
        vals0.push (Spec.PackedTensor.mk (α := α) τ (nodeData.eval ctx))) :
    NN.IR.Graph.denoteAllFrom (α := α) (g := g) (payload := payload)
        (input := input) (i := i) (vals := vals0) =
      .ok (denoteAllState (α := α) inShape st' x) := by
  unfold NN.IR.Graph.denoteAllFrom
  simp [hi, hEval]
  simpa [hStep] using hTail
end IRExec
end Autograd
end Runtime
