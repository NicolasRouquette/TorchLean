/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Semantics
public import NN.Proofs.Autograd.Runtime.Link

/-!
# IRExec

IR → executable SSA graph bridge.

This module lets us *run* an op-tagged `NN.IR.Graph` by compiling it into an executable
`Proofs.Autograd.Algebra.GraphData` (the SSA/DAG representation used by the proof-compiled runtime).

Why this exists:
- Verification tooling already targets `NN.IR.Graph` (an op-tagged DAG with external payloads).
- The runtime `.compiled` path executes `GraphData` (closures for each node).
- To enforce a single shared IR contract, we provide a checked translation `IR.Graph → GraphData`.
  The supported-fragment forward-correctness theorem connecting `GraphData.eval` to the IR denotation
  (`NN.IR.Graph.denote*`) lives in `NN.Runtime.Autograd.Compiled.IRExec.Correctness.SemanticEquivalence`
  (split out so routine runtime imports do not pull in the full semantic proof).

Important:
- The produced `GraphData` is meant for forward execution; the theorem layer states exactly which
  fragment is forward-correct and which domain side conditions are assumed.
- Today `jvp`/`vjp` are forward-only sentinels; this bridge is intended for forward execution and
  for closing the shared-IR semantics gap, not for training-style gradient computation.

### PyTorch intuition

If you’re coming from PyTorch:
- This is closer in spirit to *compiled graph execution* (TorchScript / `torch.compile`) than to
  eager mode.
- `NN.IR.Graph` is the "shared IR" we want verifiers and runtimes to agree on.
- `GraphData` is the executable SSA/DAG form: each node becomes a closure that reads parent values
  from a typed runtime context.
- PyTorch’s autograd engine computes gradients by recording an eager tape; this bridge is about
  running the forward pass of an IR graph with a proof that it matches the IR semantics.

## Main declarations

- `ExecGraphData` packages a compiled graph with its input shape.
- `IRExec.dValsOfCtx` converts typed runtime contexts back into IR-style value arrays.
- `IRExec.buildFrom` is the compiler from `NN.IR.Graph` to executable graph data.
- `IRExec.execGraphOfIR` is the main user-facing bridge entry point.

## Main definitions

- `ExecGraphData`: compiled executable graph package.
- `IRExec.mkIdx`: checked parent-id to typed-index bridge.
- `IRExec.mkFwdNode`: forward-only node constructor used during lowering.
- `IRExec.buildFrom`: recursive compiler from IR graph to executable SSA graph.
- `IRExec.execGraphOfIR`: user-facing compile entrypoint.

## Implementation notes

- This bridge covers forward semantics; gradient compilation is a separate contract.
- `jvp`/`vjp` are sentinels in this layer because gradient compilation is a separate concern
  from proving forward semantic equivalence.
- Lowering untyped numeric ids goes through typed indices (`Idx`) and explicit shape checks.

## References

- [Lean community documentation style](https://leanprover-community.github.io/contribute/doc.html)
- [PyTorch graph execution intuition](https://pytorch.org/docs/stable/generated/torch.compile.html)

## Tags

ir, compiler, runtime, graphdata, forward-semantics
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Compiled

open Spec
open Tensor
open Proofs.Autograd.Algebra
open NN.IR

/--
`simp` rule for `Except`-`do` chains: binding an `.ok` value is just function application.
-/
@[simp] theorem Except.ok_bind {ε α β : Type} (a : α) (f : α → Except ε β) :
    (Except.ok a >>= f) = f a := by
  simp [Bind.bind, Except.bind]

/--
`simp` rule for `Except`-`do` chains: binding an `.error` short-circuits.

Used heavily when discharging impossible branches in compilation correctness proofs.
-/
@[simp] theorem Except.error_bind {ε α β : Type} (e : ε) (f : α → Except ε β) :
    (Except.error e >>= f) = Except.error e := by
  simp [Bind.bind, Except.bind]

/-- Definitional simplification for `Except.bind` on `.ok`. -/
@[simp] theorem Except.bind_ok {ε α β : Type} (a : α) (f : α → Except ε β) :
    Except.bind (Except.ok a) f = f a := by
  rfl

/-- Definitional simplification for `Except.bind` on `.error`. -/
@[simp] theorem Except.bind_error {ε α β : Type} (e : ε) (f : α → Except ε β) :
    Except.bind (Except.error e) f = Except.error e := by
  rfl

/--
A forward-executable SSA graph derived from an `NN.IR.Graph`.

The compiled graph stores:
- one distinguished input shape (`inShape`),
- one shape per compiled node (`ss`, corresponding to IR node ids `1..n-1`),
- and executable node closures (`g`) consumed by `GraphData.eval`.
-/
structure ExecGraphData (α : Type) where
  /-- The distinguished IR input node’s shape (node id 0). -/
  inShape : Shape
  /-- Shapes of the IR nodes 1..(n-1) (one per executable SSA node). -/
  ss : List Shape
  /-- Executable SSA/DAG graph for nodes 1..(n-1); inputs live in `Γ := [inShape]`. -/
  g : GraphData α Unit [inShape] ss

namespace ExecGraphData

variable {α : Type}

/--
Evaluate the compiled executable SSA graph on a concrete input tensor.

The result is the full typed runtime context `[inShape] ++ ss`, i.e. input followed by every
compiled node value in topological order.
-/
def eval (e : ExecGraphData α) (x : Tensor α e.inShape) : TList α ([e.inShape] ++ e.ss) :=
  GraphData.eval (α := α) (Δ := Unit) (Γ := [e.inShape]) (ss := e.ss) e.g (.cons x .nil) ()

end ExecGraphData

/-!
## Denotation Table Helper

`ExecGraphData.eval` produces a typed runtime context `TList α ([inShape] ++ ss)`.

For debugging and for the forward-correctness development in
`NN.Runtime.Autograd.Compiled.IRExec.Correctness`,
we provide a helper that erases this context
into an IR-style value table `Array (NN.IR.DVal α)` in node-id order.
-/

namespace IRExec

/-- Convert a runtime `AnyTensor` (shape carried as a field) into an IR denotation value `DVal`. -/
def dValOfAny {α : Type} [Context α] (v : Runtime.AnyTensor α) : NN.IR.DVal α :=
  ⟨v.s, v.t⟩

/--
Convert a typed runtime context `TList α ss` into an IR-style value table.

This is phrased in terms of `Array (DVal α)` because the IR denotation functions (`denoteAll*`)
are array-based, while the compiled runtime evaluates into a typed context (`TList`).
-/
def dValsOfCtx {α : Type} [Context α] {ss : List Shape}
    (ctx : Proofs.Autograd.Algebra.TList α ss) : Array (NN.IR.DVal α) :=
  (Proofs.Autograd.Algebra.TList.toAnyArray (α := α) (ss := ss) ctx).map (dValOfAny (α := α))

end IRExec

namespace ExecGraphData

variable {α : Type} [Context α]

/--
Convert the full evaluated context into an IR-style value table (one `DVal` per node id).

This is the concrete bridge used in semantic equivalence statements that compare compiled evaluation against
`NN.IR.Graph.denoteAll*`.
-/
def denoteAll (e : Runtime.Autograd.Compiled.ExecGraphData α)
    (x : Tensor α e.inShape) : Array (NN.IR.DVal α) :=
  IRExec.dValsOfCtx (α := α) (ss := [e.inShape] ++ e.ss)
    (Runtime.Autograd.Compiled.ExecGraphData.eval e x)

end ExecGraphData

end Compiled
end Autograd
end Runtime
