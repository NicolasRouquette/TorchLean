/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.BackwardOptim

/-!
# Typed Executable Graphs

Typed SSA graphs with forward, JVP, and VJP entry points. The graph records the computation once
and reuses the same shape-indexed operation data for repeated execution.

This is an executable graph representation, not an optimizing compiler. Lowering to `TypedGraph`
does not perform fusion, scheduling, native code generation, or kernel selection.
-/


@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec
open Tensor
open Proofs.Autograd.Algebra

/-!
Imperative sessions live in:
- `Runtime.Autograd.TorchLean.Session` (unified eager/typed graph execution, recommended),
- `Runtime.Autograd.Torch.Internal.TypedGraphSession` (typed graph recording session, internal).

Reusable typed graph representation.

This record contains executable `GraphData`: shape-indexed forward, JVP, and VJP functions. It
records once, then callers can evaluate or differentiate it repeatedly with new inputs. `GraphData`
does not itself prove that a node's JVP and VJP are derivatives of its forward map. Those local
laws live in `Proofs.Autograd.Algebra.Node`, and graphs assembled from such nodes use the separate
proof-carrying `Proofs.Autograd.Algebra.Graph` type.

Note: this does *not* cache a `Runtime.Autograd.Tape` for reuse across different inputs.
The current tape lowering pass bakes the forward context into backward closures, so reusing a single
tape across changing inputs would be unsound without redesigning the runtime node API.

This representation is also distinct from `Internal.TypedGraphSession`: `TypedGraph` is a
persistent value with explicit inputs, whereas the session is a mutable recorder used by the
imperative API.

It is also distinct from `Runtime.Autograd.IRExec.ForwardGraph`, which is the forward-only
executable form of the canonical operation-tagged `NN.IR.Graph`.
-/

/--
Typed graph with differentiable tensor inputs `Γ`, auxiliary runtime input `Δ`, and tensor output
of shape `τ`.

`Δ` is reserved for non-differentiable data such as token ids, labels, gather indices, and masks.
It affects evaluation but does not appear in the returned input gradients.
-/
structure TypedGraphWithAux (α : Type) (Δ : Type) (Γ : List Shape) (τ : Shape) where
  /-- Shapes of the recorded SSA nodes. -/
  nodeShapes : List Shape
  /-- Executable data for all recorded nodes. -/
  data : Proofs.Autograd.Algebra.GraphData α Δ Γ nodeShapes
  /-- Typed reference to the graph output, which may be an input or any recorded node. -/
  output : Proofs.Autograd.Algebra.Idx (Γ ++ nodeShapes) τ

namespace TypedGraphWithAux

/-- Evaluate the output tensor for leaf values `x` and auxiliary input `d`. -/
def forward {α Δ : Type} {Γ : List Shape} {τ : Shape}
    (c : TypedGraphWithAux α Δ Γ τ) (x : TList α Γ) (d : Δ) : Tensor α τ :=
  getIdx (Proofs.Autograd.Algebra.GraphData.eval (g := c.data) x d) c.output

/-- Forward-mode Jacobian-vector product at `x`, with `d` held fixed. -/
def jvp {α Δ : Type} {Γ : List Shape} {τ : Shape}
    (c : TypedGraphWithAux α Δ Γ τ) (x dx : TList α Γ) (d : Δ) : Tensor α τ :=
  getIdx (Proofs.Autograd.Algebra.GraphData.jvpCtx (g := c.data) x dx d) c.output

/-- Reverse-mode vector-Jacobian product with an explicit output cotangent seed. -/
def vjpWithSeed {α Δ : Type} [Add α] [Zero α]
    {Γ : List Shape} {τ : Shape} (c : TypedGraphWithAux α Δ Γ τ)
    (x : TList α Γ) (d : Δ) (seedOut : Tensor α τ) : TList α Γ :=
  Proofs.Autograd.Algebra.GraphData.backpropCtx
    (α := α) (Δ := Δ) (Γ := Γ) (g := c.data) x d (TList.single c.output seedOut)

end TypedGraphWithAux

/-- Typed graph with no auxiliary, non-differentiable runtime inputs. -/
abbrev TypedGraph (α : Type) (Γ : List Shape) (τ : Shape) : Type :=
  TypedGraphWithAux α Unit Γ τ

/-- Scalar-output typed graph with no auxiliary, non-differentiable runtime inputs. -/
abbrev TypedScalarGraph (α : Type) (Γ : List Shape) : Type :=
  TypedGraph α Γ Shape.scalar

namespace TypedGraph

/-- Evaluate the output tensor for leaf values `x`. -/
def forward {α : Type} {Γ : List Shape} {τ : Shape}
  (c : TypedGraph α Γ τ) (x : TList α Γ) : Tensor α τ :=
  TypedGraphWithAux.forward c x ()

/-- Forward-mode Jacobian-vector product (JVP) at `x` with tangent `dx`. -/
def jvp {α : Type} {Γ : List Shape} {τ : Shape}
  (c : TypedGraph α Γ τ) (x dx : TList α Γ) : Tensor α τ :=
  TypedGraphWithAux.jvp c x dx ()

/--
Reverse-mode vector-Jacobian product (VJP) with an explicit output cotangent seed.

This is the tensor-valued analogue of `TypedScalarGraph.backwardWithSeed`.
PyTorch comparison: `out.backward(gradient=seedOut)` (for a tensor output).
-/
def vjpWithSeed {α : Type} [Add α] [Zero α]
    {Γ : List Shape} {τ : Shape}
    (c : TypedGraph α Γ τ) (x : TList α Γ) (seedOut : Tensor α τ) : TList α Γ :=
  TypedGraphWithAux.vjpWithSeed c x () seedOut

end TypedGraph

namespace TypedScalarGraph

/-- Evaluate the scalar output for leaf values `x`. -/
def forward {α : Type} {Γ : List Shape}
    (c : TypedScalarGraph α Γ) (x : TList α Γ) : Tensor α Shape.scalar :=
  TypedGraph.forward c x

/-- Forward-mode Jacobian-vector product at `x` with tangent `dx`. -/
def jvp {α : Type} {Γ : List Shape}
    (c : TypedScalarGraph α Γ) (x dx : TList α Γ) : Tensor α Shape.scalar :=
  TypedGraph.jvp c x dx

/-- Reverse-mode backpropagation for a scalar output with implicit cotangent seed `1`. -/
def backward {α : Type} [Add α] [Zero α] [One α]
    {Γ : List Shape} (c : TypedScalarGraph α Γ) (x : TList α Γ) : TList α Γ :=
  TypedGraph.vjpWithSeed c x (Tensor.scalar (1 : α))

/-- Reverse-mode backpropagation for a scalar output with an explicit scalar seed. -/
def backwardWithSeed {α : Type} [Add α] [Zero α]
    {Γ : List Shape} (c : TypedScalarGraph α Γ) (x : TList α Γ) (seedOut : α) : TList α Γ :=
  TypedGraph.vjpWithSeed c x (Tensor.scalar seedOut)

end TypedScalarGraph

/-- Lower a graph builder with an auxiliary runtime environment into a reusable typed graph. -/
def lowerToTypedGraphWithAux {α Δ : Type} [DecidableEq Shape] {Γ : List Shape} {τ : Shape}
    (build : Runtime.Autograd.TypedGraph.GraphM.MWith α Δ Γ
      (Runtime.Autograd.TypedGraph.GraphM.Var τ)) :
    Runtime.Autograd.Result (TypedGraphWithAux α Δ Γ τ) := do
  let (outVar, st) ← StateT.run build Runtime.Autograd.TypedGraph.GraphM.emptyWith
  let output ← Runtime.Autograd.TypedGraph.GraphM.mkIdx
    (_α := α) (Γ := Γ) st.1 outVar
  pure { nodeShapes := st.1, data := st.2, output := output }

/--
Lower a scalar-output graph builder into a `TypedScalarGraph`.

The builder is expressed in the `TypedGraph.GraphM` monad. Its returned scalar variable may be an
input or any recorded node; lowering preserves that reference as the graph output.
-/
def lowerScalarToTypedGraph {α : Type} [DecidableEq Shape] {Γ : List Shape}
    (build : Runtime.Autograd.TypedGraph.GraphM.M α Γ
      (Runtime.Autograd.TypedGraph.GraphM.Var Shape.scalar)) :
    Runtime.Autograd.Result (TypedScalarGraph α Γ) :=
  lowerToTypedGraphWithAux (α := α) (Δ := Unit) (Γ := Γ) (τ := Shape.scalar) build

/--
Lower a tensor-output graph builder into a `TypedGraph`.

The returned variable may reference an input or any recorded node. Lowering validates its runtime
index and preserves its statically known output shape.
-/
def lowerToTypedGraph {α : Type} [DecidableEq Shape] {Γ : List Shape} {τ : Shape}
  (build : Runtime.Autograd.TypedGraph.GraphM.M α Γ (Runtime.Autograd.TypedGraph.GraphM.Var τ)) :
  Runtime.Autograd.Result (TypedGraph α Γ τ) :=
  lowerToTypedGraphWithAux (α := α) (Δ := Unit) (Γ := Γ) (τ := τ) build
end Torch
end Autograd
end Runtime
