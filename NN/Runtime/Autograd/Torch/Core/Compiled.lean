/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.BackwardOptim

/-!
# Compiled Torch Wrappers

Thin `torch.compile`-style wrappers around the proof-backed graph compiler. These definitions expose
forward, JVP, and VJP/backward entry points while keeping the compiled graph semantics explicit.
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
- `Runtime.Autograd.TorchLean.Session` (unified eager/compiled, recommended),
- `Runtime.Autograd.Torch.Internal.SessionIR` (proof-linked imperative session, internal).

`torch.compile`-style wrapper (cached static graph).

This record contains the proof-compiled graph model (`GraphData`) and its proven-correct
reverse-mode accumulator (`GraphData.backpropCtx`). It compiles once, then you can call
`forward/backward` repeatedly with new inputs.

Note: this does *not* cache a `Runtime.Autograd.Tape` for reuse across different inputs.
The current tape compiler bakes the forward context into backward closures, so reusing a single
tape across changing inputs would be unsound without redesigning the runtime node API.
-/

/--
`torch.compile`-style wrapper for a scalar-valued computation over leaf context `Γ`.

This stores a *proved* node (`NodeData`) together with the preceding graph prefix so it can be
evaluated and differentiated without rebuilding the whole graph each time.
-/
structure CompiledScalarWith (α : Type) (Δ : Type) (Γ : List Shape) where
  /-- Shapes of internal SSA nodes preceding the scalar output node. -/
  ssPrev : List Shape
  /-- Proved graph prefix that computes all preceding SSA nodes. -/
  gPrev : Proofs.Autograd.Algebra.GraphData α Δ Γ ssPrev
  /-- Final scalar output node over the leaf context plus graph prefix. -/
  node : Proofs.Autograd.Algebra.NodeData α Δ (Γ ++ ssPrev) Shape.scalar

/-- Scalar compiled graph with no auxiliary, non-differentiable runtime inputs. -/
abbrev CompiledScalar (α : Type) (Γ : List Shape) : Type :=
  CompiledScalarWith α Unit Γ

namespace CompiledScalar

/-- Evaluate the scalar output for leaf values `x`. -/
def forward {α : Type} {Γ : List Shape}
  (c : CompiledScalar α Γ) (x : TList α Γ) : Tensor α Shape.scalar :=
  c.node.forward (Proofs.Autograd.Algebra.GraphData.eval (g := c.gPrev) x ()) ()

/-- Forward-mode Jacobian-vector product (JVP) at `x` with tangent `dx`. -/
def jvp {α : Type} {Γ : List Shape}
  (c : CompiledScalar α Γ) (x dx : TList α Γ) : Tensor α Shape.scalar :=
  let ctx := Proofs.Autograd.Algebra.GraphData.eval (g := c.gPrev) x ()
  let dctx := Proofs.Autograd.Algebra.GraphData.jvpCtx (g := c.gPrev) x dx ()
  c.node.jvp ctx dctx ()

/--
Reverse-mode backprop for a scalar output with implicit seed `1`.

Returns a `TList` of gradients aligned with the leaf context `Γ`.
-/
def backward {α : Type} [Add α] [Zero α] [One α]
  {Γ : List Shape} (c : CompiledScalar α Γ) (x : TList α Γ) : TList α Γ :=
  let ssPrev := c.ssPrev
  let full : Proofs.Autograd.Algebra.GraphData α Unit Γ (ssPrev ++ [Shape.scalar]) :=
    .snoc (ss := ssPrev) c.gPrev c.node
  let seedPrev : TList α (Γ ++ ssPrev) := TList.zero (α := α) (ss := Γ ++ ssPrev)
  let seed' : TList α ((Γ ++ ssPrev) ++ [Shape.scalar]) :=
    TList.snoc (α := α) (ss := Γ ++ ssPrev) (τ := Shape.scalar) seedPrev (Tensor.scalar (1 : α))
  let seed : TList α (Γ ++ (ssPrev ++ [Shape.scalar])) :=
    TList.cast (α := α) (h := List.append_assoc Γ ssPrev [Shape.scalar]) seed'
  Proofs.Autograd.Algebra.GraphData.backpropCtx (α := α) (Δ := Unit) (Γ := Γ) (g := full) x () seed

/--
Reverse-mode backprop for a scalar output with an explicit scalar seed.

PyTorch comparison: `loss.backward(gradient=seedOut)` for a scalar loss.
-/
def backwardWithSeed {α : Type} [Add α] [Zero α]
    {Γ : List Shape} (c : CompiledScalar α Γ) (x : TList α Γ) (seedOut : α) : TList α Γ :=
  let ssPrev := c.ssPrev
  let full : Proofs.Autograd.Algebra.GraphData α Unit Γ (ssPrev ++ [Shape.scalar]) :=
    .snoc (ss := ssPrev) c.gPrev c.node
  let seedPrev : TList α (Γ ++ ssPrev) := TList.zero (α := α) (ss := Γ ++ ssPrev)
  let seed' : TList α ((Γ ++ ssPrev) ++ [Shape.scalar]) :=
    TList.snoc (α := α) (ss := Γ ++ ssPrev) (τ := Shape.scalar) seedPrev (Tensor.scalar seedOut)
  let seed : TList α (Γ ++ (ssPrev ++ [Shape.scalar])) :=
    TList.cast (α := α) (h := List.append_assoc Γ ssPrev [Shape.scalar]) seed'
  Proofs.Autograd.Algebra.GraphData.backpropCtx (α := α) (Δ := Unit) (Γ := Γ) (g := full) x () seed

end CompiledScalar

/-!
`torch.compile`-style wrapper for tensor-valued outputs.

This is the same idea as `CompiledScalar`, but parameterized by an arbitrary output shape `τ`.
It supports:
- `forward` (evaluate the output),
- `jvp` (forward-mode JVP, provided all ops supply `jvp`),
- `vjpWithSeed` (reverse-mode VJP with an explicit cotangent seed at the output).
-/

/--
`torch.compile`-style wrapper for a tensor-valued output of shape `τ`.

This generalizes `CompiledScalar` to arbitrary output shapes and provides forward-mode JVP and
reverse-mode VJP (with explicit seed).
-/
structure CompiledGraph (α : Type) (Γ : List Shape) (τ : Shape) where
  /-- Shapes of internal SSA nodes preceding the output node. -/
  ssPrev : List Shape
  /-- Proved graph prefix that computes all preceding SSA nodes. -/
  gPrev : Proofs.Autograd.Algebra.GraphData α Unit Γ ssPrev
  /-- Final output node over the leaf context plus graph prefix. -/
  node : Proofs.Autograd.Algebra.NodeData α Unit (Γ ++ ssPrev) τ

namespace CompiledGraph

/-- Evaluate the output tensor for leaf values `x`. -/
def forward {α : Type} {Γ : List Shape} {τ : Shape}
  (c : CompiledGraph α Γ τ) (x : TList α Γ) : Tensor α τ :=
  c.node.forward (Proofs.Autograd.Algebra.GraphData.eval (g := c.gPrev) x ()) ()

/-- Forward-mode Jacobian-vector product (JVP) at `x` with tangent `dx`. -/
def jvp {α : Type} {Γ : List Shape} {τ : Shape}
  (c : CompiledGraph α Γ τ) (x dx : TList α Γ) : Tensor α τ :=
  let ctx := Proofs.Autograd.Algebra.GraphData.eval (g := c.gPrev) x ()
  let dctx := Proofs.Autograd.Algebra.GraphData.jvpCtx (g := c.gPrev) x dx ()
  c.node.jvp ctx dctx ()

/--
Reverse-mode vector-Jacobian product (VJP) with an explicit output cotangent seed.

This is the tensor-valued analogue of `CompiledScalar.backwardWithSeed`.
PyTorch comparison: `out.backward(gradient=seedOut)` (for a tensor output).
-/
def vjpWithSeed {α : Type} [Add α] [Zero α]
    {Γ : List Shape} {τ : Shape}
    (c : CompiledGraph α Γ τ) (x : TList α Γ) (seedOut : Tensor α τ) : TList α Γ :=
  let ssPrev := c.ssPrev
  let full : Proofs.Autograd.Algebra.GraphData α Unit Γ (ssPrev ++ [τ]) :=
    .snoc (ss := ssPrev) c.gPrev c.node
  let seedPrev : TList α (Γ ++ ssPrev) := TList.zero (α := α) (ss := Γ ++ ssPrev)
  let seed' : TList α ((Γ ++ ssPrev) ++ [τ]) :=
    TList.snoc (α := α) (ss := Γ ++ ssPrev) (τ := τ) seedPrev seedOut
  let seed : TList α (Γ ++ (ssPrev ++ [τ])) :=
    TList.cast (α := α) (h := List.append_assoc Γ ssPrev [τ]) seed'
  Proofs.Autograd.Algebra.GraphData.backpropCtx (α := α) (Δ := Unit) (Γ := Γ) (g := full) x () seed

end CompiledGraph

/--
Compile a scalar-output graph whose nodes may read an auxiliary runtime environment `Δ`.

The environment is not part of the differentiable leaf context `Γ`. It is intended for discrete
inputs such as token ids, labels, gather indices, and masks that affect evaluation but do not
receive gradients.
-/
def compileScalarWith {α Δ : Type} [DecidableEq Shape] {Γ : List Shape}
    (build : Runtime.Autograd.Compiled.GraphM.MWith α Δ Γ
      (Runtime.Autograd.Compiled.GraphM.Var Shape.scalar)) :
    Runtime.Autograd.Result (CompiledScalarWith α Δ Γ) := do
  let (outVar, st) ← StateT.run build Runtime.Autograd.Compiled.GraphM.emptyWith
  finish outVar st.1 st.2
where
  finish {α Δ : Type} {Γ : List Shape}
      (outVar : Runtime.Autograd.Compiled.GraphM.Var Shape.scalar) :
      (ss : List Shape) →
        Runtime.Autograd.Compiled.GraphM.PGraphData α Δ Γ ss →
          Runtime.Autograd.Result (CompiledScalarWith α Δ Γ)
    | _, .nil =>
        .error "torch.compile: graph produced no nodes (need a scalar output node)"
    | _, .snoc (ss := ssPrev) (τ := τ) gPrev node =>
        match τ with
        | .scalar =>
            let expectedOutId := Γ.length + ssPrev.length
            if _h : outVar.id = expectedOutId then
              .ok { ssPrev := ssPrev, gPrev := gPrev, node := node }
            else
              .error
                (s!"torch.compile: output Var is not the last node (got id={outVar.id}, " ++
                  s!"expected id={expectedOutId})")
        | _ =>
            .error "torch.compile: output node is not scalar (expected Shape.scalar)"

/--
Compile a scalar-output graph builder into a `CompiledScalar`.

The builder is expressed in the `Compiled.GraphM` monad. We expect it to produce at least one node
and return a variable of scalar shape.
-/
def compileScalar {α : Type} [DecidableEq Shape] {Γ : List Shape}
    (build : Runtime.Autograd.Compiled.GraphM.M α Γ
      (Runtime.Autograd.Compiled.GraphM.Var Shape.scalar)) :
    Runtime.Autograd.Result (CompiledScalar α Γ) :=
  compileScalarWith (α := α) (Δ := Unit) (Γ := Γ) build

/--
Compile a tensor-output graph builder into a `CompiledGraph`.

We require that the returned `Var τ` is the *last* node produced by the builder, so the wrapper can
store the prefix graph and final output node cleanly.
-/
def compileGraph {α : Type} [DecidableEq Shape] {Γ : List Shape} {τ : Shape}
  (build : Runtime.Autograd.Compiled.GraphM.M α Γ (Runtime.Autograd.Compiled.GraphM.Var τ)) :
  Runtime.Autograd.Result (CompiledGraph α Γ τ) := do
  let (outVar, st) ← Runtime.Autograd.Compiled.GraphM.run (α := α) (Γ := Γ) build
  finish outVar st.1 st.2
where
  finish {α : Type} {Γ : List Shape} {τ : Shape}
      (outVar : Runtime.Autograd.Compiled.GraphM.Var τ) :
      (ss : List Shape) →
        Runtime.Autograd.Compiled.GraphM.PGraphData α Unit Γ ss →
          Runtime.Autograd.Result (CompiledGraph α Γ τ)
    | _, .nil =>
        .error "torch.compileGraph: graph produced no nodes (need an explicit output node)"
    | _, .snoc (ss := ssPrev) (τ := τ') gPrev node =>
        let expectedOutId := Γ.length + ssPrev.length
        if _hOut : outVar.id = expectedOutId then
          if hτ : τ' = τ then
            match hτ with
            | rfl => .ok { ssPrev := ssPrev, gPrev := gPrev, node := node }
          else
            .error
              (s!"torch.compileGraph: output node shape mismatch (expected " ++
                s!"{Shape.pretty τ}, got {Shape.pretty τ'})")
        else
          .error
            (s!"torch.compileGraph: output Var is not the last node (got " ++
              s!"id={outVar.id}, expected id={expectedOutId})")
end Torch
end Autograd
end Runtime
