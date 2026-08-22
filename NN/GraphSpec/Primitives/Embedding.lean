/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Core

/-!
# Primitive Embedding

GraphSpec’s DAG language (`NN.GraphSpec.DAG`) is the “general graph” IR. To avoid duplicating the
operator semantics, unary sequential primitives

`p : Primitive ps σ τ`

are embedded into DAG primitive ops

`LowerToDAG.Primitive.toDAGPrimOp p : DAG.PrimOp (ps ++ [σ]) τ`.

This file proves the bookkeeping theorem that keeps the “no duplicated semantics” contract explicit:
if you take a sequential primitive, embed it into DAG form, and feed it the obvious argument list
`params ++ [x]`, you get exactly the same pure forward computation.
-/

@[expose] public section


namespace NN
namespace GraphSpec
namespace Primitive

open _root_.Spec
open Spec.Tensor
open NN.Tensor

open Runtime.Autograd.Torch (TList)

/--
`splitAppend` undoes `append` in the one-input case.

This typed-list fact is used internally by `LowerToDAG.Primitive.toDAGPrimOp`.
-/
theorem splitAppend_appendSingleton
    {α : Type} [Context α] :
    {ps : List Shape} → {σ : Shape} →
      (params : TList α ps) → (x : Spec.Tensor α σ) →
        Proofs.Autograd.Algebra.TList.splitAppend (α := α)
            (ss₁ := ps) (ss₂ := [σ])
            (Proofs.Autograd.Algebra.TList.append (α := α)
              (ss₁ := ps) (ss₂ := [σ]) params (.cons x .nil))
          =
        (params, .cons x .nil)
  | [], _σ, .nil, x => rfl
  | _s :: ps, σ, .cons p params, x => by
      simp
        [ Proofs.Autograd.Algebra.TList.append
        , Proofs.Autograd.Algebra.TList.splitAppend
        , splitAppend_appendSingleton (α := α) (ps := ps) (σ := σ) params x
        ]

/--
Embedding a sequential primitive into DAG form preserves its pure `specFwd` semantics.

This theorem states that “the DAG primitive is the sequential primitive with
its parameters made explicit as ordinary inputs”.
-/
theorem toDAGPrimOp_specFwd_eq
    {α : Type} [Context α]
    {ps : List Shape} {σ τ : Shape}
    (p : Primitive ps σ τ)
    (params : TList α ps) (x : Spec.Tensor α σ) :
    (LowerToDAG.Primitive.toDAGPrimOp (ps := ps) (σ := σ) (τ := τ) p).specFwd (α := α)
        (Proofs.Autograd.Algebra.TList.append (α := α)
          (ss₁ := ps) (ss₂ := [σ]) params (.cons x .nil))
    =
    p.specFwd (α := α) params x := by
  -- Unfold the embedding and use `splitAppend_appendSingleton` to simplify the `splitAppend`.
  simp
    [ LowerToDAG.Primitive.toDAGPrimOp
    , splitAppend_appendSingleton (α := α) (ps := ps) (σ := σ) params x
    ]

end Primitive
end GraphSpec
end NN
