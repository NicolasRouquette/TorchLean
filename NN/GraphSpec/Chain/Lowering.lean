/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.Chain.Syntax

/-!
# Executable lowering of sequential GraphSpec chains

This module lowers a chain directly to an execution-polymorphic `TorchLean.Program`. Parameters
remain ordered by the chain's type-level ABI and are split at the same structural boundaries as in
the pure interpreter.
-/

@[expose] public section

namespace NN
namespace GraphSpec

open _root_.Spec

namespace Chain

/-- Lower a sequential `Chain` to an execution-polymorphic TorchLean program. -/
def toProgram
    {ps : List Shape} {σ τ : Shape}
    (g : Chain ps σ τ)
    {α : Type 0} [Context α] [DecidableEq Shape] :
    Runtime.Autograd.TorchLean.Program α (ps ++ [σ]) τ :=
  fun {m} _instM _instOps =>
    let Ref := fun s => Runtime.Autograd.Torch.Ops.Ref (m := m) (α := α) s

    let rec splitParamsRef :
        {ps₁ ps₂ : List Shape} →
          Runtime.Autograd.Torch.RefList Ref (ps₁ ++ ps₂) →
            Runtime.Autograd.Torch.RefList Ref ps₁ × Runtime.Autograd.Torch.RefList Ref ps₂
      | [], _ps₂, xs => (.nil, xs)
      | _s :: ps₁, ps₂, .cons x xs =>
          let (l, r) := splitParamsRef (ps₁ := ps₁) (ps₂ := ps₂) xs
          (.cons x l, r)

    let rec lowerRefList
        {ps : List Shape} {σ τ : Shape}
        (g : Chain ps σ τ)
        (rs : Runtime.Autograd.Torch.RefList Ref (ps ++ [σ])) :
        m (Ref τ) :=
      match g with
      | .id _ =>
          match rs with
          | .cons x .nil => pure x
      | .prim p =>
          Runtime.Autograd.Torch.CurriedRef.uncurry (ss := ps ++ [σ]) (Ref := Ref)
            (p.program (α := α)) rs
      | .seq (ps₁ := ps₁) (ps₂ := ps₂) (τ := τm) g₁ g₂ => do
          let (params12, x) :=
            Runtime.Autograd.Torch.RefList.splitLast (Ref := Ref) (ss := ps₁ ++ ps₂) (τ := σ) rs
          let (params₁, params₂) := splitParamsRef (ps₁ := ps₁) (ps₂ := ps₂) params12
          let rs₁ :=
            Runtime.Autograd.Torch.RefList.append (Ref := Ref) (ss₁ := ps₁) (ss₂ := [σ])
              params₁ (.cons x .nil)
          let y ← lowerRefList (ps := ps₁) (σ := σ) (τ := τm) g₁ rs₁
          let rs₂ :=
            Runtime.Autograd.Torch.RefList.append (Ref := Ref) (ss₁ := ps₂) (ss₂ := [τm])
              params₂ (.cons y .nil)
          lowerRefList (ps := ps₂) (σ := τm) (τ := τ) g₂ rs₂

    Runtime.Autograd.Torch.CurriedRef.curry (ss := ps ++ [σ]) (Ref := Ref)
      (fun rs => lowerRefList (ps := ps) (σ := σ) (τ := τ) g rs)

end Chain

end GraphSpec
end NN
