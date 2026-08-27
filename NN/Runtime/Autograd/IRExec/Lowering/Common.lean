/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Semantics
public import NN.Runtime.Autograd.IRExec.Lowering.Primitives
public import NN.Spec.Layers.Attention

/-!
# Shared Checked-Lowering Context

Common dependent context and result types used by the operation-family lowering modules.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace IRExec

open Spec
open Tensor
open Proofs.Autograd.Algebra
open NN.IR

namespace Internal

/-- Data shared by checked lowerers for one IR node. -/
structure NodeLoweringContext (α : Type) [Context α] (Γ : List Shape) where
  graph : NN.IR.Graph
  payload : Payload α
  index : Nat
  node : NN.IR.Node
  parentIdx : (pid : Nat) → (s : Shape) → Except String (Idx Γ s)

/-- The checked executable node produced for a lowering context. -/
abbrev NodeLoweringResult {α : Type} [Context α] {Γ : List Shape}
    (ctx : NodeLoweringContext α Γ) : Type :=
  Except String (ForwardNode α Γ ctx.node.outShape)

/-- Extract a dynamically shaped evaluator result after lowering has checked its output shape. -/
def packedResultOrPanic {α : Type} [Context α] [DecidableEq Shape]
    (expected : Shape) (result : Except String (Spec.SomeTensor α)) : Tensor α expected :=
  match result with
  | .error message => panic! s!"IRExec: checked evaluator failed at runtime: {message}"
  | .ok ⟨actual, tensor⟩ =>
      if h : actual = expected then h ▸ tensor
      else panic! s!"IRExec: checked evaluator returned {repr actual}, expected {repr expected}"


end Internal
end IRExec
end Autograd
end Runtime
