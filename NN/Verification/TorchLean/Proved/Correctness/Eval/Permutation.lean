/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Correctness.Eval.ShapeOps

/-!
# Permutation IR Evaluation

Local semantics for axis permutation.  The theorem is stated against `Graph.permuteDVal`, the shared
permutation interpreter used by `permute`, non-last-axis softmax, and axis-generic concat.
-/

@[expose] public section

namespace NN.Verification.TorchLean.Proved

open _root_.Spec
open _root_.Spec.Tensor
open NN.IR

namespace Correctness

namespace IRStep

/-- Local IR semantics for `permute`, using the shared dynamic-value permutation interpreter. -/
theorem evalAt_permute_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {s out : Shape} (perm : List Nat) (x : Tensor α s) (vOut : DVal α)
    (hPerm : Graph.permuteDVal (α := α) (v := DVal.mk (α := α) s x) perm = .ok vOut)
    (hShape : vOut.shape = out) :
    Graph.evalAt (α := α) (g := unaryGraphOut (.permute perm) s out)
        (payload := {})
        (input := DVal.mk (α := α) s x)
        (vals := #[DVal.mk (α := α) s x]) (i := 1)
      =
      Except.ok (DVal.mk (α := α) out (hShape ▸ vOut.tensor)) := by
  simp [Graph.evalAt, unaryGraphOut, unaryNodeOut, Graph.getNode, Graph.getNode?,
    Bind.bind, Except.bind, Pure.pure, Except.pure]
  have hPerm' : Graph.permuteDVal (α := α) (v := ⟨s, x⟩) perm = .ok vOut := by
    simpa [DVal.mk] using hPerm
  have hShape' : vOut.1 = out := by
    simpa [DVal.shape] using hShape
  rw [hPerm']
  simp [hShape']

end IRStep

end Correctness

end NN.Verification.TorchLean.Proved
