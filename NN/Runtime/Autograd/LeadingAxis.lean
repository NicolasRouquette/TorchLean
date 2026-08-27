/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Shape

/-!
# Leading-Axis Reference Maps

A backend-independent recursion for applying an operation pointwise over the outer axis of a
shape-typed reference. Eager and typed graph execution supply their own slicing, reshaping, and
concatenation operations; the traversal itself remains shared.
-/

@[expose] public section

namespace Runtime.Autograd

open Spec

/--
Apply a reference-level operation independently to every entry of a leading axis.

The callbacks isolate the four structural operations needed by the recursion. Device backends may
replace this reference traversal with a fused primitive when the fused operation has the same
per-entry semantics.
-/
def mapOuterAxisWith {m : Type → Type} [Monad m] {Ref : Shape → Type} {σ τ : Shape}
    (empty : m (Ref (.dim 0 τ)))
    (slice : ∀ {n : Nat}, Ref (.dim n σ) → (start len : Nat) →
      (h : start + len ≤ n) → m (Ref (.dim len σ)))
    (reshape : ∀ {s₁ s₂ : Shape}, Ref s₁ → Shape.size s₁ = Shape.size s₂ → m (Ref s₂))
    (concat : ∀ {n k : Nat}, Ref (.dim n τ) → Ref (.dim k τ) →
      m (Ref (.dim (n + k) τ)))
    (f : Ref σ → m (Ref τ)) {n : Nat} (x : Ref (.dim n σ)) : m (Ref (.dim n τ)) :=
  match n with
  | 0 => empty
  | k + 1 => do
      let headBatch ← slice (n := k + 1) x 0 1 (by simp)
      let head ← reshape (s₁ := .dim 1 σ) (s₂ := σ) headBatch (by simp [Shape.size])
      let yHead ← f head
      let yHeadBatch ← reshape (s₁ := τ) (s₂ := .dim 1 τ) yHead (by simp [Shape.size])
      let tail ← slice (n := k + 1) x 1 k (by simp [Nat.add_comm])
      let yTail ← mapOuterAxisWith empty slice reshape concat f (n := k) tail
      let y ← concat (n := 1) (k := k) yHeadBatch yTail
      pure (by simpa [Nat.one_add] using y)
termination_by n

end Runtime.Autograd
