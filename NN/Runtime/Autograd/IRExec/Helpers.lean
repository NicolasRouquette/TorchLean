/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Core

/-!
# IR Graph Lowering

Checked helpers for lowering the shared operation-tagged IR into executable forward graph data.
The public graph representation and denotation helpers live in `IRExec.Core`; implementation
details in this file remain under `IRExec.Internal`.
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

/--
Internal lowering state used by `buildFrom`.

It is a dependent pair of:
- `ss`: shapes of already-lowered IR nodes,
- `ForwardData α [inShape] ss`: forward closures for exactly that shape list.
-/
abbrev State (α : Type) (inShape : Shape) : Type :=
  Σ ss : List Shape, ForwardData α [inShape] ss

/--
Build a typed runtime index (`Idx`) for a numeric IR parent id.

The forward executor's context is typed by `[inShape] ++ ss`, matching `ForwardData`'s
input-plus-node representation. `mkIdx` checks that:
- `id` is in bounds, and
- the context shape at that position matches the expected shape `s`.

On failure, this returns a descriptive error string used directly by `buildFrom`.
-/
def mkIdx [DecidableEq Shape]
    (inShape : Shape) (ss : List Shape) (id : Nat) (s : Shape) :
    Except String (Idx ([inShape] ++ ss) s) := by
  let ctxShapes : List Shape := [inShape] ++ ss
  if h : id < ctxShapes.length then
    let fin : Fin ctxShapes.length := ⟨id, h⟩
    let got : Shape := ctxShapes.get fin
    if hg : got = s then
      exact .ok ⟨fin, hg⟩
    else
      exact .error
        s!"IRExec: shape mismatch at id={id}: expected {Shape.pretty s}, got {Shape.pretty got}"
  else
    exact .error s!"IRExec: invalid id={id} for ctxLen={ctxShapes.length}"

/--
Read a tensor from the single-input IR execution context using a checked parent index.

Keeping the context spelling `[inShape] ++ ss` explicit prevents dependent elaboration from
normalizing the singleton append differently at lowering pass and correctness-proof call sites.
-/
def getIRValue {α : Type} {inShape : Shape} {ss : List Shape} {s : Shape}
    (ctx : TList α ([inShape] ++ ss)) (idx : Idx ([inShape] ++ ss) s) : Tensor α s :=
  getIdx (α := α) (xs := ctx) idx

/-- Package a typed forward closure as one node of the executable IR graph. -/
def mkForwardNode {α : Type} {Γ : List Shape} {τ : Shape}
    (forward : TList α Γ → Tensor α τ) : ForwardNode α Γ τ :=
  ⟨forward⟩

/--
Evaluation projection for `mkForwardNode`.
-/
@[simp] theorem mkForwardNode_eval {α : Type} {Γ : List Shape} {τ : Shape}
    (f : TList α Γ → Tensor α τ) (ctx : TList α Γ) :
    (mkForwardNode (α := α) (Γ := Γ) (τ := τ) f).eval ctx = f ctx := by
  rfl

/--
Apply a list of adjacent swaps (specified by swap depths) to a shape.

This is the shape-level companion of `applySwapsTensor`, and mirrors IR permutation lowering.
-/
def swapShapeBySwaps (s : Shape) : List Nat → Shape
  | [] => s
  | d :: ds => swapShapeBySwaps (s.swapAdjacentAtDepth d) ds

/--
Apply the same swap sequence as `swapShapeBySwaps`, but to a tensor value.

This uses `Tensor.swap_at_depth_helper` repeatedly; it is the runtime companion of the IR-side
`swapDepthsForPerm` lowering used by `.permute`.
-/
def applySwapsTensor {α : Type} [Context α] :
    {s : Shape} → (swaps : List Nat) → Tensor α s → Tensor α (swapShapeBySwaps s swaps)
  | _s, [], t => t
  | s, d :: ds, t =>
      let t' : Tensor α (s.swapAdjacentAtDepth d) := Tensor.swapAtDepthHelper (tensor := t) d
      applySwapsTensor (s := s.swapAdjacentAtDepth d) (swaps := ds) t'

/--
Concatenate a list of tensors (all with shape `.dim nP rest`) along dimension 0.

The input list is expressed as typed indices into the runtime context `Γ`; the result tracks the
total concatenated size as a sigma.

This helper supports lowering of IR concat-style operators while preserving shape information.
-/
def concatLeadingAxisFromInfos
    {α : Type} [Context α] {Γ : List Shape} {rest : Shape} (ctx : TList α Γ) :
    (infos : List (Sigma fun nP => Idx Γ (.dim nP rest))) →
      Sigma fun nSum => Tensor α (.dim nSum rest)
  | [] =>
      ⟨0, Spec.fill (α := α) 0 (.dim 0 rest)⟩
  | info :: infos =>
      let s0 : Sigma fun n => Tensor α (.dim n rest) :=
        ⟨info.1, getIdx (α := α) (xs := ctx) info.2⟩
      infos.foldl
        (fun acc nxt =>
          match acc, nxt with
  | ⟨n1, t1⟩, ⟨n2, idx2⟩ =>
              let t2 := getIdx (α := α) (xs := ctx) idx2
              ⟨n1 + n2, Tensor.concatLeadingAxisSpec (α := α) (n := n1) (m := n2) (s := rest) t1 t2⟩)
        s0

/--
The concatenated size reported by `concatLeadingAxisFromInfos` is the sum of the input sizes.

This theorem is used to justify the output-shape side conditions in concat lowering branches.
-/
theorem concatLeadingAxisFromInfos_size_eq_sum
    {α : Type} [Context α] {Γ : List Shape} {rest : Shape}
    (ctx : TList α Γ) (infos : List (Sigma fun nP => Idx Γ (.dim nP rest))) :
    (concatLeadingAxisFromInfos (α := α) (Γ := Γ) (rest := rest) ctx infos).1 =
      infos.foldl (fun acc info => acc + info.1) 0 := by
  cases infos with
  | nil =>
      simp [concatLeadingAxisFromInfos]
  | cons info infosTail =>
      -- `concatLeadingAxisFromInfos` is a foldl over `infosTail` starting from a sigma whose `.1` is
      -- `info.1`.
      -- Its `.1` component is therefore the `Nat` foldl over the same list of `nP`s.
      let f :
          (Sigma fun n => Tensor α (.dim n rest)) →
            (Sigma fun nP => Idx Γ (.dim nP rest)) →
              (Sigma fun n => Tensor α (.dim n rest)) :=
        fun acc nxt =>
          match acc, nxt with
          | ⟨n1, t1⟩, ⟨n2, idx2⟩ =>
              let t2 := getIdx (α := α) (xs := ctx) idx2
              ⟨n1 + n2, Tensor.concatLeadingAxisSpec (α := α) (n := n1) (m := n2) (s := rest) t1 t2⟩
      have hfold :
          ∀ acc0 : Sigma fun n => Tensor α (.dim n rest),
            (infosTail.foldl f acc0).1 = infosTail.foldl (fun acc nxt => acc + nxt.1) acc0.1 := by
        intro acc0
        induction infosTail generalizing acc0 with
        | nil =>
            simp
        | cons nxt infos ih =>
            simp [List.foldl, f, ih]
      -- Now rewrite the outer fold (starting at 0) and finish.
      simpa [concatLeadingAxisFromInfos, List.foldl] using
        (hfold ⟨info.1, getIdx (α := α) (xs := ctx) info.2⟩)

end Internal
end IRExec
end Autograd
end Runtime
