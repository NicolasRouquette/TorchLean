/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.IRExec.Core

/-!
# IR Lowering Primitives

Typed state, indices, and shape operations used by the checked IR lowering pass. These declarations
remain under `IRExec.Internal`; the public lowering entry point lives in `IRExec.Lowering`.
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
    (ctx : _root_.TorchLean.TensorPack α ([inShape] ++ ss)) (idx : Idx ([inShape] ++ ss) s) : Tensor α s :=
  getIdx (α := α) (xs := ctx) idx

/-- Package a typed forward closure as one node of the executable IR graph. -/
def mkForwardNode {α : Type} {Γ : List Shape} {τ : Shape}
    (forward : _root_.TorchLean.TensorPack α Γ → Tensor α τ) : ForwardNode α Γ τ :=
  ⟨forward⟩

/--
Evaluation projection for `mkForwardNode`.
-/
@[simp] theorem mkForwardNode_eval {α : Type} {Γ : List Shape} {τ : Shape}
    (f : _root_.TorchLean.TensorPack α Γ → Tensor α τ) (ctx : _root_.TorchLean.TensorPack α Γ) :
    (mkForwardNode (α := α) (Γ := Γ) (τ := τ) f).eval ctx = f ctx := by
  rfl

/-- Internal list recursion used to track the dependent output shape of adjacent swaps. -/
def swapShapeBySwapsList (s : Shape) : List Nat → Shape
  | [] => s
  | d :: ds => swapShapeBySwapsList (s.swapAdjacentAtDepth d) ds

/-- Apply adjacent swaps, represented by their axis depths, to a shape. -/
def swapShapeBySwaps (s : Shape) (swaps : Array Nat) : Shape :=
  swapShapeBySwapsList s swaps.toList

/-- Internal dependent recursion underlying `applySwapsTensor`. -/
def applySwapsTensorList {α : Type} [Context α] :
    {s : Shape} → (swaps : List Nat) → Tensor α s → Tensor α (swapShapeBySwapsList s swaps)
  | _s, [], t => t
  | s, d :: ds, t =>
      let t' : Tensor α (s.swapAdjacentAtDepth d) := Tensor.swapAdjacentAxes (tensor := t) d
      applySwapsTensorList (s := s.swapAdjacentAtDepth d) (swaps := ds) t'

/-- Apply the same adjacent-swap sequence as `swapShapeBySwaps` to a tensor value. -/
def applySwapsTensor {α : Type} [Context α] {s : Shape} (swaps : Array Nat)
    (tensor : Tensor α s) : Tensor α (swapShapeBySwaps s swaps) :=
  applySwapsTensorList swaps.toList tensor

def concatLeadingAxisFromInfosList
    {α : Type} [Context α] {Γ : List Shape} {rest : Shape} (ctx : _root_.TorchLean.TensorPack α Γ) :
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
              ⟨n1 + n2, Tensor.concatAxisSpec .scalar (α := α) (n := n1) (m := n2)
                (suffix := rest) t1 t2⟩)
        s0

/-- Concatenate tensors selected by typed indices along their leading axis. -/
def concatLeadingAxisFromInfos
    {α : Type} [Context α] {Γ : List Shape} {rest : Shape} (ctx : _root_.TorchLean.TensorPack α Γ)
    (infos : Array (Sigma fun nP => Idx Γ (.dim nP rest))) :
    Sigma fun nSum => Tensor α (.dim nSum rest) :=
  concatLeadingAxisFromInfosList ctx infos.toList

/--
The concatenated size reported by `concatLeadingAxisFromInfos` is the sum of the input sizes.

This theorem is used to justify the output-shape side conditions in concat lowering branches.
-/
theorem concatLeadingAxisFromInfos_size_eq_sum
    {α : Type} [Context α] {Γ : List Shape} {rest : Shape}
    (ctx : _root_.TorchLean.TensorPack α Γ) (infos : Array (Sigma fun nP => Idx Γ (.dim nP rest))) :
    (concatLeadingAxisFromInfos (α := α) (Γ := Γ) (rest := rest) ctx infos).1 =
      infos.foldl (fun acc info => acc + info.1) 0 := by
  rw [← Array.foldl_toList]
  change
    (concatLeadingAxisFromInfosList (α := α) (Γ := Γ) (rest := rest) ctx infos.toList).1 =
      infos.toList.foldl (fun acc info => acc + info.1) 0
  cases hInfos : infos.toList with
  | nil =>
      simp [concatLeadingAxisFromInfosList]
  | cons info infosTail =>
      clear hInfos infos
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
              ⟨n1 + n2, Tensor.concatAxisSpec .scalar (α := α) (n := n1) (m := n2)
                (suffix := rest) t1 t2⟩
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
      simpa [concatLeadingAxisFromInfosList, List.foldl] using
        (hfold ⟨info.1, getIdx (α := α) (xs := ctx) info.2⟩)

end Internal
end IRExec
end Autograd
end Runtime
