/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Compiled.IRExec.Core

/-!
# IR Graph Lowering

Checked lowering from the shared operation-tagged IR into executable proof-compiled graph data.
The representation and denotation helpers live in `IRExec.Core`.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Compiled

open Spec
open Tensor
open Proofs.Autograd.Algebra
open NN.IR

namespace IRExec

/--
Internal compilation state used by `buildFrom`.

It is a dependent pair of:
- `ss`: shapes of already-compiled IR nodes,
- `GraphData α Unit [inShape] ss`: executable closures for exactly that shape list.
-/
abbrev State (α : Type) (inShape : Shape) : Type :=
  Σ ss : List Shape, GraphData α Unit [inShape] ss

/--
Build a typed runtime index (`Idx`) for a numeric IR parent id.

The compiled runtime context is typed by `[inShape] ++ ss`, matching `GraphData`'s input-plus-node
context representation. `mkIdx` checks that:
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
normalizing the singleton append differently at compiler and correctness-proof call sites.
-/
def getIRValue {α : Type} {inShape : Shape} {ss : List Shape} {s : Shape}
    (ctx : TList α ([inShape] ++ ss)) (idx : Idx ([inShape] ++ ss) s) : Tensor α s :=
  getIdx (α := α) (xs := ctx) idx

/--
Construct a `NodeData` for forward execution only.

The compiled runtime `GraphData` expects each node to supply `forward`, `jvp`, and `vjp`.
For this IR bridge we only care about forward correctness, so `jvp`/`vjp` are populated with
forward-only sentinels that `panic!` if called.

This is intentional: `IRExec` closes the forward semantics gap; full gradient behavior is handled
by other runtime/autograd layers. Using `panic!` here is a safety measure: it prevents silently
wrong gradients if someone accidentally routes differentiation through an `IRExec`-compiled graph.
-/
def mkFwdNode {α : Type} [Context α] {Γ : List Shape} {τ : Shape}
    (forward : TList α Γ → Tensor α τ) : NodeData α Unit Γ τ :=
  -- `panic!` requires an `Inhabited` instance for the return type; provide one for contexts.
  let rec defaultTList : {Γ : List Shape} → TList α Γ
    | [] => .nil
    | s :: Γ => .cons (Tensor.default (α := α) (s := s)) (defaultTList (Γ := Γ))
  let _inst : Inhabited (TList α Γ) := ⟨defaultTList (Γ := Γ)⟩
  { forward := fun ctx _d => forward ctx
    jvp := fun _ctx _dctx _d =>
      panic! "IRExec: JVP requested for a forward-only node (use an autograd-capable backend)."
    vjp := fun _ctx _d _δ =>
      panic! "IRExec: VJP requested for a forward-only node (use an autograd-capable backend)." }

/--
Forward projection for `mkFwdNode`.

The JVP/VJP fields are sentinels in this bridge, but the forward field is exactly the function
passed to the constructor. This small simp lemma is used by the IR semantic-equivalence proof.
-/
@[simp] theorem mkFwdNode_forward {α : Type} [Context α] {Γ : List Shape} {τ : Shape}
    (f : TList α Γ → Tensor α τ) (ctx : TList α Γ) (d : Unit) :
    (mkFwdNode (α := α) (Γ := Γ) (τ := τ) f).forward ctx d = f ctx := by
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

end IRExec
end Compiled
end Autograd
end Runtime
