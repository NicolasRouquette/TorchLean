/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Compiled.GraphM.ShapeIndex
public import NN.Runtime.Autograd.LeadingAxis

/-!
# GraphM Neural Layers

Normalization and attention builders for proof-compiled graphs.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Compiled
namespace GraphM

open Spec
open Tensor
open Proofs.Autograd.Algebra
open Runtime.Autograd.TorchLean

/--
Layer normalization (sequence-first), producing the same shape as the input.

PyTorch comparison: `torch.nn.LayerNorm` / `torch.nn.functional.layer_norm` (modulo exact layout).

Forward-mode status: implemented by `Spec.layerNormJvp`, including parameter tangents for
`gamma` and `beta`.
-/
def layerNorm {α : Type} {Δ : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape]
  {Γ : List Shape} {seqLen embedDim : Nat}
  (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (x : Var (.dim seqLen (.dim embedDim .scalar)))
  (gamma : Var (.dim embedDim .scalar))
  (beta : Var (.dim embedDim .scalar)) :
  MWith α Δ Γ (Var (.dim seqLen (.dim embedDim .scalar))) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let ig ← liftM (mkIdx (_α := α) (Γ := Γ) ss gamma)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss beta)
  let node : NodeData α Δ (Γ ++ ss) (.dim seqLen (.dim embedDim .scalar)) :=
    { forward := fun ctx _d =>
        Spec.layerNorm (α := α) (seqLen := seqLen) (embedDim := embedDim)
          (x := getIdx (α := α) (xs := ctx) ix)
          (gamma := getIdx (α := α) (xs := ctx) ig)
          (beta := getIdx (α := α) (xs := ctx) ib)
          (h_seq_pos := h_seq_pos) (h_embed_pos := h_embed_pos)
      jvp := fun ctx dctx _d =>
        let xv := getIdx (α := α) (xs := ctx) ix
        let gv := getIdx (α := α) (xs := ctx) ig
        let bv := getIdx (α := α) (xs := ctx) ib
        let dx := getIdx (α := α) (xs := dctx) ix
        let dg := getIdx (α := α) (xs := dctx) ig
        let db := getIdx (α := α) (xs := dctx) ib
        Spec.layerNormJvp (α := α) (seqLen := seqLen) (embedDim := embedDim)
          (h_seq_pos := h_seq_pos) (h_embed_pos := h_embed_pos)
          (x := xv) (tangent := dx) (gamma := gv) (dgamma := dg) (_beta := bv) (dbeta := db)
      vjp := fun ctx _d dLdy =>
        let xv := getIdx (α := α) (xs := ctx) ix
        let gv := getIdx (α := α) (xs := ctx) ig
        let bv := getIdx (α := α) (xs := ctx) ib
        let (dx, dgamma, dbeta) :=
          Spec.layerNormBackward (α := α) (seqLen := seqLen) (embedDim := embedDim)
            (h_seq_pos := h_seq_pos) (h_embed_pos := h_embed_pos)
            (x := xv) (gamma := gv) (_beta := bv) (grad_output := dLdy)
        let z0 :=
          TList.add (α := α) (ss := Γ ++ ss)
            (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim seqLen (.dim embedDim .scalar)) ix dx)
            (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim embedDim .scalar) ig dgamma)
        TList.add (α := α) (ss := Γ ++ ss) z0
          (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim embedDim .scalar) ib dbeta) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := (.dim seqLen (.dim embedDim .scalar))) g node

/--
Batch normalization in channel-first layout (no running statistics; spec-level functional form).

PyTorch comparison: `torch.nn.BatchNorm2d` in NCHW layout (modulo exact semantics/parameters).

Forward-mode status: implemented by `Spec.batchNorm2dJvp`, including parameter tangents for
`gamma` and `beta`.
-/
def batchnormChannelFirst {α : Type} {Δ : Type} [Context α] [DecidableRel ((· > ·) : α → α →
  Prop)] [DecidableEq Shape]
  {Γ : List Shape} {channels height width : Nat}
  (h_c : channels > 0) (h_h : height > 0) (h_w : width > 0)
  (x : Var (.dim channels (.dim height (.dim width .scalar))))
  (gamma : Var (.dim channels .scalar))
  (beta : Var (.dim channels .scalar)) :
  MWith α Δ Γ (Var (.dim channels (.dim height (.dim width .scalar)))) := do
  let ⟨ss, g⟩ ← get
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let ig ← liftM (mkIdx (_α := α) (Γ := Γ) ss gamma)
  let ib ← liftM (mkIdx (_α := α) (Γ := Γ) ss beta)
  let outS : Shape := .dim channels (.dim height (.dim width .scalar))
  let node : NodeData α Δ (Γ ++ ss) outS :=
    { forward := fun ctx _d =>
        Spec.batchNorm2d (α := α) (channels := channels) (height := height) (width := width)
          (x := getIdx (α := α) (xs := ctx) ix)
          (gamma := getIdx (α := α) (xs := ctx) ig)
          (beta := getIdx (α := α) (xs := ctx) ib)
          (h_c := h_c) (h_h := h_h) (h_w := h_w)
      jvp := fun ctx dctx _d =>
        let xv := getIdx (α := α) (xs := ctx) ix
        let gv := getIdx (α := α) (xs := ctx) ig
        let bv := getIdx (α := α) (xs := ctx) ib
        let dx := getIdx (α := α) (xs := dctx) ix
        let dg := getIdx (α := α) (xs := dctx) ig
        let db := getIdx (α := α) (xs := dctx) ib
        Spec.batchNorm2dJvp (α := α) (channels := channels) (height := height) (width := width)
          (x := xv) (tangent := dx) (gamma := gv) (dgamma := dg) (_beta := bv) (dbeta := db)
          (_h_c := h_c) (_h_h := h_h) (_h_w := h_w)
      vjp := fun ctx _d dLdy =>
        let xv := getIdx (α := α) (xs := ctx) ix
        let gv := getIdx (α := α) (xs := ctx) ig
        let (dx, dgamma, dbeta) :=
          Spec.batchNorm2dBackward (α := α) (channels := channels) (height := height) (width := width)
            (x := xv) (gamma := gv) (grad_output := dLdy)
            (_h_c := h_c) (_h_h := h_h) (_h_w := h_w)
        let z0 :=
          TList.add (α := α) (ss := Γ ++ ss)
            (TList.single (α := α) (Γ := Γ ++ ss) (s := outS) ix dx)
            (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim channels .scalar) ig dgamma)
        TList.add (α := α) (ss := Γ ++ ss) z0
          (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim channels .scalar) ib dbeta) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := outS) g node

/--
Multi-head attention primitive (shape-specialized).

PyTorch comparison: `torch.nn.MultiheadAttention` / scaled dot-product attention.

Forward-mode status: implemented by `Spec.MultiHeadAttentionJvp`, including tangents for the
input and all four projection matrices.
-/
def multiHeadAttention {α : Type} {Δ : Type} [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {Γ : List Shape} {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (wq : Var (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wk : Var (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wv : Var (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wo : Var (.dim (numHeads * headDim) (.dim dModel .scalar)))
  (x : Var (.dim n (.dim dModel .scalar)))
  (mask : Option (Tensor Bool (.dim n (.dim n .scalar))) := none) :
  MWith α Δ Γ (Var (.dim n (.dim dModel .scalar))) := do
  let ⟨ss, g⟩ ← get
  let iwq ← liftM (mkIdx (_α := α) (Γ := Γ) ss wq)
  let iwk ← liftM (mkIdx (_α := α) (Γ := Γ) ss wk)
  let iwv ← liftM (mkIdx (_α := α) (Γ := Γ) ss wv)
  let iwo ← liftM (mkIdx (_α := α) (Γ := Γ) ss wo)
  let ix ← liftM (mkIdx (_α := α) (Γ := Γ) ss x)
  let node : NodeData α Δ (Γ ++ ss) (.dim n (.dim dModel .scalar)) :=
    { forward := fun ctx _d =>
        let mha : Spec.MultiHeadAttention α numHeads dModel headDim :=
          { Wq := getIdx (α := α) (xs := ctx) iwq
            Wk := getIdx (α := α) (xs := ctx) iwk
            Wv := getIdx (α := α) (xs := ctx) iwv
            Wo := getIdx (α := α) (xs := ctx) iwo }
        Spec.MultiHeadAttention.forward (α := α) (n := n) (h1 := h1)
          (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
          (mha := mha) (x := getIdx (α := α) (xs := ctx) ix) (mask := mask)
      jvp := fun ctx dctx _d =>
        let mha : Spec.MultiHeadAttention α numHeads dModel headDim :=
          { Wq := getIdx (α := α) (xs := ctx) iwq
            Wk := getIdx (α := α) (xs := ctx) iwk
            Wv := getIdx (α := α) (xs := ctx) iwv
            Wo := getIdx (α := α) (xs := ctx) iwo }
        let dmha : Spec.MultiHeadAttention α numHeads dModel headDim :=
          { Wq := getIdx (α := α) (xs := dctx) iwq
            Wk := getIdx (α := α) (xs := dctx) iwk
            Wv := getIdx (α := α) (xs := dctx) iwv
            Wo := getIdx (α := α) (xs := dctx) iwo }
        Spec.MultiHeadAttentionJvp (α := α) (h1 := h1)
          (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
          (mha := mha) (dmha := dmha)
          (x := getIdx (α := α) (xs := ctx) ix)
          (dx := getIdx (α := α) (xs := dctx) ix)
          (mask := mask)
      vjp := fun ctx _d dLdy =>
        let mha : Spec.MultiHeadAttention α numHeads dModel headDim :=
          { Wq := getIdx (α := α) (xs := ctx) iwq
            Wk := getIdx (α := α) (xs := ctx) iwk
            Wv := getIdx (α := α) (xs := ctx) iwv
            Wo := getIdx (α := α) (xs := ctx) iwo }
        let xv := getIdx (α := α) (xs := ctx) ix
        let (dx, dWq, dWk, dWv, dWo) :=
          Spec.MultiHeadAttentionBackward (α := α) (h1 := h1)
            (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
            (mha := mha) (x := xv) (mask := mask) (grad_output := dLdy)
        let z0 :=
          TList.add (α := α) (ss := Γ ++ ss)
            (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim dModel (.dim (numHeads * headDim)
              .scalar)) iwq dWq)
            (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim dModel (.dim (numHeads * headDim)
              .scalar)) iwk dWk)
        let z1 :=
          TList.add (α := α) (ss := Γ ++ ss) z0
            (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim dModel (.dim (numHeads * headDim)
              .scalar)) iwv dWv)
        let z2 :=
          TList.add (α := α) (ss := Γ ++ ss) z1
            (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim (numHeads * headDim) (.dim dModel
              .scalar)) iwo dWo)
        TList.add (α := α) (ss := Γ ++ ss) z2
          (TList.single (α := α) (Γ := Γ ++ ss) (s := .dim n (.dim dModel .scalar)) ix dx) }
  push (α := α) (Δ := Δ) (Γ := Γ) (ss := ss) (s := (.dim n (.dim dModel .scalar))) g node

/--
Leading-axis map of the proved multi-head-attention node.

The compiled graph intentionally records the per-sample nodes. This keeps its forward, JVP, and
VJP definitions inherited directly from `multiHeadAttention`, while an eager device backend may
execute the same map as one batched contraction.
-/
def batchedMultiHeadAttention {α : Type} {Δ : Type} [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {Γ : List Shape} {batch n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (wq : Var (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wk : Var (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wv : Var (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wo : Var (.dim (numHeads * headDim) (.dim dModel .scalar)))
  (x : Var (.dim batch (.dim n (.dim dModel .scalar))))
  (mask : Option (Tensor Bool (.dim n (.dim n .scalar))) := none) :
  MWith α Δ Γ (Var (.dim batch (.dim n (.dim dModel .scalar)))) :=
  _root_.Runtime.Autograd.mapLeadingAxisWith
    (const (α := α) (Δ := Δ) (Γ := Γ) <| Tensor.dim (fun i : Fin 0 => Fin.elim0 i))
    (fun x start len h =>
      sliceLeadingAxisRange (α := α) (Δ := Δ) (Γ := Γ) x start len h)
    (fun x h => reshape (α := α) (Δ := Δ) (Γ := Γ) x h)
    (fun x y => concatLeadingAxis (α := α) (Δ := Δ) (Γ := Γ) x y)
    (fun sample => multiHeadAttention (α := α) (Δ := Δ) (Γ := Γ)
      h1 wq wk wv wo sample mask)
    x

end GraphM
end Compiled
end Autograd
end Runtime
