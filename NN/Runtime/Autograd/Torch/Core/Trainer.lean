/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Trainer.Parameters

/-!
# Torch Trainer Helpers

`Ops` instances and scalar trainer construction for the Torch-style runtime. Shape-indexed mutable
parameter storage lives in `Trainer.Parameters`.
-/


@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec
open Tensor
open Proofs.Autograd.Algebra

/--
Monad used for the eager `Ops` instance: read an `Internal.EagerSession α` and execute in `IO`.

This is the backend that makes `Ops` programs execute immediately by mutating a hidden runtime tape.
-/
abbrev Internal.EagerM (α : Type) := ReaderT (Internal.EagerSession α) IO

/--
`Ops` instance for the eager Torch-style runtime.

This interprets `Ops` primitives by immediately executing them against the hidden mutable tape in
the current `Internal.EagerSession`.
-/
instance {α : Type} [Context α] [Internal.CudaBridge.TensorConv α] [DecidableEq Shape] :
    Ops (Internal.EagerM α) α where
  Ref := fun s => TensorRef α s
  NatTensorRef := fun s => Tensor Nat s
  natTensorConst := fun x => x
  mapNatTensor := fun f x => f x
  const := fun {s} t => fun sess => Internal.EagerSession.const (α := α) sess (sh := s) t
  add := fun {s} a b => fun sess => Internal.EagerSession.add (α := α) sess (sh := s) a b
  sub := fun {s} a b => fun sess => Internal.EagerSession.sub (α := α) sess (sh := s) a b
  mul := fun {s} a b => fun sess => Internal.EagerSession.mul (α := α) sess (sh := s) a b
  scale := fun {s} x c => fun sess => Internal.EagerSession.scale (α := α) sess (sh := s) x c
  abs := fun {s} x => fun sess => Internal.EagerSession.abs (α := α) sess (sh := s) x
  sqrt := fun {s} x => fun sess => Internal.EagerSession.sqrt (α := α) sess (sh := s) x
  clamp := fun {s} x minVal maxVal => fun sess =>
    Internal.EagerSession.clamp (α := α) sess (sh := s) x minVal maxVal
  max := fun {s} a b => fun sess => Internal.EagerSession.max (α := α) sess (sh := s) a b
  min := fun {s} a b => fun sess => Internal.EagerSession.min (α := α) sess (sh := s) a b
  broadcastTo := fun {s₁ s₂} cb x => fun sess =>
    Internal.EagerSession.broadcastTo (α := α) sess (sh1 := s₁) (sh2 := s₂) cb x
  reshape := fun {s₁ s₂} x h => fun sess =>
    Internal.EagerSession.reshape (α := α) sess (sh1 := s₁) (sh2 := s₂) x h
  transpose2d := fun {mDim nDim} x => fun sess =>
    Internal.EagerSession.transpose2d (α := α) sess (m := mDim) (n := nDim) x
  transpose3dFirstToLast := fun {a b c} x => fun sess =>
    Internal.EagerSession.transpose3dFirstToLast (α := α) sess (a := a) (b := b) (c := c) x
  transpose3dLastToFirst := fun {a b c} x => fun sess =>
    Internal.EagerSession.transpose3dLastToFirst (α := α) sess (a := a) (b := b) (c := c) x
  transpose3dLastTwo := fun {a b c} x => fun sess =>
    Internal.EagerSession.transpose3dLastTwo (α := α) sess (a := a) (b := b) (c := c) x
  swapAdjacentAtDepth := fun {s} depth x => fun sess =>
    Internal.EagerSession.swapAdjacentAtDepth (α := α) sess (sh := s) depth x
  reduceSum := fun {s} axis => fun x => fun sess =>
    Internal.EagerSession.reduceSum (α := α) sess (sh := s) axis x
  reduceMean := fun {s} axis => fun x => fun sess =>
    Internal.EagerSession.reduceMean (α := α) sess (sh := s) axis x
  gatherScalar := fun {n} x i => fun sess =>
    Internal.EagerSession.gatherScalar (α := α) sess (n := n) x i
  gatherRow := fun {rows cols} x i => fun sess =>
    Internal.EagerSession.gatherRow (α := α) sess (rows := rows) (cols := cols) x i
  gatherScalarNat := fun {n} x i => fun sess =>
    Internal.EagerSession.gatherScalarNat (α := α) sess (n := n) x i
  gatherVecNat := fun {n k} x idx => fun sess =>
    Internal.EagerSession.gatherVecNat (α := α) sess (n := n) (k := k) x idx
  gatherRowsNat := fun {rows cols k} x idx => fun sess =>
    Internal.EagerSession.gatherRowsNat (α := α) sess (rows := rows) (cols := cols) (k := k) x idx
  scatterAddVec := fun {n} x v i => fun sess =>
    Internal.EagerSession.scatterAddVec (α := α) sess (n := n) x v i
  scatterAddRow := fun {rows cols} x v i => fun sess =>
    Internal.EagerSession.scatterAddRow (α := α) sess (rows := rows) (cols := cols) x v i
  matmul := fun {mDim nDim pDim} a b => fun sess =>
    Internal.EagerSession.matmul (α := α) sess (m := mDim) (n := nDim) (p := pDim) a b
  bmm := fun {batch mDim nDim pDim} a b => fun sess =>
    Internal.EagerSession.bmm (α := α) sess (batch := batch) (m := mDim) (n := nDim) (p := pDim) a b
  concatVectors := fun {nDim mDim} a b => fun sess =>
    Internal.EagerSession.concatVectors (α := α) sess (n := nDim) (m := mDim) a b
  concatLeadingAxis := fun {nDim mDim} {s} a b => fun sess =>
    Internal.EagerSession.concatLeadingAxis (α := α) sess (n := nDim) (m := mDim) (sh := s) a b
  sliceLeadingAxisRange := fun {nDim} {s} start len h x => fun sess =>
    Internal.EagerSession.sliceLeadingAxisRange (α := α) sess (n := nDim) (sh := s) x start len h
  maxPool := fun {d C} {inSpatial kernel stride padding} {hKernel} x => fun sess =>
    Internal.EagerSession.maxPool (α := α) sess
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel)
      x
  avgPool := fun {d C} {inSpatial kernel stride padding} hKernel x => fun sess =>
    Internal.EagerSession.avgPool (α := α) sess
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      hKernel
      x
  smoothMaxPool := fun {d C} {inSpatial kernel stride padding} {hKernel} x beta => fun sess =>
    Internal.EagerSession.smoothMaxPool (α := α) sess
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel)
      x beta
  maxPool2d := fun {kH kW inH inW inC stride} {h1 h2} x => fun sess =>
    Internal.EagerSession.maxPool2d (α := α) sess
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := h1) (h2 := h2) x
  maxPool2dPad := fun {kH kW inH inW inC stride padding} {h1 h2} x => fun sess =>
    Internal.EagerSession.maxPool2dPad (α := α) sess
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
        padding)
      (h1 := h1) (h2 := h2) x
  smoothMaxPool2d := fun {kH kW inH inW inC stride} {h1 h2} x beta => fun sess =>
    Internal.EagerSession.smoothMaxPool2d (α := α) sess
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := h1) (h2 := h2) x beta
  avgPool2d := fun {kH kW inH inW inC stride} h1 h2 x => fun sess =>
    Internal.EagerSession.avgPool2d (α := α) sess
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      h1 h2 x
  avgPool2dPad := fun {kH kW inH inW inC stride padding} h1 h2 x => fun sess =>
    Internal.EagerSession.avgPool2dPad (α := α) sess
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
        padding)
      h1 h2 x
  relu := fun {s} x => fun sess => Internal.EagerSession.relu (α := α) sess (sh := s) x
  sigmoid := fun {s} x => fun sess => Internal.EagerSession.sigmoid (α := α) sess (sh := s) x
  tanh := fun {s} x => fun sess => Internal.EagerSession.tanh (α := α) sess (sh := s) x
  gelu := fun {s} x => fun sess => Internal.EagerSession.gelu (α := α) sess (sh := s) x
  softmax := fun {s} x => fun sess => Internal.EagerSession.softmax (α := α) sess (sh := s) x
  logSoftmax := fun {s} x => fun sess => Internal.EagerSession.logSoftmax (α := α) sess (sh := s) x
  softplus := fun {s} x => fun sess => Internal.EagerSession.softplus (α := α) sess (sh := s) x
  exp := fun {s} x => fun sess => Internal.EagerSession.exp (α := α) sess (sh := s) x
  log := fun {s} x => fun sess => Internal.EagerSession.log (α := α) sess (sh := s) x
  inv := fun {s} x => fun sess => Internal.EagerSession.inv (α := α) sess (sh := s) x
  detach := fun {s} x => fun sess => Internal.EagerSession.detach (α := α) sess (sh := s) x
  safeLog := fun {s} x ε => fun sess => Internal.EagerSession.safeLog (α := α) sess (sh := s) x (ε
    := ε)
  sum := fun {s} x => fun sess => Internal.EagerSession.sum (α := α) sess (sh := s) x
  flatten := fun {s} x => fun sess => Internal.EagerSession.flatten (α := α) sess (sh := s) x
  linear := fun {inDim outDim} w b x => fun sess =>
    Internal.EagerSession.linear (α := α) sess (inDim := inDim) (outDim := outDim) w b x
  mseLoss := fun {s} yhat target => fun sess => Internal.EagerSession.mseLoss (α := α) sess (sh :=
    s) yhat target
  layerNorm := fun {seqLen embedDim} hSeq hEmb x gamma beta => fun sess =>
    Internal.EagerSession.layerNorm (α := α) sess (seqLen := seqLen) (embedDim := embedDim)
      (h_seq_pos := hSeq) (h_embed_pos := hEmb) x gamma beta
  batchNormChannelFirst := fun {channels height width} hC hH hW x gamma beta => fun sess =>
    Internal.EagerSession.batchNormChannelFirst (α := α) sess
      (channels := channels) (height := height) (width := width) (h_c := hC) (h_h := hH) (h_w := hW)
      x gamma beta
  multiHeadAttention := fun {n numHeads dModel headDim} h1 wq wk wv wo x mask => fun sess =>
    Internal.EagerSession.multiHeadAttention (α := α) sess (n := n) (numHeads := numHeads)
      (dModel := dModel) (headDim := headDim) h1 wq wk wv wo x (mask := mask)
  batchedMultiHeadAttention :=
    fun {batch n numHeads dModel headDim} hBatch h1 wq wk wv wo x mask => fun sess =>
      Internal.EagerSession.batchedMultiHeadAttention (α := α) sess
        (batch := batch) (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
        hBatch h1 wq wk wv wo x (mask := mask)
  conv := fun {d inC outC} {kernel stride padding} {inSpatial} {hInC hKernel} w b x => fun sess =>
    Internal.EagerSession.conv (α := α) sess
      (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      (hInC := hInC) (hKernel := hKernel)
      w b x
  convTranspose := fun {d inC outC} {kernel stride padding} {inSpatial} {hInC hKernel} w b x =>
    fun sess =>
      Internal.EagerSession.convTranspose (α := α) sess
        (d := d) (inC := inC) (outC := outC)
        (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
        (hInC := hInC) (hKernel := hKernel)
        w b x
  conv2d := fun {inC outC kH kW stride padding inH inW} {h1 h2 h3} kernel bias input => fun sess =>
    Internal.EagerSession.conv2d (α := α) sess (inC := inC) (outC := outC) (kH := kH) (kW := kW)
      (stride := stride) (padding := padding) (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 :=
        h3)
      kernel bias input
  convTranspose2d := fun {inC outC kH kW stride padding inH inW} {h1 h2 h3} kernel bias input =>
    fun sess =>
      Internal.EagerSession.convTranspose2d (α := α) sess
        (inC := inC) (outC := outC) (kH := kH) (kW := kW)
        (stride := stride) (padding := padding) (inH := inH) (inW := inW)
        (h1 := h1) (h2 := h2) (h3 := h3)
        kernel bias input
  randUniform := fun {s} seed => fun sess =>
    Internal.EagerSession.randUniform (α := α) sess (sh := s) seed
  bernoulliMask := fun {s} keepProb seed => fun sess =>
    Internal.EagerSession.bernoulliMask (α := α) sess (sh := s) keepProb seed

/--
`Ops` instance for the typed graph builder monad `GraphM`.

This interprets `Ops` primitives by recording typed SSA nodes rather than executing them
immediately. `Runtime.Autograd.TypedGraph.GraphM` builds the graph data;
`Runtime.Autograd.Torch.TypedGraph` packages it for repeated execution.
-/
instance {α Δ : Type} [Context α] [DecidableEq Shape] {Γ : List Shape} :
    Ops (Runtime.Autograd.TypedGraph.GraphM.MWith α Δ Γ) α where
  Ref := fun s => Runtime.Autograd.TypedGraph.GraphM.Var s
  NatTensorRef := fun s => Δ → Tensor Nat s
  natTensorConst := fun x _ => x
  mapNatTensor := fun f x d => f (x d)
  const := fun {s} t => Runtime.Autograd.TypedGraph.GraphM.const (α := α) (Γ := Γ) (s := s) t
  add := fun {s} a b => Runtime.Autograd.TypedGraph.GraphM.add (α := α) (Γ := Γ) (s := s) a b
  sub := fun {s} a b => Runtime.Autograd.TypedGraph.GraphM.sub (α := α) (Γ := Γ) (s := s) a b
  mul := fun {s} a b => Runtime.Autograd.TypedGraph.GraphM.mul (α := α) (Γ := Γ) (s := s) a b
  scale := fun {s} x c => Runtime.Autograd.TypedGraph.GraphM.scale (α := α) (Γ := Γ) (s := s) x c
  abs := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.abs (α := α) (Γ := Γ) (s := s) x
  sqrt := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.sqrt (α := α) (Γ := Γ) (s := s) x
  clamp := fun {s} x minVal maxVal =>
    Runtime.Autograd.TypedGraph.GraphM.clamp (α := α) (Γ := Γ) (s := s) x minVal maxVal
  max := fun {s} a b => Runtime.Autograd.TypedGraph.GraphM.max (α := α) (Γ := Γ) (s := s) a b
  min := fun {s} a b => Runtime.Autograd.TypedGraph.GraphM.min (α := α) (Γ := Γ) (s := s) a b
  broadcastTo := fun {s₁ s₂} cb x =>
    Runtime.Autograd.TypedGraph.GraphM.broadcastTo (α := α) (Γ := Γ) (s₁ := s₁) (s₂ := s₂) cb x
  reshape := fun {s₁ s₂} x h =>
    Runtime.Autograd.TypedGraph.GraphM.reshape (α := α) (Γ := Γ) (s₁ := s₁) (s₂ := s₂) x h
  transpose2d := fun {mDim nDim} x =>
    Runtime.Autograd.TypedGraph.GraphM.transpose2d (α := α) (Γ := Γ) (m := mDim) (n := nDim) x
  transpose3dFirstToLast := fun {a b c} x =>
    Runtime.Autograd.TypedGraph.GraphM.transpose3dFirstToLast (α := α) (Γ := Γ) (a := a) (b := b)
      (c := c) x
  transpose3dLastToFirst := fun {a b c} x =>
    Runtime.Autograd.TypedGraph.GraphM.transpose3dLastToFirst (α := α) (Γ := Γ) (a := a) (b := b)
      (c := c) x
  transpose3dLastTwo := fun {a b c} x =>
    Runtime.Autograd.TypedGraph.GraphM.transpose3dLastTwo (α := α) (Γ := Γ) (a := a) (b := b) (c :=
      c) x
  swapAdjacentAtDepth := fun {s} depth x =>
    Runtime.Autograd.TypedGraph.GraphM.swapAdjacentAtDepth (α := α) (Γ := Γ) (s := s) depth x
  reduceSum := fun {s} axis => fun x =>
    Runtime.Autograd.TypedGraph.GraphM.reduceSum (α := α) (Γ := Γ) (s := s) axis x
  reduceMean := fun {s} axis => fun x =>
    Runtime.Autograd.TypedGraph.GraphM.reduceMean (α := α) (Γ := Γ) (s := s) axis x
  gatherScalar := fun {n} x i =>
    Runtime.Autograd.TypedGraph.GraphM.gatherScalar (α := α) (Γ := Γ) (n := n) x i
  gatherRow := fun {rows cols} x i =>
    Runtime.Autograd.TypedGraph.GraphM.gatherRow (α := α) (Γ := Γ) (rows := rows) (cols := cols) x i
  gatherScalarNat := fun {n} x i =>
    Runtime.Autograd.TypedGraph.GraphM.gatherScalarNat (α := α) (Γ := Γ) (n := n) x i
  gatherVecNat := fun {n k} x idx =>
    Runtime.Autograd.TypedGraph.GraphM.gatherVecNat (α := α) (Γ := Γ) (n := n) (k := k) x idx
  gatherRowsNat := fun {rows cols k} x idx =>
    Runtime.Autograd.TypedGraph.GraphM.gatherRowsNat (α := α) (Γ := Γ) (rows := rows) (cols := cols)
      (k := k) x idx
  scatterAddVec := fun {n} x v i =>
    Runtime.Autograd.TypedGraph.GraphM.scatterAddVec (α := α) (Γ := Γ) (n := n) x v i
  scatterAddRow := fun {rows cols} x v i =>
    Runtime.Autograd.TypedGraph.GraphM.scatterAddRow (α := α) (Γ := Γ) (rows := rows) (cols := cols)
      x v i
  matmul := fun {mDim nDim pDim} a b =>
    Runtime.Autograd.TypedGraph.GraphM.matmul (α := α) (Γ := Γ) (m := mDim) (n := nDim) (p := pDim) a
      b
  bmm := fun {batch mDim nDim pDim} a b =>
    Runtime.Autograd.TypedGraph.GraphM.bmm (α := α) (Γ := Γ) (batch := batch) (m := mDim) (n := nDim)
      (p := pDim) a b
  concatVectors := fun {nDim mDim} a b =>
    Runtime.Autograd.TypedGraph.GraphM.concatVectors (α := α) (Γ := Γ) (n := nDim) (m := mDim) a b
  concatLeadingAxis := fun {nDim mDim} {s} a b =>
    Runtime.Autograd.TypedGraph.GraphM.concatLeadingAxis (α := α) (Γ := Γ) (n := nDim) (m := mDim) (s := s)
      a b
  sliceLeadingAxisRange := fun {nDim} {s} start len h x =>
    Runtime.Autograd.TypedGraph.GraphM.sliceLeadingAxisRange (α := α) (Γ := Γ) (n := nDim) (s := s) x start len
      h
  maxPool := fun {d C} {inSpatial kernel stride padding} {hKernel} x =>
    Runtime.Autograd.TypedGraph.GraphM.maxPool (α := α) (Γ := Γ)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel)
      x
  avgPool := fun {d C} {inSpatial kernel stride padding} hKernel x =>
    Runtime.Autograd.TypedGraph.GraphM.avgPool (α := α) (Γ := Γ)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel)
      x
  smoothMaxPool := fun {d C} {inSpatial kernel stride padding} {hKernel} x beta =>
    Runtime.Autograd.TypedGraph.GraphM.smoothMaxPool (α := α) (Γ := Γ)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel)
      x beta
  maxPool2d := fun {kH kW inH inW inC stride} {h1 h2} x =>
    Runtime.Autograd.TypedGraph.GraphM.maxPool2d (α := α) (Γ := Γ)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := h1) (h2 := h2) x
  maxPool2dPad := fun {kH kW inH inW inC stride padding} {h1 h2} x =>
    Runtime.Autograd.TypedGraph.GraphM.maxPool2dPad (α := α) (Γ := Γ)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
        padding)
      (h1 := h1) (h2 := h2) x
  smoothMaxPool2d := fun {kH kW inH inW inC stride} {h1 h2} x beta =>
    Runtime.Autograd.TypedGraph.GraphM.smoothMaxPool2d (α := α) (Γ := Γ)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      (h1 := h1) (h2 := h2) x beta
  avgPool2d := fun {kH kW inH inW inC stride} h1 h2 x =>
    Runtime.Autograd.TypedGraph.GraphM.avgPool2d (α := α) (Γ := Γ)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
      h1 h2 x
  avgPool2dPad := fun {kH kW inH inW inC stride padding} h1 h2 x =>
    Runtime.Autograd.TypedGraph.GraphM.avgPool2dPad (α := α) (Γ := Γ)
      (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding :=
        padding)
      h1 h2 x
  relu := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.relu (α := α) (Γ := Γ) (s := s) x
  sigmoid := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.sigmoid (α := α) (Γ := Γ) (s := s) x
  tanh := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.tanh (α := α) (Γ := Γ) (s := s) x
  gelu := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.gelu (α := α) (Γ := Γ) (s := s) x
  softmax := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.softmax (α := α) (Γ := Γ) (s := s) x
  logSoftmax := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.logSoftmax (α := α) (Γ := Γ) (s := s)
    x
  softplus := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.softplus (α := α) (Γ := Γ) (s := s) x
  exp := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.exp (α := α) (Γ := Γ) (s := s) x
  log := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.log (α := α) (Γ := Γ) (s := s) x
  inv := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.inv (α := α) (Γ := Γ) (s := s) x
  detach := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.detach (α := α) (Γ := Γ) (s := s) x
  safeLog := fun {s} x ε => Runtime.Autograd.TypedGraph.GraphM.safeLog (α := α) (Γ := Γ) (s := s) x
    (ε := ε)
  sum := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.sum (α := α) (Γ := Γ) (s := s) x
  flatten := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.flatten (α := α) (Γ := Γ) (s := s) x
  linear := fun {inDim outDim} w b x =>
    Runtime.Autograd.TypedGraph.GraphM.linear (α := α) (Γ := Γ) (inDim := inDim) (outDim := outDim) w
      b x
  mseLoss := fun {s} yhat target =>
    Runtime.Autograd.TypedGraph.GraphM.mseLoss (α := α) (Γ := Γ) (s := s) yhat target
  layerNorm := fun {seqLen embedDim} hSeq hEmb x gamma beta =>
    Runtime.Autograd.TypedGraph.GraphM.layerNorm (α := α) (Γ := Γ) (seqLen := seqLen) (embedDim :=
      embedDim)
      (h_seq_pos := hSeq) (h_embed_pos := hEmb) x gamma beta
  batchNormChannelFirst := fun {channels height width} hC hH hW x gamma beta =>
    Runtime.Autograd.TypedGraph.GraphM.batchNormChannelFirst (α := α) (Γ := Γ)
      (channels := channels) (height := height) (width := width) (h_c := hC) (h_h := hH) (h_w := hW)
      x gamma beta
  multiHeadAttention := fun {n numHeads dModel headDim} h1 wq wk wv wo x mask =>
    Runtime.Autograd.TypedGraph.GraphM.multiHeadAttention (α := α) (Γ := Γ) (n := n) (numHeads :=
      numHeads)
      (dModel := dModel) (headDim := headDim) h1 wq wk wv wo x (mask := mask)
  batchedMultiHeadAttention :=
    fun {batch n numHeads dModel headDim} _hBatch h1 wq wk wv wo x mask =>
      Runtime.Autograd.TypedGraph.GraphM.batchedMultiHeadAttention (α := α) (Γ := Γ)
        (batch := batch) (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim)
        h1 wq wk wv wo x (mask := mask)
  conv := fun {d inC outC} {kernel stride padding} {inSpatial} {hInC hKernel} w b x =>
    Runtime.Autograd.TypedGraph.GraphM.conv (α := α) (Γ := Γ)
      (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      (hInC := hInC) (hKernel := hKernel)
      w b x
  convTranspose := fun {d inC outC} {kernel stride padding} {inSpatial} {hInC hKernel} w b x =>
    Runtime.Autograd.TypedGraph.GraphM.convTranspose (α := α) (Γ := Γ)
      (d := d) (inC := inC) (outC := outC)
      (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
      (hInC := hInC) (hKernel := hKernel)
      w b x
  conv2d := fun {inC outC kH kW stride padding inH inW} {h1 h2 h3} kernel bias input =>
    Runtime.Autograd.TypedGraph.GraphM.conv2d (α := α) (Γ := Γ) (inC := inC) (outC := outC) (kH := kH)
      (kW := kW)
      (stride := stride) (padding := padding) (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 :=
        h3)
      kernel bias input
  convTranspose2d := fun {inC outC kH kW stride padding inH inW} {h1 h2 h3} kernel bias input =>
    Runtime.Autograd.TypedGraph.GraphM.convTranspose2d (α := α) (Γ := Γ)
      (inC := inC) (outC := outC) (kH := kH) (kW := kW)
      (stride := stride) (padding := padding) (inH := inH) (inW := inW)
      (h1 := h1) (h2 := h2) (h3 := h3)
      kernel bias input
  randUniform := fun {s} seed => do
    Runtime.Autograd.TypedGraph.GraphM.randUniform (α := α) (Γ := Γ) (s := s) (seed := seed)
  bernoulliMask := fun {s} keepProb seed => do
    Runtime.Autograd.TypedGraph.GraphM.bernoulliMask (α := α) (Γ := Γ) (s := s) keepProb (seed :=
      seed)

/--
Persistence hooks for optimizer state owned by a backend-specific trainer.

The payload is intentionally opaque to callers. CUDA eager training, for example, owns Adam's
moment buffers on the device and streams them without first constructing host tensors. A trainer
that has no hidden optimizer state leaves this hook absent.
-/
structure OptimizerStateCheckpoint where
  /-- Save the complete backend-owned optimizer state. -/
  save : System.FilePath → IO Unit
  /-- Replace the backend-owned optimizer state from a previously saved payload. -/
  load : System.FilePath → IO Unit

/--
Bundle a scalar-loss training loop for fixed module state and an input signature.

This is the low-level trainer object used by module-backed execution:
- `loss` computes the scalar objective,
- `lossAndGradState` computes that loss and its state-shaped gradients from one tape,
- `gradState` exposes just the gradients when the loss is not needed,
- `stepWithLoss` applies an SGD update and returns the loss from the same tape,
- `step` applies the update without requiring callers to read the loss,
- `getState` reads the current parameters and persistent buffers.
-/
structure ScalarTrainer (α : Type) (paramShapes inputShapes : List Shape)
    (natInputShapes : List Shape := []) where
  /-- Mutable module state. Entries marked `requiresGrad = false` are persistent buffers. -/
  state : ParamList α paramShapes
  /-- Compute the scalar loss for a curried input pack. -/
  loss :
    Curried.Fn α inputShapes
      (Curried.Fn Nat natInputShapes (IO (Tensor α Shape.scalar)))
  /-- Compute the scalar loss and parameter gradients from one forward tape. -/
  lossAndGradState :
    Curried.Fn α inputShapes
      (Curried.Fn Nat natInputShapes
        (IO (Tensor α Shape.scalar × TList α paramShapes)))
  /-- Compute gradients aligned with `paramShapes` for a curried input pack. -/
  gradState :
    Curried.Fn α inputShapes
      (Curried.Fn Nat natInputShapes (IO (TList α paramShapes)))
  /-- Apply one SGD-style update and return the loss used to compute that update. -/
  stepWithLoss : α →
    Curried.Fn α inputShapes
      (Curried.Fn Nat natInputShapes (IO (Tensor α Shape.scalar)))
  /-- Apply one SGD-style update for a curried input pack. -/
  step : α → Curried.Fn α inputShapes (Curried.Fn Nat natInputShapes (IO Unit))
  /--
  Optional Adam update path.

  In eager CUDA mode this is a device-gradient/device-moment update path.  Other backends expose
  `none` and should use the generic optimizer wrappers.
  -/
  adamStep? : Option (α → α → α → α →
    Curried.Fn α inputShapes (Curried.Fn Nat natInputShapes (IO Unit))) := none
  /-- CUDA-native Adam update that also returns the loss from its forward tape. -/
  adamStepWithLoss? :
    Option (α → α → α → α →
      Curried.Fn α inputShapes
        (Curried.Fn Nat natInputShapes (IO (Tensor α Shape.scalar)))) := none
  /--
  Optional AdamW update path.

  In eager CUDA mode this is a device-gradient/device-moment update path with decoupled weight
  decay. Other backends expose `none` and should use the generic optimizer wrappers.
  -/
  adamWStep? : Option (α → α → α → α → α →
    Curried.Fn α inputShapes (Curried.Fn Nat natInputShapes (IO Unit))) := none
  /-- CUDA-native AdamW update that also returns the loss from its forward tape. -/
  adamWStepWithLoss? :
    Option (α → α → α → α → α →
      Curried.Fn α inputShapes
        (Curried.Fn Nat natInputShapes (IO (Tensor α Shape.scalar)))) := none
  /-- Save and restore optimizer state retained inside the selected runtime backend. -/
  optimizerStateCheckpoint? : Option OptimizerStateCheckpoint := none
  /-- Read current module state, synchronizing device mirrors if needed. -/
  getState : IO (TList α paramShapes)

namespace Internal

/--
Extract gradients (as a typed `TList`) for a list of eager `TensorRef`s from a dense gradient array.
-/
def gradsOfRefs {α : Type} [DecidableEq Shape] :
    {ss : List Shape} → Array (Runtime.AnyTensor α) → RefList (TensorRef α) ss → IO (TList α ss)
  | [], _grads, .nil => pure .nil
  | s :: ss, grads, .cons r rs => do
      let g ← Internal.EagerSession.grad (α := α) (sh := s) grads r
      let gs ← gradsOfRefs (α := α) (ss := ss) grads rs
      pure (.cons g gs)

/--
Record all parameters as tape leaves in an eager session, returning their corresponding
  `TensorRef`s.

This is the eager analogue of "using" a parameter pack during a forward pass.
-/
def useParams {α : Type} [CudaBridge.TensorConv α] [DecidableEq Shape] :
    {ss : List Shape} → ParamList α ss → EagerM α (RefList (TensorRef α) ss)
  | [], .nil => pure .nil
  | s :: ss, .cons p ps => fun sess => do
      let r ← Internal.EagerSession.use (α := α) (sh := s) sess p
      let rs ← useParams (α := α) (ss := ss) ps sess
      pure (.cons r rs)

/--
Record all input tensors as tape leaves in an eager session, returning their corresponding
  `TensorRef`s.
-/
def useInputs {α : Type} [CudaBridge.TensorConv α] [DecidableEq Shape] :
    {ss : List Shape} → TList α ss → EagerM α (RefList (TensorRef α) ss)
  | [], .nil => pure .nil
  | s :: ss, .cons x xs => fun sess => do
      let r ← Internal.EagerSession.input (α := α) (sh := s) sess x
      let rs ← useInputs (α := α) (ss := ss) xs sess
      pure (.cons r rs)

end Internal

/--
Build a `ScalarTrainer` from an initial parameter pack and an operation-generic loss definition.

`loss` is written once against the `Ops` interface over a concatenated context
`paramShapes ++ inputShapes`. Depending on `opts.execution`, TorchLean either records the loss once
as a typed SSA graph or executes it immediately while building a dynamic tape.
-/
def scalarTrainer {α : Type} [Context α] [Internal.CudaBridge.TensorConv α] [DecidableEq Shape]
    {paramShapes inputShapes natInputShapes : List Shape}
    (opts : Options := {})
    (initRequiresGrad : List Bool := List.replicate paramShapes.length true)
    (loss :
      ∀ {m : Type → Type}, [Monad m] → [Ops (m := m) (α := α)] →
        CurriedRef (fun s => Ops.Ref (m := m) (α := α) s) (paramShapes ++ inputShapes)
          (CurriedRef (fun s => Ops.NatTensorRef (m := m) (α := α) s) natInputShapes
            (m (Ops.Ref (m := m) (α := α) Shape.scalar)))) :
    Curried.Fn α paramShapes
      (IO (ScalarTrainer α paramShapes inputShapes natInputShapes)) :=
    Curried.curry (α := α) (ss := paramShapes)
      (β := IO (ScalarTrainer α paramShapes inputShapes natInputShapes))
    (fun initParams => do
    let ps ← ParamList.ofTListWithRequiresGrad (α := α) initParams initRequiresGrad
    match opts.execution with
    | .typedGraph =>
        if opts.device != .cpu then
          throw <| IO.userError
            s!"typed graph execution currently supports device `cpu`; requested `{opts.deviceName}`"
        let Γ : List Shape := paramShapes ++ inputShapes
        let Δ : Type := TList Nat natInputShapes
        let build : Runtime.Autograd.TypedGraph.GraphM.MWith α Δ Γ
            (Runtime.Autograd.TypedGraph.GraphM.Var Shape.scalar) := do
          let vs ← Runtime.Autograd.TypedGraph.GraphM.args (α := α) (Γ := Γ)
          let withNatInputs :=
            CurriedRef.applyVarList (Γ := Γ)
              (β := CurriedRef (fun s => Δ → Tensor Nat s) natInputShapes
                (Runtime.Autograd.TypedGraph.GraphM.MWith α Δ Γ
                  (Runtime.Autograd.TypedGraph.GraphM.Var Shape.scalar)))
              (loss (m := Runtime.Autograd.TypedGraph.GraphM.MWith α Δ Γ)) vs
          CurriedRef.applyTListProjections (full := natInputShapes) id withNatInputs
        let graph ← okOrThrow
          (lowerToTypedGraphWithAux (α := α) (Δ := Δ) (Γ := Γ) (τ := Shape.scalar) build)
        let ssFull : List Shape := graph.nodeShapes
        let fullGraph : Proofs.Autograd.Algebra.GraphData α Δ Γ ssFull :=
          graph.data
        let outId : Nat := graph.output.i.val

        let getScalarFromTape (t : Runtime.Autograd.Tape α) : IO (Tensor α Shape.scalar) := do
          let any ← match t.getValue? outId with
            | some v => pure v
            | none => throw <| IO.userError "typed graph execution: missing output value in tape"
          if h : any.s = Shape.scalar then
            pure (Tensor.castShape any.t h)
          else
            throw <| IO.userError
              s!"typed graph execution: output shape mismatch (expected scalar, got {Shape.pretty any.s})"

        let rec gradsPrefix :
            {ss : List Shape} → Array (Runtime.AnyTensor α) → Nat → IO (TList α ss)
          | [], _grads, _off => pure .nil
          | s :: ss, grads, off => do
              let any ← match grads[off]? with
                | some v => pure v
                | none => throw <| IO.userError "typed graph execution: gradient array too small"
              if h : any.s = s then
                let g : Tensor α s := Tensor.castShape any.t h
                let gs ← gradsPrefix (ss := ss) grads (off + 1)
                pure (.cons g gs)
              else
                throw <| IO.userError <|
                  s!"typed graph execution: gradient shape mismatch at idx={off} (expected "
                    ++ s!"{Shape.pretty s}, got "
                    ++ s!"{Shape.pretty any.s})"

        let runTape (xs : TList α inputShapes) (natInputs : Δ) :
            IO (Runtime.Autograd.Tape α) := do
          let pv ← ParamList.values (α := α) ps
          let args := Proofs.Autograd.Algebra.TList.append (α := α) (ss₁ := paramShapes)
            (ss₂ := inputShapes) pv xs
          let (tape, _ctx) :=
            Proofs.Autograd.Algebra.Graph.lowerGraphDataToTape
              (α := α) (Δ := Δ) (Γ := Γ) (ss := ssFull) fullGraph args natInputs
          pure tape
        let lossFn :
            Curried.Fn α inputShapes
              (Curried.Fn Nat natInputShapes (IO (Tensor α Shape.scalar))) :=
          Curried.curry (α := α) (ss := inputShapes)
            (β := Curried.Fn Nat natInputShapes (IO (Tensor α Shape.scalar))) (fun xs =>
              Curried.curry (α := Nat) (ss := natInputShapes)
                (β := IO (Tensor α Shape.scalar)) (fun natInputs =>
                  runTape xs natInputs >>= getScalarFromTape))
        let lossAndGradState :
            Curried.Fn α inputShapes (Curried.Fn Nat natInputShapes
              (IO (Tensor α Shape.scalar × TList α paramShapes))) :=
          Curried.curry (α := α) (ss := inputShapes)
            (β := Curried.Fn Nat natInputShapes
              (IO (Tensor α Shape.scalar × TList α paramShapes))) (fun xs =>
              Curried.curry (α := Nat) (ss := natInputShapes)
                (β := IO (Tensor α Shape.scalar × TList α paramShapes)) (fun natInputs => do
                  let tape ← runTape xs natInputs
                  let lossValue ← getScalarFromTape tape
                  let grads ← okOrThrow
                    (Runtime.Autograd.TypedGraph.backwardDenseAllFrom
                      (α := α) (Γ := Γ) (ss := ssFull) tape graph.output
                      (Tensor.scalar (1 : α)))
                  let paramGrads ← gradsPrefix (ss := paramShapes) grads 0
                  pure (lossValue, paramGrads)))
        let gradState :
            Curried.Fn α inputShapes
              (Curried.Fn Nat natInputShapes (IO (TList α paramShapes))) :=
          Curried.curry (α := α) (ss := inputShapes)
            (β := Curried.Fn Nat natInputShapes (IO (TList α paramShapes))) (fun xs =>
              Curried.curry (α := Nat) (ss := natInputShapes)
                (β := IO (TList α paramShapes)) (fun natInputs => do
                  let tape ← runTape xs natInputs
                  let grads ← okOrThrow
                    (Runtime.Autograd.TypedGraph.backwardDenseAllFrom
                      (α := α) (Γ := Γ) (ss := ssFull) tape graph.output
                      (Tensor.scalar (1 : α)))
                  gradsPrefix (ss := paramShapes) grads 0))
        let stepWithLoss (lr : α) :
            Curried.Fn α inputShapes
              (Curried.Fn Nat natInputShapes (IO (Tensor α Shape.scalar))) :=
          Curried.curry (α := α) (ss := inputShapes)
            (β := Curried.Fn Nat natInputShapes (IO (Tensor α Shape.scalar))) (fun xs =>
              Curried.curry (α := Nat) (ss := natInputShapes)
                (β := IO (Tensor α Shape.scalar)) (fun natInputs => do
                  let lossAndGradStateForNat :=
                    Curried.uncurry (α := α) (ss := inputShapes)
                      (β := Curried.Fn Nat natInputShapes
                        (IO (Tensor α Shape.scalar × TList α paramShapes)))
                      lossAndGradState xs
                  let (lossValue, grads) ←
                    Curried.uncurry (α := Nat) (ss := natInputShapes)
                      (β := IO (Tensor α Shape.scalar × TList α paramShapes))
                      lossAndGradStateForNat natInputs
                  ParamList.sgdStep (α := α) (ss := paramShapes) ps lr grads
                  pure lossValue))
        let step (lr : α) :
            Curried.Fn α inputShapes (Curried.Fn Nat natInputShapes (IO Unit)) :=
          Curried.curry (α := α) (ss := inputShapes)
            (β := Curried.Fn Nat natInputShapes (IO Unit)) (fun xs =>
              Curried.curry (α := Nat) (ss := natInputShapes) (β := IO Unit)
                (fun natInputs => do
                  let stepForNat :=
                    Curried.uncurry (α := α) (ss := inputShapes)
                      (β := Curried.Fn Nat natInputShapes
                        (IO (Tensor α Shape.scalar))) (stepWithLoss lr) xs
                  let _ ← Curried.uncurry (α := Nat) (ss := natInputShapes)
                    (β := IO (Tensor α Shape.scalar)) stepForNat natInputs
                  pure ()))
        pure
          { state := ps
            loss := lossFn
            lossAndGradState := lossAndGradState
            gradState := gradState
            stepWithLoss := stepWithLoss
            step := step
            adamStep? := none
            adamStepWithLoss? := none
            adamWStep? := none
            adamWStepWithLoss? := none
            optimizerStateCheckpoint? := none
            getState := ParamList.values (α := α) (ss := paramShapes) ps }
    | .eager =>
        let sess ← Internal.EagerSession.new (α := α) opts
        let adamStateRef ← IO.mkRef (Std.HashMap.emptyWithCapacity : Internal.EagerSession.CudaAdamState)
        let adamConfigRef ← IO.mkRef (none : Option Internal.EagerSession.CudaAdamConfig)
        let adamSchema : Internal.OptimizerCheckpoint.ParameterSchema :=
          { shapes := paramShapes, requiresGrad := ParamList.requiresGradList ps }
        let lossEager := loss (m := Internal.EagerM α)
        let recordLoss (xs : TList α inputShapes) (natInputs : TList Nat natInputShapes) :
            IO (TensorRef α Shape.scalar × RefList (TensorRef α) paramShapes) := do
          sess.resetTape
          (do
            let pRefs ← Internal.useParams (α := α) (ss := paramShapes) ps
            let xRefs ← Internal.useInputs (α := α) (ss := inputShapes) xs
            let allRefs := RefList.append (ss₁ := paramShapes) (ss₂ := inputShapes) pRefs xRefs
            let lossWithNat :=
              CurriedRef.uncurry (ss := paramShapes ++ inputShapes) lossEager allRefs
            let lossRef ←
              CurriedRef.uncurryTList (α := Nat) (ss := natInputShapes) lossWithNat natInputs
            pure (lossRef, pRefs)) |>.run sess
        let finishCudaStep : IO Unit := do
          Internal.EagerSession.releaseCudaTapeAfterOptimizerStep sess
          sess.cudaTape.set Runtime.Autograd.Cuda.Tape.empty
          sess.paramsByLeaf.set (Std.HashMap.emptyWithCapacity)
          sess.nats.set #[]
          Internal.EagerSession.collectCudaAllocator
        let lossFn :
            Curried.Fn α inputShapes
              (Curried.Fn Nat natInputShapes (IO (Tensor α Shape.scalar))) :=
          Curried.curry (α := α) (ss := inputShapes)
            (β := Curried.Fn Nat natInputShapes (IO (Tensor α Shape.scalar))) (fun xs =>
              Curried.curry (α := Nat) (ss := natInputShapes)
                (β := IO (Tensor α Shape.scalar)) (fun natInputs => do
                  let (lossRef, _) ← recordLoss xs natInputs
                  Internal.EagerSession.getValue (α := α) sess (sh := Shape.scalar) lossRef))
        let lossAndGradState :
            Curried.Fn α inputShapes (Curried.Fn Nat natInputShapes
              (IO (Tensor α Shape.scalar × TList α paramShapes))) :=
          Curried.curry (α := α) (ss := inputShapes)
            (β := Curried.Fn Nat natInputShapes
              (IO (Tensor α Shape.scalar × TList α paramShapes))) (fun xs =>
              Curried.curry (α := Nat) (ss := natInputShapes)
                (β := IO (Tensor α Shape.scalar × TList α paramShapes)) (fun natInputs => do
                  let (lossRef, pRefs) ← recordLoss xs natInputs
                  let lossValue ←
                    Internal.EagerSession.getValue (α := α) sess (sh := Shape.scalar) lossRef
                  let grads ← Internal.EagerSession.backwardScalarDenseAll (α := α) sess lossRef
                  let paramGrads ← Internal.gradsOfRefs (α := α) (ss := paramShapes) grads pRefs
                  pure (lossValue, paramGrads)))
        let gradState :
            Curried.Fn α inputShapes
              (Curried.Fn Nat natInputShapes (IO (TList α paramShapes))) :=
          Curried.curry (α := α) (ss := inputShapes)
            (β := Curried.Fn Nat natInputShapes (IO (TList α paramShapes))) (fun xs =>
              Curried.curry (α := Nat) (ss := natInputShapes)
                (β := IO (TList α paramShapes)) (fun natInputs => do
                  let (lossRef, pRefs) ← recordLoss xs natInputs
                  let grads ← Internal.EagerSession.backwardScalarDenseAll (α := α) sess lossRef
                  Internal.gradsOfRefs (α := α) (ss := paramShapes) grads pRefs))
        let stepWithLoss (lr : α) :
            Curried.Fn α inputShapes
              (Curried.Fn Nat natInputShapes (IO (Tensor α Shape.scalar))) :=
          Curried.curry (α := α) (ss := inputShapes)
            (β := Curried.Fn Nat natInputShapes (IO (Tensor α Shape.scalar))) (fun xs =>
              Curried.curry (α := Nat) (ss := natInputShapes)
                (β := IO (Tensor α Shape.scalar)) (fun natInputs => do
                  if opts.usesCuda then
                    let (lossRef, _) ← recordLoss xs natInputs
                    let lossValue ←
                      Internal.EagerSession.getValue (α := α) sess (sh := Shape.scalar) lossRef
                    let gradsDev ←
                      Internal.EagerSession.backwardScalarParamGradsCuda (α := α) sess lossRef
                    Internal.EagerSession.withCudaGradMap gradsDev fun gradsDev =>
                      Internal.EagerSession.sgdStepAllCudaMap (α := α) sess lr gradsDev
                    finishCudaStep
                    pure lossValue
                  else
                    let lossAndGradStateForNat :=
                      Curried.uncurry (α := α) (ss := inputShapes)
                        (β := Curried.Fn Nat natInputShapes
                          (IO (Tensor α Shape.scalar × TList α paramShapes)))
                        lossAndGradState xs
                    let (lossValue, grads) ←
                      Curried.uncurry (α := Nat) (ss := natInputShapes)
                        (β := IO (Tensor α Shape.scalar × TList α paramShapes))
                        lossAndGradStateForNat natInputs
                    ParamList.sgdStep (α := α) (ss := paramShapes) ps lr grads
                    pure lossValue))
        let step (lr : α) :
            Curried.Fn α inputShapes (Curried.Fn Nat natInputShapes (IO Unit)) :=
          Curried.curry (α := α) (ss := inputShapes)
            (β := Curried.Fn Nat natInputShapes (IO Unit)) (fun xs =>
              Curried.curry (α := Nat) (ss := natInputShapes) (β := IO Unit)
                (fun natInputs => do
                  if opts.usesCuda then
                    let (lossRef, _) ← recordLoss xs natInputs
                    let gradsDev ←
                      Internal.EagerSession.backwardScalarParamGradsCuda (α := α) sess lossRef
                    Internal.EagerSession.withCudaGradMap gradsDev fun gradsDev =>
                      Internal.EagerSession.sgdStepAllCudaMap (α := α) sess lr gradsDev
                    finishCudaStep
                  else
                    let gradStateForNat :=
                      Curried.uncurry (α := α) (ss := inputShapes)
                        (β := Curried.Fn Nat natInputShapes (IO (TList α paramShapes))) gradState xs
                    let g ← Curried.uncurry (α := Nat) (ss := natInputShapes)
                      (β := IO (TList α paramShapes)) gradStateForNat natInputs
                    ParamList.sgdStep (α := α) (ss := paramShapes) ps lr g))
        let adamStep? : Option (α → α → α → α →
            Curried.Fn α inputShapes (Curried.Fn Nat natInputShapes (IO Unit))) :=
          if opts.usesCuda then
            some (fun lr beta1 beta2 epsilon =>
              Curried.curry (α := α) (ss := inputShapes)
                (β := Curried.Fn Nat natInputShapes (IO Unit)) (fun xs =>
                  Curried.curry (α := Nat) (ss := natInputShapes) (β := IO Unit)
                    (fun natInputs => do
                      let (lossRef, _) ← recordLoss xs natInputs
                      let gradsDev ←
                        Internal.EagerSession.backwardScalarParamGradsCuda (α := α) sess lossRef
                      Internal.EagerSession.withCudaGradMap gradsDev fun gradsDev =>
                        Internal.EagerSession.adamStepAllCudaMap
                          (α := α) sess adamConfigRef adamStateRef
                          lr beta1 beta2 epsilon gradsDev
                      finishCudaStep)))
          else
            none
        let adamStepWithLoss? :
            Option (α → α → α → α →
              Curried.Fn α inputShapes
                (Curried.Fn Nat natInputShapes (IO (Tensor α Shape.scalar)))) :=
          if opts.usesCuda then
            some (fun lr beta1 beta2 epsilon =>
              Curried.curry (α := α) (ss := inputShapes)
                (β := Curried.Fn Nat natInputShapes (IO (Tensor α Shape.scalar))) (fun xs =>
                  Curried.curry (α := Nat) (ss := natInputShapes)
                    (β := IO (Tensor α Shape.scalar)) (fun natInputs => do
                      let (lossRef, _) ← recordLoss xs natInputs
                      let lossValue ←
                        Internal.EagerSession.getValue (α := α) sess (sh := Shape.scalar) lossRef
                      let gradsDev ←
                        Internal.EagerSession.backwardScalarParamGradsCuda (α := α) sess lossRef
                      Internal.EagerSession.withCudaGradMap gradsDev fun gradsDev =>
                        Internal.EagerSession.adamStepAllCudaMap (α := α) sess
                          adamConfigRef adamStateRef lr beta1 beta2 epsilon gradsDev
                      finishCudaStep
                      pure lossValue)))
          else
            none
        let adamWStep? : Option (α → α → α → α → α →
            Curried.Fn α inputShapes (Curried.Fn Nat natInputShapes (IO Unit))) :=
          if opts.usesCuda then
            some (fun lr weightDecay beta1 beta2 epsilon =>
              Curried.curry (α := α) (ss := inputShapes)
                (β := Curried.Fn Nat natInputShapes (IO Unit)) (fun xs =>
                  Curried.curry (α := Nat) (ss := natInputShapes) (β := IO Unit)
                    (fun natInputs => do
                      let (lossRef, _) ← recordLoss xs natInputs
                      let gradsDev ←
                        Internal.EagerSession.backwardScalarParamGradsCuda (α := α) sess lossRef
                      Internal.EagerSession.withCudaGradMap gradsDev fun gradsDev =>
                        Internal.EagerSession.adamWStepAllCudaMap
                          (α := α) sess adamConfigRef adamStateRef
                          lr weightDecay beta1 beta2 epsilon gradsDev
                      finishCudaStep)))
          else
            none
        let adamWStepWithLoss? :
            Option (α → α → α → α → α →
              Curried.Fn α inputShapes
                (Curried.Fn Nat natInputShapes (IO (Tensor α Shape.scalar)))) :=
          if opts.usesCuda then
            some (fun lr weightDecay beta1 beta2 epsilon =>
              Curried.curry (α := α) (ss := inputShapes)
                (β := Curried.Fn Nat natInputShapes (IO (Tensor α Shape.scalar))) (fun xs =>
                  Curried.curry (α := Nat) (ss := natInputShapes)
                    (β := IO (Tensor α Shape.scalar)) (fun natInputs => do
                      let (lossRef, _) ← recordLoss xs natInputs
                      let lossValue ←
                        Internal.EagerSession.getValue (α := α) sess (sh := Shape.scalar) lossRef
                      let gradsDev ←
                        Internal.EagerSession.backwardScalarParamGradsCuda (α := α) sess lossRef
                      Internal.EagerSession.withCudaGradMap gradsDev fun gradsDev =>
                        Internal.EagerSession.adamWStepAllCudaMap (α := α) sess
                          adamConfigRef adamStateRef lr weightDecay beta1 beta2 epsilon gradsDev
                      finishCudaStep
                      pure lossValue)))
          else
            none
        let optimizerStateCheckpoint? : Option OptimizerStateCheckpoint :=
          if opts.usesCuda then
            some
              { save := fun path =>
                  Internal.EagerSession.writeCudaAdamStateFloat32
                    path adamSchema adamConfigRef adamStateRef
                load := fun path =>
                  Internal.EagerSession.readCudaAdamStateFloat32
                    path adamSchema adamConfigRef adamStateRef }
          else
            none
        pure
          { state := ps
            loss := lossFn
            lossAndGradState := lossAndGradState
            gradState := gradState
            stepWithLoss := stepWithLoss
            step := step
            adamStep? := adamStep?
            adamStepWithLoss? := adamStepWithLoss?
            adamWStep? := adamWStep?
            adamWStepWithLoss? := adamWStepWithLoss?
            optimizerStateCheckpoint? := optimizerStateCheckpoint?
            getState := ParamList.valuesSynced (α := α) (ss := paramShapes) ps })
end Torch
end Autograd
end Runtime
