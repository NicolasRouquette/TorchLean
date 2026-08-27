/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core.Trainer.Parameters
public import NN.Tensor.ShapeErasure

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
instance {α : Type} [Context α] [TensorTransfer α] [DecidableEq Shape] :
    Ops (Internal.EagerM α) α where
  Ref := fun s => TensorRef α s
  DataRef := fun β s => Tensor β s
  dataConst := fun x => x
  mapData := fun f x => f x
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
  swapAdjacentAtDepth := fun {s} depth x => fun sess =>
    Internal.EagerSession.swapAdjacentAtDepth (α := α) sess (sh := s) depth x
  reduceSum := fun {s} axis => fun x => fun sess =>
    Internal.EagerSession.reduceSum (α := α) sess (sh := s) axis x
  reduceMean := fun {s} axis => fun x => fun sess =>
    Internal.EagerSession.reduceMean (α := α) sess (sh := s) axis x
  select := fun {s} axis _axisInBounds x index => fun sess =>
    Internal.EagerSession.select (α := α) sess (shape := s) axis x index
  indexSelect := fun {s} axis count _axisInBounds x indices => fun sess =>
    Internal.EagerSession.indexSelect (α := α) sess (shape := s) axis count x indices
  scatterAdd := fun {s} axis count _axisInBounds base source indices => fun sess =>
    Internal.EagerSession.scatterAdd (α := α) sess (shape := s) axis count base source indices
  matmul := fun {batchA batchB batch : Shape} {mDim nDim pDim : Nat}
      {broadcastA} {broadcastB} a b => fun sess =>
    Internal.EagerSession.matmul (α := α) sess (batchA := batchA) (batchB := batchB)
      (batch := batch) (m := mDim) (n := nDim) (p := pDim)
      (broadcastA := broadcastA) (broadcastB := broadcastB) a b
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
  smoothMaxPool := fun {d C} {inSpatial kernel stride padding} {hKernel}
      [_decidableEq : DecidableEq α] x beta => fun sess =>
    Internal.EagerSession.smoothMaxPool (α := α) sess
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel)
      x beta
  relu := fun {s} x => fun sess => Internal.EagerSession.relu (α := α) sess (sh := s) x
  sigmoid := fun {s} x => fun sess => Internal.EagerSession.sigmoid (α := α) sess (sh := s) x
  tanh := fun {s} x => fun sess => Internal.EagerSession.tanh (α := α) sess (sh := s) x
  gelu := fun {s} x => fun sess => Internal.EagerSession.gelu (α := α) sess (sh := s) x
  softmaxLast := fun {s} x => fun sess =>
    Internal.EagerSession.softmaxLast (α := α) sess (sh := s) x
  logSoftmaxLast := fun {s} x => fun sess =>
    Internal.EagerSession.logSoftmaxLast (α := α) sess (sh := s) x
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
  batchNorm := fun {channels sSpatial} hWellFormed x gamma beta => fun sess =>
    Internal.EagerSession.batchNorm (α := α) sess
      (channels := channels) (sSpatial := sSpatial) hWellFormed x gamma beta
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
  DataRef := fun β s => Δ → Tensor β s
  dataConst := fun x _ => x
  mapData := fun f x d => f (x d)
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
  swapAdjacentAtDepth := fun {s} depth x =>
    Runtime.Autograd.TypedGraph.GraphM.swapAdjacentAtDepth (α := α) (Γ := Γ) (s := s) depth x
  reduceSum := fun {s} axis => fun x =>
    Runtime.Autograd.TypedGraph.GraphM.reduceSum (α := α) (Γ := Γ) (s := s) axis x
  reduceMean := fun {s} axis => fun x =>
    Runtime.Autograd.TypedGraph.GraphM.reduceMean (α := α) (Γ := Γ) (s := s) axis x
  select := fun {s} axis _axisInBounds x index =>
    Runtime.Autograd.TypedGraph.GraphM.select (α := α) (Γ := Γ) (s := s) axis x index
  indexSelect := fun {s} axis count _axisInBounds x indices =>
    Runtime.Autograd.TypedGraph.GraphM.indexSelect
      (α := α) (Γ := Γ) (s := s) axis count x indices
  scatterAdd := fun {s} axis count _axisInBounds base source indices =>
    Runtime.Autograd.TypedGraph.GraphM.scatterAdd
      (α := α) (Γ := Γ) (s := s) axis count base source indices
  matmul := fun {batchA batchB batch : Shape} {mDim nDim pDim : Nat}
      {broadcastA} {broadcastB} a b =>
    Runtime.Autograd.TypedGraph.GraphM.matmul (α := α) (Γ := Γ)
      (batchA := batchA) (batchB := batchB) (batch := batch)
      (m := mDim) (n := nDim) (p := pDim)
      (broadcastA := broadcastA) (broadcastB := broadcastB) a b
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
  smoothMaxPool := fun {d C} {inSpatial kernel stride padding} {hKernel}
      [_decidableEq : DecidableEq α] x beta =>
    Runtime.Autograd.TypedGraph.GraphM.smoothMaxPool (α := α) (Γ := Γ)
      (d := d) (C := C)
      (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
      (hKernel := hKernel)
      x beta
  relu := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.relu (α := α) (Γ := Γ) (s := s) x
  sigmoid := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.sigmoid (α := α) (Γ := Γ) (s := s) x
  tanh := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.tanh (α := α) (Γ := Γ) (s := s) x
  gelu := fun {s} x => Runtime.Autograd.TypedGraph.GraphM.gelu (α := α) (Γ := Γ) (s := s) x
  softmaxLast := fun {s} x =>
    Runtime.Autograd.TypedGraph.GraphM.softmaxLast (α := α) (Γ := Γ) (s := s) x
  logSoftmaxLast := fun {s} x =>
    Runtime.Autograd.TypedGraph.GraphM.logSoftmaxLast (α := α) (Γ := Γ) (s := s) x
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
  batchNorm := fun {channels sSpatial} hWellFormed x gamma beta =>
    Runtime.Autograd.TypedGraph.GraphM.batchNorm (α := α) (Γ := Γ)
      (channels := channels) (sSpatial := sSpatial) hWellFormed x gamma beta
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
structure ScalarTrainer (α δ : Type) (paramShapes inputShapes : List Shape)
    (dataInputShapes : List Shape := []) where
  /-- Mutable module state. Entries marked `requiresGrad = false` are persistent buffers. -/
  state : ParamList α paramShapes
  /-- Compute the scalar loss for a curried input pack. -/
  loss :
    Curried.Fn α inputShapes
      (Curried.Fn δ dataInputShapes (IO (Tensor α .scalar)))
  /-- Compute the scalar loss and parameter gradients from one forward tape. -/
  lossAndGradState :
    Curried.Fn α inputShapes
      (Curried.Fn δ dataInputShapes
        (IO (Tensor α .scalar × _root_.TorchLean.TensorPack α paramShapes)))
  /-- Compute gradients aligned with `paramShapes` for a curried input pack. -/
  gradState :
    Curried.Fn α inputShapes
      (Curried.Fn δ dataInputShapes (IO (_root_.TorchLean.TensorPack α paramShapes)))
  /-- Apply one SGD-style update and return the loss used to compute that update. -/
  stepWithLoss : α →
    Curried.Fn α inputShapes
      (Curried.Fn δ dataInputShapes (IO (Tensor α .scalar)))
  /-- Apply one SGD-style update for a curried input pack. -/
  step : α → Curried.Fn α inputShapes (Curried.Fn δ dataInputShapes (IO Unit))
  /--
  Optional Adam update path.

  In eager CUDA mode this is a device-gradient/device-moment update path.  Other backends expose
  `none` and should use the generic optimizer wrappers.
  -/
  adamStep? : Option (α → α → α → α →
    Curried.Fn α inputShapes (Curried.Fn δ dataInputShapes (IO Unit))) := none
  /-- CUDA-native Adam update that also returns the loss from its forward tape. -/
  adamStepWithLoss? :
    Option (α → α → α → α →
      Curried.Fn α inputShapes
        (Curried.Fn δ dataInputShapes (IO (Tensor α .scalar)))) := none
  /--
  Optional AdamW update path.

  In eager CUDA mode this is a device-gradient/device-moment update path with decoupled weight
  decay. Other backends expose `none` and should use the generic optimizer wrappers.
  -/
  adamWStep? : Option (α → α → α → α → α →
    Curried.Fn α inputShapes (Curried.Fn δ dataInputShapes (IO Unit))) := none
  /-- CUDA-native AdamW update that also returns the loss from its forward tape. -/
  adamWStepWithLoss? :
    Option (α → α → α → α → α →
      Curried.Fn α inputShapes
        (Curried.Fn δ dataInputShapes (IO (Tensor α .scalar)))) := none
  /-- Save and restore optimizer state retained inside the selected runtime backend. -/
  optimizerStateCheckpoint? : Option OptimizerStateCheckpoint := none
  /-- Read current module state, synchronizing device mirrors if needed. -/
  getState : IO (_root_.TorchLean.TensorPack α paramShapes)

namespace Internal

/--
Extract gradients (as a typed `_root_.TorchLean.TensorPack`) for a list of eager `TensorRef`s from a dense gradient array.
-/
def gradsOfRefs {α : Type} [DecidableEq Shape] :
    {ss : List Shape} → Array (Spec.SomeTensor α) → RefList (TensorRef α) ss → IO (_root_.TorchLean.TensorPack α ss)
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
def useParams {α : Type} [TensorTransfer α] [DecidableEq Shape] :
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
def useInputs {α : Type} [TensorTransfer α] [DecidableEq Shape] :
    {ss : List Shape} → _root_.TorchLean.TensorPack α ss → EagerM α (RefList (TensorRef α) ss)
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
def scalarTrainer {α δ : Type} [Context α] [TensorTransfer α]
    [DecidableEq Shape] {paramShapes inputShapes dataInputShapes : List Shape}
    (opts : Options := {})
    (initRequiresGrad : Array Bool := Array.replicate paramShapes.length true)
    (validateDataInputs : _root_.TorchLean.TensorPack δ dataInputShapes → Except String Unit :=
      fun _ => pure ())
    (loss :
      ∀ {m : Type → Type}, [Monad m] → [Ops (m := m) (α := α)] →
        CurriedRef (fun s => Ops.Ref (m := m) (α := α) s) (paramShapes ++ inputShapes)
          (CurriedRef (fun s => Ops.DataRef (m := m) (α := α) δ s) dataInputShapes
            (m (Ops.Ref (m := m) (α := α) Shape.scalar)))) :
    Curried.Fn α paramShapes
      (IO (ScalarTrainer α δ paramShapes inputShapes dataInputShapes)) :=
    Curried.curry (α := α) (ss := paramShapes)
      (β := IO (ScalarTrainer α δ paramShapes inputShapes dataInputShapes))
    (fun initParams => do
    let ps ← ParamList.ofPackWithRequiresGrad (α := α) initParams initRequiresGrad
    let validateDataInputsIO (inputs : _root_.TorchLean.TensorPack δ dataInputShapes) : IO Unit :=
      match validateDataInputs inputs with
      | .error message => throw <| IO.userError message
      | .ok () => pure ()
    match opts.execution with
    | .typedGraph =>
        if opts.device != .cpu then
          throw <| IO.userError
            s!"typed graph execution currently supports device `cpu`; requested `{opts.deviceName}`"
        let Γ : List Shape := paramShapes ++ inputShapes
        let Δ : Type := _root_.TorchLean.TensorPack δ dataInputShapes
        let build : Runtime.Autograd.TypedGraph.GraphM.MWith α Δ Γ
            (Runtime.Autograd.TypedGraph.GraphM.Var Shape.scalar) := do
          let vs ← Runtime.Autograd.TypedGraph.GraphM.args (α := α) (Γ := Γ)
          let withDataInputs :=
            CurriedRef.applyVarList (Γ := Γ)
              (β := CurriedRef (fun s => Δ → Tensor δ s) dataInputShapes
                (Runtime.Autograd.TypedGraph.GraphM.MWith α Δ Γ
                  (Runtime.Autograd.TypedGraph.GraphM.Var Shape.scalar)))
              (loss (m := Runtime.Autograd.TypedGraph.GraphM.MWith α Δ Γ)) vs
          CurriedRef.applyPackProjections (full := dataInputShapes) id withDataInputs
        let graph ← okOrThrow
          (lowerToTypedGraphWithData (α := α) (Δ := Δ) (Γ := Γ) (τ := Shape.scalar) build)
        let ssFull : List Shape := graph.nodeShapes
        let fullGraph : Proofs.Autograd.Algebra.GraphData α Δ Γ ssFull :=
          graph.data
        let outId : Nat := graph.output.i.val

        let getScalarFromTape (t : Runtime.Autograd.Tape α) : IO (Tensor α .scalar) := do
          let any ← match t.getValue? outId with
            | some v => pure v
            | none => throw <| IO.userError "typed graph execution: missing output value in tape"
          if h : any.shape = Shape.scalar then
            pure (any.cast h)
          else
            throw <| IO.userError
              s!"typed graph execution: output shape mismatch (expected scalar, got {Shape.pretty any.shape})"

        let runTape (xs : _root_.TorchLean.TensorPack α inputShapes) (dataInputs : Δ) :
            IO (Runtime.Autograd.Tape α) := do
          validateDataInputsIO dataInputs
          let pv ← ParamList.values (α := α) ps
          let args := TorchLean.TensorPack.append (α := α) (ss₁ := paramShapes)
            (ss₂ := inputShapes) pv xs
          let (tape, _ctx) :=
            Proofs.Autograd.Algebra.Graph.lowerGraphDataToTape
              (α := α) (Δ := Δ) (Γ := Γ) (ss := ssFull) fullGraph args dataInputs
          pure tape
        let lossFn :
            Curried.Fn α inputShapes
              (Curried.Fn δ dataInputShapes (IO (Tensor α .scalar))) :=
          Curried.curry (α := α) (ss := inputShapes)
            (β := Curried.Fn δ dataInputShapes (IO (Tensor α .scalar))) (fun xs =>
              Curried.curry (α := δ) (ss := dataInputShapes)
                (β := IO (Tensor α .scalar)) (fun dataInputs =>
                  runTape xs dataInputs >>= getScalarFromTape))
        let lossAndGradState :
            Curried.Fn α inputShapes (Curried.Fn δ dataInputShapes
              (IO (Tensor α .scalar × _root_.TorchLean.TensorPack α paramShapes))) :=
          Curried.curry (α := α) (ss := inputShapes)
            (β := Curried.Fn δ dataInputShapes
              (IO (Tensor α .scalar × _root_.TorchLean.TensorPack α paramShapes))) (fun xs =>
              Curried.curry (α := δ) (ss := dataInputShapes)
                (β := IO (Tensor α .scalar × _root_.TorchLean.TensorPack α paramShapes)) (fun dataInputs => do
                  let tape ← runTape xs dataInputs
                  let lossValue ← getScalarFromTape tape
                  let grads ← okOrThrow
                    (Runtime.Autograd.TypedGraph.backwardDenseAllFrom
                      (α := α) (Γ := Γ) (ss := ssFull) tape graph.output
                      (Tensor.scalar (1 : α)))
                  let paramGrads ← okOrThrow
                    (TorchLean.TensorPack.ofShapeErasedArray
                      (α := α) grads (shapes := paramShapes))
                  pure (lossValue, paramGrads)))
        let gradState :
            Curried.Fn α inputShapes
              (Curried.Fn δ dataInputShapes (IO (_root_.TorchLean.TensorPack α paramShapes))) :=
          Curried.curry (α := α) (ss := inputShapes)
            (β := Curried.Fn δ dataInputShapes (IO (_root_.TorchLean.TensorPack α paramShapes))) (fun xs =>
              Curried.curry (α := δ) (ss := dataInputShapes)
                (β := IO (_root_.TorchLean.TensorPack α paramShapes)) (fun dataInputs => do
                  let tape ← runTape xs dataInputs
                  let grads ← okOrThrow
                    (Runtime.Autograd.TypedGraph.backwardDenseAllFrom
                      (α := α) (Γ := Γ) (ss := ssFull) tape graph.output
                      (Tensor.scalar (1 : α)))
                  okOrThrow (TorchLean.TensorPack.ofShapeErasedArray
                    (α := α) grads (shapes := paramShapes))))
        let stepWithLoss (lr : α) :
            Curried.Fn α inputShapes
              (Curried.Fn δ dataInputShapes (IO (Tensor α .scalar))) :=
          Curried.curry (α := α) (ss := inputShapes)
            (β := Curried.Fn δ dataInputShapes (IO (Tensor α .scalar))) (fun xs =>
              Curried.curry (α := δ) (ss := dataInputShapes)
                (β := IO (Tensor α .scalar)) (fun dataInputs => do
                  let lossAndGradStateForData :=
                    Curried.uncurry (α := α) (ss := inputShapes)
                      (β := Curried.Fn δ dataInputShapes
                        (IO (Tensor α .scalar × _root_.TorchLean.TensorPack α paramShapes)))
                      lossAndGradState xs
                  let (lossValue, grads) ←
                    Curried.uncurry (α := δ) (ss := dataInputShapes)
                      (β := IO (Tensor α .scalar × _root_.TorchLean.TensorPack α paramShapes))
                      lossAndGradStateForData dataInputs
                  ParamList.sgdStep (α := α) (ss := paramShapes) ps lr grads
                  pure lossValue))
        let step (lr : α) :
            Curried.Fn α inputShapes (Curried.Fn δ dataInputShapes (IO Unit)) :=
          Curried.curry (α := α) (ss := inputShapes)
            (β := Curried.Fn δ dataInputShapes (IO Unit)) (fun xs =>
              Curried.curry (α := δ) (ss := dataInputShapes) (β := IO Unit)
                (fun dataInputs => do
                  let stepForData :=
                    Curried.uncurry (α := α) (ss := inputShapes)
                      (β := Curried.Fn δ dataInputShapes
                        (IO (Tensor α .scalar))) (stepWithLoss lr) xs
                  let _ ← Curried.uncurry (α := δ) (ss := dataInputShapes)
                    (β := IO (Tensor α .scalar)) stepForData dataInputs
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
          { shapes := paramShapes.toArray, requiresGrad := ParamList.requiresGradArray ps }
        let lossEager := loss (m := Internal.EagerM α)
        let recordLoss (xs : _root_.TorchLean.TensorPack α inputShapes) (dataInputs : _root_.TorchLean.TensorPack δ dataInputShapes) :
            IO (TensorRef α Shape.scalar × RefList (TensorRef α) paramShapes) := do
          validateDataInputsIO dataInputs
          sess.resetTape
          (do
            let pRefs ← Internal.useParams (α := α) (ss := paramShapes) ps
            let xRefs ← Internal.useInputs (α := α) (ss := inputShapes) xs
            let allRefs := RefList.append (ss₁ := paramShapes) (ss₂ := inputShapes) pRefs xRefs
            let lossWithData :=
              CurriedRef.uncurry (ss := paramShapes ++ inputShapes) lossEager allRefs
            let lossRef ←
              CurriedRef.uncurryPack (α := δ) (ss := dataInputShapes) lossWithData dataInputs
            pure (lossRef, pRefs)) |>.run sess
        let finishCudaStep : IO Unit := do
          Internal.EagerSession.releaseCudaTapeAfterOptimizerStep sess
          sess.cudaTape.set Runtime.Autograd.Cuda.Tape.empty
          sess.paramsByLeaf.set (Std.HashMap.emptyWithCapacity)
          sess.nats.set #[]
          Internal.EagerSession.collectCudaAllocator
        let lossFn :
            Curried.Fn α inputShapes
              (Curried.Fn δ dataInputShapes (IO (Tensor α .scalar))) :=
          Curried.curry (α := α) (ss := inputShapes)
            (β := Curried.Fn δ dataInputShapes (IO (Tensor α .scalar))) (fun xs =>
              Curried.curry (α := δ) (ss := dataInputShapes)
                (β := IO (Tensor α .scalar)) (fun dataInputs => do
                  let (lossRef, _) ← recordLoss xs dataInputs
                  Internal.EagerSession.getValue (α := α) sess (sh := Shape.scalar) lossRef))
        let lossAndGradState :
            Curried.Fn α inputShapes (Curried.Fn δ dataInputShapes
              (IO (Tensor α .scalar × _root_.TorchLean.TensorPack α paramShapes))) :=
          Curried.curry (α := α) (ss := inputShapes)
            (β := Curried.Fn δ dataInputShapes
              (IO (Tensor α .scalar × _root_.TorchLean.TensorPack α paramShapes))) (fun xs =>
              Curried.curry (α := δ) (ss := dataInputShapes)
                (β := IO (Tensor α .scalar × _root_.TorchLean.TensorPack α paramShapes)) (fun dataInputs => do
                  let (lossRef, pRefs) ← recordLoss xs dataInputs
                  let lossValue ←
                    Internal.EagerSession.getValue (α := α) sess (sh := Shape.scalar) lossRef
                  let grads ← Internal.EagerSession.backwardScalarDenseAll (α := α) sess lossRef
                  let paramGrads ← Internal.gradsOfRefs (α := α) (ss := paramShapes) grads pRefs
                  pure (lossValue, paramGrads)))
        let gradState :
            Curried.Fn α inputShapes
              (Curried.Fn δ dataInputShapes (IO (_root_.TorchLean.TensorPack α paramShapes))) :=
          Curried.curry (α := α) (ss := inputShapes)
            (β := Curried.Fn δ dataInputShapes (IO (_root_.TorchLean.TensorPack α paramShapes))) (fun xs =>
              Curried.curry (α := δ) (ss := dataInputShapes)
                (β := IO (_root_.TorchLean.TensorPack α paramShapes)) (fun dataInputs => do
                  let (lossRef, pRefs) ← recordLoss xs dataInputs
                  let grads ← Internal.EagerSession.backwardScalarDenseAll (α := α) sess lossRef
                  Internal.gradsOfRefs (α := α) (ss := paramShapes) grads pRefs))
        let stepWithLoss (lr : α) :
            Curried.Fn α inputShapes
              (Curried.Fn δ dataInputShapes (IO (Tensor α .scalar))) :=
          Curried.curry (α := α) (ss := inputShapes)
            (β := Curried.Fn δ dataInputShapes (IO (Tensor α .scalar))) (fun xs =>
              Curried.curry (α := δ) (ss := dataInputShapes)
                (β := IO (Tensor α .scalar)) (fun dataInputs => do
                  if opts.usesCuda then
                    let (lossRef, _) ← recordLoss xs dataInputs
                    let lossValue ←
                      Internal.EagerSession.getValue (α := α) sess (sh := Shape.scalar) lossRef
                    let gradsDev ←
                      Internal.EagerSession.backwardScalarParamGradsCuda (α := α) sess lossRef
                    Internal.EagerSession.withCudaGradMap gradsDev fun gradsDev =>
                      Internal.EagerSession.sgdStepAllCudaMap (α := α) sess lr gradsDev
                    finishCudaStep
                    pure lossValue
                  else
                    let lossAndGradStateForData :=
                      Curried.uncurry (α := α) (ss := inputShapes)
                        (β := Curried.Fn δ dataInputShapes
                          (IO (Tensor α .scalar × _root_.TorchLean.TensorPack α paramShapes)))
                        lossAndGradState xs
                    let (lossValue, grads) ←
                      Curried.uncurry (α := δ) (ss := dataInputShapes)
                        (β := IO (Tensor α .scalar × _root_.TorchLean.TensorPack α paramShapes))
                        lossAndGradStateForData dataInputs
                    ParamList.sgdStep (α := α) (ss := paramShapes) ps lr grads
                    pure lossValue))
        let step (lr : α) :
            Curried.Fn α inputShapes (Curried.Fn δ dataInputShapes (IO Unit)) :=
          Curried.curry (α := α) (ss := inputShapes)
            (β := Curried.Fn δ dataInputShapes (IO Unit)) (fun xs =>
              Curried.curry (α := δ) (ss := dataInputShapes) (β := IO Unit)
                (fun dataInputs => do
                  if opts.usesCuda then
                    let (lossRef, _) ← recordLoss xs dataInputs
                    let gradsDev ←
                      Internal.EagerSession.backwardScalarParamGradsCuda (α := α) sess lossRef
                    Internal.EagerSession.withCudaGradMap gradsDev fun gradsDev =>
                      Internal.EagerSession.sgdStepAllCudaMap (α := α) sess lr gradsDev
                    finishCudaStep
                  else
                    let gradStateForData :=
                      Curried.uncurry (α := α) (ss := inputShapes)
                        (β := Curried.Fn δ dataInputShapes (IO (_root_.TorchLean.TensorPack α paramShapes))) gradState xs
                    let g ← Curried.uncurry (α := δ) (ss := dataInputShapes)
                      (β := IO (_root_.TorchLean.TensorPack α paramShapes)) gradStateForData dataInputs
                    ParamList.sgdStep (α := α) (ss := paramShapes) ps lr g))
        let adamStep? : Option (α → α → α → α →
            Curried.Fn α inputShapes (Curried.Fn δ dataInputShapes (IO Unit))) :=
          if opts.usesCuda then
            some (fun lr beta1 beta2 epsilon =>
              Curried.curry (α := α) (ss := inputShapes)
                (β := Curried.Fn δ dataInputShapes (IO Unit)) (fun xs =>
                  Curried.curry (α := δ) (ss := dataInputShapes) (β := IO Unit)
                    (fun dataInputs => do
                      let (lossRef, _) ← recordLoss xs dataInputs
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
                (Curried.Fn δ dataInputShapes (IO (Tensor α .scalar)))) :=
          if opts.usesCuda then
            some (fun lr beta1 beta2 epsilon =>
              Curried.curry (α := α) (ss := inputShapes)
                (β := Curried.Fn δ dataInputShapes (IO (Tensor α .scalar))) (fun xs =>
                  Curried.curry (α := δ) (ss := dataInputShapes)
                    (β := IO (Tensor α .scalar)) (fun dataInputs => do
                      let (lossRef, _) ← recordLoss xs dataInputs
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
            Curried.Fn α inputShapes (Curried.Fn δ dataInputShapes (IO Unit))) :=
          if opts.usesCuda then
            some (fun lr weightDecay beta1 beta2 epsilon =>
              Curried.curry (α := α) (ss := inputShapes)
                (β := Curried.Fn δ dataInputShapes (IO Unit)) (fun xs =>
                  Curried.curry (α := δ) (ss := dataInputShapes) (β := IO Unit)
                    (fun dataInputs => do
                      let (lossRef, _) ← recordLoss xs dataInputs
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
                (Curried.Fn δ dataInputShapes (IO (Tensor α .scalar)))) :=
          if opts.usesCuda then
            some (fun lr weightDecay beta1 beta2 epsilon =>
              Curried.curry (α := α) (ss := inputShapes)
                (β := Curried.Fn δ dataInputShapes (IO (Tensor α .scalar))) (fun xs =>
                  Curried.curry (α := δ) (ss := dataInputShapes)
                    (β := IO (Tensor α .scalar)) (fun dataInputs => do
                      let (lossRef, _) ← recordLoss xs dataInputs
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
