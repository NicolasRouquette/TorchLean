/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Adapters
public import NN.API.Common
public import NN.API.Data
public import NN.API.Macros
public import NN.API.Public.TensorPack
public import NN.API.Rand
public import NN.API.RL
public import NN.API.Runtime
public import NN.API.Samples
public import NN.API.SelfSupervised
public import NN.API.Text
public import NN.API.Text.Bpe
public import NN.Spec.Layers.PositionalEncoding

import Mathlib.Algebra.Order.Algebra

@[expose] public section

namespace NN
namespace API

/-!
# Public neural-network builders

This module defines the `NN.API.nn.Internal` layer builders and block constructors.  The names stay
under `NN.API.nn`; the split only keeps the source file small enough to maintain.
-/

namespace nn

/-- Sequential model type (TorchLean `Seq`), analogous to PyTorch `nn.Sequential`. -/
abbrev Sequential := TorchLean.LayerCore.Seq

/-- Single-layer definition type (TorchLean `LayerDef`), analogous to PyTorch `nn.Module`. -/
abbrev LayerDef := TorchLean.LayerCore.LayerDef

/-!
Re-export common `Seq` helpers under `API.nn.*` so examples can use the stable public API.

The names mirror the TorchLean runtime layer so users can move between the public API and
runtime layer code without learning a second vocabulary.
-/
export TorchLean.LayerCore.Seq
  (paramShapes paramRequiresGrad initParams runtimeInit? hasBufferUpdates updateBuffers
   programWithMode forwardProgram
   scalarModuleDefWithMode scalarModuleDef
   mseScalarModuleDefWithMode mseScalarModuleDef
   crossEntropyOneHotScalarModuleDefWithMode crossEntropyOneHotScalarModuleDef
   compileForwardWithMode compileForward
   forward predict forwardArtifact)

/-- Lift a single layer definition into a sequential model. -/
def of {σ τ : Spec.Shape} (layer : LayerDef σ τ) : Sequential σ τ :=
  TorchLean.Layers.of layer

/-!
All explicit-seed layer constructors live under `nn.Internal.*`.

The top-level `nn.*` namespace is reserved for the *seeded builder* API that allocates
initialization seeds automatically (PyTorch-style ergonomics).
-/
namespace Internal

/-- Parameter initialization for an affine layer. `none` selects Xavier-uniform weights. -/
structure Linear where
  weightInit? : Option _root_.Runtime.Autograd.Torch.Init.Scheme := none
  biasInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .zeros

/--
Linear layer on the last axis (prefix-shape preserving).

PyTorch analogue: `torch.nn.linear`.
See `https://pytorch.org/docs/stable/generated/torch.nn.linear.html`.

Unlike the runtime TorchLean layer constructor (which is vector-only),
this public layer constructor follows PyTorch’s convention:

- if `x` has shape `[..., inDim]`, `linear inDim outDim` returns a model of shape `[..., outDim]`.

The leading “prefix” dimensions are treated as a batch (they are flattened to `(numel(prefix),
  inDim)`,
the affine map is applied once, and the result is reshaped back).
-/
def linearWith (inDim outDim : Nat) (cfg : Linear) (seedW seedB : Nat := 0)
    (pfx : Spec.Shape := Spec.Shape.scalar) :
    Sequential (pfx.appendDim inDim) (pfx.appendDim outDim) :=
  let WShape : Spec.Shape := .dim outDim (.dim inDim .scalar)
  let bShape : Spec.Shape := .dim outDim .scalar
  let weightInit := cfg.weightInit?.getD (.xavierUniform inDim outDim)
  let w0 : Spec.Tensor Float WShape := _root_.Runtime.Autograd.Torch.Init.tensor
    (s := WShape) (sch := weightInit) (seed := seedW)
  let b0 : Spec.Tensor Float bShape := _root_.Runtime.Autograd.Torch.Init.tensor
    (s := bShape) (sch := cfg.biasInit) (seed := seedB)
  let batch : Nat := Spec.Shape.size pfx
  of
    { kind := s!"Linear({inDim}, {outDim})"
      paramShapes := [WShape, bShape]
      initParams := tensorpack! w0, b0
      runtimeInit := some (.cons
        (_root_.Runtime.Autograd.TorchLean.Module.RuntimeInit.FloatInit.ofScheme weightInit seedW)
        (.cons
          (_root_.Runtime.Autograd.TorchLean.Module.RuntimeInit.FloatInit.ofScheme cfg.biasInit seedB)
          .nil))
      paramRequiresGrad := [true, true]
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun w b x =>
            let sIn : Spec.Shape := pfx.appendDim inDim
            let sOut : Spec.Shape := pfx.appendDim outDim
            ((do
              let x2D ←
                TorchLean.reshape (m := m) (α := α)
                  (s₁ := sIn)
                  (s₂ := .dim batch (.dim inDim .scalar))
                  x (by
                    -- size(sIn) = size(pfx) * inDim = batch * inDim = size(Mat batch inDim)
                    simp [sIn, batch, Spec.Shape.size_appendDim, Spec.Shape.size])

              let wT ←
                TorchLean.transpose2d (m := m) (α := α)
                  (mDim := outDim) (nDim := inDim) w
              let y ← TorchLean.matmul (m := m) (α := α)
                (mDim := batch) (nDim := inDim) (pDim := outDim) x2D wT
              let y2D ←
                TorchLean.F.addB (m := m) (α := α) (t := .dim batch (.dim outDim .scalar)) y b
              TorchLean.reshape (m := m) (α := α)
                (s₁ := .dim batch (.dim outDim .scalar))
                (s₂ := sOut)
                y2D (by
                  -- size(Mat batch outDim) = batch * outDim = size(pfx) * outDim = size(sOut)
                  simp [sOut, batch, Spec.Shape.size_appendDim, Spec.Shape.size])
            ) : m (TorchLean.RefTy (m := m) (α := α) sOut))
    }

/-- Linear layer with Xavier-uniform weights and zero bias. -/
def linear (inDim outDim : Nat) (seedW seedB : Nat := 0)
    (pfx : Spec.Shape := Spec.Shape.scalar) :
    Sequential (pfx.appendDim inDim) (pfx.appendDim outDim) :=
  linearWith inDim outDim {} seedW seedB pfx

/--
Vanilla RNN layer (time-major sequence, no batch axis).

Semantics:

$$
h_t=\tanh\!\left(W[x_t;h_{t-1}]+b\right),\qquad h_{-1}=0.
$$

This is implemented by unrolling `seqLen` steps using existing TorchLean ops, so it runs on both
CPU and CUDA backends.

PyTorch analogy: `torch.nn.RNN(inputSize, hiddenSize, nonlinearity="tanh")` with
`batch_first=false`, specialized to a single batch element.
-/
def rnn (seqLen inputSize hiddenSize : Nat) (seedW seedB : Nat := 0) :
    Sequential
      (.dim seqLen (.dim inputSize .scalar))
      (.dim seqLen (.dim hiddenSize .scalar)) :=
  of (TorchLean.LayerCore.rnn (seqLen := seqLen) (inputSize := inputSize) (hiddenSize := hiddenSize) seedW
    seedB)

/--
GRU layer (time-major sequence, no batch axis).

This is implemented by unrolling `seqLen` steps using existing TorchLean ops, so it runs on both
CPU and CUDA backends.

PyTorch analogy: `torch.nn.GRU(inputSize, hiddenSize)` with `batch_first=false`, specialized to a
single batch element.
-/
def gru (seqLen inputSize hiddenSize : Nat) (seedW seedB : Nat := 0) :
    Sequential
      (.dim seqLen (.dim inputSize .scalar))
      (.dim seqLen (.dim hiddenSize .scalar)) :=
  of (TorchLean.LayerCore.gru (seqLen := seqLen) (inputSize := inputSize) (hiddenSize := hiddenSize) seedW
    seedB)

/--
Trainable Mamba-style gated diagonal state-space layer.

The layer is time-major and single-batch, matching the simple `rnn`/`gru`/`lstm` constructors:
input `(seqLen × inputSize)`, output `(seqLen × hiddenSize)`.  It is unrolled with differentiable
TorchLean ops, so CPU and CUDA training use the same API.
-/
def mamba (seqLen inputSize hiddenSize : Nat) (seedW seedB : Nat := 0) :
    Sequential
      (.dim seqLen (.dim inputSize .scalar))
      (.dim seqLen (.dim hiddenSize .scalar)) :=
  of (TorchLean.LayerCore.mamba (seqLen := seqLen) (inputSize := inputSize) (hiddenSize := hiddenSize)
    seedW seedB)

/--
LSTM layer (time-major sequence, no batch axis).

This is implemented by unrolling `seqLen` steps using existing TorchLean ops, so it runs on both
CPU and CUDA backends.

PyTorch analogy: `torch.nn.LSTM(inputSize, hiddenSize)` with `batch_first=false`, specialized to a
single batch element.
-/
def lstm (seqLen inputSize hiddenSize : Nat) (seedW seedB : Nat := 0) :
    Sequential
      (.dim seqLen (.dim inputSize .scalar))
      (.dim seqLen (.dim hiddenSize .scalar)) :=
  of (TorchLean.LayerCore.lstm (seqLen := seqLen) (inputSize := inputSize) (hiddenSize := hiddenSize) seedW
    seedB)

/--
Embedding table initialization configuration (one-hot / token-distribution inputs).

TorchLean-friendly analogue of `torch.nn.Embedding` in the common setting where token ids are
represented as one-hot vectors (or soft token distributions), so lookup is a matrix multiplication
rather than integer indexing.
-/
structure Embedding where
  /-- Seed for deterministic embedding-table initialization. -/
  seedW : Nat := 0
  /-- Initialization scheme for the embedding table. -/
  wInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .uniform (-0.02) 0.02

/--
Embedding layer for one-hot / token-distribution inputs (no bias).

Input shape: `[..., vocab]`
Output shape: `[..., embedDim]`

PyTorch analogue: conceptually `nn.Embedding(vocab, embedDim)` but applied to one-hot inputs.
-/
def embedding (vocab embedDim : Nat) (cfg : Embedding := {}) (pfx : Spec.Shape := Spec.Shape.scalar) :
    Sequential (pfx.appendDim vocab) (pfx.appendDim embedDim) :=
  let WShape : Spec.Shape := .dim vocab (.dim embedDim .scalar)
  let w0 : Spec.Tensor Float WShape :=
    _root_.Runtime.Autograd.Torch.Init.tensor (s := WShape) (sch := cfg.wInit) (seed := cfg.seedW)
  let batch : Nat := Spec.Shape.size pfx
  of
    { kind := s!"Embedding({vocab}, {embedDim})"
      paramShapes := [WShape]
      initParams := tensorpack! w0
      runtimeInit := some (.cons
        (_root_.Runtime.Autograd.TorchLean.Module.RuntimeInit.FloatInit.ofScheme
          cfg.wInit cfg.seedW) .nil)
      paramRequiresGrad := [true]
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun w x =>
            let sIn : Spec.Shape := pfx.appendDim vocab
            let sOut : Spec.Shape := pfx.appendDim embedDim
            ((do
              let x2D ←
                TorchLean.reshape (m := m) (α := α)
                  (s₁ := sIn)
                  (s₂ := .dim batch (.dim vocab .scalar))
                  x (by
                    -- size(sIn) = size(pfx) * vocab = batch * vocab
                    simp [sIn, batch, Spec.Shape.size_appendDim, Spec.Shape.size])
              let y ←
                TorchLean.matmul (m := m) (α := α)
                  (mDim := batch) (nDim := vocab) (pDim := embedDim)
                  x2D w
              TorchLean.reshape (m := m) (α := α)
                (s₁ := .dim batch (.dim embedDim .scalar))
                (s₂ := sOut)
                y (by
                  -- size(Mat batch embedDim) = batch * embedDim = size(pfx) * embedDim = size(sOut)
                  simp [sOut, batch, Spec.Shape.size_appendDim, Spec.Shape.size])
            ) : m (TorchLean.RefTy (m := m) (α := α) sOut))
    }

/--
Learned positional embedding configuration.

This is a trainable parameter tensor of shape `(seqLen × embedDim)` that is broadcast across the
leading batch dimension and added to the input.
-/
structure LearnedPositionalEmbedding where
  /-- Seed for deterministic initialization. -/
  seedPos : Nat := 0
  /-- Initialization scheme for the positional embedding table. -/
  posInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .uniform (-0.02) 0.02

/--
Add learned positional embeddings to a batched `(batch × seqLen × embedDim)` tensor.

PyTorch analogue: `x + pos[:seqLen]` where `pos` is a parameter table.
-/
def learnedPositionalEmbedding {batch seqLen embedDim : Nat} (cfg : LearnedPositionalEmbedding := {}) :
    Sequential
      (.dim batch (.dim seqLen (.dim embedDim .scalar)))
      (.dim batch (.dim seqLen (.dim embedDim .scalar))) :=
  let posShape : Spec.Shape := .dim seqLen (.dim embedDim .scalar)
  let xShape : Spec.Shape := .dim batch posShape
  let pos0 : Spec.Tensor Float posShape :=
    _root_.Runtime.Autograd.Torch.Init.tensor (s := posShape) (sch := cfg.posInit) (seed := cfg.seedPos)
  of
    { kind := "LearnedPositionalEmbedding"
      paramShapes := [posShape]
      initParams := tensorpack! pos0
      runtimeInit := some (.cons
        (_root_.Runtime.Autograd.TorchLean.Module.RuntimeInit.FloatInit.ofScheme
          cfg.posInit cfg.seedPos) .nil)
      paramRequiresGrad := [true]
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun pos x =>
            -- Broadcast `(seqLen × embedDim)` positional embeddings across the leading `batch` axis.
            (TorchLean.F.addB (m := m) (α := α) (s₁ := posShape) (s₂ := xShape) (t := xShape) pos x)
    }

/--
Sinusoidal positional encoding configuration.

Classic non-trainable Transformer sinusoidal encoding, added to token embeddings. `startPos` is an
absolute-position offset for KV-cache decoding.
-/
structure SinusoidalPositionalEncoding where
  /-- Absolute position offset for the first row of the encoding table. -/
  startPos : Nat := 0

/--
Add sinusoidal positional encodings to a batched `(batch × seqLen × embedDim)` tensor.

Implementation:
- precompute `PE : (seqLen × embedDim)` at initialization time (stored as a non-trainable buffer),
- broadcast it across the leading `batch` axis and add to the input.
-/
def sinusoidalPositionalEncoding {batch seqLen embedDim : Nat}
    (cfg : SinusoidalPositionalEncoding := {}) :
    Sequential
      (.dim batch (.dim seqLen (.dim embedDim .scalar)))
      (.dim batch (.dim seqLen (.dim embedDim .scalar))) :=
  let peShape : Spec.Shape := .dim seqLen (.dim embedDim .scalar)
  let xShape : Spec.Shape := .dim batch peShape
  let pe0 : Spec.Tensor Float peShape :=
    Spec.sinusoidalPositionalEncodingSpec (α := Float) seqLen embedDim cfg.startPos
  of
    { kind := "SinusoidalPositionalEncoding"
      paramShapes := [peShape]
      initParams := tensorpack! pe0
      paramRequiresGrad := [false]
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun pe x =>
            -- Broadcast `PE : (seqLen × embedDim)` across the leading `batch` axis.
            (TorchLean.F.addB (m := m) (α := α) (s₁ := peShape) (s₂ := xShape) (t := xShape) pe x)
    }

/--
Rotary positional embedding (RoPE) configuration.

`startPos` is an absolute-position offset for KV-cache decoding.
-/
structure RoPE where
  /-- Absolute position offset for the first row of RoPE angles. -/
  startPos : Nat := 0

/--
Apply RoPE to a batched multi-head tensor `(batch × numHeads × seqLen × headDim)`.

This matches the standard identity:

$$
\operatorname{rope}(x)
  = x \odot \cos + \operatorname{rotatePairs}(x) \odot \sin
$$

where `cos`/`sin` depend only on `(pos, dim)` and broadcast across `(batch, numHeads)`.

Notes:
- This layer is *differentiable* (gradients flow through the rotation), but it has no trainable
  parameters; the precomputed `cos`/`sin` tables are stored as non-trainable buffers.
- The pure spec version is in `NN.Spec.Layers.PositionalEncoding` (`Spec.rope_apply_heads_spec`).
-/
def rope {batch numHeads seqLen headDim : Nat} (cfg : RoPE := {}) :
    Sequential
      (.dim batch (.dim numHeads (.dim seqLen (.dim headDim .scalar))))
      (.dim batch (.dim numHeads (.dim seqLen (.dim headDim .scalar)))) :=
  let xShape : Spec.Shape := .dim batch (.dim numHeads (.dim seqLen (.dim headDim .scalar)))
  let csShape : Spec.Shape := .dim seqLen (.dim headDim .scalar)

  -- Precompute cos/sin tables (as Float buffers). These depend only on `(seqLen, headDim, startPos)`.
  let cos0 : Spec.Tensor Float csShape :=
    Spec.Tensor.dim (fun (pos : Fin seqLen) =>
      Spec.ropeCosLastdimSpec (α := Float) (cfg.startPos + pos.val) headDim)
  let sin0 : Spec.Tensor Float csShape :=
    Spec.Tensor.dim (fun (pos : Fin seqLen) =>
      Spec.ropeSinLastdimSpec (α := Float) (cfg.startPos + pos.val) headDim)

  -- Column permutation indices implementing pairwise swap `(0↔1, 2↔3, ...)`.
  -- When `headDim` is odd, the last index is left unchanged.
  let permIdx : Spec.Tensor Nat (.dim headDim .scalar) :=
    Spec.Tensor.dim (fun (j : Fin headDim) =>
      let idx := j.val
      let out : Nat :=
        if idx % 2 = 0 then
          if idx + 1 < headDim then idx + 1 else idx
        else
          idx - 1
      Spec.Tensor.scalar out)

  of
    { kind := "RoPE"
      paramShapes := [csShape, csShape]
      initParams := tensorpack! cos0, sin0
      paramRequiresGrad := [false, false]
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun cos sin x =>
            ((do
            -- Rotate last-dim pairs by a fixed 2D permutation/sign pattern.
            let rowsFold : Nat := batch * numHeads * seqLen
            let flatShape : Spec.Shape := .dim rowsFold (.dim headDim .scalar)

            let x2d ←
              TorchLean.reshape (m := m) (α := α)
                (s₁ := xShape) (s₂ := flatShape)
                x (by
                  -- size(xShape) = batch * numHeads * seqLen * headDim = rowsFold * headDim = size(flatShape)
                  simp [xShape, flatShape, rowsFold, Spec.Shape.size,
                    Nat.mul_left_comm, Nat.mul_comm])

            let xT ←
              TorchLean.transpose2d (m := m) (α := α)
                (mDim := rowsFold) (nDim := headDim) x2d

            let xPerm ←
              TorchLean.gatherRowsNat (m := m) (α := α)
                (rows := headDim) (cols := rowsFold) (k := headDim)
                xT permIdx

            let xBack ←
              TorchLean.transpose2d (m := m) (α := α)
                (mDim := headDim) (nDim := rowsFold) xPerm

            -- Sign pattern for `rotatePairs`: even outputs get a negation (except the final unpaired entry).
            let signT : Spec.Tensor α (.dim headDim .scalar) :=
              Spec.Tensor.dim (fun (j : Fin headDim) =>
                let idx := j.val
                let v : α :=
                  if idx % 2 = 0 ∧ idx + 1 < headDim then (-1 : α) else (1 : α)
                Spec.Tensor.scalar v)
            let sign ← TorchLean.const (m := m) (α := α) (s := .dim headDim .scalar) signT

            let xRot2d ←
              TorchLean.F.mulB (m := m) (α := α)
                (s₁ := flatShape) (s₂ := .dim headDim .scalar) (t := flatShape)
                xBack sign

            let xRot ←
              TorchLean.reshape (m := m) (α := α)
                (s₁ := flatShape) (s₂ := xShape)
                xRot2d (by
                  simp [xShape, flatShape, rowsFold, Spec.Shape.size,
                    Nat.mul_left_comm, Nat.mul_comm])

            -- Apply the RoPE formula with broadcasting of `cos/sin : (seqLen × headDim)`.
            let xCos ←
              TorchLean.F.mulB (m := m) (α := α)
                (s₁ := xShape) (s₂ := csShape) (t := xShape)
                x cos
            let rotSin ←
              TorchLean.F.mulB (m := m) (α := α)
                (s₁ := xShape) (s₂ := csShape) (t := xShape)
                xRot sin
            TorchLean.add (m := m) (α := α) (s := xShape) xCos rotSin
            ) : m (TorchLean.RefTy (m := m) (α := α) xShape))
    }

/-- Elementwise ReLU. PyTorch analogue: `torch.nn.relu` / `torch.nn.functional.relu`. -/
def relu {s : Spec.Shape} : Sequential s s := TorchLean.Layers.relu (s := s)
/-- Elementwise SiLU/Swish. PyTorch analogue: `torch.nn.silu` / `torch.nn.functional.silu`. -/
def silu {s : Spec.Shape} : Sequential s s := TorchLean.Layers.silu (s := s)
/-- Elementwise GELU. PyTorch analogue: `torch.nn.gelu` / `torch.nn.functional.gelu`. -/
def gelu {s : Spec.Shape} : Sequential s s := TorchLean.Layers.gelu (s := s)
/-- Elementwise sigmoid. PyTorch analogue: `torch.nn.sigmoid` / `torch.nn.functional.sigmoid`. -/
def sigmoid {s : Spec.Shape} : Sequential s s := TorchLean.Layers.sigmoid (s := s)
/-- Elementwise tanh. PyTorch analogue: `torch.nn.tanh` / `torch.nn.functional.tanh`. -/
def tanh {s : Spec.Shape} : Sequential s s := TorchLean.Layers.tanh (s := s)
/-- Softmax. PyTorch analogue: `torch.nn.softmax` / `torch.nn.functional.softmax`. -/
def softmax {s : Spec.Shape} : Sequential s s := TorchLean.Layers.softmax (s := s)
/-- Reduce-sum to a scalar. PyTorch analogue: `torch.sum`. -/
def sum {s : Spec.Shape} : Sequential s Spec.Shape.scalar := TorchLean.Layers.sum (s := s)
/-- Flatten any tensor into a 1D vector of length `size s`. PyTorch analogue: `torch.flatten`. -/
def flatten {s : Spec.Shape} : Sequential s (.dim (Spec.Shape.size s) .scalar) :=
  TorchLean.Layers.flatten (s := s)

/--
Flatten a batched tensor `N × σ` into a matrix `N × (size σ)`.

PyTorch analogue: `torch.flatten(x, start_dim=1)`.
-/
def flattenBatch {n : Nat} {s : Spec.Shape} :
    Sequential (.dim n s) (.dim n (.dim (Spec.Shape.size s) .scalar)) :=
  of
    { kind := "FlattenBatch"
      paramShapes := []
      initParams := .nil
      paramRequiresGrad := []
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun x =>
            TorchLean.reshape (m := m) (α := α)
              (s₁ := .dim n s)
              (s₂ := .dim n (.dim (Spec.Shape.size s) .scalar))
              x (by simp [Spec.Shape.size])
    }

/--
Dropout layer (active in train mode, identity in eval mode).

PyTorch analogue: `torch.nn.Dropout`.
-/
def dropout {s : Spec.Shape} (p : Float) (seed : Nat := 0) : Sequential s s :=
  TorchLean.Layers.dropout (s := s) p seed
/--
Convenience block: `Flatten -> Linear`.

This is common for "image to classifier head" models.
-/
def flattenLinear {s : Spec.Shape} (outDim : Nat) (seedW seedB : Nat := 0) :
    Sequential s (.dim outDim .scalar) :=
  TorchLean.Layers.flattenLinear (s := s) outDim seedW seedB
