/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Macros
public import NN.API.Runtime
public import NN.Tensor
public import NN.Runtime.Autograd.TorchLean.Functional
public import NN.Runtime.Autograd.TorchLean.Module.RuntimeInit
public import NN.Runtime.Autograd.TorchLean.NN
public import NN.Spec.Core.Shape

import Mathlib.Algebra.Order.Algebra
import NN.Spec.Core.Tensor
import NN.Spec.Core.TensorReductionShape.Reductions

@[expose] public section

namespace TorchLean

/-!
# Layer Construction

This module defines explicit-seed layer builders under `TorchLean.nn.Internal`. The seeded builders
in `NN.API.Seeded` allocate initialization seeds from a deterministic stream.
-/

namespace nn

/-- Sequential model type (TorchLean `Seq`), analogous to PyTorch `nn.Sequential`. -/
abbrev Sequential := _root_.Runtime.Autograd.TorchLean.NN.Seq

export _root_.Runtime.Autograd.TorchLean.NN (Layer)

/-!
Expose common `Seq` helpers under `TorchLean.nn`.

The names mirror the TorchLean runtime layer so users can move between the public API and
runtime layer code without learning a second vocabulary.
-/
export _root_.Runtime.Autograd.TorchLean.NN.Seq
  (stateShapes requiresGrad initState runtimeInit? hasBufferUpdates updateBuffers)

/-! Constructors that pair an immutable model with a scalar training loss. -/
namespace Objective

export _root_.Runtime.Autograd.TorchLean.NN.Seq.Objective
  (createWithMode create mseWithMode mse oneHotCrossEntropyWithMode oneHotCrossEntropy)

end Objective

/-- Lift a single layer definition into a sequential model. -/
def of {σ τ : Spec.Shape} (layer : Layer σ τ) : Sequential σ τ :=
  _root_.Runtime.Autograd.TorchLean.NN.singleLayer layer

/-!
All explicit-seed layer constructors live under `nn.Internal.*`.

The top-level `nn.*` namespace is reserved for the *seeded builder* API that allocates
initialization seeds automatically (PyTorch-style ergonomics).
-/
namespace Internal

universe u v

/-- Convert a layer-like value to a sequential model for `seq!` composition. -/
class AsSequential (F : Spec.Shape → Spec.Shape → Sort u) where
  asSequential : {σ τ : Spec.Shape} → F σ τ → Sequential σ τ

instance : AsSequential Layer where
  asSequential := _root_.Runtime.Autograd.TorchLean.NN.singleLayer

instance : AsSequential Sequential where
  asSequential := id

/-- Compose layers and sequential models accepted by the `seq!` syntax. -/
def compose {σ τ υ : Spec.Shape}
    {F : Spec.Shape → Spec.Shape → Sort u} {G : Spec.Shape → Spec.Shape → Sort v}
    [AsSequential F] [AsSequential G] (f : F σ τ) (g : G τ υ) : Sequential σ υ :=
  _root_.Runtime.Autograd.TorchLean.NN.Seq.comp
    (AsSequential.asSequential f) (AsSequential.asSequential g)

/-- Parameter initialization for an affine layer. `none` selects Xavier-uniform weights. -/
structure Linear where
  weightInit? : Option _root_.Runtime.Autograd.Torch.Init.Scheme := none
  biasInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .zeros

/--
Linear layer on the last axis (prefix-shape preserving).

PyTorch analogue: `torch.nn.Linear`.
See `https://pytorch.org/docs/stable/generated/torch.nn.Linear.html`.

Unlike the runtime TorchLean layer constructor (which is vector-only),
this public layer constructor follows PyTorch’s convention:

- if `x` has shape `[..., inDim]`, `linear inDim outDim` returns a model of shape `[..., outDim]`.

The leading “prefix” dimensions are treated as a batch (they are flattened to `(numel(prefix),
  inDim)`,
the affine map is applied once, and the result is reshaped back).
-/
def linearWith (inDim outDim : Nat) (cfg : Linear) (seedW seedB : Nat := 0)
    (leading : List Nat := []) :
    Sequential (leading ++ [inDim]) (leading ++ [outDim]) :=
  let leadingShape := Spec.Shape.ofList leading
  let WShape : Spec.Shape := [outDim, inDim]
  let bShape : Spec.Shape := [outDim]
  let weightInit := cfg.weightInit?.getD (.xavierUniform inDim outDim)
  let w0 : Spec.Tensor Float WShape := _root_.Runtime.Autograd.Torch.Init.tensor
    (s := WShape) (sch := weightInit) (seed := seedW)
  let b0 : Spec.Tensor Float bShape := _root_.Runtime.Autograd.Torch.Init.tensor
    (s := bShape) (sch := cfg.biasInit) (seed := seedB)
  let batch : Nat := Spec.Shape.size leadingShape
  of
    { kind := s!"Linear({inDim}, {outDim})"
      stateShapes := [WShape, bShape]
      initState := _root_.TorchLean.TensorPack! w0, b0
      runtimeInit := some (.cons
        (_root_.Runtime.Autograd.TorchLean.Module.RuntimeInit.FloatInit.ofScheme weightInit seedW)
        (.cons
          (_root_.Runtime.Autograd.TorchLean.Module.RuntimeInit.FloatInit.ofScheme cfg.biasInit seedB)
          .nil))
      requiresGrad := #[true, true]
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun w b x =>
            let sIn : Spec.Shape := leading ++ [inDim]
            let sOut : Spec.Shape := leading ++ [outDim]
            ((do
              let x2d ←
                _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
                  (s₁ := sIn)
                  (s₂ := [batch, inDim])
                  x (by
                    -- size(sIn) = size(leading) * inDim = batch * inDim = size(Mat batch inDim)
                    simp [sIn, batch, leadingShape, Spec.Shape.size_concat,
                      Spec.Shape.size_ofList, Spec.Shape.size])

              let wT ←
                _root_.Runtime.Autograd.Torch.swapAdjacentAtDepth (m := m) (α := α)
                  (s := [outDim, inDim]) 0 w
              let y ← _root_.Runtime.Autograd.Torch.matmul (m := m) (α := α)
                (batchA := .scalar) (batchB := .scalar) (batch := .scalar)
                (mDim := batch) (nDim := inDim) (pDim := outDim) x2d wT
              let y2d ←
                _root_.Runtime.Autograd.TorchLean.F.addB (m := m) (α := α)
                  (t := [batch, outDim]) y b
              _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
                (s₁ := [batch, outDim])
                (s₂ := sOut)
                y2d (by
                  -- size(Mat batch outDim) = batch * outDim = size(leading) * outDim = size(sOut)
                  simp [sOut, batch, leadingShape, Spec.Shape.size_concat,
                    Spec.Shape.size_ofList, Spec.Shape.size])
            ) : m (_root_.TorchLean.Runtime.ValueRef (m := m) (α := α) sOut))
    }

/-- Linear layer with Xavier-uniform weights and zero bias. -/
def linear (inDim outDim : Nat) (seedW seedB : Nat := 0)
    (leading : List Nat := []) :
    Sequential (leading ++ [inDim]) (leading ++ [outDim]) :=
  linearWith inDim outDim {} seedW seedB leading

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
      [seqLen, inputSize]
      [seqLen, hiddenSize] :=
  of (_root_.Runtime.Autograd.TorchLean.NN.rnn (seqLen := seqLen) (inputSize := inputSize) (hiddenSize := hiddenSize) seedW
    seedB)

/--
GRU layer (time-major sequence, no batch axis).

This is implemented by unrolling `seqLen` Cho-style steps using existing TorchLean ops, so it runs
on both CPU and CUDA backends. PyTorch uses a different reset-after candidate parameterization;
its GRU checkpoints are not directly compatible with this constructor.
-/
def gru (seqLen inputSize hiddenSize : Nat) (seedW seedB : Nat := 0) :
    Sequential
      [seqLen, inputSize]
      [seqLen, hiddenSize] :=
  of (_root_.Runtime.Autograd.TorchLean.NN.gru (seqLen := seqLen) (inputSize := inputSize) (hiddenSize := hiddenSize) seedW
    seedB)

/--
Trainable Mamba-style gated diagonal state-space layer.

The layer is time-major and single-batch, matching the simple `rnn`/`gru`/`lstm` constructors:
input `(seqLen × inputSize)`, output `(seqLen × hiddenSize)`.  It is unrolled with differentiable
TorchLean ops, so CPU and CUDA training use the same API.
-/
def mamba (seqLen inputSize hiddenSize : Nat) (seedW seedB : Nat := 0) :
    Sequential
      [seqLen, inputSize]
      [seqLen, hiddenSize] :=
  of (_root_.Runtime.Autograd.TorchLean.NN.mamba (seqLen := seqLen) (inputSize := inputSize) (hiddenSize := hiddenSize)
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
      [seqLen, inputSize]
      [seqLen, hiddenSize] :=
  of (_root_.Runtime.Autograd.TorchLean.NN.lstm (seqLen := seqLen) (inputSize := inputSize) (hiddenSize := hiddenSize) seedW
    seedB)


/-- Elementwise ReLU. PyTorch analogue: `torch.nn.ReLU` / `torch.nn.functional.relu`. -/
def relu {s : Spec.Shape} : Sequential s s :=
  of <| _root_.Runtime.Autograd.TorchLean.NN.relu (s := s)
/-- Elementwise SiLU/Swish. PyTorch analogue: `torch.nn.SiLU` / `torch.nn.functional.silu`. -/
def silu {s : Spec.Shape} : Sequential s s :=
  of <| _root_.Runtime.Autograd.TorchLean.NN.silu (s := s)
/-- Elementwise GELU. PyTorch analogue: `torch.nn.GELU` / `torch.nn.functional.gelu`. -/
def gelu {s : Spec.Shape} : Sequential s s :=
  of <| _root_.Runtime.Autograd.TorchLean.NN.gelu (s := s)
/-- Elementwise sigmoid. PyTorch analogue: `torch.nn.Sigmoid` / `torch.sigmoid`. -/
def sigmoid {s : Spec.Shape} : Sequential s s :=
  of <| _root_.Runtime.Autograd.TorchLean.NN.sigmoid (s := s)
/-- Elementwise tanh. PyTorch analogue: `torch.nn.Tanh` / `torch.tanh`. -/
def tanh {s : Spec.Shape} : Sequential s s :=
  of <| _root_.Runtime.Autograd.TorchLean.NN.tanh (s := s)
/-- Softmax over any valid tensor dimension. -/
def softmax {s : Spec.Shape} (axis : Nat) [Spec.Shape.AxisInBounds axis s] : Sequential s s :=
  of <| _root_.Runtime.Autograd.TorchLean.NN.softmax (s := s) axis
/-- Stable log-softmax over any valid tensor dimension. -/
def logSoftmax {s : Spec.Shape} (axis : Nat) [Spec.Shape.AxisInBounds axis s] : Sequential s s :=
  of <| _root_.Runtime.Autograd.TorchLean.NN.logSoftmax (s := s) axis
/-- Reduce-sum to a scalar. PyTorch analogue: `torch.sum`. -/
def sum {s : Spec.Shape} : Sequential s Spec.Shape.scalar :=
  of <| _root_.Runtime.Autograd.TorchLean.NN.sum (s := s)
/-- Flatten any tensor into a 1D vector of length `size s`. PyTorch analogue: `torch.flatten`. -/
def flatten {s : Spec.Shape} : Sequential s [Spec.Shape.size s] :=
  of <| _root_.Runtime.Autograd.TorchLean.NN.flatten (s := s)

/--
View a tensor with a new shape containing the same number of scalar entries.

This is the shape-typed counterpart of `torch.reshape`: the equality argument records at
construction time that the source and target shapes have equal size.
-/
def reshape (source target : Spec.Shape)
    (sameSize : Spec.Shape.size source = Spec.Shape.size target) :
    Sequential source target :=
  of
    { kind := "Reshape"
      stateShapes := []
      initState := .nil
      requiresGrad := #[]
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun x =>
            _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
              (s₁ := source) (s₂ := target) x sameSize }

/--
Flatten each tensor after an arbitrary collection of leading dimensions.

For `leading = [batch]`, this is the typed counterpart of
`torch.flatten(x, start_dim=1)`. Multiple leading dimensions are preserved without introducing a
separate batched tensor type.
-/
def flattenAfter (leading : List Nat := []) {shape : List Nat} :
    Sequential (leading ++ shape) (leading ++ [shape.prod]) :=
  let source : Spec.Shape := leading ++ shape
  let target : Spec.Shape := leading ++ [shape.prod]
  of
    { kind := "FlattenLeading"
      stateShapes := []
      initState := .nil
      requiresGrad := #[]
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun x =>
            _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
              (s₁ := source)
              (s₂ := target)
              x (by
                simp [source, target, Spec.Shape.size_concat, Spec.Shape.size_ofList,
                  Spec.Shape.size])
    }

/--
Dropout layer (active in train mode, identity in eval mode).

PyTorch analogue: `torch.nn.Dropout`.
-/
def dropout {s : Spec.Shape} (p : Float) (seed : Nat := 0) : Sequential s s :=
  of <| _root_.Runtime.Autograd.TorchLean.NN.dropout (s := s) p seed
