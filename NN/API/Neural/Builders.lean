/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Macros
public import NN.API.TensorPack
public import NN.Runtime.Autograd.TorchLean
public import NN.Spec.Layers.PositionalEncoding

import Mathlib.Algebra.Order.Algebra
import NN.Spec.Core.Tensor.API
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

/-- Immutable definition of one shape-checked layer, including initialization and execution. -/
abbrev Layer := _root_.Runtime.Autograd.TorchLean.NN.Layer

/--
A shape-typed model whose input is a tensor of natural-number indices.

Parameters use the selected model scalar, while the input remains a discrete `Nat` tensor. This is
the public model type for embeddings, token models, and other indexed computations: indices cannot
accidentally acquire gradients or be reinterpreted as floating-point data.
-/
structure IndexedModel (σ τ : Spec.Shape) where
  /-- Model label used in summaries and diagnostics. -/
  kind : String := "IndexedModel"
  /-- Shapes of trainable parameters and persistent buffers. -/
  stateShapes : List Spec.Shape
  /-- Semantic initial values for the complete model state. -/
  initState : _root_.Runtime.Autograd.Torch.TList Float stateShapes
  /-- Optional storage-first initialization plan for executable `Float` backends. -/
  runtimeInit : Option
    (_root_.Runtime.Autograd.TorchLean.Module.RuntimeInit.Plan stateShapes) := none
  /-- Gradient flags aligned with `stateShapes`; persistent buffers carry `false`. -/
  requiresGrad : List Bool := List.replicate stateShapes.length true
  /-- Validate concrete indices before they reach a backend gather operation. -/
  validateInput : Spec.Tensor Nat σ → Except String Unit := fun _ => pure ()
  /-- Execution-polymorphic forward computation over one discrete input tensor. -/
  forward : ∀ (_mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
      {α : Type}, [_root_.Context α] → [DecidableEq Spec.Shape] →
      _root_.Runtime.Autograd.TorchLean.ProgramWithNatInputs α stateShapes [σ] τ

namespace IndexedModel

/-- Append an ordinary sequential model after an indexed-input model. -/
def andThen {σ τ υ : Spec.Shape} (first : IndexedModel σ τ) (rest : Sequential τ υ) :
    IndexedModel σ υ where
  kind := s!"{first.kind} → Sequential"
  stateShapes := first.stateShapes ++ _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes rest
  initState := TensorPack.append first.initState
    (_root_.Runtime.Autograd.TorchLean.NN.Seq.initState rest)
  runtimeInit :=
    match first.runtimeInit, _root_.Runtime.Autograd.TorchLean.NN.Seq.runtimeInit? rest with
    | some firstPlan, some restPlan => some (firstPlan.append restPlan)
    | _, _ => none
  requiresGrad := first.requiresGrad ++
    _root_.Runtime.Autograd.TorchLean.NN.Seq.requiresGrad rest
  validateInput := first.validateInput
  forward := fun mode {α} _ _ =>
    fun {m} _ _ =>
      _root_.Runtime.Autograd.Torch.CurriedRef.curry
        (Ref := fun s => _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) s)
        (ss := first.stateShapes ++ _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes rest)
        (β := _root_.Runtime.Autograd.Torch.CurriedRef
          (fun s => _root_.Runtime.Autograd.Torch.NatTensorRef (m := m) (α := α) s)
          [σ] (m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) υ)))
        (fun state =>
          let (firstState, restState) := _root_.Runtime.Autograd.Torch.RefList.split
            (Ref := fun s => _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) s)
            (ss₁ := first.stateShapes)
            (ss₂ := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes rest) state
          _root_.Runtime.Autograd.Torch.CurriedRef.curry
            (Ref := fun s => _root_.Runtime.Autograd.Torch.NatTensorRef
              (m := m) (α := α) s)
            (ss := [σ])
            (fun natInputs =>
              match natInputs with
              | .cons indices .nil => do
                  let embedded ←
                    (_root_.Runtime.Autograd.Torch.CurriedRef.uncurry
                      (Ref := fun s =>
                        _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) s)
                      (ss := first.stateShapes)
                      (β := _root_.Runtime.Autograd.Torch.CurriedRef
                        (fun s => _root_.Runtime.Autograd.Torch.NatTensorRef
                          (m := m) (α := α) s)
                        [σ]
                        (m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) τ)))
                      (first.forward mode (α := α)) firstState) indices
                  _root_.Runtime.Autograd.TorchLean.NN.Seq.forwardState
                    (model := rest) (α := α) (m := m) mode restState embedded))

/-! ### Scalar objectives -/

namespace Objective

/--
Pair an indexed-input model with a scalar loss under an explicit train/eval mode.

The resulting training module accepts one ordinary target tensor followed by the model's discrete
input tensor. Keeping those packs separate is important: token ids and table indices remain `Nat`
throughout execution and therefore cannot receive gradients or be silently rounded through a
floating-point scalar type.

The target shape is independent of the model output shape, so this constructor also supports losses
whose labels use a different representation from the prediction.
-/
def createWithMode {σ τ υ : Spec.Shape}
    (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
    (model : IndexedModel σ τ)
    (loss : ∀ {α : Type}, [_root_.Context α] → [DecidableEq Spec.Shape] →
      _root_.Runtime.Autograd.TorchLean.Program α [τ, υ] Spec.Shape.scalar) :
    _root_.Runtime.Autograd.TorchLean.Module.ObjectiveDef
      model.stateShapes [υ] [σ] :=
  { initState := model.initState
    runtimeInit := model.runtimeInit
    requiresGrad := model.requiresGrad
    validateNatInputs := fun
      | .cons input .nil => model.validateInput input
    loss := fun {α} => by
      intro _ _
      exact fun {m} _ _ =>
        _root_.Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := fun s => _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) s)
          (ss := model.stateShapes ++ [υ])
          (β := _root_.Runtime.Autograd.Torch.CurriedRef
            (fun s => _root_.Runtime.Autograd.Torch.NatTensorRef (m := m) (α := α) s)
            [σ]
            (m (_root_.Runtime.Autograd.TorchLean.RefTy
              (m := m) (α := α) Spec.Shape.scalar)))
          (fun args =>
            let (state, targets) := _root_.Runtime.Autograd.Torch.RefList.split
              (Ref := fun s => _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) s)
              (ss₁ := model.stateShapes) (ss₂ := [υ]) args
            let .cons target .nil := targets
            _root_.Runtime.Autograd.Torch.CurriedRef.curry
              (Ref := fun s =>
                _root_.Runtime.Autograd.Torch.NatTensorRef (m := m) (α := α) s)
              (ss := [σ])
              (β := m (_root_.Runtime.Autograd.TorchLean.RefTy
                (m := m) (α := α) Spec.Shape.scalar))
              (fun natInputs =>
                match natInputs with
                | .cons indices .nil => do
                    let withInput := _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
                      (Ref := fun s =>
                        _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) s)
                      (ss := model.stateShapes)
                      (β := _root_.Runtime.Autograd.Torch.CurriedRef
                        (fun s => _root_.Runtime.Autograd.Torch.NatTensorRef
                          (m := m) (α := α) s)
                        [σ]
                        (m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) τ)))
                      (model.forward mode (α := α)) state
                    let prediction ← withInput indices
                    _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
                      (Ref := fun s =>
                        _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) s)
                      (ss := [τ, υ]) (loss (α := α) (m := m))
                      (.cons prediction (.cons target .nil)))) }

/-- Pair an indexed-input model with a scalar loss in training mode. -/
def create {σ τ υ : Spec.Shape} (model : IndexedModel σ τ)
    (loss : ∀ {α : Type}, [_root_.Context α] → [DecidableEq Spec.Shape] →
      _root_.Runtime.Autograd.TorchLean.Program α [τ, υ] Spec.Shape.scalar) :
    _root_.Runtime.Autograd.TorchLean.Module.ObjectiveDef
      model.stateShapes [υ] [σ] :=
  createWithMode .train model loss

/-- Pair an indexed-input model with mean-squared error under an explicit layer mode. -/
def mseWithMode {σ τ : Spec.Shape}
    (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
    (model : IndexedModel σ τ)
    (reduction : _root_.TorchLean.Loss.Reduction := .mean) :
    _root_.Runtime.Autograd.TorchLean.Module.ObjectiveDef
      model.stateShapes [τ] [σ] :=
  createWithMode mode model (fun {α} _ _ => fun {m} _ _ => fun prediction target =>
    _root_.TorchLean.Loss.mse (m := m) (α := α) (s := τ) prediction target
      (reduction := reduction))

/-- Pair an indexed-input model with mean-squared error in training mode. -/
def mse {σ τ : Spec.Shape} (model : IndexedModel σ τ)
    (reduction : _root_.TorchLean.Loss.Reduction := .mean) :
    _root_.Runtime.Autograd.TorchLean.Module.ObjectiveDef
      model.stateShapes [τ] [σ] :=
  mseWithMode .train model reduction

end Objective

end IndexedModel

/--
A reusable trainable embedding table with `vocab` rows and vectors of length `embedDim`.

Unlike `IndexedModel`, this definition is independent of the eventual token-tensor shape. Calling
`table.model indices` specializes it to an input shape while retaining `Tensor Nat indices` at the
public boundary. Instantiating that model produces mutable weight storage; the definition itself
remains immutable so it can be lowered and used in proofs.
-/
structure Embedding (vocab embedDim : Nat) where
  /-- Initial table values, with one row per vocabulary item. -/
  initialWeight : Spec.Tensor Float (.dim vocab (.dim embedDim .scalar))
  /-- Storage-first initializer equivalent to `initialWeight`. -/
  runtimeInit : _root_.Runtime.Autograd.TorchLean.Module.RuntimeInit.Plan
    [.dim vocab (.dim embedDim .scalar)]
  /-- Whether reverse mode accumulates gradients for the table. -/
  requiresGrad : Bool := true

namespace Embedding

/-- Construction options for a freshly initialized embedding table. -/
structure Config where
  /-- Seed for deterministic table initialization. Seeded builders replace this field. -/
  seed : Nat := 0
  /--
  Initialization scheme for the table.

  The default agrees with `torch.nn.Embedding.reset_parameters`: independent samples from the
  standard normal distribution. Language-model constructors normally override this with their
  architecture-specific initialization, such as GPT-2's standard deviation `0.02`.
  -/
  weightInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .normal 0.0 1.0
  /-- Freeze the table by excluding it from reverse-mode parameter gradients. -/
  freeze : Bool := false

/--
Construct an embedding from an exact initial weight table.

This is the typed counterpart of passing `_weight` to `torch.nn.Embedding`. The supplied tensor
fixes both dimensions at compile time and is also recorded as an exact row-major runtime
initializer, so CPU and CUDA module construction start from the same payload.
-/
def ofWeight {vocab embedDim : Nat}
    (weight : Spec.Tensor Float (.dim vocab (.dim embedDim .scalar)))
    (freeze : Bool := false) : Embedding vocab embedDim :=
  { initialWeight := weight
    runtimeInit := .cons
      (.flat (FloatArray.mk (Spec.toList weight).toArray)) .nil
    requiresGrad := !freeze }

/-- Specialize an embedding table to a concrete index-tensor shape. -/
def model {vocab embedDim : Nat} (table : Embedding vocab embedDim) (inputShape : Spec.Shape) :
    IndexedModel inputShape (inputShape.appendDim embedDim) :=
  let weightShape : Spec.Shape := .dim vocab (.dim embedDim .scalar)
  { kind := s!"Embedding({vocab}, {embedDim})"
    stateShapes := [weightShape]
    initState := TensorPack! table.initialWeight
    runtimeInit := some table.runtimeInit
    requiresGrad := [table.requiresGrad]
    validateInput := fun tokenIds =>
      if Spec.Tensor.allSpec (fun tokenId => decide (tokenId < vocab)) tokenIds then
        pure ()
      else
        throw s!"embedding: token index outside vocabulary of size {vocab}"
    forward := fun _ {α} _ _ =>
      fun {m} _ _ => fun weight =>
        _root_.Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := fun s => _root_.Runtime.Autograd.Torch.NatTensorRef
            (m := m) (α := α) s)
          (ss := [inputShape])
          (fun natInputs =>
            match natInputs with
            | .cons tokenIds .nil =>
                _root_.Runtime.Autograd.TorchLean.F.embeddingNatOrZero
                  (m := m) (α := α) (vocab := vocab) (dim := embedDim) weight tokenIds) }

end Embedding

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
    (leading : Spec.Shape := Spec.Shape.scalar) :
    Sequential (leading.appendDim inDim) (leading.appendDim outDim) :=
  let WShape : Spec.Shape := .dim outDim (.dim inDim .scalar)
  let bShape : Spec.Shape := .dim outDim .scalar
  let weightInit := cfg.weightInit?.getD (.xavierUniform inDim outDim)
  let w0 : Spec.Tensor Float WShape := _root_.Runtime.Autograd.Torch.Init.tensor
    (s := WShape) (sch := weightInit) (seed := seedW)
  let b0 : Spec.Tensor Float bShape := _root_.Runtime.Autograd.Torch.Init.tensor
    (s := bShape) (sch := cfg.biasInit) (seed := seedB)
  let batch : Nat := Spec.Shape.size leading
  of
    { kind := s!"Linear({inDim}, {outDim})"
      stateShapes := [WShape, bShape]
      initState := TensorPack! w0, b0
      runtimeInit := some (.cons
        (_root_.Runtime.Autograd.TorchLean.Module.RuntimeInit.FloatInit.ofScheme weightInit seedW)
        (.cons
          (_root_.Runtime.Autograd.TorchLean.Module.RuntimeInit.FloatInit.ofScheme cfg.biasInit seedB)
          .nil))
      requiresGrad := [true, true]
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun w b x =>
            let sIn : Spec.Shape := leading.appendDim inDim
            let sOut : Spec.Shape := leading.appendDim outDim
            ((do
              let x2d ←
                _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
                  (s₁ := sIn)
                  (s₂ := .dim batch (.dim inDim .scalar))
                  x (by
                    -- size(sIn) = size(leading) * inDim = batch * inDim = size(Mat batch inDim)
                    simp [sIn, batch, Spec.Shape.size_appendDim, Spec.Shape.size])

              let wT ←
                _root_.Runtime.Autograd.Torch.transpose2d (m := m) (α := α)
                  (mDim := outDim) (nDim := inDim) w
              let y ← _root_.Runtime.Autograd.Torch.mm (m := m) (α := α)
                (mDim := batch) (nDim := inDim) (pDim := outDim) x2d wT
              let y2d ←
                _root_.Runtime.Autograd.TorchLean.F.addB (m := m) (α := α)
                  (t := .dim batch (.dim outDim .scalar)) y b
              _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
                (s₁ := .dim batch (.dim outDim .scalar))
                (s₂ := sOut)
                y2d (by
                  -- size(Mat batch outDim) = batch * outDim = size(leading) * outDim = size(sOut)
                  simp [sOut, batch, Spec.Shape.size_appendDim, Spec.Shape.size])
            ) : m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) sOut))
    }

/-- Linear layer with Xavier-uniform weights and zero bias. -/
def linear (inDim outDim : Nat) (seedW seedB : Nat := 0)
    (leading : Spec.Shape := Spec.Shape.scalar) :
    Sequential (leading.appendDim inDim) (leading.appendDim outDim) :=
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
      (.dim seqLen (.dim inputSize .scalar))
      (.dim seqLen (.dim hiddenSize .scalar)) :=
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
      (.dim seqLen (.dim inputSize .scalar))
      (.dim seqLen (.dim hiddenSize .scalar)) :=
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
      (.dim seqLen (.dim inputSize .scalar))
      (.dim seqLen (.dim hiddenSize .scalar)) :=
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
      (.dim seqLen (.dim inputSize .scalar))
      (.dim seqLen (.dim hiddenSize .scalar)) :=
  of (_root_.Runtime.Autograd.TorchLean.NN.lstm (seqLen := seqLen) (inputSize := inputSize) (hiddenSize := hiddenSize) seedW
    seedB)

/--
Linear projection for one-hot or soft token-distribution inputs.

Input shape: `[..., vocab]`
Output shape: `[..., embedDim]`

This is not an indexed embedding: it multiplies the final input axis by a trainable table. Use
`embedding` for integer token ids.
-/
def oneHotEmbedding (vocab embedDim : Nat) (cfg : Embedding.Config := {})
    (leading : Spec.Shape := Spec.Shape.scalar) :
    Sequential (leading.appendDim vocab) (leading.appendDim embedDim) :=
  let WShape : Spec.Shape := .dim vocab (.dim embedDim .scalar)
  let w0 : Spec.Tensor Float WShape :=
    _root_.Runtime.Autograd.Torch.Init.tensor
      (s := WShape) (sch := cfg.weightInit) (seed := cfg.seed)
  let batch : Nat := Spec.Shape.size leading
  of
    { kind := s!"OneHotEmbedding({vocab}, {embedDim})"
      stateShapes := [WShape]
      initState := TensorPack! w0
      runtimeInit := some (.cons
        (_root_.Runtime.Autograd.TorchLean.Module.RuntimeInit.FloatInit.ofScheme
          cfg.weightInit cfg.seed) .nil)
      requiresGrad := [!cfg.freeze]
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun w x =>
            let sIn : Spec.Shape := leading.appendDim vocab
            let sOut : Spec.Shape := leading.appendDim embedDim
            ((do
              let x2d ←
                _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
                  (s₁ := sIn)
                  (s₂ := .dim batch (.dim vocab .scalar))
                  x (by
                    -- size(sIn) = size(leading) * vocab = batch * vocab
                    simp [sIn, batch, Spec.Shape.size_appendDim, Spec.Shape.size])
              let y ←
                _root_.Runtime.Autograd.Torch.mm (m := m) (α := α)
                  (mDim := batch) (nDim := vocab) (pDim := embedDim)
                  x2d w
              _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
                (s₁ := .dim batch (.dim embedDim .scalar))
                (s₂ := sOut)
                y (by
                  -- size(Mat batch embedDim) = batch * embedDim = size(leading) * embedDim = size(sOut)
                  simp [sOut, batch, Spec.Shape.size_appendDim, Spec.Shape.size])
            ) : m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) sOut))
    }

/--
Look up rows of a trainable embedding table using a tensor of natural-number indices.

For input shape `σ`, `embedding vocab embedDim σ` returns shape `σ.appendDim embedDim`. This matches
`torch.nn.Embedding`: the input stores integer indices, the output appends the embedding dimension,
and gradients flow to selected rows of the table rather than to the indices.
-/
def embedding (vocab embedDim : Nat) (cfg : Embedding.Config := {}) :
    Embedding vocab embedDim :=
  let weightShape : Spec.Shape := .dim vocab (.dim embedDim .scalar)
  let initialWeight : Spec.Tensor Float weightShape :=
    _root_.Runtime.Autograd.Torch.Init.tensor
      (s := weightShape) (sch := cfg.weightInit) (seed := cfg.seed)
  { initialWeight
    runtimeInit := .cons
      (_root_.Runtime.Autograd.TorchLean.Module.RuntimeInit.FloatInit.ofScheme
        cfg.weightInit cfg.seed) .nil
    requiresGrad := !cfg.freeze }

/--
Learned positional embedding configuration.

This is a trainable parameter tensor of shape `(seqLen × embedDim)` that is broadcast across any
leading dimensions and added to the input.
-/
structure LearnedPositionalEmbedding where
  /-- Seed for deterministic initialization. -/
  seedPos : Nat := 0
  /-- Initialization scheme for the positional embedding table. -/
  posInit : _root_.Runtime.Autograd.Torch.Init.Scheme := .uniform (-0.02) 0.02

/--
Add learned positional embeddings to the `(seqLen × embedDim)` suffix of a tensor.

PyTorch analogue: `x + pos[:seqLen]` where `pos` is a parameter table.
-/
def learnedPositionalEmbedding (leading : Spec.Shape := .scalar) {seqLen embedDim : Nat}
    (cfg : LearnedPositionalEmbedding := {}) :
    Sequential
      (leading.concat (.dim seqLen (.dim embedDim .scalar)))
      (leading.concat (.dim seqLen (.dim embedDim .scalar))) :=
  let posShape : Spec.Shape := .dim seqLen (.dim embedDim .scalar)
  let xShape : Spec.Shape := leading.concat posShape
  let pos0 : Spec.Tensor Float posShape :=
    _root_.Runtime.Autograd.Torch.Init.tensor (s := posShape) (sch := cfg.posInit) (seed := cfg.seedPos)
  letI : Spec.Shape.BroadcastTo posShape xShape :=
    ⟨Spec.Shape.CanBroadcastTo.prependTarget leading posShape⟩
  letI : Spec.Shape.BroadcastTo xShape xShape :=
    ⟨Spec.Shape.CanBroadcastTo.refl xShape⟩
  of
    { kind := "LearnedPositionalEmbedding"
      stateShapes := [posShape]
      initState := TensorPack! pos0
      runtimeInit := some (.cons
        (_root_.Runtime.Autograd.TorchLean.Module.RuntimeInit.FloatInit.ofScheme
          cfg.posInit cfg.seedPos) .nil)
      requiresGrad := [true]
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun pos x =>
            -- Broadcast the positional table across every leading axis.
            (_root_.Runtime.Autograd.TorchLean.F.addB (m := m) (α := α)
              (s₁ := posShape) (s₂ := xShape) (t := xShape) pos x)
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
Add sinusoidal positional encodings to the `(seqLen × embedDim)` suffix of a tensor.

Implementation:
- precompute `PE : (seqLen × embedDim)` at initialization time (stored as a non-trainable buffer),
- broadcast it across every leading axis and add it to the input.
-/
def sinusoidalPositionalEncoding (leading : Spec.Shape := .scalar) {seqLen embedDim : Nat}
    (cfg : SinusoidalPositionalEncoding := {}) :
    Sequential
      (leading.concat (.dim seqLen (.dim embedDim .scalar)))
      (leading.concat (.dim seqLen (.dim embedDim .scalar))) :=
  let peShape : Spec.Shape := .dim seqLen (.dim embedDim .scalar)
  let xShape : Spec.Shape := leading.concat peShape
  let pe0 : Spec.Tensor Float peShape :=
    Spec.sinusoidalPositionalEncodingSpec (α := Float) seqLen embedDim cfg.startPos
  letI : Spec.Shape.BroadcastTo peShape xShape :=
    ⟨Spec.Shape.CanBroadcastTo.prependTarget leading peShape⟩
  letI : Spec.Shape.BroadcastTo xShape xShape :=
    ⟨Spec.Shape.CanBroadcastTo.refl xShape⟩
  of
    { kind := "SinusoidalPositionalEncoding"
      stateShapes := [peShape]
      initState := TensorPack! pe0
      requiresGrad := [false]
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun pe x =>
            -- Broadcast `PE : (seqLen × embedDim)` across every leading axis.
            (_root_.Runtime.Autograd.TorchLean.F.addB (m := m) (α := α)
              (s₁ := peShape) (s₂ := xShape) (t := xShape) pe x)
    }

/--
Rotary positional embedding (RoPE) configuration.

`startPos` is an absolute-position offset for KV-cache decoding.
-/
structure RotaryEmbeddingConfig where
  /-- Absolute position offset for the first row of RoPE angles. -/
  startPos : Nat := 0

/--
Apply RoPE to the `(seqLen × headDim)` suffix of a tensor.

This matches the standard identity:

$$
\operatorname{rope}(x)
  = x \odot \cos + \operatorname{rotatePairs}(x) \odot \sin
$$

where `cos` and `sin` depend only on `(pos, dim)` and broadcast across every leading axis.

Notes:
- This layer is *differentiable* (gradients flow through the rotation), but it has no trainable
  parameters; the precomputed `cos`/`sin` tables are stored as non-trainable buffers.
- The pure spec version is in `NN.Spec.Layers.PositionalEncoding` (`Spec.rope_apply_heads_spec`).
-/
def rope (leading : Spec.Shape := .scalar) {seqLen headDim : Nat}
    (cfg : RotaryEmbeddingConfig := {}) :
    Sequential
      (leading.concat (.dim seqLen (.dim headDim .scalar)))
      (leading.concat (.dim seqLen (.dim headDim .scalar))) :=
  let xShape : Spec.Shape := leading.concat (.dim seqLen (.dim headDim .scalar))
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

  letI : Spec.Shape.BroadcastTo csShape xShape :=
    ⟨Spec.Shape.CanBroadcastTo.prependTarget leading csShape⟩
  letI : Spec.Shape.BroadcastTo xShape xShape :=
    ⟨Spec.Shape.CanBroadcastTo.refl xShape⟩

  of
    { kind := "RoPE"
      stateShapes := [csShape, csShape]
      initState := TensorPack! cos0, sin0
      requiresGrad := [false, false]
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun cos sin x =>
            ((do
            -- Rotate last-dim pairs by a fixed 2D permutation/sign pattern.
            let rowsFold : Nat := Spec.Shape.size leading * seqLen
            let flatShape : Spec.Shape := .dim rowsFold (.dim headDim .scalar)

            let x2d ←
              _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
                (s₁ := xShape) (s₂ := flatShape)
                x (by
                  simp [xShape, flatShape, rowsFold, Spec.Shape.size_concat,
                    Spec.Shape.size, Nat.mul_assoc])

            let xT ←
              _root_.Runtime.Autograd.Torch.transpose2d (m := m) (α := α)
                (mDim := rowsFold) (nDim := headDim) x2d

            let xPerm ←
              _root_.Runtime.Autograd.Torch.gatherRowsNatOrZero (m := m) (α := α)
                (rows := headDim) (cols := rowsFold) (k := headDim)
                xT (_root_.Runtime.Autograd.Torch.natTensorConst (m := m) (α := α) permIdx)

            let xBack ←
              _root_.Runtime.Autograd.Torch.transpose2d (m := m) (α := α)
                (mDim := headDim) (nDim := rowsFold) xPerm

            -- Sign pattern for `rotatePairs`: even outputs get a negation (except the final unpaired entry).
            let signT : Spec.Tensor α (.dim headDim .scalar) :=
              Spec.Tensor.dim (fun (j : Fin headDim) =>
                let idx := j.val
                let v : α :=
                  if idx % 2 = 0 ∧ idx + 1 < headDim then (-1 : α) else (1 : α)
                Spec.Tensor.scalar v)
            let sign ← _root_.Runtime.Autograd.Torch.const (m := m) (α := α) (s := .dim headDim .scalar) signT

            let xRot2d ←
              _root_.Runtime.Autograd.TorchLean.F.mulB (m := m) (α := α)
                (s₁ := flatShape) (s₂ := .dim headDim .scalar) (t := flatShape)
                xBack sign

            let xRot ←
              _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
                (s₁ := flatShape) (s₂ := xShape)
                xRot2d (by
                  simp [xShape, flatShape, rowsFold, Spec.Shape.size_concat,
                    Spec.Shape.size, Nat.mul_assoc])

            -- Apply the RoPE formula with broadcasting of `cos/sin : (seqLen × headDim)`.
            let xCos ←
              _root_.Runtime.Autograd.TorchLean.F.mulB (m := m) (α := α)
                (s₁ := xShape) (s₂ := csShape) (t := xShape)
                x cos
            let rotSin ←
              _root_.Runtime.Autograd.TorchLean.F.mulB (m := m) (α := α)
                (s₁ := xShape) (s₂ := csShape) (t := xShape)
                xRot sin
            _root_.Runtime.Autograd.Torch.add (m := m) (α := α) (s := xShape) xCos rotSin
            ) : m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) xShape))
    }

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
def flatten {s : Spec.Shape} : Sequential s (.dim (Spec.Shape.size s) .scalar) :=
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
      requiresGrad := []
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun x =>
            _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
              (s₁ := source) (s₂ := target) x sameSize }

/--
Flatten each tensor after an arbitrary collection of leading dimensions.

For `leading = shape![batch]`, this is the typed counterpart of
`torch.flatten(x, start_dim=1)`. Multiple leading dimensions are preserved without introducing a
separate batched tensor type.
-/
def flattenLeading (leading : Spec.Shape := .scalar) {s : Spec.Shape} :
    Sequential (leading.concat s) (leading.appendDim (Spec.Shape.size s)) :=
  of
    { kind := "FlattenLeading"
      stateShapes := []
      initState := .nil
      requiresGrad := []
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun x =>
            _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
              (s₁ := leading.concat s)
              (s₂ := leading.appendDim (Spec.Shape.size s))
              x (by
                simp [Spec.Shape.size_concat, Spec.Shape.size_appendDim])
    }

/--
Dropout layer (active in train mode, identity in eval mode).

PyTorch analogue: `torch.nn.Dropout`.
-/
def dropout {s : Spec.Shape} (p : Float) (seed : Nat := 0) : Sequential s s :=
  of <| _root_.Runtime.Autograd.TorchLean.NN.dropout (s := s) p seed
