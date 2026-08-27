/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Neural.Builders

@[expose] public section

namespace TorchLean

/-!
# Indexed Models and Embeddings

This module defines models with non-differentiable tensor inputs, their scalar objectives, and
embedding-table builders. Import `NN.API.Neural.Indexed` when constructing or executing an indexed
model; ordinary sequential builders remain in `NN.API.Neural.Builders`.
-/

namespace nn

/-- A shape-typed model with one non-differentiable tensor input. -/
structure IndexedModel (σ τ : Spec.Shape) (β : Type) where
  /-- Model label used in summaries and diagnostics. -/
  kind : String := "IndexedModel"
  /-- Shapes of trainable parameters and persistent buffers. -/
  stateShapes : List Spec.Shape
  /-- Semantic initial values for the complete model state. -/
  initState : _root_.TorchLean.TensorPack Float stateShapes
  /-- Optional storage-first initialization plan for executable `Float` backends. -/
  runtimeInit : Option
    (_root_.Runtime.Autograd.TorchLean.Module.RuntimeInit.Plan stateShapes) := none
  /-- Gradient flags aligned with `stateShapes`; persistent buffers carry `false`. -/
  requiresGrad : Array Bool := Array.replicate stateShapes.length true
  /-- Validate concrete data before it reaches the backend program. -/
  validateInput : Tensor β σ → Except String Unit := fun _ => pure ()
  /-- Execution-polymorphic forward computation over one discrete input tensor. -/
  forward : ∀ (_mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
      {α : Type}, [_root_.Context α] → [DecidableEq Spec.Shape] →
      _root_.Runtime.Autograd.TorchLean.ProgramWithDataInputs α β stateShapes [σ] τ

namespace IndexedModel

/-- Append an ordinary sequential model after an indexed-input model. -/
def andThen {β : Type} {σ τ υ : Spec.Shape} (first : IndexedModel σ τ β)
    (rest : Sequential τ υ) : IndexedModel σ υ β where
  kind := s!"{first.kind} → Sequential"
  stateShapes := first.stateShapes ++ _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes rest
  initState := _root_.TorchLean.TensorPack.append first.initState
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
        (Ref := fun s => _root_.TorchLean.Runtime.ValueRef (m := m) (α := α) s)
        (ss := first.stateShapes ++ _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes rest)
        (β := _root_.Runtime.Autograd.Torch.CurriedRef
          (fun s => _root_.Runtime.Autograd.Torch.DataRef (m := m) (α := α) β s)
          [σ] (m (_root_.TorchLean.Runtime.ValueRef (m := m) (α := α) υ)))
        (fun state =>
          let (firstState, restState) := _root_.Runtime.Autograd.Torch.RefList.split
            (Ref := fun s => _root_.TorchLean.Runtime.ValueRef (m := m) (α := α) s)
            (ss₁ := first.stateShapes)
            (ss₂ := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes rest) state
          _root_.Runtime.Autograd.Torch.CurriedRef.curry
            (Ref := fun s => _root_.Runtime.Autograd.Torch.DataRef
              (m := m) (α := α) β s)
            (ss := [σ])
            (fun dataInputs =>
              match dataInputs with
              | .cons indices .nil => do
                  let embedded ←
                    (_root_.Runtime.Autograd.Torch.CurriedRef.uncurry
                      (Ref := fun s =>
                        _root_.TorchLean.Runtime.ValueRef (m := m) (α := α) s)
                      (ss := first.stateShapes)
                      (β := _root_.Runtime.Autograd.Torch.CurriedRef
                        (fun s => _root_.Runtime.Autograd.Torch.DataRef
                          (m := m) (α := α) β s)
                        [σ]
                        (m (_root_.TorchLean.Runtime.ValueRef (m := m) (α := α) τ)))
                      (first.forward mode (α := α)) firstState) indices
                  _root_.Runtime.Autograd.TorchLean.NN.Seq.forwardState
                    (model := rest) (α := α) (m := m) mode restState embedded))

/-! ### Scalar objectives -/

namespace Objective

/--
Pair an indexed-input model with a scalar loss under an explicit train/eval mode.

The resulting training module accepts one ordinary target tensor followed by the model's
non-differentiable input tensor. Keeping those packs separate ensures that indices cannot receive
gradients or be reinterpreted through the model's floating-point scalar type.

The target shape is independent of the model output shape, so this constructor also supports losses
whose labels use a different representation from the prediction.
-/
def createWithMode {β : Type} {σ τ υ : Spec.Shape}
    (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
    (model : IndexedModel σ τ β)
    (loss : ∀ {α : Type}, [_root_.Context α] → [DecidableEq Spec.Shape] →
      _root_.Runtime.Autograd.TorchLean.Program α [τ, υ] Spec.Shape.scalar) :
    _root_.Runtime.Autograd.TorchLean.Module.ObjectiveDef β
      model.stateShapes [υ] [σ] :=
  { initState := model.initState
    runtimeInit := model.runtimeInit
    requiresGrad := model.requiresGrad
    validateDataInputs := fun
      | .cons input .nil => model.validateInput input
    loss := fun {α} => by
      intro _ _
      exact fun {m} _ _ =>
        _root_.Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := fun s => _root_.TorchLean.Runtime.ValueRef (m := m) (α := α) s)
          (ss := model.stateShapes ++ [υ])
          (β := _root_.Runtime.Autograd.Torch.CurriedRef
            (fun s => _root_.Runtime.Autograd.Torch.DataRef (m := m) (α := α) β s)
            [σ]
            (m (_root_.TorchLean.Runtime.ValueRef
              (m := m) (α := α) Spec.Shape.scalar)))
          (fun args =>
            let (state, targets) := _root_.Runtime.Autograd.Torch.RefList.split
              (Ref := fun s => _root_.TorchLean.Runtime.ValueRef (m := m) (α := α) s)
              (ss₁ := model.stateShapes) (ss₂ := [υ]) args
            let .cons target .nil := targets
            _root_.Runtime.Autograd.Torch.CurriedRef.curry
              (Ref := fun s =>
                _root_.Runtime.Autograd.Torch.DataRef (m := m) (α := α) β s)
              (ss := [σ])
              (β := m (_root_.TorchLean.Runtime.ValueRef
                (m := m) (α := α) Spec.Shape.scalar))
              (fun dataInputs =>
                match dataInputs with
                | .cons indices .nil => do
                    let withInput := _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
                      (Ref := fun s =>
                        _root_.TorchLean.Runtime.ValueRef (m := m) (α := α) s)
                      (ss := model.stateShapes)
                      (β := _root_.Runtime.Autograd.Torch.CurriedRef
                        (fun s => _root_.Runtime.Autograd.Torch.DataRef
                          (m := m) (α := α) β s)
                        [σ]
                        (m (_root_.TorchLean.Runtime.ValueRef (m := m) (α := α) τ)))
                      (model.forward mode (α := α)) state
                    let prediction ← withInput indices
                    _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
                      (Ref := fun s =>
                        _root_.TorchLean.Runtime.ValueRef (m := m) (α := α) s)
                      (ss := [τ, υ]) (loss (α := α) (m := m))
                      (.cons prediction (.cons target .nil)))) }

/-- Pair an indexed-input model with a scalar loss in training mode. -/
def create {β : Type} {σ τ υ : Spec.Shape} (model : IndexedModel σ τ β)
    (loss : ∀ {α : Type}, [_root_.Context α] → [DecidableEq Spec.Shape] →
      _root_.Runtime.Autograd.TorchLean.Program α [τ, υ] Spec.Shape.scalar) :
    _root_.Runtime.Autograd.TorchLean.Module.ObjectiveDef β
      model.stateShapes [υ] [σ] :=
  createWithMode .train model loss

/-- Pair an indexed-input model with mean-squared error under an explicit layer mode. -/
def mseWithMode {β : Type} {σ τ : Spec.Shape}
    (mode : _root_.Runtime.Autograd.TorchLean.NN.Mode)
    (model : IndexedModel σ τ β)
    (reduction : _root_.TorchLean.Loss.Reduction := .mean) :
    _root_.Runtime.Autograd.TorchLean.Module.ObjectiveDef β
      model.stateShapes [τ] [σ] :=
  createWithMode mode model (fun {α} _ _ => fun {m} _ _ => fun prediction target =>
    _root_.TorchLean.Loss.mse (m := m) (α := α) (s := τ) prediction target
      (reduction := reduction))

/-- Pair an indexed-input model with mean-squared error in training mode. -/
def mse {β : Type} {σ τ : Spec.Shape} (model : IndexedModel σ τ β)
    (reduction : _root_.TorchLean.Loss.Reduction := .mean) :
    _root_.Runtime.Autograd.TorchLean.Module.ObjectiveDef β
      model.stateShapes [τ] [σ] :=
  mseWithMode .train model reduction

end Objective

end IndexedModel

/--
A reusable trainable embedding table with `vocab` rows and vectors of length `embedDim`.

Unlike `IndexedModel`, this definition is independent of the eventual token-tensor shape. Calling
`table.model indices` specializes it to an input shape with bounded `Fin vocab` indices.
Instantiating that model produces mutable weight storage; the definition itself remains immutable
so it can be lowered and used in proofs.
-/
structure Embedding (vocab embedDim : Nat) where
  /-- Initial table values, with one row per vocabulary item. -/
  initialWeight : Tensor Float [vocab, embedDim]
  /-- Storage-first initializer equivalent to `initialWeight`. -/
  runtimeInit : _root_.Runtime.Autograd.TorchLean.Module.RuntimeInit.Plan
    [[vocab, embedDim]]
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
    (weight : Tensor Float [vocab, embedDim])
    (freeze : Bool := false) : Embedding vocab embedDim :=
  { initialWeight := weight
    runtimeInit := .cons
      (.flat (FloatArray.mk weight.toArray)) .nil
    requiresGrad := !freeze }

/-- Specialize an embedding table to a concrete index-tensor shape. -/
def model {vocab embedDim : Nat} [NeZero vocab]
    (table : Embedding vocab embedDim) (inputShape : List Nat) :
    IndexedModel inputShape ((Spec.Shape.ofList inputShape).appendDim embedDim) (Fin vocab) :=
  let weightShape : Spec.Shape := [vocab, embedDim]
  { kind := s!"Embedding({vocab}, {embedDim})"
    stateShapes := [weightShape]
    initState := _root_.TorchLean.TensorPack! table.initialWeight
    runtimeInit := some table.runtimeInit
    requiresGrad := #[table.requiresGrad]
    forward := fun _ {α} _ _ =>
      fun {m} _ _ => fun weight =>
        _root_.Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := fun s => _root_.Runtime.Autograd.Torch.DataRef
            (m := m) (α := α) (Fin vocab) s)
          (ss := [inputShape])
          (fun dataInputs =>
            match dataInputs with
            | .cons tokenIds .nil =>
                _root_.Runtime.Autograd.TorchLean.F.embedding
                  (m := m) (α := α) (vocab := vocab) (dim := embedDim) weight tokenIds) }

end Embedding

namespace Internal

/--
Linear projection for one-hot or soft token-distribution inputs.

Input shape: `[..., vocab]`
Output shape: `[..., embedDim]`

This is not an indexed embedding: it multiplies the final input axis by a trainable table. Use
`embedding` for bounded token ids.
-/
def oneHotEmbedding (vocab embedDim : Nat) (cfg : Embedding.Config := {})
    (leading : List Nat := []) :
    Sequential (leading ++ [vocab]) (leading ++ [embedDim]) :=
  let leadingShape := Spec.Shape.ofList leading
  let WShape : Spec.Shape := [vocab, embedDim]
  let w0 : Tensor Float WShape :=
    _root_.Runtime.Autograd.Torch.Init.tensor
      (s := WShape) (sch := cfg.weightInit) (seed := cfg.seed)
  let batch : Nat := Spec.Shape.size leadingShape
  of
    { kind := s!"OneHotEmbedding({vocab}, {embedDim})"
      stateShapes := [WShape]
      initState := _root_.TorchLean.TensorPack! w0
      runtimeInit := some (.cons
        (_root_.Runtime.Autograd.TorchLean.Module.RuntimeInit.FloatInit.ofScheme
          cfg.weightInit cfg.seed) .nil)
      requiresGrad := #[!cfg.freeze]
      forward := fun _ {α} _ _ =>
        fun {m} _ _ =>
          fun w x =>
            let sIn : Spec.Shape := leading ++ [vocab]
            let sOut : Spec.Shape := leading ++ [embedDim]
            ((do
              let x2d ←
                _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
                  (s₁ := sIn)
                  (s₂ := [batch, vocab])
                  x (by
                    -- size(sIn) = size(leading) * vocab = batch * vocab
                    simp [sIn, batch, leadingShape, Spec.Shape.size_concat,
                      Spec.Shape.size_ofList, Spec.Shape.size])
              let y ←
                _root_.Runtime.Autograd.Torch.matmul (m := m) (α := α)
                  (batchA := .scalar) (batchB := .scalar) (batch := .scalar)
                  (mDim := batch) (nDim := vocab) (pDim := embedDim)
                  x2d w
              _root_.Runtime.Autograd.Torch.reshape (m := m) (α := α)
                (s₁ := [batch, embedDim])
                (s₂ := sOut)
                y (by
                  -- size(Mat batch embedDim) = batch * embedDim = size(leading) * embedDim = size(sOut)
                  simp [sOut, batch, leadingShape, Spec.Shape.size_concat,
                    Spec.Shape.size_ofList, Spec.Shape.size])
            ) : m (_root_.TorchLean.Runtime.ValueRef (m := m) (α := α) sOut))
    }

/-- Build a trainable table for bounded token ids. -/
def embedding (vocab embedDim : Nat) (cfg : Embedding.Config := {}) :
    Embedding vocab embedDim :=
  let weightShape : Spec.Shape := [vocab, embedDim]
  let initialWeight : Tensor Float weightShape :=
    _root_.Runtime.Autograd.Torch.Init.tensor
      (s := weightShape) (sch := cfg.weightInit) (seed := cfg.seed)
  { initialWeight
    runtimeInit := .cons
      (_root_.Runtime.Autograd.TorchLean.Module.RuntimeInit.FloatInit.ofScheme
        cfg.weightInit cfg.seed) .nil
    requiresGrad := !cfg.freeze }

end Internal
end nn
end TorchLean
