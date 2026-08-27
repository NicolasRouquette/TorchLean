/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.Autodiff
public import NN.Runtime.Autograd.TorchLean.Loss
public import NN.Runtime.Autograd.TorchLean.Module
public import NN.Runtime.Autograd.TorchLean.Norm

import Mathlib.Algebra.Order.Algebra

/-!
# NN

`TorchLean.NN`: a compact `torch.nn`-style builder layer.

This module defines a small `torch.nn`-style builder layer for constructing shape-typed models.
It packages model-state shapes and initial values together with an execution-polymorphic forward program, so
example code does not have to spell `paramShapes := [...]` / `inputShapes := [...]` everywhere.

## Main definitions

- `Layer σ τ` packages a shape-typed layer with explicit state (parameters and buffers) and
  a polymorphic `forward` program.
- `Seq σ τ` composes layers sequentially (PyTorch analogy: `torch.nn.Sequential`), written `f >>>
  g`.
- `Seq.Objective` bundles a `Seq` model together with a scalar loss, producing a
  `TorchLean.Module.ObjectiveDef` that the runtime training code can execute.

## PyTorch analogies

- `Layer` is like a small `nn.Module` definition, except parameters are an explicit list instead
  of fields, and the forward pass is a typed TorchLean program.
- `Mode` is like `module.train()` vs `module.eval()` (dropout and batchnorm-like layers branch on
  it).
- The `updateBuffers` mechanism is like updating non-gradient buffers (e.g. BatchNorm running
  stats).

The surface here is narrow by design: it supports TorchLean's executable model constructors and
training helpers without trying to mirror the full `torch.nn` API.

## References

- PyTorch `torch.nn`: https://pytorch.org/docs/stable/nn.html
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor
open Proofs.Autograd.Algebra

namespace NN

/-!
### Mode

TorchLean keeps "train vs eval" behavior explicit. This affects layers like dropout and
batch-normalization that behave differently during training vs inference.
-/

/--
Execution mode for layers that branch between training-time and inference-time behavior.

PyTorch analogy: `model.train()` / `model.eval()` (affects dropout, batchnorm, etc.).
-/
inductive Mode where
  | train
  | eval
deriving Repr, DecidableEq

/-! ## Layer definitions -/

/--
A shape-typed layer definition with explicit model state and an execution-polymorphic forward
program.

`Layer σ τ` is the core building block used by `Seq` (sequential composition). It stores:
- the shapes of parameters and persistent buffers,
- initial values for that state (as `Float` tensors, for reproducible
  initialization),
- per-parameter `requiresGrad` flags, and
- a `forward` program that is polymorphic over the backend monad and scalar type.

PyTorch analogy: a small `nn.Module`, where:
- `stateShapes`/`initState` contain parameters and persistent buffers,
- `forward` corresponds to `Module.forward`,
- `updateBuffers` corresponds to updating things like `running_mean`/`running_var` in BatchNorm.
-/
structure Layer (σ τ : Shape) where
  /-- Layer label used by public model summaries. -/
  kind : String := "Layer"
  /-- Shapes of parameters and persistent buffers, in the order expected by `forward`. -/
  stateShapes : List Shape
  /-- Initial model state, stored as `Float` tensors for convenient initialization. -/
  initState : TorchLean.TensorPack Float stateShapes
  /--
  Optional storage-first initialization plan for executable `Float` backends.

  This does not replace `initState`: the tensor-valued initializers remain available to the
  specification and proof layers. The plan lets a runtime create equivalent parameter storage
  without first enumerating those tensors on the host.
  -/
  runtimeInit : Option (TorchLean.Module.RuntimeInit.Plan stateShapes) :=
    if h : stateShapes = [] then some (h ▸ .nil) else none
  /--
  Gradient flags for model state (defaults to all `true`). Buffers use `false`.

  PyTorch analogy: `tensor.requires_grad_(...)` on parameters/buffers.
  -/
  requiresGrad : Array Bool := Array.replicate stateShapes.length true
  /--
  Optional buffer update function (used for running-statistics style layers).

  This is called during a forward pass (typically in `Mode.train`) to produce updated
    parameter/buffer
  state values. A canonical example is BatchNorm updating its `running_mean` / `running_var`
  buffers.
  -/
  updateBuffers :
    Option (
      Mode → ∀ {α : Type}, [Context α] → [DecidableEq Shape] →
        TorchLean.TensorPack α stateShapes → Tensor α σ → IO (TorchLean.TensorPack α stateShapes)
    ) := none
  /--
  Forward pass as a typed TorchLean program.

  The program expects `(stateShapes ++ [σ])` inputs (model state, then the layer input) and
  produces an output of shape `τ`.
  -/
  forward :
    Mode → ∀ {α : Type}, [Context α] → [DecidableEq Shape] →
      TorchLean.Program α (stateShapes ++ [σ]) τ

/--
Update running statistics of any shape using momentum.

This implements an exponential moving average:

`next = (1 - momentum) * running + momentum * batch`.

PyTorch analogy: the update performed for `running_mean` / `running_var` in BatchNorm.
-/
def updateRunning {α : Type} [Context α] {s : Shape}
    (running batch : Tensor α s) (momentum : Tensor α .scalar) : Tensor α s :=
  match momentum with
  | .scalar mom =>
      addSpec (scaleSpec running ((1 : α) - mom)) (scaleSpec batch mom)

/--
Convert the biased variance used by BatchNorm's training forward pass into the unbiased estimate
stored in its running buffer. For a singleton sample set there is no unbiased estimate; TorchLean
keeps the finite biased value rather than dividing by zero.
-/
def unbiasedRunningVariance {α : Type} [Context α] {s : Shape}
    (biased : Tensor α s) (sampleCount : Nat) : Tensor α s :=
  if sampleCount > 1 then
    scaleSpec biased ((sampleCount : α) / (sampleCount - 1 : Nat))
  else
    biased

/--
Compute per-channel mean and biased variance for a tensor with an arbitrary spatial suffix and no
batch dimension.

This reduces over every spatial axis and returns `(mean, var)` vectors of length `channels`.

The biased variance is the value used to normalize the current input. Convert it with
`unbiasedRunningVariance` before updating a PyTorch-compatible running-variance buffer.
-/
def channelStats {α : Type} [Context α]
    {channels : Nat} {spatial : Shape}
    (x : Tensor α (.dim channels spatial)) :
    Tensor α [channels] × Tensor α [channels] :=
  let spatialSize := Shape.size spatial
  let flatShape : Shape := .dim channels (.dim spatialSize .scalar)
  let xFlat : Tensor α flatShape := reshapeSpec x (by
    simp only [flatShape, spatialSize, Shape.size, Nat.mul_one])
  let means : Tensor α [channels] :=
    Tensor.dim (fun c =>
      let channelData := get xFlat c
      let channelSum :=
        (List.finRange spatialSize).foldl (fun acc i =>
          if hI : i < spatialSize then
            addSpec acc (get channelData ⟨i, hI⟩)
          else acc
        ) (Tensor.scalar 0)
      divSpec channelSum (Tensor.scalar (spatialSize : α)))
  let vars : Tensor α [channels] :=
    Tensor.dim (fun c =>
      let channelData := get xFlat c
      let mean := get means c
      let varianceSum :=
        (List.finRange spatialSize).foldl (fun acc i =>
          if hI : i < spatialSize then
            let d := subSpec (get channelData ⟨i, hI⟩) mean
            addSpec acc (mulSpec d d)
          else acc
        ) (Tensor.scalar 0)
      divSpec varianceSum (Tensor.scalar (spatialSize : α)))
  (means, vars)

/--
Compute per-channel mean and biased variance for a batched tensor.

The first two axes are batch and channel; every remaining axis is reduced. The result is a pair of
vectors indexed by channel. A running-variance update uses
`unbiasedRunningVariance vars (batch * spatial.size)` instead.
-/
def batchChannelStats {α : Type} [Context α]
    {batch channels : Nat} {spatial : Shape}
    (x : Tensor α (.dim batch (.dim channels spatial))) :
    Tensor α [channels] × Tensor α [channels] :=
  let spatialSize := Shape.size spatial
  let flatShape : Shape := .dim batch (.dim channels (.dim spatialSize .scalar))
  let xFlat : Tensor α flatShape := reshapeSpec x (by
    simp only [flatShape, spatialSize, Shape.size, Nat.mul_one])
  let sampleCount := batch * spatialSize
  let means : Tensor α [channels] :=
    Tensor.dim (fun ch =>
      let total :=
        (List.finRange batch).foldl (fun accBatch ni =>
          (List.finRange spatialSize).foldl (fun accSpatial i =>
            if hN : ni < batch then
              if hI : i < spatialSize then
                let channel := get (get xFlat ⟨ni, hN⟩) ch
                addSpec accSpatial (get channel ⟨i, hI⟩)
              else accSpatial
            else accSpatial
          ) accBatch
        ) (Tensor.scalar 0)
      divSpec total (Tensor.scalar (sampleCount : α)))
  let vars : Tensor α [channels] :=
    Tensor.dim (fun ch =>
      let mean := get means ch
      let total :=
        (List.finRange batch).foldl (fun accBatch ni =>
          (List.finRange spatialSize).foldl (fun accSpatial i =>
            if hN : ni < batch then
              if hI : i < spatialSize then
                let channel := get (get xFlat ⟨ni, hN⟩) ch
                let d := subSpec (get channel ⟨i, hI⟩) mean
                addSpec accSpatial (mulSpec d d)
              else accSpatial
            else accSpatial
          ) accBatch
        ) (Tensor.scalar 0)
      divSpec total (Tensor.scalar (sampleCount : α)))
  (means, vars)

namespace Layer

/--
Construct a layer from an uncurried reference-level forward function.

This is the general constructor for non-sequential modules: branches, parameter sharing, and
other graph structures can inspect the parameter list directly without repeating the
`CurriedRef` packing boilerplate. Reusing the same reference in `run` reuses one autograd leaf;
there is no parameter copy or alias table hidden by this constructor.
-/
def ofRef {σ τ : Shape} {ps : List Shape}
    (kind : String)
    (initState : TorchLean.TensorPack Float ps)
    (run : Mode → ∀ {α : Type}, [Context α] → [DecidableEq Shape] →
      ∀ {m : Type → Type}, [Monad m] → [Torch.Ops (m := m) (α := α)] →
        Torch.RefList (RefTy (m := m) (α := α)) ps →
        RefTy (m := m) (α := α) σ → m (RefTy (m := m) (α := α) τ))
    (runtimeInit : Option (TorchLean.Module.RuntimeInit.Plan ps) := none)
    (requiresGrad : Array Bool := Array.replicate ps.length true)
    (updateBuffers : Option (
      Mode → ∀ {α : Type}, [Context α] → [DecidableEq Shape] →
        TorchLean.TensorPack α ps → Tensor α σ → IO (TorchLean.TensorPack α ps)) := none) :
    Layer σ τ :=
  { kind
    stateShapes := ps
    initState
    runtimeInit
    requiresGrad
    updateBuffers
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        Torch.CurriedRef.curry
          (Ref := RefTy (m := m) (α := α))
          (ss := ps ++ [σ])
          (β := m (RefTy (m := m) (α := α) τ))
          (fun args =>
            let (params, x) :=
              Torch.RefList.splitLast (Ref := RefTy (m := m) (α := α))
                (ss := ps) (τ := σ) args
            run mode params x) }

/--
Run a `Layer` forward given parameter refs and an input ref.

This is the "module forward" operation at the reference level.

PyTorch analogy: calling `layer(x)` where the layer's parameters are already allocated.
-/
def forwardRef {σ τ : Shape} (l : Layer σ τ) {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Torch.Ops (m := m) (α := α)]
    (mode : Mode)
    (ps : Torch.RefList (RefTy (m := m) (α := α)) l.stateShapes)
    (x : RefTy (m := m) (α := α) σ) : m (RefTy (m := m) (α := α) τ) :=
  Torch.CurriedRef.uncurry (ss := l.stateShapes ++ [σ]) (Ref := RefTy (m := m) (α := α))
    (l.forward mode (α := α) (m := m)) (Torch.RefList.append ps (.cons x .nil))

/--
Run a `Layer` on concrete tensors by lowering its forward program to a typed graph.

This is primarily used by runtime utilities (e.g. sequential `updateBuffers`) where we want to run
forward to obtain intermediate activations.

PyTorch analogy: running a forward pass eagerly on concrete tensors.
-/
def forwardTensor {σ τ : Shape} (l : Layer σ τ) (mode : Mode)
    {α : Type} [Context α] [DecidableEq Shape]
    (ps : TorchLean.TensorPack α l.stateShapes) (x : Tensor α σ) : IO (Tensor α τ) := do
  let graph ← _root_.Runtime.Autograd.TorchLean.Autodiff.lowerToTypedGraph (α := α)
    (paramShapes := l.stateShapes) (inputShapes := [σ]) (τ := τ)
    (l.forward mode)
  let args : TorchLean.TensorPack α (l.stateShapes ++ [σ]) :=
    TorchLean.TensorPack.append (α := α) (ss₁ := l.stateShapes) (ss₂ := [σ]) ps
      (.cons x .nil)
  pure <| _root_.Runtime.Autograd.Torch.TypedGraph.forward graph args

end Layer
end NN

end TorchLean
end Autograd
end Runtime
