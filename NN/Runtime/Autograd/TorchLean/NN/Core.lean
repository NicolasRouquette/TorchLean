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
  initState : Torch.TList Float stateShapes
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
  requiresGrad : List Bool := List.replicate stateShapes.length true
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
        Torch.TList α stateShapes → Tensor α σ → IO (Torch.TList α stateShapes)
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
Update rule for a running statistics vector using momentum.

This implements an exponential moving average:

`next = (1 - momentum) * running + momentum * batch`.

PyTorch analogy: the update performed for `running_mean` / `running_var` in BatchNorm.
-/
def updateRunningVec {α : Type} [Context α] {c : Nat}
    (running batch : Tensor α (.dim c .scalar)) (momentum : Tensor α Shape.scalar) :
    Tensor α (.dim c .scalar) :=
  match running, batch, momentum with
  | .dim runningF, .dim batchF, .scalar mom =>
      let keep : Tensor α Shape.scalar := Tensor.scalar ((1 : α) - mom)
      Tensor.dim (fun i =>
        addSpec
          (mulSpec (runningF i) keep)
          (mulSpec (batchF i) (Tensor.scalar mom)))

/--
Convert the biased variance used by BatchNorm's training forward pass into the unbiased estimate
stored in its running buffer. For a singleton sample set there is no unbiased estimate; TorchLean
keeps the finite biased value rather than dividing by zero.
-/
def unbiasedRunningVariance {α : Type} [Context α] {c : Nat}
    (biased : Tensor α (.dim c .scalar)) (sampleCount : Nat) :
    Tensor α (.dim c .scalar) :=
  if sampleCount > 1 then
    scaleSpec biased ((sampleCount : α) / (sampleCount - 1 : Nat))
  else
    biased

/--
Compute per-channel mean and biased variance for a CHW tensor (no batch dimension).

This reduces over the spatial axes `(H, W)` and returns `(mean, var)` vectors of length `channels`.

The biased variance is the value used to normalize the current input. Convert it with
`unbiasedRunningVariance` before updating a PyTorch-compatible running-variance buffer.
-/
def chwBatchStats {α : Type} [Context α]
    {channels height width : Nat}
    (x : Tensor α (.dim channels (.dim height (.dim width .scalar)))) :
    Tensor α (.dim channels .scalar) × Tensor α (.dim channels .scalar) :=
  let means : Tensor α (.dim channels .scalar) :=
    Tensor.dim (fun c =>
      let channelData := getAtSpec x c
      let channelSum :=
        (List.finRange height).foldl (fun accH i =>
          (List.finRange width).foldl (fun accW j =>
            if hI : i < height then
              if hJ : j < width then
                addSpec accW (getAtSpec (getAtSpec channelData ⟨i, hI⟩) ⟨j, hJ⟩)
              else accW
            else accW
          ) accH
        ) (Tensor.scalar 0)
      divSpec channelSum (Tensor.scalar ((height * width : Nat) : α)))
  let vars : Tensor α (.dim channels .scalar) :=
    Tensor.dim (fun c =>
      let channelData := getAtSpec x c
      let mean := getAtSpec means c
      let varianceSum :=
        (List.finRange height).foldl (fun accH i =>
          (List.finRange width).foldl (fun accW j =>
            if hI : i < height then
              if hJ : j < width then
                let v := getAtSpec (getAtSpec channelData ⟨i, hI⟩) ⟨j, hJ⟩
                let d := subSpec v mean
                addSpec accW (mulSpec d d)
              else accW
            else accW
          ) accH
        ) (Tensor.scalar 0)
      divSpec varianceSum (Tensor.scalar ((height * width : Nat) : α)))
  (means, vars)

/--
Compute per-channel mean and biased variance for an NCHW tensor.

This reduces over `(N, H, W)` and returns `(mean, var)` vectors of length `c`.

These are the statistics used by `torch.nn.BatchNorm2d` to normalize the current batch. Its
running-variance update stores `unbiasedRunningVariance vars (n * h * w)` instead.
-/
def nchwBatchStats {α : Type} [Context α]
    {n c h w : Nat}
    (x : Tensor α (.dim n (.dim c (.dim h (.dim w .scalar))))) :
    Tensor α (.dim c .scalar) × Tensor α (.dim c .scalar) :=
  let means : Tensor α (.dim c .scalar) :=
    Tensor.dim (fun ch =>
      let total :=
        (List.finRange n).foldl (fun accN ni =>
          (List.finRange h).foldl (fun accH i =>
            (List.finRange w).foldl (fun accW j =>
              if hN : ni < n then
                if hI : i < h then
                  if hJ : j < w then
                    let sample := getAtSpec x ⟨ni, hN⟩
                    let channel := getAtSpec sample ch
                    addSpec accW (getAtSpec (getAtSpec channel ⟨i, hI⟩) ⟨j, hJ⟩)
                  else accW
                else accW
              else accW
            ) accH
          ) accN
        ) (Tensor.scalar 0)
      divSpec total (Tensor.scalar ((n * h * w : Nat) : α)))
  let vars : Tensor α (.dim c .scalar) :=
    Tensor.dim (fun ch =>
      let mean := getAtSpec means ch
      let total :=
        (List.finRange n).foldl (fun accN ni =>
          (List.finRange h).foldl (fun accH i =>
            (List.finRange w).foldl (fun accW j =>
              if hN : ni < n then
                if hI : i < h then
                  if hJ : j < w then
                    let sample := getAtSpec x ⟨ni, hN⟩
                    let channel := getAtSpec sample ch
                    let v := getAtSpec (getAtSpec channel ⟨i, hI⟩) ⟨j, hJ⟩
                    let d := subSpec v mean
                    addSpec accW (mulSpec d d)
                  else accW
                else accW
              else accW
            ) accH
          ) accN
        ) (Tensor.scalar 0)
      divSpec total (Tensor.scalar ((n * h * w : Nat) : α)))
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
    (initState : Torch.TList Float ps)
    (run : Mode → ∀ {α : Type}, [Context α] → [DecidableEq Shape] →
      ∀ {m : Type → Type}, [Monad m] → [Torch.Ops (m := m) (α := α)] →
        Torch.RefList (RefTy (m := m) (α := α)) ps →
        RefTy (m := m) (α := α) σ → m (RefTy (m := m) (α := α) τ))
    (runtimeInit : Option (TorchLean.Module.RuntimeInit.Plan ps) := none)
    (requiresGrad : List Bool := List.replicate ps.length true)
    (updateBuffers : Option (
      Mode → ∀ {α : Type}, [Context α] → [DecidableEq Shape] →
        Torch.TList α ps → Tensor α σ → IO (Torch.TList α ps)) := none) :
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
    (ps : Torch.TList α l.stateShapes) (x : Tensor α σ) : IO (Tensor α τ) := do
  let graph ← _root_.Runtime.Autograd.TorchLean.Autodiff.lowerToTypedGraph (α := α)
    (paramShapes := l.stateShapes) (inputShapes := [σ]) (τ := τ)
    (l.forward mode)
  let args : Torch.TList α (l.stateShapes ++ [σ]) :=
    _root_.Proofs.Autograd.Algebra.TList.append (α := α) (ss₁ := l.stateShapes) (ss₂ := [σ]) ps
      (.cons x .nil)
  pure <| _root_.Runtime.Autograd.Torch.TypedGraph.forward graph args

end Layer
end NN

end TorchLean
end Autograd
end Runtime
