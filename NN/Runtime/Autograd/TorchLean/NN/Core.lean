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
It packages parameter shapes/initial values together with a backend-polymorphic forward program, so
example code does not have to spell `paramShapes := [...]` / `inputShapes := [...]` everywhere.

## Main definitions

- `LayerDef σ τ` packages a shape-typed layer with explicit parameters (shapes + initial values) and
  a polymorphic `forward` program.
- `Seq σ τ` composes layers sequentially (PyTorch analogy: `torch.nn.Sequential`), written `f >>>
  g`.
- `scalarModuleDef*` helpers bundle a `Seq` model together with a scalar loss, producing a
  `TorchLean.Module.ScalarModuleDef` that the runtime training code can execute.

## PyTorch analogies

- `LayerDef` is like a small `nn.Module` definition, except parameters are an explicit list instead
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
A shape-typed layer definition with explicit parameters and a backend-polymorphic forward program.

`LayerDef σ τ` is the core building block used by `Seq` (sequential composition). It stores:
- a list of parameter shapes,
- initial values for those parameters/buffers (as `Float` tensors, for reproducible
  initialization),
- per-parameter `requires_grad` flags, and
- a `forward` program that is polymorphic over the backend monad and scalar type.

PyTorch analogy: a small `nn.Module`, where:
- `paramShapes`/`initParams` correspond to parameters (and possibly buffers),
- `forward` corresponds to `Module.forward`,
- `updateBuffers` corresponds to updating things like `running_mean`/`running_var` in BatchNorm.
-/
structure LayerDef (σ τ : Shape) where
  /-- Layer label used by public model summaries. -/
  kind : String := "Layer"
  /-- Shapes of the layer's parameter tensors, in the order expected by `forward`. -/
  paramShapes : List Shape
  /-- Initial parameter values (stored as `Float` tensors for convenient seeding/init schemes). -/
  initParams : Torch.TList Float paramShapes
  /--
  Optional storage-first initialization plan for executable `Float` backends.

  This does not replace `initParams`: the tensor-valued initializers remain available to the
  specification and proof layers. The plan lets a runtime create equivalent parameter storage
  without first enumerating those tensors on the host.
  -/
  runtimeInit : Option (TorchLean.Module.RuntimeInit.Plan paramShapes) :=
    if h : paramShapes = [] then some (h ▸ .nil) else none
  /--
  Per-parameter `requires_grad` flags (defaults to all `true`).

  PyTorch analogy: `tensor.requires_grad_(...)` on parameters/buffers.
  -/
  paramRequiresGrad : List Bool := List.replicate paramShapes.length true
  /--
  Optional buffer update function (used for running-statistics style layers).

  This is called during a forward pass (typically in `Mode.train`) to produce updated
    parameter/buffer
  values. A canonical example is BatchNorm updating its `running_mean` / `running_var` buffers.
  -/
  updateBuffers :
    Option (
      Mode → ∀ {α : Type}, [Context α] → [DecidableEq Shape] →
        Torch.TList α paramShapes → Tensor α σ → IO (Torch.TList α paramShapes)
    ) := none
  /--
  Forward pass as a typed TorchLean program.

  The program expects `(paramShapes ++ [σ])` inputs (the parameters, then the layer input) and
  produces an output of shape `τ`.
  -/
  forward :
    Mode → ∀ {α : Type}, [Context α] → [DecidableEq Shape] →
      TorchLean.Program α (paramShapes ++ [σ]) τ

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
Compute per-channel mean and variance for a CHW tensor (no batch dimension).

This reduces over the spatial axes `(H, W)` and returns `(mean, var)` vectors of length `channels`.

PyTorch analogy: the statistics used by `torch.nn.BatchNorm2d` in training mode (but here for an
unbatched `C×H×W` input).
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
Compute per-channel mean and variance for an NCHW tensor.

This reduces over `(N, H, W)` and returns `(mean, var)` vectors of length `c`.

PyTorch analogy: the batch statistics computed by `torch.nn.BatchNorm2d` in training mode.
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

namespace LayerDef

/--
Backend reference type used when running a `LayerDef`.

This is the `Ref` type provided by the current `Torch.Ops` backend instance, such as an eager tape or
compiled IR.
-/
abbrev RefT (m : Type → Type) (α : Type) [Context α] [DecidableEq Shape]
    [Torch.Ops (m := m) (α := α)] (s : Shape) : Type :=
  Torch.Ops.Ref (m := m) (α := α) s

/--
Construct a layer from an uncurried reference-level forward function.

This is the general constructor for non-sequential modules: branches, parameter sharing, and
other graph structures can inspect the parameter list directly without repeating the
`CurriedRef` packing boilerplate. Reusing the same reference in `run` reuses one autograd leaf;
there is no parameter copy or alias table hidden by this constructor.
-/
def ofRef {σ τ : Shape} {ps : List Shape}
    (kind : String)
    (initParams : Torch.TList Float ps)
    (run : Mode → ∀ {α : Type}, [Context α] → [DecidableEq Shape] →
      ∀ {m : Type → Type}, [Monad m] → [Torch.Ops (m := m) (α := α)] →
        Torch.RefList (RefT (m := m) (α := α)) ps →
        RefT (m := m) (α := α) σ → m (RefT (m := m) (α := α) τ))
    (runtimeInit : Option (TorchLean.Module.RuntimeInit.Plan ps) := none)
    (paramRequiresGrad : List Bool := List.replicate ps.length true)
    (updateBuffers : Option (
      Mode → ∀ {α : Type}, [Context α] → [DecidableEq Shape] →
        Torch.TList α ps → Tensor α σ → IO (Torch.TList α ps)) := none) :
    LayerDef σ τ :=
  { kind
    paramShapes := ps
    initParams
    runtimeInit
    paramRequiresGrad
    updateBuffers
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        Torch.CurriedRef.curry
          (Ref := RefT (m := m) (α := α))
          (ss := ps ++ [σ])
          (β := m (RefT (m := m) (α := α) τ))
          (fun args =>
            let (params, x) :=
              Torch.RefList.splitLast (Ref := RefT (m := m) (α := α))
                (ss := ps) (τ := σ) args
            run mode params x) }

/--
Run a `LayerDef` forward given parameter refs and an input ref.

This is the "module forward" operation at the reference level.

PyTorch analogy: calling `layer(x)` where the layer's parameters are already allocated.
-/
def forwardRef {σ τ : Shape} (l : LayerDef σ τ) {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Torch.Ops (m := m) (α := α)]
    (mode : Mode)
    (ps : Torch.RefList (RefT (m := m) (α := α)) l.paramShapes)
    (x : RefT (m := m) (α := α) σ) : m (RefT (m := m) (α := α) τ) :=
  Torch.CurriedRef.uncurry (ss := l.paramShapes ++ [σ]) (Ref := RefT (m := m) (α := α))
    (l.forward mode (α := α) (m := m)) (Torch.RefList.append ps (.cons x .nil))

/--
Run a `LayerDef` on concrete tensors by compiling its forward program.

This is primarily used by runtime utilities (e.g. sequential `updateBuffers`) where we want to run
forward to obtain intermediate activations.

PyTorch analogy: running a forward pass eagerly on concrete tensors.
-/
def forwardTensor {σ τ : Shape} (l : LayerDef σ τ) (mode : Mode)
    {α : Type} [Context α] [DecidableEq Shape]
    (ps : Torch.TList α l.paramShapes) (x : Tensor α σ) : IO (Tensor α τ) := do
  let compiled ← _root_.Runtime.Autograd.TorchLean.Autodiff.compileGraph (α := α)
    (paramShapes := l.paramShapes) (inputShapes := [σ]) (τ := τ)
    (l.forward mode)
  let args : Torch.TList α (l.paramShapes ++ [σ]) :=
    _root_.Proofs.Autograd.Algebra.TList.append (α := α) (ss₁ := l.paramShapes) (ss₂ := [σ]) ps
      (.cons x .nil)
  pure <| _root_.Runtime.Autograd.Torch.CompiledGraph.forward compiled args

end LayerDef
end NN

end TorchLean
end Autograd
end Runtime
