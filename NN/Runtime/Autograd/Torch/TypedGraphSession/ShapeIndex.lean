/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.TypedGraphSession.GraphOps

/-!
# Typed Graph Session: Shape and Indexing Operations
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch

open Spec
open Tensor

namespace Internal

namespace TypedGraphSession

/--
N-D max-pooling for channels-first tensors `(C, spatial...)` (no batch axis).

PyTorch comparison: `torch.nn.functional.max_pool1d` / `max_pool2d` / `max_pool3d` depending on the
spatial rank `d`.
-/
def maxPool {α : Type} (s : TypedGraphSession α) [Context α] [DecidableEq Shape]
  {d C : Nat} {inSpatial kernel stride padding : Spec.Tensor Nat [d]}
  {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
  (x : TensorRef α (Shape.ofList (C :: inSpatial.toList))) :
  IO (TensorRef α (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) :=
  commitGraphM (α := α) s
    (β := TensorRef α (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList)))
    (refs := #[x.identity?])
    (fun {Γ} {ss} xv nat g => do
      let (v, st') ← runGraphM (α := α) (Γ := Γ)
        (Runtime.Autograd.TypedGraph.GraphM.maxPool (α := α) (Γ := Γ) (d := d) (C := C)
          (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
          (hKernel := hKernel) { id := x.id })
        ss g
      let ⟨ss', g'⟩ := st'
      let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
      pure ({ id := v.id }, st1))

/--
N-D smooth max-pooling (log-sum-exp surrogate) for channels-first tensors `(C, spatial...)`.

This is a differentiable approximation of max-pooling; there is no direct PyTorch primitive.
-/
def smoothMaxPool {α : Type} (s : TypedGraphSession α) [Context α] [DecidableEq α]
  [DecidableEq Shape]
  {d C : Nat} {inSpatial kernel stride padding : Spec.Tensor Nat [d]}
  {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
  (x : TensorRef α (Shape.ofList (C :: inSpatial.toList))) (beta : α) :
  IO (TensorRef α (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) :=
  commitGraphM (α := α) s
    (β := TensorRef α (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList)))
    (refs := #[x.identity?])
    (fun {Γ} {ss} xv nat g => do
      let (v, st') ← runGraphM (α := α) (Γ := Γ)
        (Runtime.Autograd.TypedGraph.GraphM.smoothMaxPool (α := α) (Γ := Γ) (d := d) (C := C)
          (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
          (hKernel := hKernel) { id := x.id } beta)
        ss g
      let ⟨ss', g'⟩ := st'
      let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
      pure ({ id := v.id }, st1))

/--
N-D average-pooling for channels-first tensors `(C, spatial...)` (no batch axis).

PyTorch comparison: `torch.nn.functional.avg_pool1d` / `avg_pool2d` / `avg_pool3d` depending on the
spatial rank `d`.
-/
def avgPool {α : Type} (s : TypedGraphSession α) [Context α] [DecidableEq Shape]
  {d C : Nat} {inSpatial kernel stride padding : Spec.Tensor Nat [d]}
  (hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0)
  (x : TensorRef α (Shape.ofList (C :: inSpatial.toList))) :
  IO (TensorRef α (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) :=
  commitGraphM (α := α) s
    (β := TensorRef α (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList)))
    (refs := #[x.identity?])
    (fun {Γ} {ss} xv nat g => do
      let (v, st') ← runGraphM (α := α) (Γ := Γ)
        (Runtime.Autograd.TypedGraph.GraphM.avgPool (α := α) (Γ := Γ) (d := d) (C := C)
          (inSpatial := inSpatial) (kernel := kernel) (stride := stride) (padding := padding)
          hKernel { id := x.id })
        ss g
      let ⟨ss', g'⟩ := st'
      let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
      pure ({ id := v.id }, st1))

/--
Record elementwise ReLU.

PyTorch comparison: `torch.relu(x)` / `torch.nn.functional.relu(x)`.
-/
def relu {α : Type} (s : TypedGraphSession α)
  [Mul α] [Add α] [Zero α] [Max α] [One α] [LT α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {sh : Shape} (x : TensorRef α sh) : IO (TensorRef α sh) :=
  commitGraphM (α := α) s (β := TensorRef α sh) (refs := #[x.identity?])
      (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.relu (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Flatten a tensor into a 1D vector of length `Spec.Shape.size sh`.

PyTorch comparison: `torch.flatten(x)` (with default `start_dim=0`).
-/
def flatten {α : Type} (s : TypedGraphSession α) [Inhabited α] [Zero α] [DecidableEq Shape] {sh : Shape}
  (x : TensorRef α sh) : IO (TensorRef α [Spec.Shape.size sh]) :=
  commitGraphM (α := α) s (β := TensorRef α [Spec.Shape.size sh]) (refs := #[x.identity?])
      (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.flatten (α := α) (Γ := Γ) (s := sh) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Reshape a tensor while preserving total number of elements.

The proof argument `h` enforces `Spec.Shape.size sh1 = Spec.Shape.size sh2`.
PyTorch comparison: `torch.reshape(x, new_shape)` / `x.view(new_shape)` (when contiguous).
-/
def reshape {α : Type} (s : TypedGraphSession α) [Inhabited α] [Zero α] [DecidableEq Shape]
  {sh1 sh2 : Shape} (x : TensorRef α sh1) (h : Spec.Shape.size sh1 = Spec.Shape.size sh2) : IO (TensorRef α
    sh2) :=
  commitGraphM (α := α) s (β := TensorRef α sh2) (refs := #[x.identity?])
      (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.reshape (α := α) (Γ := Γ) (s₁ := sh1) (s₂ := sh2) { id :=
        x.id } h)
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Swap two adjacent axes at a given `depth` inside the shape.

Arbitrary permutations are lowered to this typed-graph primitive.
-/
def swapAdjacentAtDepth {α : Type} (s : TypedGraphSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (depth : Nat) (x : TensorRef α sh) : IO (TensorRef α (sh.swapAdjacentAtDepth depth))
    :=
  commitGraphM (α := α) s (β := TensorRef α (sh.swapAdjacentAtDepth depth))
      (refs := #[x.identity?]) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.swapAdjacentAtDepth (α := α) (Γ := Γ) (s := sh) depth { id
        := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Broadcast a tensor to a larger shape.

The witness `cb : Shape.CanBroadcastTo sh1 sh2` encodes the broadcasting proof.
PyTorch comparison: `x.expand(...)` / implicit broadcasting.
-/
def broadcastTo {α : Type} (s : TypedGraphSession α) [Inhabited α] [Add α] [Zero α] [DecidableEq Shape]
  {sh1 sh2 : Shape} (cb : Shape.CanBroadcastTo sh1 sh2) (x : TensorRef α sh1) : IO (TensorRef α sh2)
    :=
  commitGraphM (α := α) s (β := TensorRef α sh2) (refs := #[x.identity?])
      (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.broadcastTo (α := α) (Γ := Γ) (s₁ := sh1) (s₂ := sh2) cb {
        id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Sum-reduce along `axis`.

PyTorch comparison: `torch.sum(x, dim=axis)`.
-/
def reduceSum {α : Type} (s : TypedGraphSession α) [Add α] [Zero α] [Inhabited α] [DecidableEq Shape]
  {sh : Shape} (axis : Nat) [valid : Shape.HasNonemptyAxis axis sh] [wf : Shape.WellFormed sh]
  (x : TensorRef α sh) : IO (TensorRef α (shapeAfterSum sh axis)) :=
  commitGraphM (α := α) s (β := TensorRef α (shapeAfterSum sh axis))
      (refs := #[x.identity?]) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.reduceSum (α := α) (Γ := Γ) (s := sh) axis { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Mean-reduce along `axis`.

PyTorch comparison: `torch.mean(x, dim=axis)`.
-/
def reduceMean {α : Type} (s : TypedGraphSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (axis : Nat) [valid : Shape.HasNonemptyAxis axis sh] [wf : Shape.WellFormed sh]
  (x : TensorRef α sh) : IO (TensorRef α (shapeAfterSum sh axis)) :=
  commitGraphM (α := α) s (β := TensorRef α (shapeAfterSum sh axis))
      (refs := #[x.identity?]) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.reduceMean (α := α) (Γ := Γ) (s := sh) axis { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/-! ## Indexing -/

/-- Select one bounded coordinate from an arbitrary tensor axis. -/
def select {α : Type} (session : TypedGraphSession α) [Zero α] [DecidableEq Shape]
    {shape : Shape} (axis : Nat) [Shape.AxisInBounds axis shape]
    (x : TensorRef α shape) (index : Fin (Shape.axisSize shape axis)) :
    IO (TensorRef α (shape.eraseAxis axis)) :=
  commitGraphM (α := α) session (refs := #[x.identity?])
      (fun {Γ} {ss} values nat graph => do
    let (output, state') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.select (α := α) (Γ := Γ)
        (s := shape) axis { id := x.id } index) ss graph
    let ⟨ss', graph'⟩ := state'
    let state : TypedGraphSessionState α :=
      { Γ := Γ, x := values, nat := nat, ss := ss', g := graph' }
    pure ({ id := output.id }, state))

/-- Select several bounded coordinates from an arbitrary tensor axis. -/
def indexSelect {α : Type} (session : TypedGraphSession α) [Add α] [Zero α]
    [DecidableEq Shape] {shape : Shape} (axis count : Nat)
    [Shape.AxisInBounds axis shape] (x : TensorRef α shape)
    (indices : Tensor (Fin (Shape.axisSize shape axis)) [count]) :
    IO (TensorRef α (shape.replaceAxis axis count)) :=
  commitGraphM (α := α) session (refs := #[x.identity?])
      (fun {Γ} {ss} values nat graph => do
    let (output, state') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.indexSelect (α := α) (Γ := Γ)
        (s := shape) axis count { id := x.id } (fun _ => indices)) ss graph
    let ⟨ss', graph'⟩ := state'
    let state : TypedGraphSessionState α :=
      { Γ := Γ, x := values, nat := nat, ss := ss', g := graph' }
    pure ({ id := output.id }, state))

/-- Add source slices into an arbitrary tensor axis at bounded coordinates. -/
def scatterAdd {α : Type} (session : TypedGraphSession α) [Add α] [Zero α]
    [DecidableEq Shape] {shape : Shape} (axis count : Nat)
    [Shape.AxisInBounds axis shape] (base : TensorRef α shape)
    (source : TensorRef α (shape.replaceAxis axis count))
    (indices : Tensor (Fin (Shape.axisSize shape axis)) [count]) : IO (TensorRef α shape) :=
  commitGraphM (α := α) session (refs := #[base.identity?, source.identity?])
      (fun {Γ} {ss} values nat graph => do
    let (output, state') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.scatterAdd (α := α) (Γ := Γ)
        (s := shape) axis count { id := base.id } { id := source.id } (fun _ => indices)) ss graph
    let ⟨ss', graph'⟩ := state'
    let state : TypedGraphSessionState α :=
      { Γ := Γ, x := values, nat := nat, ss := ss', g := graph' }
    pure ({ id := output.id }, state))

end TypedGraphSession

end Internal

end Torch
end Autograd
end Runtime
