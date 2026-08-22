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
  {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
  {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (x : TensorRef α (Shape.ofList (C :: inSpatial.toList))) :
  IO (TensorRef α (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) :=
  commitGraphM (α := α) s
    (β := TensorRef α (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList)))
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
def smoothMaxPool {α : Type} (s : TypedGraphSession α) [Context α] [DecidableEq Shape]
  {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
  {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (x : TensorRef α (Shape.ofList (C :: inSpatial.toList))) (beta : α) :
  IO (TensorRef α (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) :=
  commitGraphM (α := α) s
    (β := TensorRef α (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList)))
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
  {d C : Nat} {inSpatial kernel stride padding : Vector Nat d}
  (hKernel : ∀ i : Fin d, kernel.get i ≠ 0)
  (x : TensorRef α (Shape.ofList (C :: inSpatial.toList))) :
  IO (TensorRef α (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList))) :=
  commitGraphM (α := α) s
    (β := TensorRef α (Shape.ofList (C :: (Spec.poolOutSpatialPad inSpatial kernel stride padding).toList)))
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
2D max-pooling for channel-first images.

PyTorch comparison: `torch.nn.functional.max_pool2d` (for NCHW-like layouts, here without batch).
-/
def maxPool2d {α : Type} (s : TypedGraphSession α) [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (TensorRef α (.dim inC (.dim (Spec.poolOutDim inH kH stride 0) (.dim (Spec.poolOutDim inW kW stride 0)
    .scalar)))) :=
  commitGraphM (α := α) s
    (β := TensorRef α (.dim inC (.dim (Spec.poolOutDim inH kH stride 0) (.dim (Spec.poolOutDim inW kW stride 0)
      .scalar))))
    (fun {Γ} {ss} xv nat g => do
      let (v, st') ← runGraphM (α := α) (Γ := Γ)
        (Runtime.Autograd.TypedGraph.GraphM.maxPool2d (α := α) (Γ := Γ)
          (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
          (h1 := h1) (h2 := h2) { id := x.id })
        ss g
      let ⟨ss', g'⟩ := st'
      let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
      pure ({ id := v.id }, st1))

/--
Smooth approximation of max-pooling (softmax pooling) for channel-first images.

This is not a standard PyTorch primitive; conceptually it behaves like applying a softmax over each
pooling window with inverse-temperature `beta` and returning the expected value.
-/
def smoothMaxPool2d {α : Type} (s : TypedGraphSession α) [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) (beta : α) :
  IO (TensorRef α (.dim inC (.dim (Spec.poolOutDim inH kH stride 0) (.dim (Spec.poolOutDim inW kW stride 0)
    .scalar)))) :=
  commitGraphM (α := α) s
    (β := TensorRef α (.dim inC (.dim (Spec.poolOutDim inH kH stride 0) (.dim (Spec.poolOutDim inW kW stride 0)
      .scalar))))
    (fun {Γ} {ss} xv nat g => do
      let (v, st') ← runGraphM (α := α) (Γ := Γ)
        (Runtime.Autograd.TypedGraph.GraphM.smoothMaxPool2d (α := α) (Γ := Γ)
          (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
          (h1 := h1) (h2 := h2) { id := x.id } beta)
        ss g
      let ⟨ss', g'⟩ := st'
      let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
      pure ({ id := v.id }, st1))

/--
2D average-pooling for channel-first images.

PyTorch comparison: `torch.nn.functional.avg_pool2d` (for NCHW-like layouts, here without batch).
-/
def avgPool2d {α : Type} (s : TypedGraphSession α) [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  (x : TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (TensorRef α (.dim inC (.dim (Spec.poolOutDim inH kH stride 0) (.dim (Spec.poolOutDim inW kW stride 0)
    .scalar)))) :=
  commitGraphM (α := α) s
    (β := TensorRef α (.dim inC (.dim (Spec.poolOutDim inH kH stride 0) (.dim (Spec.poolOutDim inW kW stride 0)
      .scalar))))
    (fun {Γ} {ss} xv nat g => do
      let (v, st') ← runGraphM (α := α) (Γ := Γ)
        (Runtime.Autograd.TypedGraph.GraphM.avgPool2d (α := α) (Γ := Γ)
          (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
          h1 h2 { id := x.id })
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
  commitGraphM (α := α) s (β := TensorRef α sh) (fun {Γ} {ss} xv nat g => do
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
  (x : TensorRef α sh) : IO (TensorRef α (.dim (Spec.Shape.size sh) .scalar)) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim (Spec.Shape.size sh) .scalar)) (fun {Γ} {ss} xv nat g
    => do
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
  commitGraphM (α := α) s (β := TensorRef α sh2) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.reshape (α := α) (Γ := Γ) (s₁ := sh1) (s₂ := sh2) { id :=
        x.id } h)
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Transpose a 2D matrix (swap the two axes).

PyTorch comparison: `x.t()` for 2D tensors, or `x.transpose(0, 1)`.
-/
def transpose2d {α : Type} (s : TypedGraphSession α) [Zero α] [DecidableEq Shape]
  {m n : Nat} (x : TensorRef α (.dim m (.dim n .scalar))) : IO (TensorRef α (.dim n (.dim m
    .scalar))) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim n (.dim m .scalar))) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.transpose2d (α := α) (Γ := Γ) (m := m) (n := n) { id := x.id
        })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Permute a 3D tensor by moving the first axis to the end: `(a,b,c) → (b,c,a)`.

PyTorch comparison: `x.permute(1,2,0)` for a 3D tensor.
-/
def transpose3dFirstToLast {α : Type} (s : TypedGraphSession α) [Zero α] [DecidableEq Shape]
  {a b c : Nat} (x : TensorRef α (.dim a (.dim b (.dim c .scalar)))) :
  IO (TensorRef α (.dim b (.dim c (.dim a .scalar)))) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim b (.dim c (.dim a .scalar)))) (fun {Γ} {ss} xv nat
    g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.transpose3dFirstToLast (α := α) (Γ := Γ) (a := a) (b :=
        b) (c := c) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Permute a 3D tensor by moving the last axis to the front: `(a,b,c) → (c,a,b)`.

PyTorch comparison: `x.permute(2,0,1)` for a 3D tensor.
-/
def transpose3dLastToFirst {α : Type} (s : TypedGraphSession α) [Zero α] [DecidableEq Shape]
  {a b c : Nat} (x : TensorRef α (.dim a (.dim b (.dim c .scalar)))) :
  IO (TensorRef α (.dim c (.dim a (.dim b .scalar)))) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim c (.dim a (.dim b .scalar)))) (fun {Γ} {ss} xv nat
    g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.transpose3dLastToFirst (α := α) (Γ := Γ) (a := a) (b :=
        b) (c := c) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Swap the last two axes of a 3D tensor: `(a,b,c) → (a,c,b)`.

PyTorch comparison: `x.transpose(1,2)` for a 3D tensor.
-/
def transpose3dLastTwo {α : Type} (s : TypedGraphSession α) [Zero α] [DecidableEq Shape]
  {a b c : Nat} (x : TensorRef α (.dim a (.dim b (.dim c .scalar)))) :
  IO (TensorRef α (.dim a (.dim c (.dim b .scalar)))) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim a (.dim c (.dim b .scalar)))) (fun {Γ} {ss} xv nat
    g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.transpose3dLastTwo (α := α) (Γ := Γ) (a := a) (b := b) (c
        := c) { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Swap two adjacent axes at a given `depth` inside the shape.

This is a more general permutation helper used in some shape-manipulating models.
PyTorch comparison: like `x.transpose(dim, dim+1)` for a suitably chosen `dim`.
-/
def swapAdjacentAtDepth {α : Type} (s : TypedGraphSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (depth : Nat) (x : TensorRef α sh) : IO (TensorRef α (sh.swapAdjacentAtDepth depth))
    :=
  commitGraphM (α := α) s (β := TensorRef α (sh.swapAdjacentAtDepth depth)) (fun {Γ} {ss} xv nat g
    => do
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
  commitGraphM (α := α) s (β := TensorRef α sh2) (fun {Γ} {ss} xv nat g => do
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
  commitGraphM (α := α) s (β := TensorRef α (shapeAfterSum sh axis)) (fun {Γ} {ss} xv nat g => do
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
  commitGraphM (α := α) s (β := TensorRef α (shapeAfterSum sh axis)) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.reduceMean (α := α) (Γ := Γ) (s := sh) axis { id := x.id })
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Gather a single scalar `x[i]` from a 1D vector, with a compile-time `Fin n` index.

PyTorch comparison: `x[i]` for a 1D tensor.
-/
def gatherScalar {α : Type} (s : TypedGraphSession α) [Zero α] [DecidableEq Shape]
  {n : Nat} (x : TensorRef α (.dim n .scalar)) (i : Fin n) : IO (TensorRef α Shape.scalar) :=
  commitGraphM (α := α) s (β := TensorRef α Shape.scalar) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.gatherScalar (α := α) (Γ := Γ) (n := n) { id := x.id } i)
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Gather a row `x[i]` from a 2D tensor, with a compile-time `Fin rows` index.

PyTorch comparison: `x[i]` for a 2D tensor (row indexing).
-/
def gatherRow {α : Type} (s : TypedGraphSession α) [Zero α] [DecidableEq Shape]
  {rows cols : Nat} (x : TensorRef α (.dim rows (.dim cols .scalar))) (i : Fin rows) :
  IO (TensorRef α (.dim cols .scalar)) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim cols .scalar)) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.gatherRow (α := α) (Γ := Γ) (rows := rows) (cols := cols) {
        id := x.id } i)
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Read a `Nat` from the nat-environment.

Out-of-bounds reads return `0` (total function), which is convenient for modeling "possibly invalid"
indices without throwing.
-/
def natAtOrZero (d : NatEnv) (id : Nat) : Nat :=
  match d[id]? with
  | some v => v
  | none => 0

/--
Read a length-`k` vector of `Nat`s starting at `start` from the nat-environment.

Out-of-bounds reads fall back to `0` elementwise via `natAtOrZero`.
-/
def natVecAtOrZero {k : Nat} (d : NatEnv) (start : Nat) : Tensor Nat (.dim k .scalar) :=
  Tensor.dim (fun i => Tensor.scalar (natAtOrZero d (start + i.val)))

/--
Dynamic gather of a scalar from a 1D vector using a runtime `NatRef` index.

Out-of-range indices produce `0` instead of raising.
PyTorch comparison: similar to `x[i]` where `i` is a Python integer, except PyTorch raises on
out-of-range while this definition totalizes the behavior for ease of reasoning.
-/
def gatherScalarRefOrZero {α : Type} (s : TypedGraphSession α) [Zero α] [DecidableEq Shape]
  {n : Nat} (x : TensorRef α (.dim n .scalar)) (i : NatRef) : IO (TensorRef α Shape.scalar) :=
  commitGraphM (α := α) s (β := TensorRef α Shape.scalar) (fun {Γ} {ss} xv nat g => do
    let ix ← mkIdxOrThrow (_α := α) (Γ := Γ) (ss := ss) x.id (.dim n .scalar)
    let node : _root_.Proofs.Autograd.Algebra.NodeData α NatEnv (Γ ++ ss) Shape.scalar :=
      { forward := fun ctx d =>
          let xv := _root_.Proofs.Autograd.Algebra.getIdx (α := α) (xs := ctx) ix
          let j := natAtOrZero d i.id
          if hj : j < n then
            getAtSpec xv ⟨j, hj⟩
          else
            Tensor.scalar 0
        jvp := fun _ctx dctx d =>
          let dx := _root_.Proofs.Autograd.Algebra.getIdx (α := α) (xs := dctx) ix
          let j := natAtOrZero d i.id
          if hj : j < n then
            getAtSpec dx ⟨j, hj⟩
          else
            Tensor.scalar 0
        vjp := fun _ctx d δ =>
          let gVal : α := Tensor.item δ
          let j := natAtOrZero d i.id
          if _hj : j < n then
            let dx : Tensor α (.dim n .scalar) :=
              Tensor.dim (fun k => Tensor.scalar (if decide (k.val = j) then gVal else 0))
            _root_.Proofs.Autograd.Algebra.TList.single (α := α) (Γ := Γ ++ ss) (s := .dim n
              .scalar) ix dx
          else
            _root_.Proofs.Autograd.Algebra.TList.zero (α := α) (ss := Γ ++ ss) }
    let outId : Nat := Γ.length + ss.length
    let ss' : List Shape := ss ++ [Shape.scalar]
    let g' : _root_.Proofs.Autograd.Algebra.GraphData α NatEnv Γ ss' := .snoc g node
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := outId }, st1))

/--
Dynamic gather of a row from a 2D tensor using a runtime `NatRef` index.

Out-of-range indices yield a zero row.
PyTorch comparison: similar to `x[i]` for 2D tensors with runtime `i`, but PyTorch raises on
out-of-range whereas this definition is totalized for ease of reasoning.
-/
def gatherRowRefOrZero {α : Type} (s : TypedGraphSession α) [Zero α] [DecidableEq Shape]
  {rows cols : Nat} (x : TensorRef α (.dim rows (.dim cols .scalar))) (i : NatRef) :
  IO (TensorRef α (.dim cols .scalar)) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim cols .scalar)) (fun {Γ} {ss} xv nat g => do
    let ix ← mkIdxOrThrow (_α := α) (Γ := Γ) (ss := ss) x.id (.dim rows (.dim cols .scalar))
    let outS : Shape := .dim cols .scalar
    let inS : Shape := .dim rows (.dim cols .scalar)
    let node : _root_.Proofs.Autograd.Algebra.NodeData α NatEnv (Γ ++ ss) outS :=
      { forward := fun ctx d =>
          let xv := _root_.Proofs.Autograd.Algebra.getIdx (α := α) (xs := ctx) ix
          let j := natAtOrZero d i.id
          if hj : j < rows then
            getAtSpec xv ⟨j, hj⟩
          else
            fill (0 : α) outS
        jvp := fun _ctx dctx d =>
          let dx := _root_.Proofs.Autograd.Algebra.getIdx (α := α) (xs := dctx) ix
          let j := natAtOrZero d i.id
          if hj : j < rows then
            getAtSpec dx ⟨j, hj⟩
          else
            fill (0 : α) outS
        vjp := fun _ctx d δ =>
          let j := natAtOrZero d i.id
          let dx : Tensor α inS :=
            if _hj : j < rows then
              Tensor.dim (fun r =>
                if decide (r.val = j) then
                  δ
                else
                  fill (0 : α) outS)
            else
              fill (0 : α) inS
          _root_.Proofs.Autograd.Algebra.TList.single (α := α) (Γ := Γ ++ ss) (s := inS) ix dx }
    let outId : Nat := Γ.length + ss.length
    let ss' : List Shape := ss ++ [outS]
    let g' : _root_.Proofs.Autograd.Algebra.GraphData α NatEnv Γ ss' := .snoc g node
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := outId }, st1))

/--
Dynamic gather of `k` scalars from a 1D tensor using a runtime `NatVecRef k` of indices.

Out-of-range indices yield `0`. In the VJP, gradients are accumulated for repeated indices
(i.e. it behaves like a gather followed by a scatter-add back into the source vector).
PyTorch comparison: related to `torch.gather` / advanced indexing, but with totalized out-of-range
behavior.
-/
def gatherVecRefOrZero {α : Type} (s : TypedGraphSession α) [Add α] [Zero α] [DecidableEq Shape]
  {n k : Nat} (x : TensorRef α (.dim n .scalar)) (idx : NatVecRef k) :
  IO (TensorRef α (.dim k .scalar)) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim k .scalar)) (fun {Γ} {ss} xv nat g => do
    let ix ← mkIdxOrThrow (_α := α) (Γ := Γ) (ss := ss) x.id (.dim n .scalar)
    let outS : Shape := .dim k .scalar
    let inS : Shape := .dim n .scalar
    let node : _root_.Proofs.Autograd.Algebra.NodeData α NatEnv (Γ ++ ss) outS :=
      { forward := fun ctx d =>
          let xv := _root_.Proofs.Autograd.Algebra.getIdx (α := α) (xs := ctx) ix
          let idxT := natVecAtOrZero (k := k) d idx.start
          match idxT with
          | Tensor.dim f =>
              Tensor.dim (fun j =>
                match f j with
                | Tensor.scalar ij =>
                    if h : ij < n then
                      getAtSpec xv ⟨ij, h⟩
                    else
                      Tensor.scalar 0)
        jvp := fun _ctx dctx d =>
          let dx := _root_.Proofs.Autograd.Algebra.getIdx (α := α) (xs := dctx) ix
          let idxT := natVecAtOrZero (k := k) d idx.start
          match idxT with
          | Tensor.dim f =>
              Tensor.dim (fun j =>
                match f j with
                | Tensor.scalar ij =>
                    if h : ij < n then
                      getAtSpec dx ⟨ij, h⟩
                    else
                      Tensor.scalar 0)
        vjp := fun _ctx d δ =>
          let idxT := natVecAtOrZero (k := k) d idx.start
          let dx : Tensor α inS :=
            Tensor.dim (fun iFin =>
              let sum : α :=
                (List.finRange k).foldl (fun acc j =>
                  let ij :=
                    match getAtSpec idxT j with
                    | Tensor.scalar v => v
                  if _hij : ij < n then
                    if decide (ij = iFin.val) then
                      let gj : α :=
                        match getAtSpec δ j with
                        | Tensor.scalar v => v
                      acc + gj
                    else acc
                  else acc
                ) 0
              Tensor.scalar sum)
          _root_.Proofs.Autograd.Algebra.TList.single (α := α) (Γ := Γ ++ ss) (s := inS) ix dx }
    let outId : Nat := Γ.length + ss.length
    let ss' : List Shape := ss ++ [outS]
    let g' : _root_.Proofs.Autograd.Algebra.GraphData α NatEnv Γ ss' := .snoc g node
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := outId }, st1))

/--
Dynamic gather of `k` rows from a 2D tensor using a runtime `NatVecRef k` of row indices.

Out-of-range indices yield zero rows. In the VJP, gradients are accumulated into the selected
rows (scatter-add semantics), including accumulation for repeated indices.
PyTorch comparison: similar to `torch.index_select(x, dim=0, index=...)` or advanced indexing on
the first dimension, but with totalized out-of-range behavior.
-/
def gatherRowsRefOrZero {α : Type} (s : TypedGraphSession α) [Add α] [Zero α] [DecidableEq Shape]
  {rows cols k : Nat} (x : TensorRef α (.dim rows (.dim cols .scalar))) (idx : NatVecRef k) :
  IO (TensorRef α (.dim k (.dim cols .scalar))) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim k (.dim cols .scalar))) (fun {Γ} {ss} xv nat g =>
    do
    let ix ← mkIdxOrThrow (_α := α) (Γ := Γ) (ss := ss) x.id (.dim rows (.dim cols .scalar))
    let outS : Shape := .dim k (.dim cols .scalar)
    let inS : Shape := .dim rows (.dim cols .scalar)
    let rowS : Shape := .dim cols .scalar
    let node : _root_.Proofs.Autograd.Algebra.NodeData α NatEnv (Γ ++ ss) outS :=
      { forward := fun ctx d =>
          let xv := _root_.Proofs.Autograd.Algebra.getIdx (α := α) (xs := ctx) ix
          let idxT := natVecAtOrZero (k := k) d idx.start
          match idxT with
          | Tensor.dim f =>
              Tensor.dim (fun j =>
                match f j with
                | Tensor.scalar ij =>
                    if h : ij < rows then
                      getAtSpec xv ⟨ij, h⟩
                    else
                      fill (0 : α) rowS)
        jvp := fun _ctx dctx d =>
          let dx := _root_.Proofs.Autograd.Algebra.getIdx (α := α) (xs := dctx) ix
          let idxT := natVecAtOrZero (k := k) d idx.start
          match idxT with
          | Tensor.dim f =>
              Tensor.dim (fun j =>
                match f j with
                | Tensor.scalar ij =>
                    if h : ij < rows then
                      getAtSpec dx ⟨ij, h⟩
                    else
                      fill (0 : α) rowS)
        vjp := fun _ctx d δ =>
          let idxT := natVecAtOrZero (k := k) d idx.start
          let dx : Tensor α inS :=
            Tensor.dim (fun rFin =>
              let rowGrad : Tensor α rowS :=
                (List.finRange k).foldl (fun acc j =>
                  let ij :=
                    match getAtSpec idxT j with
                    | Tensor.scalar v => v
                  if _hij : ij < rows then
                    if decide (ij = rFin.val) then
                      addSpec acc (getAtSpec δ j)
                    else acc
                  else acc
                ) (fill (0 : α) rowS)
              rowGrad)
          _root_.Proofs.Autograd.Algebra.TList.single (α := α) (Γ := Γ ++ ss) (s := inS) ix dx }
    let outId : Nat := Γ.length + ss.length
    let ss' : List Shape := ss ++ [outS]
    let g' : _root_.Proofs.Autograd.Algebra.GraphData α NatEnv Γ ss' := .snoc g node
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := outId }, st1))

/--
Gather a scalar from a 1D vector using a raw `Nat` index.

PyTorch comparison: like `x[i]` with an integer index, but this operation is recorded into the
shape-indexed graph so it remains explicit during lowering.
-/
def gatherScalarNatOrZero {α : Type} (s : TypedGraphSession α) [Zero α] [DecidableEq Shape]
  {n : Nat} (x : TensorRef α (.dim n .scalar)) (i : Nat) : IO (TensorRef α Shape.scalar) :=
  commitGraphM (α := α) s (β := TensorRef α Shape.scalar) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.gatherScalarNatOrZero (α := α) (Γ := Γ) (n := n) { id := x.id }
        i)
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Gather `k` scalars from a 1D vector using an explicit index tensor.

PyTorch comparison: related to `torch.gather` / advanced indexing with an integer index tensor.
-/
def gatherVecNatOrZero {α : Type} (s : TypedGraphSession α) [Add α] [Zero α] [DecidableEq Shape]
  {n k : Nat} (x : TensorRef α (.dim n .scalar)) (idx : Tensor Nat (.dim k .scalar)) :
  IO (TensorRef α (.dim k .scalar)) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim k .scalar)) (fun {Γ} {ss} xv nat g => do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.gatherVecNatOrZero (α := α) (Γ := Γ) (n := n) (k := k) { id :=
        x.id } (fun _ => idx))
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Gather `k` rows from a 2D tensor using an explicit index tensor.

PyTorch comparison: similar to `torch.index_select(x, dim=0, index=...)` or advanced indexing.
-/
def gatherRowsNatOrZero {α : Type} (s : TypedGraphSession α) [Add α] [Zero α] [DecidableEq Shape]
  {rows cols k : Nat} (x : TensorRef α (.dim rows (.dim cols .scalar))) (idx : Tensor Nat (.dim k
    .scalar)) :
  IO (TensorRef α (.dim k (.dim cols .scalar))) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim k (.dim cols .scalar))) (fun {Γ} {ss} xv nat g =>
    do
    let (v, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.gatherRowsNatOrZero (α := α) (Γ := Γ) (rows := rows) (cols :=
        cols) (k := k) { id := x.id } (fun _ => idx))
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := v.id }, st1))

/--
Scatter-add into a vector: return a copy of `x` with `x[i] += v`.

PyTorch comparison: similar to `x.scatter_add_(dim=0, index=..., src=...)` in spirit, but this is
functional (returns a new tensor) and uses a single `Fin n` index.
-/
def scatterAddVec {α : Type} (s : TypedGraphSession α) [Add α] [Zero α] [DecidableEq Shape]
  {n : Nat} (x : TensorRef α (.dim n .scalar)) (v : TensorRef α Shape.scalar) (i : Fin n) :
  IO (TensorRef α (.dim n .scalar)) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim n .scalar)) (fun {Γ} {ss} xv nat g => do
    let (out, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.scatterAddVec (α := α) (Γ := Γ) (n := n) { id := x.id } {
        id := v.id } i)
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := out.id }, st1))

/--
Scatter-add into a matrix row: return a copy of `x` with `x[i, :] += v`.

PyTorch comparison: like adding a row vector into a selected row (functional analogue of an
in-place indexed add).
-/
def scatterAddRow {α : Type} (s : TypedGraphSession α) [Add α] [Zero α] [DecidableEq Shape]
  {rows cols : Nat}
  (x : TensorRef α (.dim rows (.dim cols .scalar))) (v : TensorRef α (.dim cols .scalar)) (i : Fin
    rows) :
  IO (TensorRef α (.dim rows (.dim cols .scalar))) :=
  commitGraphM (α := α) s (β := TensorRef α (.dim rows (.dim cols .scalar))) (fun {Γ} {ss} xv nat g
    => do
    let (out, st') ← runGraphM (α := α) (Γ := Γ)
      (Runtime.Autograd.TypedGraph.GraphM.scatterAddRow (α := α) (Γ := Γ) (rows := rows) (cols :=
        cols) { id := x.id } { id := v.id } i)
      ss g
    let ⟨ss', g'⟩ := st'
    let st1 : TypedGraphSessionState α := { Γ := Γ, x := xv, nat := nat, ss := ss', g := g' }
    pure ({ id := out.id }, st1))
end TypedGraphSession

end Internal

end Torch
end Autograd
end Runtime
