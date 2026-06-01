/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.LinkedSession

import Mathlib.Algebra.Order.Algebra

/-!
# Session

TorchLean unified imperative session.

## Mental Model

A `Session α` is TorchLean's runtime analogue of a PyTorch "training loop environment". It:
- owns a collection of *leaf tensors* (parameters and inputs),
- records a forward computation as a tape / dataflow graph,
- can run reverse-mode AD to produce gradients for all leaves, and
- can apply simple optimizer steps (e.g. SGD) in a session-style workflow.

TorchLean exposes a **single API** with two execution backends selected at construction time:
- `.eager`: a tape-backed runtime session (imperative autograd tape; useful for debugging and
  interactive examples),
- `.compiled`: a proof-linked session that also records a (proved) IR graph while you build the tape
  and
  then executes via `Runtime.Autograd.Torch.Internal.SessionIR`.

Importantly, *the user-facing `Session` API is the same*: each op dispatches through `Session.impl`.

## Typical Training Loop (PyTorch Analogy)

Think of the following mapping (approximately):
- `Session.param` ~ create a `torch.nn.Parameter` (and later include it in a `state_dict`-like
  bundle).
- `Session.use` ~ "read" a parameter as a tensor in the current forward graph.
- `Session.input` ~ add a leaf tensor input (like feeding a batch tensor into the forward pass).
- `Session.resetTape` ~ start a fresh forward graph (closest in spirit to `optimizer.zero_grad()` +
  new forward).
- `Session.backwardScalarDenseAll` ~ `loss.backward()` (but returns gradients explicitly as an
  array).
- `Session.sgdStepAll` ~ `optimizer.step()` (dense helper; higher-level training lives in
  `NN.API.*`).
- `Session.detach` ~ `tensor.detach()` (cut the gradient edge at a value).

TorchLean does *not* store mutable `.grad` fields on each tensor ref; instead, gradients are
  returned
explicitly (see `grad`, `vjp`, and the `backward*DenseAll` functions).

## Non-Differentiable Inputs (`NatRef`)

For labels/indices, we keep a separate non-differentiable channel (`NatRef` and `NatVecRef`), used
  by
gather/indexing ops. This mirrors the practical reality that targets are often integer tensors in
PyTorch and should not require embedding into `α`.

## Deterministic RNG (Session-Level)

`RngState` provides explicit, deterministic RNG state (closer to JAX PRNG keys than a global RNG).
`freshSeedIO` is a convenience for sampling an initial seed at the IO boundary, while the *core*
semantics remains seed-threaded and replayable.

## Connection To TorchLean IR / Graph Execution

In the `.compiled` backend, the session records an IR graph while you build the tape; that IR is the
artifact that can be linked to proofs/verifiers. Execution follows the same
tape-level semantics, and the concrete graph object can be inspected or exported through the
compiled session.

Practical note: the current `.compiled` implementation expects all leaves (tensor inputs/parameters
and `NatRef`s) to be created before any op nodes are recorded. For portability, allocate leaves and
initialize/split RNG up-front, then build the forward graph.

### PyTorch References

- `torch.autograd`: https://pytorch.org/docs/stable/autograd.html
- Tensor hooks (conceptual analogue of `backwardDenseAllWithHook`):
  https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html

### AD References

This code follows the classic "tape / Wengert list" view of reverse-mode AD:
- Andreas Griewank and Andrea Walther, *Evaluating Derivatives*, 2nd ed., 2008.
- Seppo Linnainmaa, 1970 (reverse accumulation; precursor to modern backprop/autograd).
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor

/--
Eager-only session wrapper.

This is the public eager-session record backed by the internal tape session
`Runtime.Autograd.Torch.Internal.EagerSession`. Users normally interact with the unified `Session`
API; this type exists to support backend dispatch (`SessionImpl.eager`).
-/
structure EagerSession (α : Type) where
  /-- inner. -/
  inner : _root_.Runtime.Autograd.Torch.Internal.EagerSession α

namespace EagerSession

/--
Create a new eager (tape-backed) session.

This corresponds to the `.eager` backend of `Session.new`.
-/
def new {α : Type} (opts : _root_.Runtime.Autograd.Torch.Options := {}) : IO (EagerSession α) := do
  let inner ← _root_.Runtime.Autograd.Torch.Internal.EagerSession.new (α := α) (opts := opts)
  pure { inner := inner }

/-- Reset the eager autograd tape / graph-building state. -/
def resetTape {α : Type} (s : EagerSession α) : IO Unit := do
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.resetTape (α := α) s.inner

/--
Create a learnable parameter owned by this session.

PyTorch analogy: creating a `torch.nn.Parameter` during module initialization.
-/
def param {α : Type} (s : EagerSession α) {sh : Shape}
  (init : Tensor α sh) (name : Option String := none) (requiresGrad : Option Bool := none) :
  IO (_root_.Runtime.Autograd.Torch.Param α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.param (α := α) (sh := sh) s.inner
    init (name := name) (requiresGrad := requiresGrad)

/--
Use a parameter inside the current forward graph.

PyTorch analogy: reading a parameter in `forward` (it becomes part of the autograd graph).
-/
def use {α : Type} (s : EagerSession α) {sh : Shape} [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  (p : _root_.Runtime.Autograd.Torch.Param α sh) : IO (_root_.Runtime.Autograd.Torch.TensorRef α sh)
    :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.use (α := α) (sh := sh) s.inner p

/--
Add a tensor input leaf to the current graph.

`requiresGrad` controls whether this input is recorded as a differentiable leaf.
-/
def input {α : Type} (s : EagerSession α) {sh : Shape} [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  (v : Tensor α sh) (name : Option String := none) (requiresGrad : Bool := false) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.input (α := α) (sh := sh) s.inner
    v (name := name) (requiresGrad := requiresGrad)

/--
Add a non-differentiable `Nat` leaf to the session.

Used for labels/indices and gather-style ops.
-/
def inputNat {α : Type} (s : EagerSession α) (v : Nat) : IO (_root_.Runtime.Autograd.Torch.NatRef)
  :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.inputNat (α := α) s.inner v

/-- Read a `NatRef` value. -/
def getNat {α : Type} (s : EagerSession α) (r : _root_.Runtime.Autograd.Torch.NatRef) : IO Nat :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.getNat (α := α) s.inner r

/-- Mutate a `NatRef` value. -/
def setNat {α : Type} (s : EagerSession α) (r : _root_.Runtime.Autograd.Torch.NatRef) (v : Nat) : IO
  Unit :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.setNat (α := α) s.inner r v

/-- Add a non-differentiable vector-of-`Nat` leaf. -/
def inputNatVec {α : Type} {k : Nat} (s : EagerSession α) (v : Tensor Nat (.dim k .scalar)) :
    IO (_root_.Runtime.Autograd.Torch.NatVecRef k) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.inputNatVec (α := α) (k := k) s.inner v

/-- Read back a `NatVecRef` value. -/
def getNatVec {α : Type} {k : Nat} (s : EagerSession α) (r : _root_.Runtime.Autograd.Torch.NatVecRef
  k) :
    IO (Tensor Nat (.dim k .scalar)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.getNatVec (α := α) (k := k) s.inner r

/-- Mutate a `NatVecRef` value. -/
def setNatVec {α : Type} {k : Nat} (s : EagerSession α) (r : _root_.Runtime.Autograd.Torch.NatVecRef
  k)
    (v : Tensor Nat (.dim k .scalar)) : IO Unit :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.setNatVec (α := α) (k := k) s.inner r v

/--
Insert a constant tensor into the current graph.

PyTorch analogy: using a tensor literal/constant in the forward pass (as a leaf constant node).
-/
def const {α : Type} (s : EagerSession α) {sh : Shape} [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  (v : Tensor α sh) (name : Option String := none) : IO (_root_.Runtime.Autograd.Torch.TensorRef α
    sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.const (α := α) (sh := sh) s.inner v (name :=
    name)

/-- Read the concrete value for a tensor ref (for logging/debugging). -/
def getValue {α : Type} (s : EagerSession α) {sh : Shape} [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO (Tensor α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.getValue (α := α) (sh := sh) s.inner x

/--
Detach a tensor ref from the tape (stop gradient flow through it).

PyTorch analogy: `x.detach()`.
-/
def detach {α : Type} (s : EagerSession α) {sh : Shape} [Context α] [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
    IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.detach (α := α) (sh := sh) s.inner x

/-- Elementwise addition on tensor refs (eager backend). -/
def add {α : Type} (s : EagerSession α) [Add α] [DecidableEq Shape] {sh : Shape}
  (a b : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO (_root_.Runtime.Autograd.Torch.TensorRef
    α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.add (α := α) (sh := sh) s.inner a b

/-- Elementwise subtraction on tensor refs (eager backend). -/
def sub {α : Type} (s : EagerSession α) [Sub α] [Zero α] [DecidableEq Shape] {sh : Shape}
  (a b : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO (_root_.Runtime.Autograd.Torch.TensorRef
    α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.sub (α := α) (sh := sh) s.inner a b

/-- Elementwise multiplication on tensor refs (eager backend). -/
def mul {α : Type} (s : EagerSession α) [Mul α] [DecidableEq Shape] {sh : Shape}
  (a b : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO (_root_.Runtime.Autograd.Torch.TensorRef
    α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.mul (α := α) (sh := sh) s.inner a b

/-- Elementwise scaling by a scalar constant `c` (eager backend). -/
def scale {α : Type} (s : EagerSession α) [Mul α] [DecidableEq Shape] {sh : Shape}
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) (c : α) : IO
    (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.scale (α := α) (sh := sh) s.inner x c

/-- Elementwise absolute value (eager backend). -/
def abs {α : Type} (s : EagerSession α) [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.abs (α := α) (sh := sh) s.inner x

/-- Elementwise square root (eager backend). -/
def sqrt {α : Type} (s : EagerSession α) [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.sqrt (α := α) (sh := sh) s.inner x

/-- Elementwise clamp to `[minVal, maxVal]` (eager backend). -/
def clamp {α : Type} (s : EagerSession α) [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) (minVal maxVal : α) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.clamp (α := α) (sh := sh) s.inner x minVal
    maxVal

/-- Elementwise maximum (eager backend). -/
def max {α : Type} (s : EagerSession α) [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape]
  {sh : Shape} (a b : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.max (α := α) (sh := sh) s.inner a b

/-- Elementwise minimum (eager backend). -/
def min {α : Type} (s : EagerSession α) [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape]
  {sh : Shape} (a b : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.min (α := α) (sh := sh) s.inner a b

/--
Matrix multiplication (2D) on tensor refs (eager backend).

PyTorch analogy: `torch.matmul` on rank-2 tensors (or the `@` operator).
-/
def matmul {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {m n p : Nat}
  (a : _root_.Runtime.Autograd.Torch.TensorRef α (.dim m (.dim n .scalar)))
  (b : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n (.dim p .scalar))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim m (.dim p .scalar))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.matmul (α := α) s.inner (m := m) (n := n) (p
    := p) a b

/--
Batched matrix multiplication (3D) on tensor refs (eager backend).

PyTorch analogy: `torch.bmm`.
-/
def bmm {α : Type} (s : EagerSession α) [Add α] [Mul α] [Zero α] [DecidableEq Shape]
  {batch m n p : Nat}
  (a : _root_.Runtime.Autograd.Torch.TensorRef α (.dim batch (.dim m (.dim n .scalar))))
  (b : _root_.Runtime.Autograd.Torch.TensorRef α (.dim batch (.dim n (.dim p .scalar)))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim batch (.dim m (.dim p .scalar)))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.bmm (α := α) s.inner (batch := batch) (m := m)
    (n := n) (p := p) a b

/--
Concatenate two vectors along the only dimension (eager backend).

PyTorch analogy: `torch.cat([a, b], dim=0)` for 1D tensors.
-/
def concatVectors {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {n m : Nat}
  (a : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n .scalar))
  (b : _root_.Runtime.Autograd.Torch.TensorRef α (.dim m .scalar)) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim (n + m) .scalar)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.concatVectors (α := α) s.inner (n := n) (m :=
    m) a b

/--
Concatenate along the outermost dimension (dimension 0) (eager backend).

PyTorch analogy: `torch.cat([a, b], dim=0)`.
-/
def concatDim0 {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {n m : Nat} {sh : Shape}
  (a : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n sh))
  (b : _root_.Runtime.Autograd.Torch.TensorRef α (.dim m sh)) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim (n + m) sh)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.concatDim0 (α := α) s.inner (n := n) (m := m)
    (sh := sh) a b

/--
Slice a contiguous `[start, start+len)` range from dimension 0 (eager backend).

PyTorch analogy: `x[start:start+len]` for the first dimension.
-/
def sliceRange0 {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape]
  {n : Nat} {sh : Shape}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n sh)) (start len : Nat) (h : len + start ≤
    n) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim len sh)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.sliceRange0 (α := α) s.inner (n := n) (sh :=
    sh) x start len h

/--
2D max pooling on a CHW tensor (eager backend).

PyTorch analogy: `torch.nn.functional.max_pool2d` (channel-first layout).
-/
def maxPool2d {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α
    (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1) .scalar)))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.maxPool2d (α := α) s.inner
    (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
    (h1 := h1) (h2 := h2) x

/-- Alias for `max_pool2d` (PyTorch-style shorthand). -/
abbrev maxPool {α : Type} := maxPool2d (α := α)

/--
Smooth max pooling (softmax-like pooling) on a CHW tensor (eager backend).

This is a differentiable surrogate for max pooling parameterized by `beta`.
-/
def smoothMaxPool2d {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  {kH kW inH inW inC stride : Nat} {h1 : kH ≠ 0} {h2 : kW ≠ 0}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) (beta :
    α) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α
    (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1) .scalar)))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.smoothMaxPool2d (α := α) s.inner
    (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
    (h1 := h1) (h2 := h2) x beta

/-- Alias for `smooth_max_pool2d` (PyTorch-style shorthand). -/
abbrev smoothMaxPool {α : Type} := smoothMaxPool2d (α := α)

/--
2D average pooling on a CHW tensor (eager backend).

PyTorch analogy: `torch.nn.functional.avg_pool2d` (channel-first layout).
-/
def avgPool2d {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {kH kW inH inW inC stride : Nat} (h1 : kH ≠ 0) (h2 : kW ≠ 0)
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α
    (.dim inC (.dim ((inH - kH) / stride + 1) (.dim ((inW - kW) / stride + 1) .scalar)))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.avgPool2d (α := α) s.inner
    (kH := kH) (kW := kW) (inH := inH) (inW := inW) (inC := inC) (stride := stride)
    h1 h2 x

/-- Alias for `avg_pool2d` (PyTorch-style shorthand). -/
abbrev avgPool {α : Type} := avgPool2d (α := α)

/-- Elementwise ReLU activation (eager backend). -/
def relu {α : Type} (s : EagerSession α)
  [Mul α] [Zero α] [Max α] [One α] [LT α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO
    (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.relu (α := α) (sh := sh) s.inner x

/-- Elementwise sigmoid activation (eager backend). -/
def sigmoid {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO
    (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.sigmoid (α := α) (sh := sh) s.inner x

/-- Elementwise tanh activation (eager backend). -/
def tanh {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO
    (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.tanh (α := α) (sh := sh) s.inner x

/-- Elementwise softmax activation (eager backend). -/
def softmax {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO
    (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.softmax (α := α) (sh := sh) s.inner x

/-- Stable log-softmax along the last axis (eager backend). -/
def logSoftmax {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO
    (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.logSoftmax (α := α) (sh := sh) s.inner x

/-- Elementwise softplus activation (eager backend). -/
def softplus {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO
    (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.softplus (α := α) (sh := sh) s.inner x

/-- Elementwise exponential (eager backend). -/
def exp {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO
    (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.exp (α := α) (sh := sh) s.inner x

/-- Elementwise logarithm (eager backend). -/
def log {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO
    (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.log (α := α) (sh := sh) s.inner x

/-- Elementwise `safe_log` activation (`log(softplus(x) + ε)`) (eager backend). -/
def safeLog {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) (ε : α := Numbers.epsilon) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.safeLog (α := α) (sh := sh) s.inner x (ε :=
    ε)

/-- Sum-reduce a tensor to a scalar (eager backend). -/
def sum {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.sum (α := α) (sh := sh) s.inner x

/-- Flatten a tensor into a 1D vector (eager backend). -/
def flatten {α : Type} (s : EagerSession α) [Inhabited α] [DecidableEq Shape] {sh : Shape}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim (Shape.size sh) .scalar)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.flatten (α := α) (sh := sh) s.inner x

/--
Reshape a tensor, given a proof that the total number of elements is preserved (eager backend).

PyTorch analogy: `x.reshape(...)` when the element count matches.
-/
def reshape {α : Type} (s : EagerSession α) [Inhabited α] [DecidableEq Shape] {sh1 sh2 : Shape}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α sh1) (h : Shape.size sh1 = Shape.size sh2) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh2) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.reshape (α := α) (sh1 := sh1) (sh2 := sh2)
    s.inner x h

/-- Transpose a 2D matrix (eager backend). -/
def transpose2d {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape] {m n : Nat}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim m (.dim n .scalar))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim n (.dim m .scalar))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.transpose2d (α := α) (m := m) (n := n) s.inner
    x

/-- Permute a 3D tensor by moving the first dimension to the last (eager backend). -/
def transpose3dFirstToLast {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape] {a b c :
  Nat}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim a (.dim b (.dim c .scalar)))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim b (.dim c (.dim a .scalar)))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.transpose3dFirstToLast (α := α)
    (a := a) (b := b) (c := c) s.inner x

/-- Permute a 3D tensor by moving the last dimension to the first (eager backend). -/
def transpose3dLastToFirst {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape] {a b c :
  Nat}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim a (.dim b (.dim c .scalar)))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim c (.dim a (.dim b .scalar)))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.transpose3dLastToFirst (α := α)
    (a := a) (b := b) (c := c) s.inner x

/-- Swap the last two axes of a 3D tensor (eager backend). -/
def transpose3dLastTwo {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape] {a b c : Nat}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim a (.dim b (.dim c .scalar)))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim a (.dim c (.dim b .scalar)))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.transpose3dLastTwo (α := α)
    (a := a) (b := b) (c := c) s.inner x

/--
Generic "swap adjacent axes" view operation (eager backend).

This is a shape-driven permutation helper used in some attention/transformer code.
-/
def swapAdjacentAtDepth {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape] {sh : Shape}
  (depth : Nat) (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (sh.swapAdjacentAtDepth depth)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.swapAdjacentAtDepth (α := α) (sh := sh)
    s.inner depth x

/-- Broadcast a tensor to a larger shape (eager backend). -/
def broadcastTo {α : Type} (s : EagerSession α) [Inhabited α] [Add α] [Zero α] [DecidableEq Shape]
  {sh1 sh2 : Shape} (cb : Shape.CanBroadcastTo sh1 sh2) (x : _root_.Runtime.Autograd.Torch.TensorRef
    α sh1) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh2) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.broadcastTo (α := α) (sh1 := sh1) (sh2 := sh2)
    s.inner cb x

/-- Reduce-sum along an axis (eager backend). -/
def reduceSum {α : Type} (s : EagerSession α) [Add α] [Zero α] [Inhabited α] [DecidableEq Shape]
  {sh : Shape} (axis : Nat) [valid : Shape.valid_axis_inst axis sh] [wf : Shape.WellFormed sh]
  (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (shapeAfterSum sh axis)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.reduceSum (α := α) (sh := sh) s.inner axis x

/-- Reduce-mean along an axis (eager backend). -/
def reduceMean {α : Type} (s : EagerSession α) [Context α] [Inhabited α] [DecidableEq Shape]
  {sh : Shape} (axis : Nat) [valid : Shape.valid_axis_inst axis sh] [wf : Shape.WellFormed sh]
  (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (shapeAfterSum sh axis)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.reduceMean (α := α) (sh := sh) s.inner axis x

/-- Gather a single scalar from a vector at a `Fin` index (eager backend). -/
def gatherScalar {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape]
  {n : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n .scalar)) (i : Fin n) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.gatherScalar (α := α) (n := n) s.inner x i

/-- Gather a row from a matrix at a `Fin` index (eager backend). -/
def gatherRow {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape]
  {rows cols : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim rows (.dim cols .scalar)))
    (i : Fin rows) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim cols .scalar)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.gatherRow (α := α) (rows := rows) (cols :=
    cols) s.inner x i

/-- Gather a scalar from a vector using a `NatRef` index (eager backend). -/
def gatherScalarRef {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape]
  {n : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n .scalar)) (i :
    _root_.Runtime.Autograd.Torch.NatRef) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.gatherScalarRef (α := α) (n := n) s.inner x
    i

/-- Gather a row from a matrix using a `NatRef` index (eager backend). -/
def gatherRowRef {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  {rows cols : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim rows (.dim cols .scalar)))
    (i : _root_.Runtime.Autograd.Torch.NatRef) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim cols .scalar)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.gatherRowRef (α := α) (rows := rows) (cols
    := cols) s.inner x i

/-- Gather a scalar using a raw `Nat` index (eager backend). -/
def gatherScalarNat {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape]
  {n : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n .scalar)) (i : Nat) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.gatherScalarNat (α := α) (n := n) s.inner x
    i

/--
Gather a vector of entries from a vector using an index tensor (eager backend).

PyTorch analogy: `x[idx]` where `idx` is an integer tensor (1D).
-/
def gatherVecNat {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {n k : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n .scalar)) (idx : Tensor Nat
    (.dim k .scalar)) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim k .scalar)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.gatherVecNat (α := α) (n := n) (k := k)
    s.inner x idx

/-- Gather multiple rows from a matrix using an index tensor (eager backend). -/
def gatherRowsNat {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {rows cols k : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim rows (.dim cols
    .scalar)))
  (idx : Tensor Nat (.dim k .scalar)) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim k (.dim cols .scalar))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.gatherRowsNat (α := α)
    (rows := rows) (cols := cols) (k := k) s.inner x idx

/-- `gather_vec_nat`, but the indices are provided as a `NatVecRef` leaf (eager backend). -/
def gatherVecRef {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {n k : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n .scalar))
  (idx : _root_.Runtime.Autograd.Torch.NatVecRef k) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim k .scalar)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.gatherVecRef (α := α) (n := n) (k := k)
    s.inner x idx

/-- `gather_rows_nat`, but the indices are provided as a `NatVecRef` leaf (eager backend). -/
def gatherRowsRef {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {rows cols k : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim rows (.dim cols
    .scalar)))
  (idx : _root_.Runtime.Autograd.Torch.NatVecRef k) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim k (.dim cols .scalar))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.gatherRowsRef (α := α) (rows := rows) (cols
    := cols) (k := k) s.inner x idx

/--
Scatter-add into a vector at a `Fin` index (eager backend).

PyTorch analogy: `x.index_add_(dim=0, index=[i], source=v)` for a single index.
-/
def scatterAddVec {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {n : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n .scalar))
  (v : _root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) (i : Fin n) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim n .scalar)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.scatterAddVec (α := α) (n := n) s.inner x v
    i

/-- Scatter-add into a matrix row at a `Fin` index (eager backend). -/
def scatterAddRow {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {rows cols : Nat}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim rows (.dim cols .scalar)))
  (v : _root_.Runtime.Autograd.Torch.TensorRef α (.dim cols .scalar)) (i : Fin rows) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim rows (.dim cols .scalar))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.scatterAddRow (α := α) (rows := rows) (cols
    := cols) s.inner x v i

/--
Fully-connected (affine) layer on vectors: `y = w·x + b` (eager backend).

PyTorch analogue: `torch.nn.functional.linear` (with weight shape `(outDim, inDim)`).
-/
def linear {α : Type} (s : EagerSession α) [Inhabited α] [Add α] [Mul α] [Zero α] [DecidableEq
  Shape] [_root_.Runtime.Autograd.FastKernels.FastMatmul α]
  {inDim outDim : Nat}
  (w : _root_.Runtime.Autograd.Torch.TensorRef α (.dim outDim (.dim inDim .scalar)))
  (b : _root_.Runtime.Autograd.Torch.TensorRef α (.dim outDim .scalar))
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim inDim .scalar)) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim outDim .scalar)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.linear (α := α) (inDim := inDim) (outDim :=
    outDim)
    s.inner w b x

/--
Mean squared error loss returning a scalar (eager backend).

PyTorch analogue: `torch.nn.functional.mse_loss(..., reduction='mean')`.
-/
def mseLoss {α : Type} (s : EagerSession α)
  [Inhabited α] [Add α] [Sub α] [Mul α] [Div α] [Zero α] [One α] [Coe Nat α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  {sh : Shape}
  (yhat target : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.mseLoss (α := α) (sh := sh) s.inner yhat
    target

/--
LayerNorm over a `seqLen × embedDim` tensor (eager backend).

PyTorch analogue: `torch.nn.LayerNorm(embedDim)` applied per token.
-/
def layerNorm {α : Type} (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim seqLen (.dim embedDim .scalar)))
  (gamma : _root_.Runtime.Autograd.Torch.TensorRef α (.dim embedDim .scalar))
  (beta : _root_.Runtime.Autograd.Torch.TensorRef α (.dim embedDim .scalar)) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim seqLen (.dim embedDim .scalar))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.layerNorm (α := α)
    (seqLen := seqLen) (embedDim := embedDim) (h_seq_pos := h_seq_pos) (h_embed_pos := h_embed_pos)
    s.inner x gamma beta

/--
BatchNorm over a CHW tensor (eager backend).

PyTorch analogue: `torch.nn.BatchNorm2d` (channel-first layout).
-/
def batchnormChannelFirst {α : Type} (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {channels height width : Nat} (h_c : channels > 0) (h_h : height > 0) (h_w : width > 0)
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim channels (.dim height (.dim width .scalar))))
  (gamma : _root_.Runtime.Autograd.Torch.TensorRef α (.dim channels .scalar))
  (beta : _root_.Runtime.Autograd.Torch.TensorRef α (.dim channels .scalar)) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim channels (.dim height (.dim width .scalar))))
    :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.batchnormChannelFirst (α := α)
    (channels := channels) (height := height) (width := width) (h_c := h_c) (h_h := h_h) (h_w :=
      h_w)
    s.inner x gamma beta

/--
N-D convolution over a channels-first tensor `(inC, spatial...)` (eager backend).

This is the generic counterpart to `conv2d`.

PyTorch analogue: `torch.nn.functional.conv{d}d` specialized to a single sample.
-/
def conv {α : Type} (s : EagerSession α) [Context α]
  [DecidableEq Shape]
  {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (w : _root_.Runtime.Autograd.Torch.TensorRef α (Shape.ofList (outC :: inC :: kernel.toList)))
  (b : _root_.Runtime.Autograd.Torch.TensorRef α (.dim outC .scalar))
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (Shape.ofList (inC :: inSpatial.toList))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α
    (Shape.ofList (outC :: (Spec.convOutSpatial inSpatial kernel stride padding).toList))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.conv (α := α)
    (d := d) (inC := inC) (outC := outC)
    (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
    (hInC := hInC) (hKernel := hKernel)
    s.inner w b x

/--
N-D transpose convolution over a channels-first tensor `(inC, spatial...)` (eager backend).

PyTorch analogue: `torch.nn.functional.conv_transpose{d}d` specialized to a single sample.
-/
def convTranspose {α : Type} (s : EagerSession α) [Context α]
  [DecidableEq Shape]
  {d inC outC : Nat}
  {kernel stride padding : Vector Nat d}
  {inSpatial : Vector Nat d}
  {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.get i ≠ 0}
  (w : _root_.Runtime.Autograd.Torch.TensorRef α (Shape.ofList (inC :: outC :: kernel.toList)))
  (b : _root_.Runtime.Autograd.Torch.TensorRef α (.dim outC .scalar))
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (Shape.ofList (inC :: inSpatial.toList))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α
    (Shape.ofList (outC :: (Spec.convTransposeOutSpatial inSpatial kernel stride padding).toList)))
    :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.convTranspose (α := α)
    (d := d) (inC := inC) (outC := outC)
    (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
    (hInC := hInC) (hKernel := hKernel)
    s.inner w b x

/--
2D convolution over a CHW tensor (eager backend).

PyTorch analogue: `torch.nn.functional.conv2d` (channel-first layout).
-/
def conv2d {α : Type} (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (kernel : _root_.Runtime.Autograd.Torch.TensorRef α (.dim outC (.dim inC (.dim kH (.dim kW
    .scalar)))))
  (bias : _root_.Runtime.Autograd.Torch.TensorRef α (.dim outC .scalar))
  (input : _root_.Runtime.Autograd.Torch.TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α
    (.dim outC (.dim ((inH + 2 * padding - kH) / stride + 1) (.dim ((inW + 2 * padding - kW) /
      stride + 1) .scalar)))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.conv2d (α := α)
    (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
    (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 := h3)
    s.inner kernel bias input

/--
2D transpose convolution over a CHW tensor (eager backend).

PyTorch analogue: `torch.nn.functional.conv_transpose2d` (channel-first layout).
-/
def convTranspose2d {α : Type} (s : EagerSession α) [Context α]
  [DecidableEq Shape]
  {inC outC kH kW stride padding inH inW : Nat}
  {h1 : inC ≠ 0} {h2 : kH ≠ 0} {h3 : kW ≠ 0}
  (kernel : _root_.Runtime.Autograd.Torch.TensorRef α (.dim inC (.dim outC (.dim kH (.dim kW
    .scalar)))))
  (bias : _root_.Runtime.Autograd.Torch.TensorRef α (.dim outC .scalar))
  (input : _root_.Runtime.Autograd.Torch.TensorRef α (.dim inC (.dim inH (.dim inW .scalar)))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α
    (.dim outC (.dim ((inH - 1) * stride - 2 * padding + kH)
      (.dim ((inW - 1) * stride - 2 * padding + kW) .scalar)))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.convTranspose2d (α := α)
    (inC := inC) (outC := outC) (kH := kH) (kW := kW) (stride := stride) (padding := padding)
    (inH := inH) (inW := inW) (h1 := h1) (h2 := h2) (h3 := h3)
    s.inner kernel bias input

/--
Multi-head self-attention (eager backend).

This is the eager-backend implementation used by the transformer examples (approximately analogous to
`torch.nn.MultiheadAttention` in self-attention mode).
-/
def multiHeadAttention {α : Type} (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (wq : _root_.Runtime.Autograd.Torch.TensorRef α (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wk : _root_.Runtime.Autograd.Torch.TensorRef α (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wv : _root_.Runtime.Autograd.Torch.TensorRef α (.dim dModel (.dim (numHeads * headDim) .scalar)))
  (wo : _root_.Runtime.Autograd.Torch.TensorRef α (.dim (numHeads * headDim) (.dim dModel .scalar)))
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n (.dim dModel .scalar)))
  (mask : Option (Tensor Bool (.dim n (.dim n .scalar))) := none) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim n (.dim dModel .scalar))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.multiHeadAttention (α := α)
    (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim) (h1 := h1)
    s.inner wq wk wv wo x (mask := mask)

/--
Run a backward pass and return dense gradients for all leaves (eager backend).

See the unified version `Session.backwardDenseAll` for the public API.
-/
def backwardDenseAll {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  {sh : Shape} (out : _root_.Runtime.Autograd.Torch.TensorRef α sh) (seed : Tensor α sh) :
  IO (Array (Runtime.AnyTensor α)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.backwardDenseAll (α := α) (sh := sh) s.inner
    out seed

/-- Backward pass specialized to scalar losses (seed is implicitly `1`) (eager backend). -/
def backwardScalarDenseAll {α : Type} (s : EagerSession α) [Add α] [Zero α] [One α] [DecidableEq
  Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  (loss : _root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) :
  IO (Array (Runtime.AnyTensor α)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.backwardScalarDenseAll (α := α) s.inner loss

/--
Apply an SGD step to all learnable parameters given a dense gradient array (eager backend).

PyTorch analogy: `optimizer.step()` for an SGD optimizer, with gradients supplied explicitly.
-/
def sgdStepAll {α : Type} (s : EagerSession α)
  [Sub α] [Mul α] [Add α] [Zero α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  (lr : α) (grads : Array (Runtime.AnyTensor α)) : IO Unit :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.sgdStepAll (α := α) s.inner lr grads

end EagerSession

end TorchLean
end Autograd
end Runtime
