/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.TypedGraphSession
public import NN.Runtime.Autograd.TorchLean.Functional.ShapeOps

import Mathlib.Algebra.Order.Algebra

/-!
# Session

TorchLean unified imperative session.

## Session state

A `Session α` is TorchLean's runtime analogue of a PyTorch "training loop environment". It:
- owns a collection of *leaf tensors* (parameters and inputs),
- records an eager computation on a tape or builds typed SSA graph data,
- can run reverse-mode AD to produce gradients for all leaves, and
- can apply simple optimizer steps (e.g. SGD) in a session-style workflow.

TorchLean exposes a **single API** with two execution modes selected at construction time:
- `.eager`: a tape-backed runtime session (imperative autograd tape; useful for debugging and
  interactive examples),
- `.typedGraph`: a session that records shape-indexed graph data while building the runtime
  tape, then executes via `Runtime.Autograd.Torch.Internal.TypedGraphSession`.

Both modes use the same `Session` API; each operation dispatches through `Session.state`.

## Typical Training Loop (PyTorch Analogy)

Think of the following mapping (approximately):
- `Session.param` ~ create a `torch.nn.Parameter` (and later include it in a `state_dict`-like
  bundle).
- `Session.use` ~ read a parameter as a tensor in the current recording phase.
- `Session.input` ~ add a leaf tensor input (like feeding a batch tensor into the forward pass).
- `Session.resetTape` ~ start a fresh recording phase (closest in spirit to `optimizer.zero_grad()` +
  new forward).
- `Session.backwardScalarDenseAll` ~ `loss.backward()` (but returns gradients explicitly as an
  array).
- `Session.sgdStepAll` ~ `optimizer.step()` (dense helper; higher-level training lives in
  `NN.API.*`).
- `Session.detach` ~ `tensor.detach()` (cut the gradient edge at a value).

TorchLean does *not* store mutable `.grad` fields on each tensor ref; instead, gradients are
  returned
explicitly (see `grad`, `vjp`, and the `backward*DenseAll` functions).

## Non-Differentiable State (`NatRef`)

`NatRef` stores the seed and counter used by the explicit random stream. Non-differentiable tensor
inputs use the element-polymorphic data-input channel rather than a second session tensor type.

## Deterministic RNG (Session-Level)

`RngState` provides explicit, deterministic RNG state (closer to JAX PRNG keys than a global RNG).
`freshSeedIO` is a convenience for sampling an initial seed at the IO boundary, while the *core*
semantics remains seed-threaded and replayable.

## Connection To TorchLean IR / Graph Execution

In `.typedGraph` execution, the session records executable `GraphData` while building the tape.
Each call to `resetTape` starts a new recording phase. Callers that need one reusable artifact
should use `Runtime.Autograd.Torch.TypedGraph` directly; the high-level scalar trainer uses that
artifact and records its loss graph once.

The type-level context checks graph shapes. Correctness of a stored JVP or VJP is a separate claim,
proved only for operations connected to the proof-carrying `Proofs.Autograd.Algebra.Node` layer.

Practical note: the current `.typedGraph` implementation expects all leaves (tensor inputs/parameters
and `NatRef`s) to be created before any op nodes are recorded. For portability, allocate leaves and
initialize/split RNG up-front, then build the typed graph.

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
API; this type exists to support execution-mode dispatch (`SessionState.eager`).
-/
structure EagerSession (α : Type) where
  /-- inner. -/
  inner : _root_.Runtime.Autograd.Torch.Internal.EagerSession α

namespace EagerSession

/--
Create a new eager (tape-backed) session.

This corresponds to the `.eager` execution mode of `Session.new`.
-/
def new {α : Type} (opts : _root_.Runtime.Autograd.Torch.Options := {}) : IO (EagerSession α) := do
  let inner ← _root_.Runtime.Autograd.Torch.Internal.EagerSession.new (α := α) (opts := opts)
  pure { inner := inner }

/-- Reset the eager autograd tape and begin a fresh recording phase. -/
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
Use a parameter in the current eager recording.

PyTorch analogy: reading a parameter in `forward` (it becomes part of the autograd graph).
-/
def use {α : Type} (s : EagerSession α) {sh : Shape} [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.TensorTransfer α]
  (p : _root_.Runtime.Autograd.Torch.Param α sh) : IO (_root_.Runtime.Autograd.Torch.TensorRef α sh)
    :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.use (α := α) (sh := sh) s.inner p

/--
Add a tensor input leaf to the current graph.

`requiresGrad` controls whether this input is recorded as a differentiable leaf.
-/
def input {α : Type} (s : EagerSession α) {sh : Shape} [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.TensorTransfer α]
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

/--
Insert a constant tensor into the current graph.

PyTorch analogy: using a tensor literal/constant in the forward pass (as a leaf constant node).
-/
def const {α : Type} (s : EagerSession α) {sh : Shape} [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.TensorTransfer α]
  (v : Tensor α sh) (name : Option String := none) : IO (_root_.Runtime.Autograd.Torch.TensorRef α
    sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.const (α := α) (sh := sh) s.inner v (name :=
    name)

/-- Read the concrete value for a tensor ref (for logging/debugging). -/
def getValue {α : Type} (s : EagerSession α) {sh : Shape} [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.TensorTransfer α]
  (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO (Tensor α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.getValue (α := α) (sh := sh) s.inner x

/--
Detach a tensor ref from the tape (stop gradient flow through it).

PyTorch analogy: `x.detach()`.
-/
def detach {α : Type} (s : EagerSession α) {sh : Shape} [Context α] [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.TensorTransfer α]
    (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
    IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.detach (α := α) (sh := sh) s.inner x

/-- Elementwise addition on tensor refs (eager execution path). -/
def add {α : Type} (s : EagerSession α) [Add α] [DecidableEq Shape] {sh : Shape}
  (a b : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO (_root_.Runtime.Autograd.Torch.TensorRef
    α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.add (α := α) (sh := sh) s.inner a b

/-- Elementwise subtraction on tensor refs (eager execution path). -/
def sub {α : Type} (s : EagerSession α) [Sub α] [Zero α] [DecidableEq Shape] {sh : Shape}
  (a b : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO (_root_.Runtime.Autograd.Torch.TensorRef
    α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.sub (α := α) (sh := sh) s.inner a b

/-- Elementwise multiplication on tensor refs (eager execution path). -/
def mul {α : Type} (s : EagerSession α) [Mul α] [DecidableEq Shape] {sh : Shape}
  (a b : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO (_root_.Runtime.Autograd.Torch.TensorRef
    α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.mul (α := α) (sh := sh) s.inner a b

/-- Elementwise scaling by a scalar constant `c` (eager execution path). -/
def scale {α : Type} (s : EagerSession α) [Mul α] [DecidableEq Shape] {sh : Shape}
  [_root_.Runtime.Autograd.Torch.TensorTransfer α]
  (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) (c : α) : IO
    (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.scale (α := α) (sh := sh) s.inner x c

/-- Elementwise absolute value (eager execution path). -/
def abs {α : Type} (s : EagerSession α) [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.abs (α := α) (sh := sh) s.inner x

/-- Elementwise square root (eager execution path). -/
def sqrt {α : Type} (s : EagerSession α) [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.sqrt (α := α) (sh := sh) s.inner x

/-- Elementwise clamp to `[minVal, maxVal]` (eager execution path). -/
def clamp {α : Type} (s : EagerSession α) [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.TensorTransfer α]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) (minVal maxVal : α) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.clamp (α := α) (sh := sh) s.inner x minVal
    maxVal

/-- Elementwise maximum (eager execution path). -/
def max {α : Type} (s : EagerSession α) [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape]
  {sh : Shape} (a b : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.max (α := α) (sh := sh) s.inner a b

/-- Elementwise minimum (eager execution path). -/
def min {α : Type} (s : EagerSession α) [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  [DecidableEq Shape]
  {sh : Shape} (a b : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.min (α := α) (sh := sh) s.inner a b

/-- Matrix multiplication with broadcasted batch prefixes (eager execution path). -/
def matmul {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {batchA batchB batch : Shape} {m n p : Nat}
  [broadcastA : Shape.BroadcastTo batchA batch]
  [broadcastB : Shape.BroadcastTo batchB batch]
  (a : _root_.Runtime.Autograd.Torch.TensorRef α (batchA.concat [m, n]))
  (b : _root_.Runtime.Autograd.Torch.TensorRef α (batchB.concat [n, p])) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (batch.concat [m, p])) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.matmul (α := α) s.inner
    (batchA := batchA) (batchB := batchB) (batch := batch)
    (m := m) (n := n) (p := p) a b

/--
Concatenate along the outermost dimension (dimension 0) (eager execution path).

PyTorch analogy: `torch.cat([a, b], dim=0)`.
-/
def concatLeadingAxis {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {n m : Nat} {sh : Shape}
  (a : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n sh))
  (b : _root_.Runtime.Autograd.Torch.TensorRef α (.dim m sh)) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim (n + m) sh)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.concatLeadingAxis (α := α) s.inner (n := n) (m := m)
    (sh := sh) a b

/--
Slice a contiguous `[start, start+len)` range from dimension 0 (eager execution path).

PyTorch analogy: `x[start:start+len]` for the first dimension.
-/
def sliceLeadingAxisRange {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape]
  {n : Nat} {sh : Shape}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n sh)) (start len : Nat) (h : start + len ≤
    n) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim len sh)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.sliceLeadingAxisRange (α := α) s.inner (n := n) (sh :=
    sh) x start len h

/-- Apply max pooling over an arbitrary number of spatial axes. -/
def maxPool {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
    {d channels : Nat} {spatial kernel stride padding : Spec.Tensor Nat [d]}
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    (x : _root_.Runtime.Autograd.Torch.TensorRef α
      (Shape.ofList (channels :: spatial.toList))) :
    IO (_root_.Runtime.Autograd.Torch.TensorRef α
      (Shape.ofList
        (channels :: (Spec.poolOutSpatialPad spatial kernel stride padding).toList))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.maxPool (α := α) s.inner
    (d := d) (C := channels) (inSpatial := spatial)
    (kernel := kernel) (stride := stride) (padding := padding)
    (hKernel := hKernel) x

/-- Apply smooth max pooling over an arbitrary number of spatial axes. -/
def smoothMaxPool {α : Type} (s : EagerSession α) [Context α] [DecidableEq α]
    [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.TensorTransfer α]
    {d channels : Nat} {spatial kernel stride padding : Spec.Tensor Nat [d]}
    {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
    (x : _root_.Runtime.Autograd.Torch.TensorRef α
      (Shape.ofList (channels :: spatial.toList))) (beta : α) :
    IO (_root_.Runtime.Autograd.Torch.TensorRef α
      (Shape.ofList
        (channels :: (Spec.poolOutSpatialPad spatial kernel stride padding).toList))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.smoothMaxPool (α := α) s.inner
    (d := d) (C := channels) (inSpatial := spatial)
    (kernel := kernel) (stride := stride) (padding := padding)
    (hKernel := hKernel) x beta

/-- Apply average pooling over an arbitrary number of spatial axes. -/
def avgPool {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
    {d channels : Nat} {spatial kernel stride padding : Spec.Tensor Nat [d]}
    (hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0)
    (x : _root_.Runtime.Autograd.Torch.TensorRef α
      (Shape.ofList (channels :: spatial.toList))) :
    IO (_root_.Runtime.Autograd.Torch.TensorRef α
      (Shape.ofList
        (channels :: (Spec.poolOutSpatialPad spatial kernel stride padding).toList))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.avgPool (α := α) s.inner
    (d := d) (C := channels) (inSpatial := spatial)
    (kernel := kernel) (stride := stride) (padding := padding) hKernel x

/-- Elementwise ReLU activation (eager execution path). -/
def relu {α : Type} (s : EagerSession α)
  [Mul α] [Zero α] [Max α] [One α] [LT α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO
    (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.relu (α := α) (sh := sh) s.inner x

/-- Elementwise sigmoid activation (eager execution path). -/
def sigmoid {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO
    (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.sigmoid (α := α) (sh := sh) s.inner x

/-- Elementwise tanh activation (eager execution path). -/
def tanh {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO
    (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.tanh (α := α) (sh := sh) s.inner x

/-- Softmax along an explicitly selected tensor dimension (eager execution path). -/
def softmax {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
    {sh : Shape} (axis : Nat) [Shape.AxisInBounds axis sh]
    (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
    IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  let action : _root_.Runtime.Autograd.Torch.Internal.EagerM α
      (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
    F.softmax (m := _root_.Runtime.Autograd.Torch.Internal.EagerM α)
      (α := α) (s := sh) axis x
  action s.inner

/-- Stable log-softmax along an explicitly selected tensor dimension (eager execution path). -/
def logSoftmax {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
    {sh : Shape} (axis : Nat) [Shape.AxisInBounds axis sh]
    (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
    IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  let action : _root_.Runtime.Autograd.Torch.Internal.EagerM α
      (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
    F.logSoftmax (m := _root_.Runtime.Autograd.Torch.Internal.EagerM α)
      (α := α) (s := sh) axis x
  action s.inner

/-- Elementwise softplus activation (eager execution path). -/
def softplus {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO
    (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.softplus (α := α) (sh := sh) s.inner x

/-- Elementwise exponential (eager execution path). -/
def exp {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO
    (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.exp (α := α) (sh := sh) s.inner x

/-- Elementwise logarithm (eager execution path). -/
def log {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) : IO
    (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.log (α := α) (sh := sh) s.inner x

/-- Elementwise `safe_log` activation (`log(softplus(x) + ε)`) (eager execution path). -/
def safeLog {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.TensorTransfer α]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) (ε : α := Numbers.epsilon) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.safeLog (α := α) (sh := sh) s.inner x (ε :=
    ε)

/-- Sum-reduce a tensor to a scalar (eager execution path). -/
def sum {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.sum (α := α) (sh := sh) s.inner x

/-- Flatten a tensor into a 1D vector (eager execution path). -/
def flatten {α : Type} (s : EagerSession α) [Inhabited α] [DecidableEq Shape] {sh : Shape}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α [Spec.Shape.size sh]) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.flatten (α := α) (sh := sh) s.inner x

/--
Reshape a tensor, given a proof that the total number of elements is preserved (eager execution path).

PyTorch analogy: `x.reshape(...)` when the element count matches.
-/
def reshape {α : Type} (s : EagerSession α) [Inhabited α] [DecidableEq Shape] {sh1 sh2 : Shape}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α sh1) (h : Spec.Shape.size sh1 = Spec.Shape.size sh2) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh2) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.reshape (α := α) (sh1 := sh1) (sh2 := sh2)
    s.inner x h

/--
Generic "swap adjacent axes" view operation (eager execution path).

This is a shape-driven permutation helper used in some attention/transformer code.
-/
def swapAdjacentAtDepth {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape] {sh : Shape}
  (depth : Nat) (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (sh.swapAdjacentAtDepth depth)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.swapAdjacentAtDepth (α := α) (sh := sh)
    s.inner depth x

/-- Broadcast a tensor to a larger shape (eager execution path). -/
def broadcastTo {α : Type} (s : EagerSession α) [Inhabited α] [Add α] [Zero α] [DecidableEq Shape]
  {sh1 sh2 : Shape} (cb : Shape.CanBroadcastTo sh1 sh2) (x : _root_.Runtime.Autograd.Torch.TensorRef
    α sh1) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh2) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.broadcastTo (α := α) (sh1 := sh1) (sh2 := sh2)
    s.inner cb x

/-- Reduce-sum along an axis (eager execution path). -/
def reduceSum {α : Type} (s : EagerSession α) [Add α] [Zero α] [Inhabited α] [DecidableEq Shape]
  {sh : Shape} (axis : Nat) (x : _root_.Runtime.Autograd.Torch.TensorRef α sh)
  [valid : Shape.HasNonemptyAxis axis sh] [wf : Shape.WellFormed sh] :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (shapeAfterSum sh axis)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.reduceSum (α := α) (sh := sh) s.inner axis x

/-- Reduce-mean along an axis (eager execution path). -/
def reduceMean {α : Type} (s : EagerSession α) [Context α] [DecidableEq Shape]
  {sh : Shape} (axis : Nat) (x : _root_.Runtime.Autograd.Torch.TensorRef α sh)
  [valid : Shape.HasNonemptyAxis axis sh] [wf : Shape.WellFormed sh] :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (shapeAfterSum sh axis)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.reduceMean (α := α) (sh := sh) s.inner axis x

/-- Select one bounded coordinate from an arbitrary tensor axis. -/
def select {α : Type} (s : EagerSession α) [Zero α] [DecidableEq Shape]
    {shape : Shape} (axis : Nat) (x : _root_.Runtime.Autograd.Torch.TensorRef α shape)
    [Shape.AxisInBounds axis shape]
    (index : Fin (Shape.axisSize shape axis)) :
    IO (_root_.Runtime.Autograd.Torch.TensorRef α (shape.eraseAxis axis)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.select (α := α) s.inner axis x index

/-- Select several bounded coordinates from an arbitrary tensor axis. -/
def indexSelect {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
    {shape : Shape} (axis count : Nat) (x : _root_.Runtime.Autograd.Torch.TensorRef α shape)
    [Shape.AxisInBounds axis shape]
    (indices : Tensor (Fin (Shape.axisSize shape axis)) [count]) :
    IO (_root_.Runtime.Autograd.Torch.TensorRef α (shape.replaceAxis axis count)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.indexSelect (α := α) s.inner axis count
    x indices

/-- Add source slices into an arbitrary tensor axis at bounded coordinates. -/
def scatterAdd {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
    {shape : Shape} (axis count : Nat) (base : _root_.Runtime.Autograd.Torch.TensorRef α shape)
    [Shape.AxisInBounds axis shape]
    (source : _root_.Runtime.Autograd.Torch.TensorRef α (shape.replaceAxis axis count))
    (indices : Tensor (Fin (Shape.axisSize shape axis)) [count]) :
    IO (_root_.Runtime.Autograd.Torch.TensorRef α shape) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.scatterAdd (α := α) s.inner axis count
    base source indices

/--
Fully-connected (affine) layer on vectors: `y = w·x + b` (eager execution path).

PyTorch analogue: `torch.nn.functional.linear` (with weight shape `(outDim, inDim)`).
-/
def linear {α : Type} (s : EagerSession α) [Inhabited α] [Add α] [Mul α] [Zero α] [DecidableEq
  Shape]
  {inDim outDim : Nat}
  (w : _root_.Runtime.Autograd.Torch.TensorRef α [outDim, inDim])
  (b : _root_.Runtime.Autograd.Torch.TensorRef α [outDim])
  (x : _root_.Runtime.Autograd.Torch.TensorRef α [inDim]) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α [outDim]) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.linear (α := α) (inDim := inDim) (outDim :=
    outDim)
    s.inner w b x

/--
Mean squared error loss returning a scalar (eager execution path).

PyTorch analogue: `torch.nn.functional.mse_loss(..., reduction='mean')`.
-/
def mseLoss {α : Type} (s : EagerSession α)
  [Inhabited α] [Add α] [Sub α] [Mul α] [Div α] [Zero α] [One α] [Coe Nat α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.TensorTransfer α]
  {sh : Shape}
  (yhat target : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.mseLoss (α := α) (sh := sh) s.inner yhat
    target

/--
LayerNorm over a `seqLen × embedDim` tensor (eager execution path).

PyTorch analogue: `torch.nn.LayerNorm(embedDim)` applied per token.
-/
def layerNorm {α : Type} (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {seqLen embedDim : Nat} (h_seq_pos : seqLen > 0) (h_embed_pos : embedDim > 0)
  (x : _root_.Runtime.Autograd.Torch.TensorRef α [seqLen, embedDim])
  (gamma : _root_.Runtime.Autograd.Torch.TensorRef α [embedDim])
  (beta : _root_.Runtime.Autograd.Torch.TensorRef α [embedDim]) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α [seqLen, embedDim]) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.layerNorm (α := α)
    (seqLen := seqLen) (embedDim := embedDim) (h_seq_pos := h_seq_pos) (h_embed_pos := h_embed_pos)
    s.inner x gamma beta

/-- Batch normalization over every spatial axis of a channel-first tensor. -/
def batchNorm {α : Type} (s : EagerSession α) [Context α]
    [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
    {channels : Nat} {sSpatial : Shape}
    (hWellFormed : (Shape.dim channels sSpatial).wellFormed)
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim channels sSpatial))
  (gamma : _root_.Runtime.Autograd.Torch.TensorRef α [channels])
  (beta : _root_.Runtime.Autograd.Torch.TensorRef α [channels]) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim channels sSpatial)) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.batchNorm (α := α)
    (channels := channels) (sSpatial := sSpatial) s.inner hWellFormed x gamma beta

/--
N-D convolution over a channels-first tensor `(inC, spatial...)` (eager execution path).

PyTorch analogue: `torch.nn.functional.conv{d}d` specialized to a single sample.
-/
def conv {α : Type} (s : EagerSession α) [Context α]
  [DecidableEq Shape]
  {d inC outC : Nat}
  {kernel stride padding : Spec.Tensor Nat [d]}
  {inSpatial : Spec.Tensor Nat [d]}
  {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
  (w : _root_.Runtime.Autograd.Torch.TensorRef α (Shape.ofList (outC :: inC :: kernel.toList)))
  (b : _root_.Runtime.Autograd.Torch.TensorRef α [outC])
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (Shape.ofList (inC :: inSpatial.toList))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α
    (Shape.ofList (outC :: (Spec.convOutSpatial inSpatial kernel stride padding).toList))) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.conv (α := α)
    (d := d) (inC := inC) (outC := outC)
    (kernel := kernel) (stride := stride) (padding := padding) (inSpatial := inSpatial)
    (hInC := hInC) (hKernel := hKernel)
    s.inner w b x

/--
N-D transpose convolution over a channels-first tensor `(inC, spatial...)` (eager execution path).

PyTorch analogue: `torch.nn.functional.conv_transpose{d}d` specialized to a single sample.
-/
def convTranspose {α : Type} (s : EagerSession α) [Context α]
  [DecidableEq Shape]
  {d inC outC : Nat}
  {kernel stride padding : Spec.Tensor Nat [d]}
  {inSpatial : Spec.Tensor Nat [d]}
  {hInC : inC ≠ 0} {hKernel : ∀ i : Fin d, kernel.getScalar i ≠ 0}
  (w : _root_.Runtime.Autograd.Torch.TensorRef α (Shape.ofList (inC :: outC :: kernel.toList)))
  (b : _root_.Runtime.Autograd.Torch.TensorRef α [outC])
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
Multi-head self-attention (eager execution path).

This is the eager implementation used by the transformer examples (approximately analogous to
`torch.nn.MultiheadAttention` in self-attention mode).
-/
def multiHeadAttention {α : Type} (s : EagerSession α) [Context α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {n numHeads dModel headDim : Nat} (h1 : n ≠ 0)
  (wq : _root_.Runtime.Autograd.Torch.TensorRef α [dModel, numHeads * headDim])
  (wk : _root_.Runtime.Autograd.Torch.TensorRef α [dModel, numHeads * headDim])
  (wv : _root_.Runtime.Autograd.Torch.TensorRef α [dModel, numHeads * headDim])
  (wo : _root_.Runtime.Autograd.Torch.TensorRef α [numHeads * headDim, dModel])
  (x : _root_.Runtime.Autograd.Torch.TensorRef α [n, dModel])
  (mask : Option (Tensor Bool [n, n]) := none) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α [n, dModel]) :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.multiHeadAttention (α := α)
    (n := n) (numHeads := numHeads) (dModel := dModel) (headDim := headDim) (h1 := h1)
    s.inner wq wk wv wo x (mask := mask)

/--
Run a backward pass and return dense gradients for all leaves (eager execution path).

See the unified version `Session.backwardDenseAll` for the public API.
-/
def backwardDenseAll {α : Type} (s : EagerSession α) [Add α] [Zero α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.TensorTransfer α]
  {sh : Shape} (out : _root_.Runtime.Autograd.Torch.TensorRef α sh) (seed : Tensor α sh) :
  IO (Array (Spec.SomeTensor α)) := do
  s.inner.validateTensorRef out
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.backwardDenseAll (α := α) (sh := sh) s.inner
    out seed

/-- Backward pass specialized to scalar losses (seed is implicitly `1`) (eager execution path). -/
def backwardScalarDenseAll {α : Type} (s : EagerSession α) [Add α] [Zero α] [One α] [DecidableEq
  Shape]
  [_root_.Runtime.Autograd.Torch.TensorTransfer α]
  (loss : _root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) :
  IO (Array (Spec.SomeTensor α)) := do
  s.inner.validateTensorRef loss
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.backwardScalarDenseAll (α := α) s.inner loss

/--
Apply an SGD step to all learnable parameters given a dense gradient array (eager execution path).

PyTorch analogy: `optimizer.step()` for an SGD optimizer, with gradients supplied explicitly.
-/
def sgdStepAll {α : Type} (s : EagerSession α)
  [Sub α] [Mul α] [Add α] [Zero α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.TensorTransfer α]
  (lr : α) (grads : Array (Spec.SomeTensor α)) : IO Unit :=
  _root_.Runtime.Autograd.Torch.Internal.EagerSession.sgdStepAll (α := α) s.inner lr grads

end EagerSession

end TorchLean
end Autograd
end Runtime
