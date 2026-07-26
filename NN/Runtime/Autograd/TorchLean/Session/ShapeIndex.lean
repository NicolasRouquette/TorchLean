/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.Session.Ops

/-!
# Session Shape and Index Operations

This file contains the session-level operations that preserve or rearrange tensor shape: activation
helpers, reshapes, indexing, gathers, broadcasts, and reductions. Each operation dispatches through
the same eager/compiled session boundary as the lower-level tensor ops.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor

namespace Session

/--
Rectified Linear Unit (ReLU) activation.

This is the pointwise nonlinearity $\operatorname{ReLU}(x)=\max(x,0)$, recorded as part of the
session’s autograd
graph.

PyTorch analogy: `torch.relu(x)` / `torch.nn.functional.relu(x)`.
-/
def relu {α : Type} (s : Session α)
  [Mul α] [Add α] [Zero α] [Max α] [One α] [LT α]
  [DecidableRel ((· > ·) : α → α → Prop)] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.impl with
  | .eager sess => EagerSession.relu (α := α) sess (sh := sh) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.relu (α := α) sess (sh := sh) x

/--
Sigmoid (logistic) activation, applied pointwise.

PyTorch analogy: `torch.sigmoid(x)`.
-/
def sigmoid {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.impl with
  | .eager sess => EagerSession.sigmoid (α := α) sess (sh := sh) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.sigmoid (α := α) sess (sh := sh) x

/--
Hyperbolic tangent activation, applied pointwise.

PyTorch analogy: `torch.tanh(x)`.
-/
def tanh {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.impl with
  | .eager sess => EagerSession.tanh (α := α) sess (sh := sh) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.tanh (α := α) sess (sh := sh) x

/--
Softmax along the last axis (recursing over outer dimensions), shape-preserving.

This matches the spec-layer `Activation.softmax_spec` and uses a standard VJP implementation in the
backend (so we do not materialize an explicit Jacobian).

PyTorch analogy: `torch.softmax(x, dim=-1)`.
-/
def softmax {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.impl with
  | .eager sess => EagerSession.softmax (α := α) sess (sh := sh) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.softmax (α := α) sess (sh := sh) x

/--
Stable log-softmax along the last axis.

PyTorch analogy: `torch.nn.functional.log_softmax(x, dim=-1)`.
-/
def logSoftmax {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.impl with
  | .eager sess => EagerSession.logSoftmax (α := α) sess (sh := sh) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.logSoftmax (α := α) sess (sh := sh) x

/--
Softplus activation, applied pointwise:
$\operatorname{softplus}(x)=\log(1+\exp x)$.

PyTorch analogy: `torch.nn.functional.softplus(x)`.
-/
def softplus {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.impl with
  | .eager sess => EagerSession.softplus (α := α) sess (sh := sh) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.softplus (α := α) sess (sh := sh) x

/--
Elementwise exponential.

PyTorch analogy: `torch.exp(x)`.
-/
def exp {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.impl with
  | .eager sess => EagerSession.exp (α := α) sess (sh := sh) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.exp (α := α) sess (sh := sh) x

/--
Elementwise natural logarithm.

PyTorch analogy: `torch.log(x)`.

If you need a total (always-defined) "log-like" surrogate without positivity side conditions, see
`safe_log`.
-/
def log {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.impl with
  | .eager sess => EagerSession.log (α := α) sess (sh := sh) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.log (α := α) sess (sh := sh) x

/--
Elementwise safe-log surrogate:
$\operatorname{safeLog}(x;\varepsilon)=\log(\operatorname{softplus}(x)+\varepsilon)$.

We use this when we want something log-like but would rather not carry side conditions about inputs
being strictly positive.

PyTorch analogy: `torch.log(torch.nn.functional.softplus(x) + eps)`.
-/
def safeLog {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  {sh : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) (ε : α := Numbers.epsilon) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh) := do
  match s.impl with
  | .eager sess => EagerSession.safeLog (α := α) sess (sh := sh) x (ε := ε)
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.safeLog (α := α) sess (sh := sh) x (ε := ε)

/--
Sum-reduce all elements of a tensor to a scalar.

PyTorch analogy: `x.sum()` (with no `dim` argument).
-/
def sum {α : Type} (s : Session α) [Context α] [DecidableEq Shape] {sh : Shape}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) := do
  match s.impl with
  | .eager sess =>
      EagerSession.sum (α := α) sess (sh := sh) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.sum (α := α) sess (sh := sh) x

/--
Flatten a tensor to a 1D vector of length `Spec.Shape.size sh`.

PyTorch analogy: `torch.flatten(x)` or `x.reshape(-1)`.
-/
def flatten {α : Type} (s : Session α) [Inhabited α] [Zero α] [DecidableEq Shape] {sh : Shape}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim (Spec.Shape.size sh) .scalar)) := do
  match s.impl with
  | .eager sess => EagerSession.flatten (α := α) sess (sh := sh) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.flatten (α := α) sess (sh := sh) x

/--
Reshape a tensor without changing the number of elements.

The proof `h : Spec.Shape.size sh1 = Spec.Shape.size sh2` plays the role of PyTorch’s runtime check performed
by `reshape`/`view`.

PyTorch analogy: `x.reshape(new_shape)` (when the element count matches).
-/
def reshape {α : Type} (s : Session α) [Inhabited α] [Zero α] [DecidableEq Shape] {sh1 sh2 : Shape}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α sh1) (h : Spec.Shape.size sh1 = Spec.Shape.size sh2) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh2) := do
  match s.impl with
  | .eager sess => EagerSession.reshape (α := α) sess (sh1 := sh1) (sh2 := sh2) x h
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.reshape (α := α) sess
        (sh1 := sh1) (sh2 := sh2) x h

/--
Transpose a rank-2 tensor (matrix transpose): `m×n → n×m`.

PyTorch analogy: `x.transpose(0, 1)` (or `x.T` for 2D tensors).
-/
def transpose2d {α : Type} (s : Session α) [Zero α] [DecidableEq Shape] {m n : Nat}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim m (.dim n .scalar))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim n (.dim m .scalar))) := do
  match s.impl with
  | .eager sess => EagerSession.transpose2d (α := α) sess (m := m) (n := n) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.transpose2d (α := α) sess (m := m) (n := n) x

/--
Permute a rank-3 tensor by moving the first axis to the end: `(a, b, c) → (b, c, a)`.

PyTorch analogy: `x.permute(1, 2, 0)`.
-/
def transpose3dFirstToLast {α : Type} (s : Session α) [Zero α] [DecidableEq Shape] {a b c : Nat}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim a (.dim b (.dim c .scalar)))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim b (.dim c (.dim a .scalar)))) := do
  match s.impl with
  | .eager sess => EagerSession.transpose3dFirstToLast (α := α) sess (a := a) (b := b) (c := c) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.transpose3dFirstToLast (α := α) sess
        (a := a) (b := b) (c := c) x

/--
Permute a rank-3 tensor by moving the last axis to the front: `(a, b, c) → (c, a, b)`.

PyTorch analogy: `x.permute(2, 0, 1)`.
-/
def transpose3dLastToFirst {α : Type} (s : Session α) [Zero α] [DecidableEq Shape] {a b c : Nat}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim a (.dim b (.dim c .scalar)))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim c (.dim a (.dim b .scalar)))) := do
  match s.impl with
  | .eager sess => EagerSession.transpose3dLastToFirst (α := α) sess (a := a) (b := b) (c := c) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.transpose3dLastToFirst (α := α) sess
        (a := a) (b := b) (c := c) x

/--
Swap the last two axes of a rank-3 tensor: `(a, b, c) → (a, c, b)`.

PyTorch analogy: `x.permute(0, 2, 1)`.
-/
def transpose3dLastTwo {α : Type} (s : Session α) [Zero α] [DecidableEq Shape] {a b c : Nat}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim a (.dim b (.dim c .scalar)))) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim a (.dim c (.dim b .scalar)))) := do
  match s.impl with
  | .eager sess => EagerSession.transpose3dLastTwo (α := α) sess (a := a) (b := b) (c := c) x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.transpose3dLastTwo (α := α) sess (a := a)
        (b := b) (c := c) x

/--
Generic "swap adjacent axes" view operation.

This is a shape-driven permutation helper used in some attention/transformer code.
-/
def swapAdjacentAtDepth {α : Type} (s : Session α) [Context α] [DecidableEq Shape] {sh : Shape}
  (depth : Nat) (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (sh.swapAdjacentAtDepth depth)) := do
  match s.impl with
  | .eager sess => EagerSession.swapAdjacentAtDepth (α := α) sess (sh := sh) depth x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.swapAdjacentAtDepth (α := α) sess
        (sh := sh) depth x

/-- Broadcast a tensor to a larger shape (dispatches to eager vs compiled backend). -/
def broadcastTo {α : Type} (s : Session α) [Inhabited α] [Add α] [Zero α] [DecidableEq Shape]
  {sh1 sh2 : Shape} (cb : Shape.CanBroadcastTo sh1 sh2) (x : _root_.Runtime.Autograd.Torch.TensorRef
    α sh1) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α sh2) := do
  match s.impl with
  | .eager sess => EagerSession.broadcastTo (α := α) sess (sh1 := sh1) (sh2 := sh2) cb x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.broadcastTo (α := α) sess (sh1 := sh1)
        (sh2 := sh2) cb x

/-- Reduce-sum along an axis (dispatches to eager vs compiled backend). -/
def reduceSum {α : Type} (s : Session α) [Add α] [Zero α] [Inhabited α] [DecidableEq Shape]
  {sh : Shape} (axis : Nat) [valid : Shape.valid_axis_inst axis sh] [wf : Shape.WellFormed sh]
  (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (shapeAfterSum sh axis)) := do
  match s.impl with
  | .eager sess => EagerSession.reduceSum (α := α) sess (sh := sh) axis x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.reduceSum (α := α) sess (sh := sh) axis x

/-- Reduce-mean along an axis (dispatches to eager vs compiled backend). -/
def reduceMean {α : Type} (s : Session α) [Context α] [DecidableEq Shape]
  {sh : Shape} (axis : Nat) [valid : Shape.valid_axis_inst axis sh] [wf : Shape.WellFormed sh]
  (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (shapeAfterSum sh axis)) := do
  match s.impl with
  | .eager sess => EagerSession.reduceMean (α := α) sess (sh := sh) axis x
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.reduceMean (α := α) sess (sh := sh) axis x

/-- Gather a scalar from a vector at a `Fin` index (dispatches to eager vs compiled backend). -/
def gatherScalar {α : Type} (s : Session α) [Zero α] [DecidableEq Shape]
  {n : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n .scalar)) (i : Fin n) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) := do
  match s.impl with
  | .eager sess => EagerSession.gatherScalar (α := α) sess (n := n) x i
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.gatherScalar (α := α) sess (n := n) x i

/-- Gather a row from a matrix at a `Fin` index (dispatches to eager vs compiled backend). -/
def gatherRow {α : Type} (s : Session α) [Zero α] [DecidableEq Shape]
  {rows cols : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim rows (.dim cols .scalar)))
    (i : Fin rows) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim cols .scalar)) := do
  match s.impl with
  | .eager sess => EagerSession.gatherRow (α := α) sess (rows := rows) (cols := cols) x i
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.gatherRow (α := α) sess (rows := rows)
        (cols := cols) x i

/-- Gather a scalar from a vector using a `NatRef` index (dispatches to eager vs compiled backend).
  -/
def gatherScalarRef {α : Type} (s : Session α) [Zero α] [DecidableEq Shape]
  {n : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n .scalar)) (i :
    _root_.Runtime.Autograd.Torch.NatRef) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) := do
  match s.impl with
  | .eager sess => EagerSession.gatherScalarRef (α := α) sess (n := n) x i
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.gatherScalarRef (α := α) sess (n := n) x i

/-- Gather a row from a matrix using a `NatRef` index (dispatches to eager vs compiled backend). -/
def gatherRowRef {α : Type} (s : Session α) [Zero α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
  {rows cols : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim rows (.dim cols .scalar)))
    (i : _root_.Runtime.Autograd.Torch.NatRef) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim cols .scalar)) := do
  match s.impl with
  | .eager sess => EagerSession.gatherRowRef (α := α) sess (rows := rows) (cols := cols) x i
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.gatherRowRef (α := α) sess (rows := rows)
        (cols := cols) x i

/-- Gather a scalar using a raw `Nat` index (dispatches to eager vs compiled backend). -/
def gatherScalarNat {α : Type} (s : Session α) [Zero α] [DecidableEq Shape]
  {n : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n .scalar)) (i : Nat) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) := do
  match s.impl with
  | .eager sess => EagerSession.gatherScalarNat (α := α) sess (n := n) x i
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.gatherScalarNat (α := α) sess (n := n) x i

/-- Gather a vector of entries using an index tensor (dispatches to eager vs compiled backend). -/
def gatherVecNat {α : Type} (s : Session α) [Add α] [Zero α] [DecidableEq Shape]
  {n k : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n .scalar)) (idx : Tensor Nat
    (.dim k .scalar)) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim k .scalar)) := do
  match s.impl with
  | .eager sess => EagerSession.gatherVecNat (α := α) sess (n := n) (k := k) x idx
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.gatherVecNat (α := α) sess
        (n := n) (k := k) x idx

/-- Gather multiple rows using an index tensor (dispatches to eager vs compiled backend). -/
def gatherRowsNat {α : Type} (s : Session α) [Add α] [Zero α] [DecidableEq Shape]
  {rows cols k : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim rows (.dim cols
    .scalar)))
  (idx : Tensor Nat (.dim k .scalar)) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim k (.dim cols .scalar))) := do
  match s.impl with
  | .eager sess =>
      EagerSession.gatherRowsNat (α := α) sess (rows := rows) (cols := cols) (k := k) x idx
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.gatherRowsNat (α := α) sess
        (rows := rows) (cols := cols) (k := k) x idx

/-- `gather_vec_nat`, but indices are stored in a `NatVecRef` leaf (dispatches to eager vs compiled
  backend). -/
def gatherVecRef {α : Type} (s : Session α) [Add α] [Zero α] [DecidableEq Shape]
  {n k : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n .scalar))
  (idx : _root_.Runtime.Autograd.Torch.NatVecRef k) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim k .scalar)) := do
  match s.impl with
  | .eager sess => EagerSession.gatherVecRef (α := α) sess (n := n) (k := k) x idx
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.gatherVecRef (α := α) sess
        (n := n) (k := k) x idx

/-- `gather_rows_nat`, but indices are stored in a `NatVecRef` leaf (dispatches to eager vs compiled
  backend). -/
def gatherRowsRef {α : Type} (s : Session α) [Add α] [Zero α] [DecidableEq Shape]
  {rows cols k : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim rows (.dim cols
    .scalar)))
  (idx : _root_.Runtime.Autograd.Torch.NatVecRef k) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim k (.dim cols .scalar))) := do
  match s.impl with
  | .eager sess =>
      EagerSession.gatherRowsRef (α := α) sess (rows := rows) (cols := cols) (k := k) x idx
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.gatherRowsRef (α := α) sess (rows := rows)
        (cols := cols) (k := k) x idx

/-- Scatter-add into a vector at a `Fin` index (dispatches to eager vs compiled backend). -/
def scatterAddVec {α : Type} (s : Session α) [Add α] [Zero α] [DecidableEq Shape]
  {n : Nat} (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim n .scalar))
  (v : _root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) (i : Fin n) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim n .scalar)) := do
  match s.impl with
  | .eager sess => EagerSession.scatterAddVec (α := α) sess (n := n) x v i
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.scatterAddVec (α := α) sess (n := n) x v i

/-- Scatter-add into a matrix row at a `Fin` index (dispatches to eager vs compiled backend). -/
def scatterAddRow {α : Type} (s : Session α) [Add α] [Zero α] [DecidableEq Shape]
  {rows cols : Nat}
  (x : _root_.Runtime.Autograd.Torch.TensorRef α (.dim rows (.dim cols .scalar)))
  (v : _root_.Runtime.Autograd.Torch.TensorRef α (.dim cols .scalar)) (i : Fin rows) :
  IO (_root_.Runtime.Autograd.Torch.TensorRef α (.dim rows (.dim cols .scalar))) := do
  match s.impl with
  | .eager sess => EagerSession.scatterAddRow (α := α) sess (rows := rows) (cols := cols) x v i
  | .compiled sess =>
      _root_.Runtime.Autograd.Torch.Internal.SessionIR.scatterAddRow (α := α) sess (rows := rows)
        (cols := cols) x v i

end Session

end TorchLean
end Autograd
end Runtime
