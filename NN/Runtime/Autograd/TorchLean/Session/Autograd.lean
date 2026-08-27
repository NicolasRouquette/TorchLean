/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.Session.Neural

/-!
Session-level autograd operations.

This module exposes backward and gradient-readback helpers for session tensors while preserving the
host/CUDA synchronization invariants maintained by the runtime.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor

namespace Session

/--
Run a backward pass and return a dense array of gradients for *all* leaf tensors.

This is the explicit dense-array version of calling backward and then reading every leaf gradient.
-/
def backwardDenseAll {α : Type} (s : Session α) [Add α] [Zero α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.TensorTransfer α]
  {sh : Shape} (out : _root_.Runtime.Autograd.Torch.TensorRef α sh) (seed : Tensor α sh) :
  IO (Array (_root_.Spec.SomeTensor α)) := do
  match s.state with
  | .eager sess =>
      sess.inner.validateTensorRef out
      EagerSession.backwardDenseAll (α := α) sess (sh := sh) out seed
  | .typedGraph sess =>
      sess.validateTensorRef out
      _root_.Runtime.Autograd.Torch.Internal.TypedGraphSession.backwardDenseAll (α := α) sess (sh := sh) out
        seed

namespace Internal

/--
Apply a gradient hook pointwise to a dense gradient array.

Invariant: the hook must preserve each gradient tensor's shape; we check this and throw if it
changes.
-/
def applyGradHook {α : Type}
    (grads : Array (_root_.Spec.SomeTensor α))
    (hook : Nat → _root_.Spec.SomeTensor α → IO (_root_.Spec.SomeTensor α)) :
    IO (Array (_root_.Spec.SomeTensor α)) := do
  let mut out : Array (_root_.Spec.SomeTensor α) := #[]
  for i in List.finRange grads.size do
    let g := grads[i]
    let g' ← hook i.1 g
    if h : g'.shape = g.shape then
      out := out.push ⟨g.shape, g'.cast h⟩
    else
      throw <| IO.userError <|
        s!"torchlean: grad hook changed shape at id={i.1} (expected {Shape.pretty g.shape}, got "
          ++ s!"{Shape.pretty g'.shape})"
  pure out

end Internal

/--
Backward pass with an optional gradient hook applied to the *dense* gradient array.

This is a runtime utility (similar in spirit to PyTorch hooks), not part of the proof semantics.
-/
def backwardDenseAllWithHook {α : Type} (s : Session α) [Add α] [Zero α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.TensorTransfer α]
    {sh : Shape} (out : _root_.Runtime.Autograd.Torch.TensorRef α sh) (seed : Tensor α sh)
    (hook : Nat → _root_.Spec.SomeTensor α → IO (_root_.Spec.SomeTensor α)) :
    IO (Array (_root_.Spec.SomeTensor α)) := do
  Internal.applyGradHook (α := α) (grads := (← backwardDenseAll (α := α) s (sh := sh) out seed))
    hook

/-- Backward pass for a scalar loss, returning the dense gradient array (seed is implicitly `1`). -/
def backwardScalarDenseAll {α : Type} (s : Session α) [Add α] [Zero α] [One α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.TensorTransfer α]
  (loss : _root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar) :
  IO (Array (_root_.Spec.SomeTensor α)) :=
  backwardDenseAll (α := α) s loss (Tensor.scalar (1 : α))

/-- `backwardScalarDenseAll` with a per-leaf gradient hook applied. -/
def backwardScalarDenseAllWithHook {α : Type} (s : Session α) [Add α] [Zero α] [One α] [DecidableEq
  Shape]
  [_root_.Runtime.Autograd.Torch.TensorTransfer α]
    (loss : _root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar)
    (hook : Nat → _root_.Spec.SomeTensor α → IO (_root_.Spec.SomeTensor α)) :
    IO (Array (_root_.Spec.SomeTensor α)) := do
  Internal.applyGradHook (α := α) (grads := (← backwardScalarDenseAll (α := α) s loss)) hook

/--
Extract the gradient for a particular tensor ref from a dense gradient array.

This is the non-mutating counterpart of reading `x.grad`.
-/
def grad {α : Type} (s : Session α) {sh : Shape} [DecidableEq Shape]
  (grads : Array (_root_.Spec.SomeTensor α)) (x : _root_.Runtime.Autograd.Torch.TensorRef α sh) :
  IO (Tensor α sh) := do
  match s.state with
  | .eager sess => sess.inner.validateTensorRef x
  | .typedGraph sess => sess.validateTensorRef x
  let gAny ← match grads[x.id]? with
    | some g => pure g
    | none => throw <| IO.userError "torchlean: gradient array out of bounds"
  if h : gAny.shape = sh then
    pure (gAny.cast h)
  else
    throw <| IO.userError
      s!"torchlean: grad shape mismatch (expected {Shape.pretty sh}, got {Shape.pretty gAny.shape})"

/-- Vector-Jacobian product: `vjp(out, seed)[x]`. -/
def vjp {α : Type} (s : Session α) [Add α] [Zero α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.TensorTransfer α]
    {shOut shX : Shape}
    (out : _root_.Runtime.Autograd.Torch.TensorRef α shOut)
    (seed : Tensor α shOut)
    (x : _root_.Runtime.Autograd.Torch.TensorRef α shX) :
    IO (Tensor α shX) := do
  let grads ← backwardDenseAll (α := α) s (sh := shOut) out seed
  grad (α := α) s (sh := shX) grads x

/-- Scalar-loss VJP with implicit seed `1`: `∇_x loss`. -/
def vjpScalar {α : Type} (s : Session α) [Add α] [Zero α] [One α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.TensorTransfer α]
    {shX : Shape}
    (loss : _root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar)
    (x : _root_.Runtime.Autograd.Torch.TensorRef α shX) :
    IO (Tensor α shX) := do
  let grads ← backwardScalarDenseAll (α := α) s loss
  grad (α := α) s (sh := shX) grads x

/-! ## Forward-mode: JVP -/

/--
Jacobian-vector product for a single leaf (typed graph execution only).

For eager sessions, use the typed graph execution if you need JVPs.
-/
def jvpLeaf {α : Type} (s : Session α) [Zero α] [DecidableEq Shape]
    {shOut shX : Shape}
    (out : _root_.Runtime.Autograd.Torch.TensorRef α shOut)
    (x : _root_.Runtime.Autograd.Torch.TensorRef α shX)
    (dx : Tensor α shX) :
    IO (Tensor α shOut) := do
  match s.state with
  | .eager _ =>
      throw <| IO.userError "torchlean: jvpLeaf is only supported for typed graph sessions"
  | .typedGraph sess =>
      _root_.Runtime.Autograd.Torch.Internal.TypedGraphSession.jvpLeaf (α := α) sess
        (shOut := shOut) (shX := shX) out x dx

/-- Scalar-loss JVP for a single leaf (typed graph execution only). -/
def jvpScalarLeaf {α : Type} (s : Session α) [Zero α] [DecidableEq Shape]
    (loss : _root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar)
    {shX : Shape} (x : _root_.Runtime.Autograd.Torch.TensorRef α shX) (dx : Tensor α shX) :
    IO α := do
  match s.state with
  | .eager _ =>
      throw <| IO.userError "torchlean: jvpScalarLeaf is only supported for typed graph sessions"
  | .typedGraph sess =>
      _root_.Runtime.Autograd.Torch.Internal.TypedGraphSession.jvpScalarLeaf (α := α) sess loss x dx

/-! ## Forward-mode: dense JVP (typed graph execution only) -/

/--
Jacobian-vector product with explicit tangents for all *leaf* tensors.

`dxs[i]` is the tangent for leaf `i` (same indexing as `grad`/`backwardDenseAll`).
-/
def jvpDenseAll {α : Type} (s : Session α) [Zero α] [DecidableEq Shape]
    {shOut : Shape}
    (out : _root_.Runtime.Autograd.Torch.TensorRef α shOut)
    (dxs : Array (_root_.Spec.SomeTensor α)) :
    IO (Tensor α shOut) := do
  match s.state with
  | .eager _ =>
      throw <| IO.userError "torchlean: jvpDenseAll is only supported for typed graph sessions"
  | .typedGraph sess =>
      _root_.Runtime.Autograd.Torch.Internal.TypedGraphSession.jvpDenseAll (α := α) sess (sh := shOut) out
        dxs

/--
Apply a dense SGD step to all learnable parameters.

This is an optimizer helper used by examples; for a higher-level API see
  `TorchLean.Trainer.Manual`.
-/
def sgdStepAll {α : Type} (s : Session α)
  [Sub α] [Mul α] [Add α] [Zero α] [DecidableEq Shape]
  [_root_.Runtime.Autograd.Torch.TensorTransfer α]
  (lr : α) (grads : Array (_root_.Spec.SomeTensor α)) : IO Unit := do
  match s.state with
  | .eager sess => EagerSession.sgdStepAll (α := α) sess lr grads
  | .typedGraph sess =>
      _root_.Runtime.Autograd.Torch.Internal.TypedGraphSession.sgdStepAll (α := α) sess lr grads

/-- Reset the tape, then run one fresh graph-building action. -/
def withFreshTape {α β : Type} (s : Session α) (act : IO β) : IO β := do
  resetTape (α := α) s
  act

/--
Build one scalar-loss graph, run backward on it, and apply a dense SGD step.

This is the "session-style training step" helper for imperative workflows that are lower-level than
`_root_.TorchLean.Trainer.Manual`, but should still avoid manual
`resetTape` / `backwardScalarDenseAll` / `sgdStepAll` wiring.
-/
def sgdStepScalarGraph {α : Type} (s : Session α)
    [Sub α] [Mul α] [Add α] [Zero α] [One α] [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.TensorTransfer α]
    (lr : α)
    (buildLoss : IO (_root_.Runtime.Autograd.Torch.TensorRef α Shape.scalar)) :
    IO α :=
  withFreshTape (α := α) s do
    let loss ← buildLoss
    let lossTensor ← getValue (α := α) s (sh := Shape.scalar) loss
    let grads ← backwardScalarDenseAll (α := α) s loss
    sgdStepAll (α := α) s lr grads
    pure (Tensor.item lossTensor)

/--
Apply a dense SGD step to all parameters after transforming gradients with a user hook.

The `hook` is applied to each gradient tensor (given its leaf index) and can implement common
training tricks like gradient clipping, normalization, or noise injection.

PyTorch analogy:
- tensor hooks (`Tensor.register_hook`) on gradients, or
- manually postprocessing gradients before calling `optimizer.step()`.
-/
def sgdStepAllWithHook {α : Type} (s : Session α)
    [Sub α] [Mul α] [Add α] [Zero α] [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.TensorTransfer α]
    (lr : α) (grads : Array (_root_.Spec.SomeTensor α))
    (hook : Nat → _root_.Spec.SomeTensor α → IO (_root_.Spec.SomeTensor α)) : IO Unit := do
  sgdStepAll (α := α) s lr (← Internal.applyGradHook (α := α) grads hook)

end Session

end TorchLean
end Autograd
end Runtime
