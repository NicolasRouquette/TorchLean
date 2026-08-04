/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Utils
public import NN.Runtime.Autograd.TorchLean.Backend
public import NN.Runtime.Autograd.TorchLean.Training

import Mathlib.Algebra.Order.Algebra

/-!
# Module

TorchLean module wrappers with PyTorch-style ergonomics.

TorchLean already provides the core ingredients:
- a small `Ops` interface, so you write a model once and run it on different backends;
- `scalarTrainer`, which builds an eager or compiled training loop for scalar losses.

This file adds a thin “`nn.Module`-style” wrapper so users can:
- package **initial parameters** + a **loss definition** as a single object,
- instantiate it under a chosen backend (`.eager` / `.compiled`),
- call `forward / backward / step / params` with a small, consistent API.

Important: dtype selection is handled in `TorchLean.Runtime.DType` (because it picks the Lean type `α`).
The module definitions here are **polymorphic in `α`**, so the same module can be:
- used in executables with `Float` / `IEEE32Exec`, or
- instantiated at `ℝ` in proofs (noncomputable; not for `IO` execution).
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor
open Proofs.Autograd.Algebra

/-! ## Small helpers -/

namespace Module

/--
Cast a Float tensor to a backend scalar type `α` by mapping a scalar cast function.

This is mainly used to turn `tensorOfList!`-authored Float initializers into `Float`/`IEEE32Exec`/etc.
-/
def castTensor {α : Type} (cast : Float → α) {s : Shape} (t : Tensor Float s) : Tensor α s :=
  Spec.mapTensor cast t

/-- List-shaped version of `castTensor` for TorchLean's `TList` parameter bundles. -/
def castTList {α : Type} (cast : Float → α) : {ss : List Shape} → Torch.TList Float ss → Torch.TList
  α ss
  | [], .nil => .nil
  | _s :: ss, .cons x xs => .cons (castTensor cast x) (castTList (cast := cast) (ss := ss) xs)

/-! ## Runtime Float Initializers -/

namespace RuntimeInit

/--
Runtime initializer for a Float parameter.

The usual `ScalarModuleDef.initParams` path stores initializers as typed Lean tensors. That is the
right representation when the initial value itself is part of the Lean object being inspected.
For large Float runs, it is better to allocate runtime storage from a compact initialization scheme
and synchronize the host tensor only when parameters are explicitly read back.

The design mirrors the storage-first APIs used by mainstream runtimes:

- PyTorch exposes in-place initializers such as `torch.nn.init.uniform_`,
  `torch.nn.init.xavier_uniform_`, and `torch.nn.init.kaiming_uniform_` for already-allocated
  tensors: `https://pytorch.org/docs/stable/nn.init.html`.
- PyTorch's meta-device / `to_empty` path separates "module structure exists" from "real storage is
  materialized", after which users explicitly initialize parameters:
  `https://docs.pytorch.org/docs/main/meta.html`.

TorchLean keeps the semantic parameter type (`Tensor Float s`) available, but this runtime path lets
CPU/CUDA execution initialize real storage directly.
-/
inductive FloatInit where
  /-- Fill with zeros. PyTorch analogue: `torch.nn.init.zeros_`. -/
  | zeros
  /-- Fill with ones. PyTorch analogue: `torch.nn.init.ones_`. -/
  | ones
  /-- Uniform distribution over `[lo, hi)`, using TorchLean's deterministic runtime RNG. -/
  | uniform (lo hi : Float) (seed : Nat := 0)
  /-- Normal distribution with explicit mean and standard deviation. -/
  | normal (mean std : Float) (seed : Nat := 0)
  /-- Xavier/Glorot uniform with explicit fan-in and fan-out. -/
  | xavierUniform (fanIn fanOut : Nat) (seed : Nat := 0)
  /-- Kaiming/He uniform with explicit fan-in. -/
  | kaimingUniform (fanIn : Nat) (seed : Nat := 0)
  /-- Exact row-major payload. Used for imported checkpoints or generated tensors. -/
  | flat (values : FloatArray)

/-- Translate a proof-visible initializer scheme into its storage-first runtime form. -/
def FloatInit.ofScheme (scheme : Torch.Init.Scheme) (seed : Nat := 0) : FloatInit :=
  match scheme with
  | .zeros => .zeros
  | .ones => .ones
  | .uniform lo hi => .uniform lo hi seed
  | .normal mean std => .normal mean std seed
  | .xavierUniform fanIn fanOut => .xavierUniform fanIn fanOut seed
  | .kaimingUniform fanIn => .kaimingUniform fanIn seed

/--
A shape-indexed initialization plan.

This is the typed runtime-initialization API for modules with a known parameter shape list.  It is
the initialization analogue of `TList`: the type says there is exactly one initializer
for each parameter shape, in the same order.  That removes the annoying runtime failure mode where a
plain list is one element too short or too long.

The initializers themselves are runtime schemes rather than proof objects.  The proof layer story
is still the ordinary `Tensor Float s` parameter value; this plan only controls how the executable
Float runtime materializes those tensors on CPU or CUDA.
-/
inductive Plan : List Shape → Type where
  /-- No parameters, no initializers. -/
  | nil : Plan []
  /-- Initializer for the head parameter, followed by the plan for the remaining parameters. -/
  | cons {s : Shape} {ss : List Shape} (init : FloatInit) (rest : Plan ss) : Plan (s :: ss)

namespace Plan

/-- Concatenate two shape-indexed initializer plans. -/
def append : {ss₁ ss₂ : List Shape} → Plan ss₁ → Plan ss₂ → Plan (ss₁ ++ ss₂)
  | [], _, .nil, ys => ys
  | _ :: _, _, .cons x xs, ys => .cons x (append xs ys)

/-- Forget the shape index when interoperating with list-based callers. -/
def toList : {ss : List Shape} → Plan ss → List FloatInit
  | [], .nil => []
  | _ :: _, .cons init rest => init :: toList rest

/--
The type index is not decorative: forgetting a `Plan ss` to a list produces exactly `ss.length`
initializers.  This checked fact lets the runtime API avoid the usual
"initializer list does not match parameter list" class of bugs once a plan has been built.
-/
theorem length_toList : {ss : List Shape} → (plan : Plan ss) → plan.toList.length = ss.length
  | [], .nil => rfl
  | _ :: _, .cons _ rest => by
      simp [toList, length_toList rest]

/--
Recover a shape-indexed plan from a plain list.

List-based callers still enter through this boundary, but the runtime converts them immediately
into the shape-indexed representation before touching any parameters.
-/
def ofList? : (ss : List Shape) → List FloatInit → Except String (Plan ss)
  | [], [] => .ok .nil
  | [], _ :: _ => .error "torch.runtimeInit: initializer list longer than parameter list"
  | _ :: _, [] => .error "torch.runtimeInit: initializer list shorter than parameter list"
  | _ :: ss, init :: rest => do
      let restPlan ← ofList? ss rest
      pure (.cons init restPlan)

end Plan

/-- Product of a list of dimensions, used for convolutional receptive-field sizes. -/
def dimProduct (xs : List Nat) : Nat :=
  xs.foldl (fun acc x => acc * x) 1

/--
Infer `(fanIn, fanOut)` from a parameter shape using the common linear/conv convention.

For a matrix shaped `[out, in]`, this returns `(in, out)`. For convolution-like weights shaped
`[outChannels, inChannels, k1, ..., kd]`, it returns:

$$
\begin{aligned}
\operatorname{fanIn}
  &=\operatorname{inChannels}\,k_1\cdots k_d,\\
\operatorname{fanOut}
  &=\operatorname{outChannels}\,k_1\cdots k_d.
\end{aligned}
$$

This is the same fan convention documented by PyTorch's Xavier/Kaiming initialization utilities.
-/
def fanInOut? (s : Shape) : Option (Nat × Nat) :=
  match Shape.toList s with
  | outDim :: inDim :: spatial =>
      let receptive := dimProduct spatial
      some (inDim * receptive, outDim * receptive)
  | _ => none

/-- Build a Xavier initializer by deriving fan-in/fan-out from a Linear/Conv-style weight shape. -/
def xavierUniformForShape (s : Shape) (seed : Nat := 0) : Except String FloatInit :=
  match fanInOut? s with
  | some (fanIn, fanOut) => .ok (.xavierUniform fanIn fanOut seed)
  | none =>
      .error s!"torch.runtimeInit: Xavier initialization expects at least 2 dimensions, got {Shape.pretty s}"

/-- Build a Kaiming initializer by deriving fan-in from a Linear/Conv-style weight shape. -/
def kaimingUniformForShape (s : Shape) (seed : Nat := 0) : Except String FloatInit :=
  match fanInOut? s with
  | some (fanIn, _fanOut) => .ok (.kaimingUniform fanIn seed)
  | none =>
      .error s!"torch.runtimeInit: Kaiming initialization expects at least 2 dimensions, got {Shape.pretty s}"

/-- Convenience initializer for a matrix weight stored as `[outDim, inDim]`. -/
def xavierLinearWeight (outDim inDim : Nat) (seed : Nat := 0) : FloatInit :=
  .xavierUniform inDim outDim seed

/-- Convenience initializer for a ReLU-style matrix weight stored as `[outDim, inDim]`. -/
def kaimingLinearWeight (_outDim inDim : Nat) (seed : Nat := 0) : FloatInit :=
  .kaimingUniform inDim seed

/--
Deterministic unit sample shared with the pure tensor initializer.

Calling the canonical sampler here keeps CPU storage-first initialization equal to the semantic
tensor initializer. The CUDA path uses the same SplitMix64 key/index sequence, materialized as
float32 device values.
-/
def unitAt (seed idx : Nat) : Float :=
  Torch.Init.Internal.rand01 seed idx

/-- Scalar value generated by a `FloatInit` at a row-major flat index. -/
def sampleAt : FloatInit → Nat → Float
  | .zeros, _ => 0.0
  | .ones, _ => 1.0
  | .uniform lo hi seed, idx => lo + unitAt seed idx * (hi - lo)
  | .normal mean std seed, idx =>
      Torch.Init.Internal.sampleAt (.normal mean std) seed idx
  | .xavierUniform fanIn fanOut seed, idx =>
      let denom := Float.ofNat fanIn + Float.ofNat fanOut
      let limit := Float.sqrt (6.0 / denom)
      (-limit) + unitAt seed idx * (2.0 * limit)
  | .kaimingUniform fanIn seed, idx =>
      let limit := Float.sqrt (6.0 / Float.ofNat fanIn)
      (-limit) + unitAt seed idx * (2.0 * limit)
  | .flat values, idx => values.get! idx

/--
Materialize an initializer as a host `FloatArray`.

CPU execution uses this path directly. CUDA uses it only when the initializer already is an exact
flat payload; analytic initializers such as uniform/Xavier/Kaiming are created on the runtime side.
-/
def floatArrayOf (n : Nat) (init : FloatInit) : IO FloatArray := do
  match init with
  | .flat values =>
      if values.size = n then
        pure values
      else
        throw <| IO.userError
          s!"torch.runtimeInit: flat initializer length mismatch (expected {n}, got {values.size})"
  | _ =>
      let mut out : Array Float := Array.mkEmpty n
      for i in [0:n] do
        out := out.push (sampleAt init i)
      pure (FloatArray.mk out)

/-- Checked conversion to the current CUDA buffer API's `UInt32` element count. -/
def natToU32Checked (ctx : String) (n : Nat) : IO UInt32 := do
  let u := UInt32.ofNat n
  if u.toNat = n then
    pure u
  else
    throw <| IO.userError s!"{ctx}: tensor too large for CUDA buffer API ({n} elements)"

/--
Allocate a CUDA buffer filled with `U(lo, hi)`.

The implementation keeps all element generation on the runtime side: first create a CUDA uniform
buffer in `[0,1)`, then perform `lo + (hi-lo) * u` with CUDA buffer ops.
-/
def cudaUniformBuffer (n : Nat) (lo hi : Float) (seed : Nat) :
    IO _root_.Runtime.Autograd.Cuda.Buffer := do
  let n32 ← natToU32Checked "torch.runtimeInit" n
  let key := _root_.Runtime.Autograd.TorchLean.Random.keyOf seed 0
  let u := _root_.Runtime.Autograd.Cuda.Buffer.randUniform n32 key
  let shift := _root_.Runtime.Autograd.Cuda.Buffer.full n32 lo
  let out := _root_.Runtime.Autograd.Cuda.Buffer.axpy shift u (hi - lo)
  pure <| _root_.Runtime.Autograd.Cuda.Buffer.releaseThen u <|
    _root_.Runtime.Autograd.Cuda.Buffer.releaseThen shift out

/--
Allocate a CUDA buffer for a `FloatInit`.

For analytic schemes (`zeros`, `ones`, `uniform`, `xavierUniform`, `kaimingUniform`), this avoids
building a large nested Lean tensor. For `.flat`, the caller already supplied the exact payload, so
we upload that payload directly.
-/
def cudaBufferOf (n : Nat) (init : FloatInit) : IO _root_.Runtime.Autograd.Cuda.Buffer := do
  let n32 ← natToU32Checked "torch.runtimeInit" n
  match init with
  | .zeros => pure <| _root_.Runtime.Autograd.Cuda.Buffer.zeros n32
  | .ones => pure <| _root_.Runtime.Autograd.Cuda.Buffer.full n32 1.0
  | .uniform lo hi seed =>
      cudaUniformBuffer n lo hi seed
  | .normal mean std seed =>
      let key := _root_.Runtime.Autograd.TorchLean.Random.keyOf seed 0
      pure <| _root_.Runtime.Autograd.Cuda.Buffer.randNormal n32 mean std key
  | .xavierUniform fanIn fanOut seed =>
      let denom := Float.ofNat fanIn + Float.ofNat fanOut
      let limit := Float.sqrt (6.0 / denom)
      cudaUniformBuffer n (-limit) limit seed
  | .kaimingUniform fanIn seed =>
      let limit := Float.sqrt (6.0 / Float.ofNat fanIn)
      cudaUniformBuffer n (-limit) limit seed
  | .flat values =>
      if values.size = n then
        _root_.Runtime.Autograd.Cuda.Buffer.ofFloatArrayIO values
      else
        throw <| IO.userError
          s!"torch.runtimeInit: flat initializer length mismatch (expected {n}, got {values.size})"

/-- Materialize a runtime initializer as a normal host tensor. Used for CPU execution. -/
def hostTensorOf {s : Shape} (init : FloatInit) : IO (Tensor Float s) := do
  let values ← floatArrayOf (Spec.Shape.size s) init
  pure <| _root_.Runtime.Autograd.Cuda.Convert.unflattenFloatUnsafe (s := s) values

/--
Host slots for a parameter list before runtime initialization installs the real values.

CUDA runtime initialization immediately replaces these with CUDA mirrors and marks the host values
stale. These entries still give the existing `Param` type a valid host slot for later explicit
readback.
-/
def zeroFloatTList : {ss : List Shape} → Torch.TList Float ss
  | [] => .nil
  | s :: ss => .cons (Spec.fill 0.0 s) (zeroFloatTList (ss := ss))

/--
Apply a shape-indexed initialization plan to an already-created parameter list.

The key point is that the shape list appears on both sides of the type:

```lean
Torch.ParamList Float ss → RuntimeInit.Plan ss → IO Unit
```

So Lean checks the bookkeeping that Python frameworks usually check at runtime: every parameter gets
one initializer, and no extra initializer is silently ignored.
-/
def applyFloatPlan (opts : Torch.Options) :
    {ss : List Shape} → Torch.ParamList Float ss → Plan ss → IO Unit
  | [], .nil, .nil => pure ()
  | s :: ss, .cons p ps, .cons init rest => do
      if opts.usesCuda then
        let buf ← cudaBufferOf (Spec.Shape.size s) init
        _root_.Runtime.Autograd.Torch.Internal.setParamCudaValue (α := Float) (sh := s) p
          { s := s, buf := buf }
      else
        let t ← hostTensorOf (s := s) init
        _root_.Runtime.Autograd.Torch.Internal.setParamHostValue (α := Float) (sh := s) p t
      applyFloatPlan (opts := opts) (ss := ss) ps rest

/--
Apply a runtime list of initializers after checking it against the parameter shapes.

`Plan ss` is the typed form used by the initializer engine.  This entrypoint is for places where
the initializer list comes from outside Lean's typechecker, such as a checkpoint, JSON file, or CLI
experiment.
-/
def applyFloatInits (opts : Torch.Options) {ss : List Shape}
    (ps : Torch.ParamList Float ss) (inits : List FloatInit) : IO Unit := do
  match Plan.ofList? ss inits with
  | .ok plan => applyFloatPlan (opts := opts) ps plan
  | .error msg => throw <| IO.userError msg

end RuntimeInit

/-! ## Scalar-loss module (training) -/

/--
A scalar-loss module definition:
- `initParams` is stored as Float constants (easy to write with `tensorOfList!`),
- `loss` is *polymorphic in the scalar backend* (same code works for Float/IEEE32Exec/…).

You can instantiate this definition as a `ScalarModule` under a chosen backend and dtype.
-/
structure ScalarModuleDef (paramShapes inputShapes : List Shape)
    (natInputShapes : List Shape := []) where
  /-- Initial parameter values, stored as `Float` tensors and cast at instantiation time. -/
  initParams : Torch.TList Float paramShapes
  /--
  Optional storage-first initialization plan for executable `Float` runs.

  The ordinary tensors remain the semantic initial values. This plan records how a runtime may
  materialize the same initialization directly in backend storage without traversing a large
  nested Lean tensor first.
  -/
  runtimeInit : Option (RuntimeInit.Plan paramShapes) := none
  /-- Per-parameter `requiresGrad` flags aligned with `paramShapes`. -/
  initRequiresGrad : List Bool := List.replicate paramShapes.length true
  /--
  Scalar loss over differentiable tensors followed by discrete natural-number tensors.

  The second curried input pack carries labels, token ids, and gather indices without converting
  them through the model's floating-point scalar type.
  -/
  loss :
    ∀ {α : Type}, [Context α] → [DecidableEq Shape] →
      ∀ {m : Type → Type}, [Monad m] → [Torch.Ops (m := m) (α := α)] →
        Torch.CurriedRef (fun s => Torch.Ops.Ref (m := m) (α := α) s)
          (paramShapes ++ inputShapes)
          (Torch.CurriedRef (fun s => Torch.Ops.NatTensorRef (m := m) (α := α) s)
            natInputShapes (m (Torch.Ops.Ref (m := m) (α := α) Shape.scalar)))

/--
Runtime module instance (the thing you "run").

This wraps `Torch.ScalarTrainer`, but exposes a more `Module`-like set of methods.
-/
structure ScalarModule (α : Type) [Context α] [DecidableEq Shape]
    (paramShapes inputShapes : List Shape) (natInputShapes : List Shape := []) where
  /-- Trainer that owns the parameters and runs the scalar-loss program. -/
  trainer : Torch.ScalarTrainer α paramShapes inputShapes natInputShapes
  /-- Runtime options used to instantiate the module. -/
  opts : Torch.Options
  /-- Concrete host/device tensor conversion selected when the module was instantiated. -/
  tensorConv : _root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α

/--
A reusable no-gradient evaluator over an existing parameter list.

The evaluator accepts ordinary tensors and discrete `Nat` tensors separately. It can therefore run
both scalar objectives and tensor-valued forward programs without rebuilding a session for each
input batch.
-/
structure Evaluator (α : Type) (paramShapes inputShapes natInputShapes : List Shape)
    (outputShape : Shape) where
  /-- Evaluate one input pack and return the program output. -/
  evaluate : Torch.Curried.Fn α inputShapes
    (Torch.Curried.Fn Nat natInputShapes (IO (Tensor α outputShape)))

/-- Scalar-output specialization of `Evaluator`. -/
abbrev ScalarEvaluator (α : Type) (paramShapes inputShapes : List Shape)
    (natInputShapes : List Shape := []) :=
  Evaluator α paramShapes inputShapes natInputShapes Shape.scalar

namespace ScalarModule

/--
Create a runtime scalar-loss module from an explicit loss program and initial parameter values.

This is the low-level constructor; public training code starts from a `ScalarModuleDef` and calls
`ScalarModuleDef.instantiate`.
-/
def create {α : Type} [Context α] [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    {paramShapes inputShapes natInputShapes : List Shape}
    (opts : Torch.Options := {})
    (initRequiresGrad : List Bool := List.replicate paramShapes.length true)
    (loss :
      ∀ {m : Type → Type}, [Monad m] → [Torch.Ops (m := m) (α := α)] →
        Torch.CurriedRef (fun s => Torch.Ops.Ref (m := m) (α := α) s) (paramShapes ++ inputShapes)
          (Torch.CurriedRef (fun s => Torch.Ops.NatTensorRef (m := m) (α := α) s)
            natInputShapes (m (Torch.Ops.Ref (m := m) (α := α) Shape.scalar))))
    (initParams : Torch.TList α paramShapes) :
    IO (ScalarModule α paramShapes inputShapes natInputShapes) := do
  let mkTr :=
    Torch.scalarTrainer (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes)
      (natInputShapes := natInputShapes)
      (opts := opts) (initRequiresGrad := initRequiresGrad) (loss := loss)
  let tr ← Torch.Curried.uncurry (α := α) (ss := paramShapes)
    (β := IO (Torch.ScalarTrainer α paramShapes inputShapes natInputShapes)) mkTr initParams
  pure { trainer := tr, opts := opts, tensorConv := inferInstance }

/-- Run the forward pass and return the scalar loss value. -/
def forward {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes inputShapes natInputShapes : List Shape}
    (m : ScalarModule α paramShapes inputShapes natInputShapes)
    (xs : Torch.TList α inputShapes) (natInputs : Torch.TList Nat natInputShapes) :
    IO (Tensor α Shape.scalar) :=
  Torch.ScalarTrainer.forwardT (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes)
    m.trainer xs natInputs

/-- Run one forward/backward pass and return gradients for all parameters. -/
def backward {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes inputShapes natInputShapes : List Shape}
    (m : ScalarModule α paramShapes inputShapes natInputShapes)
    (xs : Torch.TList α inputShapes) (natInputs : Torch.TList Nat natInputShapes) :
    IO (Torch.TList α paramShapes) :=
  Torch.ScalarTrainer.backwardT (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes)
    m.trainer xs natInputs

/-- Return a scalar loss and its parameter gradients from one forward tape. -/
def lossAndBackward {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes inputShapes natInputShapes : List Shape}
    (m : ScalarModule α paramShapes inputShapes natInputShapes)
    (xs : Torch.TList α inputShapes) (natInputs : Torch.TList Nat natInputShapes) :
    IO (Tensor α Shape.scalar × Torch.TList α paramShapes) :=
  Torch.ScalarTrainer.lossAndBackwardT
    (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes) m.trainer xs natInputs

/-- Convenience "one-step SGD": compute gradients and apply an SGD update with learning rate `lr`.
  -/
def step {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes inputShapes natInputShapes : List Shape}
    (m : ScalarModule α paramShapes inputShapes natInputShapes)
    (lr : α) (xs : Torch.TList α inputShapes) (natInputs : Torch.TList Nat natInputShapes) :
    IO Unit :=
  Torch.ScalarTrainer.stepT (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes)
    m.trainer lr xs natInputs

/-- Apply one SGD update and return the loss from the tape that produced its gradients. -/
def stepWithLoss {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes inputShapes natInputShapes : List Shape}
    (m : ScalarModule α paramShapes inputShapes natInputShapes)
    (lr : α) (xs : Torch.TList α inputShapes) (natInputs : Torch.TList Nat natInputShapes) :
    IO (Tensor α Shape.scalar) :=
  Torch.ScalarTrainer.stepWithLossT
    (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes) m.trainer lr xs natInputs

/-- Initialize an optimizer state for this module's parameters. -/
def initOptim {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes inputShapes natInputShapes : List Shape}
    (m : ScalarModule α paramShapes inputShapes natInputShapes)
    (opt : TorchLean.Optim.Optimizer α paramShapes) :
    IO opt.State := do
  -- Generic optimizer states are initialized from host tensors. Synchronize device-backed
  -- parameters first so a CUDA module cannot seed those states from stale host mirrors.
  let _ ← m.trainer.getParams
  opt.init m.trainer.params

/--
Run one optimizer step using an explicit optimizer + state.

This mirrors a PyTorch training step:
1. compute gradients (`backwardT`)
2. update parameters via `opt.step` and return the new optimizer state
-/
def stepWith {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes inputShapes natInputShapes : List Shape}
    (m : ScalarModule α paramShapes inputShapes natInputShapes)
    (opt : TorchLean.Optim.Optimizer α paramShapes) (st : opt.State)
    (xs : Torch.TList α inputShapes) (natInputs : Torch.TList Nat natInputShapes) :
    IO opt.State := do
  match ← opt.trainerStep? m.trainer st xs natInputs with
  | some st' =>
      pure st'
  | none =>
      let grads ← Torch.ScalarTrainer.backwardT (α := α)
        (paramShapes := paramShapes) (inputShapes := inputShapes) m.trainer xs natInputs
      let _ ← m.trainer.getParams
      opt.step st m.trainer.params grads

/--
Run an explicit optimizer step and return both the next optimizer state and the loss used for it.

Native trainer hooks may keep gradients and optimizer state on the selected device. The fallback
path computes the loss and gradients once, synchronizes parameter mirrors when needed, and applies
the optimizer to those gradients.
-/
def stepWithOptimizerAndLoss {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes inputShapes natInputShapes : List Shape}
    (m : ScalarModule α paramShapes inputShapes natInputShapes)
    (opt : TorchLean.Optim.Optimizer α paramShapes) (st : opt.State)
    (xs : Torch.TList α inputShapes) (natInputs : Torch.TList Nat natInputShapes) :
    IO (opt.State × Tensor α Shape.scalar) := do
  match ← opt.trainerStepWithLoss? m.trainer st xs natInputs with
  | some result =>
      pure result
  | none =>
      let (loss, grads) ← Torch.ScalarTrainer.lossAndBackwardT (α := α)
        (paramShapes := paramShapes) (inputShapes := inputShapes) m.trainer xs natInputs
      let _ ← m.trainer.getParams
      let st' ← opt.step st m.trainer.params grads
      pure (st', loss)

/-- Fetch the current parameter values as a shape-indexed list. -/
def params {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes inputShapes natInputShapes : List Shape}
    (m : ScalarModule α paramShapes inputShapes natInputShapes) : IO (Torch.TList α paramShapes) :=
  m.trainer.getParams

/-- Overwrite all parameter values. -/
def setParams {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes inputShapes natInputShapes : List Shape}
    (m : ScalarModule α paramShapes inputShapes natInputShapes)
    (ps : Torch.TList α paramShapes) : IO Unit :=
  Torch.ParamList.setValues (α := α) (ss := paramShapes) m.trainer.params ps

/-- Train with vanilla SGD for a fixed number of steps on a fixed list of samples. -/
def trainSGD {α : Type} [Context α] [DecidableEq Shape] [ToString α]
    {paramShapes inputShapes : List Shape}
    (m : ScalarModule α paramShapes inputShapes)
    (lr : α) (steps : Nat) (samples : List (Torch.TList α inputShapes))
    (logEvery : Nat := 1) : IO Unit :=
  Torch.trainCycleSGD (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes)
    m.trainer lr steps samples (logEvery := logEvery)

/-- Like `trainSGD`, but with an explicit optimizer + mutable optimizer state. -/
def trainWith {α : Type} [Context α] [DecidableEq Shape] [ToString α]
    {paramShapes inputShapes : List Shape}
    (m : ScalarModule α paramShapes inputShapes)
    (opt : TorchLean.Optim.Optimizer α paramShapes) (st0 : opt.State)
    (steps : Nat) (samples : List (Torch.TList α inputShapes))
    (logEvery : Nat := 1) : IO opt.State :=
  TorchLean.trainCycleOptim (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes)
    m.trainer opt st0 steps samples (logEvery := logEvery)

/-- Compute the mean loss over a list of samples (no parameter updates). -/
def meanLoss {α : Type} [Context α] [DecidableEq Shape] [ToString α]
    {paramShapes inputShapes : List Shape}
    (m : ScalarModule α paramShapes inputShapes)
    (samples : List (Torch.TList α inputShapes)) : IO α :=
  Torch.meanLoss (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes) m.trainer
    samples

end ScalarModule

namespace Evaluator

/-- Apply a reusable evaluator to shape-indexed ordinary and discrete input lists. -/
def evaluateT {α : Type} {paramShapes inputShapes natInputShapes : List Shape}
    {outputShape : Shape}
    (evaluator : Evaluator α paramShapes inputShapes natInputShapes outputShape)
    (xs : Torch.TList α inputShapes) (natInputs : Torch.TList Nat natInputShapes) :
    IO (Tensor α outputShape) :=
  let withNat := Torch.Curried.uncurry (α := α) (ss := inputShapes)
    (β := Torch.Curried.Fn Nat natInputShapes (IO (Tensor α outputShape)))
    evaluator.evaluate xs
  Torch.Curried.uncurry (α := Nat) (ss := natInputShapes)
    (β := IO (Tensor α outputShape)) withNat natInputs

/--
Create a reusable no-gradient evaluator for a backend-polymorphic program.

The evaluator shares the supplied live parameter objects. Its eager session is reset after every
call, so validation and generation do not retain one execution graph per input batch.
-/
def withParams
    {α : Type} [Context α] [DecidableEq Shape]
    [tensorConv : _root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    {paramShapes inputShapes natInputShapes : List Shape} {outputShape : Shape}
    (program : ProgramWithNatInputs α (paramShapes ++ inputShapes) natInputShapes outputShape)
    (opts : Torch.Options)
    (params : Torch.ParamList α paramShapes) :
    IO (Evaluator α paramShapes inputShapes natInputShapes outputShape) := do
  let opts := { opts with trackGradients := false }
  let sess ← Torch.Internal.EagerSession.new (α := α) opts
  let programEager := program (m := Torch.Internal.EagerM α)
  let evaluate : Torch.Curried.Fn α inputShapes
      (Torch.Curried.Fn Nat natInputShapes (IO (Tensor α outputShape))) :=
    Torch.Curried.curry (α := α) (ss := inputShapes)
      (β := Torch.Curried.Fn Nat natInputShapes (IO (Tensor α outputShape))) (fun xs =>
        Torch.Curried.curry (α := Nat) (ss := natInputShapes)
          (β := IO (Tensor α outputShape)) (fun natInputs => do
            sess.resetTape
            try
              let outRef ← (do
                let pRefs ← Torch.Internal.useParams (α := α) (ss := paramShapes) params
                let xRefs ← Torch.Internal.useInputs (α := α) (ss := inputShapes) xs
                let allRefs := Torch.RefList.append
                  (ss₁ := paramShapes) (ss₂ := inputShapes) pRefs xRefs
                let withNat := Torch.CurriedRef.uncurry
                  (ss := paramShapes ++ inputShapes) programEager allRefs
                Torch.CurriedRef.uncurryTList
                  (α := Nat) (ss := natInputShapes) withNat natInputs) |>.run sess
              Torch.Internal.EagerSession.getValue (α := α) sess outRef
            finally
              sess.resetTape
              if opts.usesCuda then
                Torch.Internal.EagerSession.collectCudaAllocator))
  pure { evaluate := evaluate }

end Evaluator

namespace ScalarModuleDef

/--
Create a reusable no-gradient evaluator over an existing live parameter list.

This is useful when two definitions share the same parameter layout but differ in execution mode,
for example training and evaluation losses for a model containing dropout. The evaluator shares
the parameter objects and their current backend storage with the training module. Its eager session
is reset after every call, so repeated validation does not retain one execution graph per batch.
-/
def evaluatorWithParams
    {α : Type} [Context α] [DecidableEq Shape]
    [tensorConv : _root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    {paramShapes inputShapes natInputShapes : List Shape}
    (d : ScalarModuleDef paramShapes inputShapes natInputShapes)
    (opts : Torch.Options)
    (params : Torch.ParamList α paramShapes) :
    IO (ScalarEvaluator α paramShapes inputShapes natInputShapes) :=
  Evaluator.withParams (program := d.loss (α := α)) opts params

/--
Evaluate a scalar module definition once against an existing live parameter list.

Use `evaluatorWithParams` when evaluating more than one batch. It reuses one no-gradient session
and avoids repeatedly allocating the session state associated with a large model.
-/
def forwardWithParams
    {α : Type} [Context α] [DecidableEq Shape]
    [tensorConv : _root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    {paramShapes inputShapes natInputShapes : List Shape}
    (d : ScalarModuleDef paramShapes inputShapes natInputShapes)
    (opts : Torch.Options)
    (params : Torch.ParamList α paramShapes)
    (xs : Torch.TList α inputShapes) (natInputs : Torch.TList Nat natInputShapes) :
    IO (Tensor α Shape.scalar) := do
  let evaluator ← evaluatorWithParams d opts params
  Evaluator.evaluateT evaluator xs natInputs

/--
Instantiate a `ScalarModuleDef` by casting Float initializers to `α` and choosing Torch options.

This is the most general constructor. The shorter `instantiate` entrypoint chooses standard runtime
options before calling this function.
-/
def instantiateWith {α : Type} [Context α] [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    {paramShapes inputShapes natInputShapes : List Shape}
    (d : ScalarModuleDef paramShapes inputShapes natInputShapes)
    (cast : Float → α) (opts : Torch.Options) :
    IO (ScalarModule α paramShapes inputShapes natInputShapes) := do
  let initParams : Torch.TList α paramShapes := castTList (α := α) cast d.initParams
  ScalarModule.create (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes)
    (natInputShapes := natInputShapes)
    (opts := opts) (initRequiresGrad := d.initRequiresGrad)
    (loss := d.loss (α := α)) initParams

/--
Instantiate a Float module using runtime parameter initializers.

This is the runtime-initialized sibling of `instantiateWith`.  Instead of first building every initial
parameter as a full Lean tensor, it creates minimal zero host tensors and then applies a
shape-indexed runtime plan to the module parameters.  In CUDA mode those initializers allocate
device buffers directly and mark the host copies stale; public parameter readback still
synchronizes them through the existing CUDA mirror machinery.
-/
def instantiateFloatWithRuntimePlan {paramShapes inputShapes natInputShapes : List Shape}
    (d : ScalarModuleDef paramShapes inputShapes natInputShapes)
    (opts : Torch.Options)
    (plan : RuntimeInit.Plan paramShapes) :
    IO (ScalarModule Float paramShapes inputShapes natInputShapes) := do
  let initParams := RuntimeInit.zeroFloatTList (ss := paramShapes)
  let module ← ScalarModule.create (α := Float) (paramShapes := paramShapes)
    (inputShapes := inputShapes) (natInputShapes := natInputShapes)
    (opts := opts) (initRequiresGrad := d.initRequiresGrad)
    (loss := d.loss (α := Float)) initParams
  RuntimeInit.applyFloatPlan (opts := opts) module.trainer.params plan
  pure module

/--
Instantiate a Float module from a plain initializer list.

This wrapper is useful at file/JSON boundaries.  Internally it immediately checks the list against
`paramShapes` and then delegates to `instantiateFloatWithRuntimePlan`, so the actual parameter
mutation still goes through the shape-indexed path.
-/
def instantiateFloatWithRuntimeInit {paramShapes inputShapes natInputShapes : List Shape}
    (d : ScalarModuleDef paramShapes inputShapes natInputShapes)
    (opts : Torch.Options)
    (inits : List RuntimeInit.FloatInit) :
    IO (ScalarModule Float paramShapes inputShapes natInputShapes) := do
  match RuntimeInit.Plan.ofList? paramShapes inits with
  | .ok plan => instantiateFloatWithRuntimePlan d opts plan
  | .error msg => throw <| IO.userError msg

/--
Instantiate a `Float` module, using a storage-first plan when the definition supplies one.

Definitions without a runtime plan retain the original tensor-valued initialization path. This
makes runtime initialization an optimization with an explicit fallback, rather than a requirement
on every custom layer.
-/
def instantiateFloat {paramShapes inputShapes natInputShapes : List Shape}
    (d : ScalarModuleDef paramShapes inputShapes natInputShapes) (opts : Torch.Options) :
    IO (ScalarModule Float paramShapes inputShapes natInputShapes) :=
  match d.runtimeInit with
  | some plan => instantiateFloatWithRuntimePlan d opts plan
  | none => instantiateWith (α := Float) d id opts

/-- Convenience instantiator that chooses only the backend (`.eager` or `.compiled`). -/
def instantiate {α : Type} [Context α] [DecidableEq Shape]
    [_root_.Runtime.Autograd.Torch.Internal.CudaBridge.TensorConv α]
    {paramShapes inputShapes natInputShapes : List Shape}
    (d : ScalarModuleDef paramShapes inputShapes natInputShapes)
    (cast : Float → α) (backend : Torch.Backend := .eager) :
    IO (ScalarModule α paramShapes inputShapes natInputShapes) := do
  instantiateWith (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes)
    (natInputShapes := natInputShapes)
    d cast { backend := backend }

end ScalarModuleDef

/-- Mutable optimizer state bound to one executable module. -/
structure OptimizerHandle (α : Type) [Context α] [DecidableEq Shape]
    (paramShapes inputShapes : List Shape) (State : Type) where
  module : ScalarModule α paramShapes inputShapes
  state : IO.Ref State
  step : Torch.TList α inputShapes → IO Unit

/-- Initialize an optimizer and bind its state and update operation to `module`. -/
def optimizerHandle {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes inputShapes : List Shape}
    (module : ScalarModule α paramShapes inputShapes)
    (optimizer : Optim.Optimizer α paramShapes) :
    IO (OptimizerHandle α paramShapes inputShapes optimizer.State) := do
  let initialState ← ScalarModule.initOptim module optimizer
  let state ← IO.mkRef initialState
  let step (sample : Torch.TList α inputShapes) : IO Unit := do
    let currentState ← state.get
    let nextState ← ScalarModule.stepWith module optimizer currentState sample .nil
    state.set nextState
  pure { module, state, step }

end Module

end TorchLean
end Autograd
end Runtime
