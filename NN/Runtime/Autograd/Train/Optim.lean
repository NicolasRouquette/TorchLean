/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Train.Core
public import NN.Runtime.Optim.Optimizers
public import NN.Runtime.Optim.Schedulers
public import Std.Data.HashMap

/-!
# Optimizer integration for Runtime.Autograd

This module is the training-loop side of autograd: it takes a gradient map produced by
`Runtime.Autograd` and applies parameter updates.

PyTorch analogy:
- `ParamTable` is like an ordered list of parameters, but we key everything by a stable `Nat` id
  (closer to `state_dict` keys than pointer identity).
- `ParamGroup` and `OptimizerState` mirror `torch.optim.Optimizer` parameter groups and state.
- `LRScheduler` is a small wrapper around our scheduler implementations, similar to
  `torch.optim.lr_scheduler.*`.

All updates are *shape checked* and implemented using the pure `Spec` tensor operators, so they
can be used in eager execution or lowered into a typed graph.

Formula ownership:
- this file owns the heterogeneous parameter-table handling, parameter groups, lazy state maps,
  scheduler stepping, and PyTorch-style coupled weight decay at the training-loop boundary;
- `NN.Runtime.Optim.Optimizers` owns the canonical per-tensor optimizer equations.

The important rule is: this file must not define a second public optimizer-formula surface. The
`step` implementation below constructs canonical optimizer states from the dynamic parameter-table
buffers and calls `NN.Runtime.Optim.Optimizers` directly. The only local algebra left here is
training-loop glue that is not represented by the canonical pure states, such as coupled
weight-decay preprocessing and PyTorch-style momentum dampening/Nesterov handling.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Train

open Spec
open Tensor

/-!
## Parameter table
-/
/-!
The declarations below provide the parameter registry used by the training loop.

Unlike PyTorch (where parameters are objects with identity), we use an explicit `Nat` id so that:
- gradients can be stored in a `HashMap Nat _`,
- optimizer state buffers can be stored in a `HashMap Nat _`,
- serialization can be done by a pure `state_dict` record.
-/
/--
A single trainable parameter entry.

This is the Runtime.Autograd equivalent of a "parameter tensor" in PyTorch, except we make the
identifier explicit (`id : Nat`) so we can key gradients and optimizer state in pure maps.
-/
structure ParamEntry (α : Type) where
  /-- Stable identifier used to key gradients and optimizer state. -/
  id : Nat
  /-- Optional label, such as a module path; used only for reporting and debugging. -/
  name : Option String := none
  /-- The shape-erased parameter value. -/
  value : Spec.PackedTensor α

/-- A flat list of parameters used by the training loop. -/
abbrev ParamTable (α : Type) := List (ParamEntry α)

namespace ParamEntry

/-!
### Constructors
-/
/--
Create a `ParamEntry` from a typed tensor.

This is mostly a convenience for assembling a `ParamTable` from known-shaped tensors.
-/
def ofTensor {α : Type} {s : Shape} (id : Nat) (t : Tensor α s) (name : Option String := none) :
  ParamEntry α :=
  { id := id, name := name, value := Spec.PackedTensor.ofTensor t }

end ParamEntry

namespace ParamTable

variable {α : Type}

/-- List of ids for membership checks. -/
def ids (ps : ParamTable α) : List Nat :=
  ps.map (·.id)

/-- Find a parameter entry by id. -/
def find? (ps : ParamTable α) (id : Nat) : Option (ParamEntry α) :=
  List.find? (fun p => p.id == id) ps

/-- Get a typed tensor from the table, with shape checking. -/
def getTensor {α : Type} [DecidableEq Shape] {s : Shape}
  (tag : String) (ps : ParamTable α) (id : Nat) : Result (Tensor α s) := by
  match find? ps id with
  | none =>
      exact .error (tagError tag s!"missing param id {id}")
  | some p =>
      if h : p.value.shape = s then
        exact .ok (Tensor.castShape p.value.tensor h)
      else
        exact .error (tagError tag s!"param shape mismatch for id {id}")

/-- Replace a parameter entry value by id. -/
def set (ps : ParamTable α) (id : Nat) (value : Spec.PackedTensor α) : ParamTable α :=
  ps.map (fun p => if p.id = id then { p with value := value } else p)

/--
Build the set of parameter identifiers, rejecting duplicate ids.

Optimizer buffers and gradients are keyed by `id`, so accepting two table entries with the same id
would make both entries consume one gradient and mutate one shared optimizer state.
-/
def checkedIdSet (ps : ParamTable α) : Result (Std.HashMap Nat Unit) := do
  let mut ids : Std.HashMap Nat Unit := {}
  for p in ps do
    if ids.contains p.id then
      throw (tagError "optim" s!"duplicate param id {p.id}")
    else
      ids := ids.insert p.id ()
  pure ids

end ParamTable

/-!
## Scheduler wrapper
-/
/--
Learning-rate scheduler wrapper used by the training loop.

PyTorch analogy: this plays the role of `torch.optim.lr_scheduler.*` objects, except we keep the
state as an inductive value and expose a pure `getLR`/`advance` API.
-/
inductive LRScheduler (α : Type) where
  | constant : Optim.ConstantScheduler α -> LRScheduler α
  | exponential : Optim.ExponentialDecayScheduler α -> LRScheduler α
  | step : Optim.StepDecayScheduler α -> LRScheduler α
  | cosine : Optim.CosineAnnealingScheduler α -> LRScheduler α
  | linearWarmup : Optim.LinearWarmupScheduler α -> LRScheduler α
  | warmupCosine : Optim.WarmupCosineScheduler α -> LRScheduler α
  | cyclic : Optim.CyclicScheduler α -> LRScheduler α
  | triangular : Optim.TriangularCycleScheduler α -> LRScheduler α
  | oneCycle : Optim.OneCycleScheduler α -> LRScheduler α
  | lrFinder : Optim.LRFinder α -> LRScheduler α
  /--
  Custom schedule with an explicit step counter.

`custom f k` means "use learning rate `f k` for this step, and increment to `k+1` on `advance`".
  -/
  | custom : (Nat -> α) -> Nat -> LRScheduler α

namespace LRScheduler

variable {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]

/-- Read current learning rate from the scheduler state. -/
def getLR : LRScheduler α -> α
  | constant s => Optim.ConstantScheduler.getLr s 0
  | exponential s => Optim.ExponentialDecayScheduler.getLr s
  | step s => Optim.StepDecayScheduler.getLr s
  | cosine s => Optim.CosineAnnealingScheduler.getLr s
  | linearWarmup s => Optim.LinearWarmupScheduler.getLr s
  | warmupCosine s => Optim.WarmupCosineScheduler.getLr s
  | cyclic s => Optim.CyclicScheduler.getLr s
  | triangular s => Optim.TriangularCycleScheduler.getLr s
  | oneCycle s => Optim.OneCycleScheduler.getLr s
  | lrFinder s => Optim.LRFinder.getLr s
  | custom f k => f k

/-- Advance scheduler state by one step. -/
def advance : LRScheduler α -> LRScheduler α
  | constant s => constant (Optim.ConstantScheduler.step s)
  | exponential s => exponential (Optim.ExponentialDecayScheduler.step s)
  | step s => step (Optim.StepDecayScheduler.step s)
  | cosine s => cosine (Optim.CosineAnnealingScheduler.step s)
  | linearWarmup s => linearWarmup (Optim.LinearWarmupScheduler.step s)
  | warmupCosine s => warmupCosine (Optim.WarmupCosineScheduler.step s)
  | cyclic s => cyclic (Optim.CyclicScheduler.step s)
  | triangular s => triangular (Optim.TriangularCycleScheduler.step s)
  | oneCycle s => oneCycle (Optim.OneCycleScheduler.step s)
  | lrFinder s => lrFinder (Optim.LRFinder.step s)
  | custom f k => custom f (k + 1)

end LRScheduler

/-!
## Optimizer configuration
-/
/--
Which optimizer update rule to apply.

PyTorch analogy: these correspond approximately to `torch.optim.SGD`, `Adam`, `AdamW`, etc.
-/
inductive OptimizerKind
  | sgd
  | momentum
  | adagrad
  | rmsprop
  | adam
  | adamw
  | adadelta
  deriving Repr, DecidableEq

/--
Optimizer hyperparameters for a subset of parameters.

PyTorch analogy: this is a single entry in the optimizer's param-group list
(`optimizer.param_groups`).
-/
structure ParamGroup (α : Type) [Context α] where
  /-- Parameter ids that belong to this group. -/
  params : List Nat
  /-- Base learning rate (possibly overridden by `scheduler` on each step). -/
  lr : α
  /-- $\ell_2$ regularization coefficient (behavior depends on the optimizer kind; see AdamW). -/
  weightDecay : α := 0
  /-- Momentum factor (SGD with momentum). -/
  momentum : α := 0
  /-- Dampening for momentum updates. -/
  dampening : α := 0
  /-- Use Nesterov variant for momentum updates. -/
  nesterov : Bool := false
  /-- Adam beta1 parameter (exponential decay for the first moment). -/
  beta1 : α := Numbers.one - Numbers.oneTenth
  /-- Adam beta2 parameter (exponential decay for the second moment). -/
  beta2 : α :=
    Numbers.one - (Numbers.one / (Numbers.ten * Numbers.ten * Numbers.ten))
  /-- Numerical stability term used by adaptive optimizers. -/
  epsilon : α := Numbers.epsilon
  /-- "Rho" decay parameter for RMSProp/AdaDelta style optimizers. -/
  rho : α := Numbers.one - Numbers.oneTenth
  /-- Optional learning-rate scheduler for this group. -/
  scheduler : Option (LRScheduler α) := none

/--
Full optimizer state used by the training loop.

This mirrors PyTorch's optimizer state:
- a global optimizer-call counter,
- hyperparameter groups,
- per-parameter Adam steps, and
- per-parameter state buffers keyed by parameter id (`Nat`).
-/
structure OptimizerState (α : Type) [Context α] where
  /-- Which update rule to apply on `step`. -/
  kind : OptimizerKind
  /-- Parameter groups (hyperparameters + membership). -/
  groups : List (ParamGroup α)
  /-- Global optimizer-call counter (increments once per `step`, including sparse steps). -/
  step : Nat := 0
  /--
  Number of gradient-bearing Adam/AdamW updates applied to each parameter.

  Adam bias correction is parameter-local: a parameter whose gradient is absent does not advance
  its moment step. Keeping this separate from `step` is necessary for sparse or conditional models.
  -/
  parameterSteps : Std.HashMap Nat Nat := {}
  /-- Momentum buffer (SGD with momentum / Nesterov), keyed by parameter id. -/
  momentumBuffer : Std.HashMap Nat (Spec.PackedTensor α) := {}
  /-- Adam first-moment estimate, keyed by parameter id. -/
  m : Std.HashMap Nat (Spec.PackedTensor α) := {}
  /-- Adam second-moment estimate, keyed by parameter id. -/
  v : Std.HashMap Nat (Spec.PackedTensor α) := {}
  /-- Accumulator buffer (AdaGrad/RMSProp/AdaDelta), keyed by parameter id. -/
  acc : Std.HashMap Nat (Spec.PackedTensor α) := {}
  /-- Second accumulator buffer (AdaDelta), keyed by parameter id. -/
  acc2 : Std.HashMap Nat (Spec.PackedTensor α) := {}

/--
A pure state snapshot for saving/restoring optimizer state.

PyTorch analogy: this is the data carried by `optimizer.state_dict()` (modulo naming/layout).
We use association lists instead of `HashMap` so the result is deterministic and easy to serialize.
-/
structure OptimStateDict (α : Type) [Context α] where
  /-- Optimizer algorithm used to interpret the stored buffers. -/
  kind : OptimizerKind
  /-- Global optimizer step at the time the snapshot was taken. -/
  step : Nat
  /-- Per-parameter Adam/AdamW moment steps used for bias correction. -/
  parameterSteps : List (Nat × Nat) := []
  /-- Parameter groups, including scheduler state and hyperparameters. -/
  groups : List (ParamGroup α)
  /-- Momentum buffers keyed by parameter id. -/
  momentumBuffer : List (Nat × Spec.PackedTensor α)
  /-- Adam-family first-moment buffers keyed by parameter id. -/
  m : List (Nat × Spec.PackedTensor α)
  /-- Adam-family second-moment buffers keyed by parameter id. -/
  v : List (Nat × Spec.PackedTensor α)
  /-- AdaGrad/RMSProp/Adadelta accumulator buffers keyed by parameter id. -/
  acc : List (Nat × Spec.PackedTensor α)
  /-- Adadelta second accumulator buffers keyed by parameter id. -/
  acc2 : List (Nat × Spec.PackedTensor α)

namespace OptimizerState

variable {α : Type} [Context α]

/--
Serialize optimizer state to a pure record.

PyTorch analogy: this is the "export" step for `state_dict()`.
-/
def toStateDict (opt : OptimizerState α) : OptimStateDict α :=
  { kind := opt.kind
  , step := opt.step
  , parameterSteps := opt.parameterSteps.toList
  , groups := opt.groups
  , momentumBuffer := opt.momentumBuffer.toList
  , m := opt.m.toList
  , v := opt.v.toList
  , acc := opt.acc.toList
  , acc2 := opt.acc2.toList
  }

/--
Restore optimizer state from a state dict.

PyTorch analogy: this is the "import" step for `load_state_dict(...)`.
-/
def ofStateDict (d : OptimStateDict α) : OptimizerState α :=
  { kind := d.kind
  , step := d.step
  , parameterSteps := Std.HashMap.ofList d.parameterSteps
  , groups := d.groups
  , momentumBuffer := Std.HashMap.ofList d.momentumBuffer
  , m := Std.HashMap.ofList d.m
  , v := Std.HashMap.ofList d.v
  , acc := Std.HashMap.ofList d.acc
  , acc2 := Std.HashMap.ofList d.acc2
  }

end OptimizerState

/-!
## Optimizer step
-/
namespace Optim

variable {α : Type} [Context α] [DecidableEq Shape] [DecidableRel ((· > ·) : α → α → Prop)]

/--
Lookup a per-parameter state buffer, initializing it with zeros if absent.

This is used for momentum/Adam accumulator initialization (PyTorch does this lazily on first step).
-/
def getOrInit
  (m : Std.HashMap Nat (Spec.PackedTensor α))
  (id : Nat) (p : Spec.PackedTensor α) : Spec.PackedTensor α :=
  m.getD id { shape := p.shape, tensor := Spec.fill (0 : α) p.shape }

/--
Shape-check and cast an optimizer state buffer to match the current parameter value.

This prevents silent shape mismatches when reloading a checkpoint into a model with different
parameter shapes.
-/
def castState
  (tag : String) (id : Nat) (buf pval : Spec.PackedTensor α) : Result (Tensor α pval.shape) := do
  if h : buf.shape = pval.shape then
    pure (Tensor.castShape buf.tensor h)
  else
    throw (tagError tag s!"state shape mismatch for id {id}")

/--
Add an $\ell_2$ regularization term to the gradient:
$g+\operatorname{weightDecay}\,\operatorname{param}$.

Note: this is the *coupled* weight decay used by classic SGD-style updates.
For AdamW the integration step delegates to the canonical optimizer's decoupled update.
-/
def addWeightDecay {s : Shape}
  (param grad : Tensor α s) (weightDecay : α) : Tensor α s :=
  addSpec grad (scaleSpec param weightDecay)

/--
Recover the previous Adam step for one parameter.

New states record this counter explicitly. The moment-buffer fallback preserves sensible behavior
when loading a state dictionary created before per-parameter counters were stored.
-/
def previousAdamStep
    (parameterSteps : Std.HashMap Nat Nat)
    (moment1 moment2 : Std.HashMap Nat (Spec.PackedTensor α))
    (globalStep id : Nat) : Nat :=
  match parameterSteps.get? id with
  | some t => t
  | none =>
      if moment1.contains id || moment2.contains id then globalStep else 0

/--
Update each group's learning rate from its scheduler (if present) and advance the scheduler state.

This matches the common training-loop pattern: "read LR, then call `scheduler.step()`".
-/
def updateGroupSchedulers {α : Type} [Context α] [DecidableRel ((· > ·) : α → α → Prop)]
  (groups : List (ParamGroup α)) : List (ParamGroup α) :=
  groups.map (fun g =>
    let lr := match g.scheduler with
      | none => g.lr
      | some s => LRScheduler.getLR s
    let sched := g.scheduler.map LRScheduler.advance
    { g with lr := lr, scheduler := sched })

/--
Build a map from parameter id to its `ParamGroup`.

Fails if an id appears in multiple groups (PyTorch also disallows overlapping param groups).
-/
def groupMap {α : Type} [Context α]
  (groups : List (ParamGroup α)) : Result (Std.HashMap Nat (ParamGroup α)) := do
  let mut m : Std.HashMap Nat (ParamGroup α) := {}
  for g in groups do
    for id in g.params do
      if m.contains id then
        throw (tagError "optim" s!"param id {id} appears in multiple groups")
      else
        m := m.insert id g
  pure m

/--
Apply one optimizer step to a parameter table.

Inputs:
- `opt` is the current optimizer state (including per-parameter buffers),
- `params` is the current parameter table,
- `grads` maps parameter ids to gradients (as produced by autograd).

Behavior:
- rejects duplicate parameter ids, overlapping groups, and group ids absent from the parameter table,
- applies LR schedulers (if configured) per group,
- shape-checks gradients and state buffers against each parameter,
- updates per-parameter state buffers (momentum / Adam m,v / accumulators),
- returns the updated optimizer state and an updated parameter table.
-/
def step
  (opt : OptimizerState α)
  (params : ParamTable α)
  (grads : Std.HashMap Nat (Spec.PackedTensor α)) : Result (OptimizerState α × ParamTable α) := do
  let parameterIds ← ParamTable.checkedIdSet params
  let groups' := updateGroupSchedulers opt.groups
  let gmap <- groupMap groups'
  for (id, _) in gmap.toList do
    if !parameterIds.contains id then
      throw (tagError "optim" s!"param group references unknown id {id}")
  let tNext := opt.step + 1
  let mut parameterSteps := opt.parameterSteps
  let mut momentumBuffer := opt.momentumBuffer
  let mut m := opt.m
  let mut v := opt.v
  let mut acc := opt.acc
  let mut acc2 := opt.acc2

  let mut updated : List (ParamEntry α) := []
  for p in params do
    let g ← match gmap.get? p.id with
      | some g => pure g
      | none =>
          throw (tagError "optim" s!"no param group for id {p.id}")

    let pval := p.value
    let gradOpt := grads.get? p.id
    match gradOpt with
    | none =>
        updated := { p with value := pval } :: updated
    | some gradPacked =>
        if h : gradPacked.shape = pval.shape then
          let grad : Tensor α pval.shape := Tensor.castShape gradPacked.tensor h
          let param : Tensor α pval.shape := pval.tensor
          match opt.kind with
          | .sgd =>
              let gradWD := addWeightDecay param grad g.weightDecay
              let state : _root_.Optim.SGD.State α pval.shape := { lr := g.lr }
              let param' := Tensor.materialize <|
                _root_.Optim.SGD.update (α := α) (s := pval.shape) state param gradWD
              updated := { p with value := Spec.PackedTensor.ofTensor param' } :: updated
          | .momentum =>
              let firstUpdate := !momentumBuffer.contains p.id
              let v0 := getOrInit momentumBuffer p.id pval
              let v0t ← castState "optim" p.id v0 pval
              let gradWD := addWeightDecay param grad g.weightDecay
              -- PyTorch initializes a momentum buffer from the first raw gradient. Dampening is
              -- applied only once a buffer already exists.
              let gradForBuffer :=
                if firstUpdate then gradWD else scaleSpec gradWD (1 - g.dampening)
              let momentumState : _root_.Optim.MomentumSGD.State α pval.shape :=
                { lr := g.lr, momentum := g.momentum, buf := v0t }
              let (momentumState', paramClassic) :=
                _root_.Optim.MomentumSGD.update (α := α) (s := pval.shape)
                  momentumState param gradForBuffer
              let updateDir :=
                if g.nesterov then
                  addSpec gradWD (scaleSpec momentumState'.buf g.momentum)
                else
                  momentumState'.buf
              let param' :=
                if g.nesterov then
                  subSpec param (scaleSpec updateDir g.lr)
                else
                  paramClassic
              let v' := momentumState'.buf
              momentumBuffer := momentumBuffer.insert p.id (Spec.PackedTensor.ofTensor (Tensor.materialize v'))
              updated := { p with value := Spec.PackedTensor.ofTensor (Tensor.materialize param') } :: updated
          | .adagrad =>
              let acc0 := getOrInit acc p.id pval
              let acc0t ← castState "optim" p.id acc0 pval
              let gradWD := addWeightDecay param grad g.weightDecay
              let state : _root_.Optim.AdaGrad.State α pval.shape :=
                { lr := g.lr, epsilon := g.epsilon, accumulator := acc0t }
              let (state', param') :=
                _root_.Optim.AdaGrad.update (α := α) (s := pval.shape) state param gradWD
              let acc' := state'.accumulator
              acc := acc.insert p.id (Spec.PackedTensor.ofTensor (Tensor.materialize acc'))
              updated := { p with value := Spec.PackedTensor.ofTensor (Tensor.materialize param') } :: updated
          | .rmsprop =>
              let acc0 := getOrInit acc p.id pval
              let acc0t ← castState "optim" p.id acc0 pval
              let gradWD := addWeightDecay param grad g.weightDecay
              let state : _root_.Optim.RMSProp.State α pval.shape :=
                { lr := g.lr, decay := g.rho, epsilon := g.epsilon, accumulator := acc0t }
              let (state', param') :=
                _root_.Optim.RMSProp.update (α := α) (s := pval.shape) state param gradWD
              let acc' := state'.accumulator
              acc := acc.insert p.id (Spec.PackedTensor.ofTensor (Tensor.materialize acc'))
              updated := { p with value := Spec.PackedTensor.ofTensor (Tensor.materialize param') } :: updated
          | .adam =>
              let m0 := getOrInit m p.id pval
              let v0 := getOrInit v p.id pval
              let m0t ← castState "optim" p.id m0 pval
              let v0t ← castState "optim" p.id v0 pval
              let gradWD := addWeightDecay param grad g.weightDecay
              let previousStep := previousAdamStep parameterSteps m v opt.step p.id
              let state : _root_.Optim.Adam.State α pval.shape :=
                { lr := g.lr
                  beta1 := g.beta1
                  beta2 := g.beta2
                  epsilon := g.epsilon
                  m := m0t
                  v := v0t
                  t := previousStep }
              let (state', param') :=
                _root_.Optim.Adam.update (α := α) (s := pval.shape) state param gradWD
              let m' := state'.m
              let v' := state'.v
              parameterSteps := parameterSteps.insert p.id state'.t
              m := m.insert p.id (Spec.PackedTensor.ofTensor (Tensor.materialize m'))
              v := v.insert p.id (Spec.PackedTensor.ofTensor (Tensor.materialize v'))
              updated := { p with value := Spec.PackedTensor.ofTensor (Tensor.materialize param') } :: updated
          | .adamw =>
              let m0 := getOrInit m p.id pval
              let v0 := getOrInit v p.id pval
              let m0t ← castState "optim" p.id m0 pval
              let v0t ← castState "optim" p.id v0 pval
              let previousStep := previousAdamStep parameterSteps m v opt.step p.id
              let state : _root_.Optim.AdamW.State α pval.shape :=
                { lr := g.lr
                  beta1 := g.beta1
                  beta2 := g.beta2
                  epsilon := g.epsilon
                  weightDecay := g.weightDecay
                  m := m0t
                  v := v0t
                  t := previousStep }
              let (state', param') :=
                _root_.Optim.AdamW.update (α := α) (s := pval.shape) state param grad
              let m' := state'.m
              let v' := state'.v
              parameterSteps := parameterSteps.insert p.id state'.t
              m := m.insert p.id (Spec.PackedTensor.ofTensor (Tensor.materialize m'))
              v := v.insert p.id (Spec.PackedTensor.ofTensor (Tensor.materialize v'))
              updated := { p with value := Spec.PackedTensor.ofTensor (Tensor.materialize param') } :: updated
          | .adadelta =>
              let acc0 := getOrInit acc p.id pval
              let acc20 := getOrInit acc2 p.id pval
              let acc0t ← castState "optim" p.id acc0 pval
              let acc20t ← castState "optim" p.id acc20 pval
              let gradWD := addWeightDecay param grad g.weightDecay
              let state : _root_.Optim.Adadelta.State α pval.shape :=
                { lr := g.lr, rho := g.rho, epsilon := g.epsilon, v := acc0t, u := acc20t }
              let (state', param') :=
                _root_.Optim.Adadelta.update (α := α) (s := pval.shape) state param gradWD
              let acc' := state'.v
              let acc2' := state'.u
              acc := acc.insert p.id (Spec.PackedTensor.ofTensor (Tensor.materialize acc'))
              acc2 := acc2.insert p.id (Spec.PackedTensor.ofTensor (Tensor.materialize acc2'))
              updated := { p with value := Spec.PackedTensor.ofTensor (Tensor.materialize param') } :: updated
        else
          throw (tagError "optim" s!"gradient shape mismatch for id {p.id}")

  let opt' : OptimizerState α :=
    { kind := opt.kind
    , groups := groups'
    , step := tNext
    , parameterSteps := parameterSteps
    , momentumBuffer := momentumBuffer
    , m := m
    , v := v
    , acc := acc
    , acc2 := acc2
    }
  pure (opt', updated.reverse)

end Optim

end Train
end Autograd
end Runtime
