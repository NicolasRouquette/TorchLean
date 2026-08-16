/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.CLI
public import NN.API.Trainer.Manual.Core
public import NN.API.Trainer.Manual.Stepper
public import NN.API.Data.Loaders
public import NN.API.TensorPack
public import NN.API.Trainer.Reporting

/-!
# Manual Training

Dependent runners, callback composition, custom batch streams, and reporting helpers for workflows
that need more control than the ordinary `Trainer.new` interface.
-/

@[expose] public section

namespace TorchLean
namespace Trainer
namespace Manual

/-!
# Manual Training

Direct control over runners, callbacks, data streams, and optimization steps. The ordinary
`Trainer` interface is built from these definitions; custom training code can use them without
crossing into a second API namespace.
-/

/--
Count correct predictions in a one-hot labeled **batched** dataset.

Minibatch analogue of `accuracyOneHot`. The task already has a leading batch axis, so the
implementation scores each row independently and accumulates totals.

Returns `(correct, total)` where `total = batch * numBatches`.
-/
def Runner.accuracyOneHotBatch
    {σ : Spec.Shape} {classes batch : Nat}
    {task : SeqTask (.dim batch σ) (.dim batch (.dim classes .scalar))}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (runner : Runner α task)
    (samples : List (TorchLean.Sample.Batch α batch σ (.dim classes .scalar)))
      :
    IO (Nat × Nat) := do
  let mut correct : Nat := 0
  let mut total : Nat := 0
  for s in samples do
    let xBatch := TorchLean.Sample.x s
    let yBatch := TorchLean.Sample.y s
    let logitsBatch ← Runner.run (task := task) runner xBatch
    for i in List.finRange batch do
      let logits := Spec.getAtSpec logitsBatch i
      let target := Spec.getAtSpec yBatch i
      if let some true := _root_.TorchLean.Metrics.correctOneHot? logits target then
        correct := correct + 1
      total := total + 1
  pure (correct, total)

/-- Callback event fired after each training step. -/
structure StepEvent (α : Type) where
  /-- Current epoch number. -/
  epoch : Nat
  /-- Global optimizer-step counter. -/
  step : Nat
  /-- Loss reported for this step. -/
  loss : α

/-- Callback event fired at the end of an epoch (how many steps ran). -/
structure EpochEvent where
  /-- Epoch number that just completed. -/
  epoch : Nat
  /-- Number of steps executed in the epoch. -/
  steps : Nat

/--
Hooks for instrumenting callback-based training loops.

Callbacks are ordinary `IO` hooks. They can print progress, update an in-memory curve, sample CUDA
allocator state, or forward events to a project-specific metrics backend.
-/
structure Callbacks (α : Type) where
  /-- Called once before training starts. -/
  onTrainStart : IO Unit := pure ()
  /-- Called after each training step. -/
  onStep : StepEvent α → IO Unit := fun _ => pure ()
  /-- Called after each epoch. -/
  onEpochEnd : EpochEvent → IO Unit := fun _ => pure ()
  /-- Called once after training finishes. -/
  onTrainEnd : _root_.TorchLean.Trainer.Manual.TrainReport α → IO Unit := fun _ => pure ()

namespace Callbacks

/-- No-op callbacks. -/
def empty {α : Type} : Callbacks α := {}

/-- Combine two callback collections by running them in sequence. -/
def append {α : Type} (a b : Callbacks α) : Callbacks α :=
  { onTrainStart := do
      a.onTrainStart
      b.onTrainStart
    onStep := fun ev => do
      a.onStep ev
      b.onStep ev
    onEpochEnd := fun ev => do
      a.onEpochEnd ev
      b.onEpochEnd ev
    onTrainEnd := fun report => do
      a.onTrainEnd report
      b.onTrainEnd report
  }

/-- `∅` for callbacks: a no-op callback collection. -/
instance {α : Type} : EmptyCollection (Callbacks α) where
  emptyCollection := empty

/-- `Callbacks` form a monoid under sequential composition. -/
instance {α : Type} : Append (Callbacks α) where
  append := append

end Callbacks

/-- Build callbacks that run `action` once at the start of training. -/
def onTrainStart {α : Type} (action : IO Unit) : Callbacks α :=
  { onTrainStart := action }

/-- Build callbacks that observe every training step. -/
def onStep {α : Type} (f : StepEvent α → IO Unit) : Callbacks α :=
  { onStep := f }

/--
Build a training callback that samples the CUDA allocator at a fixed step cadence.

The callback owns a small `IO.Ref` for the previous sample, so examples can compose it with ordinary
loss-logging callbacks without threading allocator state through their training loops.
-/
def cudaMemWatchCallbacks {α : Type} (opts : _root_.Runtime.Autograd.Torch.Options)
    (watchEvery totalSteps : Nat) : IO (Callbacks α) := do
  let stateRef ← IO.mkRef (none : Option TorchLean.Trainer.Manual.CUDAMemory.State)
  pure <| onStep (α := α) (fun ev => do
    let state ← stateRef.get
    let state ←
      TorchLean.Trainer.Manual.CUDAMemory.sample
        opts watchEvery totalSteps (ev.step + 1) state
    stateRef.set state)

/-- Build callbacks that run at the end of each epoch. -/
def onEpochEnd {α : Type} (f : EpochEvent → IO Unit) : Callbacks α :=
  { onEpochEnd := f }

/-- Build callbacks that run once at the end of training, with the final report. -/
def onTrainEnd {α : Type} (f : _root_.TorchLean.Trainer.Manual.TrainReport α → IO Unit) : Callbacks α :=
  { onTrainEnd := f }

/--
Step-indexed source of already-collated module inputs.

`TorchLean.Data.batchLoader` is the right interface when the data is a finite supervised dataset.  Other
training jobs draw batches from a rule or an external source: replay buffers, collocation samplers,
synthetic scale inputs, or file-backed sequence windows. `StepBatchStream` is the direct stream
interface for those cases.

The stream is still fully typed: each produced sample is a `_root_.Runtime.Autograd.Torch.TList` matching the module's
`inputShapes`.  The training loop below is model-agnostic and only assumes that the module can run
`forward` and `optimizerStep` on those samples.
-/
structure StepBatchStream (α : Type) (inputShapes : List Spec.Shape) where
  /-- Produce the input sample used at logical optimizer step `step`. -/
  sample : Nat → IO (_root_.Runtime.Autograd.Torch.TList α inputShapes)

namespace StepBatchStream

/-- Constant stream for fixed-batch overfit runs and fixed-sample training jobs. -/
def fixed {α : Type} {inputShapes : List Spec.Shape}
    (x : _root_.Runtime.Autograd.Torch.TList α inputShapes) : StepBatchStream α inputShapes :=
  { sample := fun _ => pure x }

/-- Build a stream from a pure step-indexed sample function. -/
def ofFn {α : Type} {inputShapes : List Spec.Shape}
    (f : Nat → _root_.Runtime.Autograd.Torch.TList α inputShapes) : StepBatchStream α inputShapes :=
  { sample := fun step => pure (f step) }

/--
Cycle through a nonempty list of samples.

This adapter lets list-backed datasets use the step-stream trainer.  The explicit nonempty proof
keeps empty datasets from turning into silent modulo-by-zero behavior.
-/
def cycle {α : Type} {inputShapes : List Spec.Shape}
    (xs : List (_root_.Runtime.Autograd.Torch.TList α inputShapes)) (h : xs ≠ []) :
    StepBatchStream α inputShapes :=
  match xs with
  | [] => False.elim (h rfl)
  | x :: rest =>
      let ys := x :: rest
      { sample := fun step => pure ((TorchLean.Data.cycleList ys (by simp)) step) }

end StepBatchStream

/--
Run an action with the runner temporarily switched to `value` mode.

Use this for callback-based validation passes during training.
-/
def Runner.withMode {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    {β : Type} (runner : Runner α task) (value : _root_.Runtime.Autograd.TorchLean.NN.Mode) (action : IO β) : IO β := do
  let prev ← TorchLean.Trainer.Manual.Runner.currentMode runner
  TorchLean.Trainer.Manual.Runner.setMode runner value
  try
    action
  finally
    TorchLean.Trainer.Manual.Runner.setMode runner prev

/--
Mean loss for an already-instantiated scalar module over a typed minibatch loader.

General streaming evaluation path used by the runtime examples. It is not CIFAR-specific: any
supervised task whose objective consumes
`[dim n σ, dim n τ]` can use the same loader.  The loader stores ordinary per-example samples
`(x : σ, y : τ)`; this definition asks `TorchLean.Data.epoch` for raw minibatches and calls
`TorchLean.Data.collateSupervised` to build one shape-typed batch at a time.

Two details matter for larger examples:

- We force `shuffle := false` for evaluation so before/after metrics are deterministic.
- We do not call `TorchLean.Data.BatchLoader.batchDataset`, because that would materialize every collated
  minibatch at once.  Streaming keeps the same API usable for image, sequence, and scientific ML
  examples where the batch tensors are much larger than small tabular datasets.
-/
def Objective.meanLoss {σ τ : Spec.Shape} {n : Nat} {stateShapes : List Spec.Shape}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    (module : TorchLean.Module.Objective α stateShapes [Spec.Shape.dim n σ, Spec.Shape.dim n τ])
    (loader : TorchLean.Data.BatchLoader α n σ τ) : IO α := do
  let evalLoader : TorchLean.Data.DataLoader (TorchLean.Sample.Supervised α σ τ) :=
    { loader.loader with shuffle := false, dropLast := true }
  let (_dlNext, rawBatches) ←
    match TorchLean.Data.epoch "train.scalarModule.meanLoss" evalLoader with
    | Except.ok out => pure out
    | Except.error msg => throw <| IO.userError s!"train.scalarModule.meanLoss: {msg}"
  let mut total : α := 0
  let mut count : Nat := 0
  for rawBatch in rawBatches do
    let sample ← TorchLean.CLI.orThrow "train.scalarModule.meanLoss" <|
      TorchLean.Data.collateSupervised (α := α) (σ := σ) (τ := τ) n rawBatch
    let lossTensor ← TorchLean.Module.loss module sample .nil
    let loss := Spec.Tensor.toScalar lossTensor
    total := total + loss
    count := count + 1
  if count = 0 then
    pure 0
  else
    pure (total / (count : α))

/--
Mean loss over a typed minibatch loader through a `Trainer.Manual.Runner`.

Runner-facing form of `Objective.meanLoss`. Use it when the example is built around
`Trainer.Manual.run`, task modes, and the proof layer trainer abstraction. Use
`Objective.meanLoss` directly when the example has already instantiated a runtime
`TorchLean.Module.Objective`, which is the common fast path for CUDA examples.
-/
def Runner.meanLossLoader {σ τ : Spec.Shape} {n : Nat}
    {task : SeqTask (.dim n σ) (.dim n τ)}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    (runner : Runner α task) (loader : TorchLean.Data.BatchLoader α n σ τ) : IO α :=
  Objective.meanLoss runner.module loader

/-- One-hot accuracy over a typed minibatch loader without materializing all collated batches. -/
def Runner.accuracyOneHotLoader
    {σ : Spec.Shape} {classes batch : Nat}
    {task : SeqTask (.dim batch σ) (.dim batch (.dim classes .scalar))}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape]
    (runner : Runner α task)
    (loader : TorchLean.Data.BatchLoader α batch σ (.dim classes .scalar)) :
    IO (Nat × Nat) := do
  let evalLoader : TorchLean.Data.DataLoader
      (TorchLean.Sample.Supervised α σ (.dim classes .scalar)) :=
    { loader.loader with shuffle := false, dropLast := true }
  let (_dlNext, rawBatches) ←
    match TorchLean.Data.epoch "Runner.accuracyOneHotLoader" evalLoader with
    | Except.ok out => pure out
    | Except.error msg => throw <| IO.userError s!"Runner.accuracyOneHotLoader: {msg}"
  let mut correct : Nat := 0
  let mut total : Nat := 0
  for rawBatch in rawBatches do
    let sample ← TorchLean.CLI.orThrow "Runner.accuracyOneHotLoader" <|
      TorchLean.Data.collateSupervised (α := α) (σ := σ) (τ := .dim classes .scalar) batch rawBatch
    let (c, t) ← runner.accuracyOneHotBatch [sample]
    correct := correct + c
    total := total + t
  pure (correct, total)

/--
Train a runtime scalar module from a typed minibatch loader.

Shared real epoch loop for model examples that already have a runtime module, including CUDA runs.
It mirrors the PyTorch structure:

1. create an optimizer state for the module parameters;
2. for each epoch, ask the general `TorchLean.Data.batchLoader` for shuffled raw batches;
3. collate each raw batch into a shape-typed `(xBatch, yBatch)` sample;
4. report the scalar loss through callbacks;
5. run `forward/backward/optimizer.step` through `TorchLean.Module.optimizerStep`.

The function is polymorphic in the input shape `σ`, target shape `τ`, batch size `n`, scalar type
`α`, parameter shapes, and optimizer. It is not image-specific. CNN, ResNet, ViT, MLP,
sequence, operator-learning, and future model examples should all be able to use this path whenever
their supervised objective has input shapes `[dim n σ, dim n τ]`.
-/
def Objective.trainLoader {σ τ : Spec.Shape} {n : Nat} {stateShapes : List Spec.Shape}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    (module : TorchLean.Module.Objective α stateShapes [Spec.Shape.dim n σ, Spec.Shape.dim n τ])
    (optimizer : _root_.Runtime.Autograd.TorchLean.Optim.Optimizer α stateShapes)
    (epochs : Nat)
    (loader : TorchLean.Data.BatchLoader α n σ τ)
    (callbacks : Callbacks α := Callbacks.empty) :
    IO (_root_.TorchLean.Trainer.Manual.TrainReport α × TorchLean.Data.BatchLoader α n σ τ) := do
  let before ← Objective.meanLoss module loader
  callbacks.onTrainStart

  let mut optState ← TorchLean.Module.initOptimizer module optimizer
  let mut dl := loader
  let mut globalStep : Nat := 0

  for epochIdx in [0:epochs] do
    let (rawNext, rawBatches) ←
      match TorchLean.Data.epoch "train.scalarModule.trainLoader" dl.loader with
      | Except.ok out => pure out
      | Except.error msg => throw <| IO.userError s!"train.scalarModule.trainLoader: {msg}"
    dl := { loader := rawNext }
    for rawBatch in rawBatches do
      let sample ← TorchLean.CLI.orThrow "train.scalarModule.trainLoader" <|
        TorchLean.Data.collateSupervised (α := α) (σ := σ) (τ := τ) n rawBatch
      let lossTensor ← TorchLean.Module.loss module sample .nil
      let loss := Spec.Tensor.toScalar lossTensor
      callbacks.onStep { epoch := epochIdx, step := globalStep, loss := loss }
      optState ← TorchLean.Module.optimizerStep module optimizer optState sample .nil
      globalStep := globalStep + 1
    callbacks.onEpochEnd { epoch := epochIdx, steps := globalStep }

  let after ← Objective.meanLoss module dl
  let report := { before := before, after := after }
  callbacks.onTrainEnd report
  pure (report, dl)

/--
Train a runtime scalar module for exactly `steps` optimizer updates.

`Objective.trainLoader` above is epoch-based: each unit means one full pass over the loader. This
variant is update-based, which is the convention used by runnable examples that expose a `--steps`
flag.

The loop still draws shuffled minibatches from `TorchLean.Data.batchLoader` epoch by epoch, but it stops as
soon as the requested number of optimizer updates has run. The returned loader is the next loader
state, so callers can continue training from the next shuffled epoch if they want to.
-/
def Objective.trainLoaderSteps
    {σ τ : Spec.Shape} {n : Nat} {stateShapes : List Spec.Shape}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    (module : TorchLean.Module.Objective α stateShapes [Spec.Shape.dim n σ, Spec.Shape.dim n τ])
    (optimizer : _root_.Runtime.Autograd.TorchLean.Optim.Optimizer α stateShapes)
    (steps : Nat)
    (loader : TorchLean.Data.BatchLoader α n σ τ)
    (callbacks : Callbacks α := Callbacks.empty) :
    IO (_root_.TorchLean.Trainer.Manual.TrainReport α × TorchLean.Data.BatchLoader α n σ τ) := do
  let before ← Objective.meanLoss module loader
  callbacks.onTrainStart

  let mut optState ← TorchLean.Module.initOptimizer module optimizer
  let mut dl := loader
  let mut globalStep : Nat := 0
  let mut epochIdx : Nat := 0

  while globalStep < steps do
    let (rawNext, rawBatches) ←
      match TorchLean.Data.epoch "train.scalarModule.trainLoaderSteps" dl.loader with
      | Except.ok out => pure out
      | Except.error msg => throw <| IO.userError s!"train.scalarModule.trainLoaderSteps: {msg}"
    if rawBatches.isEmpty then
      throw <| IO.userError "train.scalarModule.trainLoaderSteps: loader produced no batches"
    dl := { loader := rawNext }
    let epochStart := globalStep
    for rawBatch in rawBatches do
      if globalStep < steps then
        let sample ← TorchLean.CLI.orThrow "train.scalarModule.trainLoaderSteps" <|
          TorchLean.Data.collateSupervised (α := α) (σ := σ) (τ := τ) n rawBatch
        let lossTensor ← TorchLean.Module.loss module sample .nil
        let loss := Spec.Tensor.toScalar lossTensor
        callbacks.onStep { epoch := epochIdx, step := globalStep, loss := loss }
        optState ← TorchLean.Module.optimizerStep module optimizer optState sample .nil
        globalStep := globalStep + 1
    callbacks.onEpochEnd { epoch := epochIdx, steps := globalStep - epochStart }
    epochIdx := epochIdx + 1

  let after ← Objective.meanLoss module dl
  let report := { before := before, after := after }
  callbacks.onTrainEnd report
  pure (report, dl)

/--
Train a scalar module from a step-indexed batch stream.

Shared loop for workloads whose batches are produced step by step rather than by one finite
`TorchLean.Data.batchLoader` epoch:

- RL algorithms can sample replay or rollout batches,
- PDE examples can resample collocation points,
- generated workloads can stream synthetic inputs without storing a dataset.

The function is generic in `inputShapes`. It does not know whether the sample is
`[x, y]`, `[state, action, target]`, or `[]`; it only asks the stream for the next typed input list
and then runs the same `forward/backward/optimizer.step` machinery as the loader-based trainer.
-/
def Objective.trainStream {inputShapes : List Spec.Shape} {stateShapes : List Spec.Shape}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    (module : TorchLean.Module.Objective α stateShapes inputShapes)
    (optimizer : _root_.Runtime.Autograd.TorchLean.Optim.Optimizer α stateShapes)
    (steps : Nat)
    (stream : StepBatchStream α inputShapes)
    (callbacks : Callbacks α := Callbacks.empty) :
    IO (_root_.TorchLean.Trainer.Manual.TrainReport α) := do
  let sample0 ← stream.sample 0
  let beforeTensor ← TorchLean.Module.loss module sample0 .nil
  let before := Spec.Tensor.toScalar beforeTensor
  callbacks.onTrainStart

  let mut optState ← TorchLean.Module.initOptimizer module optimizer

  for step in [0:steps] do
    let sample ← stream.sample step
    let lossTensor ← TorchLean.Module.loss module sample .nil
    let loss := Spec.Tensor.toScalar lossTensor
    callbacks.onStep { epoch := 0, step := step, loss := loss }
    optState ← TorchLean.Module.optimizerStep module optimizer optState sample .nil

  callbacks.onEpochEnd { epoch := 0, steps := steps }
  let sampleAfter ← stream.sample steps
  let afterTensor ← TorchLean.Module.loss module sampleAfter .nil
  let after := Spec.Tensor.toScalar afterTensor
  let report := { before := before, after := after }
  callbacks.onTrainEnd report
  pure report

/--
Report-oriented stream-training entrypoint.

Callers pass the module, optimizer, runtime options, step count, and stream, and get standard
before/after reporting plus CUDA memory watching.
-/
def Objective.trainStreamReport
    {inputShapes : List Spec.Shape} {stateShapes : List Spec.Shape}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    (module : TorchLean.Module.Objective α stateShapes inputShapes)
    (optimizer : _root_.Runtime.Autograd.TorchLean.Optim.Optimizer α stateShapes)
    (opts : _root_.Runtime.Autograd.Torch.Options)
    (steps : Nat)
    (stream : StepBatchStream α inputShapes)
    (cudaMemWatch : Nat := 0)
    (extraCallbacks : Callbacks α := Callbacks.empty) :
    IO (_root_.TorchLean.Trainer.Manual.TrainReport α) := do
  let watchEvery := TorchLean.Trainer.Manual.CUDAMemory.cadence opts steps cudaMemWatch
  let memHooks ← cudaMemWatchCallbacks (α := α) opts watchEvery steps
  let hooks : Callbacks α :=
    memHooks
    ++ extraCallbacks
    ++ onTrainEnd (α := α) (fun report =>
      IO.println s!"  steps={steps} loss_before={report.before} loss_after={report.after}")
  Objective.trainStream module optimizer steps stream hooks

/--
Float stream trainer that records a per-step loss curve.

Generated and file-backed batches do not always have one finite loader to summarize. This entrypoint
keeps their training curves in the same JSON format as the supervised examples.
-/
def Objective.trainStreamCurve
    {inputShapes : List Spec.Shape} {stateShapes : List Spec.Shape}
    (module : TorchLean.Module.Objective Float stateShapes inputShapes)
    (optimizer : _root_.Runtime.Autograd.TorchLean.Optim.Optimizer Float stateShapes)
    (opts : _root_.Runtime.Autograd.Torch.Options)
    (steps : Nat)
    (stream : StepBatchStream Float inputShapes)
    (cudaMemWatch : Nat := 0)
    (extraCallbacks : Callbacks Float := Callbacks.empty) :
    IO (_root_.TorchLean.Trainer.Manual.TrainReport Float × _root_.Runtime.Training.Curve) := do
  let curveRef ← IO.mkRef ({} : _root_.Runtime.Training.Curve)
  let curveHooks : Callbacks Float :=
    onStep (α := Float) (fun ev =>
      curveRef.modify (fun c => c.push ev.step ev.loss))
  let report ← Objective.trainStreamReport module optimizer opts steps stream cudaMemWatch
    (extraCallbacks ++ curveHooks)
  let curve ← curveRef.get
  pure (report, curve)

/--
Train from a runner-backed loader with explicit callbacks instead of inline printing in example
code.

Runner-facing public path for PyTorch-style custom loops:
- keep the optimizer/scheduler logic in the library,
- inject logging, evaluation, and prediction reporting through callbacks.

This path keeps the `Runner` abstraction, including task modes and scheduler support.  For
CUDA-heavy entrypoints that already have a `TorchLean.Module.Objective`, prefer
`Objective.trainLoader`; both paths consume the same general `TorchLean.Data.batchLoader`.
-/
def Runner.trainLoader {σ τ : Spec.Shape} {n : Nat}
    {task : SeqTask (.dim n σ) (.dim n τ)}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α] [_root_.TorchLean.Runtime.FromFloat α]
    (runner : Runner α task) (cfg : _root_.TorchLean.Trainer.Manual.LoaderTrainConfig)
    (loader : TorchLean.Data.BatchLoader α n σ τ)
    (callbacks : Callbacks α := Callbacks.empty) :
    IO (_root_.TorchLean.Trainer.Manual.TrainReport α × TorchLean.Data.BatchLoader α n σ τ) := do
  Runner.eval runner
  let before ← runner.meanLossLoader loader
  callbacks.onTrainStart

  Runner.train runner
  let loop ← _root_.TorchLean.Trainer.Manual.stepper (task := task) runner cfg.optimizer (scheduler :=
    cfg.scheduler)
  let mut dl := loader
  let mut globalStep : Nat := 0

  for epochIdx in [0:cfg.epochs] do
    let (rawNext, rawBatches) ←
      match TorchLean.Data.epoch "Runner.trainLoader" dl.loader with
      | Except.ok out => pure out
      | Except.error msg => throw <| IO.userError s!"Runner.trainLoader: {msg}"
    dl := { loader := rawNext }
    for rawBatch in rawBatches do
      let sample ← TorchLean.CLI.orThrow "Runner.trainLoader" <|
        TorchLean.Data.collateSupervised (α := α) (σ := σ) (τ := τ) n rawBatch
      let loss ← _root_.TorchLean.Trainer.Manual.step (task := task) loop sample
      callbacks.onStep { epoch := epochIdx, step := globalStep, loss := loss }
      globalStep := globalStep + 1
    callbacks.onEpochEnd { epoch := epochIdx, steps := globalStep }

  Runner.eval runner
  let after ← runner.meanLossLoader dl
  let report := { before := before, after := after }
  callbacks.onTrainEnd report
  pure (report, dl)

namespace Report

/-!
### Small Reporting Helpers (IO)

These definitions factor out common "print a loss/accuracy table" patterns for runnable model
commands.
They do not affect semantics: they only call the underlying runner functions and print
human-facing summaries. Public examples should reach them through `Trainer.Manual` only when the
ordinary `Trainer.new` / `trainer.train` API is too small for the example.
-/

/-- Print a titled list of named report lines. -/
def probes {β : Type} (title : String) (values : List β) (lineOf : β → IO String) : IO Unit :=
  do
  IO.println title
  for value in values do
    IO.println (← lineOf value)

/-- Convenience: mean loss on a dataset, printed with a label. -/
def meanLoss
    {σ τ : Spec.Shape} {task : SeqTask σ τ}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    (runner : Runner α task)
    (dataset : _root_.Runtime.Autograd.Train.Dataset (TorchLean.Sample.Supervised α σ τ))
    (label : String) : IO Unit := do
  let loss ← _root_.TorchLean.Trainer.Manual.Runner.meanLossDataset (task := task) runner dataset
  IO.println s!"mean_loss({label}) = {loss}"

/-- Convenience: mean loss on a typed minibatch loader, streamed batch by batch. -/
def meanLossLoader
    {σ τ : Spec.Shape} {batch : Nat} {task : SeqTask (.dim batch σ) (.dim batch τ)}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    (runner : Runner α task)
    (loader : TorchLean.Data.BatchLoader α batch σ τ)
    (label : String) : IO Unit := do
  let loss ← runner.meanLossLoader loader
  IO.println s!"mean_loss({label}) = {loss}"

/--
Convenience: mean loss on a typed minibatch loader for an already-instantiated runtime module.

Use this in direct CUDA/runtime examples to avoid building a `Runner` only for logging.  The data
path is still the same public loader path: `TorchLean.Data.batchLoader` plus `TorchLean.Data.collateSupervised`.
-/
def Objective.meanLoss
    {σ τ : Spec.Shape} {batch : Nat} {stateShapes : List Spec.Shape}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    (module : TorchLean.Module.Objective α stateShapes [Spec.Shape.dim batch σ, Spec.Shape.dim
      batch τ])
    (loader : TorchLean.Data.BatchLoader α batch σ τ)
    (label : String) : IO Unit := do
  let loss ← TorchLean.Trainer.Manual.Objective.meanLoss module loader
  IO.println s!"mean_loss({label}) = {loss}"

/--
Report predicted classes on a list of named inputs.

Each entry is `(name, x, expectedClass)`.
If `includeLogits := true`, also prints the raw model outputs.
-/
def classProbes
    {σ : Spec.Shape} {classes : Nat} {task : SeqTask σ (.dim classes .scalar)}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    (runner : Runner α task)
    (probes : List (String × Spec.Tensor α σ × Nat))
    (title : String := "predictions")
    (includeLogits : Bool := false) : IO Unit := do
  Report.probes title probes (fun (name, x, expected) => do
    let logits ← Runner.run (task := task) runner x
    let pred? := _root_.TorchLean.Metrics.argmax? logits
    let predStr :=
      match pred? with
      | some k => toString k.val
      | none => "none"
    let logitsStr :=
      if includeLogits then
        s!" logits={Spec.pretty logits}"
      else
        ""
    pure s!"  {name}: expected={expected} predicted={predStr}{logitsStr}")

/--
Report predicted classes on a list of named inputs, for a **batched** model.

This expects inputs of the *unbatched* input shape `σ` and replicates each one across the batch
axis, then reports the prediction for row 0.
-/
def classProbesBatch
    {σ : Spec.Shape} {classes batch : Nat} {task : SeqTask (.dim batch σ) (.dim batch
      (.dim classes .scalar))}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    (runner : Runner α task)
    (probes : List (String × Spec.Tensor α σ × Nat))
    (title : String := "predictions")
    (includeLogits : Bool := false) : IO Unit := do
  Report.probes title probes (fun (name, xSingle, expected) => do
    let xBatch : Spec.Tensor α (.dim batch σ) :=
      Spec.Tensor.dim (fun _ => xSingle)
    let logitsBatch ← Runner.run (task := task) runner xBatch
    -- If `batch = 0`, there is no row to display. That case is not meaningful for training anyway.
    match List.finRange batch with
    | [] =>
        pure s!"  {name}: batch=0 (no prediction)"
    | i0 :: _ =>
        let logits0 := Spec.getAtSpec logitsBatch i0
        let pred? := _root_.TorchLean.Metrics.argmax? logits0
        let predStr :=
          match pred? with
          | some k => toString k.val
          | none => "none"
        let logitsStr :=
          if includeLogits then
            s!" logits={Spec.pretty logits0}"
          else
            ""
        pure s!"  {name}: expected={expected} predicted={predStr}{logitsStr}")

/-- Convenience: mean loss + one-hot accuracy on a dataset, printed with a label. -/
def oneHotMetrics
    {σ : Spec.Shape} {classes : Nat} {task : SeqTask σ (.dim classes .scalar)}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    (runner : Runner α task)
    (dataset : _root_.Runtime.Autograd.Train.Dataset
      (TorchLean.Sample.Supervised α σ (.dim classes .scalar)))
    (label : String) : IO Unit := do
  let loss ← _root_.TorchLean.Trainer.Manual.Runner.meanLossDataset (task := task) runner dataset
  let (correct, total) ← Runner.accuracyOneHot (task := task) runner dataset.toList
  IO.println s!"mean_loss({label}) = {loss}"
  IO.println s!"accuracy({label}) = {correct}/{total}"

/-- Batched variant of `oneHotMetrics`. -/
def oneHotMetricsBatch
    {σ : Spec.Shape} {classes batch : Nat}
    {task : SeqTask (.dim batch σ) (.dim batch (.dim classes .scalar))}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    (runner : Runner α task)
    (dataset : _root_.Runtime.Autograd.Train.Dataset
      (TorchLean.Sample.Batch α batch σ (.dim classes .scalar)))
    (label : String) : IO Unit := do
  let loss ← _root_.TorchLean.Trainer.Manual.Runner.meanLossDataset (task := task) runner dataset
  let (correct, total) ← runner.accuracyOneHotBatch dataset.toList
  IO.println s!"mean_loss({label}) = {loss}"
  IO.println s!"accuracy({label}) = {correct}/{total}"

/-- Loader variant of `oneHotMetricsBatch`, streaming through minibatches. -/
def oneHotMetricsLoader
    {σ : Spec.Shape} {classes batch : Nat}
    {task : SeqTask (.dim batch σ) (.dim batch (.dim classes .scalar))}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    (runner : Runner α task)
    (loader : TorchLean.Data.BatchLoader α batch σ (.dim classes .scalar))
    (label : String) : IO Unit := do
  let loss ← runner.meanLossLoader loader
  let (correct, total) ← runner.accuracyOneHotLoader loader
  IO.println s!"mean_loss({label}) = {loss}"
  IO.println s!"accuracy({label}) = {correct}/{total}"

end Report

/--
Train a runtime module for a fixed number of optimizer updates with the standard runtime reports.

Common path for direct-module training, not example-only code. It composes the
generic step loop with before/after mean-loss reporting and CUDA allocator telemetry, while still
accepting extra callbacks for projects that want their own metrics, validation, or tracing.
-/
def Objective.trainLoaderReport
    {σ τ : Spec.Shape} {n : Nat} {stateShapes : List Spec.Shape}
    {α : Type} [_root_.Context α] [DecidableEq Spec.Shape] [ToString α]
    (module : TorchLean.Module.Objective α stateShapes [Spec.Shape.dim n σ, Spec.Shape.dim n τ])
    (optimizer : _root_.Runtime.Autograd.TorchLean.Optim.Optimizer α stateShapes)
    (opts : _root_.Runtime.Autograd.Torch.Options)
    (steps : Nat)
    (loader : TorchLean.Data.BatchLoader α n σ τ)
    (cudaMemWatch : Nat := 0)
    (extraCallbacks : Callbacks α := Callbacks.empty) :
    IO (_root_.TorchLean.Trainer.Manual.TrainReport α × TorchLean.Data.BatchLoader α n σ τ) := do
  let watchEvery := TorchLean.Trainer.Manual.CUDAMemory.cadence opts steps cudaMemWatch
  let memHooks ← cudaMemWatchCallbacks (α := α) opts watchEvery steps
  let hooks : Callbacks α :=
    (onTrainStart (α := α) do
      Report.Objective.meanLoss module loader "train(before)")
    ++ extraCallbacks
    ++ memHooks
    ++ onTrainEnd (α := α) (fun _ =>
      Report.Objective.meanLoss module loader "train(after)")
  Objective.trainLoaderSteps module optimizer steps loader hooks

/--
Float-specialized module training that also records a scalar loss curve.

The training loop itself is the same as `Objective.trainLoaderReport`; this entrypoint adds the
standard `Curve` callback used by JSON logs and website widgets.
-/
def Objective.trainLoaderCurve {σ τ : Spec.Shape} {n : Nat}
    {stateShapes : List Spec.Shape}
    (module : TorchLean.Module.Objective Float stateShapes [Spec.Shape.dim n σ,
      Spec.Shape.dim n τ])
    (optimizer : _root_.Runtime.Autograd.TorchLean.Optim.Optimizer Float stateShapes)
    (opts : _root_.Runtime.Autograd.Torch.Options)
    (steps : Nat)
    (loader : TorchLean.Data.BatchLoader Float n σ τ)
    (cudaMemWatch : Nat := 0)
    (extraCallbacks : Callbacks Float := Callbacks.empty) :
    IO (_root_.TorchLean.Trainer.Manual.TrainReport Float × TorchLean.Data.BatchLoader Float n σ τ ×
      _root_.Runtime.Training.Curve) := do
  let curveRef ← IO.mkRef ({} : _root_.Runtime.Training.Curve)
  let curveHooks : Callbacks Float :=
    onStep (α := Float) (fun ev => curveRef.modify (fun c => c.push ev.step ev.loss))
  let (report, loader') ← Objective.trainLoaderReport module optimizer opts steps loader
    cudaMemWatch (extraCallbacks ++ curveHooks)
  let curve ← curveRef.get
  pure (report, loader', curve)

/--
Train a Float runtime module, write a standard scalar-curve log, and return the train report.

High-level path used by runnable training commands. The caller provides the model, optimizer,
loader, runtime options, and metadata notes; the library owns the callback composition, CUDA
telemetry, before/after reports, and JSON curve emission.
-/
def Objective.trainLoaderLogged {σ τ : Spec.Shape} {n : Nat}
    {stateShapes : List Spec.Shape}
    (module : TorchLean.Module.Objective Float stateShapes [Spec.Shape.dim n σ,
      Spec.Shape.dim n τ])
    (optimizer : _root_.Runtime.Autograd.TorchLean.Optim.Optimizer Float stateShapes)
    (opts : _root_.Runtime.Autograd.Torch.Options)
    (steps : Nat)
    (loader : TorchLean.Data.BatchLoader Float n σ τ)
    (log : _root_.Runtime.Training.LogDestination)
    (title : String)
    (notes : Array String := #[])
    (seriesName : String := "loss")
    (cudaMemWatch : Nat := 0) :
    IO (_root_.TorchLean.Trainer.Manual.TrainReport Float × TorchLean.Data.BatchLoader Float n σ τ) := do
  let (report, loader', curve) ←
    Objective.trainLoaderCurve module optimizer opts steps loader
    cudaMemWatch
  TorchLean.Training.Curve.writeLogTo curve log title seriesName notes
  pure (report, loader')

end Manual
end Trainer
end TorchLean
