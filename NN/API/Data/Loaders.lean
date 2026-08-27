/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Data.Sources

/-!
# Internal Supervised Epochs

Typed collation used by trainer implementations and low-level manual loops. Application code
should construct a `Trainer.Dataset` and use `trainer.train`.
-/

@[expose] public section

namespace TorchLean
namespace Data

/--
Shape-typed epoch loader for supervised samples.

The batch size `n` appears in the type. A full epoch therefore returns collated tensors whose
leading dimension is `n`; constructing this loader requires `dropLast = true`.
-/
structure SupervisedEpochs (α : Type) (n : Nat)
    (inputShape targetShape : List Nat) where
  /-- Loader carrying the uncollated supervised samples. -/
  loader : EpochLoader (TorchLean.Sample.Supervised α inputShape targetShape)

/--
Collate `n` supervised samples into one sample with a leading batch axis.

For samples with input shape `inputShape` and target shape `targetShape`, the result contains:

- `xBatch : Tensor α (n :: inputShape)`;
- `yBatch : Tensor α (n :: targetShape)`.
-/
def collateSupervised {α : Type} {inputShape targetShape : List Nat} (n : Nat)
    (batch : Array (_root_.TorchLean.TensorPack α [inputShape, targetShape])) :
    Except String
      (_root_.TorchLean.TensorPack α [n :: inputShape, n :: targetShape]) := do
  if h : batch.size = n then
    let getSample : Fin n →
        _root_.TorchLean.TensorPack α [inputShape, targetShape] :=
      fun i =>
        let hlt : i.val < batch.size := by simp [h]
        batch[i.val]'hlt
    let xs : Tensor α (n :: inputShape) :=
      by
        simpa [Shape.insertAxis] using Tensor.stack 0 (fun i =>
          match getSample i with
          | .cons x _ => x)
    let ys : Tensor α (n :: targetShape) :=
      by
        simpa [Shape.insertAxis] using Tensor.stack 0 (fun i =>
          match getSample i with
          | .cons _x (.cons y .nil) => y)
    pure (_root_.TorchLean.TensorPack! xs, ys)
  else
    throw s!"collate: expected batch size {n}, got {batch.size}"

/--
Turn a per-sample supervised stream into a stream of fixed-size minibatches.

This is useful for metrics and manual loops when a model expects a leading
batch axis.

Notes:
- This drops the final partial batch (PyTorch `drop_last=True` behavior).
- Batches are formed in dataset order (shuffling is the loader's job).
-/
def collateStream {α : Type} {inputShape targetShape : List Nat} (n : Nat)
    (stream : _root_.TorchLean.Data.SampleStream
      (_root_.TorchLean.TensorPack α [inputShape, targetShape])) :
    Except String (_root_.TorchLean.Data.SampleStream
      (_root_.TorchLean.TensorPack α [n :: inputShape, n :: targetShape])) := do
  if n = 0 then
    throw "batched: batch size must be > 0"
  let groups ← _root_.TorchLean.Data.SampleStream.batches "batched" n stream
  let full := groups.filter (fun g => g.size = n)
  let batches ← full.mapM
    (collateSupervised (α := α) (inputShape := inputShape)
      (targetShape := targetShape) n)
  pure (_root_.TorchLean.Data.SampleStream.ofArray batches)

namespace SupervisedEpochs

/-- Run one epoch: return the updated loader state and an array of typed minibatches. -/
def epoch {α : Type} {n : Nat} {inputShape targetShape : List Nat}
    (name : String) (dl : SupervisedEpochs α n inputShape targetShape) :
    Except String (SupervisedEpochs α n inputShape targetShape ×
      Array (TorchLean.Sample.Batch α n inputShape targetShape)) := do
  if dl.loader.batchSize != n then
    throw s!"{name}: expected typed batch size {n}, got loader.batchSize={dl.loader.batchSize}"
  if !dl.loader.dropLast then
    throw s!"{name}: typed epochs require dropLast=true"
  let (raw', batches) ←
    _root_.TorchLean.Data.EpochLoader.epochCollate name dl.loader
      (fun batch => collateSupervised (α := α) (inputShape := inputShape)
        (targetShape := targetShape) n batch)
  pure ({ loader := raw' }, batches)

/-- Like `epoch`, but post-process each minibatch with a user-supplied collate/transform `f`. -/
def epochCollate {α β : Type} {n : Nat} {inputShape targetShape : List Nat}
    (name : String) (dl : SupervisedEpochs α n inputShape targetShape)
    (f : TorchLean.Sample.Batch α n inputShape targetShape → Except String β) :
    Except String (SupervisedEpochs α n inputShape targetShape × Array β) := do
  let (dl', batches) ← epoch (α := α) (inputShape := inputShape)
    (targetShape := targetShape) name dl
  pure (dl', ← batches.mapM f)

/--
Run one epoch and require at least one full typed minibatch.

This is the shared checked boundary for examples that need a nonempty array of full batches. It
keeps the "drop partial batches, but fail if nothing remains" policy with the loader API rather than
repeating it in each dataset-specific helper.
-/
def nonemptyEpoch {α : Type} {n : Nat} {inputShape targetShape : List Nat}
    (name : String) (dl : SupervisedEpochs α n inputShape targetShape) :
    Except String (SupervisedEpochs α n inputShape targetShape ×
      Array (TorchLean.Sample.Batch α n inputShape targetShape)) := do
  let (dl', batches) ← epoch (α := α) (inputShape := inputShape)
    (targetShape := targetShape) name dl
  if batches.isEmpty then
    throw s!"{name}: no full minibatch available (batch={n}, rows={dl.loader.samples.size})"
  else
    pure (dl', batches)

/-- Run one epoch and return its first full typed minibatch. -/
def firstFullBatch {α : Type} {n : Nat} {inputShape targetShape : List Nat}
    (name : String) (dl : SupervisedEpochs α n inputShape targetShape) :
    Except String (TorchLean.Sample.Batch α n inputShape targetShape) := do
  let (_dl', batches) ← nonemptyEpoch (α := α) (inputShape := inputShape)
    (targetShape := targetShape) name dl
  match batches[0]? with
  | some b => pure b
  | none => throw s!"{name}: internal error: expected a nonempty minibatch epoch"

end SupervisedEpochs

/--
Build fixed-size supervised epochs for a manual training loop.

The underlying dataset still stores individual samples; the loader batches them and `epoch`
returns tensors with a leading batch axis. Because the batch size is reflected in the type,
the public batched path requires full batches, so `dropLast` defaults to `true`.
-/
def supervisedEpochs {α : Type} {inputShape targetShape : List Nat}
    (ds : SampleStream (TorchLean.Sample.Supervised α inputShape targetShape))
    (batchSize : Nat) (shuffle : Bool := false) (seed : Nat := 0) (dropLast : Bool := true) :
    SupervisedEpochs α batchSize inputShape targetShape :=
  { loader := EpochLoader.create ds batchSize
      (shuffle := shuffle) (seed := seed) (dropLast := dropLast) }

/--
Load a numeric supervised CSV and wrap it as typed supervised epochs.

The CSV convention is the same as `TabularSupervisedSource`: each row contains `inDim` feature
columns followed by `outDim` target columns. This shared operation belongs in the data API rather
than an individual model file.
-/
def tabularCsvEpochs {α : Type} [_root_.Context α] [_root_.TorchLean.Runtime.FromFloat α]
    (path : System.FilePath) (batchSize inDim outDim : Nat)
    (csvOptions : CsvOptions := {}) (shuffle : Bool := true) (seed : Nat := 0)
    (dropLast : Bool := true) :
    IO (Except String (SupervisedEpochs α batchSize [inDim]
      [outDim])) := do
  let src : TabularSupervisedSource :=
    { path := path, inDim := inDim, outDim := outDim, csvOptions := csvOptions }
  let dsE ← src.load (α := α)
  match dsE with
  | .error e => pure (.error e)
  | .ok ds =>
      pure <| .ok <| supervisedEpochs (α := α) ds batchSize (shuffle := shuffle)
        (seed := seed) (dropLast := dropLast)

end Data
end TorchLean
