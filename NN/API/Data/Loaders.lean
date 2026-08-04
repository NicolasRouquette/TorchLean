/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Data.Sources

/-!
# Dataset Collation and Loaders

In-memory collation, fixed-size batching, and epoch traversal.
-/

@[expose] public section

namespace TorchLean
namespace Data

/--
Build a supervised dataset from two matrices `X : n×inDim` and `Y : n×outDim` by pairing rows.
This is the simple regression case of a tensor dataset.
-/
def supervisedRows {α : Type} [_root_.Context α] [_root_.TorchLean.Runtime.FromFloat α]
    {n inDim outDim : Nat}
    (X : Spec.Tensor Float (.dim n (.dim inDim .scalar)))
    (Y : Spec.Tensor Float (.dim n (.dim outDim .scalar))) :
    Dataset (_root_.TorchLean.TensorPack α [.dim inDim .scalar, .dim outDim .scalar]) :=
  supervisedFromLeadingAxisFloat (α := α) X Y

/--
Collate a length-`n` supervised batch into a single sample with a leading batch axis.

If your samples are `(x : σ, y : τ)`, the collated sample is:

- `xBatch : (n × σ)` and
- `yBatch : (n × τ)`

In shapes: `TensorPack α [dim n σ, dim n τ]`.
-/
def collateSupervised {α : Type} {σ τ : Spec.Shape} (n : Nat)
    (batch : List (_root_.TorchLean.TensorPack α [σ, τ])) :
    Except String (_root_.TorchLean.TensorPack α [Spec.Shape.dim n σ, Spec.Shape.dim n τ]) := do
  if h : batch.length = n then
    let getSample : Fin n → _root_.TorchLean.TensorPack α [σ, τ] :=
      fun i =>
        let hlt : i.val < batch.length := by simp [h]
        batch.get ⟨i.val, hlt⟩
    let xs : Spec.Tensor α (Spec.Shape.dim n σ) :=
      _root_.Spec.Tensor.dim (fun i =>
        match getSample i with
        | .cons x _ => x)
    let ys : Spec.Tensor α (Spec.Shape.dim n τ) :=
      _root_.Spec.Tensor.dim (fun i =>
        match getSample i with
        | .cons _x (.cons y .nil) => y)
    pure (tensorpack! xs, ys)
  else
    throw s!"collate: expected batch size {n}, got {batch.length}"

/-- Split a list into consecutive length-`n` chunks, dropping any final short chunk. -/
def chunkN {a : Type} (n : Nat) (xs : List a) : List (List a) :=
  if _h : n = 0 then
    []
  else
    -- Use an explicit fuel so Lean sees termination; `take/drop` recursion doesn't.
    let rec go (xs : List a) (fuel : Nat) : List (List a) :=
      match fuel with
      | 0 => []
      | fuel + 1 =>
          if xs.length < n then
            []
          else
            let b := xs.take n
            let rest := xs.drop n
            b :: go rest fuel
    go xs xs.length

/--
Turn a per-sample supervised dataset into a dataset of fixed-size minibatches.

This is useful for metrics (`meanLossDataset`, accuracy, etc.) when your model expects a leading
batch axis.

Notes:
- This drops the final partial batch (PyTorch `drop_last=True` behavior).
- Batches are formed in dataset order (shuffling is the loader's job).
-/
def batchedSupervised {α : Type} {σ τ : Spec.Shape} (n : Nat)
    (ds : _root_.Runtime.Autograd.Train.Dataset (_root_.TorchLean.TensorPack α [σ, τ])) :
    Except String (_root_.Runtime.Autograd.Train.Dataset (_root_.TorchLean.TensorPack α [Spec.Shape.dim n σ,
      Spec.Shape.dim n τ])) := do
  if n = 0 then
    throw "batched: batch size must be > 0"
  let groups := chunkN n (_root_.Runtime.Autograd.Train.Dataset.toList ds)
  let full := groups.filter (fun g => g.length = n)
  let batches ← full.mapM (collateSupervised (α := α) (σ := σ) (τ := τ) n)
  pure (fromList batches)

namespace BatchLoader

/-- Extract the underlying per-sample dataset from a typed `BatchLoader`. -/
def dataset {α : Type} {n : Nat} {σ τ : Spec.Shape}
    (dl : BatchLoader α n σ τ) :
    Dataset (TorchLean.Sample.Supervised α σ τ) :=
  dl.raw.dataset

/-- The batch size `n` carried in the type of a `BatchLoader`. -/
def batchSize {α : Type} {n : Nat} {σ τ : Spec.Shape}
    (_dl : BatchLoader α n σ τ) : Nat :=
  n

/-- Whether the loader is configured to shuffle samples each epoch. -/
def shuffled {α : Type} {n : Nat} {σ τ : Spec.Shape}
    (dl : BatchLoader α n σ τ) : Bool :=
  dl.raw.shuffle

/-- RNG seed used for shuffling (if enabled). -/
def seed {α : Type} {n : Nat} {σ τ : Spec.Shape}
    (dl : BatchLoader α n σ τ) : Nat :=
  dl.raw.seed

/-- Materialize the dataset as a dataset of full minibatches (dropping any final partial batch). -/
def batchDataset {α : Type} {n : Nat} {σ τ : Spec.Shape}
    (dl : BatchLoader α n σ τ) :
    Except String (Dataset (TorchLean.Sample.Batch α n σ τ)) :=
  batchedSupervised (α := α) (σ := σ) (τ := τ) n dl.raw.dataset

/-- Run one epoch: return the updated loader state and a list of typed minibatches. -/
def epoch {α : Type} {n : Nat} {σ τ : Spec.Shape}
    (name : String) (dl : BatchLoader α n σ τ) :
    Except String (BatchLoader α n σ τ × List (TorchLean.Sample.Batch α n σ τ)) := do
  if dl.raw.batchSize != n then
    throw s!"{name}: expected typed batch size {n}, got loader.batchSize={dl.raw.batchSize}"
  if !dl.raw.dropLast then
    throw s!"{name}: BatchLoader requires dropLast=true"
  let (raw', batches) ←
    _root_.Runtime.Autograd.Train.DataLoader.epochCollate name dl.raw
      (fun batch => collateSupervised (α := α) (σ := σ) (τ := τ) n batch)
  pure ({ raw := raw' }, batches)

/-- Like `epoch`, but post-process each minibatch with a user-supplied collate/transform `f`. -/
def epochCollate {α β : Type} {n : Nat} {σ τ : Spec.Shape}
    (name : String) (dl : BatchLoader α n σ τ)
    (f : TorchLean.Sample.Batch α n σ τ → Except String β) :
    Except String (BatchLoader α n σ τ × List β) := do
  let (dl', batches) ← epoch (α := α) (σ := σ) (τ := τ) name dl
  pure (dl', ← batches.mapM f)

/--
Run one epoch and require at least one full typed minibatch.

This is the shared checked boundary for examples that need a nonempty list of full batches. It
keeps the "drop partial batches, but fail if nothing remains" policy with the loader API rather than
repeating it in each dataset-specific helper.
-/
def nonemptyEpoch {α : Type} {n : Nat} {σ τ : Spec.Shape}
    (name : String) (dl : BatchLoader α n σ τ) :
    Except String (BatchLoader α n σ τ × List (TorchLean.Sample.Batch α n σ τ)) := do
  let (dl', batches) ← epoch (α := α) (σ := σ) (τ := τ) name dl
  match batches with
  | _ :: _ => pure (dl', batches)
  | [] =>
      throw s!"{name}: no full minibatch available (batch={n}, rows={Data.size dl.raw.dataset})"

/-- Run one epoch and return its first full typed minibatch. -/
def firstFullBatch {α : Type} {n : Nat} {σ τ : Spec.Shape}
    (name : String) (dl : BatchLoader α n σ τ) :
    Except String (TorchLean.Sample.Batch α n σ τ) := do
  let (_dl', batches) ← nonemptyEpoch (α := α) (σ := σ) (τ := τ) name dl
  match batches with
  | b :: _ => pure b
  | [] => throw s!"{name}: internal error: expected a nonempty minibatch epoch"

end BatchLoader

/--
Public loader API: supervised datasets become fixed-size minibatch loaders by default.

The underlying dataset still stores individual samples; the loader batches them and `epoch`
returns tensors with a leading batch axis. Because the batch size is reflected in the type,
the public batched path requires full batches, so `dropLast` defaults to `true`.
-/
def batchLoader {α : Type} {σ τ : Spec.Shape}
    (ds : Dataset (TorchLean.Sample.Supervised α σ τ))
    (batchSize : Nat) (shuffle : Bool := false) (seed : Nat := 0) (dropLast : Bool := true) :
    BatchLoader α batchSize σ τ :=
  { raw := loader ds batchSize (shuffle := shuffle) (seed := seed) (dropLast := dropLast) }

/--
Load a numeric supervised CSV and immediately wrap it as a typed minibatch loader.

The CSV convention is the same as `TabularSupervisedSource`: each row contains `inDim` feature
columns followed by `outDim` target columns.  This belongs in the data API rather than in an
individual model file because tabular examples, benchmarks, and downstream users all need the same
operation: CSV -> typed dataset -> shuffled minibatch loader.
-/
def tabularCsvLoader {α : Type} [_root_.Context α] [_root_.TorchLean.Runtime.FromFloat α]
    (path : System.FilePath) (batchSize inDim outDim : Nat)
    (csvOptions : CsvOptions := {}) (shuffle : Bool := true) (seed : Nat := 0)
    (dropLast : Bool := true) :
    IO (Except String (BatchLoader α batchSize (.dim inDim .scalar)
      (.dim outDim .scalar))) := do
  let src : TabularSupervisedSource :=
    { path := path, inDim := inDim, outDim := outDim, csvOptions := csvOptions }
  let dsE ← src.load (α := α)
  match dsE with
  | .error e => pure (.error e)
  | .ok ds =>
      pure <| .ok <| batchLoader (α := α) ds batchSize (shuffle := shuffle)
        (seed := seed) (dropLast := dropLast)

/-- Build a batch loader when the batch size is only known at runtime. -/
def loaderAny {α : Type} {σ τ : Spec.Shape}
    (ds : Dataset (TorchLean.Sample.Supervised α σ τ))
    (batchSize : Nat) (shuffle : Bool := false) (seed : Nat := 0) (dropLast : Bool := true) :
    AnyBatchLoader α σ τ :=
  ⟨batchSize, batchLoader (α := α) (σ := σ) (τ := τ) ds batchSize shuffle seed dropLast⟩

end Data
end TorchLean
