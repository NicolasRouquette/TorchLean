/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

-- shake: keep-all

public import NN.API.Scalar
public import NN.API.Macros
public import NN.API.TensorPack
public import NN.Runtime.Autograd.Train.IoLoader

import Mathlib.Algebra.Order.Algebra
import Mathlib.Data.List.Basic

/-!
# Datasets, Loaders, and File Sources

TorchLean datasets keep each sample's tensor shapes in its type. A file-backed workflow usually
looks like this:

1. Convert outside-world datasets to canonical `.npy` tensors or small numeric CSV files.
2. Describe those files with `TensorSource`, `SupervisedSource`, or `LabeledSource`.
3. Load them into shape-typed TorchLean tensors and datasets.
4. Train with `batchLoader`, `BatchLoader.epoch`, `trainer.train`, or a manual trainer loop.

We keep the implementation small and predictable:
- datasets are in-memory and pure (often backed by `List`)
- loader shuffling is seed-driven and reproducible
- `.npy` is the canonical numeric interchange format
- CSV is supported for small tabular data
- MATLAB `.mat`, PyTorch `.pt/.pth`, NumPy `.npz`, and image folders should be converted to
  `.npy` with `scripts/datasets/torchlean_data_convert.py`
- there are no multiprocessing workers, memory maps, or pinned-memory support

## PyTorch Mapping

This is inspired by `torch.utils.data`:
- Dataset, DataLoader: `https://pytorch.org/docs/stable/data.html`
- TensorDataset: `https://pytorch.org/docs/stable/data.html#torch.utils.data.TensorDataset`
- DataLoader: `https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader`

TorchLean’s key difference is that samples typically carry *type-level shapes* (via `TensorPack`),
so many helpers here are shape-aware by construction.

## Main Types

- `TensorSource`: one file plus expected dimensions.
- `SupervisedSource`: two batched tensors, `X : (N, xDims...)` and `Y : (N, yDims...)`.
- `LabeledSource`: batched inputs plus integer class labels, one-hot encoded on load.
- `TabularSupervisedSource`: one CSV table with input columns followed by target columns.
- `batchLoader`: deterministic, typed minibatching.

For examples and conversion commands, see `NN/Examples/Data/README.md`.
-/

@[expose] public section


namespace TorchLean
namespace Data

export _root_.Runtime.Autograd.Train
  (Dataset CsvOptions
   readCsvFloatRows readCsvDatasetPairs readCsvVectorDataset
   readNpy readNpyLeadingAxisPrefix readNpyVector readNpyMatrix
   vectorOfList vectorOfArray matrixOfLists matrixOfArrays
   datasetOfPairs datasetOfListVectors)
/--
Typed analogue of PyTorch's `TensorDataset`.

In TorchLean, a sample is usually a `TensorPack α shapes`, i.e. a shape-tracked tuple of tensors.
-/
abbrev TensorDataset (α : Type) (shapes : List Spec.Shape) :=
  _root_.Runtime.Autograd.Train.Dataset (_root_.TorchLean.TensorPack α shapes)

/-- Build a dataset from an explicit list of samples. -/
def fromList {a : Type} (xs : List a) : _root_.Runtime.Autograd.Train.Dataset a :=
  _root_.Runtime.Autograd.Train.Dataset.ofList xs

/-- Require that all paths exist, otherwise raise a user-facing error with a shared hint. -/
def requireFiles (exeName : String) (paths : List System.FilePath) (hint : String := "") :
    IO Unit := do
  for p in paths do
    unless (← p.pathExists) do
      let suffix := if hint.isEmpty then "" else "\n" ++ hint
      throw <| IO.userError s!"{exeName}: missing data file: {p}{suffix}"

/-- Require one named data file to exist. -/
def requireFile
    (exeName : String) (label : String) (path : System.FilePath) (hint : String := "") :
    IO Unit := do
  unless (← path.pathExists) do
    let suffix := if hint.isEmpty then "" else "\n" ++ hint
    throw <| IO.userError s!"{exeName}: missing {label}: {path}{suffix}"

/-- Require paired supervised input/target files to exist. -/
def requirePairedFiles
    (exeName : String)
    (xLabel : String) (xPath : System.FilePath)
    (yLabel : String) (yPath : System.FilePath)
    (hint : String := "") : IO Unit := do
  requireFile exeName xLabel xPath hint
  requireFile exeName yLabel yPath hint

/-- Write a small CSV file, creating the parent directory if needed. -/
def writeCsv (path : System.FilePath) (header : List String) (rows : List (List String)) :
    IO Unit := do
  if let some parent := path.parent then
    IO.FS.createDirAll parent
  let lines := String.intercalate "\n" ((String.intercalate "," header) :: rows.map
    (String.intercalate ",")) ++ "\n"
  IO.FS.writeFile path lines

/--
Write a one-dimensional prediction probe CSV.

Rows are `i,x,input,target,prediction`, where $x=i/(n-1)$ for $n>1$.
This writes the compact prediction table used by plotting examples such as 1D operator learning.
-/
def writeVectorPredictionCsv {n : Nat}
    (path : System.FilePath)
    (input target prediction : Spec.Tensor Float (.dim n .scalar)) : IO Unit := do
  let rows := (List.finRange n).map (fun i =>
    let denom := Float.ofNat (Nat.max 1 (n - 1))
    let xpos := Float.ofNat i.val / denom
    [toString i.val, toString xpos, toString (Spec.Tensor.vecGet input i),
      toString (Spec.Tensor.vecGet target i), toString (Spec.Tensor.vecGet prediction i)])
  writeCsv path ["i", "x", "input", "target", "prediction"] rows

/-- Materialize a dataset as a list. -/
def toList {a : Type} (ds : _root_.Runtime.Autograd.Train.Dataset a) : List a :=
  _root_.Runtime.Autograd.Train.Dataset.toList ds

/-- Converting a list to a dataset and back yields the original list. -/
@[simp] theorem toList_fromList {a : Type} (xs : List a) : toList (fromList xs) = xs := by
  simp [toList, fromList,
    _root_.Runtime.Autograd.Train.Dataset.toList,
    _root_.Runtime.Autograd.Train.Dataset.ofList]

/-- Number of elements in the dataset. -/
def size {a : Type} (ds : _root_.Runtime.Autograd.Train.Dataset a) : Nat :=
  _root_.Runtime.Autograd.Train.Dataset.size ds

/-- The size of a dataset built from a list is the list length. -/
@[simp] theorem size_fromList {a : Type} (xs : List a) : size (fromList xs) = xs.length := by
  simp [size, fromList, _root_.Runtime.Autograd.Train.Dataset.size,
    _root_.Runtime.Autograd.Train.Dataset.ofList]

/-- Whether the dataset is empty. -/
def isEmpty {a : Type} (ds : _root_.Runtime.Autograd.Train.Dataset a) : Bool :=
  _root_.Runtime.Autograd.Train.Dataset.isEmpty ds

/--
Build a cycling index function for a *nonempty* list.

`cycleList xs h i` returns `xs[i % xs.length]`.

This is useful for in-memory datasets where a fixed-step “PyTorch-like” loop should avoid repeated
`Option` handling.
-/
def cycleList {a : Type} (xs : List a) (h : xs ≠ []) : Nat → a :=
  fun i =>
    have hlen : 0 < xs.length := by
      simpa using List.length_pos_of_ne_nil h
    xs[i % xs.length]'(Nat.mod_lt _ hlen)

/--
Like `cycleList`, but fail with a message if the list is empty.

Fixed-step dataset code can check emptiness once and then index without `Option`.
-/
def cycleListOrError {a : Type} (xs : List a) (err : String := "empty list") : Except String (Nat →
  a) :=
  match xs with
  | [] => .error err
  | x :: xs => .ok (cycleList (x :: xs) (by simp))

/--
Build a cycling index function for a *nonempty* dataset.

`cycleDataset ds h i` returns `ds[i % ds.size]`.

This is the dataset analogue of `cycleList`. It avoids per-step `Option` handling in fixed-step
training loops.
-/
def cycleDataset {a : Type} (ds : _root_.Runtime.Autograd.Train.Dataset a) (h : ds.data.size ≠ 0) :
  Nat → a :=
  fun i =>
    let n := ds.data.size
    have hn : 0 < n :=
      Nat.pos_of_ne_zero (by simpa [n] using h)
    ds.data[i % n]'(Nat.mod_lt _ hn)

/--
Like `cycleDataset`, but fail with a message if the dataset is empty.

This is the preferred helper for “PyTorch-style” fixed-step loops over in-memory datasets.
-/
def cycleDatasetOrError {a : Type} (ds : _root_.Runtime.Autograd.Train.Dataset a)
    (err : String := "empty dataset") : Except String (Nat → a) :=
  match h : ds.data.size with
  | 0 => .error err
  | n + 1 =>
      .ok (cycleDataset ds (by
        -- In this branch, `ds.data.size = n+1`, so it is nonzero.
        simp [h]))

/-- Safe indexing into a dataset. -/
def get? {a : Type} (ds : _root_.Runtime.Autograd.Train.Dataset a) (i : Nat) : Option a :=
  _root_.Runtime.Autograd.Train.Dataset.get? ds i

/-- Return the first array element, or a caller-provided error when the array is empty. -/
def firstArrayOrError {a : Type} (xs : Array a) (err : String := "empty array") : Except String a :=
  match xs[0]? with
  | some x => .ok x
  | none => .error err

/-- Map a dataset elementwise (pure, deterministic). -/
def map {a b : Type} (f : a → b) (ds : _root_.Runtime.Autograd.Train.Dataset a) :
    _root_.Runtime.Autograd.Train.Dataset b :=
  _root_.Runtime.Autograd.Train.Dataset.map f ds

/-- Append two datasets, preserving order: all samples from `x` followed by all samples from `y`. -/
def append {a : Type} (x y : _root_.Runtime.Autograd.Train.Dataset a) :
    _root_.Runtime.Autograd.Train.Dataset a :=
  _root_.Runtime.Autograd.Train.Dataset.append x y

/-- Split a dataset at position `n` (prefix, suffix). -/
def splitAt {a : Type} (n : Nat) (ds : _root_.Runtime.Autograd.Train.Dataset a) :
    _root_.Runtime.Autograd.Train.Dataset a × _root_.Runtime.Autograd.Train.Dataset a :=
  _root_.Runtime.Autograd.Train.Dataset.splitAt n ds

/--
Shuffle a dataset deterministically, returning the updated RNG seed and the shuffled dataset.

This is used to implement `DataLoader.shuffle` behavior in a purely functional way.
-/
def shuffle {a : Type} (seed : Nat) (ds : _root_.Runtime.Autograd.Train.Dataset a) :
    Nat × _root_.Runtime.Autograd.Train.Dataset a :=
  _root_.Runtime.Autograd.Train.Dataset.shuffle seed ds

/-- Deterministically shuffle a dataset when the caller does not need the updated seed. -/
def shuffled {a : Type} (seed : Nat) (ds : _root_.Runtime.Autograd.Train.Dataset a) :
    _root_.Runtime.Autograd.Train.Dataset a :=
  (shuffle seed ds).snd

/--
Shuffle once and then split at `n`.

This is a small building block for train/val splits.
-/
def randomSplitAt {a : Type} (seed : Nat) (n : Nat) (ds : _root_.Runtime.Autograd.Train.Dataset a) :
    Nat × (_root_.Runtime.Autograd.Train.Dataset a × _root_.Runtime.Autograd.Train.Dataset a) :=
  let (seed', ds') := shuffle seed ds
  (seed', splitAt n ds')

/--
Split a dataset into equal-sized minibatches (as lists), dropping the final partial batch.

This is a low-level helper; ordinary loader code should use `DataLoader.epoch` or
`Data.batchedSupervised`.
-/
def batches {a : Type} (tag : String) (batchSize : Nat) (ds : _root_.Runtime.Autograd.Train.Dataset
  a) :
    Except String (List (List a)) :=
  _root_.Runtime.Autograd.Train.Dataset.batches tag batchSize ds

/-- Like `batches`, but return each minibatch as an `Array` instead of a `List`. -/
def batchesArray {a : Type} (tag : String) (batchSize : Nat) (ds :
  _root_.Runtime.Autograd.Train.Dataset a) :
    Except String (List (Array a)) :=
  _root_.Runtime.Autograd.Train.Dataset.batchesArray tag batchSize ds

/--
Untyped analogue of PyTorch's `torch.utils.data.DataLoader`.

This is the deterministic, purely-functional loader provided by the TorchLean runtime.
-/
abbrev DataLoader (a : Type) :=
  _root_.Runtime.Autograd.Train.DataLoader a

/--
Construct a `DataLoader` from a dataset.

If `shuffle := true`, shuffling is deterministic w.r.t. `seed`.
If `dropLast := true`, incomplete final batches are discarded.
-/
def loader {a : Type} (ds : _root_.Runtime.Autograd.Train.Dataset a)
    (batchSize : Nat) (shuffle : Bool := false) (seed : Nat := 0) (dropLast : Bool := false) :
    DataLoader a :=
  { dataset := ds, batchSize := batchSize, shuffle := shuffle, seed := seed, dropLast := dropLast }

/--
Run one epoch worth of minibatching and return:
- an updated loader (with the new seed), and
- the list of minibatches.
-/
def epoch {a : Type} (name : String) (dl : DataLoader a) :
    Except String (DataLoader a × List (List a)) :=
  _root_.Runtime.Autograd.Train.DataLoader.epoch name dl

/--
Like `epoch`, but apply a user-provided `collate` function to each minibatch, matching the role of
PyTorch's `collate_fn=` option.
-/
def epochCollate {a b : Type} (name : String) (dl : DataLoader a)
    (collate : List a → Except String b) :
    Except String (DataLoader a × List b) :=
  _root_.Runtime.Autograd.Train.DataLoader.epochCollate name dl collate

/--
Shape-typed wrapper around `DataLoader` for supervised samples.

The batch size `n` is reflected in the type, and `BatchLoader.epoch` returns fully-collated
`dim n` minibatches (so `dropLast=true` is required).
-/
structure BatchLoader (α : Type) (n : Nat) (σ τ : Spec.Shape) where
  /-- Generic loader carrying the supervised samples. -/
  loader : DataLoader (TorchLean.Sample.Supervised α σ τ)

/-- Existential wrapper for loaders when the batch size is chosen at runtime. -/
abbrev AnyBatchLoader (α : Type) (σ τ : Spec.Shape) :=
  Σ n : Nat, BatchLoader α n σ τ

/--
Read the row count from an `.npy` file and check its trailing shape.

For a batched tensor with shape `(N, d₁, ..., dₖ)`, this returns `N` when the trailing dimensions
match `tailShape`.
-/
def availableNpyRows
    (path : System.FilePath) (tailShape : List Nat) (expectedDesc : String) :
    IO (Except String Nat) := do
  let npyMeta ← readNpy path
  match npyMeta with
  | .error e => pure (.error e)
  | .ok n =>
      match n.shape with
      | rows :: rest =>
          if rest = tailShape then
            pure (.ok rows)
          else
            pure (.error s!"expected {expectedDesc}, got {n.shape}")
      | _ => pure (.error s!"expected {expectedDesc}, got {n.shape}")

/--
Convert a list of `(x, y)` float tensors into a dataset of TorchLean supervised samples.

This casts float data into the selected scalar backend `α` and packs it into a
`TensorPack α [σ, τ]`.
-/
def supervised {α : Type} [_root_.Context α] [_root_.TorchLean.Runtime.FromFloat α] {σ τ : Spec.Shape}
    (xs : List (Spec.Tensor Float σ × Spec.Tensor Float τ)) :
    _root_.Runtime.Autograd.Train.Dataset (_root_.TorchLean.TensorPack α [σ, τ]) :=
  fromList <| xs.map (fun (xF, yF) =>
    let x : Spec.Tensor α σ := Spec.mapTensor (_root_.TorchLean.Runtime.ofFloat (α := α)) xF
    let y : Spec.Tensor α τ := Spec.mapTensor (_root_.TorchLean.Runtime.ofFloat (α := α)) yF
    tensorpack! x, y)

/--
Convert a list of `(x, label)` pairs into a dataset of one-hot classification samples.

Each label has type `Fin classes`, so an out-of-range class cannot enter the dataset.
-/
def labeled {α : Type} [_root_.Context α] [_root_.TorchLean.Runtime.FromFloat α] {σ : Spec.Shape}
    (classes : Nat) (xs : List (Spec.Tensor Float σ × Fin classes)) :
    _root_.Runtime.Autograd.Train.Dataset
      (_root_.TorchLean.TensorPack α [σ, .dim classes .scalar]) :=
  fromList <| xs.map (fun (xF, label) =>
    let x : Spec.Tensor α σ := Spec.mapTensor (_root_.TorchLean.Runtime.ofFloat (α := α)) xF
    let yF : Spec.Tensor Float (.dim classes .scalar) := NN.Tensor.oneHot (α := Float) classes label
    let y : Spec.Tensor α (.dim classes .scalar) :=
      Spec.mapTensor (_root_.TorchLean.Runtime.ofFloat (α := α)) yF
    tensorpack! x, y)

end Data
end TorchLean
