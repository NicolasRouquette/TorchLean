/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Data.Dataset
public import NN.API.Trainer.Run

/-!
# Datasets

Dataset constructors and file-backed loaders used by `TorchLean.Trainer`.
-/

@[expose] public section

namespace TorchLean

namespace Data

export Training (Dataset DataLoader)

/--
Runtime-polymorphic supervised dataset for `Float` tensors.

Use this for most tutorials and file-loader paths: Float data is cast into whichever runtime scalar
the command selected with `--scalar`.
-/
def tensorDataset
    {n : Nat} {σ τ : Shape}
    (X : Tensor Float (.dim n σ)) (Y : Tensor Float (.dim n τ)) :
    Trainer.DataSource σ τ :=
  { build := fun {α} _ => pure <| TensorDataset.supervisedFloat (α := α) X Y }

/--
Runtime-polymorphic supervised dataset from an explicit sample builder.

Use this when Lean code generates samples directly rather than loading them from batched tensors,
CSV, or NPY files. Sequence windows, synthetic PDE batches, and task-specific examples can keep
their own sample logic while still returning a standard `Trainer.DataSource`.
-/
def samples
    {σ τ : Shape}
    (mk : {α : Type} → [_root_.Context α] → [Runtime.FromFloat α] → List (Sample.Supervised α σ τ)) :
    Trainer.DataSource σ τ :=
  { build := fun {α} _ _ => pure <| _root_.Runtime.Autograd.Train.Dataset.ofList (mk (α := α)) }

/--
Build a singleton dataset from one runtime-polymorphic supervised sample.

Small examples can use the `Trainer.DataSource` API without fixing the runtime scalar or backend
in the dataset definition itself.
-/
def singleton
    {σ τ : Shape}
    (mk : {α : Type} → [_root_.Context α] → [Runtime.FromFloat α] → Sample.Supervised α σ τ) :
    Trainer.DataSource σ τ :=
  samples (fun {α} _ _ => [mk (α := α)])

/--
Build a singleton dataset by feeding one explicit argument into a runtime-polymorphic sample
constructor.

Use this when the sample construction depends on one user-facing payload, such as a prompt string
or one file-backed record.
-/
def singletonFrom
    {ρ : Type} {σ τ : Shape} (arg : ρ)
    (mk : {α : Type} → [_root_.Context α] → [Runtime.FromFloat α] → ρ → Sample.Supervised α σ τ) :
    Trainer.DataSource σ τ :=
  singleton (fun {α} _ _ => mk (α := α) arg)

/--
Build a singleton dataset from one `Float` sample produced inside `IO`.

Use this when the sample comes from a file-backed or runtime-loaded Float boundary. The public
trainer still owns the scalar/backend choice through `Trainer.RunConfig` and `Trainer.TrainOptions`.
-/
def singletonFloatIO
    {σ τ : Shape}
    (mk : IO (Sample.Supervised Float σ τ)) :
    Trainer.DataSource σ τ :=
  { build := fun {_} _ _ => do
      let sample ← mk
      pure <| _root_.Runtime.Autograd.Train.Dataset.ofList
        [ Sample.mk
            (Tensor.map Runtime.ofFloat (Sample.x sample))
            (Tensor.map Runtime.ofFloat (Sample.y sample)) ] }

/--
Runtime-polymorphic dataset from an in-memory list of `Float` supervised samples.

Several examples build their training windows in ordinary `Float` first because the source is text,
CSV, NPY, or another external boundary. This constructor keeps those examples on the public trainer
API: the samples are still cast to the runtime-selected scalar at training time, so callers do not
have to write their own scalar-polymorphic dataset adapter.
-/
def floatSamples
    {σ τ : Shape}
    (samples : List (Sample.Supervised Float σ τ)) :
    Trainer.DataSource σ τ :=
  { build := fun {_} _ _ =>
      pure <| _root_.Runtime.Autograd.Train.Dataset.ofList <| samples.map (fun sample =>
        Sample.mk
          (Tensor.map Runtime.ofFloat (Sample.x sample))
          (Tensor.map Runtime.ofFloat (Sample.y sample))) }

/-- Array form of `floatSamples`. -/
def floatSampleArray
    {σ τ : Shape}
    (samples : Array (Sample.Supervised Float σ τ)) :
    Trainer.DataSource σ τ :=
  floatSamples samples.toList

/--
Convert an unbatched supervised dataset into a fixed-size batched dataset.

Public adapter for examples that want to minibatch the dataset before training and let the model own
the batch axis. The returned dataset stores samples of shape `dim batch σ` and `dim batch τ`, so it
can be passed directly to `Trainer.new` with a batched model.
-/
def batchDataset
    {σ τ : Shape} (batch : Nat) (data : Trainer.DataSource σ τ)
    (shuffle : Bool := true) (seed : Nat := 0) (dropLast : Bool := true) :
    Trainer.DataSource (.dim batch σ) (.dim batch τ) :=
  { build := fun {α} _ => do
      if !dropLast then
        throw <| IO.userError
          "Data.batchDataset: dropLast=false is not supported for typed fixed-size batches"
      let samples ← data.build (α := α)
      let samples :=
        if shuffle then
          shuffled seed samples
        else
          samples
      match batchedSupervised (α := α) batch samples with
      | .ok ds => pure ds
      | .error msg => throw <| IO.userError s!"Data.batchDataset: {msg}" }

/--
Split a public dataset into deterministic train/test views.

Dataset-level analogue of `torch.utils.data.random_split`: the split happens after the
trainer materializes the runtime scalar, but callers stay on ordinary `Trainer.DataSource` values.
-/
def randomSplitDataset
    {σ τ : Shape} (trainSize : Nat) (data : Trainer.DataSource σ τ) (seed : Nat := 0) :
    Trainer.DataSource σ τ × Trainer.DataSource σ τ :=
  let mk (takeTrain : Bool) : Trainer.DataSource σ τ :=
    { build := fun {α} _ _ => do
        let samples ← data.build (α := α)
        if trainSize > samples.size then
          throw <| IO.userError
            s!"Data.randomSplitDataset: requested split {trainSize}, but dataset only has {samples.size} samples"
        let (_seed', parts) := randomSplitAt (seed := seed) trainSize samples
        pure <| if takeTrain then parts.1 else parts.2 }
  (mk true, mk false)

/--
Load a numeric CSV table as a dataset of fixed-size tabular regression batches.

Each CSV row is interpreted as `inDim` feature columns followed by `outDim` target columns.
The returned dataset already has the leading batch dimension expected by a model with input
shape `.dim batch (.dim inDim .scalar)` and output shape `.dim batch (.dim outDim .scalar)`.
-/
def tabularCsvDataset
    (path : System.FilePath) (batch inDim outDim : Nat)
    (csvOptions : CsvOptions := {}) (shuffle : Bool := true) (seed : Nat := 0)
    (dropLast : Bool := true) :
    Trainer.DataSource (.dim batch (.dim inDim .scalar)) (.dim batch (.dim outDim .scalar)) :=
  { build := fun {α} _ => do
      if !dropLast then
        throw <| IO.userError
          "Data.tabularCsvDataset: dropLast=false is not supported for typed fixed-size batches"
      let raw ← readCsvSupervised (α := α) path inDim outDim csvOptions
      let samples ←
        match raw with
        | .ok ds => pure ds
        | .error msg => throw <| IO.userError s!"Data.tabularCsvDataset: {msg}"
      let samples :=
        if shuffle then
          shuffled seed samples
        else
          samples
      match batchedSupervised (α := α) batch samples with
      | .ok ds => pure ds
      | .error msg => throw <| IO.userError s!"Data.tabularCsvDataset: {msg}" }

/--
Runtime-polymorphic supervised regression dataset from a tensor source.

Public file-data analogue of `torch.utils.data.TensorDataset(X, Y)` for examples whose targets are
tensors rather than class labels. The source records where batched features and targets live; the
trainer materializes them at the selected scalar type.
-/
def supervisedDataset (src : SupervisedSource) :
    Trainer.DataSource (Shape.ofList src.xDims) (Shape.ofList src.yDims) :=
  { build := fun {α} _ => do
      let loaded ← src.load (α := α)
      match loaded with
      | .ok ds => pure ds
      | .error msg => throw <| IO.userError s!"Data.supervisedDataset: {msg}" }

/--
Runtime-polymorphic supervised dataset backed by paired `.npy` files.

Common file-backed regression/operator-learning path: `xPath` stores samples with shape `xDims`,
and `yPath` stores matching targets with shape `yDims`.
-/
def supervisedNpyDataset
    (xPath yPath : System.FilePath) (n : Nat)
    (xDims yDims : List Nat) :
    Trainer.DataSource (Shape.ofList xDims) (Shape.ofList yDims) :=
  supervisedDataset (supervisedNpySource xPath yPath n xDims yDims)

/--
Runtime-polymorphic one-hot classification dataset from a tensor source.

Public file-data analogue of `torch.utils.data.TensorDataset`: the source records where features and
integer labels live, and the trainer materializes them at the selected scalar type.
-/
def labeledDataset (src : LabeledSource) :
    Trainer.DataSource (Shape.ofList src.xDims) (.dim src.classes .scalar) :=
  { build := fun {α} _ => do
      let loaded ← src.load (α := α)
      match loaded with
      | .ok ds => pure ds
      | .error msg => throw <| IO.userError s!"Data.labeledDataset: {msg}" }

end Data

end TorchLean
