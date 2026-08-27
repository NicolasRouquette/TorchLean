/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Data.Loaders
public import NN.API.Trainer.Dataset

/-!
# Datasets

Dataset constructors and file-backed loaders used by `TorchLean.Trainer`.
-/

@[expose] public section

namespace TorchLean

namespace Data

/--
Runtime-polymorphic supervised dataset for `Float` tensors.

This is the typed counterpart of PyTorch's `TensorDataset(inputs, targets)`. The common leading
dimension counts samples; the remaining input and target dimensions are inferred and retained in
the result type.

Use this for most tutorials and file-loader paths: Float data is cast into whichever runtime scalar
the command selected with `--scalar`.
-/
def tensorDataset
    {n : Nat} {inputShape targetShape : List Nat}
    (inputs : Tensor Float (n :: inputShape))
    (targets : Tensor Float (n :: targetShape)) :
    Trainer.Dataset inputShape targetShape :=
  { build := fun {α} _ =>
      pure <| TensorDataset.ofBatchedFloat
        (α := α) (shapes := [inputShape, targetShape])
        (_root_.TorchLean.TensorPack! inputs, targets) }

/--
Runtime-polymorphic supervised dataset from an explicit sample builder.

Use this when Lean code generates samples directly rather than loading them from batched tensors,
CSV, or NPY files. Sequence windows, synthetic PDE batches, and task-specific examples can keep
their own sample logic while still returning a standard `Trainer.Dataset`.
-/
def samples
    {inputShape targetShape : List Nat}
    (mk : {α : Type} → [_root_.Context α] → [Runtime.FromFloat α] →
      Array (Sample.Supervised α inputShape targetShape)) :
    Trainer.Dataset inputShape targetShape :=
  { build := fun {α} _ _ => pure <| _root_.TorchLean.Data.SampleStream.ofArray (mk (α := α)) }

/--
Build a singleton dataset from one runtime-polymorphic supervised sample.

Small examples can use the `Trainer.Dataset` API without fixing the runtime scalar or backend
in the dataset definition itself.
-/
def singleton
    {inputShape targetShape : List Nat}
    (mk : {α : Type} → [_root_.Context α] → [Runtime.FromFloat α] →
      Sample.Supervised α inputShape targetShape) :
    Trainer.Dataset inputShape targetShape :=
  samples (fun {α} _ _ => #[mk (α := α)])

/--
Build a singleton dataset from one `Float` sample produced inside `IO`.

Use this when the sample comes from a file-backed or runtime-loaded Float boundary. The public
trainer still owns the scalar/backend choice through `Trainer.RunConfig` and `Trainer.TrainOptions`.
-/
def singletonFloatIO
    {inputShape targetShape : List Nat}
    (mk : IO (Sample.Supervised Float inputShape targetShape)) :
    Trainer.Dataset inputShape targetShape :=
  { build := fun {_} _ _ => do
      let sample ← mk
      pure <| _root_.TorchLean.Data.SampleStream.ofArray
        #[ Sample.mk
            (Tensor.map Runtime.ofFloat (Sample.x sample))
            (Tensor.map Runtime.ofFloat (Sample.y sample)) ] }

/--
Runtime-polymorphic dataset from an in-memory array of `Float` supervised samples.

Several examples build their training windows in ordinary `Float` first because the source is text,
CSV, NPY, or another external boundary. This constructor keeps those examples on the public trainer
API: the samples are still cast to the runtime-selected scalar at training time, so callers do not
have to write their own scalar-polymorphic dataset adapter.
-/
def floatSamples
    {inputShape targetShape : List Nat}
    (samples : Array (Sample.Supervised Float inputShape targetShape)) :
    Trainer.Dataset inputShape targetShape :=
  { build := fun {_} _ _ =>
      pure <| (_root_.TorchLean.Data.SampleStream.ofArray samples).map (fun sample =>
        Sample.mk
          (Tensor.map Runtime.ofFloat (Sample.x sample))
          (Tensor.map Runtime.ofFloat (Sample.y sample))) }

/--
Convert an unbatched supervised dataset into a fixed-size batched dataset.

Public adapter for examples that want to minibatch the dataset before training and let the model own
the batch axis. The returned dataset prepends `batch` to each sample shape, so it can be passed
directly to `Trainer.new` with a batched model.
-/
def batchDataset
    {inputShape targetShape : List Nat} (batch : Nat)
    (data : Trainer.Dataset inputShape targetShape)
    (shuffle : Bool := true) (seed : Nat := 0) (dropLast : Bool := true) :
    Trainer.Dataset (batch :: inputShape) (batch :: targetShape) :=
  { build := fun {α} _ => do
      if !dropLast then
        throw <| IO.userError
          "Data.batchDataset: dropLast=false is not supported for typed fixed-size batches"
      let samples ← data.build (α := α)
      let samples :=
        if shuffle then
          SampleStream.shuffled seed samples
        else
          samples
      match collateStream (α := α) batch samples with
      | .ok ds => pure ds
      | .error msg => throw <| IO.userError s!"Data.batchDataset: {msg}" }

/--
Split a public dataset into deterministic train/test views.

Dataset-level analogue of `torch.utils.data.random_split`: the split happens after the
trainer materializes the runtime scalar, but callers stay on ordinary `Trainer.Dataset` values.
-/
def randomSplitDataset
    {inputShape targetShape : List Nat} (trainSize : Nat)
    (data : Trainer.Dataset inputShape targetShape) (seed : Nat := 0) :
    Trainer.Dataset inputShape targetShape × Trainer.Dataset inputShape targetShape :=
  let mk (takeTrain : Bool) : Trainer.Dataset inputShape targetShape :=
    { build := fun {α} _ _ => do
        let samples ← data.build (α := α)
        if trainSize > samples.size then
          throw <| IO.userError
            s!"Data.randomSplitDataset: requested split {trainSize}, but dataset only has {samples.size} samples"
        let (_seed', parts) := SampleStream.randomSplitAt seed trainSize samples
        pure <| if takeTrain then parts.1 else parts.2 }
  (mk true, mk false)

/--
Load a numeric CSV table as a dataset of fixed-size tabular regression batches.

Each CSV row is interpreted as `inDim` feature columns followed by `outDim` target columns.
The returned dataset already has the leading batch dimension expected by a model with input
shape `[batch, inDim]` and output shape `[batch, outDim]`.
-/
def tabularCsvDataset
    (path : System.FilePath) (batch inDim outDim : Nat)
    (csvOptions : CsvOptions := {}) (shuffle : Bool := true) (seed : Nat := 0)
    (dropLast : Bool := true) :
    Trainer.Dataset [batch, inDim] [batch, outDim] :=
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
          SampleStream.shuffled seed samples
        else
          samples
      match collateStream (α := α) batch samples with
      | .ok ds => pure ds
      | .error msg => throw <| IO.userError s!"Data.tabularCsvDataset: {msg}" }

/--
Runtime-polymorphic supervised regression dataset from a tensor source.

Public file-data analogue of `torch.utils.data.TensorDataset(X, Y)` for examples whose targets are
tensors rather than class labels. The source records where batched features and targets live; the
trainer materializes them at the selected scalar type.
-/
def supervisedDataset (src : SupervisedSource) :
    Trainer.Dataset src.xDims src.yDims :=
  { build := fun {α} _ => do
      let loaded ← src.load (α := α)
      match loaded with
      | .ok ds => pure ds
      | .error msg => throw <| IO.userError s!"Data.supervisedDataset: {msg}" }

/--
Runtime-polymorphic one-hot classification dataset from a tensor source.

Public file-data analogue of `torch.utils.data.TensorDataset`: the source records where features and
integer labels live, and the trainer materializes them at the selected scalar type.
-/
def labeledDataset (src : LabeledSource) :
    Trainer.Dataset src.xDims [src.classes] :=
  { build := fun {α} _ => do
      let loaded ← src.load (α := α)
      match loaded with
      | .ok ds => pure ds
      | .error msg => throw <| IO.userError s!"Data.labeledDataset: {msg}" }

end Data

end TorchLean
