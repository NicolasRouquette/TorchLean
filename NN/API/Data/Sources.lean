/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Data.TensorDataset

/-!
# Dataset Sources

Typed NPY and CSV sources for supervised and labeled datasets.
-/

@[expose] public section

namespace TorchLean
namespace Data

/-- Load an N-D tensor from a `.npy` file, checking the on-disk shape matches `dims`. -/
def readNpyTensor (path : System.FilePath) (dims : List Nat) :
    IO (Except String (Spec.Tensor Float (NN.Tensor.shapeOfDims dims))) := do
  let res ← readNpy path
  match res with
  | .error e => pure (.error e)
  | .ok data =>
      if data.shape != dims then
        pure (.error s!"npy: shape mismatch, expected {dims}, got {data.shape}")
      else
        pure <| NN.Tensor.ofList (α := Float) dims data.values.toList

/--
Load an N-D tensor from a `.npy` file, allowing the file to contain more rows on the leading axis.

This is the dataset-loader analogue of taking `tensor[:n]` in PyTorch. The rank and trailing
dimensions must still match exactly; only the leading dimension may be larger than requested.

We use this for dataset sources rather than the stricter `readNpyTensor` because an exported
dataset usually has a fixed full size, while local runs often request a bounded prefix. For example,
a CIFAR file may have shape `(50000, 3, 32, 32)` while an example command asks for `n = 80`; the
resulting TorchLean tensor has type-level shape
`(80, 3, 32, 32)`.

This is still a checked loader, not an implicit reshape:

- rank must agree;
- all trailing dimensions must agree;
- the file must contain at least the requested number of rows;
- only C-order NPY files can be prefix-loaded efficiently by the low-level parser.
-/
def readNpyTensorPrefix (path : System.FilePath) (dims : List Nat) :
    IO (Except String (Spec.Tensor Float (NN.Tensor.shapeOfDims dims))) := do
  let res ← readNpyLeadingAxisPrefix path dims
  match res with
  | .error e => pure (.error e)
  | .ok data =>
      match hDims : dims with
      | expectedN :: expectedTail =>
          match data.shape with
          | actualN :: actualTail =>
              if actualTail != expectedTail then
                pure (.error
                  s!"npy: shape mismatch, expected trailing dims {expectedTail}, got {actualTail}")
              else if actualN < expectedN then
                pure (.error s!"npy: expected at least {expectedN} rows, got {actualN}")
              else
                let dims' := expectedN :: expectedTail
                let count := dims'.foldl (fun acc n => acc * n) 1
                pure <| (NN.Tensor.ofList (α := Float) dims'
                  (data.values.toList.take count)).map (fun t => by
                    simpa [hDims] using t)
          | [] =>
              pure (.error s!"npy: shape mismatch, expected {dims}, got {data.shape}")
      | [] =>
          match data.shape with
          | [] =>
              pure <| (NN.Tensor.ofList (α := Float) [] data.values.toList).map (fun t => by
                simpa [hDims] using t)
          | _ =>
              pure (.error s!"npy: shape mismatch, expected {dims}, got {data.shape}")

/-- Parse a float-encoded class label as a `Nat` in `[0, classes)`. -/
def natLabelOfFloat (tag : String) (classes : Nat) (x : Float) : Except String Nat := do
  let n : Nat := x.toUInt64.toNat
  if (n : Float) != x then
    throw s!"{tag}: expected an integer class label, got {x}"
  else if n >= classes then
    throw s!"{tag}: class label {n} out of range (classes={classes})"
  else
    pure n

/--
Labeled dataset from a batched tensor `X : (n, σ)` and a label vector `y : (n,)`.

Labels are stored as floats (common when exporting from NumPy); we validate each label is an
integer in `[0, classes)`, then one-hot encode it.
-/
def labeledFromLeadingAxis {α : Type} [_root_.Context α] [_root_.TorchLean.Runtime.FromFloat α]
    (tag : String) (classes : Nat)
    {n : Nat} {σ : Spec.Shape}
    (X : Spec.Tensor Float (.dim n σ))
    (y : Spec.Tensor Float (.dim n .scalar)) :
    Except String (Dataset (_root_.TorchLean.TensorPack α [σ, .dim classes .scalar])) := do
  let samples : List (Spec.Tensor Float σ × Nat) ←
    (List.finRange n).mapM (fun i => do
      let x := Spec.getAtSpec X i
      let labelF : Float := Spec.Tensor.toScalar (Spec.getAtSpec y i)
      let label ← natLabelOfFloat tag classes labelF
      pure (x, label))
  pure <| labeled (α := α) (σ := σ) classes samples

/--
Load a supervised dataset from a CSV with `inDim + outDim` columns per row:

`x1, ..., x_inDim, y1, ..., y_outDim`.
-/
def readCsvSupervised {α : Type} [_root_.Context α] [_root_.TorchLean.Runtime.FromFloat α]
    (path : System.FilePath) (inDim outDim : Nat) (opts : CsvOptions := {}) :
    IO (Except String (Dataset (_root_.TorchLean.TensorPack α [.dim inDim .scalar,
      .dim outDim .scalar]))) := do
  let rowsRes ← readCsvFloatRows path opts
  match rowsRes with
  | .error e => pure (.error e)
  | .ok rows =>
      let samplesRes :
          Except String (List (Spec.Tensor Float (.dim inDim .scalar) × Spec.Tensor Float
            (.dim outDim .scalar))) :=
        rows.mapM (fun row => do
          let xs := row.take inDim
          let ys := row.drop inDim
          let xF ← vectorOfList (tag := "csv") (n := inDim) xs
          let yF ← vectorOfList (tag := "csv") (n := outDim) ys
          pure (xF, yF))
      pure <| samplesRes.map (fun samplesF => supervised (α := α) samplesF)

/-!
## File Sources

The definitions below describe tensors by path, format, and expected dimensions:

1. describe each tensor as a `TensorSource`;
2. load it as a typed TorchLean tensor;
3. build supervised/labeled datasets by slicing the leading batch axis, just like PyTorch `TensorDataset`.

Policy for external ecosystems:
- NumPy `.npy` is the canonical interchange format for numeric tensors.
- CSV is supported for small tabular data.
- MATLAB `.mat`, PyTorch checkpoints, HDF5, Parquet, and image archives should be converted by a
  small preparation script into `.npy` tensors plus metadata. The Lean runtime loader intentionally
  handles a small deterministic interchange format rather than every external binary format.
-/

/-- File formats supported directly by the Lean side unified data-source loader. -/
inductive TensorFormat where
  /-- NumPy `.npy`, supporting the subset decoded by `readNpyTensor`. -/
  | npy
  /-- Numeric CSV table. CSV sources are interpreted as 2D tensors `[rows, cols]`. -/
  | csv
deriving BEq, Repr

namespace TensorFormat

/-- Human-facing extension used by messages and examples. -/
def extension : TensorFormat → String
  | .npy => ".npy"
  | .csv => ".csv"

end TensorFormat

/--
Description of one tensor stored on disk.

`dims` is the expected tensor shape.  NPY can load any rank supported by `ofList`; CSV is treated
as a numeric table and therefore expects `dims = [rows, cols]`.
-/
structure TensorSource where
  /-- Path to the file. -/
  path : System.FilePath
  /-- Expected dimensions. -/
  dims : List Nat
  /-- Direct Lean side format. External formats should be preconverted to `.npy`. -/
  format : TensorFormat := .npy
  /-- CSV parsing options, used only when `format = .csv`. -/
  csvOptions : CsvOptions := {}

namespace TensorSource

/--
Load a numeric CSV table as a tensor.

Supported shapes:
- `[rows, cols]`: ordinary numeric table,
- `[n]`: either one column with `n` rows or one row with `n` columns.
-/
def loadCsvTensorND (path : System.FilePath) (dims : List Nat) (opts : CsvOptions := {}) :
    IO (Except String (Spec.Tensor Float (NN.Tensor.shapeOfDims dims))) := do
  match hDims : dims with
  | [rowsExpected, colsExpected] =>
      let rowsRes ← readCsvFloatRows path opts
      match rowsRes with
      | .error e => pure (.error e)
      | .ok rows =>
          if rows.length != rowsExpected then
            pure (.error s!"csv: expected {rowsExpected} rows, got {rows.length}")
          else
            let bad? := rows.zipIdx.find? (fun (row, _i) => row.length != colsExpected)
            match bad? with
            | some (row, i) =>
                pure (.error s!"csv: row {i + 1}: expected {colsExpected} columns, got {row.length}")
            | none =>
                let flat := rows.foldr (fun row acc => row ++ acc) []
                pure <| (NN.Tensor.ofList (α := Float) [rowsExpected, colsExpected] flat).map
                  (fun t => by
                    simpa [hDims] using t)
  | [n] =>
      let rowsRes ← readCsvFloatRows path opts
      match rowsRes with
      | .error e => pure (.error e)
      | .ok rows =>
          let flat? : Except String (List Float) :=
            if rows.length = n && rows.all (fun row => row.length == 1) then
              .ok (rows.map (fun row => row.getD 0 0.0))
            else
              match rows with
              | [row] =>
                  if row.length = n then .ok row
                  else .error s!"csv: expected one row with {n} columns, got {row.length}"
              | _ =>
                  .error s!"csv: expected a length-{n} vector as one column or one row"
          match flat? with
          | .error e => pure (.error e)
          | .ok flat =>
              pure <| (NN.Tensor.ofList (α := Float) [n] flat).map
                (fun t => by
                  simpa [hDims] using t)
  | _ =>
      pure (.error s!"csv: TensorSource expects dims=[rows, cols] or dims=[n], got {dims}")

/-- Load a Float tensor from a path/format/dimension tuple. -/
def loadFloatAs (format : TensorFormat) (path : System.FilePath)
    (dims : List Nat) (opts : CsvOptions := {}) :
    IO (Except String (Spec.Tensor Float (NN.Tensor.shapeOfDims dims))) := do
  match format with
  | .npy => readNpyTensor path dims
  | .csv => loadCsvTensorND path dims opts

/--
Load a Float tensor, allowing NPY files to contain more rows than requested on the leading axis.

`TensorSource.loadFloatAs` is exact: the file shape must equal `dims`.  This prefix variant is for
dataset-style sources where `dims` starts with the number of rows requested by the current run.  CSV
sources remain exact because CSV has no binary prefix contract; NPY sources use
`readNpyTensorPrefix`.
-/
def loadFloatLeadingPrefixAs (format : TensorFormat) (path : System.FilePath)
    (dims : List Nat) (opts : CsvOptions := {}) :
    IO (Except String (Spec.Tensor Float (NN.Tensor.shapeOfDims dims))) := do
  match format with
  | .npy => readNpyTensorPrefix path dims
  | .csv => loadCsvTensorND path dims opts

/-- Load a `TensorSource` as a Float tensor with the statically reflected `shapeOfDims src.dims`. -/
def loadFloat (src : TensorSource) :
    IO (Except String (Spec.Tensor Float (NN.Tensor.shapeOfDims src.dims))) := do
  loadFloatAs src.format src.path src.dims src.csvOptions

end TensorSource

/--
Two tensor sources representing supervised data:
- `x` must have shape `(n, xDims...)`,
- `y` must have shape `(n, yDims...)`.
-/
structure SupervisedSource where
  /-- Number of samples along the leading batch axis. -/
  n : Nat
  /-- Per-sample input dimensions. -/
  xDims : List Nat
  /-- Per-sample target dimensions. -/
  yDims : List Nat
  /-- Source for the batched input tensor. -/
  x : TensorSource
  /-- Source for the batched target tensor. -/
  y : TensorSource

namespace SupervisedSource

/-- Construct a supervised source from paths using the same file format for `x` and `y`. -/
def ofPaths (format : TensorFormat) (xPath yPath : System.FilePath)
    (n : Nat) (xDims yDims : List Nat) (csvOptions : CsvOptions := {}) : SupervisedSource :=
  { n, xDims, yDims
    x := { path := xPath, dims := n :: xDims, format, csvOptions }
    y := { path := yPath, dims := n :: yDims, format, csvOptions } }

/--
Load a supervised dataset by slicing the leading batch axis from the two tensors.

This is the preferred public loader for regression/operator-learning examples, regardless of
whether the backing files are `.npy` or small numeric CSV tables.
-/
def load {α : Type} [_root_.TorchLean.Runtime.FromFloat α] (src : SupervisedSource) :
    IO (Except String (Dataset (_root_.TorchLean.TensorPack α [NN.Tensor.shapeOfDims src.xDims,
      NN.Tensor.shapeOfDims src.yDims]))) := do
  -- Dataset sources interpret `src.n` as "number of rows to use in this run."  For NPY files, the
  -- physical file is allowed to contain more rows; for CSV files, the requested shape remains exact.
  let xRes ← TensorSource.loadFloatLeadingPrefixAs src.x.format src.x.path (src.n :: src.xDims)
    src.x.csvOptions
  let yRes ← TensorSource.loadFloatLeadingPrefixAs src.y.format src.y.path (src.n :: src.yDims)
    src.y.csvOptions
  match xRes with
  | .error e => pure (.error e)
  | .ok X =>
      match yRes with
      | .error e => pure (.error e)
      | .ok Y => pure (.ok (supervisedFromLeadingAxisFloat (α := α) X Y))

end SupervisedSource

/-- Paired `.npy` source for supervised regression or operator-learning datasets. -/
def supervisedNpySource
    (xPath yPath : System.FilePath) (n : Nat)
    (xDims yDims : List Nat) : SupervisedSource :=
  SupervisedSource.ofPaths .npy xPath yPath n xDims yDims

/--
Load paired `.npy` files as concrete `Float` supervised samples.

This is useful for reporting, custom evaluation loops, and native kernels that need concrete
`Float` tensors outside the high-level trainer API.
-/
def loadSupervisedNpyFloatSamples
    (xPath yPath : System.FilePath) (n : Nat)
    (xDims yDims : List Nat) :
    IO (Except String (Array (TorchLean.Sample.Supervised Float (NN.Tensor.shapeOfDims xDims)
      (NN.Tensor.shapeOfDims yDims)))) := do
  let ds ← SupervisedSource.load (α := Float) (supervisedNpySource xPath yPath n xDims yDims)
  pure <| ds.map (fun d => toList d |>.toArray)

/--
Two tensor sources representing labeled classification data:
- `x` must have shape `(n, xDims...)`,
- `y` must have shape `(n,)` and contain integer-valued labels.
-/
structure LabeledSource where
  /-- Number of samples along the leading batch axis. -/
  n : Nat
  /-- Per-sample input dimensions. -/
  xDims : List Nat
  /-- Number of classes for one-hot targets. -/
  classes : Nat
  /-- Source for the batched input tensor. -/
  x : TensorSource
  /-- Source for the label vector. -/
  y : TensorSource

namespace LabeledSource

/-- Construct a labeled source from paths using the same file format for `x` and `y`. -/
def ofPaths (format : TensorFormat) (xPath yPath : System.FilePath)
    (n : Nat) (xDims : List Nat) (classes : Nat) (csvOptions : CsvOptions := {}) : LabeledSource :=
  { n, xDims, classes
    x := { path := xPath, dims := n :: xDims, format, csvOptions }
    y := { path := yPath, dims := [n], format, csvOptions } }

/--
Load a labeled classification dataset by slicing the leading batch axis and one-hot encoding labels.

For CSV label vectors, store labels as a single-column table with `dims = [n, 1]` and use a custom
`TensorSource` if needed; the path constructor above is aimed at `.npy` label vectors.
-/
def load {α : Type} [_root_.Context α] [_root_.TorchLean.Runtime.FromFloat α] (src : LabeledSource) :
    IO (Except String (Dataset (_root_.TorchLean.TensorPack α [NN.Tensor.shapeOfDims src.xDims,
      .dim src.classes .scalar]))) := do
  -- Labels use the same prefix-row convention as supervised tensors. This lets one full exported
  -- label vector back different bounded runs without making separate copies on disk.
  let xRes ← TensorSource.loadFloatLeadingPrefixAs src.x.format src.x.path (src.n :: src.xDims)
    src.x.csvOptions
  let yRes ← TensorSource.loadFloatLeadingPrefixAs src.y.format src.y.path [src.n] src.y.csvOptions
  match xRes with
  | .error e => pure (.error e)
  | .ok X =>
      match yRes with
      | .error e => pure (.error e)
      | .ok y =>
          pure <| labeledFromLeadingAxis (α := α) (σ := NN.Tensor.shapeOfDims src.xDims)
            "data-source" src.classes X y

end LabeledSource

/--
Single-table supervised CSV source.

Use this when one CSV row contains both input and target columns:
`x1, ..., x_inDim, y1, ..., y_outDim`.
-/
structure TabularSupervisedSource where
  /-- CSV file path. -/
  path : System.FilePath
  /-- Number of input feature columns. -/
  inDim : Nat
  /-- Number of target columns. -/
  outDim : Nat
  /-- CSV parsing options. -/
  csvOptions : CsvOptions := {}

namespace TabularSupervisedSource

/-- Load a single-table supervised CSV source. -/
def load {α : Type} [_root_.Context α] [_root_.TorchLean.Runtime.FromFloat α]
    (src : TabularSupervisedSource) :
    IO (Except String (Dataset (_root_.TorchLean.TensorPack α [.dim src.inDim .scalar,
      .dim src.outDim .scalar]))) :=
  readCsvSupervised (α := α) src.path src.inDim src.outDim src.csvOptions

end TabularSupervisedSource

end Data
end TorchLean
