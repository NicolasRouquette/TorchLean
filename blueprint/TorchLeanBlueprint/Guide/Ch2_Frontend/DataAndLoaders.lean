import VersoManual

open Verso.Genre Manual

#doc (Manual) "From Files To Typed Minibatches" =>
%%%
tag := "datasets-loaders"
%%%

A model type tells Lean the shape of one input and one output. A dataset must eventually provide
values of exactly those shapes, but real data begins in a less orderly form: rows in a CSV file,
arrays in an NPY file, text tokens, simulator output, or tensors generated on demand.

TorchLean treats loading as a boundary. Parsing and dimension checks happen before a value becomes
a typed training sample. Once the boundary succeeds, the training loop does not need to ask on
every step whether a row had the right number of columns.

# The Public Dataset Type

The public type is:

```
Trainer.Dataset inputShape targetShape
```

It describes one training item. The scalar type is intentionally absent. Its `build` field
materializes a concrete dataset after the trainer chooses `Float`, `IEEE32Exec`, or another
supported executable scalar.

This lets the same Float-authored data feed several scalar runtimes while keeping the conversion
visible at materialization. It also means a proof-level real tensor is not accidentally passed to
an IO training loop.

# Begin With Four Samples

The XOR table is small enough to see in full:

```
import NN.API
open TorchLean

def xs : Tensor.T Float (shape![4, 2]) :=
  tensor! [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
  ]

def ys : Tensor.T Float (shape![4, 1]) :=
  tensor! [[0.0], [1.0], [1.0], [0.0]]

def xorData : Trainer.Dataset (shape![2]) (shape![1]) :=
  Data.tensorDataset xs ys
```

The leading dimension of `xs` and `ys` is the sample count. `Data.tensorDataset` checks that both
counts agree and removes that leading axis from the item shapes:

```
whole input tensor    [4, 2]
one input sample         [2]

whole target tensor   [4, 1]
one target sample        [1]
```

Try changing the target annotation to `shape![3,1]` while leaving four rows in the literal. The
literal itself fails to elaborate. If the mismatch instead arrives from runtime files, the loader
returns an error before constructing the dataset.

`Data.samples`, `Data.singleton`, and `Data.floatSamples` serve list-backed or generated data.
`Data.regressionGrid` builds the deterministic grid used by the running MLP.

# A Real CSV Run

TorchLean includes a 25-row regression file with columns `x1,x2,y`. Run:

```
lake exe torchlean data_csv \
  --device cpu --batch 5 --steps 5 --seed 2026
```

The loader prints the model and boundary choices before training:

```
model:
Sequential: [5, 2] -> [5, 1], layers=3, params=33
  [0] Linear(2, 8): [5, 2] -> [5, 8]
  [1] ReLU: [5, 8] -> [5, 8]
  [2] Linear(8, 1): [5, 8] -> [5, 1]
data_dir = NN/Examples/Data
csv_path = NN/Examples/Data/small_regression.csv
seed = 2026
train = Adam(lr=0.05), steps=5,
        batch_size=5, shuffle=true, drop_last=true
dataset size = 5
mean_loss(before) = 1.367492
mean_loss(after) = 0.323823
```

“Dataset size = 5” now counts materialized minibatches, not source rows. Each item already has input
shape `[5,2]` and target shape `[5,1]`.

The constructor in
[`Csv.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Csv.lean)
is:

```
let csvOptions : Data.CsvOptions :=
  { skipHeader := true }

let data :=
  Data.tabularCsvDataset csvPath batch 2 1
    (csvOptions := csvOptions)
    (shuffle := true)
    (seed := seed)
```

The arguments `2` and `1` state how many columns belong to the input and target. Malformed numbers,
wrong column counts, missing files, or too few rows become explicit `IO` errors when `build` runs.

# Break The File Boundary Deliberately

Run the same command with a nonexistent path:

```
lake exe torchlean data_csv --csv /tmp/no-such-data.csv
```

The program stops at `Data.requireFile`; no randomly initialized model is reported as having
trained on an empty dataset.

For a second experiment, copy the small CSV and remove one value from a row. The CSV parser may
still recognize the row as text, but the supervised loader rejects its column count. These two
failures are different:

- file existence is an operating-system boundary;
- row width is a data-schema boundary.

Neither is a theorem about the data-generating process. They establish that the accepted artifact
has the structure requested by the model.

# NPY And Other Numeric Sources

NPY preserves numeric dtype and array dimensions, making it a cleaner boundary for already prepared
numeric tensors:

```
def source : Data.SupervisedSource :=
  Data.SupervisedSource.ofPaths
    .npy
    "data/x.npy"
    "data/y.npy"
    100
    [2]
    [1]

def data : Trainer.Dataset (shape![2]) (shape![1]) :=
  Data.supervisedDataset source
```

The source declares:

- the file format;
- input and target paths;
- sample count;
- one-sample input dimensions;
- one-sample target dimensions.

`Data.supervisedNpyDataset` is the convenience constructor. `Data.LabeledSource` reads integer
labels and constructs one-hot targets for classification.

TorchLean does not maintain a second parser for every ecosystem format. `.pt`, `.pth`, `.npz`,
image folders, and specialized scientific containers can be converted with Python into a small
numeric boundary such as NPY or CSV. The converter is an untrusted producer; the Lean loader checks
the resulting artifact.

# Text Is Not A Numeric Matrix

Language models need a vocabulary, tokenization rule, context window, and target shift. Those are
semantic choices, not generic CSV parsing.

A next-token dataset typically turns tokens:

$$`t_0,t_1,\ldots,t_n`

into windows:

$$`
x_i=(t_i,\ldots,t_{i+L-1}),\qquad
y_i=(t_{i+1},\ldots,t_{i+L}).
`

The shapes may both be `[batch,L]`, but the one-position shift is the learning problem. TorchLean's
text helpers make integer tokens and window construction explicit. A tokenizer file or vocabulary
mapping should be stored with the run because changing it changes the meaning of every integer in
the dataset.

# Two Meanings Of “Batch Size”

The distinction here is important.

## Tensor minibatches

`Data.batchDataset` changes the item shapes:

```
def batched :
    Trainer.Dataset (shape![5, 2]) (shape![5, 1]) :=
  Data.batchDataset 5 xorData
    (shuffle := true)
    (seed := 42)
```

The model must accept `[5,2]` and return `[5,1]`. One forward/backward operation processes the whole
tensor minibatch.

## Groups of unbatched samples

`Trainer.TrainOptions.batchSize` on a model `[2] → [1]` controls how many samples are consumed in an
outer step. In the current in-memory loop, an optimizer update is still applied for each sample in
that group. The option affects scheduling and logging; it does not insert a leading tensor axis.

The two paths can have different optimization behavior. Calling both of them “batch size” without
examining the model shape would hide that difference.

# Why The Final Partial Batch Is Dropped

A model accepting `[5,2]` cannot receive `[3,2]`; these are different Lean types. Therefore the
fixed-size typed batching path uses `dropLast := true`. In a 23-sample dataset with batch size five,
four full minibatches are produced and three samples remain.

Dropping is one policy, not a law of machine learning. Alternatives include:

- padding to five and carrying a validity mask;
- bucketing examples so each group has a fixed length;
- using a dynamic wrapper at a lower runtime layer;
- choosing a batch size that divides the dataset.

Each alternative changes the data contract. TorchLean refuses to silently change the shape of the
last item.

# Materialized Loaders And Epoch State

The public `Trainer.Dataset` delays scalar choice. Lower-level manual code may already own a
materialized:

```
Data.Dataset α inputShape targetShape
```

`Data.batchLoader` constructs a loader whose batch size appears in its type.
`Data.BatchLoader.epoch name loader` returns both:

- the full typed batches for this epoch;
- updated deterministic shuffle state for the next epoch.

This functional state transition makes data order reproducible. There is no hidden global RNG whose
position depends on unrelated code.

Use the lower loader API for a custom epoch loop. Ordinary model training should prefer
`Data.batchDataset`, `Data.tabularCsvDataset`, or another public constructor.

# Generated Streams

Some workloads are not finite passes over a stored dataset. PINNs resample collocation points,
reinforcement-learning agents collect new transitions, and language models may generate windows
from a large file on demand.

`Trainer.Manual.StepBatchStream α shapes` represents a source indexed by the training step. The
shape list remains fixed in the type, while values can be generated or loaded lazily.

This gives the loop explicit control over:

- the step number;
- generator state;
- file position;
- simulator state;
- checkpoint restoration.

A generated stream should log enough state to reproduce a batch. “Seed 42” is insufficient if the
generator also depends on an evolving environment or file cursor.

# Reproducibility Needs More Than One Seed

At minimum, record:

:::table +header
*
  * Choice
  * Example
*
  * model initialization
  * trainer seed
*
  * sample order
  * loader/shuffle seed
*
  * source identity
  * file path and preferably content hash
*
  * preprocessing
  * tokenizer, normalization, column split
*
  * batch policy
  * size, shuffling, dropping, padding
*
  * runtime
  * scalar semantics, backend, device
:::

The CSV example uses `2026` for both model initialization and shuffling for convenience. They are
conceptually separate choices and may be configured independently in a larger experiment.

Training can write a JSON `TrainLog`:

```
let trained ← trainer.train data
  { steps := 200
    log := .json outPath
    title := "small regression" }
```

The log is an execution artifact. It can support debugging and reproducibility, but it is not a
certificate of convergence or generalization.

# Continue With Training

The complete minibatch example is runnable as:

```
lake exe torchlean quickstart_minibatch_mlp \
  --device cpu --batch 5 --steps 5 --seed 2026
```

It exercises CSV parsing, deterministic shuffling, fixed-size typed batching, model execution,
autograd, Adam, and prediction. The next chapter opens the training loop and explains what state
changes at each optimizer step.

Sources:

- [`NN/API/Data/README.md`](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Data/README.md);
- [`NN/Examples/Data/README.md`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/README.md);
- [`Npy.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Npy.lean);
- [`Cifar10Images.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Cifar10Images.lean).
