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

# The Dataset Type

The trainer-facing type is:

```
Trainer.DataSource inputShape targetShape
```

It describes one training item. The scalar type is intentionally absent. Its `build` field
materializes a concrete dataset after the trainer chooses native `Float32`, `IEEE32Exec`, or
another supported executable scalar. Materialization chooses one arithmetic semantics for every
numeric input and target in that run; it is not a per-column dtype schema.

This lets the same host-`Float` data source feed several scalar runtimes while keeping the
conversion visible at materialization. It also means a proof-level real tensor is not accidentally
passed to an IO training loop.

The relevant field is small enough to read directly:

```
structure Trainer.DataSource (σ τ : Shape) where
  build :
    {α : Type} →
    [Context α] →
    [Runtime.FromFloat α] →
    IO (Training.Dataset (Sample.Supervised α σ τ))
```

The `IO` is doing real work. A dataset may open a file, parse text, allocate arrays, or discover a
bad row. The shape indices describe each accepted sample; they do not claim that file access is
pure or that every source record is valid.

# When The Checks Happen

I find data bugs much easier to diagnose when I write down the boundary at which each fact becomes
known:

:::table +header
*
  * Stage
  * What is known
  * Typical failure
*
  * source description
  * paths, format, requested dimensions, tokenizer or preprocessing choice
  * unsupported format or incomplete configuration
*
  * parsing
  * bytes or text can be decoded into values
  * missing file, malformed number, corrupt header
*
  * shape validation
  * element count and sample dimensions agree
  * short row, mismatched leading counts, invalid token id
*
  * scalar materialization
  * every numeric value can be represented by the selected runtime scalar
  * unsupported dtype or rejected conversion
*
  * batching
  * each emitted item has the model's input and target shape
  * final partial batch under a fixed-size policy
*
  * training
  * the runner receives typed samples
  * numerical failure, unsupported operation, or backend rejection
:::

Moving a check earlier improves the error message, but it does not change its logical strength. A
CSV width check proves that accepted rows have the requested number of fields. It says nothing
about whether the second field is the physical quantity the experiment intended to measure.

# Begin With Four Samples

The XOR table is small enough to see in full:

```
import NN.API
open TorchLean

def xs : Tensor Float (shape![4, 2]) :=
  tensor! [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
  ]

def ys : Tensor Float (shape![4, 1]) :=
  tensor! [[0.0], [1.0], [1.0], [0.0]]

def xorData : Trainer.DataSource (shape![2]) (shape![1]) :=
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
The running MLP constructs a deterministic input grid with `Data.Synthetic.squareGrid`, maps its
target over the leading axis with `Tensor.mapLeading`, and passes both tensors to
`Data.tensorDataset`.

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

“Dataset size = 5” counts materialized minibatches, not source rows. Each item already has input
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

def data : Trainer.DataSource (shape![2]) (shape![1]) :=
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

## GPT-2 Byte-Level BPE Files

`NN.API.Text.Bpe` loads the conventional GPT-2 tokenizer pair directly in Lean:

- `vocab.json` is a flat JSON object from byte-escaped token spellings to natural-number ids;
- `merges.txt` gives one adjacent symbol pair per non-comment line. File order determines rank, and
  a lower rank is applied first.

The two files form one artifact: using a vocabulary with a merge table from a different tokenizer
can change token boundaries or leave a merged piece absent from the vocabulary. Pass both paths and
record both files, preferably with content hashes, alongside a run. A malformed vocabulary or a
missing file produces an error. The current merge parser skips malformed non-comment lines, so a
successful load alone is not a complete integrity check for an untrusted `merges.txt`.

Encoding has four visible stages. It applies the GPT-2 pre-tokenizer, converts each fragment's UTF-8
bytes through GPT-2's reversible byte-to-Unicode table, repeatedly applies the best-ranked adjacent
merge, and finally looks up every piece in `vocab.json`. `GPT2BPE.encode` returns an error when a
piece has no id. `GPT2BPE.decode?` performs the inverse vocabulary lookup and byte decoding, and
reports an unknown id or invalid UTF-8 rather than inventing text.

Here is a direct round-trip from the repository root after placing a matching tokenizer pair at the
shown paths:

```
import NN.API.Text.Bpe
open TorchLean.text

def inspectBpe : IO Unit := do
  let tok ← GPT2BPE.load
    "data/real/gpt2/vocab.json"
    "data/real/gpt2/merges.txt"
  match GPT2BPE.encode tok "naïve café" with
  | .error e => throw <| IO.userError e
  | .ok ids =>
      IO.println s!"ids = {repr ids}"
      match GPT2BPE.decode? tok ids with
      | .error e => throw <| IO.userError e
      | .ok decoded => IO.println s!"decoded = {decoded}"
```

`decodeOrEmpty` and the generic-tokenizer adapter are intended for display and convenience: they collapse
decode or encode failures to empty output. Use the error-reporting `encode` and `decode?` functions
at an artifact-validation boundary.

## Unicode Is Part Of The Tokenizer Version

GPT-2 pre-tokenization distinguishes Unicode letters, numbers, whitespace, contractions, and other
symbols. `NN.API.Text.Unicode` supplies checked-in Unicode 15.1 category ranges for `L*` and `N*`
plus the Unicode `White_Space` set. Tokenization therefore does not silently inherit an operating
system regex engine, but the checked-in tables are still a semantic dependency: category changes
between Unicode versions can change segmentation. Record the TorchLean revision as well as the BPE
files when exact reproducibility matters.

## Run The Maintained BPE Trainer

First prepare the example corpus, then run the CUDA trainer with both tokenizer paths:

```
python3 scripts/datasets/download_example_data.py --tiny-shakespeare

lake -R -K cuda=true exe torchlean text_gpt2 --device cuda \
  --data-file data/real/text/tiny_shakespeare.txt \
  --bpe-vocab data/real/gpt2/vocab.json \
  --bpe-merges data/real/gpt2/merges.txt \
  --allow-small-data --max-chars 20000 --steps 1 \
  --prompt "First Citizen:" --generate 0
```

The data script prepares the corpus; it does not supply the tokenizer pair. The command prints BPE
loading progress, a first shifted token window, and before/after loss. Both BPE flags are required
together. Omitting both selects the runner's byte-token path instead.

This runner is CUDA-only. It does not load OpenAI or Hugging Face model weights. Its BPE mode trains
a randomly initialized TorchLean Transformer with batch size two, a one-token context, and a local
projection of at most 512 observed GPT-2 ids; ids outside that retained set map to the local fallback
id. It exercises real file parsing, tokenization, shifted-window construction, training, and decode
plumbing, but it is neither GPT-2-small nor evidence of checkpoint-level tokenizer equivalence.

That completes the text-specific detour. The rest of the chapter returns to a question shared by
numeric, image, and text datasets: what a batch means and when it is materialized.

# Two Meanings Of “Batch Size”

TorchLean uses “batch size” for two related operations:

## Tensor minibatches

`Data.batchDataset` changes the item shapes:

```
def batched :
    Trainer.DataSource (shape![2, 2]) (shape![2, 1]) :=
  Data.batchDataset 2 xorData
    (shuffle := true)
    (seed := 42)
```

The model must accept `[2,2]` and return `[2,1]`. One forward/backward operation processes the whole
tensor minibatch.

Using batch size five for this four-row XOR dataset would produce no typed batches because
`Data.batchDataset` defaults to `dropLast := true`. Choosing two here makes both batches full and
keeps the example executable.

## Groups of unbatched samples

`Trainer.TrainOptions.batchSize` counts dataset items per optimizer update. On a model `[2] → [1]`,
ordinary dataset items are individual samples. Every selected item is differentiated at the same
parameter point, the resulting gradient packs are averaged, and the optimizer is applied once. The
reported loss is the corresponding mean of the item losses. This path does not insert a leading
tensor axis.

For a `Data.batchDataset`, each item already contains a fixed-size tensor minibatch. Keep
`TrainOptions.batchSize := 1` to perform one vectorized device pass per update. A larger value
accumulates gradients across several tensor minibatches. On CUDA, generic accumulation currently
synchronizes the parameter pack for the host optimizer update, so one typed batch item per update
is the fast path for large batches.

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

`Trainer.DataSource` delays scalar choice. Lower-level manual code may already own a
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
`Data.batchDataset`, `Data.tabularCsvDataset`, or another dataset constructor.

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
- [`Bpe.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Text/Bpe.lean);
- [`Unicode.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/Text/Unicode.lean);
- [`TextGpt2.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Sequence/TextGpt2.lean);
- [`Npy.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Npy.lean);
- [`Cifar10Images.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Data/Loaders/Cifar10Images.lean).
