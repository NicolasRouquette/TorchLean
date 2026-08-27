# Data Examples

TorchLean keeps the data boundary focused. Python handles ecosystem formats and
downloads; Lean receives `.npy`, numeric CSV, or text, then checks shapes and builds typed datasets.

```text
external dataset -> Python converter/downloader -> .npy or numeric CSV -> TorchLean.Data -> typed loader
```

Use Python for JPEG folders, `.pt`, `.npz`, `.mat`, and dataset downloads. Use Lean for typed
tensors, typed batching, training, verification, and reproducible evaluation.

Data enters TorchLean through a small boundary file. Loaders record where the samples came from,
their expected shape, and whether labels are integer ids or one-hot targets. Those details remain
available when a trained model, prediction artifact, or certificate is inspected later.

## Local Tutorial Artifacts

Generate the small ignored artifacts used by the loader tutorials:

```bash
python3 NN/Examples/Data/generate_small_data.py
```

This writes:

| File | Shape / role |
| --- | --- |
| `small_regression.csv` | numeric rows `x1,x2,y` |
| `small_regression_X.npy` | `(25, 2)` `float32` features |
| `small_regression_y.npy` | `(25, 1)` `float32` targets |
| `small_cifar10like_X.npy` | `(200, 3, 32, 32)` `float32` image-shaped features |
| `small_cifar10like_y.npy` | `(200,)` integer labels stored as `float32` |
| `small_fno1d_X.npy` | `(4, 32)` deterministic periodic inputs |
| `small_fno1d_y.npy` | `(4, 32)` deterministic operator-learning targets |

## Dataset Shapes

| Task | Input file | Target file | Loader |
| --- | --- | --- | --- |
| regression / operator learning | `X.npy : (N, ...)` | `Y.npy : (N, ...)` | `Data.SupervisedSource` |
| classification | `X.npy : (N, ...)` | `y.npy : (N,)` labels | `Data.LabeledSource` |
| tabular regression | numeric CSV with `x...,y...` | same CSV | `Data.TabularSupervisedSource` |
| text modeling | UTF-8 text | tokenizer/windowing builds targets | `TorchLean.text` |
| images | `X.npy : (N, C, H, W)` | `y.npy : (N,)` labels | `Data.LabeledSource` |

Classification labels are numeric values such as `0.0`, `1.0`, ..., and are converted to one-hot
targets on the Lean side.

## Boundary Files And Manifests

For custom data, prefer converter commands that write a manifest. The manifest is a provenance note:
it tells later readers which file was converted, which key was read, and what shape was exported.

Useful converter patterns:

```bash
python3 scripts/datasets/torchlean_data_convert.py tensor \
  --input features.pt --key x --output data/real/mytask/X.npy --manifest

python3 scripts/datasets/torchlean_data_convert.py pair \
  --x-input dataset.npz --x-key images --x-output data/real/mytask/X.npy \
  --y-input dataset.npz --y-key labels --y-output data/real/mytask/y.npy \
  --manifest

python3 scripts/datasets/torchlean_data_convert.py image-folder \
  --input imagenette2/train --height 160 --width 160 --labels-from-dirs \
  --x-output data/real/imagenette/train_X.npy \
  --y-output data/real/imagenette/train_y.npy \
  --manifest
```

The Lean loader then checks the exported shape before training code sees the tensor. If the data is
later used by a verifier, the verifier should cite the same shape/provenance boundary instead of
assuming the dataset layout from a filename.

## Downloaded Example Data

```bash
python3 scripts/datasets/download_example_data.py --auto-mpg --cifar10 --tiny-shakespeare
python3 scripts/datasets/download_example_data.py --household-power --household-power-windows 512
python3 scripts/datasets/download_example_data.py --tinystories-valid
```

Common outputs:

| Dataset | Output path |
| --- | --- |
| CIFAR-10 | `data/real/cifar10/cifar10_{train,test}_{X,y}.npy` |
| UCI Auto MPG | `data/real/auto_mpg/auto_mpg.csv` |
| UCI household power | `data/real/household_power/household_power_{X,Y}.npy` |
| Tiny Shakespeare | `data/real/text/tiny_shakespeare.txt` |
| TinyStories validation | `data/real/text/tinystories_valid.txt` |

The CIFAR-10 export defaults to a small real subset for interactive examples. Pass
`--cifar10-limit-train -1 --cifar10-limit-test -1` to export the full split.

## Run The Loader Examples

```bash
lake exe torchlean data_csv
lake exe torchlean data_npy
lake exe torchlean data_cifar10 --check-only --epochs 1 --batch 4 --train-size 8 --n-total 20
```

Model examples use the same API:

```bash
lake -R -K cuda=true exe torchlean mlp --device cuda --steps 10
lake -R -K cuda=true exe torchlean cnn --device cuda --n-total 200 --steps 2
lake -R -K cuda=true exe torchlean lstm_regression --device cuda --steps 200 --windows 96
lake -R -K cuda=true exe torchlean gpt2 --device cuda --tiny-shakespeare --steps 1 --windows 1 --generate 0
```

## Training Data Versus Verification Data

Most model-zoo commands use data for training or evaluation. Verification commands use data more
strictly: an input box, coordinate sample, residual dataset, or prediction artifact becomes part of
the checked claim.

Keep those roles separate:

- training data may be large, shuffled, and summarized by logs;
- verification fixtures should be small, reproducible, and explicit about tolerances;
- generated prediction artifacts should record the command and shape metadata needed to reload
  them;
- external datasets and producers should be listed in `THIRD_PARTY_NOTICES.md` when they become
  part of a documented workflow.

## Lean API

Ordinary loader code should use the public API:

```lean
import NN.API
open TorchLean
```

Classification source:

```lean
def src : Data.LabeledSource :=
  Data.LabeledSource.ofPaths
    .npy
    "data/real/imagenette/train_X.npy"
    "data/real/imagenette/train_y.npy"
    1000
    [3, 160, 160]
    10
```

Supervised regression source:

```lean
def src : Data.SupervisedSource :=
  Data.SupervisedSource.ofPaths
    .npy
    "data/real/fno/burgers_X.npy"
    "data/real/fno/burgers_Y.npy"
    1000
    [64]
    [64]
```

Wrap loaded datasets with `Data.supervisedEpochs` for typed minibatches:

```lean
let loader := Data.supervisedEpochs ds 8 (shuffle := true) (seed := 0) (dropLast := true)
let (_loader', batches) ← CLI.orThrow "trainer" <| Data.SupervisedEpochs.epoch "trainer" loader
```

The batch axis is reflected in the type. For example, a batch of channel-first images and its
class labels can have the following shapes:

```text
X : Tensor α [batch, channels, height, width]
Y : Tensor α [batch, classes]
```

These are ordinary tensors; channel-first is an axis convention rather than a separate image type.

## Provenance

CIFAR-10, UCI Auto MPG, UCI household power, Tiny Shakespeare, and TinyStories keep their upstream
licenses and citations. The generated tutorial artifacts are local examples and are ignored by git.
