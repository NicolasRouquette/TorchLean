# `TorchLean.Data`

`TorchLean.Data` contains constructors and file sources for `Trainer.Dataset`.

Application code uses one supervised-data type:

```lean
Trainer.Dataset inputShape targetShape
```

It keeps the sample shapes in the type while delaying scalar conversion until the trainer chooses
its runtime. Constructors such as `Data.tensorDataset`, `Data.floatSamples`, and
`Data.supervisedDataset` all return this type.

Manual loops that already own a runtime scalar can import `NN.Data.SampleStream`. A sample stream is
finite and lazily indexed: wrapping an array does not copy it, and slicing a batched tensor happens
when a sample is requested. It is deliberately not a second application-level dataset API.

The data boundary is focused:

- `.npy` for numeric tensors;
- numeric CSV for small tabular data;
- UTF-8 text for language-model examples.

For other formats, use the converter:

```bash
python3 scripts/datasets/torchlean_data_convert.py --help
```

The design is intentionally conservative. TorchLean does not need to own every dataset format in
the ML ecosystem. Python handles ecosystem formats and writes a simple boundary file. TorchLean then
checks shape, scalar type, batch structure, and labels before the data reaches a trainer, graph
export, or verifier.

## Which Source Should I Use?

| Data shape | Use |
| --- | --- |
| one tensor file | `Data.TensorSource` |
| supervised `X.npy`, `Y.npy` | `Data.SupervisedSource` |
| image/classification labels | `Data.LabeledSource` |
| small numeric CSV | `Data.TabularSupervisedSource` |
| fixed-size tensor batches | `Data.batchDataset` |
| generated or file-backed batches | typed step streams through the trainer API |
| text windows | `TorchLean.text` helpers, then bounded-token samples |

The main Lean entry points are:

- `Data.TensorSource`: one tensor file plus expected dimensions.
- `Data.SupervisedSource`: two batched tensors, `X : (N, xDims...)` and `Y : (N, yDims...)`.
- `Data.LabeledSource`: batched inputs plus label vector, one-hot encoded when loaded.
- `Data.TabularSupervisedSource`: one CSV where each row contains `x..., y...`.
- `Data.batchDataset`: fixed-size tensor minibatches represented as another `Trainer.Dataset`.

Use `Tensor.map` and `Sample.mapX` or `Sample.mapY` for preprocessing. Manual stream code may use
`SampleStream.map`; it transforms samples only when they are requested.

## Boundary Discipline

Every loader should make these facts visible:

- the number of samples,
- the input shape after removing the batch axis,
- the target or label shape,
- the scalar interpretation,
- whether labels are one-hot encoded or integer ids,
- whether shuffling is deterministic and which seed controls it.

A classifier margin certificate, PINN dataset containment check, or FNO prediction artifact depends
on the data shape and ordering supplied to its checker.

The loader is therefore part of the trust boundary. It does not certify that an external dataset is
scientifically correct, but it does make the imported tensor shape, label interpretation, and sample
order explicit before the data reaches a trainer or checker.

## Common Paths

| Workflow | Typical data path |
| --- | --- |
| MLP/KAN tabular examples | numeric CSV or `X.npy`/`Y.npy` with shape `(N, features)` and `(N, targets)` |
| CNN/ViT examples | image tensors `(N, C, H, W)` plus label vector `(N,)` |
| GPT/Mamba examples | UTF-8 text, tokenizer assets, bounded-token windows, shifted targets |
| FNO Burgers | `.mat` or simulator output converted to `X.npy`/`Y.npy`, plus metadata for grid and split |
| PINN checks | JSON/certificate artifacts plus coordinate/value samples checked by `NN.Verification.PINN` |
| VNN-COMP-style checks | network/property artifacts loaded by the verification layer, not by an ordinary trainer |

The same boundary file may feed multiple layers. For example, a supervised `.npy` pair can train a
model, export predictions, and later provide a verification fixture. The loader's job is to make
the shape and provenance obvious before those later claims are made.

## What To Document

When adding a new data path, document:

- the external source or producer;
- the command that creates the boundary file;
- the expected tensor shapes after conversion;
- label encoding and class count, if applicable;
- whether shuffling is deterministic;
- where generated logs, predictions, or manifests are written.

If the source is a public dataset or third-party artifact, update `THIRD_PARTY_NOTICES.md`. If the
data feeds a checker, also update the relevant `NN/Verification` or `NN/Examples/Verification`
README so the checked predicate names the data boundary it relies on.

Examples and conversion recipes live in:

```text
NN/Examples/Data/README.md
```
