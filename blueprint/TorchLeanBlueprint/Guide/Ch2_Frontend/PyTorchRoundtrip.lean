import VersoManual

open Verso.Genre Manual

#doc (Manual) "PyTorch Round Trip" =>
%%%
tag := "pytorch-roundtrip"
%%%

Many TorchLean workflows will still involve Python. A model may be trained in PyTorch because the
dataset, optimizer, or engineering environment belongs there. The question for TorchLean is what
happens after that training run. Can its required names and shapes be checked before the values
enter a typed family record?

The round trip is narrow by design. TorchLean does not import arbitrary `nn.Module` objects. It
supports known model families with known layouts. A bridge this small can be audited, tested, and
connected to later graph and verification work.

The checked-in
[round-trip program](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Interop/PyTorch/Roundtrip.lean)
wires three family-specific importers: an MLP, a convolutional network, and a transformer encoder.
The shared JSON core parses named nested arrays, while each family chooses its accepted keys and
expected tensor shapes. Python supplies a named payload, and Lean either reconstructs that typed
family record or rejects it.

# The Contract

The round trip has five steps:

1. Lean defines the expected model family.
2. Lean exports matching PyTorch code.
3. Python trains or modifies the weights.
4. Python writes a named tensor payload.
5. The chosen Lean family importer looks up its required names and parses every nested array at the
   statically expected shape.

The boundary between Lean and PyTorch stays small enough that failures are usually local: a missing
name, a nonnumeric entry, or a mismatched nested-array shape.

If any of those checks fail, the artifact stops at the boundary. It does not become a TorchLean
model by accident.

The shape of the boundary is intentionally closer to a `state_dict` contract than to Python object
serialization. PyTorch users are used to seeing names such as:

```
layer1.weight
layer1.bias
layer2.weight
layer2.bias
```

TorchLean wants those names to become a checked parameter payload, not an implicit module object.
The import code should be able to say exactly which TorchLean parameter each Python tensor is meant
to fill.

# What Lean Checks On Import

The importer is strict about required tensors. It checks them before constructing the typed family
record used by the example.

The checks actually performed are:

- the root is a JSON object, optionally with a `params` object;
- every required family-specific key exists under one of the accepted naming conventions;
- every scalar leaf is a JSON number;
- every nested array has exactly the length required by the expected Lean shape.

JSON object order is irrelevant: the importer looks up names and constructs a fixed typed record.
The current family loaders do not validate the optional `meta.format` or `meta.dtype` strings, and
extra keys are ignored. Those fields are useful provenance, not acceptance evidence. Imported
numbers become host `Float` values; this format does not preserve exact binary32 payload bits.

For example, the checked-in MLP expects:

```
layers.0.weight : shape![3, 2]
```

If its JSON value has two rows of length three instead of three rows of length two, recursive tensor
parsing fails. There is no separate shape annotation to trust:

```
"layers.0.weight": [[1, 2, 3], [4, 5, 6]]
```

The import fails before the value becomes a model parameter. A round trip should either reconstruct
the same typed layout or stop at the boundary with a concrete error.

# What Gets Serialized

The round trip does not serialize an arbitrary Python object graph. Instead it serializes a small
amount of model family metadata plus a named tensor payload. The exact schema depends on the family,
but the shape of the contract is always the same:

- choose a known architecture family,
- agree on required parameter names and tensor layout,
- serialize tensors by name into JSON,
- check that layout again on the Lean side before constructing typed parameters.

Lean then reconstructs the family-specific typed record used by this example's specification
forward pass. Converting that record into another model API is a separate, explicit adapter. The
JSON file is transport, not semantics.

For the checked-in `2 → 3 → 1` MLP, the payload has the following shape:

```
{
  "params": {
    "layers.0.weight": [[...], [...], [...]],
    "layers.0.bias":   [...],
    "layers.2.weight": [[...]],
    "layers.2.bias":   [...]
  },
  "meta": {
    "format": "TorchLean.MLP",
    "dtype": "float32"
  }
}
```

The MLP loader also accepts `fc1.*` and `fc2.*` keys. The CNN and transformer loaders have their
own fixed key sets. The nested-array shape, rather than a separate `shape` field, is what the parser
checks against the Lean type.

This small exchange format handles known TorchLean families. Keep the full object in Python when a
training pipeline needs PyTorch's complete serialization graph, and export only the checked payload
that TorchLean understands.

# Model Families

The repository uses three family examples:

- MLP: the smallest parameter and shape contract.
- CNN: convolution weights and image layouts.
- Transformer encoder: attention projections and many named parameters.

## MLP

The MLP example is the smallest place to read the round trip as a contract rather than as infrastructure.
The architecture is small enough to inspect exported names, compare them against PyTorch's
`state_dict`, and see exactly what Lean checks on re-import.

The same path also shows the common failure modes. If the exported JSON is wrong, Lean rejects it
when a required key is absent or its nested array does not match the expected typed shape.

Example command sequence:

```
lake exe torchlean pytorch_roundtrip --model mlp --action export
python3 NN/Examples/Interop/PyTorch/MLP/train_mlp.py
lake exe torchlean pytorch_roundtrip --model mlp --action import
```

The Python step is
[`train_mlp.py`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Interop/PyTorch/MLP/train_mlp.py);
it writes `NN/Examples/Interop/PyTorch/MLP/mlp.json`.

## CNN

The CNN example exercises nontrivial tensor shapes and weight layouts, while remaining small enough
to debug by hand when the importer reports a mismatch.

Example command sequence:

```
lake exe torchlean pytorch_roundtrip --model cnn --action export
python3 NN/Examples/Interop/PyTorch/CNN/train_cnn.py
lake exe torchlean pytorch_roundtrip --model cnn --action import
```

Here
[`train_cnn.py`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Interop/PyTorch/CNN/train_cnn.py)
writes `NN/Examples/Interop/PyTorch/CNN/cnn.json`.

## Transformer (Encoder)

The transformer example shows why the boundary must stay explicit. Once attention layers and
multiple projections appear, Lean needs a known layout to check, not a vague promise that every
PyTorch module can be imported.

The same pattern scales to larger model families: Lean checks the declared shapes and parameter
layout, Python runs the training loop, and the JSON payload transports only named parameters back
across the boundary.

Example command sequence:

```
lake exe torchlean pytorch_roundtrip --model transformer --action export
python3 NN/Examples/Interop/PyTorch/Transformer/train_transformer.py
lake exe torchlean pytorch_roundtrip --model transformer --action import
```

The companion
[`train_transformer.py`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Interop/PyTorch/Transformer/train_transformer.py)
writes `NN/Examples/Interop/PyTorch/Transformer/transformer_encoder.json`.

# TorchLean, IR, and Generated PyTorch Code

The
[IR export example](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/DeepDives/TorchIRPyTorch.lean)
does the reverse kind of work: it compiles a TorchLean model to the shared IR and emits runnable
PyTorch code for a curated set of architectures:

- `linear`, `mlp`, `sum`, `autoencoder`
- `mha`, `mha-mask`
- `transformer`

Typical usage:

```
lake exe torchlean torch_ir_pytorch --arch mlp > exported_model.py
python3 exported_model.py
```

PyTorch users get a translation example for architectures richer than a single linear layer, and the
example shows where the compiled IR acts as the interchange format.

# A Small Python View

The Python half stays ordinary. TorchLean does not replace the PyTorch workflow; it makes the
boundary between PyTorch and Lean explicit enough to audit.

```
import torch

model = build_exported_model()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(steps):
    pred = model(x_train)
    loss = torch.nn.functional.mse_loss(pred, y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()

save_torchlean_json(model, "mlp.json")
```

The round trip has exactly that shape: Lean defines the expected structure, Python performs the
training, and Lean checks the returned payload against the same structure.

# Tensor Exchange Versus Model Import

It is useful to distinguish three interop layers:

- *tensor exchange* moves arrays between frameworks;
- *parameter import* fills a known TorchLean model family with checked weights;
- *semantic import* claims that a foreign program has the same meaning as a TorchLean graph.

DLPack belongs mostly to the first layer: it is a standard in-memory tensor exchange format used by
array and tensor libraries. It can help avoid extra copies, but it does not by itself tell Lean that
a Python model has the same architecture, parameter names, or proof semantics.

TorchLean's current round trip implements the second layer for three known families: the selected
loader checks required names and tensor shapes. Semantic import is stronger and requires an IR
denotation and a proof relating the foreign program to it.

# Choosing The Interop Boundary

Use the round trip when the training workflow belongs in Python but the returned named tensors
should enter a known TorchLean family with checked shapes. Run the training loop in Lean when its
state transitions must remain explicit. Lower to IR when the model must be inspected as a graph or
connected to a theorem.

# Guarantees And Limits

When the round trip succeeds, the result is a controlled bridge between Lean and Python:

- Lean can emit a model skeleton and companion files.
- Python can train or export weights using a matching layout.
- Lean can parse the required named arrays into family-specific `Tensor Float` fields and run the
  checked-in specification forward example.

The scope is narrow:

- arbitrary PyTorch training needs its own semantic or artifact bridge,
- the importer covers the supported artifact formats rather than the full PyTorch ecosystem,
- optional metadata, optimizer state, unlisted buffers, extra keys, and exact float bit patterns are
  not certified by these family loaders,
- binary32 behavior is handled by the floating point bridge rather than by the round trip format.

Binary32 claims still require the relevant TorchLean float backend and the theorems in the
floating-point chapters.

The round trip is a checked artifact workflow, not a universal conversion tool. Python can remain
the right place to train. Lean becomes the place where the returned object is named, shaped,
inspected, and prepared for graph analysis or verification work.

# References

- George et al., [“TorchLean”](https://arxiv.org/abs/2602.22631), 2026, for the project overview and
  shared IR architecture.
- PyTorch, [serialization notes](https://pytorch.org/docs/stable/notes/serialization.html), for
  `state_dict`-style save and load workflows.
- PyTorch, [`nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html), for
  parameter naming and module conventions.
- [DLPack documentation](https://dmlc.github.io/dlpack/latest/), for tensor exchange as distinct
  from model-family import.
