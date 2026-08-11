# VNN-COMP-Style MNIST-FC Artifacts

This folder documents TorchLean's current VNN-COMP-facing shape: exported JSON artifacts for a
small MNIST fully-connected benchmark family, checked by Lean through
`NN.Verification.VNNComp.MnistFC`.

VNN-COMP uses standard benchmark packages, network formats, property files, timeouts, and solver
reporting rules. TorchLean currently converts a benchmark instance into explicit JSON objects, then
runs a Lean checker over the exported weights, properties, and optional verifier metadata.

Expected local layout:

```text
_external/vnncomp/mnist_fc/model_weights.json
_external/vnncomp/mnist_fc/suite.json
_external/vnncomp/mnist_fc/alphas_crownobj.json   # optional
```

Run with:

```bash
lake exe verify -- vnncomp-mnistfc \
  --weights=_external/vnncomp/mnist_fc/model_weights.json \
  --suite=_external/vnncomp/mnist_fc/suite.json \
  --max=2
```

If the artifacts live somewhere else, pass `--weights=...`, `--suite=...`, and optionally
`--alphas=...`.

The checked object is the exported suite item: network weights, input region, expected label or
margin property, and any optional bound data attached to the item. The external preparation step is
responsible for converting the original benchmark files into this JSON shape. Full benchmark dumps,
large model files, and solver outputs belong in `_external/` or another local data directory rather
than in git.

The long-term direction is a native Lean verifier path that can read standard benchmark objects
more directly. This example is the small, reviewable bridge: VNN-COMP-style problem data enters
TorchLean as explicit artifacts, and Lean checks the part of the property represented in those
artifacts.
