# VNN-COMP Verification Assets

`MnistFcVerify.lean` is a runnable checker for VNN-COMP style JSON exports, but large benchmark
snapshots are easier to work with as local `_external/` artifacts.

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

If you keep the artifacts somewhere else, pass `--weights=...`, `--suite=...`, and optionally
`--alphas=...`.

This directory keeps the Lean checker, small documentation, and conversion entrypoints. Full
VNN-COMP model dumps and suite exports live in `_external/` or another local data directory.
