import VersoManual

open Verso.Genre Manual

#doc (Manual) "Command-Line Reference" =>
%%%
tag := "cli"
%%%

The first command-line decision is simpler than the number of examples suggests. Ask whether you
want to *run a model or demonstration* or *check a verification claim*. TorchLean gives those jobs
separate dispatchers:

```
lake exe torchlean <example> [flags...]
lake exe verify -- <tool> [args...]
```

The first runs examples, training applications, data checks, and numerical deep dives. The second
runs verifiers and certificate checkers. Keeping them separate makes a successful training command
harder to confuse with acceptance of a certificate. Small source-local programs can also be
executed directly:

```
lake env lean --run NN/Examples/Quickstart/TensorBasics.lean
```

The dispatch tables are ordinary Lean definitions:

- [`NN.Examples.Models.Runner`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Models/Runner.lean)
  routes the `torchlean` subcommands;
- [`NN.Verification.CLI`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/CLI.lean)
  routes the `verify` tools.

The help printed by your checkout is authoritative for command names and flags because it is
generated from the executable being run.

# Discovering Commands

Start with:

```
lake exe torchlean --help
```

The current help begins:

```
TorchLean runnable examples

Usage:
  lake exe torchlean <example> [flags...]
  lake exe torchlean --choose <example> [flags...]
  lake exe torchlean <example> --help

Start here:
  lake exe torchlean quickstart_tensors
  lake exe torchlean quickstart_autograd
  lake exe torchlean quickstart_mlp --steps 20
```

The shorter list in top-level help is a starting point, not the complete dispatch table. Verification
tools are listed independently:

```
lake exe verify -- list
```

That command currently includes in-memory TorchLean-to-IR workflows, LiRPA artifact checkers,
PINN and geometry certificate checkers, numerical ODE tools, and VNN-COMP-style applications. Use
the list printed by your checkout rather than copying a stale inventory from a paper or issue.

# Command Grammar

The normal form is:

```
lake exe torchlean <subcommand> [runtime flags] [command flags]
```

For example:

```
lake exe torchlean quickstart_mlp \
  --device cpu --steps 20 --seed 2026
```

Runtime flags may also precede the subcommand:

```
lake exe torchlean --device cpu quickstart_mlp \
  --steps 20 --seed 2026
```

Both forms reach the same parser. Documentation uses the first because the application name appears
before its options.

A leading separator is accepted for wrappers that require one:

```
lake exe torchlean -- quickstart_mlp --device cpu --steps 20
```

# The Interactive Device Chooser

Use `--choose` when running a command by hand and you do not want to remember the device flag:

```
lake exe torchlean --choose quickstart_mlp --steps 1
```

The prompt is:

```
TorchLean runtime chooser
Runtime device:
  1) CPU    portable default
  2) CUDA   GPU runtime, requires `lake -R -K cuda=true exe ...`
Select device [1]:
```

Pressing Enter selects CPU. The chooser is opt-in so shell scripts, tests, and continuous
integration never block waiting for input. It currently chooses only between implemented CPU and
CUDA execution. Other target names belong to the backend registry but do not yet have runnable
training engines.

# Runtime Flags

The common runtime parser recognizes:

:::table +header
*
  * Flag
  * Current meaning
*
  * `--device auto|cpu|cuda|...`
  * requested execution device
*
  * `--scalar float32|ieee32-exec|complex64`
  * scalar runtime, where the command supports it
*
  * `--execution eager|typed-graph`
  * eager autograd or shape-indexed typed graph host path where supported
*
  * `--seed N`
  * explicit random seed
*
  * `--show-backend`
  * print selected backend capsules
:::

The parser also accepts names such as `rocm`, `metal`, `wasm`, `tpu`, `trainium`, `custom`, and
`external`. They are planning targets in the current registry, not completed runtimes. Requesting
one fails validation rather than silently falling back to CPU.

Most model commands use native `Float32`. Scalar-polymorphic quickstarts and numerical workflows
also accept `--scalar ieee32-exec`. A command reports an error when it cannot implement the selected
semantics; the shared parser never treats a recognized name as proof that every runtime supports it.

`--execution typed-graph` selects the shape-indexed graph path for commands that implement it. It
does not request a generic CUDA graph mode, perform compiler optimization, or by itself establish a
derivative theorem. CUDA-only specialized applications may require eager execution.

# CPU And CUDA Builds

CPU execution uses the ordinary build:

```
lake build
lake exe torchlean quickstart_mlp --device cpu --steps 20
```

Real CUDA execution must be compiled with the Lake option:

```
lake -R -K cuda=true build
lake -R -K cuda=true exe torchlean chargpt --device cuda \
  --tiny-shakespeare --preset smoke
```

Keep `-R` when changing a Lake configuration so affected native archives are rebuilt. Without
`cuda=true`, TorchLean links portable CUDA stubs. An explicit `--device cuda` request then fails;
it does not pretend to have used the GPU.

Add `--show-backend` to inspect the selected contracts:

```
lake -R -K cuda=true exe torchlean quickstart_mlp \
  --device cuda --steps 1 --show-backend
```

The printed capsules name the operation, provider, layouts, numerical policy, forward and backward
implementation responsibility, and evidence level. Device selection answers “where did this
run?”; capsule reporting answers the more precise question “which implementation was selected for
each operation?”

# Command-Specific Flags

Ask the subcommand for help:

```
lake exe torchlean chargpt --help
```

The CharGPT command documents presets and structural overrides:

```
Presets:
  smoke       two-update end-to-end CUDA check
  karpathy    full Tiny Shakespeare lecture experiment

Architecture:
  --width N       embedding width
  --heads N       attention heads; must divide width
  --layers N      Transformer blocks
  --dropout P     dropout probability in [0, 1)
  --batch N       training windows per update
  --seq-len N     context length
  --steps N       optimizer updates
```

Not every application has equally detailed command-specific help yet. If a generic runtime page is
printed, inspect the module documentation or parser next to that application. The source links in
the preceding chapters point to the maintained definitions.

# Strict Parsing

TorchLean parsers remove the flags they understand and then reject whatever remains. This catches
misspellings and copied options from another command.

For example, GridWorld has a fixed source-level horizon and does not accept `--rollout`:

```
lake exe torchlean ppo_gridworld \
  --device cpu --updates 1 --rollout 8
```

The command fails with:

```
torchlean ppo_gridworld: unexpected arguments: [--rollout, 8]
```

The FNO command expects `--x` and `--y` for training arrays, not `--train-x` and `--train-y`.
Unknown path flags fail in the same way.

Shared parsing lives in
[`NN.API.CLI`](https://github.com/lean-dojo/TorchLean/blob/main/NN/API/CLI.lean)
under the `TorchLean.CLI` namespace. A lightweight command can write:

```
import NN.API.CLI

open TorchLean

#check CLI.takeFlagValueOnce
#check CLI.takeNatFlagOnce
#check CLI.takeBoolValueFlagOnce
#check CLI.takePathFlagOnce
#check CLI.checkNoArgs
```

Value flags accept both `--key value` and `--key=value`. Duplicate occurrences are rejected.
Parsers return the unconsumed argument list, and `checkNoArgs` closes the loop.

# Preparing Data

The small dataset downloader prepares the common datasets:

```
python3 scripts/datasets/download_example_data.py \
  --auto-mpg --tiny-shakespeare --tinystories-valid --cifar10
```

The forecasting and Burgers applications use dedicated preparation:

```
python3 scripts/datasets/download_example_data.py \
  --household-power --household-power-windows 512

python3 NN/Examples/Data/prepare_fno1d_burgers.py \
  --download --grid 32 --ntrain 128 --ntest 32
```

Convert a local labeled image folder with:

```
python3 scripts/datasets/torchlean_data_convert.py image-folder \
  --input /path/to/images \
  --x-output data/real/imagenet64/imagenet64_train_X.npy \
  --y-output data/real/imagenet64/imagenet64_train_y.npy \
  --height 64 --width 64 \
  --labels-from-dirs --limit 2000
```

These Python tools are data producers. Lean checks file existence, array metadata, dimensions,
finite values where the loader contract requires them, and typed conversion. It does not prove the
downloaded labels or preprocessing script semantically correct.

# Artifact Paths

Pass explicit paths when an output will be inspected later:

```
lake exe torchlean autoencoder --device cpu \
  --n-total 1 --steps 1 \
  --log /tmp/autoencoder-trainlog.json
```

```
lake exe torchlean ppo_gridworld --device cpu \
  --updates 1 --eval-every 1 \
  --log /tmp/ppo-trainlog.json \
  --policy /tmp/ppo-policy.json \
  --path /tmp/ppo-path.json
```

```
lake -R -K cuda=true exe torchlean diffusion --device cuda \
  --dataset cifar10 --n-total 8 --steps 20 --T 20 \
  --sample-ppm /tmp/diffusion-sample.ppm
```

The path is part of the experiment. It tells a widget, plotting script, or checker which exact
object it is reading.

# Verification Commands

List the registry, then choose one workflow:

```
lake exe verify -- list
lake exe verify -- torchlean-ibp
```

An in-memory workflow such as `torchlean-ibp` constructs a model, lowers it to IR, and runs a bound
algorithm. An artifact checker such as `pinn-cert` or `abcrown-leaf` parses a file produced
elsewhere. Their success messages have different meanings.

Representative forms are:

```
lake exe verify -- torchlean-transformer-ibp
lake exe verify -- pinn-cert path/to/pinn-cert.json
lake exe verify -- abcrown-leaf path/to/leaf-artifact.json
lake exe verify -- camera-box3d-cert path/to/camera-cert.json
```

A checker accepts only its declared schema and semantic fragment. It does not retroactively verify
the external process that produced the artifact.

The current registry is broader than those representative commands. Its families are:

- LiRPA artifacts: `lirpa-mlp`, `lirpa-cnn`, `lirpa-attention`, `lirpa-gru`, `lirpa-encoder`;
- scientific and geometry checkers: `camera-box3d-cert`, `pinn-cert`, `pinn-cli`,
  `pinn-dataset-check`, `spline-cert`, `ode`;
- TorchLean graph workflows: `torchlean-robustness`, `torchlean-ibp`,
  `torchlean-transformer-ibp`, `torchlean-crown-ops`, `torchlean-mlp-workflow`;
- robustness suites: `margin-cert`, `digits`, `digits-train-certify`, `vnncomp-mnistfc`;
- imported and two-stage evidence: `abcrown-leaf`, `twostage-pythononly-certgen`,
  `twostage-hybrid-van-stage2`, `twostage-torchlean-cegis-van`.

`lake exe verify -- all` runs only entries whose registry record sets `includeInAll := true`.
Interactive, externally dependent, or long-running tools are intentionally excluded, so a green
`all` run is not evidence that every line printed by `list` was executed.

# A Practical Validation Sequence

After a fresh clone or a substantial local change, run these commands sequentially:

```
lake build
lake build NNExamples
lake exe torchlean quickstart_tensors
lake exe torchlean quickstart_autograd
lake exe torchlean quickstart_mlp --device cpu --steps 20 --seed 2026
lake exe torchlean numerical_certificate
lake exe verify -- torchlean-ibp
```

They answer different questions:

:::table +header
*
  * Command
  * Question
*
  * `lake build`
  * does the project elaborate and link?
*
  * `lake build NNExamples`
  * do the curated examples elaborate?
*
  * tensor/autograd quickstarts
  * do small executable values and derivatives behave as expected?
*
  * MLP training
  * does a complete optimizer path run?
*
  * numerical certificate
  * are positive and negative certificate cases handled?
*
  * TorchLean IBP
  * does model lowering and bound propagation complete?
:::

For CUDA changes, follow with CUDA-specific builds and runtime checks. For external tools, prepare
their dependencies separately. No single green command subsumes all of these boundaries.

# Building The Website

The guide, API reference, and application pages are built by:

```
scripts/docs/build_site.sh
```

For a local Jekyll preview after the generated site exists:

```
cd home_page
bundle _2.3.14_ exec jekyll serve \
  --config _config.yml,_config_dev.yml \
  --host 127.0.0.1 --port 4001
```

Then open:

```
http://127.0.0.1:4001/
http://127.0.0.1:4001/blueprint/
http://127.0.0.1:4001/examples/
http://127.0.0.1:4001/docs/
```

If the source changed but the browser did not, rebuild the generated site before restarting
Jekyll. A running web server cannot regenerate Lean documentation on its own.
