# Examples

The examples are working paths through TorchLean. A small file starts with ordinary model code and
leaves behind something more precise than "the command ran": a typed tensor, a training log, a
graph, a certificate, an exported artifact, or a theorem statement that can be inspected afterward.

That is the rhythm to look for while reading this directory. First find the model and the data. Then
find the boundary: runtime execution, CUDA, PyTorch interop, a verifier artifact, or a proof layer
definition. The examples keep the artifact small enough to inspect because their purpose is to make
that boundary visible.

Use the curated CLIs first:

```bash
lake exe torchlean --help
lake exe verify --help
lake test
```

Runnable model examples are subcommands of `lake exe torchlean <name>`. Verification workflows are
subcommands of `lake exe verify -- <tool>`.

## First Path

Start here if you want a compact tour from typed tensors to training and verification.

| Goal | File | Command | What to look for |
| --- | --- | --- | --- |
| Typed tensors | `Quickstart/TensorBasics.lean` | `lake exe torchlean quickstart_tensors` | Shape-indexed tensors, constructors, and basic operations before any training machinery appears. |
| Editor widgets | `Quickstart/Widgets.lean` | open the file and run the `#..._view` commands | Lean side views for tensors, graphs, and small runtime objects. |
| Runtime scalar modes | `DeepDives/Floats/Float32Modes.lean` | `lake exe torchlean float32_modes` | The difference between exact/ideal values and executable Float32 paths. |
| Effective float32 rounding | `DeepDives/Floats/EffectiveRounding.lean` | `lake build NN.Examples.DeepDives.Floats.EffectiveRounding` | The same shaped tensor addition in the `FP32` proof model and the executable IEEE model, with the resulting mantissa and exponent exposed by theorem. |
| Autograd API | `Quickstart/AutogradBasics.lean` | `lake exe torchlean quickstart_autograd` | The tape records operations, runs backward, and reports gradients for closed-form checks. |
| Proof basics | `Quickstart/Proofs.lean` | `lake build NN.Examples.Quickstart.Proofs` | Small theorem statements over tensor expressions and model fragments. |
| Simple training | `Quickstart/SimpleMlpTrain.lean` | `lake exe torchlean quickstart_mlp --steps 200 --dtype float32 --backend compiled` | A `Trainer` run with compiled execution, loss reporting, and parameter updates. |
| Data loading | `Data/Loaders/Csv.lean` | `lake exe torchlean data_csv --steps 30 --batch 5 --dtype float --backend eager` | A file-backed batch stream crossing into typed TorchLean data. |
| Verification | `Verification/TorchLean/*` | `lake exe verify -- torchlean-ibp` | A TorchLean model lowered into a verifier graph and checked by a native bound workflow. |
| PyTorch import/export | `Interop/PyTorch/Roundtrip.lean` | `lake exe torchlean pytorch_roundtrip --model mlp --action import` | Weight and graph exchange at an explicit trust boundary. |

For the larger or newer workflows:

| Workflow | File or artifact | Command | What it exposes |
| --- | --- | --- | --- |
| Burgers FNO | `Models/Operators/Fno1dBurgers.lean` | `lake -R -K cuda=true exe torchlean fno1d_burgers --device cuda --steps 1` | Scientific-ML data, Fourier layers, prediction artifacts, and the path from PDE trajectories to Lean-checkable outputs. |
| alpha-beta-CROWN leaf artifact | `Verification/AbCrown/sample_abcrown_leaf_artifact_v0_1.json` | `lake exe verify -- abcrown-leaf` | Lean checks a leaf artifact schema and local certificate condition without trusting the external branch-and-bound run. |
| LiRPA fixture artifacts | `Verification/LiRPA/*.json` | `lake exe verify -- lirpa-mlp` and sibling commands | Small exported bound artifacts for MLP, CNN, attention, GRU, and transformer encoder fragments. |
| PINN residual certificate | `Verification/PINN/pinn_cert.json` | `lake exe verify -- pinn-cert` | A neural PDE residual certificate is parsed into Lean side expressions and checked against the declared grid/bounds. |
| PINN dataset containment | `Verification/PINN/sample_dataset_1d.json` | `lake exe verify -- pinn-dataset-check` | Dataset metadata and coordinate bounds are checked before a scientific ML artifact is treated as a valid verification input. |
| Digits train and certify | `Verification/Robustness/*` | `lake exe verify -- digits-train-certify --epochs=5 --max=20` | A small classifier is trained, imported, and certified by a Lean side robustness checker. |
| VNN-COMP-style MNIST-FC | `Verification/VNNComp/` | `lake exe verify -- vnncomp-mnistfc` | A VNN-COMP-shaped robustness query using TorchLean's native graph and margin-certificate machinery. |
| 3D projection certificate | `Verification/Geometry3D` and `BugZoo/Geometry3DProjection.lean` | `lake exe verify -- camera-box3d-cert` | Camera and box tensors are replayed in Lean to check positive depth, projection, and 2D enclosure. |
| Spline certificate | `Verification/Splines` | `lake exe verify -- spline-cert` | Piecewise-polynomial certificate data is parsed and checked against Lean side interval predicates. |
| Optimizer certificates | `Optimization/MuonCertificates.lean` | `lake build NN.Examples.Optimization` | Muon update rules, QR/Newton-Schulz orthogonalizer contracts, and theorem names for downstream optimizer proofs. |

## What An Example Should Leave Behind

A TorchLean example should leave one of these behind:

| Artifact | Why it matters |
| --- | --- |
| Printed tensor or loss trace | Confirms the runtime path is executing the intended model and scalar mode. |
| Training log JSON | Gives plots, regression checks, and a stable record of a run. |
| Imported/exported weights | Makes the trust boundary explicit when TorchLean compares against PyTorch or another tool. |
| Graph or IR object | Gives the verifier and graph semantics a concrete model to inspect. |
| Certificate JSON | Lets Lean check a claim produced by training, a script, or an external verifier. |
| Lean theorem target | Turns the example into a machine-checked statement over the named semantics. |

## Directory Map

| Directory | Purpose |
| --- | --- |
| `Quickstart/` | Small examples for tensors, autograd, widgets, proofs, and first training loops. |
| `Data/` | CSV/NPY loaders, tutorial artifact generation, and dataset preparation notes. |
| `Models/` | Runnable supervised, vision, sequence, generative, operator-learning, and RL commands. |
| `Verification/` | Bundled verifier examples and small local artifacts for `lake exe verify`. |
| `Interop/PyTorch/` | JSON round trips and PyTorch graph export checks. |
| `DeepDives/` | Float semantics, GraphSpec, IR, widgets, and deeper tutorial files. |
| `Optimization/` | Optimizer-law examples and certificates for Muon-style updates. |
| `BugZoo/` | Checked versions of common ML failure modes. |
| `RL/` | Viewer files for RL artifacts generated by runtime examples. |

## Build Targets

Use these when checking a whole slice:

```bash
lake build NN.Examples.Zoo
lake build NN.Examples.DeepDives
lake build NN.Examples.Optimization
lake build NN.Examples.BugZoo.All
```

The default `lake build` target is curated and does not compile every example module. Lake will build
the modules needed by a command when you run that command.

## Data

Generated tutorial artifacts are local and ignored by git:

```bash
python3 NN/Examples/Data/generate_small_data.py
```

Real-data examples use prepared files under `data/real`:

```bash
python3 scripts/datasets/download_example_data.py --auto-mpg --cifar10 --tiny-shakespeare
python3 scripts/datasets/download_example_data.py --tinystories-valid
python3 NN/Examples/Data/prepare_fno1d_burgers.py --download --grid 32 --ntrain 128 --ntest 32
```

The Lean boundary format is `.npy` or simple numeric CSV. Use
`scripts/datasets/torchlean_data_convert.py` for external formats such as image folders, `.pt`,
`.npz`, or `.mat` files.

Data examples should keep the boundary explicit. A loader may read a CSV, NPY file, or small
downloaded fixture, but the Lean side object should have declared dimensions, scalar type, labels,
and batch structure. That discipline is what lets the same data feed a runtime command, a graph
export, and a later certificate check.

## Common Commands

```bash
lake exe torchlean quickstart_mlp --steps 10 --dtype float32 --backend compiled
lake exe torchlean mlp --steps 10
lake -R -K cuda=true exe torchlean cnn --device cuda --steps 10
lake -R -K cuda=true exe torchlean gpt2 --device cuda --tiny-shakespeare --steps 10 --windows 1 --generate 0
lake -R -K cuda=true exe torchlean mamba --device cuda --tiny-shakespeare --steps 10 --windows 1 --generate 0
lake -R -K cuda=true exe torchlean ppo_gridworld --device cuda --updates 1 --eval-every 1 --eval-episodes 1 --eval-max-steps 8
lake exe verify -- torchlean-ibp
lake exe verify -- torchlean-crown-ops
lake exe verify -- abcrown-leaf
lake exe verify -- lirpa-mlp
lake exe verify -- pinn-cert
lake exe verify -- pinn-dataset-check
lake exe verify -- digits-train-certify --epochs=5 --eps=0.02 --max=20
lake exe verify -- vnncomp-mnistfc
lake exe verify -- camera-box3d-cert
lake exe verify -- spline-cert
```

CPU is the quick path for small tabular, tensor, and recurrent examples. CNN, ViT, Transformer,
GPT-style text, diffusion, FNO, and PPO examples are CUDA validation targets; their CPU paths help
with debugging but are too slow for the normal example matrix.

Training commands that expose `--log` can write a training curve:

```bash
lake exe torchlean mlp --steps 10 --log data/model_zoo/mlp_trainlog.json
python3 scripts/datasets/plot_trainlog.py data/model_zoo/mlp_trainlog.json --out-dir plots/model_zoo
```

Some commands write richer artifacts. `fno1d_burgers` can write a prediction CSV for plotting.
`diffusion` can write sampled/reconstructed image artifacts and a loss log. `gpt2` can save a
shape-indexed parameter pack that `gpt2_saved` reloads before sampling. `digits-train-certify`
trains a small classifier outside Lean, imports the weights and examples, then certifies the
resulting graph inside Lean. `abcrown-leaf` checks a bundled α,β-CROWN-style leaf artifact schema:
box nesting, array sizes, and the represented witness lower-bound comparison are checked in Lean,
while the external branch-and-bound search remains a named producer boundary.

## API Used By Examples

Runnable examples use the focused application API:

```lean
import NN.API
open TorchLean
```

Deep dives import a subsystem explicitly when they inspect its internal objects:

| API area | Use |
| --- | --- |
| `TorchLean` | Model, tensor, data, training, runtime, text, and RL names. |
| `NN.API.Neural` | Application API modules; ordinary examples should not import this directly. |
| `NN.Tensor` | Typed tensor constructors and semantics below the application API. |
| `NN.API.Runtime` | Runtime subsystem access for code that is explicitly extending the runtime layer. |
| `NN.Verification` | Verification APIs and theorems. |
| `NN.Verification.CLI` | The `lake exe verify` registry. |

New model and training examples should use the `TorchLean` names. An example that imports an
implementation module should exercise a declaration from that module directly.
