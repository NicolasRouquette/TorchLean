<h1 align="center">
  <img src="home_page/assets/media/brand/torchlean-logo.png" alt="TorchLean logo" width="88" align="center">
  Formalizing Neural Networks in Lean
</h1>

TorchLean is a Lean 4 framework for writing, running, inspecting, and verifying
neural-network programs. It provides typed tensors and model APIs, a shared graph
IR, runtime/autograd support, finite-precision semantics, certificate checkers,
CUDA/runtime boundaries, and examples across modern ML and scientific ML.

## Installation

```bash
git clone https://github.com/lean-dojo/TorchLean.git
cd TorchLean
lake exe cache get
lake build
```

For Linux, macOS, Windows/WSL, CUDA, optional LibTorch support, and an explanation of
TorchLean's backend architecture, see the [Installation guide](https://lean-dojo.github.io/TorchLean/installation/).

TorchLean is pinned by `lean-toolchain` and currently builds with
`leanprover/lean4:v4.32.0`.

## Quickstart

```bash
lake exe torchlean quickstart_mlp --device cpu --steps 10 --dtype float32 --backend eager
lake exe torchlean quickstart_mlp --device cpu --steps 10 --dtype float --backend eager

# Optional CUDA run, if the CUDA toolkit and an NVIDIA GPU are available:
lake -R -K cuda=true build
lake -R -K cuda=true exe torchlean mlp --device cuda --steps 1000
```

The first quickstart uses TorchLean's executable IEEE-style Float32 scalar. The
second uses Lean's builtin `Float` runtime path. The CUDA command uses the
native GPU runtime path and checks that the CUDA backend is available. Theorem
statements that mention CUDA cite the native-runtime boundary in
`TRUST_BOUNDARIES.md` instead of treating a kernel launch as Lean proof evidence.

Application code looks like this:

```lean
import NN.API
open TorchLean

/-- A two-layer regression model. The dimensions are checked when the layers are composed. -/
def model :=
  nn.Sequential![
    nn.Linear 2 8,
    nn.ReLU,
    nn.Linear 8 1
  ]

-- Four input rows, each containing two features.
def xs : Tensor.T Float (shape![4, 2]) :=
  tensorOfList! [4, 2] [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]

-- One regression target for each input row.
def ys : Tensor.T Float (shape![4, 1]) :=
  tensorOfList! [4, 1] [0.2, 1.0, 1.0, 1.8]

-- The dataset type records the feature and target shapes expected by the trainer.
def data : Trainer.Dataset (.dim 2 .scalar) (.dim 1 .scalar) :=
  Data.tensorDataset xs ys

def trainOnce : IO Unit := do
  -- Select the loss from the regression task and train through the compiled Float32 path.
  let trainer :=
    Trainer.new model
      { task := .regression
        optimizer := optim.sgd { lr := 0.05 }
        backend := .compiled
        dtype := .float32 }
  -- Inspect the initialized model before any parameter updates.
  let initialPrediction ← trainer.predict (tensorOfList! [2] [0.5, -0.25])
  IO.println s!"initial={Tensor.pretty initialPrediction}"
  -- Each step averages 16 sample gradients at one parameter point, then updates once.
  -- Training returns a new trainer containing the updated parameters and run history.
  let trained ← trainer.train data { steps := 200, batchSize := 16, logEvery := 25 }
  trained.printSummary
```

## First Things To Try

```bash
lake exe torchlean --help
lake exe verify --help
lake exe verify -- torchlean-ibp
```

For the maintained examples:

```bash
lake build NNExamples
```

## Use TorchLean From Another Lean Project

TorchLean is a normal Lake package. You can depend on the Git repository directly:

```lean
require TorchLean from git "https://github.com/lean-dojo/TorchLean.git" @ "main"
```

Then run:

```bash
lake update
lake exe cache get
lake build
```

Use `import NN.API` for model, data, and training code. It provides `TorchLean.nn`,
`TorchLean.Data`, `TorchLean.Trainer`, and `TorchLean.optim`, together with the mathematical model
definitions in `Spec`. Use `import NN` when the same file also needs verification, proofs, or
backend infrastructure; focused imports such as `NN.GraphSpec`, `NN.Runtime`, or `NN.Proofs` are
available for subsystem work.

Downstream model and training files should start from:

```lean
import NN.API
open TorchLean
```

The floating-point library can also be used on its own:

```lean
import NN.Floats
open TorchLean.Floats
```

This import provides generic formats and rounding, finite binary32 semantics, executable IEEE
binary32 operations, interval rounders, and scalar quantization. It does not import tensors,
models, autograd, CUDA, certificate checkers, or external numerical tools. More specialized users
can import `NN.Floats.NeuralFloat`, `NN.Floats.FP32`, `NN.Floats.IEEEExec`, or
`NN.Floats.Interval` directly. Tensor quantization and runtime-approximation proofs are separate
adapters under `NN.Spec.Quantization` and `NN.Proofs.RuntimeApprox.FP32`.

For local development against a checkout, use a path dependency instead:

```lean
require TorchLean from "../TorchLean"
```

## Repository Map

- `NN.lean`: complete import for model, tensor, data, training, verification, and proof workflows.
- `NN/API`: the application API exported by `import NN.API` and included by `import NN`.
- `NN/Spec`: mathematical tensor, layer, model, and dynamical-system definitions.
- `NN/Runtime`: executable autograd, optimizers, training loops, CUDA boundary,
  PyTorch import/export, and RL runtime support.
- `NN/Backend`: contract-carrying backend capsules, profiles, device targets, reports, and gates.
- `NN/IR` and `NN/GraphSpec`: graph IR, graph semantics, and typed architecture
  descriptions.
- `NN/Proofs`: tensor algebra, selected autograd correctness theorems, analytic derivatives,
  runtime approximation, and bridge proofs.
- `NN/Floats`: finite-precision models, IEEE-style executable semantics,
  NeuralFloat formats, and error-bound infrastructure.
- `NN/MLTheory`: learning theory, robustness, CROWN/LiRPA, generative objectives,
  optimization theory, and related proof layers.
- `NN/Verification`: certificate checkers and CLI workflows.
- `NN/Examples`: quickstarts, model zoo commands, widgets, bundled verification assets,
  and interoperability workflows.
- `blueprint/TorchLeanBlueprint/Guide`: source for the guide.
- `home_page`: project website sources.

## Current Capabilities

The same rule applies across the tree: name the object, name the artifact, and name the boundary.

- **Training and runtime.** `Trainer.new` supports supervised tasks, scalar/backend choices,
  trained handles, prediction, logs, typed step streams, generated or file-backed batches, and
  optional CUDA-backed Float32 runtime paths. Application code uses one trainer with selected
  backend options instead of one forward method per backend.
- **Graphs and compiler fragments.** TorchLean models can be lowered to a shared IR. A first-order
  forward fragment has Lean-side compiler-correctness theorems, and coverage grows operation by
  operation. GraphSpec describes typed architectures above the lower-level op-tagged DAG consumed by
  runtime, widgets, exporters, and verification passes.
- **Verification.** The repository includes IBP/CROWN-style graph checks, α/β-CROWN-style artifact
  replay, robustness workflows, VNN-COMP-style MNIST checks, PINN residual checks, ODE corridors,
  spline certificates, and 3D geometry projection certificates. External producers write compact
  artifacts; Lean parses the artifact, checks the stated predicate, and records any remaining
  producer assumptions.
- **ML theory.** The theory layer covers CROWN/LiRPA objects, optimizer laws, Muon
  orthogonalization certificates, learning-theory examples, generative objective identities,
  self-supervised-learning algebra, approximation theorems, and floating-point bridges. Optimizers
  use a generic `TensorOptimizer` interface so new update rules share finite-stream laws. A
  `StepSpec` is available when an independent mathematical recurrence must be related to the
  executable update.
- **Scientific ML.** The FNO Burgers and PINN/ODE paths show how numerical artifacts can be carried
  back into Lean checks. External simulators and optimizers remain named producers; Lean owns the
  artifact schemas, residual predicates, dataset checks, and certificate replay statements that sit
  at the verification boundary.

## What To Cite For A Claim

| Claim shape | Where to look |
| --- | --- |
| "This model runs and trains" | `NN/Examples`, `NN/Runtime`, command output, and regression tests. |
| "This graph has a checked bound/certificate" | `NN/Verification`, `NN/MLTheory/CROWN`, and the artifact schema named by the command. |
| "This compilation/evaluation fragment preserves meaning" | `NN/Verification/TorchLean/Proved` and the theorem imported through `NN.Verification`. |
| "This optimizer update follows the intended equation" | `NN/Runtime/Optim` for executable equations and `NN/MLTheory/Optimization` for reusable laws. |
| "This finite-precision statement has a formal model" | `NN/Floats`, `NN/Proofs/RuntimeApprox`, and the relevant bridge hypotheses. |
| "This CUDA/LibTorch/PyTorch/Julia/Gymnasium path was used" | `TRUST_BOUNDARIES.md`, the runtime module docstring, and the command or artifact provenance. |

## Correctness and Boundaries

For correctness claims, trust assumptions, and third-party tooling:

- `TRUST_BOUNDARIES.md`
- `AI_USAGE.md`
- `THIRD_PARTY_NOTICES.md`
- `CONTRIBUTING.md`

Lean proofs, executable checkers, Lake builds, tests, and explicit
trust-boundary documentation are the source of truth for what is proved,
checked, or assumed.

## Citation

If TorchLean is useful in your work, please cite
[*TorchLean: Formalizing Neural Networks in Lean*](https://arxiv.org/abs/2602.22631):

```bibtex
@misc{george2026torchlean,
  title         = {TorchLean: Formalizing Neural Networks in Lean},
  author        = {George, Robert Joseph and Cruden, Jennifer and Adkisson, Will and
                   Zhong, Xiangru and Zhang, Huan and Anandkumar, Anima},
  year          = {2026},
  eprint        = {2602.22631},
  archivePrefix = {arXiv},
  primaryClass  = {cs.MS},
  url           = {https://arxiv.org/abs/2602.22631}
}
```

## License

TorchLean is released under the MIT License. See `LICENSE`.
