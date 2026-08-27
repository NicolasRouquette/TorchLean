# Verification Artifacts and Workflows

This directory contains the bundled verification artifacts and runnable entry modules used by TorchLean's
unified verification CLI.

Reusable verification code lives under `NN/Verification/*`.
These files are the small assets and entry modules that keep `lake exe verify`
reproducible without pulling in large benchmark dumps.

## What To Run

- `lake exe verify -- all`
  Runs the fast certificate checkers that are safe for routine regression checks.

- `lake exe verify -- digits --eps=0.02 --max=360`
  Loads the bundled sklearn digits weights and test set, lowers the linear classifier through the
  TorchLean verifier bridge, and reports IBP/CROWN certified accuracy.

- `lake exe verify -- digits-train-certify --epochs=50 --eps=0.02 --max=100`
  Trains a fresh sklearn digits linear classifier with the local Python producer, exports weights
  and a test split, then immediately lowers and certifies those artifacts inside Lean.

- `lake exe verify -- margin-report`
  Recomputes the margin predicate and summary fields in an exported digits logit-bound report.
  The command does not establish that the supplied bounds enclose the model.

- `lake exe verify -- torchlean-robustness`
  Builds a small TorchLean classifier, lowers it to verifier IR, and checks the margin with
  IBP, forward affine CROWN, and backward objective CROWN.

- `lake exe verify -- torchlean-crown-ops`
  Exercises nonlinear verifier ops such as softmax and MSE loss on small TorchLean graphs.

- `lake exe verify -- abcrown-leaf`
  Parses the bundled alpha-beta-CROWN-style leaf artifact and recomputes the local certificate
  predicate inside Lean.

- `lake exe verify -- vnncomp-mnistfc`
  Loads a compact VNN-COMP-style fully connected MNIST network/property pair and checks the
  supported robustness condition through the TorchLean verifier path.

- `lake exe verify -- camera-box3d-cert`
  Recomputes a camera projection certificate for a 3D box artifact and checks that the claimed 2D
  envelope contains the projected corners.

- `lake exe verify -- spline-cert`
  Checks an exact rational piecewise polynomial certificate. With `--regen`, Julia is used only as
  an untrusted producer and Lean checks the regenerated JSON payload.

## Workflow Tiers

- Native TorchLean verification: `NN.Verification.Robustness.TorchLean` and
  `NN.Verification.TorchLean.*` build models in TorchLean, lower them to verifier IR, and run
  bound propagation directly.

- Exporter-backed verification: `LiRPA/*`, `VNNComp/*`, `AbCrown/*`, `Geometry3D/*`, `ODE/*`, and
  `PINN/*` hold bundled artifacts consumed by reusable CLI/checker code under `NN/Verification`.
  Python or external tools may produce candidate JSON artifacts, but Lean still parses and checks
  the artifact before accepting it.

- Artifact checkers: `LiRPA/*`, `AbCrown/*`,
  `NN.Verification.Robustness.MarginCert`, and `NN.Verification.Splines.PiecewiseLinearCLI` parse
  external artifacts and recompute their declared conditions inside Lean. The margin-report
  checker validates only report consistency. Classifier margin checks share
  `NN.Verification.Robustness.TopLabel`, so JSON reports and in-memory IBP/CROWN bounds use the same
  top-label rule.

- Data-backed robustness: `lake exe verify -- digits` runs `NN.Verification.Robustness.Digits`,
  which loads the exported sklearn digits weights and test data stored in `Robustness/`.
  `lake exe verify -- digits-train-certify` runs the producer first and then checks the newly
  exported artifacts through the same Lean lowering and bound engines.

- ODE/PINN workflows: `ODE/*` and `PINN/*` hold small certificate/dataset assets. The checker
  implementations live under `NN.Verification.ODE` and `NN.Verification.PINN`, with Python scripts
  under `scripts/verification/` used as untrusted producers for weights or candidate certificates.

- Proof map: theorem-level graph IBP/CROWN-family soundness is developed under
  `NN.MLTheory.CROWN.Proofs.*` and imported by `NN.Verification`, not defined in this
  artifact directory.

Reusable Lean code for ODE/PINN and certificate checking belongs under `NN/Verification`.
The `ODE/`, `PINN/`, `AbCrown/`, and `LiRPA/` folders here should contain small artifacts, notes,
or thin runnable entries. Producers generally belong under `scripts/verification/`.

## Trust Boundaries

External tools may produce JSON, weights, alpha slopes, or candidate bounds. Those artifacts are
not trusted. Treat a workflow as checked only when Lean parses the artifact, checks shapes, and
recomputes the relevant predicate or bound.

Some JSON checkers compare decimal serialized floating point values with an explicit
tolerance. That checks the serialized artifact against the declared tolerance; soundness of the
producer is a separate claim. For the theorem path, use the proof modules re-exported by
`NN.Verification`, which state checker-style soundness over the Lean graph semantics once
the local certificate hypotheses are discharged.

## Small Constants Versus Real Data

Small hand-written tensors are useful in TorchLean-native operator workflows because the whole
graph, input box, and property can be inspected in one file. Workflows that make data claims should
load weights and datasets from documented assets. Digits artifacts are bundled; large VNN-COMP
exports are kept outside git and passed to the checker explicitly.

## Artifact Parsers And Assets

Reusable parsing belongs in `NN.Verification`, not in individual example files. In particular,
`NN.Verification.Util.Json` provides the shared artifact boundary: read a JSON file, require a
schema `format`, and extract typed fields such as objects, arrays, natural numbers, booleans, and
float arrays with contextual errors.

Small JSON files are kept only when they make an example reproducible with one command. Larger
benchmark assets should be generated or downloaded by the documented scripts and treated like data
artifacts, not hand maintained source code.

Use the asset catalog to see or run the available regeneration commands:

```bash
python3 scripts/verification/regenerate_assets.py --list
python3 scripts/verification/regenerate_assets.py --group digits --run
python3 scripts/verification/regenerate_assets.py --group lirpa --run
```

Current asset policy:

| Asset class | Keep in git? | Regeneration path |
| --- | --- | --- |
| Small checker artifacts (`LiRPA/*.json`, `AbCrown/sample_*.json`, `Splines/*.json`) | Yes, if they keep CLI checks offline and small. | `regenerate_assets.py --group lirpa`, `lake exe verify -- spline-cert --regen`, or the local exporter. |
| Digits robustness artifacts | Yes, while they keep the certified accuracy example reproducible offline. | `regenerate_assets.py --group digits --run`. |
| PINN small certs/datasets | Keep only small curated artifacts; store trained checkpoints outside git. | `regenerate_assets.py --group pinn-small --run` and `PINN/train_*.py` for local runs. |
| PINN trained checkpoints/weight dumps | No. They are generated local outputs. | `regenerate_assets.py --group pinn-train --run`; outputs land in ignored paths. |
| ODE small certificates/weights | Yes, if they remain curated and small. | `regenerate_assets.py --group ode --run` checks the default curated artifact. |
| VNN-COMP snapshots | No. Keep model/suite exports outside git. | Store under `_external/vnncomp/...` or pass explicit `--weights=... --suite=...` paths. |
| Two stage controller/Lyapunov weights | No. Treat as local experiment output. | `regenerate_assets.py --group two-stage --run`, which writes to `_external/` by default. |
