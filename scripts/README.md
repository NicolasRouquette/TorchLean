# TorchLean Scripts

This directory contains support commands for local checks, repository hygiene audits,
site generation, dataset preparation, artifact producers, and optional sandboxed Lean checking.

Generated files such as `__pycache__/`, downloaded datasets, built documentation, and local output
artifacts belong in ignored output directories such as `data/`, `_out/`, or `home_page/_site/`.

## Directory Map

Everything is organized by purpose:

- `checks/`: local CI, repository lint, dependency audit, CUDA sanitizer helpers, and Lake's Lean
  lint driver.
- `setup/`: optional local native-backend setup helpers.
- `docs/`: public site, DocGen, and Verso post-processing.
- `datasets/`: dataset download, conversion, and training-log plotting helpers.
- `verification/`: certificate producers and artifact-regeneration workflows.
- `rl/`: optional reinforcement-learning bridge examples.
- `sandbox/`: comparator/untrusted-Lean helper tooling.

### Build and Check Support

These scripts are used by the build, documentation, and local verification paths:

- `checks/check.sh`
- `checks/example_regression.sh`
- `checks/cuda_sanitize_tests.sh`
- `checks/cuda_profile_tests.sh`
- `checks/repo_lint.py`
- `checks/TorchLeanLint.lean`
- `checks/dependency_audit.py`
- `docs/build_site.sh`
- `docs/polish_docgen.py`
- `docs/polish_verso_guide.py`
- `setup/resolve_libtorch.sh`
- `sandbox/run_comparator.py`
- `comparator/nn_ci_all.json`

`docs/polish_docgen.py` and `docs/polish_verso_guide.py` are website post-processors rather than
verification logic. DocGen and Verso generate the HTML; these scripts add the TorchLean landing page,
navigation polish, responsive figures, copy buttons, asset copying, and public-site styling.

### Reproducibility Helpers

These are referenced by examples, docs, or artifact-regeneration workflows:

- `datasets/download_example_data.py`
- `datasets/torchlean_data_convert.py`
- `datasets/plot_trainlog.py`
- `verification/regenerate_assets.py`
- `verification/robustness/*`
- `verification/lirpa/*`
- `verification/pinn/*`
- `verification/splines/*`
- `verification/two_stage/*`
- `verification/geometry3d/*`

Use the subfolder paths directly, for example
`python3 scripts/datasets/download_example_data.py --cifar10`.

### Optional Examples and Research Workflows

These scripts support documented workflows outside the core build path:

- `datasets/download_wikitext.py`
- `rl/gymnasium_server.py`
- `rl/export_gymnasium_rollout.py`
- `rl/train_ppo_cartpole_sb3.py`

### Local Output

These are generated locally and stay out of the source tree:

- `__pycache__/`
- `*.pyc`
- local checkpoints, downloaded data, generated plots, generated JSON output
- scratch notebooks or ad hoc scripts until they are promoted into one of the groups above

## Plot and Asset Policy

Most plots are generated output. The source tree tracks only small documentation assets that are
referenced by Markdown or Verso pages.

Keep tracked:

- `home_page/assets/media/**` when referenced by homepage Markdown.
- `blueprint/TorchLeanBlueprint/Guide/Assets/**` when referenced by the Verso guide.

Generated locally:

- `_external/**`: local model outputs, Geometry3D certificates, overlays, and diagnostic PNGs.
- `_out/**`: generated Verso/blueprint build output.
- `home_page/docs/**`: generated DocGen API HTML.
- `home_page/_site/**`: generated Jekyll output.
- `data/**`: downloaded datasets, training logs, audit plots, and model outputs.
- `Two-Stage_Neural_Controller_Training/**`: local research/training output.

## Checks

- `checks/check.sh`: local verification gate for `lake build`, `lake test`, `lake lint`,
  and optional CUDA / `NN.CI.All` checks.
- `checks/example_regression.sh`: sequential regression check for the `lake exe torchlean ...`
  examples. It audits every registered subcommand's `--help` path and runs compact
  tutorial/interop examples; pass `--cuda` for a short real-CUDA model regression set, or
  `--extended-cuda` for a broader one-step model-zoo CUDA run. Optional external-environment
  checks, such as ALE/Pong, live behind `--external-rl`.
- `checks/cuda_sanitize_tests.sh`: CUDA sanitizer runner for the CUDA runtime test suite.
- `checks/cuda_profile_tests.sh`: optional Nsight Systems / Nsight Compute wrapper for CUDA
  performance reports.
- `checks/repo_lint.py`: repository lint used by `lake lint`. It checks source hygiene, public API
  boundaries, import-only aggregators, fixed-rank names in public tensor/model APIs,
  trusted-axiom quarantine, public-example spellings, DocGen/Verso math markup, and module
  docstrings for `NN/` Lean files.
- `checks/dependency_audit.py`: repository-level module/import graph audit inspired by
  Li et al., "The Network Structure of Mathlib" (arXiv:2604.24797). It reports
  broad imports, layer-boundary smells, fan-in/fan-out hubs, and Markdown/JSON
  summaries for repository hygiene. Lean comments, strings, and fenced guide examples are ignored
  so imports shown in documentation do not become graph edges.
- `checks/TorchLeanLint.lean`: Lean-side lint entry point used by Lake.

Useful commands:

```bash
scripts/checks/check.sh --ci-all
scripts/checks/example_regression.sh
scripts/checks/cuda_profile_tests.sh --both
python3 scripts/checks/repo_lint.py --fail-on-warn
python3 scripts/checks/dependency_audit.py --markdown /tmp/torchlean_dependency_audit.md --fail-on-error
```

## Data

- `datasets/download_example_data.py`: downloads the small public datasets used by the
  runnable examples, including CIFAR-10 shards, UCI household-power forecasting
  windows, and tiny text corpora.
- `datasets/download_wikitext.py`: optional WikiText preparation helper for text-model
  experiments. Requires `pyarrow`.
- `datasets/prepare_gpt2_corpus.py`: tokenizes local/Hugging Face text corpora into GPT-2 id
  streams for sequence-model experiments. Requires `transformers`.
- `datasets/plot_trainlog.py`: renders TorchLean `TrainLog` JSON curves for quick local inspection.
- `datasets/torchlean_data_convert.py`: converts `.npy`, `.npz`, `.mat`, `.pt/.pth`, CSV,
  and image-folder datasets into TorchLean's `.npy` tensor format. Optional formats require the
  corresponding Python package (`scipy`, `torch`, or `pillow`).
  For ImageNet-style diffusion examples, use:

  ```bash
  python3 scripts/datasets/torchlean_data_convert.py image-folder \
    --input /path/to/imagenet/train \
    --x-output data/real/imagenet64/imagenet64_train_X.npy \
    --y-output data/real/imagenet64/imagenet64_train_y.npy \
    --height 64 --width 64 --labels-from-dirs --limit 2000
  ```

## Documentation

- `docs/build_site.sh`: rebuilds the DocGen declaration pages, the Verso Blueprint guide and
  formalization map, dependency graph JSON, and homepage bundle.
- `docs/polish_docgen.py`: post-processes DocGen HTML with the TorchLean landing page,
  navigation links, declaration legends, dependency-link rewrites, and site styling.
- `docs/polish_verso_guide.py`: post-processes the Verso guide with responsive figures and tables,
  stable KaTeX display layout, copy buttons, theorem cards, and a check that every guide page loads
  the local math runtime.

`docs/build_site.sh` is the full site build: it rebuilds Lean modules, DocGen, the Verso guide,
the dependency graph JSON, and the Jekyll site. To refresh only the graph artifact, run
`checks/dependency_audit.py` directly.

The dependency audit is a source-architecture check. Its graph is made from Lean imports, so it is
the right tool for questions about layer boundaries and module dependencies. The Blueprint
formalization map covers selected declarations and proof dependencies. The runtime graph IR
represents neural-network computations.

## Setup

- `setup/resolve_libtorch.sh`: validates an optional LibTorch checkout used by
  `lake -R -K cuda=true build -K libtorch=true`. The normal CPU build and the default CUDA build do
  not require LibTorch.

## Sandboxed Lean Checking

- `sandbox/run_comparator.py`: optional wrapper for `leanprover/comparator`.
- `comparator/nn_ci_all.json`: Comparator config for the `NN.CI.ComparatorAll`
  marker theorem.

## Reinforcement Learning

- `rl/gymnasium_server.py`: JSON-lines Gymnasium bridge used by Lean-side RL experiments.
- `rl/export_gymnasium_rollout.py`: collects a Gymnasium rollout into TorchLean's JSON format.
- `rl/train_ppo_cartpole_sb3.py`: Stable-Baselines3 CartPole baseline for comparing against the
  TorchLean PPO path.

## Verification Producers

These scripts produce artifacts for Lean checkers. The Python/Julia side is the producer; Lean is
the checker.

- `verification/regenerate_assets.py`: command catalog for refreshing curated verification
  artifacts by group.
- `verification/lirpa/cert_runner.py`: shared LiRPA/PINN certificate runner that can also invoke
  `lake exe verify`.
- `verification/lirpa/common.py`: shared interval-arithmetic and JSON-writing helpers for the
  small LiRPA certificate producers.
- `verification/lirpa/export_mlp_cert.py`, `verification/lirpa/export_cnn_cert.py`,
  `verification/lirpa/export_attention_cert.py`, `verification/lirpa/export_gru_cert.py`,
  `verification/lirpa/export_crown_cert.py`: small deterministic LiRPA/CROWN certificate
  producers.
- `verification/abcrown/export_leaf_artifact.py`: converts raw alpha-beta-CROWN-style terminal-domain dumps
  into TorchLean's `abcrown_leaf_artifact_v0_1` JSON schema and can run the Lean checker.
- `verification/robustness/train_digits_linear.py`: trains the tiny digits linear model used by
  robustness examples.
- `verification/robustness/export_margin_cert.py`: writes logit-margin certificates consumed by
  the robustness checker.
- `verification/pinn/train_pinn_1d.py`, `verification/pinn/train_pinn_2d.py`: configurable
  PyTorch PINN trainers.
- `verification/pinn/pinn_common.py`: shared dataset/model/export utilities used by the PINN
  trainers.
- `verification/pinn/export_pinn_cert.py`, `verification/pinn/export_pinn_weights.py`,
  `verification/pinn/import_burgers_shock_mat.py`: PINN certificate, weight, and dataset producers.
- `verification/pinn/safe_expr.py`: restricted expression evaluator shared by the PINN trainers.
- `verification/splines/fit_piecewise_linear.jl`: dependency-light Julia producer for the spline
  certificate example.
- `verification/two_stage/export_van_stage1_bits.py`: Stage-1 artifact producer for the Van der Pol
  two-stage workflow.
- `verification/two_stage/cegis_van_stage2_python_baseline.py`: Python baseline for comparing
  against the Lean-checked Stage-2 workflow.

## Geometry3D Producers

- `verification/geometry3d/export_hf_depth_box3d_cert.py`: DETR/depth-pipeline certificate
  producer.
- `verification/geometry3d/export_omni3d_box3d_cert.py`: Cube R-CNN / Omni3D-style certificate
  producer.
- `verification/geometry3d/export_wilddet3d_box3d_cert.py`: WildDet3D certificate producer.
- `verification/geometry3d/render_box3d_cert_overlay.py`: visual overlay renderer for certificates.
- `verification/geometry3d/plot_box3d_bbox_diagnostic.py`: numeric bbox diagnostic plotter.
- `verification/geometry3d/test_bad_box3d_certs.py`: negative-test generator that requires Lean to
  reject mutated certificates.
- `verification/geometry3d/safe_image_io.py`: shared HTTPS/local RGB image loader.
