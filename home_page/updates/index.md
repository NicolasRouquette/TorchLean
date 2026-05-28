---
title: Updates
usemathjax: true
---

# TorchLean Updates

This is the release log for public releases, recent fixes, validation notes,
and user-facing issues.

For correctness assumptions and trusted boundaries, use
[Trust Boundaries](https://github.com/lean-dojo/TorchLean/blob/main/TRUST_BOUNDARIES.md).
For source-level provenance and release hygiene, use
[Third-Party Notices](https://github.com/lean-dojo/TorchLean/blob/main/THIRD_PARTY_NOTICES.md).

## Index

- [May 2026: Lean 4.30 and Runtime API Update](#may-2026-lean-430-and-runtime-api-update)
- [May 2026: CUDA Training Stability Update](#may-2026-cuda-training-stability-update)
- [May 2026: Quickstart Data Note](#may-2026-quickstart-data-note)
- [May 2026: TorchLean Released](#may-2026-torchlean-released)

## May 2026: Lean 4.30 and Runtime API Update

TorchLean now builds with `leanprover/lean4:v4.30.0`. The Lake manifests, the
Verso guide package, and the website build metadata have been updated together
so the README, guide, and generated site agree about the toolchain.

This update also documents the runtime pieces that were added while tightening
long CUDA runs:

- runtime-side Float initializers for large CPU/CUDA modules;
- shape-indexed initializer plans, so each parameter receives exactly one
  initializer;
- step-indexed training streams for batches produced by a rule, simulator,
  replay buffer, or file-backed window source;
- integer-token embedding and row-wise cross-entropy helpers for GPT-style
  language-model training;
- CUDA allocator diagnostics and explicit workspace-buffer ownership in the eager
  runtime.

The guide now has dedicated sections for step streams, runtime initialization,
CUDA memory ownership, and integer-token GPT training.

## May 2026: CUDA Training Stability Update

Longer CUDA training runs exposed a practical bug in the eager runtime. A
GPT-style example could train normally for thousands of updates and then stop
with a CUDA allocation failure. The surprising part was that the same kind of
problem could show up even on small examples if enough per-step CUDA objects
were kept alive.

The issue was not that the model suddenly needed more parameters. It was that
some intermediate values created during training -- tape entries, gradient buffers,
and kernel workspace buffers -- could stay attached to the run longer than they
needed to. Over many optimizer updates, that turns into allocator pressure. The
fix was to make the training loop and CUDA eager backend more explicit about
which values are returned to the caller and which intermediate buffers can be
released after the step finishes.

The CUDA training path used by the public model examples now does three things:

- loader-based model commands now treat `--steps` as optimizer updates;
- CUDA eager/autograd releases ephemeral tape, gradient, and workspace buffers
  after the values needed by the caller have been extracted;
- longer CUDA example runs report allocator telemetry by default, with
  `--cuda-mem-watch N` available when a user wants an exact sampling cadence.

The allocator report is a runtime diagnostic. It prints live and peak allocator
state while a run is still in progress, and it warns if the observed free-memory
trend would run out before the requested training horizon. If a long run is
going wrong, the terminal should show that early instead of leaving the user to
discover it at the end. It does not change the Lean-side trust-boundary story;
it makes the native runtime behavior easier to inspect.

Validation for this update included:

- `lake build`
- `lake build -K cuda=true`
- `lake exe torchlean mlp --cpu --steps 10 --log false`
- `lake exe torchlean mlp --steps 10 --dtype float --backend eager --log false`
- `lake exe -K cuda=true torchlean mlp --cuda --fast-kernels --steps 1000 --log false`
- `lake exe -K cuda=true torchlean cnn --cuda --steps 1000 --log false`
- `lake exe -K cuda=true torchlean gpt2 --cuda --fast-kernels --steps 1200 --generate 0 --log false`
- `lake exe -K cuda=true torchlean fno1d_burgers --cuda --fast-kernels --steps 50 --log false`

The validation commands above checked two things: representative losses or MSE
values went down, and the CUDA allocator stayed bounded on the exercised runs.
CUDA execution remains an implementation path; the mathematical trust boundary
is still documented separately.

## May 2026: Quickstart Data Note

Some examples use real public datasets and do not download them during
`lake build`. If a model reports a missing dataset, run the downloader it
prints. For the README MLP quickstart:

```bash
python3 scripts/datasets/download_example_data.py --auto-mpg
```

For the text and vision examples:

```bash
python3 scripts/datasets/download_example_data.py --tiny-shakespeare --cifar10
```

Keeping data downloads explicit makes ordinary builds deterministic and avoids
silently committing large external data into the repository.

## May 2026: TorchLean Released

TorchLean is public as a Lean 4 framework for writing, running, inspecting, and
verifying neural-network programs. The initial release brings together the parts
of the project that are meant to be used as one system:

- typed tensors, layers, model APIs, loaders, training loops, and examples;
- a shared graph IR for execution, inspection, verification, and import/export;
- finite-precision semantics, including executable IEEE-style Float32 models and
  explicit runtime agreement boundaries;
- autograd and optimizer support for runnable examples;
- verification examples including IBP/CROWN-style bounds, certificate checkers,
  VNN-COMP-style bundles, Bug Zoo contracts, and 3D geometry certificates;
- optional CUDA/native runtime paths with documented trust boundaries;
- generated API docs, a guide, examples pages, third-party notices, and an AI
  assistance disclosure.

Start with the README quickstart, then use the Guide and Examples pages for the
longer walkthroughs.
