# NN/CI

This directory holds proof and check targets for CI rather than the everyday development loop.

Some proofs take long enough to elaborate that building them on every local edit would slow people
down. The umbrella modules here let CI build those checks explicitly, or on a schedule, while normal
development stays focused on the files being changed.

See:

* `NN/Runtime/Autograd/IRExec/Correctness.lean` (runtime lowering correctness,
  including the semantic equivalence theorem)
* `NN/CI/All.lean` (CI umbrella for broad compile checks)

If you run one of these locally and it appears to pause, Lean is often elaborating one large module
without intermediate progress output.

## CI Targets

CI targets should answer questions that are broader than a local executable regression check:

- does the curated public API still build after a refactor?
- do slow proof modules still elaborate on the pinned Lean toolchain?
- do generated or bundled verification artifacts still parse under the current checker code?
- do CUDA and non-CUDA builds still expose the same Lean names where the public API expects them?
- do import umbrellas still expose every intended public feature?

This directory is not a second documentation tree and not a dumping ground for examples. It should
contain import targets whose job is to make CI exercise a meaningful slice of the repository.

## Suggested Local Checks

For ordinary code changes that stay inside Lean definitions, examples, or docs:

```bash
lake build
lake exe verify -- all
lake exe nn_tests_suite
```

For proof-heavy or public API changes:

```bash
lake build NN.CI.All
lake build NN.API NN.Verification NN.Proofs
lake build NN.Examples.Zoo NN.Examples.BugZoo.All
```

For CUDA changes:

```bash
lake -R -K cuda=true build NN.CI.All
lake -R -K cuda=true exe nn_tests_suite
scripts/checks/cuda_sanitize_tests.sh --all-tools
```

The CUDA sanitizer run is expensive, but it is the right evidence for memory and synchronization
hazards at the native boundary. A Lean proof about the spec does not replace that native check.

For public command or website changes:

```bash
scripts/checks/example_regression.sh --skip-help
cd home_page
bundle _2.3.14_ exec jekyll build --config _config.yml,_config_dev.yml
```

For docs that mention verification tools, compare the command names with:

```bash
lake exe verify --help
lake exe torchlean --help
```

CI should keep these evidence types distinct. A theorem target proves a mathematical statement. A
verifier command checks a concrete artifact against its schema and predicate. A runtime regression
exercises the code path users run. A sanitizer run checks native CUDA behavior around the Lean FFI
boundary. A site build proves the public pages can be regenerated from source.
