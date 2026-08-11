# ODE Verification Artifacts

The ODE verifier checks finite corridor certificates for differential equations. A producer may
learn a lower/upper neural bound, fit a simple model, or write a hand-curated certificate. Lean then
checks the exported JSON object: the equation, the time/domain boxes, the proposed lower and upper
functions, and the margins needed for the enclosure argument represented in the artifact.

The reusable verifier lives under `NN/Verification/ODE`. This folder contains small JSON artifacts
for examples and regression checks:

- `sample_ode_cert.json`: the default compact certificate used by the CLI.
- `sin_cert.json`: a direct trigonometric example.
- `logistic_trivial_cert.json`, `logistic_learned_cert.json`: logistic-equation certificates.
- `zero_mlp.json`, `one_mlp.json`, `zero_siren.json`: tiny neural-function fixtures used by the
  checker path.
- `logistic_lower_learned.json`, `logistic_upper_learned.json`: learned lower/upper side artifacts.

Check a bundled certificate directly with:

```bash
lake exe verify -- ode --cert=NN/Examples/Verification/ODE/sample_ode_cert.json
```

Regenerate or recheck the curated ODE group with:

```bash
python3 scripts/verification/regenerate_assets.py --group ode --run
```

The data flow is:

```text
ODE and candidate tube
  -> finite JSON certificate
  -> Lean parses the equation and artifact
  -> checker accepts only if the represented enclosure conditions hold
```

These files are small enough to keep in the public repository. Larger learned ODE/PINN weights
should be stored outside git and passed to `lake exe verify -- ode ...` explicitly.
