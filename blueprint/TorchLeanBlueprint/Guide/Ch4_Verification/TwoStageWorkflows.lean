import VersoManual

open Verso.Genre Manual

#doc (Manual) "Two-Stage Verification Workflows" =>
%%%
tag := "twostage"
%%%

A branch-and-bound verifier may spend hours splitting boxes, optimizing linear relaxations, and
running GPU kernels. Reimplementing that search inside Lean would make the trusted story simpler,
but it would also discard mature solvers and make large examples impractical. A two-stage workflow
separates the expensive search from the small object that must be checked.

The producer is allowed to be complicated:

```
trained model + input property
  -> external verifier
  -> branch-and-bound leaves and claimed lower bounds
```

The consumer should be narrow:

```
JSON artifact
  -> finite parser
  -> schema and local predicate checks
  -> accept or reject
```

This architecture is only valuable when the boundary is stated exactly. “Checked by Lean” may mean
anything from parsing a JSON file to replaying every bound computation and deriving a theorem.
TorchLean’s current α,β-CROWN leaf checker implements the former kind of boundary: it checks a
small structural leaf format and a local threshold predicate. It does not rerun α,β-CROWN.

# Begin With The Producer

The previous chapter established exactly what the leaf checker accepts, including the JSON schema,
the local threshold predicate, and negative controls. Here we begin with the external producer and
follow its output into the controller workflows.

After an external verifier has collected terminal domains, the handoff is:

```
raw terminal-domain dump
  -> TorchLean conversion helper
  -> abcrown_leaf_artifact_v0_1.json
  -> Lean structural checker
```

The producer may be large and stateful. The handoff remains small: root and leaf boxes, exported
lower bounds and thresholds, and an optional witness coordinate. The certificate chapter is the
reference for the meaning and limitations of those fields.

# Convert The Producer Output

Vanilla α,β-CROWN does not emit TorchLean’s JSON schema. The adapter
[`export_leaf_artifact.py`](https://github.com/lean-dojo/TorchLean/blob/main/scripts/verification/abcrown/export_leaf_artifact.py)
accepts common raw names such as `x_L`, `x_U`, `lower_bounds`, and `thresholds`, normalizes them,
computes a positive witness coordinate, and writes the checked format.

Run the complete bundled producer-to-checker path:

```
python3 scripts/verification/abcrown/export_leaf_artifact.py \
  --input NN/Examples/Verification/AbCrown/example_raw_leaf_dump.json \
  --out /tmp/torchlean-abcrown-artifact.json \
  --check
```

The verified output is:

```
Wrote TorchLean alpha-beta-CROWN-style leaf artifact to /tmp/torchlean-abcrown-artifact.json
[artifact] Checked 1 leaves: ok=1, bad=0
```

For integration inside a producer process, the same script exports a Python function:

```
from scripts.verification.abcrown.export_leaf_artifact import \
    write_abcrown_leaf_artifact

write_abcrown_leaf_artifact(
    root_lo=original_property_lo,
    root_hi=original_property_hi,
    leaves=terminal_verified_leaves,
    out_path="leaf_artifact.json",
)
```

The call belongs after the external verifier has collected terminal leaves. Each leaf must provide
an input box, lower-bound vector, and threshold vector. Supplying the original property box is
important. If no root is provided, the converter can infer the componentwise envelope of the
leaves, which is useful for fixtures but does not establish that the leaves cover the intended
property domain.

TorchLean does not vendor α,β-CROWN or the Two-Stage neural-controller repository. Their Python,
CUDA, solver, and model dependencies remain in separate environments. The core Lean build needs
only the exported artifact and checker.

# Carry Forward The Exact Claim

The certificate chapter demonstrates the positive check and its negative controls. For the
workflows below, carry forward its exact conclusion: acceptance establishes structural consistency
and the exported local threshold predicate, not the provenance of the lower bound or coverage of
the original root region.

# Neural Controllers

TorchLean also registers three concrete Lyapunov workflow runners. They differ in where candidate
generation and numerical checking occur:

- [`twostage-pythononly-certgen`](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/CROWN/Lyapunov/TwoStage/PipelineIPythonOnly.lean)
  invokes the external CROWN producer and writes a Lean module. The generated bounds enter through
  `CrownOracleWitness`; Lean proves the final real inequalities from that oracle witness and the
  sign conditions, but does not replay the external verifier.
- [`twostage-hybrid-van-stage2`](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/CROWN/Lyapunov/TwoStage/PipelineIIHybrid.lean)
  treats PyTorch's bit-exact float32 parameter export as an untrusted initialization, then performs
  refinement and the final IBP/CROWN box check in Lean with `IEEE32Exec`.
- [`twostage-torchlean-cegis-van`](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/CROWN/Lyapunov/TwoStage/PipelineIIIAllInLean.lean)
  performs initialization, sampled training, PGD-style candidate search, refinement, and the final
  bound check in Lean with `IEEE32Exec`; it does not require the external stage-one exporter.

Run them by their exact registry names:

```
lake exe verify -- twostage-pythononly-certgen --model model.pth \
  --region "[-1,1]x[-1,1]" --dynamics van_der_pol
lake exe verify -- twostage-hybrid-van-stage2
lake exe verify -- twostage-torchlean-cegis-van
```

“Runs in Lean” identifies the execution and checker boundary; it is not by itself a theorem that
the learned controller satisfies the analytic Lyapunov conditions on an arbitrary region. The
runtime result must still be connected to the real `LyapunovCert` theorem, and any native or
`@[implemented_by]` path retains its documented implementation boundary.

In a controller workflow, the producer may search for a policy `u_θ` and a Lyapunov candidate `V`.
The target inequalities often look like

$$`V(x)\geq 0,\qquad
\nabla V(x)\cdot f(x,u_\theta(x))
\leq-\alpha\|x\|^2.`

A trustworthy two-stage artifact must identify:

- the model and parameter hash;
- the state-space root region;
- the exact dynamics and scalar semantics;
- every partition leaf;
- replayable bounds for `V` and its Lie derivative;
- coverage and boundary conditions.

Merely exporting terminal boxes with positive numeric margins does not prove those analytic
conditions. TorchLean’s other `RealCert`-style and IR-based checkers illustrate stronger replay
patterns, but they should be evaluated command by command rather than transferred to
`abcrown-leaf` by association.

# Reproducible Evidence

A useful run record includes:

- TorchLean commit and Lean toolchain;
- external verifier repository and commit;
- Python and solver versions;
- model checkpoint hash;
- original property file and root box;
- dtype, device, and numerical flags;
- raw producer dump;
- normalized TorchLean artifact;
- exact checker command and output.

This is more than administrative detail. A lower-bound vector has no stable meaning if the model,
property, or output-margin convention changes.

The present leaf workflow is best used as a transparent integration fixture and schema boundary.
It proves that the finite data passed to TorchLean satisfies the checks listed above. Stronger
root-region claims require coverage and replayable local-bound evidence, and the guide keeps that
next step visible rather than hiding it behind the name of the external solver.
