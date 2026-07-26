import VersoManual

open Verso.Genre Manual

#doc (Manual) "Verification Certificates" =>
%%%
tag := "certificates"
%%%

A certificate is useful when producing an answer is expensive but checking the answer can be made
small. A branch-and-bound verifier may explore thousands of subdomains, optimize relaxation
parameters, and use GPU kernels. Lean need not replay that entire search if the producer emits
enough evidence for a compact checker.

For every certificate format, ask three questions:

1. What finite object did the producer return?
2. Which conditions does Lean recompute?
3. Which theorem turns acceptance into the final semantic claim?

The third question matters most. A perfectly parsed file can still contain a number that was never
proved to bound the network.

TorchLean does not vendor the Two-Stage / α,β-CROWN repository. The core Lean build does not
require that Python environment. Generating fresh α,β-CROWN leaf artifacts requires external verifier
output plus TorchLean's conversion helper.

During a branch-and-bound verification run, an instrumented external verifier can expose terminal
subdomains. TorchLean's helper converts that terminal-domain data into a small JSON *leaf artifact*
for each represented subdomain. TorchLean can parse that JSON and validate several
properties entirely inside Lean: every leaf box lies inside the declared root input region; every
leaf marked as verified satisfies the exported local prune test
($`\exists i,\;lb_i>threshold_i` in the exported fields); and the document is internally consistent
(dimensions, array lengths, and cross-references line up).

These checks focus on the JSON artifact itself: boxes nest correctly, verified leaves satisfy the
stated prune rule, and the fields fit together. The numeric bound propagation that produced each
`lb` belongs to the external producer unless a separate recompute-and-compare certificate path is
added.

The concrete path is:

- α,β-CROWN performs branch and bound outside Lean;
- the producer exports or exposes terminal leaf domains;
- TorchLean's converter writes those domains in `abcrown_leaf_artifact_v0_1.json`;
- TorchLean parses the JSON;
- Lean checks the structural predicate for each leaf;
- the checker accepts or rejects the artifact.

Here, `verified` or `pruned` means that a represented leaf passes the producer's exported local
test. It does not yet mean that Lean has proved the neural-network property on the root box.

# Run The Bundled Checker

The sample artifact contains one two-dimensional leaf. Its box equals the declared root, its
exported lower bound is `1.0`, and its threshold is `0.0`.

```
lake exe verify -- abcrown-leaf
```

Lean reports:

```
[artifact] Checked 1 leaves: ok=1, bad=0
```

To see what was actually checked, make a temporary copy whose threshold is larger than the
exported lower bound:

```
jq '.leaves[0].threshold=[2.0] | .leaves[0].witness_margin=-1.0' \
  NN/Examples/Verification/AbCrown/sample_abcrown_leaf_artifact_v0_1.json \
  > /tmp/torchlean_bad_leaf.json

lake exe verify -- abcrown-leaf /tmp/torchlean_bad_leaf.json
```

The command exits unsuccessfully:

```
[artifact] Checked 1 leaves: ok=0, bad=1
uncaught exception: Artifact failed checks for 1 leaves
```

This is a useful failure: the JSON is valid, but the claimed witness no longer satisfies the prune
inequality. Try two other changes. Move a leaf coordinate outside the root box; then replace one
numeric field by an infinite value. Both should be rejected before the artifact can be treated as
checked evidence.

The checked predicate is small enough to write informally: a leaf is accepted when its box lies
inside the root box, its dimensions are coherent, and it has a witness index whose exported lower
bound is above the corresponding threshold.

The leaf prune test has the form:

$$`\exists i,\qquad lb_i>threshold_i.`

In Lean-facing pseudocode, the checked part is closer to:

```
def leafPruned (lb threshold : Array Float) : Bool :=
  any index i with lb[i] > threshold[i]

def leafInsideRoot (root leaf : Box) : Bool :=
  all coordinates k, root.lo[k] <= leaf.lo[k] && leaf.hi[k] <= root.hi[k]
```

Those checks are about exported numbers. They do not by themselves prove that the exported `lb`
values are lower bounds of the neural network. That stronger claim needs either recomputation in
Lean or a proof-backed certificate whose local transfer rules Lean can check.

Leaf nesting has the form:

$$`B_\ell\subseteq B_{\mathrm{root}}.`

A stronger branch certificate would also check coverage:

$$`B_{\mathrm{root}}\subseteq\bigcup_\ell B_\ell.`

For a semantic margin property, the target shape is:

$$`\forall x\in B_\ell,\qquad c^\top f(x)\ge threshold.`

The current checker does not establish that quantified statement. It accepts when the exported
boxes and witness fields are coherent and every represented leaf passes the finite comparison
$`lb_i>threshold_i`. Even if the lower-bound provenance were added, turning the leaves into a
root-region proof would still require separately checked coverage. In this fragment, the
certificate is structural.

# Three Levels Of Checking

The `v0.1` format intentionally stops at structural checking. Lean checks that every represented
leaf lies inside the root box, that dimensions and arrays agree, that numeric fields are finite,
and that every leaf's witness satisfies $`\exists i,\;lb_i>threshold_i`.

The current artifact checks $`lb_i>threshold_i` for an exported lower bound. A stronger artifact
would also check that $`lb_i` is a sound lower bound for the graph on the leaf.

There are three progressively stronger designs:

- *Structural checking:* the artifact is self-consistent and each exported witness passes its
  stated arithmetic test. This is what `abcrown-leaf` provides.
- *Recompute and compare:* the artifact contains a network and enough node data for Lean to
  reproduce the bound calculation. TorchLean's node-certificate checkers recompute the complete
  trace with `IEEE32Exec`: interval entries must contain the authoritative trace, while affine
  replay entries must agree exactly at the binary32 level.
- *Proof-backed soundness:* checker acceptance supplies the exact hypotheses of a theorem that
  encloses the graph semantics. This requires a proved local transfer for every supported
  operator, plus a compiler correspondence and any required floating-point bridge.

The levels can share one producer workflow, but they support different claims.

# Lean Entry Points

The leaf checker is intentionally separate from CROWN node checkers:

```
#check NN.Verification.Cert.AbCrownLeafCert.checkAbCrownLeafArtifact
#check NN.Verification.IBPCert.check
#check NN.Verification.IBPNodeCert.checkIBPNodeCertificate
#check NN.Verification.CROWNNodeCert.checkCROWNNodeCertificate
#check NN.Verification.CROWNNodeCertAlphaBeta.AlphaBetaCROWNNodeCertificate
#check NN.Verification.CROWNNodeCertAlphaBeta.checkAlphaBetaCROWNNodeCertificate
```

Use the first when the artifact is a branch-and-bound leaf summary. `IBPCert.check` checks a compact
Float output-bound artifact against a supplied graph and parameter store. `IBPNodeCert` instead
replays per-node interval data with `IEEE32Exec`. `CROWNNodeCert` adds affine CROWN data, and the
alpha-beta variant adds the corresponding relaxation parameters. These formats are related, but
they are not interchangeable transcripts.

This split is important for citations:

- `abcrown-leaf` checks a structural leaf artifact.
- `checkIBPNodeCertificate` recomputes the authoritative binary32 interval trace and requires the
  imported node boxes to contain it.
- `checkCROWNNodeCertificate` checks the additional affine transcript for the supported CROWN
  fragment.
- `checkAlphaBetaCROWNNodeCertificate` checks per-node α,β-CROWN transfer data by recomputation and
  exact binary32 transcript comparison; its interval side data may widen the recomputed boxes but
  may never shrink them.
- graph soundness theorems apply only when the certificate format and graph fragment supply the
  hypotheses those theorems demand.

# File Format: `abcrown_leaf_artifact_v0_1.json`

Top-level object:

```
{
  "format": "abcrown_leaf_artifact_v0_1",
  "input_dim": 2,
  "root": { "lo": [-4.8, -10.8], "hi": [4.8, 10.8] },
  "leaves": [
    {
      "lo": [...],
      "hi": [...],
      "lb": [...],
      "threshold": [...],
      "witness_idx": 0,
      "witness_margin": 0.123
    }
  ]
}
```

Semantics:

- `root` describes the input box being verified.
- For real verification runs, pass the original input-property box as `root`. If the raw dump does
  not contain a root, the exporter can infer the componentwise leaf envelope as a structural
  fallback, but that fallback is only the envelope of the represented leaves.
- Each `leaf` is a sub-box of `root`.
- The root and every leaf must contain finite coordinates ordered coordinatewise
  ($`lo_i\le hi_i`), and the leaf array must be nonempty.
- `lb` and `threshold` are the lower bounds and thresholds reported by the external producer for
  that leaf at the moment it was pruned or verified.
- A leaf is considered "verified" iff $`\exists i,\;lb_i>threshold_i`.
  (This matches how `complete_verifier/input_split/branching_domains.py` filters out verified domains.)
- `witness_idx` and `witness_margin` are a convenience witness for the check above:
  $`witness\_margin=lb_{witness\_idx}-threshold_{witness\_idx}`.
  When `witness_idx` is present, the checker validates that exact coordinate rather than searching
  for a different witness. When `witness_margin` is present, it must accompany the index and agree
  with the recomputed margin up to the schema tolerance.

The schema deliberately does not contain a neural-network graph, α slopes, β phases, or per-node
affine forms. It is therefore a *leaf artifact*, not a full proof certificate.
The artifact records enough to check the terminal-domain bookkeeping exported by the producer; it
does not replay the producer's bound propagation.

# From Checker To Producer

This chapter owns the accepted format and checker claim. The next chapter begins with the external
producer and shows how `scripts/verification/abcrown/export_leaf_artifact.py` converts terminal
domains into this schema, both as a command and as a function called inside a producer process.
Keeping those directions there avoids mixing artifact semantics with external-tool setup.

# How To Check In Lean

Use the unified `verify` CLI tool `abcrown-leaf` to check the converted JSON artifact against
TorchLean's structural leaf predicate.

Example:

```
lake exe verify -- abcrown-leaf
```

With no path, the command uses the bundled sample. With a path, it checks that artifact instead.
Run `lake exe verify -- list` to see the other registered certificate and workflow checkers,
including LiRPA, PINN, spline, logit-margin, and TorchLean-to-IR robustness paths.

The leaf command is intentionally modest: it answers whether the exported leaf document satisfies
the `v0.1` structural contract. The neural-network verification chapter gives the theorem chain
needed for a semantic robustness result, while the two-stage chapter shows where an external
producer enters that chain.

# References

- [α,β-CROWN project](https://github.com/Verified-Intelligence/alpha-beta-CROWN)
- [β-CROWN / α,β-CROWN paper](https://arxiv.org/abs/2103.06624), covering branch and bound with
  optimized bound propagation.
- [LiRPA on general computational graphs](https://arxiv.org/abs/2002.12920), for automatic
  perturbation analysis.
- [Branch-and-bound for neural network verification](https://arxiv.org/abs/1907.10615), as one
  representative background entry.
