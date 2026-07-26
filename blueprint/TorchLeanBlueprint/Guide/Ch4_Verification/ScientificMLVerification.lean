import VersoManual

open Verso.Genre Manual

#doc (Manual) "Scientific ML Verification" =>
%%%
tag := "scientific-ml-verification"
%%%

Scientific models make the verification boundary unusually visible. A trained PINN may look
accurate on a plot while violating its PDE between sample points. A numerical ODE trajectory may
look smooth while accumulated error takes it outside the claimed corridor. A spline fit may be
excellent at the knots and wrong inside one interval. TorchLean therefore treats the trained model
or fitted curve as a producer of a mathematical claim, not as the claim itself.

The three maintained paths share a producer-and-checker shape, but they check different things. The
ODE command tests interval subsolution, supersolution, ordering, and initial-value conditions for
candidate corridor networks. The bundled PINN command replays residual and derivative bounds for a
fixed in-source graph and parameter set. The spline command checks that rational polynomial pieces
interpolate their declared knots exactly; it does not yet bound the polynomial between those knots.

For classifier verification, the artifact is often an input box and logit bounds. For scientific ML,
the artifact may be a time corridor, a residual bound, a polynomial certificate, or a derivative
enclosure. In each case a producer proposes a finite object and Lean checks a smaller predicate.
Where a mathematical enclosure theorem exists, a separate soundness bridge must still show that
the executable predicate supplies its hypotheses.

# Three Commands To Try

The registered verification tools can be listed with:

```
lake exe verify -- list
```

Three entries correspond to the scientific paths in this chapter:

```
pinn-cert [<path>]    -- PINN certificate recomputation check
spline-cert [<path>]  -- piecewise-polynomial certificate checker
ode                   -- ODE enclosure verification
```

Start with the bundled PINN artifact:

```
lake exe verify -- pinn-cert
```

For each domain location, TorchLean prints enclosures for the first and second derivatives. A
portion of the output is:

```
Residual R(x) from PDE 'uxx': [-5.556681,-5.220827]
u'(x)∈[2.858092,2.969166]
u''(x)∈[-5.556681,-5.220827]
...
PINN artifact replay matched Lean's recomputed residual bounds.
```

This run demonstrates recomputation: the checker does not merely trust the residual interval stored
in JSON. It reconstructs the relevant derivative bounds and compares the artifact with the result.
It does not say that a small residual alone implies closeness to the true PDE solution; that
requires a separate stability or a posteriori error theorem for the PDE.

The spline sample is shorter:

```
lake exe verify -- spline-cert
```

and prints:

```
Piecewise polynomial certificate verified.
```

Here “verified” means that the knot coordinates are strictly increasing, each piece names the
matching adjacent knots, every coefficient array has the declared length, and Horner evaluation at
both endpoints equals the declared knot values over exact rationals. Change one coefficient and
rerun the checker; an endpoint mismatch should be reported.

Two flags expose useful neighboring checks:

```
lake exe verify -- spline-cert --ieee32
lake exe verify -- spline-cert --regen
```

`--ieee32` additionally requires every rational value to be exactly representable as finite
binary32 and replays the endpoint equalities with `IEEE32Exec`. `--regen` asks the Julia producer to
write a fresh JSON document before Lean checks it. Neither flag proves an interior range bound for
a polynomial piece.

The ODE tool has no meaningful default differential equation, so invoking it without a certificate
prints the required data:

```
lake exe verify -- ode
```

```
lake exe verify -- ode --model=direct \
  [--scalar=float|ieee32exec] --cert=<ode_enclosure.json>
lake exe verify -- ode --model=torchlean \
  --scalar=float --cert=<ode_enclosure.json>
```

The ODE expression always comes from the certificate. The `--model` choice controls how the lower
and upper corridor networks are evaluated: directly from the imported graph, or after compilation
through TorchLean. The `--scalar` choice controls the arithmetic used by the direct evaluator.
Today the TorchLean-compiled route supports only `--scalar=float`; pairing it with `ieee32exec` is
rejected rather than silently changing the requested semantics.

Certificate times and initial endpoints must be finite and correctly ordered; `minWidth` and
`slack` must be finite and nonnegative. Unknown backend names or wrong JSON field types are rejected
instead of being replaced by defaults. During checking, a NaN or infinity in any interval
comparison is a failure, not a successful unordered comparison.

Expression parsing is part of that boundary. In both ODE and PINN expressions, exponentiation
binds more tightly than unary minus, so `-u^2` means `-(u^2)`; write `(-u)^2` for the other tree.
Natural powers use the ordinary identities $`u^0=1` and $`u^1=u`. The parser consumes the whole
input and rejects unknown identifiers or trailing tokens, preventing a certificate from being
checked against a silently shortened equation.

# ODE Enclosures

An ODE enclosure certificate is a finite description of a corridor around a trajectory. The checker
does not depend on an external integrator's explanation of the run. It parses the ODE expression,
runs its interval-shaped endpoint calculation over each segment, and tests the candidate tube's
subsolution, supersolution, ordering, and initial-value conditions. That calculation is an
executable screening condition; a theorem about the true trajectory still needs the outward-bound
and real-analysis links described below.

The mathematical object is an ODE

$$`\dot x(t)=f(t,x(t)).`

A corridor certificate gives boxes $`X_i` over time intervals $`[t_i,t_{i+1}]`. The theorem shape is:

$$`x(t_i)\in X_i
\quad\text{and}\quad
f([t_i,t_{i+1}],X_i)\subseteq \dot X_i
\quad\Longrightarrow\quad
x(t)\in X_i\ \text{for}\ t\in[t_i,t_{i+1}].`

The executable side is exposed through the
[ODE checker API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/ODE/Verify.lean). The core pieces are the expression AST,
the interval evaluator, the segment certificate, and the final checker result.

The exported expression language and command entry point are small enough to inspect directly:

```
#check NN.Verification.ODE.Expr
#check NN.Verification.ODE.eval
#check NN.Verification.ODE.Verify.main
```

The theorem side is the real mathematical statement. In the
[ODE enclosure API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Verification/ODE/Enclosure.lean), a corridor theorem says, in
plain language:

> If the initial state lies in the first set, and each segment satisfies the enclosure condition for
> the ODE vector field, then the true solution remains inside the certified corridor for the
> covered time interval.

The backend bridge in
[ODE enclosure backends](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Verification/ODE/EnclosureBackends.lean) explains how
backend valued trajectories, including FP32 and `IEEE32Exec` views, can be related back to the real
statement through explicit interpretation maps.

Lean has a local real enclosure theorem, and the executable checker computes the kinds of corridor
inequalities that theorem consumes. There is not currently a theorem saying that a successful
`runCertificate` call supplies all of the real-analysis hypotheses of that enclosure theorem.
Continuity, derivative agreement, interval soundness, and any finite-to-real interpretation still
have to be connected explicitly. Broader neural ODE and integrator claims need their own enclosure
conditions and agreement evidence as well.

The trusted boundary is therefore:

```
external integrator/search -> proposed tube JSON
Lean parser/checker        -> interval side conditions for the tube
ODE theorem                -> statement about true trajectories, if theorem hypotheses match
runtime bridge             -> needed for a claim about a concrete finite-precision integrator
```

# PINN Certificates

PINN verification is more than "run the neural network." The artifact has at least four claims:

1. the imported parameters match the architecture;
2. the PDE expression is the one being checked;
3. the residual is bounded over the domain;
4. boundary or dataset constraints are respected.

TorchLean gives each piece a small object. The
[PINN architecture API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/PINN/Architecture.lean) names sequential network
records and graph construction. The
[PDE expression API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/PINN/PdeAst.lean) names the PDE language. The
[PyTorch parameter store API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/PINN/PyTorch/ParamStore.lean) names imported
parameters instead of letting a raw tensor dictionary float around unchecked. The
[residual affine API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/PINN/ResidualAffine.lean) contains the bound helpers,
including McCormick style pieces and branch and bound support.

The bundled `pinn-cert` path is intentionally smaller than that full target. Its graph is the fixed
`1 -> 16 -> 16 -> 1` tanh network `buildGraph`, and its deterministic parameters are
`seedParamsFloat`, both defined in Lean. The JSON supplies sample points, box radii, a PDE
expression, and expected value, derivative, and residual intervals. `verifyCert` recomputes those
quantities with the Float bound implementation and compares them with the artifact. It returns
success or an error; it does not construct a proof object for a uniform residual proposition.

For a Burgers-style residual, the mathematical claim has the shape:

$$`R_\theta(t,x)
=
\partial_t u_\theta(t,x)
+u_\theta(t,x)\partial_x u_\theta(t,x)
-\nu\partial_{xx}u_\theta(t,x).`

The certificate target is a uniform bound over the domain:

$$`\forall (t,x)\in\Omega,\qquad |R_\theta(t,x)|\le\varepsilon.`

Boundary or data conditions have the same form:

$$`\forall z\in\partial\Omega,\qquad |u_\theta(z)-g(z)|\le\varepsilon_b.`

The [PINN certificate API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/PINN/Certificate.lean), the
[dataset checker API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/PINN/DatasetCheck.lean), and the
[PINN command API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/PINN/CLI.lean) are the user-facing pieces for that path.

Important Lean objects:

```
#check NN.Verification.PINN.SequentialPINNArch
#check NN.Verification.PINN.buildGraph
#check NN.Verification.PINN.PdeAst.Expr
#check NN.Verification.PINN.PdeAst.eval
#check NN.Verification.PINN.ResidualAffine.crownUBoundsForward
#check NN.Verification.PINN.DatasetCheck.DatasetCheckOpts
```

PINNs are a good stress test because the model is only part of the claim. The PDE residual, the
domain, the boundary data, and the imported parameters all matter. TorchLean's design makes those
pieces explicit across its PINN tools. `pinn-cli` explores one- and two-dimensional residual boxes
with IBP or CROWN-style methods, while `pinn-dataset-check` performs pointwise interval containment
checks and can load an optional PyTorch parameter file. The dataset command is report-only by
default: it prints `ok` and `bad` counts but exits successfully even when misses are present. Use

```
lake exe verify -- pinn-dataset-check --strict
```

when a nonzero `bad` count should fail an automated run. Even strict success is still a checker
result until a soundness theorem connects the selected bound path and imported parameters to a
quantified PDE statement.

The reference point for the application is Raissi, Perdikaris, and Karniadakis,
["Physics-informed neural networks"](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
(Journal of Computational Physics 2019; arXiv preprint
[1711.10561](https://arxiv.org/abs/1711.10561)). That paper motivates the residual objective.
TorchLean's bundled replay makes a narrower claim: for its fixed graph and parameters, the
artifact's stored value, derivative, and residual intervals match the Float quantities recomputed
at the declared boxes. The dataset and interactive commands have their own inputs and checks; they
should not be folded into the meaning of `pinn-cert`.

# Piecewise Polynomial and Spline Certificates

The spline path is concentrated in
[NN.Verification.Splines.PiecewisePolyCert API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/Splines/PiecewisePolyCert.lean).
It parses `piecewise_poly_v0` JSON, evaluates polynomial pieces by Horner's rule, checks exact
rational interpolation at adjacent knots, and also has an `IEEE32Exec` exact conversion path.

A piecewise polynomial certificate names intervals $`I_i` and polynomial pieces

$$`p_i(x)=\sum_k a_{ik}x^k,\qquad x\in I_i.`

For a piece on $`I_i=[x_i,x_{i+1}]`, the checked equations are

$$`p_i(x_i)=y_i,
\qquad p_i(x_{i+1})=y_{i+1}.`

An interior range theorem would instead need a statement such as

$$`\forall x\in I_i,\qquad p_i(x)\in[\ell_i,u_i],`

together with data or a proof sufficient to check it. That stronger condition is not part of
`piecewise_poly_v0` today. This is precisely why a curve may interpolate every knot and still
behave badly between them.

The example follows the external-tool pattern:

1. another system may generate a piecewise polynomial artifact;
2. the artifact is serialized into a small explicit certificate format;
3. Lean parses and checks that format;
4. any remaining producer hypothesis is named instead of hidden.

# Artifact Boundary Examples

The same scientific artifact can support different strengths of claim depending on what it exports.

- If a PINN JSON contains only sampled residuals, Lean can check those samples; it cannot infer a
  uniform residual bound over the domain.
- If a PINN certificate contains interval or affine residual bounds over domain boxes, Lean can
  check those box obligations and state a uniform residual claim for the boxes covered by the
  certificate.
- If an ODE artifact contains a proposed trajectory but no interval enclosure condition, Lean can
  parse the trajectory but does not get an enclosure theorem.
- If a piecewise polynomial artifact contains rational coefficients in the current format, Lean can
  check exact knot interpolation. An interior range claim needs a richer schema and checker.

This is the same checked/proved/assumed distinction used for robustness certificates. The producer
may be a numerical solver; the theorem applies only to the artifact fields that Lean checked or to
producer hypotheses named in the statement.

# Scientific Artifacts In The Same Trust Story

Scientific ML often lives at the boundary between theorem proving and numerical tooling. The
working discipline is simple: export a small artifact, recompute as much of it as practical in
Lean, and attach it to a theorem only after proving that the accepted checks imply the theorem's
hypotheses. The plots remain useful evidence, but they no longer have to carry the logical meaning
of the result by themselves.

# References

- Maziar Raissi, Paris Perdikaris, and George Em Karniadakis,
  ["Physics-informed neural networks"](https://www.sciencedirect.com/science/article/pii/S0021999118307125),
  Journal of Computational Physics 2019; preprint
  [arXiv:1711.10561](https://arxiv.org/abs/1711.10561).
