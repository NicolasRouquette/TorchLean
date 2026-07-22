import VersoManual

open Verso.Genre Manual

#doc (Manual) "Proof Systems Beyond Bounds" =>
%%%
tag := "proof-systems-beyond-bounds"
%%%

Verification in TorchLean means connecting an artifact to the semantics it claims to represent.

IBP and CROWN enclose outputs. Compiler correctness preserves graph meaning. The autograd theorems
connect backward rules to derivatives. Runtime approximation accounts for finite precision. BugZoo
contracts make common ML failure modes precise. These are different proof systems, but they follow
the same discipline: name the object, state the relation, and prove or check the relation for the
supported fragment.

# Proof Obligations As Relations

The common structure is relational. Let `S` be a semantic object, `A` an executable or imported
artifact, and `R S A` the proposition that the artifact represents the semantics correctly. A
compiler proof establishes `R` for the executable graph it produces. A certificate checker parses
an untrusted artifact and returns evidence from which `R` follows. A backend contract records `R`
as an assumption when the implementation remains outside the proved fragment.

For a checker

$$`\operatorname{check}:A\to\operatorname{Except}(\mathrm{Error},W),`

the useful theorem is not that the parser terminates. It has the form

$$`\operatorname{check}(a)=\operatorname{ok}(w)
\quad\Longrightarrow\quad
R(S,\operatorname{decode}(a),w).`

This direction matters. The producer may be Python, CUDA, α,β-CROWN, or a remote solver; none of
those programs must be trusted merely because the checker accepts their output. Trust moves to the
Lean definition of `R`, the checker, and its soundness theorem. When no such theorem exists, the
artifact is evidence or an explicit boundary, not a certificate by vocabulary alone.

# IRExec Correctness: The Big Compiler Theorem

The theorem to care about first is:

```
Runtime.Autograd.Compiled.execGraphOfIR_semantics_eq
```

The declaration is in the
[IR execution correctness API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Compiled/IRExec/Correctness/SemanticEquivalence.lean).

In plain English:

> If `execGraphOfIR` successfully compiles an `NN.IR.Graph` and payload into executable compiled
> graph data, and the graph is in the named supported fragment, then evaluating the compiled graph
> on any input gives the same value table as the Lean denotational evaluator for the original IR
> graph.

The theorem connects three objects:

- `NN.IR.Graph.denoteAll`: the reference denotational semantics of the tagged op IR.
- `execGraphOfIR`: the compiler from IR to executable compiled graph data.
- `ExecGraphData.denoteAll`: the compiled runtime evaluator.

The theorem shape is: if `execGraphOfIR g payload` returns `ok exec`, and the named fragment
side conditions hold, then for every input `x`, the value table produced by
`ExecGraphData.denoteAll exec x` is the same value table produced by
`NN.IR.Graph.denoteAll g payload x`.

In theorem notation, the supported-fragment statement has the shape:

$$`\operatorname{execGraphOfIR}(G,P)=\operatorname{ok}(E)
\;\land\; \operatorname{NoMSELoss}(G)
\;\land\; \operatorname{NoRawLog}(G)
\quad\Longrightarrow\quad
\forall x,\;
\operatorname{ExecGraphData.denoteAll}(E,x)
=
\operatorname{Graph.denoteAll}(G,P,x)`

For the covered IRExec fragment, this theorem prevents "verified the wrong executable graph."

There are side conditions, and they matter. The graph must pass the structural checks, compilation
must succeed, and the current theorem has explicit fragment predicates:

- `NoMSELoss g`, because that op is outside this whole-graph semantic equivalence proof path.
- `NoRawLog g`, because raw real `log` needs a positivity precondition. The local positive-domain
  branch is present, but the whole-graph theorem needs per-node domain facts to use it. Use the
  epsilon-protected safe-log operation when unconditional execution is intended.

That precision is part of the compiler proof: supported ops get named coverage, and unsupported ops
or ops needing extra domain facts do not get folded into the theorem by vague prose.

What this theorem rules out:

- In PyTorch or XLA style compilation, we usually rely on the compiler and test for regressions.
- In TorchLean's supported IRExec fragment, under the named side conditions, the compiler's forward
  result is tied to the IR denotation by a Lean theorem.
- This directly targets the classic silent wrong code problem: the optimized/executable graph
  should not secretly compute a different mathematical program.

The proof is large because it recursively mirrors the compiler. The workhorse lemma is
`buildFrom_preserves_denotation`: as the compiler walks node ids and extends the compiled graph, the
IR value table and compiled context stay aligned. Each operator branch proves one local
preservation fact, then the recursive theorem stitches the branch into the whole graph.

The current proof is split for auditability:

- The [IRExec common API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Compiled/IRExec/Correctness/Common.lean) contains shared
  infrastructure for dynamic values, compiled contexts, and finishing a node step.
- The [semantic equivalence common API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Compiled/IRExec/Correctness/SemanticEquivalenceCommon.lean)
  contains helper lemmas used by the recursive proof.
- The [semantic equivalence op cases API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Compiled/IRExec/Correctness/SemanticEquivalenceOpCases.lean)
  contains heavy named cases such as `.linear` and `.conv2d`.
- `Correctness/Ops/*` contains smaller branches by op family: activations, constants, elementwise,
  linear algebra, normalization, pooling, permutation, random, reductions, structural ops, and unary
  ops.
- The [semantic equivalence theorem API](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/Compiled/IRExec/Correctness/SemanticEquivalence.lean)
  ties the cases together into `execGraphOfIR_semantics_eq`.

Here TorchLean is doing more than a normal ML framework can do with tests. A regression test says
"we tried these examples." The theorem says "for every input, if the supported compiler accepts this
graph and the named fragment side conditions hold, the compiled evaluator and IR denotation agree."

# A Tiny IRExec Example

Imagine an IR graph for:

$$`y=\operatorname{ReLU}(Wx+b)`

There are two ways to evaluate it:

1. Interpret the IR directly:

- input node `0` gives `x`;
- linear node `1` reads `W` and `b` from the payload and computes `Wx+b`;
- ReLU node `2` computes `max(0,node1)`.

2. Compile the IR into `ExecGraphData` and run the compiled graph:

- compiled node `1` has a forward closure for the affine map;
- compiled node `2` has a forward closure for ReLU;
- the compiled evaluator visits the same dependency order as the IR denotation.

The theorem says these two paths produce the same result for every `x`, provided the compiler
accepted the graph and the ops are in the proved fragment. At that boundary, the compiled runtime
stops being "some other implementation" and becomes a proved refinement of the IR semantics.

# Run The Graph Through Both Views

The `graphspec` example constructs a typed model, lowers it, trains it through the runtime, and
reports the object that crossed each layer:

```
lake exe torchlean graphspec
```

A seeded run includes:

```
Sequential: [2] -> [1], layers=3, params=13
  [0] Linear(2, 3): [2] -> [3] params=9
  [1] ReLU: [3] -> [3] params=0
  [2] Linear(3, 1): [3] -> [1] params=4
mean_loss(before) = 1.239197
mean_loss(after) = 0.247518
forward: GraphSpec MLP lowered to TorchLean and executed
```

The output establishes that this execution completed and that the loss decreased on this run. The
compiler theorem supplies the stronger statement: for every input, if this graph is in the proved
fragment and compilation succeeds, the compiled denotation equals the IR denotation. The training
log and compiler theorem answer different questions, and both are useful.

# Unsupported Cases Must Remain Visible

The axis-operator tutorial compares specification execution with compiled execution:

```
lake exe torchlean ir_axis_ops
```

For concatenation on the middle axis, both paths execute and print the same leading values:

```
[concat_middle_axis] spec outShape: ... [2,8,4]
[concat_middle_axis] compiled outShape: ... [2,8,4]
```

For middle-axis softmax and LayerNorm, the specification path runs but the compiled path reports:

```
compiled skipped: current IRExec backend supports fewer axis cases
than the spec semantics.
```

That message is a feature, not an inconvenience to hide. The semantic language can describe more
programs than a particular compiler theorem or runtime backend currently covers. A clean system
rejects or skips the unsupported lowering; it does not infer correctness from the fact that a
different implementation happened to return an array of the expected shape.

# How The Proof Systems Compose

Consider a claim about one training step on a Float32 backend. No single theorem should be expected
to prove the entire statement. The proof is assembled from relations:

$$`
\begin{aligned}
\text{source model}
&\equiv \text{IR denotation},\\
\text{IR denotation}
&\equiv \text{compiled forward denotation},\\
\text{graph VJP}
&= (D\,\text{forward})^\ast,\\
\text{rounded execution}
&\approx_\varepsilon \text{real execution},\\
\text{optimizer update}
&= \theta-\eta g.
\end{aligned}`

Each line has its own hypotheses and failure modes:

:::table +header
*
  * Relation
  * Typical obligation
*
  * source to IR
  * compiler covers every source constructor used
*
  * IR to executable graph
  * graph is well formed and satisfies fragment predicates
*
  * VJP to derivative
  * local backward laws and analytic domain conditions
*
  * rounded to real
  * finite values and an explicit error budget
*
  * gradient to update
  * optimizer state and update equation match the intended algorithm
:::

A theorem about the whole workflow composes these relations. A report about a workflow should say
which rows are proved, which were checked for one artifact, and which cross a named backend
boundary.

# Why The Relations Compose

Suppose lowering relates a source model `M` to an IR graph `G`, compilation relates `G` to an
executable graph `E`, and a numerical theorem bounds `E_float` against the real denotation of `E`.
The end-to-end argument is ordinary transitivity, but each intermediate term must be the same
mathematical object:

$$`\begin{aligned}
\llbracket M\rrbracket_{\mathbb R}
&=\llbracket G\rrbracket_{\mathbb R},\\
\llbracket G\rrbracket_{\mathbb R}
&=\llbracket E\rrbracket_{\mathbb R},\\
\left\|\llbracket E\rrbracket_{\mathrm{float}}(x)
 -\llbracket E\rrbracket_{\mathbb R}(x)\right\|
&\le\varepsilon(x).
\end{aligned}`

Therefore

$$`\left\|\llbracket E\rrbracket_{\mathrm{float}}(x)
-\llbracket M\rrbracket_{\mathbb R}(x)\right\|
\le\varepsilon(x).`

This apparently simple substitution is where many informal arguments fail. A changed parameter
layout, mask convention, axis order, or scalar interpretation means the middle terms are no longer
identical. Typed payloads and explicit denotations make those mismatches proof obligations instead
of silent conventions.

# Evidence Is Not Interchangeable

The same result may have several kinds of evidence:

- a runtime example shows that a path executes on one input;
- a regression test guards behavior on selected inputs;
- a checker validates a finite artifact;
- a refinement theorem relates two semantics for every input satisfying its hypotheses;
- a backend contract records an assumption about code outside Lean.

More evidence is welcome, but one kind does not silently become another. A CUDA parity test does
not prove a vendor kernel. A real-arithmetic CROWN theorem does not by itself prove a binary32
margin. A correct local VJP does not prove a complete recurrent network until the composition
theorem covers the unroll.

Autograd correctness, runtime approximation, optimizer laws, learning theory, scientific
certificates, reinforcement learning, and generative models use different relations. The important
question in every case is the same: which semantic object appears on both sides of the theorem?

# References

- [PyTorch autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)
- [JAX automatic differentiation guide](https://docs.jax.dev/en/latest/automatic-differentiation.html)
- [α,β-CROWN project](https://github.com/Verified-Intelligence/alpha-beta-CROWN)
- [auto LiRPA project](https://github.com/Verified-Intelligence/auto%5FLiRPA)
- [VNN-COMP](https://vnn-comp.github.io/)
