import VersoManual

open Verso.Genre Manual

#doc (Manual) "Why Running The Model Is Not The Whole Story" =>
%%%
tag := "motivation"
%%%

Suppose a classifier returns class `3` on an image. We have learned what happened at one point. A
robustness claim asks a larger question: does class `3` remain ahead of every competitor throughout
a whole neighborhood of that image?

The difference is visible in the quantifiers. A prediction is one computation:

$$`f_\theta(x_0)=y`.

A local robustness statement concerns every point in a region:

$$`\forall x,\quad \lVert x-x_0\rVert_\infty\leq\varepsilon
  \Longrightarrow
  f_\theta(x)_y-f_\theta(x)_j>0
  \quad\text{for every }j\ne y`.

The first line is a calculation. The second is a theorem about an uncountable set. No amount of
random sampling changes that quantifier. A verifier needs a description of the region and a way to
bound the network everywhere inside it.

# A Mask With The Right Shape And The Wrong Meaning

Before trying to certify a network, we have to know which function we are certifying. Shape types
remove many accidental interpretations, but they cannot decide the meaning of every well-shaped
operation. Attention masking is a useful case because the incorrect version can look convincing in
ordinary tests.

For query $`i`, let $`A_i` be the keys that are allowed to receive attention. A hard mask means

$$`
\operatorname{attention}_{ij}
=
\begin{cases}
\dfrac{\exp(s_{ij})}
      {\sum_{k\in A_i}\exp(s_{ik})}, & j\in A_i,\\[1.2ex]
0, & j\notin A_i.
\end{cases}
`

Blocked entries never enter the denominator and receive exactly zero weight. A common numerical
shortcut instead adds a large negative constant $`-C` to a blocked logit before softmax:

$$`
\widetilde{\operatorname{attention}}_{ij}
=
\frac{\exp(s_{ij}-C)}
     {\sum_{k\in A_i}\exp(s_{ik})
       +\sum_{k\notin A_i}\exp(s_{ik}-C)}.
`

For ordinary logits and a large $`C`, this value may underflow to zero in a particular floating-point
run. Mathematically, however, it is positive for every finite $`C`. Worse, the shortcut is not safe
for arbitrary logits. If a blocked score is $`C+100`, then subtracting $`C` leaves the very large
score $`100`; the supposedly blocked key can dominate the softmax.

Both implementations have the same tensor shapes. Both run. Tests with moderate random logits may
make them look identical. The bug is in the definition of masking.

TorchLean's specification uses the hard-mask equation: blocked entries have zero numerator. Runtime
providers must implement that meaning or advertise a different operation. This is a useful example
of the library's larger design. Types settle structural questions; specifications settle semantic
ones.

# Coordinates Are Part Of The Claim

The same problem can arise before the first layer. Suppose the model consumes normalized vectors:

$$`N(x)_i=\frac{x_i-\mu_i}{\sigma_i}`.

If the raw input lies in a box

$$`x_i\in[\ell_i,u_i]`,

then, when $`\sigma_i>0`, the normalized box is

$$`N(x)_i\in
  \left[
    \frac{\ell_i-\mu_i}{\sigma_i},
    \frac{u_i-\mu_i}{\sigma_i}
  \right]`.

If the verifier starts after normalization, it must receive this transformed box. Reusing the raw
bounds asks a question about the wrong points. Again, no shape error will warn us. Coordinates are
part of the mathematics.

# Run A Bound-Propagation Example

TorchLean includes a small end-to-end executable workflow. From the repository root, run:

```
lake exe verify -- torchlean-ibp
```

The current checked-in example prints:

```
=== TorchLean → IR → IBP (small MLP) workflow ===
[TorchLean] Float32 mode: IEEE32Exec: executable IEEE-754 binary32 kernel (bit-level; includes NaN/Inf)
compiled IR nodes: 20
output box lo: [1.904000]
output box hi: [2.256000]
```

The source is
[`NN/Verification/TorchLean/IBPWorkflow.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/TorchLean/IBPWorkflow.lean).
It constructs a two-input, three-hidden-unit ReLU MLP with an explicit parameter payload. It lowers
that forward program to `NN.IR.Graph`, places an $`\ell_\infty` box of radius $`0.1` around
$`(0.5,0.8)`, and runs interval bound propagation.

The two vectors are not samples. They are the lower and upper endpoints computed for the output
box. Its midpoint is

$$`\frac{1.904+2.256}{2}=2.08`

and radius

$$`\frac{2.256-1.904}{2}=0.176`.

Check those two calculations, then inspect `x0` and `eps` in the workflow source. Before looking at
`Verification.lInfBall`, write down the two input intervals that should seed the graph.

# The Chain Behind A Certificate

A useful verification claim has a chain of named objects:

$$`
\begin{aligned}
\text{model source}
&\longrightarrow \text{initialized architecture and parameters}\\
&\longrightarrow \text{semantic graph}\\
&\longrightarrow \text{input region}\\
&\longrightarrow \text{bound or certificate}\\
&\longrightarrow \text{proved property}.
\end{aligned}
`

Each arrow carries one piece of the argument.

- Did initialization produce the parameter payload that was later analyzed?
- Did lowering preserve the model's forward computation?
- Does the region describe raw inputs or already transformed inputs?
- Did the checker interpret each graph operation with the intended scalar arithmetic?
- Does acceptance imply the property written in the theorem?

The final theorem usually has the form

$$`\operatorname{check}(g,\theta,B,c)=\mathrm{true}
  \Longrightarrow
  \operatorname{Property}(\operatorname{denote}(g,\theta),B)`.

The certificate $`c` may come from a large external search. That is often the best arrangement:
search can be clever and expensive, while checking stays small and deterministic.

# See The Available Workflows

The command

```
lake exe verify -- list
```

shows the executable verification workflows in the current checkout. The list includes IBP,
certificate checkers, PINN and geometry workflows, and VNN-COMP-style runners. We will run several
of them later and open the theorem behind each checker.

# Arithmetic Cannot Be Left Implicit

The ideal expression

$$`(a+b)+c`

is associative over real numbers. Binary floating-point addition is rounded after each operation,
so regrouping may change the result. Fused multiply-add, reduction order, subnormal handling, NaNs,
and overflow introduce further distinctions.

TorchLean therefore gives several numerical interpretations distinct names:

- exact real-valued specifications;
- configurable rounded-real arithmetic with checked positive format precision;
- `FP32`, a rounded-real model with binary32 precision and gradual-underflow parameters but no
  upper exponent bound or IEEE special values;
- executable bit-level IEEE binary32;
- host `Float`, native CUDA, and external runtime providers.

A real-valued enclosure theorem cannot be silently relabeled as a theorem about every GPU execution.
A bridge theorem or an explicit backend contract must carry the result across that boundary. Later
chapters develop these numerical layers in detail; for now, the important habit is to ask which
arithmetic appears in the statement.

Even an algebraically equivalent formula can make a runtime contract safer. Smooth-max pooling,
for example, evaluates

$$`
p+\frac{1}{\beta}
  \log\!\left(\sum_i e^{\beta(x_i-p)}\right),
\qquad
p=
\begin{cases}
\max_i x_i,&\beta>0,\\
\min_i x_i,&\beta<0.
\end{cases}
`

The shift keeps every exponential argument nonpositive. TorchLean uses the same shifted weights in
the forward and derivative paths, and executable entry points reject zero or non-finite $`\beta`.
The real identity explains the transformation; the validation and backend tests address the
executable boundary.

# Why Lean Helps

Lean lets the program, its mathematical interpretation, the checker, and the theorem refer to the
same definitions. If the graph schema changes, old parsers stop compiling. If an operation changes
meaning, proofs that used the old equation must be repaired. If a certificate omits a required
hypothesis, the soundness theorem cannot be applied.

That feedback is the practical value of formalization. An assumption that once lived in a comment
becomes an argument that Lean asks us to supply.

# Further Reading

- Szegedy et al., ["Intriguing properties of neural
  networks"](https://arxiv.org/abs/1312.6199), ICLR 2014.
- Gowal et al., ["On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust
  Models"](https://arxiv.org/abs/1810.12715), 2018.
- Zhang et al., ["Efficient Neural Network Robustness Certification with General Activation
  Functions"](https://arxiv.org/abs/1811.00866), NeurIPS 2018.
- Xu et al., ["Automatic Perturbation Analysis for Scalable Certified Robustness and
  Beyond"](https://arxiv.org/abs/2002.12920), NeurIPS 2020.
- Wang et al., ["Beta-CROWN: Efficient Bound Propagation with Per-neuron Split Constraints for
  Complete and Incomplete Neural Network Robustness Verification"](https://arxiv.org/abs/2103.06624),
  NeurIPS 2021.
- Jia and Rinard, ["Exploiting Verified Neural Networks via Floating Point Numerical
  Error"](https://arxiv.org/abs/2003.03021), 2020.
