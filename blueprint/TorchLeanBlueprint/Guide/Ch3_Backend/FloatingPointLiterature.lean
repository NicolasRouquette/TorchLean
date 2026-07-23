import VersoManual

open Verso.Genre Manual

#doc (Manual) "Tracking Numerical Error Through A Network" =>
%%%
tag := "floating-point-literature"
%%%

The previous chapter explained one rounded operation. A neural network contains thousands or
billions of them. The useful question is no longer “how large can one rounding error be?” but:

> If the runtime starts near the ideal inputs, how far can its forward value, gradients, and next
> parameter state move from the corresponding real-valued computation?

TorchLean answers this compositionally for its proof-bearing operator fragment. Each covered
operation contributes a local error rule. A graph theorem combines those rules in forward order,
a reverse-mode theorem combines the covered VJP rules in backward order, and an optimizer contract
carries the resulting gradient error into the next training state.

# Two Executions Of The Same MLP

Start with the checked-in example:

```
lake exe torchlean float32_modes
```

It evaluates

$$`y=W_2\operatorname{ReLU}(W_1x+b_1)+b_2`

twice. The first run uses Lean's host `Float`; the second uses `IEEE32Exec`. Both runs then compute
the parameter and input VJPs. A shortened output is:

```
== Float (runtime) ==
y = [2.080000]
hiddenBiasGrad = [0.700000, 0.800000, 0.900000]
inputGrad = [0.760000, 1.000000]

== Float32 (IEEE32Exec) ==
y = [2.080000]
hiddenBiasGrad = [0.700000, 0.800000, 0.900000]
inputGrad = [0.760000, 1.000000]

max_abs_diff(Float vs IEEE32Exec) =
  0.0000000762939453835542735760100185871124267578125
```

The values look identical at six decimal places. The final line shows that they are not. This is a
good experiment, but we want more than the distance observed at one input. We want a bound derived
from the operations in the model.

# The Approximation Relation

Let `x` be an ideal real tensor and `x̂` its rounded counterpart. TorchLean writes the basic
coordinatewise relation as

$$`\operatorname{approxT}(x,\widehat x,\varepsilon)
  \quad\Longleftrightarrow\quad
  \forall i,\;
  |\operatorname{toSpec}(\widehat x_i)-x_i|\leq\varepsilon`.

The scalar map `toSpec` decodes the runtime scalar into the real-valued specification. For `NF`, it
is the stored real value. For another runtime type, the map can be different.

The bound is intentionally separate from the tensor. The same runtime value may be known to
approximate several ideal values with different errors, and graph propagation should update the
bound without rebuilding the tensor.

# A Linear Layer By Hand

For one output coordinate, an exact affine layer computes

$$`y_j=\sum_{k=0}^{n-1}W_{jk}x_k+b_j`.

Suppose the runtime has approximations `Ŵ`, `x̂`, and `b̂` with coordinatewise errors
`ε_W`, `ε_x`, and `ε_b`. Before accounting for arithmetic rounding, one product satisfies

$$`
|\widehat W_{jk}\widehat x_k-W_{jk}x_k|
\leq
|W_{jk}|\,\varepsilon_x
+|x_k|\,\varepsilon_W
+\varepsilon_W\varepsilon_x.
`

This identity comes from adding and subtracting `W_{jk}x̂_k`:

$$`
\widehat W\widehat x-Wx
=W(\widehat x-x)+x(\widehat W-W)
  +(\widehat W-W)(\widehat x-x).
`

Now add the local multiplication error `ρ_mul` and the rounding introduced by each accumulation.
For a fixed-left dot product, a conservative recurrence is

$$`
\begin{aligned}
E_0 &= 0,\\
P_k &=
  |W_{jk}|\,\varepsilon_x
  +|x_k|\,\varepsilon_W
  +\varepsilon_W\varepsilon_x
  +\rho_{\rm mul}(W_{jk},x_k),\\
E_{k+1} &= E_k+P_k+\rho_{\rm add}(s_k,p_k).
\end{aligned}
`

Finally add `ε_b` and the rounding of the bias addition. This explains why a linear-layer theorem
needs more than the half-ULP bound from the previous chapter. It also needs the chosen reduction
order and magnitude information for the operands.

TorchLean packages this calculation in the `NF` runtime-approximation backend. Matrix
multiplication, convolution, reductions, and scalar arithmetic each provide an error transformer
and a theorem proving that transformer valid.

# ReLU Does Not Amplify Existing Error

ReLU is easier because it is 1-Lipschitz:

$$`
|\operatorname{ReLU}(u)-\operatorname{ReLU}(v)|
\leq |u-v|.
`

If the runtime ReLU itself is exact for the declared scalar semantics, an incoming bound `ε`
remains `ε`. The interesting case is when `u` and `v` lie on opposite sides of zero. The derivative
changes discontinuously there, but the forward Lipschitz bound still holds.

This difference between forward and backward sensitivity matters. A small perturbation can leave
the ReLU output close while changing which VJP branch is selected. Backward approximation
therefore carries branch hypotheses or a bound that covers both possibilities rather than blindly
reusing the forward proof.

# Softmax And Normalization Need Range Information

For a nonlinear operation such as softmax, a global absolute-error rule is usually too weak.
TorchLean's stable softmax first subtracts the row maximum:

$$`
\operatorname{softmax}(z)_i
=
\frac{\exp(z_i-\max_j z_j)}
     {\sum_k\exp(z_k-\max_j z_j)}.
`

Now every exponent input is nonpositive, the denominator is at least one, and the output lies in
`[0,1]`. Those range facts control the local Lipschitz and rounding terms.

Layer normalization has a similar chain:

$$`
x
\longmapsto \mu
\longmapsto x-\mu
\longmapsto (x-\mu)^2
\longmapsto \sigma^2+\epsilon
\longmapsto \sqrt{\sigma^2+\epsilon}
\longmapsto
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}.
`

The positive stabilization constant gives the square root and division a domain margin. TorchLean's
local rule records that margin explicitly, then composes the mean,
subtraction, square, reduction, square root, and division bounds.

# Composition Over A Graph

A proof-bearing `RevGraph` stores, for every node:

- the exact forward operation;
- the rounded forward operation;
- an error transformer for the forward result;
- the exact and rounded VJPs;
- an error transformer for the VJP;
- proofs for both transformers.

`RevGraph.eval_approx` follows the graph in topological order. If the input context satisfies its
declared bounds, the runtime output context satisfies the bounds computed by
`RevGraph.evalBounds`.

The executable graph interpreter uses `GraphData`. The theorem

```
Proofs.RuntimeApprox.NFBackend.eval_approx_graphData
```

connects that interpreter to the same forward result. A supported graph may originate from an MLP,
CNN, transformer, or neural operator; composition depends on its operations and shapes, not its
model family name.

# Backward Error Follows The Tape In Reverse

Reverse mode starts from a seed cotangent and applies local VJPs from outputs back to inputs and
parameters. For a composition `h(x)=g(f(x))`,

$$`
\bar x
=J_f(x)^\mathsf{T}
  J_g(f(x))^\mathsf{T}\bar h.
`

The rounded pass perturbs the saved forward values, the seed, each local VJP, and every gradient
accumulation. `RevGraph.backpropBounds` mirrors the runtime traversal and computes the resulting
error context. The theorem

```
Proofs.RuntimeApprox.NFBackend.backprop_approx_graphData
```

states that `GraphData.backpropCtx` stays inside that context whenever the inputs and seed satisfy
their initial approximation relations.

This is why TorchLean keeps saved values in the numerical model. A backward rule for multiplication
uses the opposite operand; a normalization VJP uses saved statistics; attention backward uses
probabilities and mask semantics from the forward pass. Bounding only the final forward output
would throw away the information needed to analyze those gradients.

# The Optimizer Is Part Of The Numerical Story

Training does not stop at a gradient. SGD computes

$$`\theta^+=\theta-\eta g`.

If `θ̂`, `η̂`, and `ĝ` approximate their ideal values, the next parameter bound combines the old
parameter error, learning-rate error, gradient error, and local multiplication/subtraction
rounding.

Momentum adds a state recurrence. AdamW adds first and second moments, bias correction, square root,
division, and weight decay. Rather than hard-code each optimizer into the graph theorem, TorchLean
uses `NumericalStepContract`. A contract supplies:

- ideal and rounded optimizer states;
- an approximation relation for the state;
- ideal and rounded update functions;
- an error transformer for the new state and parameters;
- a theorem that the transformer is sound.

The generic theorem

```
Proofs.RuntimeApprox.NFBackend.backprop_optimizer_update_approx_graphData
```

takes one parameter gradient from reverse mode and passes it through any optimizer satisfying that
interface. SGD, momentum SGD, and AdamW reuse the same graph theorem.

# Runtime Optimizers And Proved Optimizer Contracts

The executable training layer is broader than those three numerical contracts.
`Runtime.Autograd.Train.Optim` implements parameter groups, learning-rate schedulers, shape-checked
optimizer state, and updates for SGD, momentum/Nesterov, AdaGrad, RMSProp, Adam, AdamW, and
Adadelta. `OptimStateDict` preserves the global step, per-parameter Adam steps, group configuration,
and state buffers for save/restore.

This distinction matters for conditional or sparse-gradient models. The global optimizer-call
counter advances on every step, while an Adam-family parameter advances its bias-correction counter
only when that parameter receives a gradient. Reloaded state and gradients are checked against the
current parameter shape before an update.

The proved NF numerical contracts currently cover plain SGD, momentum SGD, and AdamW; AdamW also
carries the positivity data needed for its square-root denominator. Runtime support for the other
optimizers is implemented and tested, but it should not be described as inheriting those numerical
refinement theorems automatically.

# Run A Graph Certificate

The executable companion to these approximation theorems works over the canonical IR:

```
lake exe torchlean numerical_certificate
```

The example constructs a two-layer MLP from ordinary IR operations:

```
input [1,2]
  -> matmul [2,3]
  -> add bias [1,3]
  -> ReLU
  -> matmul [3,1]
  -> add bias [1,1]
```

It generates outward-rounded binary32 ranges for every node, binds them to the selected backend
profile, and replays a concrete `IEEE32Exec` execution. The final report includes:

```
  ok  two-layer MLP certificate
  ok  two-layer MLP IEEE replay
```

The same executable also performs negative experiments. It corrupts a range, duplicates a
registry contract, changes the registry identity, violates a square-root domain, and asks a CUDA
capsule with an implementation-dependent reduction order to satisfy a fixed-left certificate. Each
case is rejected.

Open
[`GraphNumericalCertificate.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/DeepDives/Floats/GraphNumericalCertificate.lean)
and find `mlpGraph`, `mlpSources`, `mlpPayload`, `mlpCertificate`, and `mlpReplay`. Change one weight
source interval so that it no longer contains the payload value, then rerun the command. Replay
will identify the node whose value escaped the claimed enclosure.

# What We Can Display During Training

`trainingStepTrace` turns the computed bounds into a UI-friendly record:

```
structure TrainingStepTrace where
  optimizer : String
  parameterIndex : Nat
  forwardBounds : List ℝ
  backwardBounds : List ℝ
  gradientBound : ℝ
  parameterBound : ℝ
  optimizerStateBounds : List (String × ℝ)
  stepData : List (String × ℝ)
```

The trace is architecture-independent. A frontend can attach names such as
`transformer.blocks.3.attention.q_proj.weight` after lowering, but propagation itself only needs the
typed graph and parameter index. This is the basis for an InfoView or training dashboard that shows
how numerical uncertainty changes at each step.

# The Main Lesson

A half-ULP theorem is local. A model-level bound comes from composing many local theorems in the
order used by the program. Forward propagation, reverse mode, and optimizer updates each need their
own recurrence, but they share the same approximation relation and scalar rounding theory.

This is where `NeuralFloat` becomes part of TorchLean rather than a floating-point library sitting
beside it. Its generic rounding facts feed operator contracts; operator contracts feed graph
theorems; graph theorems feed a training-step bound.

# References

- Nicholas J. Higham,
  [*Accuracy and Stability of Numerical Algorithms*](https://doi.org/10.1137/1.9780898718027),
  second edition, for forward error, backward error, and the standard `γ_n` style of accumulated
  rounding analysis.
- Jean-Michel Muller et al.,
  [*Handbook of Floating-Point Arithmetic*](https://doi.org/10.1007/978-3-319-76526-6),
  second edition, for ULPs, exactness, FMA, and reduction behavior.
- Sylvie Boldo and Guillaume Melquiond,
  [“Flocq: A Unified Library for Proving Floating-Point Algorithms in
  Coq”](https://doi.org/10.1109/ARITH.2011.40), IEEE ARITH 2011.
- David Goldberg,
  [“What Every Computer Scientist Should Know About Floating-Point
  Arithmetic”](https://doi.org/10.1145/103162.103163), *ACM Computing Surveys* 23(1), 1991.
