import VersoManual

open Verso.Genre Manual

#doc (Manual) "Probability and Local Gradient Proofs" =>
%%%
tag := "probability-and-gradients"
%%%

Large-model proofs are rarely completed in one piece. A diffusion theorem needs a reusable statement
about affine images of Gaussian noise. An autograd theorem needs the derivative of each primitive.
A linear-layer proof needs to agree on whether weights are stored by input or output coordinate.
TorchLean keeps these local facts small enough to use independently.

This chapter follows two such pieces:

1. the forward noising kernel used by diffusion models;
2. scalar activation derivatives and linear-layer backward specifications.

They are related by their role in larger proofs, not because probability and differentiation are
the same subsystem.

# A Diffusion Step Is A Markov Kernel

Let `E` be a finite-dimensional real inner-product space and let `Z` have the standard Gaussian law
on `E`. Given scalars `a` and `b` and a clean state `x`, the forward noising step is

$$`X'=a x+bZ.`

In a DDPM schedule one usually takes

$$`a=\sqrt{\bar\alpha_t},
\qquad
b=\sqrt{1-\bar\alpha_t},`

so that

$$`x_t=\sqrt{\bar\alpha_t}\,x_0
      +\sqrt{1-\bar\alpha_t}\,\epsilon.`

The schedule is not built into the probability theorem. The theorem layer accepts arbitrary `a`
and `b`; a diffusion model chooses them elsewhere.

There is a closely related definition in the generative-model development:
`NN.MLTheory.Generative.Diffusion.forwardGaussian`. Both definitions scale a standard Gaussian and
translate it by a scaled clean state, but they live at different levels. `forwardNoising` works
over an arbitrary finite-dimensional real inner-product
space and also supplies a Markov-kernel wrapper; `forwardGaussian` is specialized to Euclidean
space and records the Gaussian-closure theorem used by the diffusion chapter. TorchLean does not
currently contain a bridge theorem identifying the two measures, so downstream proofs should use
the declaration their hypotheses actually mention.

The definition
[`forwardNoising`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Probability/DiffusionForward.lean)
is a measure obtained by two pushforwards:

```
def forwardNoising (a b : ℝ) (x : E) : Measure E :=
  ((stdGaussian E).map
      (b • (ContinuousLinearMap.id ℝ E))).map
    (fun y => a • x + y)
```

The staged form is useful to typeclass inference: first apply the continuous linear scaling, then
translate. The theorem `forwardNoising_eq_map` proves that this is the cleaner direct pushforward

$$`\operatorname{map}
  \bigl(z\mapsto ax+bz\bigr)
  \bigl(\mathcal N(0,I)\bigr).`

Because affine images of finite-dimensional Gaussian measures are Gaussian, Mathlib supplies the
`IsGaussian` instance. The file also proves that the measure has mass one:

```
@[simp] lemma forwardNoising_univ (a b : ℝ) (x : E) :
  forwardNoising (E := E) a b x Set.univ = 1
```

A diffusion process needs composable transitions in addition to a measure for each starting point.
`forwardKernel a b : Kernel E E` packages the operation as a Markov kernel. The bridge theorem

```
lemma forwardKernel_apply (a b : ℝ) (x : E) :
  forwardKernel (E := E) a b x =
    forwardNoising (E := E) a b x
```

says that the kernel construction and the direct measure construction denote the same transition.
`isGaussian_forwardKernel` then transfers the Gaussian fact to each kernel application.

# Reproduce The Mass-One Proof

The following is a complete scratch-file proof, not pseudocode:

```
import NN.Proofs.Probability.DiffusionForward

open MeasureTheory ProbabilityTheory
open NN.Proofs.Probability

noncomputable section

variable {E : Type*}
  [NormedAddCommGroup E]
  [InnerProductSpace ℝ E]
  [FiniteDimensional ℝ E]
  [MeasurableSpace E]
  [BorelSpace E]

example (a b : ℝ) (x : E) :
    forwardKernel (E := E) a b x Set.univ = 1 := by
  rw [forwardKernel_apply]
  exact forwardNoising_univ a b x
```

Running `lake env lean Scratch.lean` succeeds silently. The proof has only two moves: rewrite a
kernel application as its noising measure, then use the probability-mass theorem.

Try deleting `[BorelSpace E]`. Lean can no longer establish the measurability facts needed by the
kernel and reports missing measurable-space instances. Try deleting `[FiniteDimensional ℝ E]` and
the standard finite-dimensional Gaussian construction is no longer available. These failures show
which mathematical hypotheses carry the result.

This file does not sample noise, execute a diffusion network, or prove a reverse-time sampler.
Those are different objects. It proves the measure-theoretic forward transition that such
developments may cite.

# The Linear Layer’s Local Reverse Rule

TorchLean stores a linear layer’s weight tensor with shape

$$`\texttt{[outDim, inDim]},`

matching PyTorch’s `torch.nn.functional.linear` convention. For one input vector,

$$`y_i=\sum_j W_{ij}x_j+b_i.`

If $`\delta_i` is the cotangent arriving from the rest of the computation, elementary differentiation
gives

$$`\frac{\partial L}{\partial x_j}
  =\sum_i W_{ij}\delta_i,\qquad
\frac{\partial L}{\partial W_{ij}}
  =\delta_i x_j,\qquad
\frac{\partial L}{\partial b_i}
  =\delta_i.`

In matrix notation:

$$`\bar x=W^\top\bar y,\qquad
\bar W=\bar y\,x^\top,\qquad
\bar b=\bar y.`

The file
[`NN.Proofs.Gradients.Linear`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Gradients/Linear.lean)
records these equations using TorchLean tensors:

```
theorem linear_weight_gradient_correct
    (x : Tensor ℝ (.dim inDim .scalar))
    (δ : Tensor ℝ (.dim outDim .scalar)) :
  linearWeightsDerivSpec x δ = outerProductSpec δ x

theorem linear_input_gradient_correct
    (layer : LinearSpec ℝ inDim outDim)
    (δ : Tensor ℝ (.dim outDim .scalar)) :
  linearInputDerivSpec layer.weights δ =
    vecMatMulSpec δ layer.weights

theorem linear_bias_gradient_correct
    (x : Tensor ℝ (.dim inDim .scalar))
    (δ : Tensor ℝ (.dim outDim .scalar)) :
  linearBiasDerivSpec (Inhabited.default) δ x = δ
```

For example, with

$$`W=\begin{pmatrix}1&2\\-1&3\end{pmatrix},
\quad x=\begin{pmatrix}4\\5\end{pmatrix},
\quad\delta=\begin{pmatrix}2\\-1\end{pmatrix},`

the local reverse values are

$$`\bar x=W^\top\delta
=\begin{pmatrix}3\\1\end{pmatrix},\qquad
\bar W=\delta x^\top
=\begin{pmatrix}8&10\\-4&-5\end{pmatrix},\qquad
\bar b=\begin{pmatrix}2\\-1\end{pmatrix}.`

The tensor types prevent transposing the outer product accidentally: $`\delta\otimes x` has shape
`[outDim, inDim]`, while $`x\otimes\delta` has shape `[inDim, outDim]`.

# What The Linear Theorems Prove

These linear declarations are spec identities. Several are proved by unfolding the definitions
and reducing them, because `linearWeightsDerivSpec` was defined as the outer product and
`linearInputDerivSpec` was defined as the transposed weight action. They document and stabilize the
backward ABI, but they are not themselves a Fréchet-differentiability proof for the tensor map.

That distinction is worth testing. In an editor:

```
import NN.Proofs.Gradients.Linear

#check Proofs.linear_weight_gradient_correct
#check Proofs.linear_gradients_mathematical_correctness
```

The result is an equality between backward specifications. It does not mention `HasFDerivAt`,
the runtime tape, a CUDA kernel, or a finite-difference test. A whole-autograd correctness theorem
must additionally show that the tape invokes these rules with the right saved values and composes
them in reverse graph order.

The current local theorem is also for one vector. Batched accumulation, such as summing bias
cotangents over a batch dimension, belongs to the corresponding batched operator theorem; it
should not be inferred from this unbatched signature.

# Scalar Activation Calculus

Activation derivatives use a stronger style. The file
[`NN.Proofs.Gradients.Activation`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/Gradients/Activation.lean)
proves Mathlib `HasDerivAt` statements for real scalar functions. For smooth sigmoid,

$$`\sigma(x)=\frac{1}{1+e^{-x}},
\qquad
\sigma'(x)=\sigma(x)\bigl(1-\sigma(x)\bigr),`

and `sigmoid_deriv_correct` is a genuine calculus theorem. The proof constructs derivatives of
negation, exponential, addition, and inverse, then uses the chain rule.

ReLU is different:

$$`\operatorname{ReLU}(x)=\max(x,0),\qquad
\operatorname{ReLU}'(x)=
\begin{cases}
0,&x<0,\\
1,&x>0.
\end{cases}`

There is no ordinary derivative at zero. Accordingly:

```
theorem relu_deriv_correct (x : ℝ) (h : x ≠ 0) :
  HasDerivAt Activation.Math.reluSpec
    (Activation.Math.reluDerivSpec x) x
```

The same kind of nonzero hypothesis appears for leaky ReLU and general-parameter ELU. Smooth
activations such as sigmoid, tanh, softplus, SiLU, and the chosen GELU formula have global
derivative theorems; guarded functions such as `safe_log` expose their domain parameter.

Here is a complete Infoview exercise:

```
import NN.Proofs.Gradients.Activation

open Activation
open Proofs

example :
    HasDerivAt
      (Activation.Math.reluSpec : ℝ → ℝ)
      (1 : ℝ) (2 : ℝ) := by
  simpa [Activation.Math.reluDerivSpec] using
    relu_deriv_correct (2 : ℝ) (by norm_num)

example :
    HasDerivAt
      (Activation.Math.sigmoidSpec : ℝ → ℝ)
      (Activation.Math.sigmoidDerivSpec (0 : ℝ))
      (0 : ℝ) :=
  sigmoid_deriv_correct 0
```

Both examples compile without goals. Replace `2` by `0` in the ReLU example. Lean asks for
$`0\ne0`, which cannot be proved. A runtime autograd system may choose a subgradient convention at
the kink, as PyTorch does, but that convention is not an ordinary `HasDerivAt` theorem.

The same audit rule applies to the less visible domains. A theorem for raw `log` needs positivity;
one for reciprocal or division needs a nonzero denominator; square-root differentiation needs a
strictly positive point. Epsilon-protected `safe_log` and safe-division operations are different
functions, not licenses to drop those hypotheses from the raw functions. When composing a graph,
the local domain facts must be established at the actual saved forward values used by the reverse
rule.

# How The Pieces Compose

For a network

$$`x\longmapsto W_2\,\sigma(W_1x+b_1)+b_2,`

a mathematical differentiation proof uses:

1. the affine derivative in each linear layer;
2. the scalar derivative of $`\sigma`, lifted pointwise;
3. the chain rule;
4. tensor-shape and adjoint bookkeeping.

A runtime-autograd proof adds:

5. correctness of graph recording;
6. correctness of saved tensors and cotangent accumulation;
7. agreement of the selected execution backend with the operator semantics.

The probability theorem has an analogous place in diffusion: it supplies the exact forward
transition law, while a model proof must still connect that law to a schedule, a denoising
objective, and an executable sampler.

The Gaussian construction is grounded in Mathlib’s multivariate Gaussian and kernel libraries.
The diffusion equation follows Ho, Jain, and Abbeel’s DDPM formulation. The reverse-mode formulas
are standard matrix calculus; Baydin et al.’s survey gives the broader automatic-differentiation
context. TorchLean’s contribution here is not a new derivative formula, but a precise Lean object
that later graph and runtime theorems can reuse.
