import VersoManual

open Verso.Genre Manual

#doc (Manual) "Approximation Theory" =>
%%%
tag := "approximation-theory"
%%%

A trained network is an approximation in an everyday sense, but that sentence hides three
different mathematical claims. Suppose a ReLU network is intended to represent a function `f` on
an interval.

First, an approximation theorem may say that a suitable network *exists*. Second, after choosing
weights, a verifier may enclose the values of that particular network on an input box. Third, a
runtime theorem may compare ideal real arithmetic with the binary32 program that actually runs.
The three statements have different quantifiers:

$$`\begin{aligned}
\text{representation:}\quad&
  \forall f\,\forall\varepsilon>0\,\exists\theta\,
  \sup_{x\in K}|N_\theta(x)-f(x)|<\varepsilon,\\
\text{verification:}\quad&
  \forall x\in B,\quad N_\theta(x)\in\mathcal A_\theta(B),\\
\text{execution:}\quad&
  \forall x\in B,\quad
  |N_{\theta,\mathrm{run}}(x)-N_{\theta,\mathbb R}(x)|\leq\delta(x).
\end{aligned}`

Only the first line is universal approximation. It does not certify a trained checkpoint, choose
the parameters for an optimizer, or prove that floating-point evaluation is close to the real
network. TorchLean keeps these statements near one another because they eventually need to be
composed, but it does not identify them.

# Building A ReLU Approximant

The one-dimensional construction is easier to understand in hinge notation. Let

$$`H(x)=b_0+\sum_{i=0}^{N-1} c_i\,\operatorname{ReLU}(x-t_i).`

Each term changes the slope at one knot $`t_i`. By choosing knots on a sufficiently fine mesh and
choosing $`c_i` from changes in the piecewise-linear slope, $`H` interpolates a Lipschitz target.
The same expression is a one-hidden-layer network: the first linear layer computes $`x-t_i`, ReLU
applies the hinges, and the second linear layer forms their weighted sum.

The theorem
[`relu_universal_approximation_Icc`](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Proofs/Approximation/Universal/UniversalApproximation.lean)
makes precisely that conversion. Its current signature is:

```
theorem relu_universal_approximation_Icc
    {f : ℝ → ℝ} {a b L : ℝ}
    (h_ab : a < b)
    (hL : 0 < L)
    (h_lip :
      ∀ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b,
        |f x - f y| ≤ L * |x - y|) :
    ∀ ε > 0,
      ∃ (hidDim : ℕ)
        (l1 : LinearSpec ℝ 1 hidDim)
        (l2 : LinearSpec ℝ hidDim 1),
      ∀ x ∈ Set.Icc a b,
        |f x - mlpEval1d hidDim l1 l2 x| < ε
```

The assumptions are not decoration. $`a<b` gives a nonempty interval with positive length.
$`L>0` and `h_lip` provide a quantitative continuity bound. $`\varepsilon>0` is needed before a finite
mesh can be chosen. The conclusion supplies an actual TorchLean `LinearSpec` pair, not merely an
unnamed continuous approximating function.

The rate theorem chooses the width

$$`N(L,a,b,\varepsilon)
  = \left\lceil\frac{2L(b-a)}{\varepsilon}\right\rceil+1.`

In Lean this is
[`reluApproximationWidth`](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Proofs/Approximation/Universal/UniversalApproximationRate.lean),
and `relu_universal_approximation_Icc_rate` uses exactly that hidden dimension. The arithmetic
lemma underneath it proves

$$`\frac{2L(b-a)}{N}<\varepsilon.`

This bound is conservative. It establishes a construction with a stated size; it does not claim
that the width is minimal, nor that gradient descent will discover these hinge parameters.

# An Infoview Experiment

Create a scratch file at the repository root and ask Lean for the theorem types:

```
import NN.MLTheory.Proofs.Approximation.Universal.UniversalApproximationRate

open NN.MLTheory.Proofs.UniversalApproximation

#check reluApproximationWidth
#check relu_universal_approximation_Icc_rate
#check two_mul_mul_sub_div_relu_approximation_width_lt
```

Run it with `lake env lean Scratch.lean`, or hover over each declaration in an editor. The second
line should display a conclusion of the form

```
∃ l1 l2, ∀ x ∈ Set.Icc a b,
  |f x - mlpEval1d (reluApproximationWidth L a b ε) l1 l2 x| < ε
```

There is a revealing variation. Try:

```
#eval reluApproximationWidth 1 0 1 (1 / 10)
```

Lean rejects this with a `dependsOnNoncomputable` error. The width uses `Nat.ceil` on mathematical
real numbers, so it is a proof-level construction rather than a compiled numerical routine. The
error marks the boundary between an existence proof over exact reals and an executable
parameter-selection program. A runtime tool could compute the same formula from rational inputs,
but that would be a separate executable definition with a refinement theorem.

# From Real Parameters To Binary32

The real theorem is only the first leg of a finite-precision result. Once knots, coefficients, and
the bias are stored in binary32, the total error naturally splits into

$$`\begin{aligned}
|f(x)-H_{\mathrm{IEEE}}(x)|
\leq{}&
|f(x)-H_{\mathbb R}(x)|\\
&+|H_{\mathbb R}(x)-H_{\mathrm{embedded}}(x)|\\
&+|H_{\mathrm{embedded}}(x)-H_{\mathrm{IEEE}}(x)|.
\end{aligned}`

These are, respectively:

1. approximation error of the real hinge network;
2. parameter-quantization or reference error;
3. rounded evaluation error.

The distinction is visible in
[`reluApproximationIccIEEE32Exec_threeTerm`](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Proofs/Approximation/Universal/UniversalApproximationIEEE32Exec.lean).
The theorem takes real hinge parameters `tR` and `cR`, executable `IEEE32Exec` parameters `t`, `c`,
and `b0`, and separate assumptions for the real approximation and quantization terms. It also
requires finiteness witnesses for the intermediate hinge sum and output. Its conclusion adds
`hingeFunErrorBound` to the two supplied tolerances.

That theorem deliberately does not say that every real parameter can be converted to binary32
without overflow. The more concrete
`reluApproximationIccIEEE32Exec_dyadicHalfUlp` starts from dyadic parameters and uses a half-ULP
rounding bound, but it still asks for finite evaluation witnesses. NaNs and infinities do not
disappear because the target function was continuous.

TorchLean also has an intermediate
[`FP32` theorem](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Proofs/Approximation/Universal/UniversalApproximationFP32.lean).
`relu_universal_approximation_Icc_fp32` evaluates the hinge construction in the clean finite
rounding model and proves a pointwise bound of the form

$$`|f(x)-H_{\mathrm{FP32}}(x)|
  < \varepsilon+\operatorname{hingeFunErrorBound}(x).`

`FP32` is convenient for error analysis because values are represented by reals rounded at
binary32 precision with gradual underflow and no upper exponent cutoff. `IEEE32Exec` is the
explicit bit-level model with overflow, signed zero, subnormals, infinities, and NaNs. A proof in
the former is not silently promoted to the latter.

# Exact Finite Interval Images

Approximation theory also appears in a finite semantic form. The module
[`FloatInterval.Semantics`](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Proofs/Approximation/FloatInterval/Semantics.lean)
defines intervals of `IEEE32Exec` values and computes abstract operations by enumerating the finite
concrete image and taking its hull. For addition, the central theorem is:

```
theorem add_sound (A B : I) :
  ∀ {x y : F}, x ∈ A → y ∈ B →
    IEEE32Exec.add x y ∈ addSharp A B
```

The multiplication and ReLU theorems have the same shape. They are then composed through an affine
layer and a two-layer ReLU network:

```
theorem eval_sound [OpsExact.Sound]
    (net : Net d h) (B : I.Box d)
    (hW1 : ...)
    (hb1 : ...)
    (hW2 : ...)
    (hb2 : ...) :
  ∀ {x}, x ∈ I.γ B → eval net x ∈ evalSharp net B
```

The weight and bias hypotheses rule out NaN parameters. The conclusion is about the explicit
`IEEE32Exec` evaluation, not an ideal real network. Because binary32 is finite, the exact abstract
operators can in principle enumerate every concrete pair in an interval. That makes them excellent
reference semantics and very poor large-scale kernels: their cost grows with the number of
represented values.

Run the proof modules directly:

```
lake env lean \
  NN/MLTheory/Proofs/Approximation/FloatInterval/Semantics.lean

lake env lean \
  NN/MLTheory/Proofs/Approximation/FloatInterval/ExactImageTheorem.lean
```

A successful run is silent and exits with status zero. To inspect the composition point, use:

```
import NN.MLTheory.Proofs.Approximation.FloatInterval.Semantics

open NN.MLTheory.Proofs.Approximation.FloatInterval

#check OpsExact.add_sound
#check aff_sound
#check eval_sound
```

As a deliberate failure, remove one of the `isNaN ... = false` hypotheses from a attempted use of
`aff_sound`. Lean leaves exactly that missing premise as a goal. A point interval containing NaN
cannot be treated as an ordinary ordered singleton, so the proof correctly refuses to proceed.

# Higher-Dimensional Domains

The one-dimensional hinge proof is constructive and quantitative. The higher-dimensional
development uses a different route. In
[`UniversalApproximationND`](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Proofs/Approximation/Universal/UniversalApproximationND.lean),
`TensorVec n` is identified with `Fin n → ℝ` by a homeomorphism. Coordinate functions generate a
subalgebra of continuous functions, and the proof establishes that this subalgebra separates
points. Stone-Weierstrass then supplies density on compact domains.

That topological argument answers a broad representation question, but it does not produce the
same explicit width formula as the one-dimensional Lipschitz construction. The two proofs are
complementary:

- the hinge mesh exposes parameters and a rate on $`[a,b]`;
- the coordinate-subalgebra proof handles compact multidimensional domains at a more abstract
  level.

# What The Result Buys

Two endpoints complete the implemented path beyond the scalar hinge construction. The
[two-layer IEEE32Exec approximation API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Proofs/Approximation/Universal/UniversalApproximationIEEE32ExecTwoLayerMlp.lean)
packages representation, parameter-rounding, and executable-rounding error in
`relu_twoLayerMlp_ieee32exec_threeTerm`. For finite interval semantics, the
[constant-target API](https://github.com/lean-dojo/TorchLean/blob/main/NN/MLTheory/Proofs/Approximation/FloatInterval/ConstantTarget.lean)
proves `exactIntervalImage_constant`: a finite constant has the exact singleton interval image.
The latter is a useful base case for certificate composition, not a claim that an arbitrary
nonconstant network has an exact interval image.

After these layers are composed, one can make a statement with all errors visible:

$$`\text{target error}
\leq
\text{representation error}
+\text{parameter rounding error}
+\text{execution error}.`

A CROWN or interval certificate can then add a fourth component: a sound enclosure over an input
region for the chosen network. Each term comes from a different argument: model construction,
parameter conversion, runtime arithmetic, and regional verification. Writing the sum explicitly
lets an application decide where to spend its error budget.

The constructions follow the classical universal-approximation tradition, including the
Cybenko and Hornik results, while the explicit ReLU viewpoint is closer to modern constructive
piecewise-linear proofs. The finite interval development is instead an abstract-interpretation
argument over the executable binary32 carrier.
