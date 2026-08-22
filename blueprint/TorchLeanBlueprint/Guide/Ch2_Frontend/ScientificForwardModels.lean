import VersoManual

open Verso.Genre Manual

#doc (Manual) "Differentiable Scientific Models" =>
%%%
tag := "scientific-forward-models"
%%%

Many scientific models are small compositions of affine maps and nonlinear scalar functions. A
remote-sensing retrieval may contain exponential attenuation; a chemical rate law may contain
`exp`; a likelihood contains `log`; a PINN residual differentiates a neural field with respect to
space and time.

These examples are modest compared with a transformer, but they demand careful semantics. A wrong
sign in an exponential derivative can invalidate an inverse problem even when training appears to
run.

# One Attenuation Model

Consider:

$$`f(x)=0.8e^{-2x}+0.1`.

Write it once:

```
import NN.API
open TorchLean

def attenuated :
    autograd.func.TensorFunction Shape.scalar Shape.scalar :=
  fun x => do
    let neg ← nn.functional.scale x (-2.0)
    let e ← nn.functional.exp neg
    nn.functional.affine e 0.8 0.1
```

The operations correspond directly to:

$$`
u=-2x,\qquad
v=e^u,\qquad
f=0.8v+0.1.
`

By the chain rule:

$$`
f'(x)
=0.8e^{-2x}(-2)
=-1.6e^{-2x}.
`

At $`x=0.5`:

$$`
f'(0.5)=-1.6e^{-1}.
`

The derivative is negative. Dropping the $`-2` factor or flipping its sign is an easy implementation
mistake if forward and backward formulas are maintained separately.

# Functional Tensor Primitives

The currently implemented `nn.functional` primitives include:

```
exp x
log x
scale x c
shift x k
affine x c k
square x
mean x
detach x
```

along with broadcasting and other tensor operations. Each builds the same checked tensor program
used by autograd. There is no special “scientific mode.”

`scale x c` computes $`c\,x`; `shift x k` computes $`x+k`; and `affine x c k` computes $`c\,x+k`.
Keeping these as named primitives gives graph lowering and derivative proofs an explicit operation
to recognize.

# Run Positive And Negative Controls

```
lake exe transcendentals_check
```

The current output is:

```
[PASS] exp: grad = 1.648721 ≈ 1.648721
[PASS-NEG] exp≠1:
  grad = 1.648721 ≠ 1.000000
[PASS] affine(3x+1):
  grad = 3.000000 ≈ 3.000000
[PASS-NEG] affine≠1:
  grad = 3.000000 ≠ 1.000000
[PASS] exp(-2x):
  grad = -0.735759 ≈ -0.735759
[PASS-NEG] exp(-2x) sign:
  grad = -0.735759 ≠ 0.735759
[transcendentals] all positive + negative controls passed
```

The first case evaluates:

$$`\frac{d}{dx}e^x=e^x`

at $`x=0.5`, giving $`e^{0.5}\approx1.648721`. The affine case checks a slope of three. The third
case checks:

$$`
\frac{d}{dx}e^{-2x}=-2e^{-2x}
`

at $`x=0.5`, giving approximately $`-0.735759`.

Each positive control is paired with a wrong answer that must *not* match. This matters because a
test that always returns gradient one would pass a poorly chosen identity example. The negative
controls demonstrate that the test can distinguish the defect it is meant to detect.

# Change The Chain Rule

Open:

[`Transcendentals.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Functional/Transcendentals.lean)

and change:

```
let u ← nn.functional.scale x (-Numbers.two)
```

to:

```
let u ← nn.functional.scale x (-Numbers.three)
```

The existing expected derivative fails. Update it to:

$$`-3e^{-3x}`.

This experiment shows that autograd differentiates the forward program that actually exists. The
closed form in the test remains an independent oracle and must be updated separately.

# The Test And The Theorem Are Different

The executable checks above use the Float tape at three concrete inputs. The file also imports a
proof object:

```
noncomputable def expProofSurface :
    Proofs.Autograd.OpSpecFDerivCorrect 1 1 :=
  Proofs.Autograd.OpSpecFDerivCorrect.exp (n := 1)
```

This structure packages:

- the real-valued forward operation;
- its JVP candidate;
- its backward candidate;
- a proof that the JVP is the Fréchet derivative.

From it, TorchLean proves:

```
theorem expBackward_eq_adjoint_fderiv
    (x δ : Spec.Tensor ℝ (.dim 1 .scalar)) :
    Proofs.Autograd.toVecE
      (expProofSurface.correct.op.backward x δ) =
    Proofs.Autograd.vjp
      expProofSurface.forwardVec
      (Proofs.Autograd.toVecE x)
      (Proofs.Autograd.toVecE δ) := ...
```

This is a universal mathematical statement over real tensors. It is stronger than the three Float
checks, but it speaks about the proof-layer operation. Relating it to host Float, CUDA, or LibTorch
requires the runtime refinement boundary for the selected implementation.

# `log` Has A Domain

The real logarithm is defined for positive arguments in the intended scientific interpretation.
Its derivative:

$$`\frac{d}{dx}\log x=\frac1x`

also assumes $`x>0`.

The eager CPU tape, explicit IR evaluator, and proved forward fragment reject nonpositive raw-log
inputs. A lowered typed graph fails on an invalid domain rather than inventing a real logarithm of
a negative value. CUDA follows the native buffer operation, whose agreement needs a backend
contract.

A common attempted fix is:

$$`\log(x+\epsilon)`.

For arbitrary negative $`x`, adding a small positive $`\varepsilon` does not guarantee positivity. The
actual alternatives are:

- prove a precondition $`x>0`;
- construct a positive quantity, such as $`\operatorname{softplus}(x)+\varepsilon`, with a supported
  operation;
- use a deliberately total guarded operation such as the lower `safeLog` primitive and state its
  semantics.

The guard is part of the model. It must appear in the theorem and runtime contract rather than
living only in a comment.

There is also a specification distinction to keep visible. Mathlib's real `log` and `sqrt` are
total functions, and the rounded-real `NF`/`FP32` operations inherit that totalization. They do not
produce IEEE NaN or infinity for invalid inputs. Scientific theorems should therefore state the
positive-domain hypotheses they mean, while executable IEEE behavior belongs to `IEEE32Exec` or a
named runtime contract.

# Multiple Inputs

A scientific equation may use parameters and observations:

$$`
g(a,b,x)=a e^{-bx}+c.
`

One clean representation is a typed input pack or a tensor whose final axis has a declared feature
order. The choice must be explicit because the shape `[3]` alone does not tell us whether coordinate
zero means $`a`, $`b`, or $`x`.

The functional tensor namespace does not export every indexing operation. For a
multi-feature program, use the implemented gather/projection operation at the appropriate layer or
define a typed pack. Writing nonexistent `index1d` syntax in documentation would produce an example
that cannot run.

# From A Forward Model To An Inverse Problem

Suppose observations satisfy:

$$`y_i=f_\theta(x_i)+\eta_i`.

A least-squares estimate minimizes:

$$`
L(\theta)=
\frac1N\sum_{i=1}^{N}
\left(f_\theta(x_i)-y_i\right)^2.
`

TorchLean can:

1. express $`f_\theta` as a checked tensor program;
2. build the MSE objective;
3. differentiate with respect to the parameter pack;
4. train through a selected runtime;
5. state real-valued derivative theorems for supported primitives;
6. attach numerical or backend evidence to the executable path.

These are separate layers of one workflow. An optimizer finding a low loss does not prove parameter
identifiability. A correct derivative theorem does not prove the observations follow the assumed
model. A native result does not inherit real semantics without a bridge.

# PINN Residuals

For a PDE:

$$`\mathcal N[u]=0`,

a PINN uses a neural field $`u_\theta(x,t)` and a residual loss:

$$`
L_{\mathrm{PDE}}(\theta)
=
\frac1N\sum_i
\left|\mathcal N[u_\theta](x_i,t_i)\right|^2.
`

The derivative with respect to input coordinates is not the same derivative as the gradient with
respect to model parameters:

- input derivatives form $`u_t`, $`u_x`, $`u_{xx}`, and similar PDE terms;
- parameter derivatives optimize the residual loss.

TorchLean's function JVP/Jacobian/Hessian APIs can express input derivatives for supported programs,
while model autograd supplies parameter gradients. The scientific-ML verification chapter later
explains how a finite residual artifact becomes a checked claim and what remains between grid
samples.

# Numerical Semantics Matter More For Transcendentals

For exact reals:

$$`e^{a+b}=e^ae^b`.

In floating point, evaluating the two sides may round differently and may overflow along different
paths. A theorem about the real identity is not a theorem about either evaluation order.
Transcendental implementations also come from host libraries, CUDA `libdevice`, LibTorch, or
TorchLean reference code; agreement needs a stated approximation or refinement contract.

TorchLean's executable `IEEE32Exec` transcendental functions are deterministic definitions, but
they do not currently carry a universal correctly-rounded contract. In particular, a small
derivative regression such as $`\exp(-2x)` supports that tested point and code path; it does not
establish global accuracy for `exp`, `log`, `sin`, or `cos` over all finite bit patterns.

# Continue The Experiment

Three useful modifications are:

1. replace $`-2` by a parameter and inspect its gradient sign;
2. add a sum or mean over a vector of observation locations and compare the scale of the gradient;
3. run the same finite example with `IEEE32Exec` and compare against host Float.

Keep the forward equation unchanged while changing only the derivative query. That is the advantage
of an autograd-aware scientific model: JVPs, VJPs, parameter gradients, and residual derivatives all
refer to the same program.

Sources and references:

- [`Transcendentals.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Functional/Transcendentals.lean);
- [`Functional/Core.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Runtime/Autograd/TorchLean/Functional/Core.lean);
- [PyTorch `torch.exp`](https://docs.pytorch.org/docs/stable/generated/torch.exp.html);
- [PyTorch `torch.log`](https://docs.pytorch.org/docs/stable/generated/torch.log.html);
- Baydin et al.,
  [Automatic Differentiation in Machine Learning](https://arxiv.org/abs/1502.05767).
