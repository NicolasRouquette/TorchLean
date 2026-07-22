import VersoManual

open Verso.Genre Manual

#doc (Manual) "Worked Examples" =>
%%%
tag := "examples"
%%%

The examples in this chapter form a progression. The first prints typed values. The second
differentiates a concrete linear map. The third trains a small network. The fourth lowers a model
to the verification IR and propagates an input box. The final example checks a numerical
certificate with explicit binary32 semantics.

Run them in order the first time. Each introduces one new object while keeping the earlier ones
visible.

# Lab One: A Tensor Has A Shape And A Scalar Meaning

Run:

```
lake exe torchlean quickstart_tensors
```

The current output is:

```
== Quickstart: tensor basics ==
[Float] [0.100000, 0.200000, 0.300000, 0.400000]
[ℚ] [1/10, 1/5, 3/10, 2/5]
[Int] [1, 2, 3, 4]
[IEEE32Exec] [0.100000, 0.200000, 0.300000, 0.400000]
[Float] [[[1.000000, 2.000000], [3.000000, 4.000000]],
         [[5.000000, 6.000000], [7.000000, 8.000000]]]
Expected failure printing Tensor ℝ: Refusing to print `Tensor ℝ` (proof-level);
cast to `Float`/`IEEE32Exec`/`ℚ` to display.
```

The first four tensors have the same vector shape and superficially similar entries, but their
scalar meanings differ:

- `Float` is Lean's native executable floating-point value;
- `ℚ` is exact rational arithmetic;
- `Int` is exact integer arithmetic;
- `IEEE32Exec` is TorchLean's explicit executable IEEE-754 binary32 model;
- `ℝ` is suitable for proofs but is not an object the runtime should pretend to print.

The shape of the final tensor is `2 × 2 × 2`. It is represented by nested, length-indexed
dimensions, so the eight entries cannot accidentally be interpreted as a `4 × 2` matrix without an
explicit reshape proof.

Read
[`TensorBasics.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/TensorBasics.lean)
beside the output. Change one scalar type at a time and let Lean show which operations require a
different algebraic context.

# Lab Two: Reverse And Forward Differentiation

The autograd quickstart uses a linear map from two inputs to three outputs. Run:

```
lake exe torchlean quickstart_autograd
```

For input

$$`x=(0.5,-1.2),`

the Jacobian of `y=Wx+b` with respect to `W` has three block rows. The command prints those rows:

```
jacrevOutParams rows = 3 (should be size(out)=3)
  row[0] dW = [[0.500000, -1.200000],
               [0.000000, 0.000000],
               [0.000000, 0.000000]]
  row[1] dW = [[0.000000, 0.000000],
               [0.500000, -1.200000],
               [0.000000, 0.000000]]
  row[2] dW = [[0.000000, 0.000000],
               [0.000000, 0.000000],
               [0.500000, -1.200000]]
```

That output is easy to check by hand:

$$`\frac{\partial y_i}{\partial W_{jk}}
=\begin{cases}
x_k,&i=j,\\
0,&i\ne j.
\end{cases}`

The same program then prints VJPs, JVPs, an HVP, a Hessian, and gradients through `detach`. The
detach check is especially useful:

```
loss(mse ∘ detach) = 0.165133
gradParams (mse ∘ detach) gW = [[0.000000, 0.000000],
                                [0.000000, 0.000000],
                                [0.000000, 0.000000]]
gradParams (mse ∘ detach) gb = [0.000000, 0.000000, 0.000000]
```

The value is unchanged while the gradient path is cut. This is an executable check against a
closed form, not by itself a theorem about every autograd operation. Proof coverage for individual
forward and backward rules lives in the proof modules described earlier in the guide.

Source:
[`AutogradBasics.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/AutogradBasics.lean).

# Lab Three: Train A Small MLP

The training quickstart generates 25 regression samples for a two-input, one-output function and
uses a hidden layer of width eight.

```
lake exe torchlean quickstart_mlp \
  --device cpu --steps 20 --seed 2026
```

The command reports the same two probes before and after optimization:

```
== Quickstart: simple MLP training ==
seed  = 2026
steps = 20
dataset size = 25
mean_loss(before) = 0.761530
predictions(before)
  center:  x=(0.000000,0.000000)   target=0.200000  pred=[0.000000]
  heldout: x=(0.250000,-0.750000) target=0.200000  pred=[0.043283]
step 0: loss=0.000866
mean_loss(after) = 0.459876
predictions(after)
  center:  x=(0.000000,0.000000)   target=0.200000  pred=[-0.132460]
  heldout: x=(0.250000,-0.750000) target=0.200000  pred=[0.012967]
steps=20 loss0=0.761530 loss1=0.459876
predict(heldout) = [0.012967]
```

The per-step loss at step zero is not the dataset mean. It is the loss for the first streamed
sample. Reading the two as the same statistic would be a subtle reporting error; the labels keep
them distinct.

## Change The Numerical Meaning

The quickstart is scalar-polymorphic enough to run with the executable binary32 model:

```
lake exe torchlean quickstart_mlp \
  --device cpu --dtype ieee754exec \
  --steps 2 --seed 2026
```

This selects `IEEE32Exec` for the covered path. The general model-zoo trainers are mostly native
`Float` applications, so do not assume that every subcommand accepts this dtype. Command-specific
validation rejects unsupported combinations.

## Train Longer

At 200 updates with the same seed, the held-out prediction is close to its target:

```
lake exe torchlean quickstart_mlp \
  --device cpu --steps 200 --seed 2026
```

```
mean_loss(before) = 0.761530
mean_loss(after) = 0.003234
heldout x=(0.25,-0.75), target=0.2, prediction(after)=[0.210239]
```

This demonstrates learning on one generated problem. It is neither a convergence theorem nor a
generalization bound.

Source:
[`SimpleMlpTrain.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Quickstart/SimpleMlpTrain.lean).

# Lab Four: Lower A Model And Bound Its Output

The previous labs evaluated one concrete input at a time. Interval bound propagation starts from an
input box

$$`x\in[\ell,u]`

and computes an output box enclosing every supported model evaluation in that region.

Run the registered workflow:

```
lake exe verify -- torchlean-ibp
```

Current output:

```
=== TorchLean → IR → IBP (small MLP) workflow ===
[TorchLean] Float32 mode: IEEE32Exec: executable IEEE-754 binary32 kernel
compiled IR nodes: 20
output box lo: [1.904000]
output box hi: [2.256000]
```

The path is:

```
TorchLean model
  -> canonical IR
  -> supported-operation check
  -> IEEE32Exec interval propagation
  -> output box
```

For an affine layer `y=Wx+b`, IBP separates positive and negative weights:

$$`\begin{aligned}
\ell'_i
&=\sum_j
  \left(\max(W_{ij},0)\ell_j+\min(W_{ij},0)u_j\right)+b_i,\\
u'_i
&=\sum_j
  \left(\max(W_{ij},0)u_j+\min(W_{ij},0)\ell_j\right)+b_i.
\end{aligned}`

For ReLU,

$$`[\ell_i,u_i]
\longmapsto
[\max(0,\ell_i),\max(0,u_i)].`

The command's result belongs to the supported IR fragment and numerical policy used by this
workflow. It does not certify an arbitrary model command merely because that command also contains
linear layers and ReLUs.

Discover the other registered workflows with:

```
lake exe verify -- list
```

The implementation of this path is reachable from
[`NN/Verification/CLI.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Verification/CLI.lean).

# Lab Five: Replay A Numerical Certificate

The numerical-certificate example moves from an in-memory bound result to a checked artifact. Run:

```
lake exe torchlean numerical_certificate
```

The command checks positive and negative cases:

```
TorchLean numerical runtime certificate
  ok  base certificate
  ok  base IEEE replay
  ok  tampered range rejected
  ok  misplaced source rejected
  ok  duplicate contract rejected
  ok  registry mismatch rejected
  ok  unsupported operation rejected
  ok  fixed-left reduction
  ok  portable matmul
  ok  CUDA matmul policy rejected
  ok  directed sqrt
  ok  negative sqrt domain rejected
  ok  portable LayerNorm
  ok  CUDA LayerNorm policy rejected
  ok  stable softmax
  ok  two-layer MLP certificate
  ok  two-layer MLP IEEE replay
All numerical certificate checks passed.
```

A certificate is useful only if malformed evidence is rejected. That is why the example includes a
tampered range, an unsupported operation, a bad square-root domain, and numerical policies that do
not match the declared backend.

The graph checker and its proof layer are:

- [`GraphNumericalCertificate.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/DeepDives/Floats/GraphNumericalCertificate.lean);
- [`NN.Proofs.RuntimeApprox.Graph.NumericalCertificate`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Graph/NumericalCertificate.lean).

The certificate uses outward-rounded `IEEE32Exec` ranges and can replay concrete inputs in the
bit-level interpreter. A native runtime remains a separate provider whose agreement requires the
appropriate backend evidence.

# Lab Six: An External Graph Producer

TorchLean can ask PyTorch to export model graphs and then parse the resulting artifact:

```
lake exe torchlean pytorch_export_check
```

The command accepts supported examples such as small MLP, CNN, normalization, Transformer-shaped,
and single-head-attention graphs. It deliberately rejects unsupported structures, including
multi-head attention outside the currently implemented lowering fragment.

The important chain is:

```
Python/PyTorch producer
  -> JSON value graph
  -> Lean parser
  -> canonical TorchLean IR
  -> WellShaped result or explicit rejection
```

Acceptance proves that the parsed graph is well shaped through the parser theorem. It does not
prove PyTorch's exporter, Python interpreter, or original model source correct.

Source:
[`TorchExportCheck.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/Interop/PyTorch/TorchExportCheck.lean).

# Building The Example Suite

Before changing public examples, build the curated umbrella:

```
lake build NNExamples
```

That checks elaboration across the example tree. It does not execute every long-running model or
external dependency. Runtime regressions, CUDA checks, dataset preparation, and certificate replay
remain separate validation steps.
