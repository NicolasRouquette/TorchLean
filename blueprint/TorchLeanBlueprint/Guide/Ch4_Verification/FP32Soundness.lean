import VersoManual

open Verso.Genre Manual

#doc (Manual) "Float32 Soundness" =>
%%%
tag := "fp32-soundness"
%%%

Suppose a real-valued verifier proves that the winning logit leads a competitor by `0.12`. That is
not yet a statement about Float32 execution. If each rounded logit may move by `0.03`, the winner
may move down while the competitor moves up, leaving only

$$`0.12-0.03-0.03=0.06.`

The classification still survives, but for a numerical reason that must appear in the proof. If
the original margin had been `0.04`, the same real theorem would no longer settle the Float32
question.

To justify the `0.03`, we must choose a Float32 model, derive an operator-level error budget, compose
those bounds through the network, and account for the result in the verifier's final margin.

# Two Float32 Objects, Two Jobs

`FP32` is TorchLean's rounded-real proof model:

```
abbrev FP32 := NF binaryRadix fexp32 rnd32
```

Arithmetic is performed exactly over the reals and rounded to the binary32-precision,
gradual-underflow grid with nearest-even rounding. This makes error expressions readable. The model
does not contain NaN, infinity, signed zero, exception flags, or binary32 overflow to infinity; its
exponent policy has no upper cutoff.

`IEEE32Exec` instead stores a `UInt32` bit pattern and executes the binary32 special cases. It is the
right reference when a claim depends on NaN, infinity, subnormals, signed zero, or exact result bits.
The two APIs begin at the
[FP32 proof semantics](https://github.com/lean-dojo/TorchLean/blob/main/NN/Floats/FP32.lean) and the
[IEEE32Exec semantics](https://github.com/lean-dojo/TorchLean/blob/main/NN/Floats/IEEEExec/Exec32.lean).

Exact bridge theorems join the models for covered finite-path operations: basic arithmetic, FMA,
square root, order, and min/max. Their qualifications are operation-specific; for example, division
needs a nonzero denominator and a composite expression needs finite intermediate results rather
than merely finite inputs. Executable `log` and `tanh` still need a separate accuracy or refinement
contract. A subnormal result is finite; `0/0`, overflow to infinity, and square root of a negative
value are not paths that the rounded-real theorem silently absorbs.

# Run The Two Float32 Views

TorchLean includes a small forward-and-backward comparison using the same MLP parameters in host
`Float` arithmetic and in the executable `IEEE32Exec` semantics:

```
lake exe torchlean float32_modes
```

The command first names the available meanings:

```
Float32 mode: FP32: proof semantics (round-on-ℝ), finite-only; no NaN/Inf
Float32 mode: IEEE32Exec: executable IEEE-754 binary32 kernel (bit-level; includes NaN/Inf)
```

It then prints the output, parameter gradients, and input gradient for both executable paths. The
final comparison on the bundled example is:

```
max_abs_diff(Float vs IEEE32Exec) =
  0.0000000762939453835542735760100185871124267578125
```

This number is an observation about one input and one network. It is not a uniform error theorem.
The proof task is to derive a bound $`\varepsilon` from input ranges, parameter ranges, and the sequence of
rounded operations, then prove that every execution covered by those hypotheses differs from the
real specification by at most $`\varepsilon`.

Try changing the example's weights by a power of two and by a nearby non-power-of-two decimal. The
former often passes through binary arithmetic exactly; the latter exposes rounding earlier. The
experiment gives intuition for the formal representability and ULP theorems developed in the
floating-point chapters.

# Follow The Error Through One Linear Layer

The smallest useful network calculation is already more than one rounded operation:

$$`y_i=b_i+\sum_{j=0}^{n-1}W_{ij}x_j.`

The runtime weights, bias, and input may begin near their real counterparts. Each product introduces
another rounded result, the dot product accumulates those results in a declared order, and the bias
addition rounds once more. `linearErrorBudget` is the explicit expression obtained by composing the
matrix-vector and final-addition bounds. `approxT_linear_fp32` proves that this expression bounds
every output coordinate.

This matters more than a theorem that merely says some tolerance exists. A caller can inspect the
budget, compare it with a safety margin, and see whether wider inputs or larger weights caused the
loss of precision.

```
import NN.Proofs.RuntimeApprox.FP32.Layers

open NN.Proofs.RuntimeApprox.FP32

#check linearErrorBudget
#check approxT_linear_fp32
```

The real side uses `LinearSpec ℝ inDim outDim`; the rounded side uses the same tensor
specification at scalar type `R`, the FP32 abbreviation in this proof namespace. `approxT` relates
the two tensors componentwise after interpreting the rounded values as reals. Thus the theorem is
about a shaped layer, not an isolated scalar multiply.

# Compose The Layer Bounds Into A Network

The two-layer ReLU theorem feeds the first linear budget through the rounded ReLU rule, then uses the
result as the input budget for the second linear layer. The three-layer tanh theorem repeats the
same pattern through two smooth activations. Both expose their final expressions rather than hiding
them behind an existential tolerance.

```
import NN.Proofs.RuntimeApprox.FP32.MLP

open NN.Proofs.RuntimeApprox.FP32

#check reluTwoLayerMlpErrorBudget
#check approxT_reluTwoLayerMlp_float32
#check tanhMlp3ErrorBudget
#check approxT_tanhMlp3_fp32
```

These are architecture-shaped theorems for `Linear → ReLU → Linear` and
`Linear → tanh → Linear → tanh → Linear`. They demonstrate composition and cover the
corresponding examples; they are not a claim that every model assembled from arbitrary operations
already has an FP32 theorem.

# Spend The Budget In A Verification Result

Suppose real IBP proves that an output lies in a box $`[\mathrm{lo},\mathrm{hi}]`, while the rounded network theorem
gives an error `epsOut` at a fixed input. Inflating every output coordinate by that amount gives

$$`[\mathrm{lo}-\varepsilon_{\mathrm{out}},\;\mathrm{hi}+\varepsilon_{\mathrm{out}}].`

`ibpBound_contains_reluTwoLayerMlp_float32` proves this construction pointwise for the two-layer
ReLU MLP. It combines the real IBP theorem with `approxT_reluTwoLayerMlp_float32`; its named budget
is `ibpReluTwoLayerErrorBudget`. The budget depends on the chosen real input, while
`inflateBoxUniform` merely applies one chosen amount uniformly across output coordinates. To obtain
one rounded enclosure valid for every input in the input box, first prove a domain-wide upper bound
on that pointwise budget.

```
import NN.Proofs.RuntimeApprox.FP32.CROWN

open NN.Proofs.RuntimeApprox.FP32

#check ibpReluTwoLayerErrorBudget
#check ibpBound_contains_reluTwoLayerMlp_float32
#check fp32_le_of_real_le_sub_margin
#check fp32_ge_of_real_ge_add_margin
```

For a scalar threshold, `fp32_le_of_real_le_sub_margin` and
`fp32_ge_of_real_ge_add_margin` package the same arithmetic. For a classifier, apply it to both
logits: the true logit may fall by its budget and the competitor may rise by its budget. The real
margin must pay both costs, just as in the `0.12` example at the start.

# From The Proof Model To An Executed Program

The result so far concerns the rounded-real `FP32` semantics. For a network whose primitives are
covered, an `IEEE32Exec` claim can use the exact finite-path bridges and discharge their domain and
finiteness hypotheses. The current bridges cover basic arithmetic, FMA, square root, order, and
min/max behavior; they do not yet give an accuracy or refinement theorem connecting executable
`log` or `tanh` to the rounded-real operations. In particular, the tanh MLP theorem above remains a
result about the `FP32` proof model until such a transcendental contract is supplied.

To cite either result for Lean's host `Float32`, CUDA, cuBLAS, or LibTorch, one more agreement
statement must connect that provider to the executable or rounded model and must account for
reduction order, contraction, and exceptional behavior.

The `float32_modes` difference printed earlier is valuable regression evidence for one execution;
it is not that provider agreement theorem. A complete deployment claim therefore has a visible
chain:

```
real property
  + FP32 approximation budget
  + finite IEEE bridge for each covered primitive
  + native-provider agreement
  = property of the selected execution path
```

Some applications stop earlier because their theorem is intentionally about the rounded-real
model. That is a legitimate claim as long as it says so. For the construction of the two numerical
representations, read *Floating-Point Semantics*; for the graph and checker side of the argument,
return to *Neural Network Verification*.

# References

- IEEE 754-2019 standard: https://standards.ieee.org/standard/754-2019.html
- Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic",
  https://dl.acm.org/doi/10.1145/103162.103163
- Flocq: https://flocq.gitlabpages.inria.fr/
- Higham, *Accuracy and Stability of Numerical Algorithms* (2nd ed., SIAM, 2002), the standard
  numerical analysis reference for forward error and stability arguments of the kind TorchLean
  packages into margin-transfer lemmas.
