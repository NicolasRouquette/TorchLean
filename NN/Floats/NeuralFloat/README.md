# NeuralFloat (`NN/Floats/NeuralFloat`)

This folder provides generic rounded arithmetic over `ℝ`. It is the reusable layer underneath
TorchLean's proof-oriented floating-point models: precision metadata, exponent formats, rounding
modes, rounded scalar values, and small error lemmas.

This layer models rounded arithmetic independently of any executable kernel. Executable binary32
behavior lives in
`NN/Floats/IEEEExec/`; CUDA, ATen/libtorch, and native runtime behavior are separate trust-boundary
topics.

That distinction is important. A theorem about `NeuralFloat` says something like "this expression is
the result of applying a declared rounding model to real arithmetic." Claims about a particular GPU
kernel, BLAS call, or libtorch operator should cite the executable bridge or trust-boundary statement
that connects the backend to that rounding model.

## Relationship To Flocq

This is a native Lean development that follows Flocq's mathematical organization; it is not a
line-by-line translation of the Coq library. The implemented generic theory includes radix powers
and magnitude, valid exponent functions, FIX/FLX/FLT and abrupt-underflow formats, canonical
representations, directed and nearest rounding, round-away and round-to-odd, ULPs and neighboring
values, relative and absolute error bounds, directed double rounding, and Sterbenz subtraction.

Flocq is larger than this layer. In particular, its specialized proofs for individual arithmetic
algorithms, effective-computation infrastructure, and Coq-specific application modules have not
all been reproduced here. TorchLean's executable IEEE-754 binary32 development is instead under
`NN/Floats/IEEEExec/`, with explicit bridge theorems connecting finite executable results to the
rounded-real model. The calculation layer under `NN/Floats/Calc/` supplies bracket refinement,
canonical truncation, and representation-level arithmetic used by the rounded-real proofs.

## Structure

- `Core.lean` defines radix powers and the exact mantissa/exponent carrier `NeuralFloat`.
- `Metadata.lean` contains provenance annotations used by conversion and runtime-refinement code. It
  does not define numerical semantics.
- `Format/` defines magnitudes, digit counts, exponent functions, and representable grids.
- `Rounding/` defines rounding modes and proves their order and double-rounding properties.
- `Scalar/` packages rounded-real semantics as `NF` and supplies ordinary scalar operations.
- `Analysis/` studies ULP spacing, neighboring values, and exact subtraction.
- `Error/` proves absolute, relative, directed, and exact-residual results.
- `Special/` contains execution policies such as flush-to-zero that intentionally differ from the
  generic gradual-underflow model.

Each directory has one umbrella module with the same name.  Import the narrow folder umbrella when
possible; import `NN.Floats.NeuralFloat` only when the full generic theory is required.
Storage-width-independent affine quantization lives in `NN.Floats.Quantization`. Its
rank-polymorphic tensor adapter lives separately in `NN.Spec.Quantization`, so using the numerical
library does not require TorchLean's tensor specifications.

## When To Use It

Use `NeuralFloat`/`NF` when the theorem should be parametric in a rounded arithmetic model. Use
`NN.Floats.FP32` when the theorem specifically needs the binary32-sized rounded-real model. Use
`IEEE32Exec` when the object is executable IEEE-754 binary32 behavior with special values.

A useful mental model is:

- `ℝ`: ideal mathematical semantics.
- `NeuralFloat` / `NF`: a configurable rounded-real semantics suitable for proofs.
- `FP32`: the binary32-sized rounded-real specialization used by many neural-network statements.
- `IEEE32Exec`: executable IEEE-754-style behavior, including finite/special-value cases.
- runtime floats: what the backend actually computed, subject to the runtime boundary documented in
  `TRUST_BOUNDARIES.md`.

For a fixed grid with positive spacing `step`, `neuralRoundAtScale` applies any valid integer
rounding rule without introducing a second format semantics. The positivity proof prevents the
totalized real division at a zero or negative scale from masquerading as a rounding operation.
`NN.Floats.Quantization` builds scalar affine
quantizers from that rounding theory, using a positive scale, zero point, and bounded integer code
interval; `NN.Spec.Quantization` supplies the tensor lift. The theorem
`neuralRoundAtScale_nearestEven_after_odd_binary_extra` proves that round-to-odd on a sufficiently
fine binary intermediate avoids nearest-even double rounding on the final grid.

Likewise, FLX, FLT, and FTZ require a positive radix-digit precision. Code receiving an integer
configuration should call `NeuralFormatPrecision.ofInt?` and use the checked value's exponent
selector. The lower exponent `emin` may be any integer; it is a location on the radix grid, not a
scale or precision. The raw exponent formulas are retained for theorem statements, while generic
rounding remains gated by `NeuralValidExp`.

`NF` deliberately permits direct construction from a real because comparison and approximation
proofs sometimes need values that are not on the grid. Use `NF.ofReal` for rounded values and carry
`NF.IsRepresentable` when a theorem relies on operand representability. Primitive arithmetic rounds
its result even when an input was introduced through the raw constructor.

## What Belongs Here

This layer is a good home for generic rounding definitions, format predicates, conversion metadata,
and reusable error lemmas. It is not the place for CUDA performance claims, ATen kernel behavior, or
end-to-end model accuracy results. Those claims need to name the backend and the evidence being used.

## References

- S. Boldo and G. Melquiond, "Flocq: A Unified Library for Proving Floating-Point Algorithms in
  Coq," *IEEE ARITH*, 2011, doi:10.1109/ARITH.2011.40.
- IEEE, *IEEE Standard for Floating-Point Arithmetic*, IEEE 754-2019.
- D. Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic,"
  *ACM Computing Surveys* 23(1), 1991, doi:10.1145/103162.103163.
- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, second edition, SIAM, 2002.
