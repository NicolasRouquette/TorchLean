# NeuralFloat (`NN/Floats/NeuralFloat`)

This folder defines generic rounded arithmetic over `ℝ`: radix and exponent formats, rounding
policies, rounded scalar operations, ULPs, and error bounds. It is independent of an executable
kernel. Executable binary32 arithmetic lives under `NN/Floats/IEEEExec/`.

## Relationship To Flocq

This Lean development follows Flocq's mathematical organization rather than translating the Coq
library line by line. The generic theory includes radix powers
and magnitude, valid exponent functions, FIX/FLX/FLT and abrupt-underflow formats, canonical
representations, directed and nearest rounding, round-away and round-to-odd, ULPs and neighboring
values, relative and absolute error bounds, directed double rounding, and Sterbenz subtraction.

Flocq also contains specialized arithmetic algorithms, effective-computation infrastructure, and
Coq-specific application modules that are not reproduced here. TorchLean's executable binary32 development is under
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

Each directory has an umbrella module with the same name. Import the narrow folder umbrella when
possible; import `NN.Floats.NeuralFloat` only when the full generic theory is required.
Storage-width-independent affine quantization lives in `NN.Floats.Quantization`. Its
rank-polymorphic tensor adapter lives separately in `NN.Spec.Quantization`, so using the numerical
library does not require TorchLean's tensor specifications.

## Choosing A Representation

Use `NeuralFloat`/`NF` when the theorem should be parametric in a rounded arithmetic model. Use
`NN.Floats.FP32` when the theorem specifically needs the binary32-sized rounded-real model. Use
`IEEE32Exec` when the object is executable IEEE-754 binary32 behavior with special values.

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

## References

- S. Boldo and G. Melquiond, "Flocq: A Unified Library for Proving Floating-Point Algorithms in
  Coq," *IEEE ARITH*, 2011, doi:10.1109/ARITH.2011.40.
- IEEE, *IEEE Standard for Floating-Point Arithmetic*, IEEE 754-2019.
- D. Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic,"
  *ACM Computing Surveys* 23(1), 1991, doi:10.1145/103162.103163.
- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, second edition, SIAM, 2002.
