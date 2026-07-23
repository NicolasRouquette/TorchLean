# Formats

This folder describes which real numbers belong to a generic radix/exponent grid.  It does not pick
a rounding mode and it does not claim that a runtime backend implements the grid.

- `Magnitude.lean` locates a nonzero real between adjacent radix powers.
- `Digits.lean` relates integer digit counts to those magnitude bounds.
- `Formats.lean` defines fixed-exponent (`FIX`), unbounded (`FLX`), and bounded-precision (`FLT`)
  formats. `Special/FTZ.lean` defines abrupt-underflow exponent selection.
- `Generic.lean` defines canonical representability and grid inclusion.
- `Theorems.lean` proves structural properties shared by arithmetic and error analysis.

This separation is important for quantization.  A fixed-point or arbitrary-float quantizer first
declares its representable grid here; rounding and saturation are separate policy choices.

FLX, FLT, and FTZ precision is required to be positive. Their explicit format predicates reject
nonpositive integers, and their `NeuralValidExp` witnesses require a positivity proof. At a parsed
or otherwise untrusted configuration boundary, use `NeuralFormatPrecision.ofInt?`; a checked value
provides `flxExp`, `fltExp`, and `ftzExp` selectors with validity instances. The raw integer formulas
remain available for symbolic proofs, but a negative integer is never treated as a positive digit
count through `natAbs`.

The principal reference is Boldo and Melquiond's Flocq library and paper (IEEE ARITH 2011,
doi:10.1109/ARITH.2011.40).  Concrete binary interchange formats are governed by IEEE 754-2019.
