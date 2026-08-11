import VersoManual

open Verso.Genre Manual

#doc (Manual) "Floating-Point Semantics" =>
%%%
tag := "floats"
%%%

Neural-network papers write equations over real numbers, while hardware evaluates a sequence of
finite operations. Rounding, overflow, underflow, subnormals, infinities, and NaNs can therefore
change the value computed by a model. TorchLean gives these effects explicit mathematical and
executable meanings instead of treating them as details of the runtime.

# Using The Numerical Library By Itself

The floating-point development is a reusable numerical library inside the TorchLean package. A
downstream Lean project can depend on TorchLean and import only:

```
import NN.Floats
open TorchLean.Floats
```

That import does not bring in tensors, model definitions, autograd, CUDA, certificate checkers, or
external numerical tools. It provides the generic format and rounding theory, the binary32-precision
rounded-real model, executable IEEE binary32 arithmetic, proved interval rounders, and scalar affine
quantization. Narrower imports such as `NN.Floats.NeuralFloat`, `NN.Floats.FP32`,
`NN.Floats.IEEEExec`, and `NN.Floats.Interval` are useful when a file needs only one layer.

Connections to the rest of TorchLean point in the other direction. `NN.Spec.Quantization` lifts the
scalar quantizer to shape-indexed tensors. `NN.Proofs.RuntimeApprox.FP32` connects rounded-real FP32
semantics to runtime-approximation proofs. Arb-based transcendental checks are optional and require
the explicit import `NN.Floats.Arb`. The numerical core therefore remains usable without adopting
TorchLean's model or runtime APIs.

# Begin With A Calculation

Take the three exact real numbers

$$`a=2^{24},\qquad b=1,\qquad c=-2^{24}`.

Over the reals, both parenthesizations are equal:

$$`(a+b)+c=a+(b+c)=1`.

Binary32 has 24 bits of significand precision. Around $`2^{24}`, adjacent representable numbers are
two units apart. The exact value $`2^{24}+1` sits halfway between them, and ties-to-even rounds it
back to $`2^{24}`. Therefore

$$`
\operatorname{fl}(\operatorname{fl}(a+b)+c)=0,
\qquad
\operatorname{fl}(a+\operatorname{fl}(b+c))=1.
`

TorchLean can run this example from the bit patterns themselves:

```
import NN.Floats

open TorchLean.Floats.IEEE754

def a : IEEE32Exec := IEEE32Exec.ofBits 0x4b800000  --  2^24
def b : IEEE32Exec := IEEE32Exec.ofBits 0x3f800000  --  1
def c : IEEE32Exec := IEEE32Exec.ofBits 0xcb800000  -- -2^24

#eval (IEEE32Exec.add (IEEE32Exec.add a b) c).bits
-- 0          (0x00000000, +0)

#eval (IEEE32Exec.add a (IEEE32Exec.add b c)).bits
-- 1065353216 (0x3f800000, 1)
```

Save the snippet as `FloatPlayground.lean` in the repository root and run:

```
lake env lean FloatPlayground.lean
```

There are two quick variations worth trying. Replace $`b` by $`2`; both parenthesizations
return $`2`, because $`2^{24}+2` is representable. Replace it by the binary32 value $`0.5`; both
paths lose the small addend. The interesting case $`b=1` sits exactly at the tie between
neighboring values.

This small calculation already raises three different questions.

First, *why* did the first addition round down? That is a theorem about the spacing of a binary
format and the nearest-even rule. Second, *which bits* should the complete operation return? That
requires an executable account of binary32, including signs, exponents, subnormals, infinities, and
NaNs. Third, will a CUDA reduction use either of these parenthesizations, a balanced tree, or a
fused instruction? That is a question about the selected backend.

Trying to answer all three with one definition makes every proof carry irrelevant machinery.
TorchLean instead builds a small ladder:

```
format and rounding theory
        ↓
rounded-real arithmetic
        ↓
binary32-precision rounded-real specialization
        ↓
executable IEEE bit patterns
        ↓
CPU, CUDA, and external providers
```

The rest of the chapter constructs this ladder from the bottom of the mathematics upward.

# Where `NeuralFloat` Came From

TorchLean's generic floating-point theory was inspired by
[Flocq](https://flocq.gitlabpages.inria.fr/), the mature floating-point library developed in Rocq
(formerly Coq) by Sylvie Boldo, Guillaume Melquiond, and contributors. Flocq's central design
decision is wonderfully simple: a floating-point *format* and a *rounding rule* are different
objects.

A format tells us which real numbers are available. A rounding rule chooses one of those numbers.
The same binary32 format can round downward, upward, toward zero, or to nearest-even. Conversely, the
same nearest-even idea can be used with binary16, binary32, a fixed-point grid, or an experimental
low-precision format. Once those choices are separated, theorems about monotonicity, exactness,
neighboring values, and ULP error can be reused.

TorchLean follows that organization but is not a port of Flocq. The definitions and proofs are
native Lean. The library concentrates on the theory needed by tensor semantics, interval
enclosures, quantization, and runtime approximation. Flocq remains broader in areas such as its
effective-operation infrastructure and its collection of verified floating-point algorithms.
TorchLean adds a separate executable binary32 development and graph-level numerical machinery aimed
at machine learning.

The correspondence is useful when reading either library:

:::table +header
*
  * Mathematical role
  * Flocq vocabulary
  * TorchLean vocabulary
*
  * radix power $`\beta^e`
  * `bpow`
  * `neuralBpow`
*
  * mantissa/exponent value
  * `F2R`-style representation
  * `NeuralFloat`, `neuralToReal`
*
  * exponent policy
  * `fexp`
  * `fexp` with `NeuralValidExp`
*
  * representable grid
  * `generic_format`
  * `neuralGenericFormat`
*
  * fixed, unbounded, gradual formats
  * `FIX`, `FLX`, `FLT`
  * `FIXExp`, `FLXExp`, `FLTExp`
*
  * rounding to the grid
  * `round`
  * `neuralRound`
:::

The `neural` prefix is not claiming that floating-point arithmetic is unique to neural networks.
It marks TorchLean's generic floating layer and avoids colliding with Lean's host `Float`. The
theorems themselves are ordinary numerical analysis and can be used independently of a model.

# The Exact Carrier Comes First

Before imposing a precision, define a radix-$`\beta` value by an integer mantissa and an integer
exponent:

$$`\operatorname{value}_{\beta}(m,e)=m\beta^e`.

This is the structure `NeuralFloat β`:

```
structure NeuralFloat (β : NeuralRadix) where
  mantissa : ℤ
  exponent : ℤ
```

For example,

```
open TorchLean.Floats

def threeQuarters : NeuralFloat binaryRadix :=
  { mantissa := 3, exponent := -2 }
```

denotes

$$`\operatorname{neuralToReal}(\texttt{threeQuarters})
  =3\cdot2^{-2}=\frac34`.

There is intentionally no field saying “24 bits of precision.” The pair $`(3,-2)` and the pair
$`(6,-3)` denote the same real number. A raw mantissa/exponent pair is a representation, not yet a
machine format. Keeping it general lets later proofs normalize representations and reuse the same
carrier at different precisions.

A format is added as a predicate on the real value. Informally,

$$`\operatorname{neuralGenericFormat}(\beta,f_{\rm exp},x)`

means that after scaling $`x` by the exponent selected by `fexp`, the resulting mantissa is an
integer. The exponent policy is where precision and underflow enter.

# Formats As Exponent Policies

The central format parameter is an exponent function

$$`f_{\mathrm{exp}}:\mathbb Z\to\mathbb Z`.

For a nonzero real $`x`, its magnitude identifies the power of $`\beta` immediately above
$`|x|`. Applying `fexp` to that magnitude gives the canonical exponent at which the mantissa must
be integral. TorchLean's `neuralGenericFormat β fexp x` says precisely that $`x` lies on this grid.

Three standard policies explain most uses:

- `FIXExp emin` always returns `emin`. This is a fixed-point grid with constant spacing
  $`\beta^{e_{\min}}`.
- `FLXExp prec` returns `e - prec`. This models a precision of `prec` radix digits with no lower
  exponent bound.
- `FLTExp emin prec` returns `max (e - prec) emin`. This is the gradual-underflow format: normal
  values receive `prec` digits, while values near zero stay on the fixed subnormal grid
  $`\beta^{e_{\min}}`.

Precision must be positive. The raw integer formulas remain useful inside symbolic theorem
statements, where positivity is carried as a hypothesis. Code reading a format configuration uses
`NeuralFormatPrecision.ofNat?` or `NeuralFormatPrecision.ofInt?`; the checked value supplies valid
FLX, FLT, and FTZ exponent selectors. Thus zero and negative inputs are rejected instead of being
silently reinterpreted through an absolute value.

It helps to see this on a toy system. Take radix two, precision three, and minimum exponent $`-4`.
Between $`1` and $`2`, three-bit numbers are spaced by $`1/4`:

$$`1,\quad 1.25,\quad 1.5,\quad 1.75,\quad 2`.

Near zero, `FLTExp (-4) 3` stops decreasing the exponent, so the spacing becomes the constant
subnormal step $`2^{-4}=1/16`. `FLXExp 3` would continue creating smaller normal scales forever;
`FIXExp (-4)` would use the $`1/16` grid everywhere. These three policies are not unrelated format
implementations. They are three choices for the same exponent function interface.

Binary32 uses radix two, precision 24, and the least subnormal exponent $`-149`, so TorchLean defines

```
def fexp32 : ℤ → ℤ :=
  FLTExp (-149) 24
```

The choice $`-149` is not the minimum *normal* exponent. Binary32 normal numbers begin at
$`2^{-126}`, but the 23 fraction bits extend the gradual-underflow grid down to $`2^{-149}`.
Encoding that fact in
`FLTExp` is what allows the same representability predicate to cover normal and subnormal finite
values.

The benefit of the generic definition is visible in theorem statements. Monotonicity of rounding,
the half-ULP nearest-rounding bound, and fixed-grid exactness do not need separate proofs for every
precision. A binary32 theorem specializes the generic result by supplying `binaryRadix`, `fexp32`,
and nearest-even rounding.

# Rounding Is A Separate Choice

A format answers "which values are available?" Rounding answers "which available value should
replace this exact real?" These are independent decisions. The same binary format supports rounding
toward negative infinity, toward positive infinity, toward zero, and to nearest with ties to even.

Conceptually, TorchLean computes

$$`\operatorname{round}_{\beta,f,r}(x)
   = \beta^{e}\,r(x\beta^{-e}),
   \qquad e=f(\operatorname{mag}_{\beta}(x))`,

where $`r:\mathbb R\to\mathbb Z` rounds the scaled mantissa to an integer. The type class `NeuralValidRnd r`
records the order properties required of that integer rounder. `NeuralRoundingMode` packages the
standard choices used by APIs.

Return to the toy three-bit format and round $`1.375`. Its canonical exponent in this binade is
$`-2`, so the scaled mantissa is

$$`1.375\cdot2^2=5.5`.

Rounding downward chooses mantissa $`5`, giving $`1.25`. Rounding upward chooses $`6`, giving
$`1.5`. Nearest-even also chooses $`6`, because the two candidates are equally distant and $`6` is
even. The format selected the scale $`2^{-2}`; the rounding mode selected the integer mantissa.

The separation gives us reusable theorems:

- rounding a representable value leaves it unchanged;
- every rounded value belongs to the format;
- directed rounding returns the greatest representable value below, or least one above, the input;
- nearest rounding has error at most half an ULP;
- rounding is monotone;
- an initial round-to-odd can prevent a later nearest-even double-rounding error.

The abstract definition is excellent for proofs: it says what rounding means without committing to
an implementation. `NN.Floats.Calc` supplies the constructive middle layer. It brackets an exact
value between representable neighbors, applies the rounding decision, and returns concrete
mantissa/exponent data. `FP32.round_eq_computed` proves that this calculation agrees with the
abstract binary32 rounder.

# `NF`: Arithmetic In A Declared Format

The generic theorems above discuss individual real values and rounding functions. Neural-network
proofs need an object on which `+`, `*`, division, activations, and reductions can be written without
repeating all format parameters. That object is

```
NF β fexp rnd
```

An `NF` value carries a real number, while its type fixes the radix, format, and rounding rule.
Primitive arithmetic computes the exact real operation and rounds the result back to the declared
grid. Schematically,

$$`\operatorname{NF.add}(a,b)
  = \operatorname{round}_{\beta,f,r}(a_{\mathbb R}+b_{\mathbb R})`.

Why store a real instead of the mantissa/exponent pair directly? Error proofs constantly compare an
ideal value with a rounded one. Keeping the real projection immediate makes expressions such as

$$`|\operatorname{NF.toReal}(\widehat x)-x|`

pleasant to state, while the type still fixes the format and rounding rule. `NF.ofReal` rounds a
real into the format. The low-level constructor is available for approximation relations, so
theorems that need a genuine grid value ask for `NF.IsRepresentable`.

# `FP32`: The Binary32-Precision Rounded-Real Specialization

`FP32` is the specialization

```
abbrev FP32 : Type :=
  NF binaryRadix fexp32 rnd32
```

where `rnd32` is nearest-even. It is the right model for a theorem whose intended reading is
"perform this real operation and round it at binary32 precision with gradual underflow." Its
exponent policy has no upper cutoff, so it is not the finite set of IEEE bit patterns. The aliases
`round32`, `ulp32`, and `eps32` expose the rounder, local spacing, and half-ULP scale directly over
`ℝ`.

That separation is deliberate. `FP32` omits:

- NaN and positive or negative infinity;
- signed zero and NaN payloads;
- overflow to infinity;
- IEEE exception flags.

That makes `FP32` the convenient layer for ordinary forward-error analysis. A theorem about a linear
layer can compare the exact dot product with a sequence of rounded operations without splitting
every line into finite, infinite, and NaN cases. When exceptional values matter, we move down one
level to `IEEE32Exec`.

# Total Operations Still Need Domain Hypotheses

`NF` and `FP32` use Lean's total real operations internally. Real division by zero and square root
or logarithm outside their usual analytical domains therefore do not produce IEEE exceptions, and
the rounded-real format has no upper exponent bound that models overflow to infinity. These choices
keep algebraic definitions total; they do not turn invalid-domain calculations into faithful IEEE
executions.

Accordingly, an IEEE correspondence theorem states finiteness, nonzero-denominator, domain, and
no-overflow hypotheses where needed. At the bit level, `IEEE32Exec.toReal?` returns `none` for NaN
and infinity. The convenience projection `toReal` maps those encodings to zero and should be used
only under a finiteness hypothesis; interval work that includes infinities uses the extended-real
semantics instead.

# `IEEE32Exec`: Bits And Exceptional Behavior

`IEEE32Exec` stores a raw `UInt32` bit pattern. Its classifiers distinguish zero, subnormal, normal,
infinite, and NaN encodings. Core arithmetic is implemented in Lean using integer and dyadic
calculations, so it can be evaluated without delegating the operation to the host's floating-point
instruction.

For example, these are the binary32 encodings of $`1`, $`2^{-25}`, and their nearest-even sum:

```
open TorchLean.Floats.IEEE754

def oneBits : IEEE32Exec :=
  IEEE32Exec.ofBits 0x3f800000

def tinyBits : IEEE32Exec :=
  IEEE32Exec.ofBits 0x33000000

#eval (IEEE32Exec.add oneBits tinyBits).bits
-- 1065353216, hexadecimal 0x3f800000
```

At $`1`, one ULP is $`2^{-23}`; half an ULP is $`2^{-24}`. The exact addend $`2^{-25}` is smaller
than half an ULP, so nearest-even returns $`1`. The executable `absorbs` predicate records precisely
this event:
the accumulator is unchanged even though the exact real increment is positive.

Change `tinyBits` to `0x33800000`, the encoding of exactly half an ULP at $`1`. The sum still rounds
to $`1` because its significand is even. Change it once more to `0x34000000`, one full ULP, and the
result advances to `0x3f800001`. These three evaluations are the bit-level version of the spacing
picture developed above.

The value-only operations have status-bearing variants. `IEEEOutcome` pairs the result bits with an
`IEEEStatus` containing the invalid, divide-by-zero, overflow, underflow, and inexact flags supported
by the model. Underflow follows the documented tininess-after-rounding policy and is raised only for
an inexact tiny result.

For example:

```
#eval IEEE32Exec.divWithStatus IEEE32Exec.posOne IEEE32Exec.posZero
```

returns positive infinity together with `divideByZero := true`. In contrast, $`0/0` returns the
canonical NaN and sets `invalid := true`. The status is derived from the same exact dyadic or
rational intermediate used by the arithmetic operation; no host floating-point instruction is
called to guess the flag.

Transcendentals have a different status from basic arithmetic because IEEE 754 does not prescribe
one correctly rounded bit pattern for every elementary function. TorchLean provides deterministic
wrappers and approximation contracts; the runtime chapter explains how a concrete `libm`,
`libdevice`, or LibTorch implementation can be related to them.

Deterministic is not synonymous with correctly rounded. The executable transcendental kernels are
algorithms with explicit special-value behavior; a numerical claim needs a range-specific error or
interval contract for the particular function. The small Taylor bounds in the rules library do not
by themselves certify every executable `sin`, `cos`, or `tanh` input.

# Lean `Float32`

Lean gives core `Float32` operations a kernel-visible semantics through `Float32.Model`. The model
defines bit conversion, classification, comparison, addition, subtraction, multiplication,
division, negation, absolute value, and square root. Compiled programs may replace these logical
definitions with native instructions through `@[extern]`; native conformance is recorded separately
for each runtime provider.

TorchLean compares `Float32.Model` with `IEEE32Exec`, its independent raw-bit implementation. The
classification theorem has no runtime assumption:

```
open TorchLean.Floats.IEEE754
open TorchLean.Floats.IEEE754.Float32Bridge

example (a : Float32) :
    Float32.isFinite a = IEEE32Exec.isFinite (toIEEE32Exec a) := by
  exact float32_isFinite_eq_ieee32 a
```

The theorem ranges over every canonical binary32 value. Addition has a corresponding arithmetic
theorem:

```
example (a b : Float32) :
    toIEEE32Exec (a + b) =
      canonicalize (IEEE32Exec.add (toIEEE32Exec a) (toIEEE32Exec b)) := by
  exact toIEEE32Exec_add a b
```

The proof covers NaNs, signed infinities and zeros, subnormals, cancellation, normal rounding,
underflow, and overflow. Subtraction follows from addition and the sign-bit negation bridge.
Multiplication, division, square root, negation, and absolute value have analogous theorems.

`Float32.Model` is a canonical binary32 representation. It stores a `UInt32` together with a proof
that every NaN uses Lean's chosen canonical encoding. Addition, subtraction, multiplication,
division, negation, absolute value, square root, comparisons, and classification operate by
unpacking that value, computing in Lean's `UnpackedFloat` model, and repacking. Transcendental
functions such as `sin`, `exp`, and `log` remain opaque and need separate contracts.

`IEEE32Exec` accepts every raw binary32 bit pattern, retains deterministic NaN payload information,
reports IEEE exception status, and connects finite executions to the rounded-real and interval
developments. `Float32.Model` uses one canonical NaN representation. The bridge therefore
canonicalizes arithmetic results before comparing their bits.

The bridge has two parts:

```
Float32 operation
  = Float32.Model operation

Float32.Model operation
  = canonicalize (TorchLean IEEE32Exec operation)
                                             -- independent algorithm-equivalence obligation
```

TorchLean proves both equalities for the modeled core operations. The second proof is not a test or
an assumption about an opaque runtime call: it compares two pure Lean algorithms. Canonicalization
appears because Lean stores one NaN bit pattern while `IEEE32Exec` retains payload and sign bits.

The resulting interfaces have three separate scopes:

:::table +header
*
  * Claim
  * Current status
*
  * A Lean `Float32` core operation denotes its `Float32.Model` operation.
  * Defined by Lean.
*
  * `Float32.Model` and `IEEE32Exec` classify canonical bit patterns identically.
  * Proved for finiteness, infinity, and NaN.
*
  * `Float32.Model` and `IEEE32Exec` compute the same canonical result.
  * Proved for comparison, addition, subtraction, multiplication, division, square root, negation,
    and absolute value.
*
  * Compiled CPU, CUDA, or library code computes the logical result.
  * A provider-specific backend contract, checked separately.
:::

The bridge makes `IEEE32Exec` error, interval, and reduction results available to logical
`Float32` proofs. Compiled execution uses the contract attached to its runtime provider.

# Following One Addition Through The Layers

The addition above has four useful interpretations.

First, the ideal real expression is

$$`z = 1 + 2^{-25}`.

Nothing is lost in $`\mathbb R`. Second, $`\operatorname{round}_{32}(z)=1`, justified by the binary32
format and nearest-even rounding theory. Third, constructing `FP32` operands and adding them applies
that same `round32` policy to the exact sum. Fourth, `IEEE32Exec.add` runs the bit-level algorithm
and returns `0x3f800000`.

The bridge theorem supplies the nontrivial connection:

$$`\operatorname{toReal}
    (\operatorname{IEEE32Exec.add}(a,b))
  = \operatorname{round32}
    (\operatorname{toReal}(a)+\operatorname{toReal}(b))`,

under the theorem's finite-path and result hypotheses. The ULP bridge identifies the exponent
returned by executable `ulpExp?` with `ulp32` in the rounded-real model. The absorption theorem then
states that a successful finite executable absorption check implies the corresponding `round32`
addition leaves the left operand unchanged.

The representations line up as follows:

```
exact real expression
  -> generic format and rounding theorem
  -> FP32 finite specialization
  -> IEEE32Exec bit-level operation
  <-> Lean Float32.Model
  -> compiled CPU, CUDA, or external provider
```

The mathematical layers and both bit-level models are Lean definitions. The last arrow is supplied
by a native backend contract and its validation evidence.

# Exact Subtraction: A More Interesting Example

Sterbenz's lemma says that subtraction can be exact even in floating-point arithmetic. If positive,
representable $`x` and $`y` are within a factor of two,

$$`\frac{y}{2}\leq x\leq 2y`,

then $`x-y` is representable in the same format. TorchLean first proves the fixed-grid and
unbounded-exponent results, then extends the argument to the gradual-underflow `FLT` format. The
extension matters near zero: a proof only about normal values would miss subtraction across the
normal/subnormal boundary.

`FP32.sub_exact_of_sterbenz` specializes the result to finite rounded-real binary32.
`IEEE32Exec.toReal_sub_eq_sub_of_sterbenz` goes further: finite bit patterns are decoded, proved
representable on the `fexp32` grid, passed through the rounded-real Sterbenz theorem, and related
back to executable subtraction. The theorem derives result finiteness rather than quietly assuming
it.

This example shows why the representation stack is useful. The generic proof captures the
mathematics once; the binary32 specialization supplies the machine format; the executable bridge
connects actual bit patterns to that theorem.

# Tensors, Reductions, And Quantization

Pointwise tensor operations lift the scalar semantics coordinate by coordinate. A tensor theorem
over `NF` therefore inherits the declared rounding at each scalar operation.

Reductions add another choice: order. A left fold, balanced tree, warp reduction, atomic
accumulation, and library matrix multiplication may all use binary32 addition and still disagree.
Contraction adds the same issue for $`ab+c`: FMA rounds once, while separate multiplication and
addition round twice. Backend capsules record these policies so the numerical analysis can select
the matching expression.

Affine quantization reuses the generic rounding layer rather than inventing another scalar
semantics. An `AffineQuantizer` has a positive scale $`s`, zero point $`z`, and integer code bounds:

$$`q(x)=\operatorname{clamp}
  \left(\operatorname{round}\left(\frac{x}{s}\right)+z\right)`,

$$`\widehat{x}(q)=s(q-z)`.

The scalar definition and its arithmetic theorems live in `NN.Floats.Quantization`. The separate
`NN.Spec.Quantization` adapter applies the same equations at every coordinate of a shape-indexed
tensor. Together they prove code range, monotonicity, in-range code round trips, and the half-step
reconstruction bound when saturation is inactive. Later runtime work can add packed int8 or int4
storage without changing these scalar theorems.

# From Lean Semantics To A Training Run

Lean `Float32`, C and CUDA `float`, cuBLAS reductions, and LibTorch tensors can all appear in an
execution path. TorchLean proves that `Float32.Model` agrees with its independent executable
reference for classification, comparison, addition, subtraction, multiplication, division, square
root, negation, and absolute value. Native instructions are described by provider contracts.

Native providers follow the same pattern. A kernel capsule names:

- the operation and device;
- its forward and backward providers;
- shape and layout requirements;
- rounding, contraction, subnormal, and reduction policies;
- the evidence level attached to the numerical contract.

The capsule makes the provider's numerical policies available to graph-level error analysis. A
provider may come with a proved bridge, a checked guard, parity evidence, or an explicit external
assumption. Those evidence levels were defined in the introduction, so later reports can name
the one attached to each operation.

# Choosing The Right Representation

Use the smallest layer that states the claim accurately:

:::table +header
*
  * Question
  * Representation
*
  * Which values lie on a radix/precision grid?
  * `NeuralFloat` formats
*
  * How does a declared format round exact real arithmetic?
  * `NF`
*
  * What is the binary32-precision rounded-real error?
  * `FP32`
*
  * What bits and IEEE exceptional cases result?
  * `IEEE32Exec`
*
  * What does a Lean `Float32` core operation mean in the logic?
  * `Float32.Model`
*
  * Does an interval enclose an operation?
  * directed `IEEE32Exec` or proved interval rounders
*
  * What did CPU, CUDA, or LibTorch execute?
  * runtime result plus an explicit bridge or boundary
:::

The table is also a useful debugging guide. If a theorem is cluttered with NaN cases while the
algorithm assumes a finite path, move up to `FP32`. If a proof needs the sign of zero or an exception
flag, move down to `IEEE32Exec`. If two GPU runs disagree, inspect reduction and contraction policy
before blaming the real-valued model.

# References

- Lean FRO,
  [`Float32.Model` source](https://github.com/leanprover/lean4/blob/v4.33.0/src/lean/Init/Data/Float/Model/Float32.lean).
- The Lean Language Reference,
  [Floating-Point Numbers](https://lean-lang.org/doc/reference/latest/Basic-Types/Floating-Point-Numbers/).
- Sylvie Boldo and Guillaume Melquiond,
  [“Flocq: A Unified Library for Proving Floating-Point Algorithms in
  Coq”](https://doi.org/10.1109/ARITH.2011.40), IEEE ARITH 2011.
- [Flocq in a Nutshell](https://flocq.gitlabpages.inria.fr/theos.html), an overview of formats,
  rounding, ULP results, double rounding, and effective operators.
- IEEE Computer Society,
  [IEEE Standard for Floating-Point Arithmetic,
  IEEE 754-2019](https://doi.org/10.1109/IEEESTD.2019.8766229).
- Jean-Michel Muller et al.,
  [*Handbook of Floating-Point Arithmetic*](https://doi.org/10.1007/978-3-319-76526-6),
  second edition.
- Nicholas J. Higham,
  [*Accuracy and Stability of Numerical Algorithms*](https://doi.org/10.1137/1.9780898718027),
  second edition.
- Pat H. Sterbenz, *Floating-Point Computation*, Prentice-Hall, 1974.
