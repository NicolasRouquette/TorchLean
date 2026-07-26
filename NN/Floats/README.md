# Floats (`NN/Floats`)

This directory contains TorchLean's floating-point semantics and their supporting numerical theory.

- `FP32/`: a proof oriented, finite float32 model based on rounding over `ℝ`.
- `NeuralFloat/`: generic rounding over `ℝ` (`NeuralRadix`, `NF`, rounding, ULPs, and error bounds).
- `Calc/`: mantissa/exponent calculations, including brackets, rounding decisions, exact
  addition and multiplication, and rounded division and square root.
- `IEEEExec/`: an executable IEEE-754 binary32 kernel (`IEEE32Exec`) plus bridge theorems to `FP32`.
- `Interval/`: interval and enclosure utilities, including quantized intervals over `ℝ` and executable endpoint intervals.
- `Arb/`: an optional external Arb/FLINT oracle adapter (python-flint) for ball and interval
  enclosures. It is not imported by `NN.Floats`.

The generic theory is a native Lean development informed by Flocq's organization and results. It is
not a claim that every Coq module has been translated. TorchLean keeps the parts used by its tensor,
error-analysis, verification, and runtime-refinement developments, and proves executable binary32
behavior separately under `IEEEExec/`.

Import `NN.Floats` for the complete Lean-native numerical library. This import does not include
tensors, models, autograd, CUDA, certificate checkers, or external processes. Import a narrower
umbrella such as `NN.Floats.NeuralFloat`, `NN.Floats.FP32`, `NN.Floats.IEEEExec`, or
`NN.Floats.Interval` when only one layer is needed. The tensor adapter for affine quantization lives
in `NN.Spec.Quantization`, and runtime-approximation results live in
`NN.Proofs.RuntimeApprox.FP32`.

Executable examples that exercise this infrastructure live under `NN/Examples/`.

## Which Layer Should I Cite?

| If your claim is about... | Use this layer |
| --- | --- |
| executable binary32 values inside Lean | `IEEEExec` |
| finite float32-as-rounded-real error bounds | `FP32` |
| precision-parametric rounding and ULP facts | `NeuralFloat` / `NF` |
| the mantissa and exponent produced by a rounding operation | `Calc`, specialized through `FP32` when appropriate |
| interval enclosures with directed endpoints | `Interval` |
| high-precision external enclosure evidence | `Arb`, with the oracle boundary named and its exact rational endpoints enclosed by proved directed rounders |
| CUDA, LibTorch, or Lean runtime `Float` | a runtime bridge or `TRUST_BOUNDARIES.md` assumption |

This distinction is part of the correctness story. A theorem over `FP32` does not become a CUDA
claim until a runtime bridge or trust-boundary statement connects the executable path to that model.

## The Three Float32 Views

TorchLean uses three complementary notions of float32. They have different
strengths, and the bridge files let us move between them without blurring the
trust boundary.

### 1. `IEEE32Exec` (Executable IEEE-754 Binary32)

Folder: `NN/Floats/IEEEExec/`

This is the executable backend. Values are 32-bit encodings, and operations are implemented as pure
Lean code over those bits.

Why we keep it:
- we can execute models inside Lean without relying on opaque runtime float calls,
- we can state and prove theorems about the kernel itself (NaN/Inf propagation, signed zeros, etc.),
- and it provides a stable target for connecting to the mathematical models below.

Where to look:
- `NN/Floats/IEEEExec/Exec32.lean`: the core executable kernel (`IEEE32Exec`),
- `NN/Floats/IEEEExec/Rules/SpecialRules.lean`: NaN/Inf propagation rules,
- `NN/Floats/IEEEExec/Reductions.lean`: reduction semantics (sums/dot products) that match deployment realities,
- `NN/Floats/IEEEExec/Bridge/FP32/Core.lean`: finite bit-pattern representability in `fexp32`,
- `NN/Floats/IEEEExec/Bridge/FP32/Ops.lean`: arithmetic refinement and executable exact subtraction,
- `NN/Floats/IEEEExec/Bridge/FP32/Ulp.lean`: executable ULP exponents and absorption soundness.
  (For executable endpoint-interval arithmetic, see `NN/Floats/Interval/IEEEExec32.lean`.)

### 2. `FP32` (Finite Real Rounded Float32 Model)

Folder: `NN/Floats/FP32/`

This is the mathematical view: float32 computation is modeled as real arithmetic followed by a
float32 rounding operator. It is finite only, with no NaNs or infinities, which keeps many error
analysis statements manageable.

Why we keep it:
- most error-bound theorems are naturally stated over reals + rounding,
- it composes cleanly with `NeuralFloat`/`NF` error bounds,
- it gives a clear semantic target when we want to describe what a float32 computation means.

Where to look:
- `NN/Floats/FP32/Core.lean`: the main definitions,
- `NN/Floats/FP32/Error.lean`: error bounds,
- `NN/Floats/FP32/Sterbenz.lean`: exact subtraction for nearby representable operands,
- `NN/Floats/Interval/FP32.lean`: interval-style enclosure corollaries.

### 3. `NeuralFloat` / `NF` (Generic Format And Error Analysis)

Folder: `NN/Floats/NeuralFloat/`

This layer abstracts over radix, precision, and exponent range. It is where we develop reusable facts
about rounding, ULPs, and relative or absolute error bounds. `FP32` is the
binary32-specialized instance of this generic layer.

Why we keep it:
- we want one set of theorems that applies to multiple precisions and training phases,
- it lets us reuse the same reasoning patterns for float32 and other formats.

Where to look:
- `NN/Floats/NeuralFloat/Format.lean` for representable grids,
- `NN/Floats/NeuralFloat/Rounding.lean` for rounding semantics,
- `NN/Floats/NeuralFloat/Scalar.lean` for `NF`,
- `NN/Floats/Quantization.lean` for scalar affine quantization,
- `NN/Floats/NeuralFloat/Analysis/SterbenzFLT.lean` for exact subtraction with gradual underflow,
- `NN/Floats/NeuralFloat/Analysis.lean` and `NN/Floats/NeuralFloat/Error.lean` for numerical bounds.

## How The Bridge Layers Fit Together

`IEEE32Exec` is executable; `FP32` and `NF` are aimed at proofs. `Calc` supplies representation-level
arithmetic between them. Given a bracket for an exact value, it records the value's position between
adjacent representable numbers, makes the rounding-mode decision, and returns a `NeuralFloat`
mantissa and exponent. Theorems in `FP32/Core.lean` identify the real value of that representation
with `fp32Round`. Thus `Calc` is not another floating-point semantics: it exposes the integer
calculation underlying the rounding operation used by the proof model. Calculations starting from an
arbitrary Lean real remain noncomputable; executable runs use `IEEE32Exec` or a named runtime backend.

The bridge files then connect the executable bit model to that result:

- `NN/Floats/IEEEExec/Bridge/FP32.lean`: core refinement theorems on the **finite/no-overflow** path:
  $\operatorname{toReal}(\operatorname{op}_{\mathrm{exec}}(\ldots))=\operatorname{fp32Round}(\operatorname{op}_{\mathrm{real}}(\ldots))$.
- `NN/Floats/IEEEExec/Bridge/FP32Total.lean`: total wrappers that combine NaN/Inf propagation rules
  with the finite refinement theorems, phrased using `toReal?`.
- `NN/Floats/IEEEExec/Bridge/Expressions.lean`: a compact scalar AST + a whole-expression refinement theorem.
- `NN/Floats/IEEEExec/Bridge/ERealTotal.lean`: an `EReal`-valued semantics that distinguishes
  $+\infty$ and $-\infty$.
- `NN/Floats/IEEEExec/Bridge/RuntimeFloat32.lean`: an assumption-based bridge from Lean's runtime
  `Init.Float32` to `IEEE32Exec`. Exact bit agreement covers finite inputs and finite results;
  classification is stated for all values because runtime NaN payload propagation is opaque.

The standard rounding choices are available through `NeuralRoundingMode`: nearest-even,
toward-zero, toward-positive, and toward-negative. The lower-level function-valued API remains the
right one for round-to-odd and other proof-specific policies. Directed binary32 addition,
subtraction, multiplication, division, fused multiply-add, and square root have executable lower
and upper operations with semantic enclosure theorems.

The executable arithmetic also has status-bearing wrappers such as `addWithStatus`,
`divWithStatus`, and `sqrtWithStatus`. They return the same binary32 value together with the five
IEEE exception indicators. Underflow uses tininess after rounding and is raised only for an inexact
tiny result; this policy is part of the Lean definition rather than inherited from the host.

`neuralRoundAtScale` supplies the common fixed-grid operation used by affine quantization and
fixed-point arguments and requires a proof that the grid step is positive. The round-to-odd theorem
in `NeuralFloat/Rounding/Odd.lean` shows that a
sufficiently fine binary intermediate prevents nearest-even double rounding on an arbitrary
coarser grid.

`NN/Examples/DeepDives/Floats/EffectiveRounding.lean` follows one shaped tensor addition through the
rounded-real and executable IEEE paths. Its theorems expose the computed mantissa/exponent result on
both sides of the finite bridge.

For native CUDA and LibTorch paths, the bridge is not in this folder by default. Those backends
are runtime providers. A proof layer float claim should say which Lean model it uses and where the
runtime/backend agreement assumption is discharged or documented.

Background references that informed the design:
- IEEE 754-2019: https://doi.org/10.1109/IEEESTD.2019.8766229
- Goldberg (1991): https://doi.org/10.1145/103162.103163
- Higham (2002): *Accuracy and Stability of Numerical Algorithms* (2nd ed.), ISBN 0-89871-521-0
- Flocq (Boldo–Melquiond, 2011): https://doi.org/10.1109/ARITH.2011.40
