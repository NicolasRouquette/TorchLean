# `NN/Floats/IEEEExec`: Executable IEEE-754 Float32 Semantics

This directory contains TorchLean's Lean defined executable model of IEEE-754 binary32. It also holds
bridge theorems that connect runtime execution to proof oriented rounding models over `ℝ`.

## What Lives Here

- `Exec32.lean` and `Exec32/`: the executable kernel (`IEEE32Exec`), bit layout, arithmetic,
  comparison, directed rounding, dyadic conversion, exception-status outcomes, instances, and
  deterministic transcendental wrappers.
- `Encoding/`: interpretation of bit patterns as exact dyadics and reals, together with sign-bit
  facts.
- `Rounding/`: integer and rational lemmas used to prove the executable rounding algorithms.
- `Semantics/`: real and `EReal` interpretations, error bounds, and operation sandwiches.
- `Rules/`: proved rules for special values and the deterministic transcendental wrappers.
- `Reductions.lean`: reduction semantics for sums/dot products.
- `Bridge/`: refinement from executable bits to rounded-real, extended-real, expression, and
  runtime models.

## Interval / Directed Rounding Semantics

If your goal is interval arithmetic over float32, this folder provides the directed rounding
kernels and their soundness proofs:

- `DirectedRoundingSoundness/`: soundness of directed dyadic and rational rounding and the
  endpoint operations for addition, multiplication, fused multiply-add, division, and square
  root. The statements use `EReal`, so endpoint overflow to $\pm\infty$ remains a valid enclosure.
- `Semantics/MinMaxERealSoundness.lean`: order lemmas used by endpoint min/max rules.

The interval *API layer* that uses these results lives in `NN/Floats/Interval/`.

## Bridge To Proof Oriented Float Models

TorchLean keeps two float32 views:

- Executable (`IEEE32Exec`): what we can run inside Lean, with bit level semantics.
- Proof oriented (`FP32`): round on real semantics used for numerical error envelopes.

Bridge files connect these on the finite, no overflow path:

- `Bridge/FP32.lean` and `Bridge/FP32/`: per-operation refinement lemmas on the finite branch,
  including dyadic/rational rounding infrastructure, exact subtraction under Sterbenz's
  hypotheses, and executable ULP exponents. The ULP query succeeds exactly for finite bit patterns
  and returns `none` exactly for NaNs and infinities.
- `Bridge/Expressions.lean`: expression-level refinement that composes operation lemmas once.
- `Bridge/FP32Total.lean`: packages finite refinement and proved special-value rules using
  `toReal?`.
- `Bridge/ERealTotal.lean`: an `EReal` interpretation that distinguishes $+\infty$ and $-\infty$ while
  representing NaN as `none`.
- `Bridge/RuntimeFloat32.lean`: an assumption-based interface relating Lean runtime `Float32` to
  `IEEE32Exec`. Exact bit agreement is restricted to finite inputs and finite results. Classification
  laws cover all values, while NaN payload propagation remains outside the contract because the
  runtime is opaque to the kernel.

## How To Use It

Use `IEEE32Exec` when the object you need is executable binary32 inside Lean. Use `FP32` when the
theorem should be a rounded-real error statement. Use runtime `Float32` or CUDA only with an
explicit bridge or trust-boundary statement.

The value-only operations (`add`, `mul`, `div`, `fma`, and `sqrt`) are accompanied by
status-bearing operations (`addWithStatus`, `mulWithStatus`, `divWithStatus`, `fmaWithStatus`, and
`sqrtWithStatus`). An `IEEEOutcome` contains the computed bits and an `IEEEStatus` containing the
invalid, divide-by-zero, overflow, underflow, and inexact indicators. Tininess is detected after
rounding, and underflow implies inexactness by theorem. `Rules/SpecialRules.lean` proves the main
special-value status cases.

The useful proof shape is:

```text
IEEE32Exec operation
  -> finite/no-overflow bridge
  -> FP32 rounded-real statement
  -> real-valued error envelope or interval claim
```

## Trust boundary (important)

IEEE-754 does not specify bit level results for transcendentals (`expf`, `logf`, ...), and platform
libm implementations can differ. TorchLean therefore treats:

- core arithmetic (add/mul/div/sqrt, specials) as the proved executable kernel, and
- transcendentals as deterministic but not IEEE specified unless you use a separate rigorous
  backend (e.g. the Arb oracle under `NN/Floats/Arb/` and the interval glue in `NN/Floats/Interval/`).

`Rules/TranscendentalRules.lean` defines `UnaryApproximationContract` for connecting a finite
executable implementation to a real function on a stated domain with a stated error tolerance.
This contract does not assert that the current deterministic wrappers satisfy a particular
accuracy bound; such a theorem needs evidence for the chosen implementation.
