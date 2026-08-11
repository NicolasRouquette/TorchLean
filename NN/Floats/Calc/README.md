# Representation-Level Floating-Point Calculation

`NN.Floats.Calc` exposes the integer calculation beneath generic rounded-real semantics. It is the
link between an abstract statement such as `neuralRound rnd x` and a concrete mantissa/exponent
representation.

## Modules

- `Bracket.lean` records whether an exact value is an endpoint or lies below, at, or above a
  bracket midpoint. Its refinement theorems preserve that information when a radix block is split.
- `Operations.lean` aligns mantissas and implements exact negation, absolute value, addition,
  subtraction, and multiplication on `NeuralFloat` representations.
- `Round.lean` turns certified brackets into directed or nearest decisions. Its canonical
  truncation theorem preserves the bracket, selects `neuralCexp`, and proves nearest-even agreement
  with `neuralRound` on the positive path.
- `Arithmetic.lean` packages rounded representation arithmetic. The executable binary32 path remains
  `IEEE32Exec`; finite bridge theorems identify that result with the `FP32` proof model.

The central proved chain is:

1. exact representation arithmetic produces a value and a sufficiently precise bracket;
2. truncation discards low radix digits while refining the midpoint location;
3. the resulting exponent is the canonical exponent selected by the format;
4. nearest-even chooses one of the adjacent representable values;
5. the selected `NeuralFloat` denotes the same real as generic `neuralRound`.

Calculations beginning with an arbitrary Lean real are noncomputable. Executable scalar operations
are connected to these results by the bridges under `NN/Floats/IEEEExec/Bridge/`.

Import the complete calculation layer with:

```lean
import NN.Floats.Calc
```
