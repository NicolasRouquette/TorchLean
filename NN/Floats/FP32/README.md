# `NN.Floats.FP32`

`FP32` is TorchLean's proof-oriented rounded-real model of float32 arithmetic. It is used for
theorem statements and compositional error arguments when the proof should look like finite
precision arithmetic but does not need NaNs, infinities, signed zero, or bit-level payloads.

Use this layer when you want clean statements such as:

```text
real spec value
  -> rounded FP32 operation
  -> bounded error or interval enclosure
```

Do not use it as the executable IEEE-754 model. For bit-level binary32 behavior, including special
values, use `TorchLean.Floats.IEEE754.IEEE32Exec` from `NN/Floats/IEEEExec/`.

## Files

- `Core.lean`: canonical binary32 configuration (`fexp32`, `rnd32`) and the `FP32` type alias.
- `Notation.lean`: aliases over `ℝ` for the model, including `round32`, `ulp32`, and `eps32`.
- `Error.lean`: per-operation absolute error bounds.
- `NN/Proofs/RuntimeApprox/FP32.lean`: error bounds restated through the generic tolerance relation
  `≈[t]`.
- `Sterbenz.lean`: exact subtraction for nearby representable binary32 values.

Interval enclosures live in `NN/Floats/Interval/FP32.lean` and are imported by `NN.Floats.FP32`.

## Relationship To Runtime

The runtime may use Lean `Float`, C/CUDA `float`, or external kernels. A theorem over `FP32` becomes
a runtime claim only after an explicit bridge says the executable path agrees with, or is bounded by,
the proof model. That bridge is intentionally separate from this folder.
