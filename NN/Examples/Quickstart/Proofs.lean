/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API
public import NN.Proofs

/-!
# Quickstart: Proving Small TorchLean Facts

TorchLean examples are not only executable scripts. Many guarantees are ordinary Lean theorems:
shape round-trips, typed tensor construction, activation identities, and later full verification
statements.

The boundary where TorchLean becomes more than an executable ML library is:

- compile-time guarantees from shape-indexed tensor types, and
- ordinary mathematical lemmas about the public API.

The deeper proof libraries live under `NN.Proofs.*`, `NN.Verification.*`, and `NN.MLTheory.*`.
-/

@[expose] public section

namespace NN.Examples.Quickstart.Proofs

open TorchLean

/--
A tensor's shape is part of its type.

If this definition compiles, Lean has already checked that the literal has exactly two entries and
therefore really is a `Vec 2`. The commented shape-mismatch below is the kind of bug Lean catches
before runtime:

```lean
-- def badVector : Tensor Float (shape![3]) := tensor! [1.0, 2.0]
```
-/
def twoVector : Tensor.T Float (shape![2]) :=
  tensor! [1.0, 2.0]

/--
Runtime dimension lists can still be related back to static TorchLean shapes.

This is the compact theorem behind many JSON/CLI/data-loader paths: parse dimensions dynamically,
then recover the precise `Shape` used by the typed tensor API.
-/
theorem matrix_shape_roundtrip :
    Shape.ofDims (Shape.toList (shape![2, 3])) = shape![2, 3] := by
  simp

/-- ReLU fixes every nonnegative real number. -/
theorem relu_eq_self_of_nonnegative (x : ℝ) (hx : 0 ≤ x) :
    Activation.Math.reluSpec x = x := by
  unfold Activation.Math.reluSpec
  exact max_eq_left hx

/-- ReLU clamps nonpositive real inputs to zero. -/
theorem relu_eq_zero_of_nonpositive (x : ℝ) (hx : x ≤ 0) :
    Activation.Math.reluSpec x = 0 := by
  unfold Activation.Math.reluSpec
  exact max_eq_right hx

example : Activation.Math.reluSpec (3 : ℝ) = 3 := by
  exact relu_eq_self_of_nonnegative 3 (by norm_num)

example : Activation.Math.reluSpec (-2 : ℝ) = 0 := by
  exact relu_eq_zero_of_nonpositive (-2) (by norm_num)

end NN.Examples.Quickstart.Proofs
