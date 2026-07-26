/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.Util.Json

/-!
# Array predicates for verification artifacts

Small checkers often receive vector data as JSON float arrays before converting anything into
shape-indexed tensors. This module keeps the common pointwise predicates in one place.
-/

@[expose] public section

namespace NN.Verification.Util.Array

open NN.Verification.Json

/-- Boolean $\leq$ on floats, for executable artifact checks. -/
def floatLe (x y : Float) : Bool :=
  decide (x ≤ y)

/-- Boolean `<` on floats, for executable artifact checks. -/
def floatLt (x y : Float) : Bool :=
  decide (x < y)

/--
Check pointwise containment
$[\mathrm{lo},\mathrm{hi}]\subseteq[\mathrm{rootLo},\mathrm{rootHi}]$ for JSON-style float arrays.

The helper is strict about lengths through `allPairwise`: mismatched arrays are rejected.
-/
def boxWithin (rootLo rootHi lo hi : Array Float) : Bool :=
  allPairwise rootLo lo floatLe && allPairwise lo hi floatLe && allPairwise hi rootHi floatLe

/--
Check whether a lower-bound vector refutes a threshold vector at some coordinate.

This is the executable form of
$\exists i,\ \mathrm{threshold}[i]<\mathrm{lowerBound}[i]$.
-/
def refutesThreshold (lowerBound threshold : Array Float) : Bool :=
  anyPairwise lowerBound threshold (fun lb thr => floatLt thr lb)

/--
Check a supplied witness coordinate for threshold refutation.

Out-of-range witnesses are rejected instead of silently falling back to another coordinate.
-/
def refutesThresholdAt (lowerBound threshold : Array Float) (witnessIdx : Nat) : Bool :=
  if hLower : witnessIdx < lowerBound.size then
    if hThreshold : witnessIdx < threshold.size then
      let thresholdAt := threshold[witnessIdx]'hThreshold
      let lowerAt := lowerBound[witnessIdx]'hLower
      floatLt thresholdAt lowerAt
    else
      false
  else
    false

end NN.Verification.Util.Array
