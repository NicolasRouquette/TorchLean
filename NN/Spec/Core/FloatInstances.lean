/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Exec32
public import NN.Floats.NeuralFloat.Scalar.NF
public import NN.Spec.Core.Context

/-!
# Floating-Point Adapters For Tensor Specifications

The numerical types in `NN.Floats` are independent of TorchLean's tensor and model interfaces.
This module supplies the one-way adapters that let those types instantiate the broader `Context`
expected by scalar-polymorphic specifications.
-/

@[expose] public section

namespace TorchLean.Floats

namespace NF

variable {β : NeuralRadix} {fexp : ℤ → ℤ} {rnd : ℝ → ℤ}
variable [NeuralValidExp fexp] [NeuralValidRnd rnd]

/--
Use rounded-real `NF` arithmetic as a TorchLean specification scalar.

The general scalar interface requires a total `α ^ α`. Its adapter uses `NF.checkedRealPow`, which
handles arbitrary exponents on positive bases, integer exponents on negative bases, and positive
exponents at zero. The adapter selects its rounded-zero fallback only when `checkedRealPow` rejects
the domain, such as a negative base with a noninteger exponent or zero with a negative exponent;
an accepted computation can independently round to zero. Direct numerical code should inspect the
checked result, or use the unambiguous `NF.powNat`, rather than relying on that compatibility
fallback.
-/
noncomputable instance : Context (NF β fexp rnd) where
  pow a b :=
    (checkedRealPow (β := β) (fexp := fexp) (rnd := rnd) a b).getD
      (ofReal (β := β) (fexp := fexp) (rnd := rnd) 0)
  decidable_gt := Classical.decRel _

end NF

namespace IEEE754.IEEE32Exec

/-- Use executable binary32 arithmetic as a TorchLean specification scalar. -/
instance : Context IEEE32Exec where
  decidable_gt := fun x y => inferInstanceAs (Decidable (x > y))

end IEEE754.IEEE32Exec
end TorchLean.Floats
