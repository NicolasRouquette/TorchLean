/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Init.Data.Float32
public import NN.Floats.IEEEExec.Exec32

/-!
# Runtime `Float32` Bridge

External (assumption-based) bridge: Lean's `Init.Float32` ↔ `IEEE32Exec`.

Why assumptions are necessary:
`Init.Float32` arithmetic in Lean is implemented by *external* runtime calls. Those calls are opaque
to the Lean kernel, so (inside Lean) we cannot prove that the runtime implementation coincides
bit-for-bit with any particular float32 specification.

We package the finite-result connection as a typeclass interface. If you (or your trusted runtime)
can discharge the assumptions that finite runtime `Float32` primitives match the executable kernel
`IEEE32Exec` bit-for-bit, then you can:

1) execute with `Float32` (runtime),
2) rewrite the result to `IEEE32Exec`,
3) reuse `Bridge/FP32.lean` and `Bridge/Expressions.lean` to connect
   execution to the `FP32` rounding-on-`ℝ` model on finite/no-overflow inputs.

The trust boundary is explicit: the only unproved part is the external/runtime correctness
assumption, which is unavoidable in a pure Lean development.

Background:
- IEEE 754-2019 (what it means to “match float32 semantics”):
  https://doi.org/10.1109/IEEESTD.2019.8766229
-/

@[expose] public section


namespace TorchLean.Floats.IEEE754

open IEEE32Exec

namespace Float32Bridge

/-- Reinterpret a runtime `Float32` as the executable bit-level float32 (`IEEE32Exec`). -/
@[inline] def toIEEE32Exec (x : _root_.Float32) : IEEE32Exec :=
  IEEE32Exec.ofBits x.toBits

/-- Reinterpret an executable float32 bit-pattern as a runtime `Float32`. -/
@[inline] def ofIEEE32Exec (x : IEEE32Exec) : _root_.Float32 :=
  Float32.ofBits x.bits

/-!
## What this bridge gives us

This file does *not* prove anything about the runtime semantics of `Float32`. Instead, it gives a
clean *interface* that you can assume/provide:

- If the runtime `Float32` primitives produce the same result bits as `IEEE32Exec`,
  then runtime evaluation can be rewritten into `IEEE32Exec` evaluation.
- Once we are in `IEEE32Exec`, we can use the internal bridge theorems to connect execution to the
  `FP32` rounding-on-`ℝ` model on the finite/no-overflow path (`Bridge/FP32.lean`,
  `Bridge/FP32Total.lean`, and `Bridge/Expressions.lean`).

NaN payloads are deliberately outside the exact-bit contract: Lean runtimes may canonicalize them,
whereas `IEEE32Exec` preserves a deterministic payload. Classification fields below still relate
NaN, infinity, and finiteness for every runtime value.
-/

/-! ## External correctness assumptions -/

/--
Assumption package relating the finite-result part of Lean's runtime `Float32` primitives to
`IEEE32Exec`. Exact NaN payload agreement is intentionally not required.
-/
class RuntimeFloat32FiniteMatchesIEEE32Exec : Prop where
  toBits_ofBits_of_isFinite : ∀ b : UInt32,
    IEEE32Exec.isFinite (IEEE32Exec.ofBits b) = true → (Float32.ofBits b).toBits = b

  add_bits_of_isFinite : ∀ a b : _root_.Float32,
    Float32.isFinite a = true → Float32.isFinite b = true →
    Float32.isFinite (Float32.add a b) = true →
    (Float32.add a b).toBits = (IEEE32Exec.add (toIEEE32Exec a) (toIEEE32Exec b)).bits
  sub_bits_of_isFinite : ∀ a b : _root_.Float32,
    Float32.isFinite a = true → Float32.isFinite b = true →
    Float32.isFinite (Float32.sub a b) = true →
    (Float32.sub a b).toBits = (IEEE32Exec.sub (toIEEE32Exec a) (toIEEE32Exec b)).bits
  mul_bits_of_isFinite : ∀ a b : _root_.Float32,
    Float32.isFinite a = true → Float32.isFinite b = true →
    Float32.isFinite (Float32.mul a b) = true →
    (Float32.mul a b).toBits = (IEEE32Exec.mul (toIEEE32Exec a) (toIEEE32Exec b)).bits
  div_bits_of_isFinite : ∀ a b : _root_.Float32,
    Float32.isFinite a = true → Float32.isFinite b = true →
    Float32.isFinite (Float32.div a b) = true →
    (Float32.div a b).toBits = (IEEE32Exec.div (toIEEE32Exec a) (toIEEE32Exec b)).bits
  neg_bits_of_isFinite : ∀ a : _root_.Float32, Float32.isFinite a = true →
    (Float32.neg a).toBits = (IEEE32Exec.neg (toIEEE32Exec a)).bits
  sqrt_bits_of_isFinite : ∀ a : _root_.Float32,
    Float32.isFinite a = true → Float32.isFinite (Float32.sqrt a) = true →
    (Float32.sqrt a).toBits = (IEEE32Exec.sqrt (toIEEE32Exec a)).bits

  isNaN_bits : ∀ a : _root_.Float32,
    Float32.isNaN a = IEEE32Exec.isNaN (toIEEE32Exec a)
  isInf_bits : ∀ a : _root_.Float32,
    Float32.isInf a = IEEE32Exec.isInf (toIEEE32Exec a)
  isFinite_bits : ∀ a : _root_.Float32,
    Float32.isFinite a = IEEE32Exec.isFinite (toIEEE32Exec a)

/-! ## Derived bit-level refinement lemmas -/

namespace RuntimeFloat32FiniteMatchesIEEE32Exec

variable [RuntimeFloat32FiniteMatchesIEEE32Exec]

/-!
## Derived lemmas (rewriting runtime to executable)

The assumptions above are phrased as bit equalities. In practice we almost always want the more
convenient *value-level* rewriting lemmas below:

`toIEEE32Exec (a + b) = IEEE32Exec.add (toIEEE32Exec a) (toIEEE32Exec b)`, etc.

These are the lemmas you use to “turn a runtime evaluation into an `IEEE32Exec` evaluation”.
-/

-- `IEEE32Exec` stores a `UInt32` bit pattern; equality is extensional on `.bits`.
omit [RuntimeFloat32FiniteMatchesIEEE32Exec] in
private theorem bits_inj {x y : IEEE32Exec} (h : x.bits = y.bits) : x = y := by
  cases x
  cases y
  cases h
  rfl

-- We write these theorems using the usual notation (`a + b`, `a * b`) because that is how most
-- downstream code is written; the assumptions are stated in terms of `Float32.add`, etc.
/-- Rewrite runtime float32 addition into executable `IEEE32Exec.add`. -/
theorem toIEEE32Exec_add (a b : _root_.Float32) :
    Float32.isFinite a = true → Float32.isFinite b = true →
    Float32.isFinite (Float32.add a b) = true →
    toIEEE32Exec (Float32.add a b) = IEEE32Exec.add (toIEEE32Exec a) (toIEEE32Exec b) := by
  intro ha hb hr
  apply bits_inj
  simpa [toIEEE32Exec, IEEE32Exec.ofBits] using
    (RuntimeFloat32FiniteMatchesIEEE32Exec.add_bits_of_isFinite (a := a) (b := b) ha hb hr)

/-- Rewrite runtime float32 subtraction into executable `IEEE32Exec.sub` (value-level form). -/
theorem toIEEE32Exec_sub (a b : _root_.Float32) :
    Float32.isFinite a = true → Float32.isFinite b = true →
    Float32.isFinite (Float32.sub a b) = true →
    toIEEE32Exec (Float32.sub a b) = IEEE32Exec.sub (toIEEE32Exec a) (toIEEE32Exec b) := by
  intro ha hb hr
  apply bits_inj
  simpa [toIEEE32Exec, IEEE32Exec.ofBits] using
    (RuntimeFloat32FiniteMatchesIEEE32Exec.sub_bits_of_isFinite (a := a) (b := b) ha hb hr)

/-- Rewrite runtime float32 multiplication into executable `IEEE32Exec.mul` (value-level form). -/
theorem toIEEE32Exec_mul (a b : _root_.Float32) :
    Float32.isFinite a = true → Float32.isFinite b = true →
    Float32.isFinite (Float32.mul a b) = true →
    toIEEE32Exec (Float32.mul a b) = IEEE32Exec.mul (toIEEE32Exec a) (toIEEE32Exec b) := by
  intro ha hb hr
  apply bits_inj
  simpa [toIEEE32Exec, IEEE32Exec.ofBits] using
    (RuntimeFloat32FiniteMatchesIEEE32Exec.mul_bits_of_isFinite (a := a) (b := b) ha hb hr)

/-- Rewrite runtime float32 division into executable `IEEE32Exec.div` (value-level form). -/
theorem toIEEE32Exec_div (a b : _root_.Float32) :
    Float32.isFinite a = true → Float32.isFinite b = true →
    Float32.isFinite (Float32.div a b) = true →
    toIEEE32Exec (Float32.div a b) = IEEE32Exec.div (toIEEE32Exec a) (toIEEE32Exec b) := by
  intro ha hb hr
  apply bits_inj
  simpa [toIEEE32Exec, IEEE32Exec.ofBits] using
    (RuntimeFloat32FiniteMatchesIEEE32Exec.div_bits_of_isFinite (a := a) (b := b) ha hb hr)

/-- Rewrite runtime float32 negation into executable `IEEE32Exec.neg` (value-level form). -/
theorem toIEEE32Exec_neg (a : _root_.Float32) (ha : Float32.isFinite a = true) :
    toIEEE32Exec (Float32.neg a) = IEEE32Exec.neg (toIEEE32Exec a) := by
  apply bits_inj
  simpa [toIEEE32Exec, IEEE32Exec.ofBits] using
    (RuntimeFloat32FiniteMatchesIEEE32Exec.neg_bits_of_isFinite (a := a) ha)

/-- Rewrite runtime float32 square root into executable `IEEE32Exec.sqrt` (value-level form). -/
theorem toIEEE32Exec_sqrt (a : _root_.Float32) :
    Float32.isFinite a = true → Float32.isFinite (Float32.sqrt a) = true →
    toIEEE32Exec (Float32.sqrt a) = IEEE32Exec.sqrt (toIEEE32Exec a) := by
  intro ha hr
  apply bits_inj
  simpa [toIEEE32Exec, IEEE32Exec.ofBits] using
    (RuntimeFloat32FiniteMatchesIEEE32Exec.sqrt_bits_of_isFinite (a := a) ha hr)

/-- Converting a finite `IEEE32Exec` value to runtime `Float32` and back preserves its bits. -/
theorem toIEEE32Exec_ofIEEE32Exec (x : IEEE32Exec) (hx : IEEE32Exec.isFinite x = true) :
    toIEEE32Exec (ofIEEE32Exec x) = x := by
  apply bits_inj
  have hxBits : IEEE32Exec.isFinite (IEEE32Exec.ofBits x.bits) = true := by
    have hwrap : IEEE32Exec.ofBits x.bits = x := by cases x; rfl
    rw [hwrap]
    exact hx
  simpa [toIEEE32Exec, ofIEEE32Exec, IEEE32Exec.ofBits] using
    (RuntimeFloat32FiniteMatchesIEEE32Exec.toBits_ofBits_of_isFinite (b := x.bits) hxBits)

end RuntimeFloat32FiniteMatchesIEEE32Exec

end Float32Bridge

end TorchLean.Floats.IEEE754
