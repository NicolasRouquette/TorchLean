/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Init.Data.BitVec.Lemmas
public import NN.IR.BinaryProofs

/-!
# Typed reference execution

Binary arithmetic reuses FloatLib at the declared precision. Boolean and fixed-width integer
values can also pass through the shared storage-only tensor operations. Integers of any positive
width use Lean's `BitVec`; their signedness remains in `DType`, rather than in a second carrier.
No integer arithmetic or cross-type integer conversion is inferred from that representation.

Element words are unsigned, including signed integer encodings. Boolean words are canonically
zero or one. Loading rejects excessive words, invalid descriptors and incompatible tensor sizes;
`Reference.read_write` proves that decoding a valid word does not change its encoding.

This interpreter is a software reference, not a proof of XLA or LibTorch execution. Other element
representations and operations still need implementations before they can run.
-/

@[expose] public section

namespace NN.IR.Reference

open TorchLean FloatLib.Floats
open FloatLib.Floats.Formats.BinaryInterchange

/-- Configured binary values, Boolean values, and arbitrary-width integer bit vectors. -/
abbrev scalar : DType → Type
  | .binary format => Binary.Scalar format
  | .bool => Bool
  | .integer _ (some width) => BitVec width
  | _ => Empty

/-- Whether this reference implements a valid element representation. -/
def supported (dtype : DType) : Bool :=
  dtype.valid && match dtype with
    | .binary _ | .bool | .integer _ (some _) => true
    | _ => false

/-- Decode one complete element word, rejecting rather than truncating excessive bits. -/
def read : (dtype : DType) → Nat → Except String (scalar dtype)
  | .binary format, word =>
      if word < 2 ^ format.bitWidth then pure (ExecFloat.Binary.ofNatBits word)
      else throw "reference IR: word exceeds its binary format"
  | .bool, word =>
      if word < 2 then pure (word != 0)
      else throw "reference IR: Boolean word must be zero or one"
  | .integer signed (some width), word => do
      (DType.integer signed (some width)).check
      if h : word < 2 ^ width then pure (BitVec.ofNatLT word h)
      else throw "reference IR: word exceeds its integer width"
  | dtype, _ => throw s!"reference IR: no word representation for {dtype}"

/-- Encode one element without interpreting signed integer words as host floats. -/
def write : (dtype : DType) → scalar dtype → Except String Nat
  | .binary _, value => .ok (ExecFloat.Binary.toNatBits value)
  | .bool, value => .ok (if value then 1 else 0)
  | .integer signed (some width), value => do
      (DType.integer signed (some width)).check
      pure value.toNat
  | dtype, _ => .error s!"reference IR: no word representation for {dtype}"

/-- Reuse directed binary conversion; nonbinary conversions currently require identical types.

Binary conversions still use FloatLib when the formats are equal, since exceptional-value
handling and flags belong to the conversion semantics, not to a tensor relabeling.
-/
def cast (source target : DType) (rounding : Model.IEEERoundingMode)
    (value : scalar source) : Except String (scalar target × ExecFloat.ConversionStatus) :=
  match source, target with
  | .binary src, .binary dst => Binary.cast (.binary src) (.binary dst) rounding value
  | source, target =>
      if supported source && supported target then
        if h : source = target then by
          cases h
          exact .ok (value, {})
        else .error s!"reference IR: conversion {source} → {target} is not implemented"
      else .error s!"reference IR: no representation for {source} or {target}"

/-- A successfully decoded element retains its entire original word on encoding. -/
theorem read_write (dtype : DType) (word : Nat) (value : scalar dtype)
    (h : read dtype word = .ok value) : write dtype value = .ok word := by
  cases dtype with
  | binary format =>
      simp only [read] at h
      split at h
      · cases h
        simp only [write, ExecFloat.Binary.toNatBits_ofNatBits_of_lt word ‹_›]
      · cases h
  | bool =>
      simp only [read] at h
      split at h
      · have hw : word = 0 ∨ word = 1 := by omega
        rcases hw with rfl | rfl <;> simp_all [write, Pure.pure, Except.pure]
      · cases h
  | integer signed width =>
      cases width with
      | none => cases value
      | some width =>
          simp only [read, Bind.bind, Except.bind] at h
          cases hc : DType.check (.integer signed (some width)) with
          | error error => simp [hc] at h
          | ok checked =>
              cases checked
              simp only [hc] at h
              split at h
              · cases h
                simp only [write, hc, Bind.bind, Except.bind, Pure.pure, Except.pure,
                  BitVec.toNat_ofNatLT]
              · cases h
  | context | rational | real | decimal _ | posit _ | complex _ | named _ _ => cases value

/-- The binary part of reference conversion satisfies the same FloatLib specification. -/
theorem cast_binary_value (source target : FloatFormat) (rounding : Model.IEEERoundingMode)
    (value : Binary.Scalar source) (converted : Binary.Scalar target)
    (flags : ExecFloat.ConversionStatus)
    (h : cast (.binary source) (.binary target) rounding value = .ok (converted, flags)) :
    ExecFloat.Binary.toModel converted =
      Model.castWithRounding source target (ExecFloat.Binary.toModel value) rounding :=
  Binary.cast_value source target rounding value converted flags h

end NN.IR.Reference

namespace NN.IR.Interpreter

open TorchLean

/-- Binary numerical evaluation and storage-only Boolean/arbitrary-width integer evaluation. -/
def reference : Interpreter where
  Scalar := Reference.scalar
  storage := fun dtype => match dtype with
    | .binary format => inferInstanceAs (Storage (Binary.Scalar format))
    | other => instArrayStorage (Reference.scalar other)
  supported := Reference.supported
  arithmetic := fun dtype => match dtype with
    | .binary format => some (inferInstanceAs (Context (Binary.Scalar format)))
    | _ => none
  cast := Reference.cast

end NN.IR.Interpreter

namespace NN.IR.Reference

open TorchLean

/-- Load a typed tensor from complete element words, checking its type, size and word ranges. -/
def load (dtype : DType) (shape : Spec.Shape) (words : Array Nat) :
    Except String (Interpreter.Value Interpreter.reference) := do
  letI : Storage (Interpreter.reference.Scalar dtype) := Interpreter.reference.storage dtype
  do
  dtype.check
  unless supported dtype do
    throw s!"reference IR: no representation for {dtype}"
  unless words.size == Spec.Shape.size shape do
    throw "reference IR: word count does not match the tensor shape"
  let tensor ← @Tensor.generateFlatM (Except String) inferInstance
    (Interpreter.reference.Scalar dtype) (Interpreter.reference.storage dtype) shape
    (fun index => read dtype words[index]!)
  pure ⟨dtype, @Spec.SomeTensor.ofTensor (Interpreter.reference.Scalar dtype)
    (Interpreter.reference.storage dtype) shape tensor⟩

/-- Complete element words of a typed tensor, retaining binary special values and integer bits. -/
def words (value : Interpreter.Value Interpreter.reference) : Except String (Array Nat) := do
  letI : Storage (Interpreter.reference.Scalar value.dtype) :=
    Interpreter.reference.storage value.dtype
  do
  value.dtype.check
  unless supported value.dtype do
    throw s!"reference IR: no representation for {value.dtype}"
  let data := @Tensor.Internal.Rep.data (Interpreter.reference.Scalar value.dtype)
    (Interpreter.reference.storage value.dtype) value.tensor.shape value.tensor.tensor
  data.mapM (write value.dtype)

end NN.IR.Reference
