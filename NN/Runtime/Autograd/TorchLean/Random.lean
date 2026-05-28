/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context
public import NN.Spec.Core.Tensor.Core

/-!
# Random

Deterministic RNG utilities for TorchLean (seed-threaded, *pure*).

## Why not `IO.rand` / runtime randomness?

Lean (and mathlib) can generate random numbers via `IO`, but that gives *effectful* randomness
whose results depend on hidden runtime state. For TorchLean, that is a poor fit:

- it breaks the “one semantics” contract for compilation/verification (graphs stop being a pure
  mathematical object unless you model the RNG state explicitly),
- it makes replays / certificate checking brittle (you want reproducible runs),
- it complicates the proof-linked compilation path (effects and mixed dtypes).

Instead we use a deterministic pseudorandom generator and treat “randomness” as
**a deterministic function of an explicit seed** (and a counter/stream id). This mirrors the
JAX/functional RNG style and keeps the semantic core pure.

## What this file provides

- a small 64-bit PRNG (SplitMix64-style mixing),
- deterministic sampling utilities keyed by `(seed, counter, linearIndex)`,
- a deterministic way to build dropout-style `{0,1}` masks as shape-indexed tensors.

Today this is used by *Session-level* stochastic layers (e.g. `TorchLean.Session.dropout`), where
the RNG state is stored in `NatRef`s and used to generate masks reproducibly.

If you want “PyTorch-like” randomness at the boundary, prefer sampling an initial seed in `IO`
and then using the seeded RNG from that point onward (`TorchLean.Session.initRngFromIO`).
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace TorchLean
namespace Random

open Spec
open Tensor

/-! ### SplitMix64-style mixing -/

/--
SplitMix64-style mixing function on 64-bit words.

This is used as a compact deterministic PRNG core: we treat "randomness" as a pure function of an
explicit seed/counter/index.
-/
def splitmix64 (x : UInt64) : UInt64 :=
  let z1 := x + 0x9e3779b97f4a7c15
  let z2 := (z1 ^^^ (z1 >>> 30)) * 0xbf58476d1ce4e5b9
  let z3 := (z2 ^^^ (z2 >>> 27)) * 0x94d049bb133111eb
  z3 ^^^ (z3 >>> 31)

/-- Derive a per-call key from a `(seed, counter)` pair. -/
def keyOf (seed counter : Nat) : UInt64 :=
  splitmix64 (UInt64.ofNat seed + UInt64.ofNat counter)

/-- Advance the seed deterministically once per RNG use. -/
def nextSeed (seed counter : Nat) : Nat :=
  (splitmix64 (UInt64.ofNat seed + UInt64.ofNat counter + 0x9e3779b97f4a7c15)).toNat

/-! ### Sampling helpers -/

/-- Deterministic "random" natural number in `[0, denom)` derived from `key` and a linear index. -/
def sampleNat (key : UInt64) (linearIndex : Nat) (denom : Nat := (2:Nat) ^ 32) : Nat :=
  (splitmix64 (key + UInt64.ofNat linearIndex)).toNat % denom

/-- Convert `u/denom` into `α` using `Context` coercions. -/
def sampleUnit {α : Type} [Context α] (u denom : Nat) : α :=
  (u : α) / (denom : α)

/-- Decide whether to keep an element given `keepProb` and a sample `u ∈ [0, denom)`. -/
def keepBit {α : Type} [Context α] (keepProb : α) (u denom : Nat) : α :=
  let uα : α := sampleUnit (α := α) u denom
  have : Decidable (keepProb > uα) := (Context.decidable_gt) keepProb uα
  if keepProb > uα then (1 : α) else (0 : α)

/-! ### Uniform tensors -/

/--
Build a deterministic tensor with entries in `[0,1)` (discrete grid `u/denom`) with the requested
shape.

This is keyed by:
- `key` (typically derived from a seed and a counter), and
- `linearOffset` (to make recursion order-insensitive).
-/
def uniformAux {α : Type} [Context α] (key : UInt64) :
    ∀ {s : Shape}, Nat → Tensor α s
  | .scalar, linearOffset =>
      let denom : Nat := (2:Nat) ^ 32
      let u := sampleNat key linearOffset denom
      Tensor.scalar (sampleUnit (α := α) u denom)
  | .dim _n rest, linearOffset =>
      let block := Shape.size rest
      Tensor.dim (fun i =>
        uniformAux (α := α) key (s := rest) (linearOffset + i.1 * block))

/-- Convenience wrapper: build a uniform tensor starting at offset `0`. -/
def uniform {α : Type} [Context α] (key : UInt64) {s : Shape} : Tensor α s :=
  uniformAux (α := α) key (s := s) 0

/-! ### Mask construction -/

/--
Build a dropout-style mask with entries in `{0,1}` and the same shape as the target tensor.

The mask is deterministic given:
- `key` (derived from seed/counter),
- `keepProb` (probability of keeping a unit),
- `linearOffset` (typically `0` for the whole tensor).
-/
def maskAux {α : Type} [Context α] (key : UInt64) (keepProb : α) :
    ∀ {s : Shape}, Nat → Tensor α s
  | .scalar, linearOffset =>
      let denom : Nat := (2:Nat) ^ 32
      let u := sampleNat key linearOffset denom
      Tensor.scalar (keepBit (α := α) keepProb u denom)
  | .dim _n rest, linearOffset =>
      let block := Shape.size rest
      Tensor.dim (fun i =>
        maskAux (α := α) key keepProb (s := rest) (linearOffset + i.1 * block))

/-- Convenience wrapper: build a mask starting at offset `0`. -/
def mask {α : Type} [Context α] (key : UInt64) (keepProb : α) {s : Shape} : Tensor α s :=
  maskAux (α := α) key keepProb (s := s) 0

/-! ### Standard normal tensors -/

/--
Box–Muller transform: turn two independent uniforms `u1,u2 ∈ (0,1)` into a standard normal sample.

We return only the `cos` branch:

`z = sqrt(-2 * log u1) * cos(2π * u2)`.

Notes:
- We clamp `u1` below by `ε` to avoid `log 0`.
- This is intended as a deterministic *pseudo*-normal sampler for examples and benchmarking.
  It is not a cryptographic RNG.

Reference:
- Box & Muller (1958), "A Note on the Generation of Random Normal Deviates".
-/
def boxMullerCos {α : Type} [Context α] (u1 u2 : α) : α :=
  let u1' := Max.max u1 Numbers.epsilon
  let r : α := MathFunctions.sqrt (-Numbers.two * MathFunctions.log u1')
  let theta : α := (Numbers.two * MathFunctions.pi) * u2
  r * MathFunctions.cos theta

/--
Deterministic standard normal `N(0,1)` sample derived from `key` and a linear index.

We use two 32-bit uniforms (via `sampleNat`) per output scalar and apply the Box–Muller transform.
-/
def normalScalar {α : Type} [Context α] (key : UInt64) (linearIndex : Nat) : α :=
  let denom : Nat := (2:Nat) ^ 32
  let u1n := sampleNat key (2 * linearIndex) denom
  let u2n := sampleNat key (2 * linearIndex + 1) denom
  let u1 : α := sampleUnit (α := α) u1n denom
  let u2 : α := sampleUnit (α := α) u2n denom
  boxMullerCos (α := α) u1 u2

/--
Build a deterministic tensor with (approximate) standard normal entries.

As with `uniformAux`, this is order-insensitive: the recursion uses `linearOffset` plus a
block-size multiplier so the same tensor shape always yields the same samples.
-/
def normalAux {α : Type} [Context α] (key : UInt64) :
    ∀ {s : Shape}, Nat → Tensor α s
  | .scalar, linearOffset =>
      Tensor.scalar (normalScalar (α := α) key linearOffset)
  | .dim _n rest, linearOffset =>
      let block := Shape.size rest
      Tensor.dim (fun i =>
        normalAux (α := α) key (s := rest) (linearOffset + i.1 * block))

/-- Convenience wrapper: build a standard normal tensor starting at offset `0`. -/
def normal {α : Type} [Context α] (key : UInt64) {s : Shape} : Tensor α s :=
  normalAux (α := α) key (s := s) 0

end Random
end TorchLean
end Autograd
end Runtime
