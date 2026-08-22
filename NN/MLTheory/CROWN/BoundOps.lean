/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context

/-!
# Directed-rounding primitives for interval propagation

TorchLean’s IBP/CROWN code represents bounds as endpoint pairs (`lo`/`hi`) inside `Box`/`FlatBox`.
To make those bounds meaningful under different numeric semantics, we abstract the *primitive*
endpoint operations (directed rounding).

Intuition:
- For pure real/interval backends, using ordinary `+`/`*` is already enclosure-safe (because the
  scalar itself is an interval type with outward rounding).
- For finite-precision backends with discrete grids (e.g. `IEEE32Exec`), we want *directed rounding*
  primitives like `addDown/addUp` and `mulDown/mulUp` so that interval propagation encloses the
  corresponding exact real operation.

This file defines two small numeric interfaces:

- `BoundOps` for directed elementary arithmetic; and
- `NonlinearBoundOps` for interval transfers that may be unavailable on a backend.

There is intentionally no generic fallback for `BoundOps`: ordinary finite-precision arithmetic is
not directed rounding and must not silently enter a sound bound-propagation path. The nonlinear
interface does have a conservative fallback for bounded activations, but operations with unbounded
ranges return `none` unless the scalar backend supplies an implementation. Soundness is a separate
obligation, recorded by `LawfulNonlinearBoundOps`.

## Integration points in the current codebase

The intended usage is:

- Keep graphs/layers scalar-polymorphic over `[Context α]`.
- When a routine *propagates bounds* (IBP/affine/CROWN), also require `[BoundOps α]` and use
  `addDown/addUp/subDown/subUp/mulDown/mulUp` at the endpoints.

Concretely:

- `NN/MLTheory/CROWN/Core.lean`
  - `AffineVec.eval_on_box`: min/max over products and accumulation use `BoundOps`.
  - `IBP.linear`: interval linear layer propagation uses `BoundOps`.
- `NN.MLTheory.CROWN.Graph`
  - `boxAdd`, `boxSub`, `boxMulElem`: endpoint propagation uses `BoundOps`.

Executable certificate replay relies on this separation. A backend may support directed binary
arithmetic without having a correctly rounded `exp` or `log`; in that case the graph checker leaves
the corresponding node unresolved instead of treating an ordinary library call as an enclosure.
-/

@[expose] public section


namespace NN.MLTheory.CROWN

variable {α : Type} [Context α]

/-!
## `BoundOps α`

`BoundOps` supplies directed-rounding versions of the arithmetic
primitives that appear in IBP for affine/linear layers and basic arithmetic nodes.

If you want to swap in a quantized backend, the key is to provide an instance of `BoundOps` for
your scalar type.
-/
class BoundOps (α : Type) [Context α] where
  addDown : α → α → α
  addUp   : α → α → α
  subDown : α → α → α
  subUp   : α → α → α
  mulDown : α → α → α
  mulUp   : α → α → α
  /-- Whether ordinary scalar algebra may be reassociated without a rounding error. -/
  supportsExactAffineReassociation : Bool := false

/--
Real-semantic enclosure laws for `BoundOps`.

The executable interface above is intentionally available without this class: a backend may be
useful for diagnostics before its arithmetic has been connected to a proof.  Sound CROWN theorems
require `LawfulBoundOps` in addition to `BoundOps`. The interpretation `toReal` says what a scalar
endpoint means mathematically, and the laws compare each directed operation with exact arithmetic
on those real values. This is stronger than merely surrounding the backend's ordinary rounded
operation.

There is a global instance for `ℝ`.  There is deliberately no global instance for Lean `Float` or
for all `IEEE32Exec` bit patterns.  Host `Float` is a trusted runtime boundary, while IEEE-754 NaNs,
infinities, and overflow require finite-path hypotheses; those facts are stated at the IEEE
semantics layer rather than hidden in an invalid ordered-ring instance.
-/
class LawfulBoundOps (α : Type) [Context α] [BoundOps α] where
  /-- Mathematical value represented by an endpoint. -/
  toReal : α → ℝ
  /-- Executable endpoint comparisons agree with the mathematical order. -/
  lt_iff (a b : α) : a < b ↔ toReal a < toReal b
  addDown_le (a b : α) : toReal (BoundOps.addDown a b) ≤ toReal a + toReal b
  le_addUp (a b : α) : toReal a + toReal b ≤ toReal (BoundOps.addUp a b)
  subDown_le (a b : α) : toReal (BoundOps.subDown a b) ≤ toReal a - toReal b
  le_subUp (a b : α) : toReal a - toReal b ≤ toReal (BoundOps.subUp a b)
  mulDown_le (a b : α) : toReal (BoundOps.mulDown a b) ≤ toReal a * toReal b
  le_mulUp (a b : α) : toReal a * toReal b ≤ toReal (BoundOps.mulUp a b)

namespace BoundOps

/-- Minimum of two scalar endpoints. -/
@[inline] def min2 (a b : α) : α :=
  if decide (a > b) then b else a

/-- Maximum of two scalar endpoints. -/
@[inline] def max2 (a b : α) : α :=
  if decide (a > b) then a else b

end BoundOps

/-!
## Nonlinear enclosure operations

Each method consumes a closed interval `[lo, hi]`. A successful result is another endpoint pair;
`none` means that this backend does not implement a finite transfer for the requested operation.
Division receives both numerator and denominator intervals. The executable result alone makes no
soundness claim; `LawfulNonlinearBoundOps` supplies that claim when a theorem needs it.

The interface is deliberately operational, like `BoundOps`. The proof layer establishes soundness
for the concrete implementations used by checked workflows; an external instance without a lawful
instance remains part of the backend trust boundary.
-/

class NonlinearBoundOps (α : Type) [Context α] where
  divBounds : α → α → α → α → Option (α × α)
  expBounds : α → α → Option (α × α)
  logBounds : α → α → Option (α × α)
  sqrtBounds : α → α → Option (α × α)
  sigmoidBounds : α → α → Option (α × α)
  tanhBounds : α → α → Option (α × α)
  sinBounds : α → α → Option (α × α)
  cosBounds : α → α → Option (α × α)
  /-- Uniform absolute bound for one last-axis layer-normalization row. -/
  layerNormAbsBound : Nat → Option α
  /-- Whether coupled softmax/layer-normalization derivative formulas use exact scalar arithmetic. -/
  supportsIdealCoupledDerivatives : Bool

/--
Soundness predicate for a unary interval transfer.

Returning `none` is always permitted. If the transfer returns endpoints, every real input between
the interpreted input endpoints must map between the interpreted output endpoints.
-/
def UnaryEnclosure [BoundOps α] [LawfulBoundOps α]
    (f : ℝ → ℝ) (transfer : α → α → Option (α × α)) : Prop :=
  ∀ {lo hi outLo outHi : α} {x : ℝ}, transfer lo hi = some (outLo, outHi) →
    LawfulBoundOps.toReal lo ≤ x → x ≤ LawfulBoundOps.toReal hi →
    LawfulBoundOps.toReal outLo ≤ f x ∧ f x ≤ LawfulBoundOps.toReal outHi

/-- Soundness predicate for a binary interval transfer. -/
def BinaryEnclosure [BoundOps α] [LawfulBoundOps α]
    (f : ℝ → ℝ → ℝ) (transfer : α → α → α → α → Option (α × α)) : Prop :=
  ∀ {aLo aHi bLo bHi outLo outHi : α} {x y : ℝ},
    transfer aLo aHi bLo bHi = some (outLo, outHi) →
    LawfulBoundOps.toReal aLo ≤ x → x ≤ LawfulBoundOps.toReal aHi →
    LawfulBoundOps.toReal bLo ≤ y → y ≤ LawfulBoundOps.toReal bHi →
    LawfulBoundOps.toReal outLo ≤ f x y ∧ f x y ≤ LawfulBoundOps.toReal outHi

/--
Real-semantic enclosure laws for `NonlinearBoundOps`.

This class is deliberately separate from the executable transfer table. A backend may implement a
transfer for testing before proving it; sound verification entrypoints can require this class and
therefore cannot silently promote an unchecked implementation into a theorem.
-/
class LawfulNonlinearBoundOps (α : Type) [Context α] [BoundOps α] [LawfulBoundOps α]
    [NonlinearBoundOps α] : Prop where
  divBounds_enclosure :
    BinaryEnclosure (α := α) (· / ·) (NonlinearBoundOps.divBounds (α := α))
  expBounds_enclosure :
    UnaryEnclosure (α := α) Real.exp (NonlinearBoundOps.expBounds (α := α))
  logBounds_enclosure :
    UnaryEnclosure (α := α) Real.log (NonlinearBoundOps.logBounds (α := α))
  sqrtBounds_enclosure :
    UnaryEnclosure (α := α) Real.sqrt (NonlinearBoundOps.sqrtBounds (α := α))
  sigmoidBounds_enclosure :
    UnaryEnclosure (α := α) (fun x : ℝ => 1 / (1 + Real.exp (-x)))
      (NonlinearBoundOps.sigmoidBounds (α := α))
  tanhBounds_enclosure :
    UnaryEnclosure (α := α) Real.tanh (NonlinearBoundOps.tanhBounds (α := α))
  sinBounds_enclosure :
    UnaryEnclosure (α := α) Real.sin (NonlinearBoundOps.sinBounds (α := α))
  cosBounds_enclosure :
    UnaryEnclosure (α := α) Real.cos (NonlinearBoundOps.cosBounds (α := α))
  layerNormAbsBound_sound {n : Nat} {radius : α} :
    NonlinearBoundOps.layerNormAbsBound (α := α) n = some radius →
      Real.sqrt n ≤ LawfulBoundOps.toReal radius
  coupledDerivatives_exact :
    NonlinearBoundOps.supportsIdealCoupledDerivatives (α := α) →
      BoundOps.supportsExactAffineReassociation (α := α)

namespace NonlinearBoundOps

/-- Minimum of four endpoints. -/
def min4 (a b c d : α) : α :=
  BoundOps.min2 (BoundOps.min2 a b) (BoundOps.min2 c d)

/-- Maximum of four endpoints. -/
def max4 (a b c d : α) : α :=
  BoundOps.max2 (BoundOps.max2 a b) (BoundOps.max2 c d)

/-- The denominator interval avoids zero. -/
def denominatorAvoidsZero (lo hi : α) : Bool :=
  decide (lo > Numbers.zero) || decide (Numbers.zero > hi)

end NonlinearBoundOps

/--
Conservative nonlinear ranges available for every scalar context.

Sigmoid, tanh, sine, and cosine have format-independent codomain bounds. Unbounded operations and
layer normalization remain unavailable until a concrete backend provides directed implementations.
-/
instance (priority := 100) instNonlinearBoundOpsConservative : NonlinearBoundOps α where
  divBounds := fun _ _ _ _ => none
  expBounds := fun _ _ => none
  logBounds := fun _ _ => none
  sqrtBounds := fun _ _ => none
  sigmoidBounds := fun _ _ => some (Numbers.zero, Numbers.one)
  tanhBounds := fun _ _ => some (Numbers.negOne, Numbers.one)
  sinBounds := fun _ _ => some (Numbers.negOne, Numbers.one)
  cosBounds := fun _ _ => some (Numbers.negOne, Numbers.one)
  layerNormAbsBound := fun _ => none
  supportsIdealCoupledDerivatives := false

/-!
Exact real arithmetic needs no rounding, so its lower and upper operations coincide.
-/
noncomputable instance instBoundOpsReal : BoundOps ℝ where
  addDown := (· + ·)
  addUp   := (· + ·)
  subDown := (· - ·)
  subUp   := (· - ·)
  mulDown := (· * ·)
  mulUp   := (· * ·)
  supportsExactAffineReassociation := true

/-- Exact real endpoint arithmetic satisfies the directed-operation enclosure laws. -/
noncomputable instance instLawfulBoundOpsReal : LawfulBoundOps ℝ where
  toReal := id
  lt_iff _ _ := Iff.rfl
  addDown_le _ _ := le_rfl
  le_addUp _ _ := le_rfl
  subDown_le _ _ := le_rfl
  le_subUp _ _ := le_rfl
  mulDown_le _ _ := le_rfl
  le_mulUp _ _ := le_rfl

/-- Exact nonlinear interval transfers over the real numbers. -/
noncomputable instance instNonlinearBoundOpsReal : NonlinearBoundOps ℝ where
  divBounds aLo aHi bLo bHi :=
    if bLo > 0 || 0 > bHi then
      let p1 := aLo / bLo
      let p2 := aLo / bHi
      let p3 := aHi / bLo
      let p4 := aHi / bHi
      some (min (min p1 p2) (min p3 p4), max (max p1 p2) (max p3 p4))
    else
      none
  expBounds lo hi := some (Real.exp lo, Real.exp hi)
  logBounds lo hi :=
    if lo > 0 then some (Real.log lo, Real.log hi) else none
  sqrtBounds lo hi :=
    if hi < 0 then none else some (Real.sqrt (max lo 0), Real.sqrt hi)
  sigmoidBounds lo hi :=
    some ((1 : ℝ) / (1 + Real.exp (-lo)), (1 : ℝ) / (1 + Real.exp (-hi)))
  tanhBounds lo hi := some (Real.tanh lo, Real.tanh hi)
  sinBounds := fun _ _ => some (-1, 1)
  cosBounds := fun _ _ => some (-1, 1)
  layerNormAbsBound n := some (Real.sqrt n)
  supportsIdealCoupledDerivatives := true

/-!
## Host binary64 endpoints

Lean's `Float` operations round to nearest on the host binary64 format. For executable checking we
widen every finite result by one adjacent representable value. This is deliberately an explicit
instance rather than a generic fallback: its soundness depends on the host IEEE-754 arithmetic
boundary documented by Lean, whereas `instBoundOpsReal` is exact and the `IEEE32Exec` instance is
connected to TorchLean's bit-level binary32 proofs.
-/

namespace HostFloat

def signMask : UInt64 := 0x8000000000000000
def posInfBits : UInt64 := 0x7ff0000000000000
def negInfBits : UInt64 := 0xfff0000000000000

/-- Adjacent binary64 value above `x`, with the usual IEEE behavior at infinities and zeros. -/
def nextUp (x : Float) : Float :=
  let bits := x.toBits
  if x.isNaN || bits = posInfBits then
    x
  else if bits = signMask || bits = 0 then
    Float.ofBits 1
  else if bits &&& signMask = 0 then
    Float.ofBits (bits + 1)
  else
    Float.ofBits (bits - 1)

/-- Adjacent binary64 value below `x`, with the usual IEEE behavior at infinities and zeros. -/
def nextDown (x : Float) : Float :=
  let bits := x.toBits
  if x.isNaN || bits = negInfBits then
    x
  else if bits = signMask || bits = 0 then
    Float.ofBits (signMask + 1)
  else if bits &&& signMask = 0 then
    Float.ofBits (bits - 1)
  else
    Float.ofBits (bits + 1)

end HostFloat

/--
Outward-widened host binary64 operations.

This instance is suitable for executable certificate replay under the trusted host-Float boundary.
Use `IEEE32Exec` when the binary32 endpoint calculation itself must be connected to Lean proofs.
-/
instance instBoundOpsFloat : BoundOps Float where
  addDown a b := HostFloat.nextDown (a + b)
  addUp a b := HostFloat.nextUp (a + b)
  subDown a b := HostFloat.nextDown (a - b)
  subUp a b := HostFloat.nextUp (a - b)
  mulDown a b := HostFloat.nextDown (a * b)
  mulUp a b := HostFloat.nextUp (a * b)

/--
Nonlinear enclosures supplied by host binary64 arithmetic.

Division and square root use hardware operations widened by one adjacent binary64 value. We do not
make the same claim for host transcendental-library calls, so `exp` and `log` remain unsupported.
-/
instance instNonlinearBoundOpsFloat : NonlinearBoundOps Float where
  divBounds aLo aHi bLo bHi :=
    if NonlinearBoundOps.denominatorAvoidsZero bLo bHi then
      let p1 := aLo / bLo
      let p2 := aLo / bHi
      let p3 := aHi / bLo
      let p4 := aHi / bHi
      let lo := NonlinearBoundOps.min4 p1 p2 p3 p4
      let hi := NonlinearBoundOps.max4 p1 p2 p3 p4
      some (HostFloat.nextDown lo, HostFloat.nextUp hi)
    else
      none
  expBounds := fun _ _ => none
  logBounds := fun _ _ => none
  sqrtBounds lo hi :=
    if hi < Numbers.zero then
      none
    else
      let lo' := if lo > Numbers.zero then lo else Numbers.zero
      some (HostFloat.nextDown (MathFunctions.sqrt lo'),
        HostFloat.nextUp (MathFunctions.sqrt hi))
  sigmoidBounds := fun _ _ => some (Numbers.zero, Numbers.one)
  tanhBounds := fun _ _ => some (Numbers.negOne, Numbers.one)
  sinBounds := fun _ _ => some (Numbers.negOne, Numbers.one)
  cosBounds := fun _ _ => some (Numbers.negOne, Numbers.one)
  layerNormAbsBound := fun _ => none
  supportsIdealCoupledDerivatives := false

/-!
## Native binary32 endpoints

The same one-ULP widening policy is available for Lean's native `Float32`. The operations execute
with binary32 rounding; moving to the adjacent representable value turns each nearest-rounded
result into an outward endpoint.
-/

namespace HostFloat32

def signMask : UInt32 := 0x80000000
def posInfBits : UInt32 := 0x7f800000
def negInfBits : UInt32 := 0xff800000

/-- Adjacent binary32 value above `x`, preserving NaNs and positive infinity. -/
def nextUp (x : Float32) : Float32 :=
  let bits := x.toBits
  if x.isNaN || bits = posInfBits then
    x
  else if bits = signMask || bits = 0 then
    Float32.ofBits 1
  else if bits &&& signMask = 0 then
    Float32.ofBits (bits + 1)
  else
    Float32.ofBits (bits - 1)

/-- Adjacent binary32 value below `x`, preserving NaNs and negative infinity. -/
def nextDown (x : Float32) : Float32 :=
  let bits := x.toBits
  if x.isNaN || bits = negInfBits then
    x
  else if bits = signMask || bits = 0 then
    Float32.ofBits (signMask + 1)
  else if bits &&& signMask = 0 then
    Float32.ofBits (bits - 1)
  else
    Float32.ofBits (bits + 1)

end HostFloat32

/-- Outward-widened native binary32 operations. -/
instance instBoundOpsFloat32 : BoundOps Float32 where
  addDown a b := HostFloat32.nextDown (a + b)
  addUp a b := HostFloat32.nextUp (a + b)
  subDown a b := HostFloat32.nextDown (a - b)
  subUp a b := HostFloat32.nextUp (a - b)
  mulDown a b := HostFloat32.nextDown (a * b)
  mulUp a b := HostFloat32.nextUp (a * b)

/-- Native binary32 nonlinear enclosures for division and square root. -/
instance instNonlinearBoundOpsFloat32 : NonlinearBoundOps Float32 where
  divBounds aLo aHi bLo bHi :=
    if NonlinearBoundOps.denominatorAvoidsZero bLo bHi then
      let p1 := aLo / bLo
      let p2 := aLo / bHi
      let p3 := aHi / bLo
      let p4 := aHi / bHi
      let lo := NonlinearBoundOps.min4 p1 p2 p3 p4
      let hi := NonlinearBoundOps.max4 p1 p2 p3 p4
      some (HostFloat32.nextDown lo, HostFloat32.nextUp hi)
    else
      none
  expBounds := fun _ _ => none
  logBounds := fun _ _ => none
  sqrtBounds lo hi :=
    if hi < 0 then none
    else some (HostFloat32.nextDown (Float32.sqrt (max lo 0)),
      HostFloat32.nextUp (Float32.sqrt hi))
  sigmoidBounds := fun _ _ => none
  tanhBounds := fun _ _ => none
  sinBounds := fun _ _ => some (-1, 1)
  cosBounds := fun _ _ => some (-1, 1)
  layerNormAbsBound := fun _ => none
  supportsIdealCoupledDerivatives := false

end NN.MLTheory.CROWN
