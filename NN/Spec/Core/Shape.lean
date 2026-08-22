/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Logic.Basic

import Init.Grind
import Mathlib.Data.Nat.Init

/-!
# Shapes (`Spec.Shape`)

`Shape` is the type-level “shape descriptor” for tensors in the spec layer.

TorchLean uses *shape-indexed tensors*:

`Tensor α s`

so `Shape` is how we encode the structure of `s` in a way Lean can use for both computation and
proofs.

## Representation

`Shape` is an inductive tree:

- `.scalar`
- `.dim n s`  (a length-`n` dimension whose entries have shape `s`)

This matches the tensor definition in `NN/Spec/Core/Tensor/Core.lean`.

## Common utilities

- `Spec.Shape.size : Shape → Nat` is the total number of scalar elements (“numel”).
- `Shape.toList : Shape → List Nat` is a convenient runtime view used by front-ends and bridges.

PyTorch analogy:

- `Shape.toList s` corresponds to `tensor.shape` (a tuple of dimensions).
- `Spec.Shape.rank s` corresponds to `tensor.ndim`.
- `Spec.Shape.size s` corresponds to `tensor.numel()`.

## Broadcasting and axes

Broadcasting is encoded via `CanBroadcastTo` / `BroadcastTo`.

This is an intentionally *asymmetric* relation ("broadcast `s1` to `s2`"), because most tensor code
is naturally written by choosing the output shape and requiring each input to broadcast to it.

The typeclass wrapper `BroadcastTo` keeps higher-level specs readable: in many cases Lean can infer
the broadcast evidence automatically, so call sites don’t have to manually thread proofs around.

It also defines axis-validity helpers (`NonemptyAxis`) and a `well_formed` predicate for “all
dimensions are positive”, which is useful when you want to rule out degenerate cases in proofs.
-/

@[expose] public section


namespace Spec

/-!
We represent shapes as an inductive tree instead of a bare `List Nat` because:

- it matches the tensor representation (`Tensor α s`) structurally, so many definitions are simple
  structural recursion,
- it keeps "scalar vs dim" cases explicit (important for proofs),
- it gives definitional equalities that are friendlier than lists in many places.
-/
/--
Tensor shape descriptor used to index spec-level tensors (`Spec.Tensor α s`).

`Shape` is an outermost-first tree:
- `.scalar` for a scalar,
- `.dim n s` for a length-`n` dimension whose entries have shape `s`.
-/
inductive Shape where
  | scalar : Shape
  | dim : Nat → Shape → Shape
deriving DecidableEq, Repr

namespace Shape

/--
Output length of a floor-mode sliding window with symmetric padding.

For positive `kernel` and `stride` with
$\mathtt{kernel}\le\mathtt{input}+2\mathtt{padding}$, this is
$(\mathtt{input}+2\mathtt{padding}-\mathtt{kernel})/\mathtt{stride}+1$. Invalid geometry has length zero, so saturated
natural-number subtraction and division by zero cannot create a phantom output element.
-/
def slidingWindowOutDim (input kernel stride padding : Nat) : Nat :=
  let padded := input + 2 * padding
  if kernel = 0 || stride = 0 || padded < kernel then
    0
  else
    (padded - kernel) / stride + 1

/-- Build a shape from a list of dimensions (outermost first). -/
def ofList : List Nat → Shape
  | [] => .scalar
  | n :: ns => .dim n (ofList ns)

/-- Pretty-print a `Shape` for debugging / logs. -/
def pretty (s : Shape) : String :=
  match s with
  | .scalar => "scalar"
  | .dim n rest => s!"dim {n} ({pretty rest})"

/-- Swap two adjacent dimensions at a given depth (0‑based from the outermost). -/
def swapAdjacentAtDepth (s : Shape) (depth : Nat) : Shape :=
  match depth, s with
  | 0, .dim m (.dim n rest) => .dim n (.dim m rest)
  | d+1, .dim m rest => .dim m (swapAdjacentAtDepth rest d)
  | _, _ => s  -- invalid depth, return unchanged

/-- Swapping adjacent dims at depth `depth` twice returns the original shape. -/
theorem swapAdjacentAtDepth_involutive (s : Shape) (depth : Nat) :
    (s.swapAdjacentAtDepth depth).swapAdjacentAtDepth depth = s := by
  induction depth generalizing s with
  | zero =>
      cases s with
      | scalar => simp [swapAdjacentAtDepth]
      | dim m rest =>
          cases rest <;> simp [swapAdjacentAtDepth]
  | succ d ih =>
      cases s <;> simp [swapAdjacentAtDepth, ih]

/-- Shape obtained by applying adjacent-axis swaps from left to right. -/
def applyAdjacentSwaps : Shape → List Nat → Shape
  | s, [] => s
  | s, depth :: depths => applyAdjacentSwaps (s.swapAdjacentAtDepth depth) depths

/-- Adjacent swaps that move `axis` to the innermost position of a rank-`rank` shape. -/
def moveAxisToLastSwaps (rank axis : Nat) : List Nat :=
  (List.range (rank - (axis + 1))).map (axis + ·)

@[simp]
theorem applyAdjacentSwaps_append (s : Shape) (xs ys : List Nat) :
    applyAdjacentSwaps s (xs ++ ys) = applyAdjacentSwaps (applyAdjacentSwaps s xs) ys := by
  induction xs generalizing s with
  | nil => rfl
  | cons depth depths ih =>
      simp only [List.cons_append, applyAdjacentSwaps]
      exact ih (s.swapAdjacentAtDepth depth)

/-- Replaying adjacent-axis swaps in reverse order restores the original shape. -/
@[simp]
theorem applyAdjacentSwaps_reverse (s : Shape) (depths : List Nat) :
    applyAdjacentSwaps (applyAdjacentSwaps s depths) depths.reverse = s := by
  induction depths generalizing s with
  | nil => rfl
  | cons depth depths ih =>
      simp only [applyAdjacentSwaps, List.reverse_cons, applyAdjacentSwaps_append]
      rw [ih]
      exact swapAdjacentAtDepth_involutive s depth

/-- Append a new innermost dimension. -/
@[reducible]
def appendDim (s : Shape) (n : Nat) : Shape :=
  match s with
  | .scalar => .dim n .scalar
  | .dim m rest => .dim m (appendDim rest n)

/-- Concatenate two shapes, preserving the dimensions of the first shape as leading axes. -/
@[reducible]
def concat : Shape → Shape → Shape
  | .scalar, suffix => suffix
  | .dim n rest, suffix => .dim n (concat rest suffix)

/-- Appending one dimension is concatenation with a one-axis suffix. -/
theorem appendDim_eq_concat (s : Shape) (n : Nat) :
    s.appendDim n = s.concat (.dim n .scalar) := by
  induction s with
  | scalar => rfl
  | dim m rest ih => simp only [appendDim, concat, ih]

/-- Appending two dimensions is concatenation with a two-axis suffix. -/
theorem appendDim_appendDim_eq_concat (s : Shape) (m n : Nat) :
    (s.appendDim m).appendDim n = s.concat (.dim m (.dim n .scalar)) := by
  induction s with
  | scalar => rfl
  | dim k rest ih => simp only [appendDim, concat, ih]

/-- Appending a final dimension commutes with adding a fixed leading shape. -/
@[simp]
theorem concat_appendDim (leading suffix : Shape) (n : Nat) :
    (leading.concat suffix).appendDim n = leading.concat (suffix.appendDim n) := by
  induction leading with
  | scalar => rfl
  | dim m rest ih =>
      simp only [concat, appendDim, ih]

/-- Total number of scalar elements (a.k.a. “numel”). -/
def size : Shape → Nat
  | .scalar => 1
  | .dim n rest => n * size rest

/-- The number of entries in a list-shaped tensor is the product of its dimensions. -/
@[simp] theorem size_ofList (dims : List Nat) :
    size (ofList dims) = dims.prod := by
  induction dims with
  | nil => rfl
  | cons dim dims ih => simp [ofList, size, ih]

/-- A shape consisting only of singleton axes contains one scalar. -/
@[simp] theorem size_ofList_replicate_one (n : Nat) :
    size (ofList (List.replicate n 1)) = 1 := by
  induction n with
  | zero => rfl
  | succ n ih =>
      rw [List.replicate_succ]
      simp [ofList, size, ih]

/-- `size` for a 2D shape factors as $ab\,\operatorname{size}(s)$. -/
theorem size_dim_mul (a b : Nat) (s : Shape) :
    size (dim a (dim b s)) = a * b * size s := by
  simp [size, Nat.mul_assoc]

/--
`appendDim` multiplies the number of scalar elements by the appended dimension.

This lemma is the standard justification for reshape tricks where we:
- treat a tensor of shape `s.appendDim n` as a matrix of shape `(size s) × n`, or
- append an extra singleton dimension (`n = 1`) without changing `size`.
-/
theorem size_appendDim (s : Shape) (n : Nat) : size (appendDim s n) = size s * n := by
  induction s with
  | scalar =>
      simp [size]
  | dim m rest ih =>
      -- `appendDim` recurses to the innermost dimension; `size` is multiplicative.
      simp [size, ih, Nat.mul_assoc]

/-- The number of elements in a concatenated shape is the product of the two shape sizes. -/
theorem size_concat (leading suffix : Shape) :
    size (concat leading suffix) = size leading * size suffix := by
  induction leading with
  | scalar => simp [size]
  | dim n rest ih => simp [size, ih, Nat.mul_assoc]

/--
Shape-size identity used in Transformer attention reshapes.

If $\mathtt{dModel}=\mathtt{numHeads}\cdot\mathtt{headDim}$, then
$(\mathtt{seqLen}\times\mathtt{dModel})$ has the same `size` as
$(\mathtt{numHeads}\times\mathtt{seqLen}\times\mathtt{headDim})$.
-/
theorem size_eq_of_dModel_eq_numHeads_mul_headDim
  (seqLen numHeads dModel headDim : Nat)
  (h : dModel = numHeads * headDim) :
  size (dim seqLen (dim dModel scalar)) = size (dim numHeads (dim seqLen (dim headDim scalar))) :=
    by
  simp [size]
  rw [h]
  -- `Nat.mul_left_comm` proves `a * b * c = b * a * c`; we just reassociate to match our goal.
  simpa [Nat.mul_assoc] using Nat.mul_left_comm seqLen numHeads headDim


/-- Convert to a list of dimensions (outermost first). -/
def toList : Shape → List Nat
  | .scalar => []
  | .dim n rest => n :: toList rest

/-- `ofList` is a left inverse of `toList`. -/
@[simp] theorem ofList_toList (s : Shape) : ofList (toList s) = s := by
  induction s with
  | scalar => rfl
  | dim n s ih =>
    simp [ofList, toList, ih]

/-- `toList` is a right inverse of `ofList`. -/
@[simp] theorem toList_ofList (xs : List Nat) : toList (ofList xs) = xs := by
  induction xs with
  | nil => rfl
  | cons n ns ih =>
    simp [ofList, toList, ih]

-- Tell `grind` about the standard shape normalization lemmas.
attribute [grind =] size_appendDim size_ofList ofList_toList toList_ofList

/-- Convert to an array of dimensions (outermost first). -/
def toArray (s : Shape) : Array Nat :=
  toList s |>.toArray

/-- Boolean equality test for shapes (structural). -/
def areEqual : Shape → Shape → Bool
  | .scalar, .scalar => true
  | .dim n1 s1, .dim n2 s2 => n1 == n2 && areEqual s1 s2
  | _, _ => false

-- We keep `BEq` as an explicit structural test because it shows up in runtime checks and logs.
/-- `BEq Shape` uses the explicit structural boolean test `Shape.areEqual`. -/
instance : BEq Shape where
  beq := areEqual

/-- Default inhabitant for `Shape`, used only when Lean needs a canonical fallback value. -/
instance : Inhabited Shape where
  default := .scalar

/-- Return whether the shape has no tensor dimensions. -/
def isScalar : Shape → Bool
  | .scalar => true
  | _ => false

/-- Get dimension at index `i` (0‑based), or `none` if out of bounds. -/
def getDim : Shape → Nat → Option Nat
  | .scalar, _ => none
  | .dim n _, 0 => some n
  | .dim _ rest, i+1 => getDim rest i

/- Broadcasting support -/
/-!
### Typeclass-friendly broadcasting (`BroadcastTo`)

The `CanBroadcastTo` relation is asymmetric (“broadcast `s₁` *to* `s₂`”), matching how most
operations are written: we pick a target shape and require each operand to broadcast to it.

The `BroadcastTo` wrapper lets Lean search for a broadcast proof automatically, which is convenient
for higher-level specs (layers/models) where the broadcasting details are not the point.

PyTorch analogy:

- PyTorch broadcasting aligns shapes from the *trailing* dimensions by implicitly prepending `1`s
  to the shorter shape.
- Our `Shape` is an outermost-first tree, so the corresponding operation is `expand_dims`:
  it inserts leading/outer dimensions to reach the target rank (this is the "prepend `1`s" step).
- `dim_1_to_n` corresponds to PyTorch's "dimension 1 can expand to n" rule.
-/

/-- Rank = number of dimensions (scalar has rank 0). -/
def rank : Shape → Nat
  | Shape.scalar => 0
  | Shape.dim _ rest => 1 + rank rest

/-- Swap the first two axes after an arbitrary fixed leading shape. -/
@[simp]
theorem swapAdjacentAtDepth_concat_rank (leading suffix : Shape) (m n : Nat) :
    swapAdjacentAtDepth (leading.concat (.dim m (.dim n suffix))) leading.rank =
      leading.concat (.dim n (.dim m suffix)) := by
  induction leading with
  | scalar => rfl
  | dim _ tail ih =>
      simp only [concat, rank, Nat.one_add, swapAdjacentAtDepth, ih]

/-- Replace every dimension by one while preserving the rank of a shape. -/
def singletonAxes : Shape → Shape
  | .scalar => .scalar
  | .dim _ rest => .dim 1 (singletonAxes rest)

@[simp] theorem rank_singletonAxes (s : Shape) : rank (singletonAxes s) = rank s := by
  induction s <;> simp [singletonAxes, rank, *]

@[simp] theorem size_singletonAxes (s : Shape) : size (singletonAxes s) = 1 := by
  induction s <;> simp [singletonAxes, size, *]

/-- Proposition used by broadcast constructors that align two existing dimensions. -/
class SameRank (s₁ s₂ : Shape) : Prop where
  /-- The two shapes have the same number of dimensions. -/
  rank_eq : rank s₁ = rank s₂

instance : SameRank .scalar .scalar := ⟨rfl⟩

instance sameRankRefl (s : Shape) : SameRank s s := ⟨rfl⟩

instance {s₁ s₂ : Shape} {n₁ n₂ : Nat} [tail : SameRank s₁ s₂] :
    SameRank (.dim n₁ s₁) (.dim n₂ s₂) :=
  ⟨by simpa [rank] using congrArg Nat.succ tail.rank_eq⟩

/-- Evidence that shape `s₁` can be broadcast to shape `s₂` using right-aligned dimensions.

The equal-dimension constructors require equal-rank tails. Consequently, every rank difference is
resolved by `expand_dims` before dimensions are compared, matching NumPy and PyTorch rather than
allowing a proof term to choose between left- and right-aligned interpretations. -/
inductive CanBroadcastTo : Shape → Shape → Type where
  /-- Scalar shapes agree. Higher-rank scalar broadcasts are built with `expand_dims`. -/
  | scalar : CanBroadcastTo .scalar .scalar
  /-- Matching outer dimensions preserve broadcasting of their tails. -/
  | dim_eq {n : Nat} {s₁ s₂ : Shape} [SameRank s₁ s₂]
      (tail : CanBroadcastTo s₁ s₂) :
      CanBroadcastTo (.dim n s₁) (.dim n s₂)
  /-- An outer dimension of length one can expand to any target length. -/
  | dim_1_to_n {n : Nat} {s₁ s₂ : Shape} [SameRank s₁ s₂]
      (tail : CanBroadcastTo s₁ s₂) :
      CanBroadcastTo (.dim 1 s₁) (.dim n s₂)
  /-- A new outer target dimension aligns a source of lower rank. -/
  | expand_dims {n : Nat} {s₁ s₂ : Shape} (tail : CanBroadcastTo s₁ s₂) :
      CanBroadcastTo s₁ (Shape.dim n s₂)
deriving Repr

/-- Every shape broadcasts to itself without expanding an axis. -/
def CanBroadcastTo.refl : (s : Shape) → CanBroadcastTo s s
  | .scalar => .scalar
  | .dim _ tail => .dim_eq (refl tail)

/-- The canonical witness that broadcasts a scalar by inserting every target dimension. -/
def CanBroadcastTo.scalarTo : (s : Shape) → CanBroadcastTo .scalar s
  | .scalar => .scalar
  | .dim _ tail => .expand_dims (scalarTo tail)

/-- Broadcast a shape of singleton axes to any shape of the same rank. -/
def CanBroadcastTo.singletonAxes : (s : Shape) → CanBroadcastTo (Shape.singletonAxes s) s
  | .scalar => .scalar
  | .dim _ tail =>
      letI : SameRank (Shape.singletonAxes tail) tail := ⟨rank_singletonAxes tail⟩
      .dim_1_to_n (singletonAxes tail)

/-- Broadcast a suffix across an arbitrary collection of newly prepended target dimensions. -/
def CanBroadcastTo.prependTarget : (leading suffix : Shape) →
    CanBroadcastTo suffix (Shape.concat leading suffix)
  | .scalar, suffix => refl suffix
  | .dim _ tail, suffix => .expand_dims (prependTarget tail suffix)

/-- Typeclass wrapper for `CanBroadcastTo` so broadcast proofs can be inferred. -/
class BroadcastTo (s₁ s₂ : Shape) where
  proof : CanBroadcastTo s₁ s₂

/-- Scalar shapes broadcast directly. Leading target dimensions are inferred by `expand_dims`. -/
instance broadcastToScalar : BroadcastTo Shape.scalar Shape.scalar where
  proof := CanBroadcastTo.scalar

/-- Broadcasting preserves equal leading dimensions when the tails broadcast. -/
instance broadcastToDimEq {n : Nat} {s₁ s₂ : Shape} [SameRank s₁ s₂]
    [bc : BroadcastTo s₁ s₂] : BroadcastTo (Shape.dim n s₁) (Shape.dim n s₂) where
  proof := CanBroadcastTo.dim_eq bc.proof

/-- Dimension `1` can broadcast to any `n` (PyTorch's main broadcast rule). -/
instance broadcastToDim1ToN {n : Nat} {s₁ s₂ : Shape} [SameRank s₁ s₂]
    [bc : BroadcastTo s₁ s₂] : BroadcastTo (Shape.dim 1 s₁) (Shape.dim n s₂) where
  proof := CanBroadcastTo.dim_1_to_n bc.proof

/-- Prepend an outer dimension (the "expand_dims" step used to align ranks). -/
instance broadcastToExpandDims {n : Nat} {s₁ s₂ : Shape} [bc : BroadcastTo s₁ s₂] :
    BroadcastTo s₁ (Shape.dim n s₂) where
  proof := CanBroadcastTo.expand_dims bc.proof

/-- `true` iff two shapes have the same number of elements. -/
def isValidReshape (s₁ s₂ : Shape) : Bool :=
  Spec.Shape.size s₁ == Spec.Shape.size s₂

/-- Swap adjacent entries in an axis-ordering list, leaving invalid positions unchanged. -/
def swapAdjacentAxes (axes : List Nat) (depth : Nat) : List Nat :=
  match axes, depth with
  | [], _ => []
  | [axis], _ => [axis]
  | first :: second :: rest, 0 => second :: first :: rest
  | first :: rest, depth + 1 => first :: swapAdjacentAxes rest depth

/-- Permute axes of a shape using a runtime permutation list (0-based). Returns `none` if invalid.
  -/
def permute? (s : Shape) (perm : List Nat) : Option Shape :=
  let r := rank s
  if perm.length != r then
    none
  else if !(decide perm.Nodup) then
    none
  else
    let dims := toList s
    (perm.mapM fun i => dims[i]?).map ofList

/-!
## Axis evidence

Axes are zero-based natural numbers. `AxisInBounds axis s` says only that the axis exists;
`HasNonemptyAxis axis s` additionally says that its extent is positive. Shape-preserving operations
such as softmax need the first condition. Reductions whose definition selects an element, such as
maximum and minimum, use the second.

Negative axes are normalized by frontends before reaching this layer. The innermost axis of a
positive-rank shape is `s.rank - 1`.
-/

/-- The zero-based `axis` names a dimension of `s`. Its extent may be zero. -/
class AxisInBounds (axis : Nat) (s : Shape) : Prop where
  /-- The axis is strictly smaller than the shape rank. -/
  proof : axis < rank s

/-- Looking up an axis below the rank of a shape succeeds. -/
theorem getDim_isSome_of_lt {s : Shape} {axis : Nat} (h : axis < rank s) :
    (getDim s axis).isSome = true := by
  induction s generalizing axis with
  | scalar => simp [rank] at h
  | dim n rest ih =>
      cases axis with
      | zero => rfl
      | succ axis =>
          simp only [getDim]
          apply ih
          grind [rank]

/-- Extent of a statically valid axis. -/
def axisSize (s : Shape) (axis : Nat) [h : AxisInBounds axis s] : Nat :=
  (getDim s axis).get (getDim_isSome_of_lt h.proof)

/-- The outermost dimension is axis zero, regardless of its extent. -/
instance axisInBoundsZero {n s} : AxisInBounds 0 (.dim n s) :=
  ⟨by simpa [rank, Nat.add_comm] using Nat.zero_lt_succ s.rank⟩

/-- An inner axis remains in bounds under an additional outer dimension. -/
instance axisInBoundsSucc {n s axis} [h : AxisInBounds axis s] :
    AxisInBounds (axis + 1) (.dim n s) :=
  ⟨by simpa [rank, Nat.add_comm] using Nat.add_lt_add_right h.proof 1⟩

/-- Decide whether a natural number names a dimension of `s`, returning typed evidence. -/
def axisInBounds? (axis : Nat) (s : Shape) : Option (PLift (AxisInBounds axis s)) :=
  if h : axis < rank s then some ⟨⟨h⟩⟩ else none

/-- The decision procedure succeeds whenever static in-bounds evidence is available. -/
theorem axisInBounds?_isSome {axis : Nat} {s : Shape} [h : AxisInBounds axis s] :
    (axisInBounds? axis s).isSome := by
  simp [axisInBounds?, h.proof]

/-- A positive-rank shape always has an innermost dimension. -/
theorem axisInBoundsLast {s : Shape} (hRank : 0 < rank s) :
    AxisInBounds (rank s - 1) s :=
  ⟨Nat.sub_lt hRank Nat.zero_lt_one⟩

/--
`NonemptyAxis axis s` says that `axis` selects a positive-length dimension of `s`.

The constructors follow the recursive representation of `Shape`, so reduction definitions can
eliminate this evidence while recursing through outer dimensions.
-/
inductive NonemptyAxis : Nat → Shape → Prop
  | zero {n : Nat} {s : Shape} : NonemptyAxis 0 (.dim (n + 1) s)
  | succ {n : Nat} {s : Shape} {axis : Nat} :
      NonemptyAxis axis s → NonemptyAxis (axis + 1) (.dim n s)

/-- A nonempty axis is, in particular, a valid axis. -/
theorem NonemptyAxis.toAxisInBounds {axis : Nat} {s : Shape} (h : NonemptyAxis axis s) :
    AxisInBounds axis s := by
  induction h with
  | zero => exact axisInBoundsZero
  | succ inner ih =>
      constructor
      simpa [rank, Nat.add_comm] using Nat.add_lt_add_right ih.proof 1

/-- Axis zero is nonempty when the outer dimension is positive. -/
@[simp] theorem nonemptyAxis_zero {n : Nat} {s : Shape} :
    NonemptyAxis 0 (.dim (n + 1) s) :=
  .zero

/-- Nonemptiness of an inner axis is unchanged by an outer dimension. -/
@[simp] theorem nonemptyAxis_succ {n : Nat} {s : Shape} {axis : Nat}
    (h : NonemptyAxis axis s) : NonemptyAxis (axis + 1) (.dim n s) :=
  .succ h

/--
Return evidence that `axis` addresses a positive dimension of `s`.

Executable consumers use this to recover the proposition required by typed tensor operations from
a raw runtime axis. Invalid axes, including axes into zero-sized dimensions, return `none`.
-/
def nonemptyAxis? (axis : Nat) : (s : Shape) → Option (PLift (NonemptyAxis axis s))
  | .scalar => none
  | .dim n rest =>
      match axis, n with
      | 0, Nat.succ k => some ⟨NonemptyAxis.zero (n := k) (s := rest)⟩
      | 0, 0 => none
      | Nat.succ inner, n =>
          (nonemptyAxis? inner rest).map (fun h =>
            ⟨NonemptyAxis.succ (n := n) (s := rest) (axis := inner) h.down⟩)

/-- Typeclass wrapper used when nonempty-axis evidence can be inferred statically. -/
class HasNonemptyAxis (axis : Nat) (s : Shape) : Prop where
  /-- The selected dimension has positive extent. -/
  proof : NonemptyAxis axis s

/-- Instance: axis `0` is valid for any positive outer dimension. -/
instance hasNonemptyAxisZero {n s} : HasNonemptyAxis 0 (.dim (n + 1) s) :=
  ⟨NonemptyAxis.zero⟩

/-- Package a proof that the outer dimension is nonzero as axis evidence. -/
theorem hasNonemptyAxisZeroOfNe {n s} (h : n ≠ 0) : HasNonemptyAxis 0 (.dim n s) :=
  let ⟨m, hm⟩ := Nat.exists_eq_succ_of_ne_zero h
  ⟨by rw [hm]; exact .zero⟩

/-- Package positivity of the outer dimension as axis evidence. -/
theorem hasNonemptyAxisZeroOfPos {n s} (h : 0 < n) : HasNonemptyAxis 0 (.dim n s) :=
  hasNonemptyAxisZeroOfNe (Nat.ne_of_gt h)

/-- Instance: axis `1` is valid for a 2D shape when the selected dimension is nonzero. -/
theorem hasNonemptyAxisOne {n₁ n₂ s} (h₂ : n₂ ≠ 0) :
    HasNonemptyAxis 1 (.dim n₁ (.dim n₂ s)) :=
  ⟨NonemptyAxis.succ (hasNonemptyAxisZeroOfNe h₂).proof⟩

/-- Nonemptiness of an inner axis is unchanged by an outer dimension. -/
instance hasNonemptyAxisSucc {n s axis} [h : HasNonemptyAxis axis s] :
    HasNonemptyAxis (axis + 1) (.dim n s) :=
  ⟨NonemptyAxis.succ h.proof⟩


/-!
## Well-formedness (`well_formed`)

`well_formed s` means "all dimensions are positive".

Why this matters (and why we designed it this way):

- Many definitions use `Fin n` indexing; if `n = 0`, there is no index and you end up with either
  vacuous truths or extra cases that obscure the intent of the lemma.
- Some common ops become awkward or partial at `n = 0`. For example, a mean typically divides by
  the number of elements, so `n = 0` needs special-case semantics.
- PyTorch *does* allow zero-sized dimensions, and most ops define a sensible result for them. We
  intentionally keep that complexity out of the core spec layer because it makes proofs much more
  case-heavy. When we need zero-dimension tensors, we introduce them with explicit
  semantics instead of relying on incidental behavior.

This is a pragmatic choice: proofs and specs are shorter, and
runtime checks can still handle edge cases separately.
-/
-- Well-formed shapes have positive dimensions.
/-- `well_formed s` means "all dimensions of `s` are positive" (recursively). -/
def wellFormed : Shape → Prop
| .scalar => True
| .dim n s => n > 0 ∧ s.wellFormed

/-!
### Size positivity

If all dimensions of a shape are positive, then the total number of scalar elements is positive.

This is a small but useful bridge lemma: many reductions are only defined for nonempty dimensions,
and `WellFormed` is our standard way of expressing that assumption.
-/

/-- If `s.well_formed`, then `Spec.Shape.size s > 0`. -/
theorem size_pos_of_well_formed : ∀ {s : Shape}, s.wellFormed → 0 < Spec.Shape.size s
  | .scalar, _ => by
      simp [Spec.Shape.size]
  | .dim n s, hw => by
      rcases hw with ⟨hn, hs⟩
      simpa [Spec.Shape.size] using Nat.mul_pos hn (size_pos_of_well_formed (s := s) hs)

/--
The innermost axis of a positive-rank, well-formed shape is nonempty.
-/
theorem nonemptyLastAxis {s : Shape} (hRank : 0 < s.rank) (hw : s.wellFormed) :
    NonemptyAxis (s.rank - 1) s := by
  induction s with
  | scalar => simp [rank] at hRank
  | dim n rest ih =>
      rcases hw with ⟨hn, hRestWf⟩
      by_cases hRestRank : rest.rank = 0
      · obtain ⟨m, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (Nat.ne_of_gt hn)
        simp [rank, hRestRank]
      · have hInner := ih (Nat.pos_of_ne_zero hRestRank) hRestWf
        have hLifted := NonemptyAxis.succ (n := n) hInner
        simpa [rank, Nat.sub_add_cancel (Nat.one_le_iff_ne_zero.mpr hRestRank)] using hLifted

/-- Package `nonemptyLastAxis` as inferred reduction-axis evidence. -/
theorem hasNonemptyLastAxis {s : Shape} (hRank : 0 < s.rank) (hw : s.wellFormed) :
    HasNonemptyAxis (s.rank - 1) s :=
  ⟨nonemptyLastAxis hRank hw⟩

/--
Typeclass wrapper for `Shape.well_formed`.

We use a typeclass (instead of passing a `well_formed` proof everywhere) because it mirrors how
other "side conditions" are handled in the library: call sites stay clean, and instances can be
provided locally (e.g. `letI : Shape.WellFormed s := ...`) when needed.
-/
class WellFormed (s : Shape) : Prop where
  proof : s.wellFormed

-- Scalars are always well-formed.
/-- Scalars are always well-formed. -/
instance : WellFormed .scalar where
  proof := trivial

/-- Adding a nonzero dimension preserves well-formedness. -/
instance {n s} [Shape.WellFormed s] [NeZero n] : Shape.WellFormed (.dim n s) :=
  ⟨⟨Nat.pos_of_ne_zero (NeZero.ne n), Shape.WellFormed.proof⟩⟩

/-- Infer that the innermost axis is nonempty from `WellFormed s`. -/
theorem inferNonemptyLastAxis {s : Shape} [hw : WellFormed s] (hRank : 0 < s.rank) :
    HasNonemptyAxis (s.rank - 1) s :=
  hasNonemptyLastAxis hRank hw.proof

/-!
`padLeft n s` prepends `n` singleton dimensions to a shape.

PyTorch analogy: `unsqueeze(0)` repeated `n` times (or equivalently viewing a tensor as having
extra leading dimensions of size 1). This is also the "prepend 1s" step you see in broadcasting.
-/
/-- Prepend `n` leading singleton dimensions (size `1`) to a shape. -/
def padLeft : Nat → Shape → Shape
| 0, s => s
| (n+1), s => dim 1 (padLeft n s)

-- Padding with leading `1`s increases rank by exactly `n`.
/-- `padLeft n s` increases the rank by exactly `n`. -/
theorem padLeft_rank : ∀ n s, (padLeft n s).rank = n + s.rank
| 0, s => by simp [padLeft]
| n+1, s => by
  simp [padLeft, rank]
  rw [padLeft_rank n]
  grind

end Shape
end Spec
