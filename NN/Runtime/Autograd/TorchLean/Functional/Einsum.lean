/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.Functional.Core

@[expose] public section


namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor

namespace F

/-! ## Einsum-ish building blocks -/

/-- Matrix matmul: `[m,n] × [n,p] → [m,p]`. -/
def matmul2d {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {mDim nDim pDim : Nat}
    (a : RefTy (m := m) (α := α) (.dim mDim (.dim nDim .scalar)))
    (b : RefTy (m := m) (α := α) (.dim nDim (.dim pDim .scalar))) :
    m (RefTy (m := m) (α := α) (.dim mDim (.dim pDim .scalar))) :=
  _root_.Runtime.Autograd.Torch.matmul (m := m) (α := α) (mDim := mDim) (nDim := nDim) (pDim :=
    pDim) a b

/-- Batched matmul: `[batch,m,n] × [batch,n,p] → [batch,m,p]`. -/
def bmm {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch mDim nDim pDim : Nat}
    (a : RefTy (m := m) (α := α) (.dim batch (.dim mDim (.dim nDim .scalar))))
    (b : RefTy (m := m) (α := α) (.dim batch (.dim nDim (.dim pDim .scalar)))) :
    m (RefTy (m := m) (α := α) (.dim batch (.dim mDim (.dim pDim .scalar)))) :=
  _root_.Runtime.Autograd.Torch.bmm (m := m) (α := α) (batch := batch) (mDim := mDim) (nDim := nDim)
    (pDim := pDim) a b

/-!
## Typed einsum wrappers (fast, total)

These are non-`Option` equivalents for the most common einsum contractions in ML code.
They are intended to be used directly (no string parsing), and serve as the fast-path targets for
`einsumDyn`.
-/

/-- `einsum("ij,jk->ik", A, B)` as a typed matmul. -/
def einsumIjJkIk {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {iDim jDim kDim : Nat}
    (a : RefTy (m := m) (α := α) (.dim iDim (.dim jDim .scalar)))
    (b : RefTy (m := m) (α := α) (.dim jDim (.dim kDim .scalar))) :
    m (RefTy (m := m) (α := α) (.dim iDim (.dim kDim .scalar))) :=
  matmul2d (m := m) (α := α) (mDim := iDim) (nDim := jDim) (pDim := kDim) a b

/-- `einsum("bij,bjk->bik", A, B)` as a typed batched matmul. -/
def einsumBijBjkBik {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch iDim jDim kDim : Nat}
    (a : RefTy (m := m) (α := α) (.dim batch (.dim iDim (.dim jDim .scalar))))
    (b : RefTy (m := m) (α := α) (.dim batch (.dim jDim (.dim kDim .scalar)))) :
    m (RefTy (m := m) (α := α) (.dim batch (.dim iDim (.dim kDim .scalar)))) :=
  bmm (m := m) (α := α) (batch := batch) (mDim := iDim) (nDim := jDim) (pDim := kDim) a b

/-- Einsum pattern used in attention: `bhid,bhjd -> bhij` (batched Q·Kᵀ per head). -/
def einsumBhidBhjdBhij {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch heads iDim jDim dDim : Nat}
    (q : RefTy (m := m) (α := α) (.dim batch (.dim heads (.dim iDim (.dim dDim .scalar)))))
    (k : RefTy (m := m) (α := α) (.dim batch (.dim heads (.dim jDim (.dim dDim .scalar))))) :
    m (RefTy (m := m) (α := α) (.dim batch (.dim heads (.dim iDim (.dim jDim .scalar))))) := do
  let bh : Nat := batch * heads
  let sQ4 : Shape := .dim batch (.dim heads (.dim iDim (.dim dDim .scalar)))
  let sK4 : Shape := .dim batch (.dim heads (.dim jDim (.dim dDim .scalar)))
  let sQ3 : Shape := .dim bh (.dim iDim (.dim dDim .scalar))
  let sK3 : Shape := .dim bh (.dim jDim (.dim dDim .scalar))
  let sKT : Shape := .dim bh (.dim dDim (.dim jDim .scalar))
  let sOut3 : Shape := .dim bh (.dim iDim (.dim jDim .scalar))
  let sOut4 : Shape := .dim batch (.dim heads (.dim iDim (.dim jDim .scalar)))
  have hQ : Shape.size sQ4 = Shape.size sQ3 := by
    simp [sQ4, sQ3, Shape.size, bh, Nat.mul_left_comm, Nat.mul_comm]
  have hK : Shape.size sK4 = Shape.size sK3 := by
    simp [sK4, sK3, Shape.size, bh, Nat.mul_left_comm, Nat.mul_comm]
  have hOut : Shape.size sOut3 = Shape.size sOut4 := by
    simp [sOut3, sOut4, Shape.size, bh, Nat.mul_left_comm, Nat.mul_comm]
  let q3 ← reshape (m := m) (α := α) (s₁ := sQ4) (s₂ := sQ3) q hQ
  let k3 ← reshape (m := m) (α := α) (s₁ := sK4) (s₂ := sK3) k hK
  let kt ← transpose3dLastTwo (m := m) (α := α) (a := bh) (b := jDim) (c := dDim) k3
  let out3 ← bmm (m := m) (α := α) (batch := bh) (mDim := iDim) (nDim := dDim) (pDim := jDim) q3 kt
  reshape (m := m) (α := α) (s₁ := sOut3) (s₂ := sOut4) out3 hOut

/-- Einsum pattern used in attention: `bhij,bhjd -> bhid` (batched Attn·V per head). -/
def einsumBhijBhjdBhid {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch heads iDim jDim dDim : Nat}
    (attn : RefTy (m := m) (α := α) (.dim batch (.dim heads (.dim iDim (.dim jDim .scalar)))))
    (v : RefTy (m := m) (α := α) (.dim batch (.dim heads (.dim jDim (.dim dDim .scalar))))) :
    m (RefTy (m := m) (α := α) (.dim batch (.dim heads (.dim iDim (.dim dDim .scalar))))) := do
  let bh : Nat := batch * heads
  let sA4 : Shape := .dim batch (.dim heads (.dim iDim (.dim jDim .scalar)))
  let sV4 : Shape := .dim batch (.dim heads (.dim jDim (.dim dDim .scalar)))
  let sA3 : Shape := .dim bh (.dim iDim (.dim jDim .scalar))
  let sV3 : Shape := .dim bh (.dim jDim (.dim dDim .scalar))
  let sOut3 : Shape := .dim bh (.dim iDim (.dim dDim .scalar))
  let sOut4 : Shape := .dim batch (.dim heads (.dim iDim (.dim dDim .scalar)))
  have hA : Shape.size sA4 = Shape.size sA3 := by
    simp [sA4, sA3, Shape.size, bh, Nat.mul_left_comm, Nat.mul_comm]
  have hV : Shape.size sV4 = Shape.size sV3 := by
    simp [sV4, sV3, Shape.size, bh, Nat.mul_left_comm, Nat.mul_comm]
  have hOut : Shape.size sOut3 = Shape.size sOut4 := by
    simp [sOut3, sOut4, Shape.size, bh, Nat.mul_left_comm, Nat.mul_comm]
  let a3 ← reshape (m := m) (α := α) (s₁ := sA4) (s₂ := sA3) attn hA
  let v3 ← reshape (m := m) (α := α) (s₁ := sV4) (s₂ := sV3) v hV
  let out3 ← bmm (m := m) (α := α) (batch := bh) (mDim := iDim) (nDim := jDim) (pDim := dDim) a3 v3
  reshape (m := m) (α := α) (s₁ := sOut3) (s₂ := sOut4) out3 hOut

/-! ## General einsum (PyTorch-style subscripts; runtime-checked) -/

namespace Einsum

-- Needed for runtime checks (e.g. to build a `valid_axis_inst` for reductions).
/-- Decidable instance for `Shape.well_formed`, used by the dynamic einsum lowering. -/
def wellFormedDec : (s : Shape) → Decidable s.wellFormed
  | .scalar => isTrue trivial
  | .dim n s =>
      match (inferInstance : Decidable (n > 0)) with
      | isTrue hn =>
          match wellFormedDec s with
          | isTrue hs => isTrue ⟨hn, hs⟩
          | isFalse hs => isFalse (fun h => hs h.2)
      | isFalse hn =>
          isFalse (fun h => hn h.1)

/-- Local decidability instance for `Shape.well_formed` (used by the dynamic einsum lowering). -/
instance (s : Shape) : Decidable s.wellFormed :=
  wellFormedDec s

/--
Label used by the dynamic einsum parser.

`chr c` is a concrete axis label like `'i'` or `'j'`, while `ell k` is a generated ellipsis label
that stands for "some number of unnamed batch-like axes".
-/
inductive Label where
  | chr (c : Char)
  | ell (k : Nat)
  deriving BEq, DecidableEq, Repr

/--
One operand’s subscript, split around an optional ellipsis.

For example, parsing `"ab...cd"` yields:
- `pre = ['a','b']`
- `post = ['c','d']`
- `hasEll = true`
-/
structure Subscript where
  /-- pre. -/
  pre : List Char
  /-- post. -/
  post : List Char
  /-- has Ell. -/
  hasEll : Bool
  deriving Repr

/-- Remove ASCII whitespace to simplify the hand-rolled parser. -/
def stripSpaces (s : String) : String :=
  String.ofList <| s.toList.filter (fun c => c != ' ' && c != '\n' && c != '\t' && c != '\r')

/-- Parse a single operand subscript (with at most one `...`). -/
def parseSubscript (raw : String) : Except String Subscript := do
  let s := stripSpaces raw
  let parts := s.splitOn "..."
  match parts with
  | [one] =>
      pure { pre := one.toList, post := [], hasEll := false }
  | [a, b] =>
      pure { pre := a.toList, post := b.toList, hasEll := true }
  | _ =>
      throw s!"einsum: invalid subscript `{raw}` (ellipsis `...` may appear at most once)"

/-- Parsed `einsum` equation: input subscripts and an optional explicit output subscript. -/
structure Parsed where
  /-- inputs. -/
  inputs : List Subscript
  output? : Option Subscript
  deriving Repr

/-- Parse an equation of the form `"a,b->c"` or `"a,b"` (implicit output). -/
def parseEquation (raw : String) : Except String Parsed := do
  let s := stripSpaces raw
  let parts := s.splitOn "->"
  match parts with
  | [lhs] =>
      let insRaw := lhs.splitOn ","
      let ins ← insRaw.mapM parseSubscript
      pure { inputs := ins, output? := none }
  | [lhs, rhs] =>
      let insRaw := lhs.splitOn ","
      let ins ← insRaw.mapM parseSubscript
      let out ← parseSubscript rhs
      pure { inputs := ins, output? := some out }
  | _ =>
      throw s!"einsum: invalid equation `{raw}` (expected `lhs` or `lhs->rhs`)"

/-- Detect whether a list of labels contains any duplicates (order-preserving scan). -/
def hasDupLabels (xs : List Label) : Bool :=
  let rec go (seen : List Label) : List Label → Bool
    | [] => false
    | x :: xs => if seen.contains x then true else go (x :: seen) xs
  go [] xs

/-- Find the first index of a label (like `List.findIdx?`, but returning an `Option Nat`). -/
def findIndex? (xs : List Label) (x : Label) : Option Nat :=
  let rec go (i : Nat) : List Label → Option Nat
    | [] => none
    | y :: ys => if y == x then some i else go (i + 1) ys
  go 0 xs

/-- Swap adjacent elements at position `d` (used to implement permutations via adjacent swaps). -/
def swapAt {α : Type} (xs : List α) (d : Nat) : List α :=
  match xs, d with
  | [], _ => []
  | [x], _ => [x]
  | x :: y :: rest, 0 => y :: x :: rest
  | x :: rest, d + 1 => x :: swapAt rest d

/--
Convert a permutation of axes into a sequence of adjacent swaps.

This mirrors the IR-side lowering strategy: represent a general permutation as a list of swaps at
depths, then implement swaps with `swapAdjacentAtDepth`.
-/
def swapDepthsForPerm? (perm : List Nat) (r : Nat) : Option (List Nat) :=
  let rec bubbleLeft (cur : List Nat) (swapsRev : List Nat) (i j : Nat) : List Nat × List Nat :=
    if j ≤ i then
      (cur, swapsRev)
    else
      bubbleLeft (swapAt cur (j - 1)) ((j - 1) :: swapsRev) i (j - 1)
  if perm.length = r && perm.all (fun d => d < r) then
    let rec go (i : Nat) (targets : List Nat) (cur : List Nat) (swapsRev : List Nat) :
        Option (List Nat) :=
      match targets with
      | [] => some swapsRev.reverse
      | target :: targets' =>
          match (cur.findIdx? (fun z => z = target)) with
          | none => none
          | some j =>
              let (cur', swapsRev') := bubbleLeft cur swapsRev i j
              go (i + 1) targets' cur' swapsRev'
    go 0 perm (List.range r) []
  else
    none

/--
Expand an input operand’s labels to a full label list matching the operand’s rank.

If the subscript contains an ellipsis, this inserts fresh `Label.ell` labels so that the total
label count matches `Shape.rank s`.
-/
def expandInputLabels (sub : Subscript) (s : Shape) (maxEll : Nat) : Except String (List Label) :=
  do
  let r := Shape.rank s
  let fixed := sub.pre.length + sub.post.length
  if sub.hasEll then
    if r < fixed then
      throw s!"einsum: subscript has too many labels for shape rank={r}"
    let ellCount := r - fixed
    if ellCount > maxEll then
      throw s!"einsum: internal error (ellipsis count {ellCount} > maxEll {maxEll})"
    let offset := maxEll - ellCount
    let ellLabels := (List.range ellCount).map (fun i => Label.ell (offset + i))
    pure <| sub.pre.map Label.chr ++ ellLabels ++ sub.post.map Label.chr
  else
    if r != fixed then
      throw s!"einsum: subscript label count {fixed} does not match shape rank={r}"
    pure <| sub.pre.map Label.chr

/-- Expand output labels, materializing the full ellipsis range `[0..maxEll)` when present. -/
def expandOutputLabels (sub : Subscript) (maxEll : Nat) : Except String (List Label) := do
  if sub.hasEll then
    let ellLabels := (List.range maxEll).map Label.ell
    pure <| sub.pre.map Label.chr ++ ellLabels ++ sub.post.map Label.chr
  else
    pure <| sub.pre.map Label.chr

/-!
### Small association-list helpers

To keep this file dependency-light, we represent maps as association lists and use small helpers
instead of `Std.HashMap`.
-/

/-- Occurrence counts for labels, represented as an association list. -/
abbrev Counts : Type := List (Label × Nat)

/-- Look up how many times a parsed einsum label occurs in the current signature. -/
def countsFind? (cs : Counts) (x : Label) : Option Nat :=
  (cs.find? (fun p => p.1 == x)).map (fun p => p.2)

/-- Lookup with default for `Counts`. -/
def countsFindD (cs : Counts) (x : Label) (dflt : Nat) : Nat :=
  (countsFind? cs x).getD dflt

/-- Increment a label’s count (inserting it if absent). -/
def countsInc (cs : Counts) (x : Label) : Counts :=
  let rec go : Counts → Counts
    | [] => [(x, 1)]
    | (y, n) :: ys =>
        if y == x then
          (y, n + 1) :: ys
        else
          (y, n) :: go ys
  go cs

/-- Count all labels across a list of operands. -/
def labelCounts (xss : List (List Label)) : Counts :=
  xss.foldl (fun acc xs => xs.foldl countsInc acc) []

/-- Keep first occurrences of labels, preserving order. -/
def orderedUnique (xs : List Label) : List Label :=
  let rec go (seen : List Label) : List Label → List Label
    | [] => []
    | x :: xs =>
        if seen.contains x then go seen xs
        else x :: go (x :: seen) xs
  go [] xs

/-- Map each label to its concrete dimension size (association list). -/
abbrev DimMap : Type := List (Label × Nat)

/-- Lookup a label’s dimension size. -/
def dimFind? (mp : DimMap) (x : Label) : Option Nat :=
  (mp.find? (fun p => p.1 == x)).map (fun p => p.2)

/-- Lookup with default for `DimMap`. -/
def dimFindD (mp : DimMap) (x : Label) (dflt : Nat) : Nat :=
  (dimFind? mp x).getD dflt

/-- Insert/update a label’s dimension size in a `DimMap`. -/
def dimUpdate (mp : DimMap) (x : Label) (d : Nat) : DimMap :=
  let rec go : DimMap → DimMap
    | [] => [(x, d)]
    | (y, d0) :: ys =>
        if y == x then
          (y, d) :: ys
        else
          (y, d0) :: go ys
  go mp

/--
Infer a consistent label-to-dimension map from operand label lists and operand shapes.

This implements standard einsum broadcast rules: if a label is seen with both `d` and `1`, we keep
`d`; if two non-`1` sizes disagree, we error.
-/
def labelDimMap (xss : List (List Label)) (shapes : List Shape) : Except String DimMap := do
  let mut mp : DimMap := []
  for (xs, s) in List.zip xss shapes do
    let dims := Shape.toList s
    if xs.length != dims.length then
      throw "einsum: internal error (label/dim length mismatch)"
    for (lbl, d) in List.zip xs dims do
      match dimFind? mp lbl with
      | none => mp := dimUpdate mp lbl d
      | some d0 =>
          if d0 = d then
            pure ()
          else if d0 = 1 then
            mp := dimUpdate mp lbl d
          else if d = 1 then
            pure ()
          else
            throw s!"einsum: incompatible sizes for label {repr lbl}: {d0} vs {d}"
  pure mp

/--
Compute a `Shape.CanBroadcastTo` witness at runtime.

This mirrors the typeclass-based broadcasting used elsewhere, but returns an `Option` so the dynamic
einsum lowering can fail gracefully.
-/
def canBroadcastTo? : (s₁ s₂ : Shape) → Option (Shape.CanBroadcastTo s₁ s₂)
  | .scalar, s₂ => some (.scalar_to_any s₂)
  | .dim n s₁, .dim m s₂ =>
      if h : n = m then
        by
          cases h
          match canBroadcastTo? s₁ s₂ with
          | some tail => exact some (Shape.CanBroadcastTo.dim_eq (n := n) (s₁ := s₁) (s₂ := s₂)
            tail)
          | none => exact none
      else if h1 : n = 1 then
        by
          cases h1
          match canBroadcastTo? s₁ s₂ with
          | some tail => exact some (Shape.CanBroadcastTo.dim_1_to_n (n := m) (s₁ := s₁) (s₂ := s₂)
            tail)
          | none => exact none
      else
        none
  | _, _ => none

/--
Apply a permutation expressed as adjacent swap depths to an existentially-shaped tensor.

This is the runtime “apply swaps” primitive used by both `.permute` and the dynamic einsum output
reordering.
-/
def permuteBySwaps {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    (x : Σ s : Shape, RefTy (m := m) (α := α) s) (swaps : List Nat) :
    m (Σ s : Shape, RefTy (m := m) (α := α) s) := do
  let mut cur := x
  for d in swaps do
    -- This is definitional: each swap updates the shape index.
    let cur' : RefTy (m := m) (α := α) (cur.fst.swapAdjacentAtDepth d) ←
      swapAdjacentAtDepth (m := m) (α := α) (s := cur.fst) d cur.snd
    cur := ⟨cur.fst.swapAdjacentAtDepth d, cur'⟩
  pure cur

/-- Reflexive broadcast witness constructor (`s` can always broadcast to itself). -/
def canBroadcastRefl : (s : Shape) → Shape.CanBroadcastTo s s
  | .scalar => Shape.CanBroadcastTo.scalar_to_any .scalar
  | .dim n s => Shape.CanBroadcastTo.dim_eq (n := n) (s₁ := s) (s₂ := s) (canBroadcastRefl s)

/-- Remove the element at index `n` (0-based), leaving the list unchanged if out of bounds. -/
def removeAt {α : Type} : List α → Nat → List α
  | [], _ => []
  | _ :: xs, 0 => xs
  | x :: xs, n + 1 => x :: removeAt xs n

/--
Compute a permutation that maps `src` to `tgt` when duplicates are present.

This is used for the “diagonal embedding” case when output labels contain repeats: we temporarily
expand the output with extra axes, then permute back to the requested (possibly-duplicated) order.
-/
def permForDuplicateLabels? (src tgt : List Label) : Option (List Nat) :=
  if _hLen : src.length = tgt.length then
    let rec findUnusedIndex? (used : List Nat) (l : Label) (i : Nat) : List Label → Option Nat
      | [] => none
      | x :: xs =>
          if x == l && !(used.contains i) then
            some i
          else
            findUnusedIndex? used l (i + 1) xs
    let rec go (used : List Nat) (tgt : List Label) (acc : List Nat) : Option (List Nat) :=
      match tgt with
      | [] => some acc.reverse
      | l :: ls =>
          match findUnusedIndex? used l 0 src with
          | none => none
          | some i => go (i :: used) ls (i :: acc)
    go [] tgt []
  else
    none

/--
Build a diagonal mask tensor (spec-level) for diagonal embedding/extraction.

Given axes `p` and `q`, the resulting tensor is `1` when the indices along those axes agree, and `0`
otherwise. The `ip`/`iq` parameters track the first-seen indices while recursing over dimensions.
-/
def diagMaskSpecAux {α : Type} [Zero α] [One α] :
    (dims : List Nat) → (p q : Nat) → (ip iq : Option Nat) → Tensor α (Shape.ofList dims)
  | [], _, _, ip, iq =>
      Tensor.scalar <|
        match ip, iq with
        | some i, some j => if i = j then 1 else 0
        | _, _ => 1
  | d :: ds, p, q, ip, iq =>
      Tensor.dim fun i : Fin d =>
        let ip' :=
          match p with
          | 0 =>
              match ip with
              | none => some i.1
              | some v => some v
          | _ + 1 => ip
        let iq' :=
          match q with
          | 0 =>
              match iq with
              | none => some i.1
              | some v => some v
          | _ + 1 => iq
        diagMaskSpecAux ds (Nat.pred p) (Nat.pred q) ip' iq'

/-- Diagonal mask specification with fresh index-tracking state. -/
def diagMaskSpec {α : Type} [Zero α] [One α] (dims : List Nat) (p q : Nat) :
    Tensor α (Shape.ofList dims) :=
  diagMaskSpecAux (α := α) dims p q none none

/-- Specialize `diagMaskSpec` to a concrete `Shape` by rewriting through `Shape.toList`. -/
def diagMaskForShape {α : Type} [Zero α] [One α] (s : Shape) (p q : Nat) : Tensor α s := by
  have hs : Shape.ofList (Shape.toList s) = s := Shape.ofList_toList s
  exact hs ▸ diagMaskSpec (α := α) (Shape.toList s) p q

/-- Return the first duplicate label in `xs`, along with its original and duplicate positions. -/
def firstDup? (xs : List Label) : Option (Label × Nat × Nat) :=
  let rec go (seen : List (Label × Nat)) (i : Nat) : List Label → Option (Label × Nat × Nat)
    | [] => none
    | x :: xs =>
        match seen.find? (fun p => p.1 == x) with
        | some p => some (x, p.2, i)
        | none => go ((x, i) :: seen) (i + 1) xs
  go [] 0 xs

/-- `Shape.appendDim s 1` preserves `Shape.size` (used to justify reshape tricks). -/
theorem size_appendDim_one (s : Shape) : Shape.size (Shape.appendDim s 1) = Shape.size s := by
  induction s with
  | scalar => simp [Shape.appendDim, Shape.size]
  | dim n s ih => simp [Shape.appendDim, Shape.size, ih]

/-- Permutation list that moves `axis` to the last position (keeping relative order of others). -/
def permMoveAxisToLast (r axis : Nat) : List Nat :=
  (List.range axis) ++ ((List.range (r - (axis + 1))).map (fun i => axis + 1 + i)) ++ [axis]

end Einsum

open Einsum

/--
Runtime-checked `einsum` that returns an existential output shape.

Supported:
- multiple inputs, explicit/implicit output, and ellipsis (`...`).
- repeated labels within an operand (diagonal extraction / trace semantics).
- repeated labels in the output (diagonal embedding / zeroing off-diagonal entries).

Currently unsupported (returns `none`):
- non-broadcastable size mismatches.
- any case that would require gather/scatter-style indexing (not in the verifier-friendly op set).

This is implemented purely by reordering, reshaping, broadcasting, elementwise multiplication,
and summing contracted axes.
-/
def einsumDyn {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    (equation : String)
    (xs : List (Σ s : Shape, RefTy (m := m) (α := α) s)) :
    m (Option (Σ s : Shape, RefTy (m := m) (α := α) s)) := do
  let computation : OptionT m (Σ s : Shape, RefTy (m := m) (α := α) s) := do
    let parsed : Einsum.Parsed ←
      match Einsum.parseEquation equation with
      | .ok p => pure p
      | .error _ => failure
    if parsed.inputs.length != xs.length then
      failure

    -- Fast paths for common contractions.
    --
    -- The generic lowering below broadcasts all operands to a common label shape, multiplies,
    -- then sums contracted axes. This is verifier-friendly but can allocate extremely large
    -- intermediate tensors for common patterns like matmul/bmm/attention.
    --
    -- These fast paths dispatch to existing primitives (`matmul`/`bmm` plus small reshapes/
    -- transposes), preserving semantics while avoiding the huge broadcasted intermediate.
    let attemptFast : OptionT m (Σ s : Shape, RefTy (m := m) (α := α) s) := do
      if parsed.inputs.length != 2 then
        failure
      if xs.length != 2 then
        failure
      let some sub0 := listGet? parsed.inputs 0 | failure
      let some sub1 := listGet? parsed.inputs 1 | failure
      let some x0 := listGet? xs 0 | failure
      let some x1 := listGet? xs 1 | failure
      let out? := parsed.output?

      -- Require "simple" subscripts for fast paths: no ellipsis and no post-ellipsis labels.
      if sub0.hasEll || sub1.hasEll then
        failure
      match sub0.post, sub1.post with
      | [], [] => pure ()
      | _, _ => failure

      let expectOut (expected : List Char) : OptionT m Unit := do
        match out? with
        | some o =>
            if o.hasEll then failure
            match o.post with
            | [] =>
                if o.pre = expected then pure () else failure
            | _ => failure
        | none =>
            -- Implicit output for these fast paths matches the standard Einstein convention:
            -- keep labels that appear exactly once, in first-appearance order.
            pure ()

      -- Case dispatch is primarily driven by shapes.
      match x0 with
      -- Matmul: `[i,j] × [j,k] → [i,k]` and subscripts `ij,jk->ik` (or implicit output).
      | ⟨.dim iDim (.dim jDim .scalar), a⟩ =>
          match x1 with
          | ⟨.dim jDim2 (.dim kDim .scalar), b⟩ =>
              if hJ : jDim = jDim2 then
                match hJ with
                | rfl =>
                    -- Extract labels (must be length-2 on both operands).
                    let some li0 := listGet? sub0.pre 0 | failure
                    let some lj0 := listGet? sub0.pre 1 | failure
                    if sub0.pre.length != 2 then failure
                    let some lj1 := listGet? sub1.pre 0 | failure
                    let some lk1 := listGet? sub1.pre 1 | failure
                    if sub1.pre.length != 2 then failure
                    if lj0 != lj1 then failure
                    expectOut [li0, lk1]
                    let out ← OptionT.lift <|
                      einsumIjJkIk (m := m) (α := α)
                        (iDim := iDim) (jDim := jDim) (kDim := kDim) a b
                    pure ⟨.dim iDim (.dim kDim .scalar), out⟩
              else
                failure
          | _ => failure
      -- BMM: `[b,i,j] × [b,j,k] → [b,i,k]` and subscripts `bij,bjk->bik` (or implicit output).
      | ⟨.dim batch (.dim iDim (.dim jDim .scalar)), a⟩ =>
          match x1 with
          | ⟨.dim batch2 (.dim jDim2 (.dim kDim .scalar)), b⟩ =>
              if hB : batch = batch2 then
                match hB with
                | rfl =>
                    if hJ : jDim = jDim2 then
                      match hJ with
                      | rfl =>
                          let some lb0 := listGet? sub0.pre 0 | failure
                          let some li0 := listGet? sub0.pre 1 | failure
                          let some lj0 := listGet? sub0.pre 2 | failure
                          if sub0.pre.length != 3 then failure
                          let some lb1 := listGet? sub1.pre 0 | failure
                          let some lj1 := listGet? sub1.pre 1 | failure
                          let some lk1 := listGet? sub1.pre 2 | failure
                          if sub1.pre.length != 3 then failure
                          if lb0 != lb1 then failure
                          if lj0 != lj1 then failure
                          expectOut [lb0, li0, lk1]
                          let out ← OptionT.lift <|
                            einsumBijBjkBik (m := m) (α := α)
                              (batch := batch) (iDim := iDim) (jDim := jDim) (kDim := kDim) a b
                          pure ⟨.dim batch (.dim iDim (.dim kDim .scalar)), out⟩
                    else
                      failure
              else
                failure
          | _ => failure
      -- 4D attention-like contractions (Q·Kᵀ or Attn·V), selected by label patterns.
      | ⟨.dim batch (.dim heads (.dim iDim (.dim tDim .scalar))), x0Ref⟩ =>
          match x1 with
          | ⟨.dim batch2 (.dim heads2 (.dim jDim (.dim dDim .scalar))), x1Ref⟩ =>
              if hB : batch = batch2 then
                match hB with
                | rfl =>
                    if hH : heads = heads2 then
                      match hH with
                      | rfl =>
                          let some lb0 := listGet? sub0.pre 0 | failure
                          let some lh0 := listGet? sub0.pre 1 | failure
                          let some l2_0 := listGet? sub0.pre 2 | failure
                          let some l3_0 := listGet? sub0.pre 3 | failure
                          if sub0.pre.length != 4 then failure
                          let some lb1 := listGet? sub1.pre 0 | failure
                          let some lh1 := listGet? sub1.pre 1 | failure
                          let some l2_1 := listGet? sub1.pre 2 | failure
                          let some l3_1 := listGet? sub1.pre 3 | failure
                          if sub1.pre.length != 4 then failure
                          if lb0 != lb1 then failure
                          if lh0 != lh1 then failure
                          -- Two attention-like cases:
                          -- 1) Q·Kᵀ: `bhid,bhjd -> bhij`  (shared last label across inputs).
                          -- 2) Attn·V: `bhij,bhjd -> bhid` (contract sub0 last label with sub1
                          -- third label).
                          if l3_0 = l3_1 then
                            -- Q·Kᵀ: sub0 = [b,h,i,d], sub1 = [b,h,j,d], and shapes must agree on
                            -- `d`.
                            if hD : dDim = tDim then
                              match hD with
                              | rfl =>
                                  expectOut [lb0, lh0, l2_0, l2_1]
                                  let out ← OptionT.lift <|
                                    einsumBhidBhjdBhij (m := m) (α := α)
                                      (batch := batch) (heads := heads) (iDim := iDim) (jDim :=
                                        jDim) (dDim := dDim) x0Ref x1Ref
                                  pure ⟨.dim batch (.dim heads (.dim iDim (.dim jDim .scalar))),
                                    out⟩
                            else
                              failure
                          else if l3_0 = l2_1 then
                            -- Attn·V: sub0 = [b,h,i,j], sub1 = [b,h,j,d], and shapes must agree on
                            -- `j`.
                            if hJ : jDim = tDim then
                              match hJ with
                              | rfl =>
                                  expectOut [lb0, lh0, l2_0, l3_1]
                                  let out ← OptionT.lift <|
                                    einsumBhijBhjdBhid (m := m) (α := α)
                                      (batch := batch) (heads := heads) (iDim := iDim) (jDim :=
                                        jDim) (dDim := dDim) x0Ref x1Ref
                                  pure ⟨.dim batch (.dim heads (.dim iDim (.dim dDim .scalar))),
                                    out⟩
                            else
                              failure
                          else
                            failure
                    else
                      failure
              else
                failure
          | _ => failure
      | _ => failure

    let fastRes? ← OptionT.lift attemptFast.run
    if let some r := fastRes? then
      return r

    let shapes0 : List Shape := xs.map Sigma.fst
    let ranks : List Nat := shapes0.map Shape.rank

    -- Compute max ellipsis length across inputs.
    let mut maxEll : Nat := 0
    for (sub, r) in List.zip parsed.inputs ranks do
      if sub.hasEll then
        let fixed := sub.pre.length + sub.post.length
        if r < fixed then
          failure
        maxEll := Nat.max maxEll (r - fixed)

    -- Expand per-input labels (including ellipsis mapped to `ell k` labels), and apply diagonal
    -- extraction for repeated labels inside an operand (PyTorch semantics).
    let mut processed : List (Σ s : Shape, RefTy (m := m) (α := α) s) := []
    let mut inLabels : List (List Label) := []
    let mut inLabelsRaw : List (List Label) := []

    let rec diagonalizeOperand (fuel : Nat)
        (cur : Σ s : Shape, RefTy (m := m) (α := α) s) (labs : List Label) :
        OptionT m ((Σ s : Shape, RefTy (m := m) (α := α) s) × List Label) := do
      match fuel with
      | 0 =>
          if Einsum.hasDupLabels labs then
            failure
          else
            pure (cur, labs)
      | fuel + 1 =>
          match Einsum.firstDup? labs with
          | none => pure (cur, labs)
          | some (_, p, q) =>
              let curShape := cur.fst
              let dims := Shape.toList curShape
              let dp := dims.getD p 0
              let dq := dims.getD q 0
              if dp = 0 || dq = 0 then
                failure
              if dp != dq then
                failure
              let maskT : Tensor α curShape := Einsum.diagMaskForShape (α := α) curShape p q
              let mask ← OptionT.lift <| const (m := m) (α := α) (s := curShape) maskT
              let xMasked ← OptionT.lift <| mul (m := m) (α := α) (s := curShape) cur.snd mask
              let r := Shape.rank curShape
              if hRank : r > 0 then
                if hq : q < r then
                  let perm := Einsum.permMoveAxisToLast r q
                  let swaps : List Nat ←
                    match Einsum.swapDepthsForPerm? perm r with
                    | some ss => pure ss
                    | none => failure
                  let ⟨sPerm, xPerm⟩ ← OptionT.lift <|
                    Einsum.permuteBySwaps (α := α) (m := m) (x := ⟨curShape, xMasked⟩) swaps
                  let rPerm := Shape.rank sPerm
                  if hRankPerm : rPerm > 0 then
                    if hw : sPerm.wellFormed then
                      let axis : Nat := rPerm - 1
                      let nextRef :=
                        (← OptionT.lift <|
                          (by
                            letI : Shape.WellFormed sPerm := ⟨hw⟩
                            haveI : Shape.valid_axis_inst axis sPerm :=
                              Shape.validAxisLastInst (s := sPerm) hRankPerm hw
                            exact reduceSum (m := m) (α := α) (s := sPerm) axis xPerm))
                      let nextShape : Shape := Spec.Tensor.shapeAfterSum sPerm axis
                      diagonalizeOperand fuel ⟨nextShape, nextRef⟩ (Einsum.removeAt labs q)
                    else
                      failure
                  else
                    failure
                else
                  failure
              else
                failure

    for (sub, xSigma) in List.zip parsed.inputs xs do
      let s := xSigma.fst
      let x := xSigma.snd
      let labs0 : List Label ←
        match Einsum.expandInputLabels sub s maxEll with
        | .ok v => pure v
        | .error _ => failure
      inLabelsRaw := inLabelsRaw ++ [labs0]
      let (cur', labs') ← diagonalizeOperand labs0.length ⟨s, x⟩ labs0
      processed := processed ++ [cur']
      inLabels := inLabels ++ [labs']

    -- Use diagonalized shapes for the remaining checks/alignments.
    let shapes : List Shape := processed.map Sigma.fst

    -- Determine output labels.
    let counts := Einsum.labelCounts inLabels
    let countsRaw := Einsum.labelCounts inLabelsRaw
    let allInOrder : List Label :=
      Einsum.orderedUnique (inLabels.foldl (fun acc xs => acc ++ xs) [])
    let allInOrderRaw : List Label :=
      Einsum.orderedUnique (inLabelsRaw.foldl (fun acc xs => acc ++ xs) [])
    let outLabelsRaw : List Label ←
      match parsed.output? with
      | some outSub =>
          let labs : List Label ←
            match Einsum.expandOutputLabels outSub maxEll with
            | .ok v => pure v
            | .error _ => failure
          for l in labs do
            if !(allInOrder.contains l) then
              failure
          pure labs
      | none =>
          -- Implicit output: all ellipsis dims first, then labels that appear exactly once,
          -- in order of first appearance.
          let ellLabs := (List.range maxEll).map Label.ell
          let mut labs : List Label := ellLabs
          for l in allInOrderRaw do
            if ellLabs.contains l then
              continue
            -- Important: use *raw* label multiplicities (before diagonal extraction) so that
            -- `ii` (implicit output) computes a trace (scalar), not a diagonal vector.
            if Einsum.countsFindD countsRaw l 0 = 1 then
              labs := labs ++ [l]
          pure labs

    let outLabels : List Label :=
      -- If explicit output repeats labels (e.g. `i->ii`), we contract w.r.t. unique labels
      -- and then "diag-embed" to the repeated output at the end.
      Einsum.orderedUnique outLabelsRaw

    -- Contracted labels are everything not in the output (in first-appearance order).
    let contracted : List Label :=
      allInOrder.filter (fun l => !(outLabels.contains l))
    let fullLabels : List Label := outLabels ++ contracted

    -- Infer broadcasted label sizes.
    let dimMap ←
      match Einsum.labelDimMap inLabels shapes with
      | .ok mp => pure mp
      | .error _ => failure

    let fullDims : List Nat :=
      fullLabels.map (fun l => Einsum.dimFindD dimMap l 1)
    let sCommon : Shape := Shape.ofList fullDims

    -- Align each operand to `fullLabels` (permute -> reshape insert ones -> broadcast).
    let mut aligned : List (RefTy (m := m) (α := α) sCommon) := []
    for ((⟨sIn, xIn⟩), labsIn) in List.zip processed inLabels do
      let targetOrder := fullLabels.filter (fun l => labsIn.contains l)
      let mut perm : List Nat := []
      for l in targetOrder do
        match Einsum.findIndex? labsIn l with
        | none => failure
        | some i => perm := perm ++ [i]
      let swaps : List Nat ←
        match Einsum.swapDepthsForPerm? perm (Shape.rank sIn) with
        | some ss => pure ss
        | none => failure
      let ⟨sPerm, xPerm⟩ ← OptionT.lift <|
        Einsum.permuteBySwaps (α := α) (m := m) (x := ⟨sIn, xIn⟩) swaps
      let dimsPerm := Shape.toList sPerm
      -- Reshape to insert singleton dims for missing labels.
      let mut di : Nat := 0
      let mut insertedDims : List Nat := []
      for l in fullLabels do
        if labsIn.contains l then
          let d := dimsPerm.getD di 1
          insertedDims := insertedDims ++ [d]
          di := di + 1
        else
          insertedDims := insertedDims ++ [1]
      let sInserted : Shape := Shape.ofList insertedDims
      let xInserted : RefTy (m := m) (α := α) sInserted ←
        if h : Shape.size sPerm = Shape.size sInserted then
          OptionT.lift <| reshape (m := m) (α := α) (s₁ := sPerm) (s₂ := sInserted) xPerm h
        else
          failure
      let cb : Shape.CanBroadcastTo sInserted sCommon ←
        match Einsum.canBroadcastTo? sInserted sCommon with
        | some cb => pure cb
        | none => failure
      let xb ← OptionT.lift <|
        broadcastTo (m := m) (α := α) (s₁ := sInserted) (s₂ := sCommon) cb xInserted
      aligned := aligned ++ [xb]

    -- Multiply all aligned operands elementwise.
    let some prod0 := aligned.head? | failure
    let mut prod : RefTy (m := m) (α := α) sCommon := prod0
    for x in aligned.drop 1 do
      prod ← OptionT.lift <| mul (m := m) (α := α) (s := sCommon) prod x

    -- Sum-reduce contracted axes (a suffix by construction).
    let rec reduceContracted (n : Nat)
        (cur : Σ s : Shape, RefTy (m := m) (α := α) s) :
        OptionT m (Σ s : Shape, RefTy (m := m) (α := α) s) := do
      match n with
      | 0 => pure cur
      | n + 1 =>
          let curShape := cur.fst
          if hRank : Shape.rank curShape > 0 then
            if hw : curShape.wellFormed then
              let axis : Nat := Shape.rank curShape - 1
              let nextRef :=
                (← OptionT.lift <|
                  (by
                    letI : Shape.WellFormed curShape := ⟨hw⟩
                    haveI : Shape.valid_axis_inst axis curShape :=
                      Shape.validAxisLastInst (s := curShape) hRank hw
                    exact reduceSum (m := m) (α := α) (s := curShape) axis cur.snd))
              let nextShape : Shape :=
                Spec.Tensor.shapeAfterSum curShape axis
              reduceContracted n ⟨nextShape, nextRef⟩
            else
              failure
          else
            failure

    let out0 ← reduceContracted contracted.length ⟨sCommon, prod⟩
    if outLabelsRaw = outLabels then
      pure out0
    else
      -- Diagonal embedding for repeated output labels:
      -- insert new axes (via reshape+broadcast) and zero out off-diagonal entries with a mask.
      let extras : List Label :=
        let rec go (seen : List Label) : List Label → List Label
          | [] => []
          | l :: ls =>
              if seen.contains l then
                l :: go seen ls
              else
                go (l :: seen) ls
        go [] outLabelsRaw
      let outCanon : List Label := outLabels ++ extras
      let mut cur : Σ s : Shape, RefTy (m := m) (α := α) s := out0
      for l in extras do
        let some baseIdx := Einsum.findIndex? outLabels l | failure
        let d := Einsum.dimFindD dimMap l 1
        let sReshape : Shape := Shape.appendDim cur.fst 1
        have hSz : Shape.size cur.fst = Shape.size sReshape := by
          simpa [sReshape] using (Eq.symm (Einsum.size_appendDim_one cur.fst))
        let xReshaped ← OptionT.lift <|
          reshape (m := m) (α := α) (s₁ := cur.fst) (s₂ := sReshape) cur.snd hSz
        let sBroad : Shape := Shape.appendDim cur.fst d
        let cb : Shape.CanBroadcastTo sReshape sBroad ←
          match Einsum.canBroadcastTo? sReshape sBroad with
          | some cb => pure cb
          | none => failure
        let xExpanded ← OptionT.lift <|
          broadcastTo (m := m) (α := α) (s₁ := sReshape) (s₂ := sBroad) cb xReshaped
        let qIdx : Nat := Shape.rank sBroad - 1
        let maskT : Tensor α sBroad := Einsum.diagMaskForShape (α := α) sBroad baseIdx qIdx
        let mask ← OptionT.lift <| const (m := m) (α := α) (s := sBroad) maskT
        let xMasked ← OptionT.lift <| mul (m := m) (α := α) (s := sBroad) xExpanded mask
        cur := ⟨sBroad, xMasked⟩

      let some perm := Einsum.permForDuplicateLabels? outCanon outLabelsRaw | failure
      let some swaps := Einsum.swapDepthsForPerm? perm (Shape.rank cur.fst) | failure
      OptionT.lift <| Einsum.permuteBySwaps (α := α) (m := m) (x := cur) swaps
  computation.run

/-- `einsum` with an expected output shape. -/
def einsum {α : Type} [Context α] [DecidableEq Shape]
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {sOut : Shape}
    (equation : String)
    (xs : List (Σ s : Shape, RefTy (m := m) (α := α) s)) :
    m (Option (RefTy (m := m) (α := α) sOut)) := do
  let r? ← einsumDyn (α := α) (m := m) equation xs
  match r? with
  | none => pure none
  | some ⟨s, r⟩ =>
      if h : s = sOut then
        pure (some (h ▸ r))
      else
        pure none
end F
end TorchLean
end Autograd
end Runtime
