/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.Float32
public import NN.Spec.Core.Complex
public import NN.Spec.Core.Tensor.API
import Mathlib.Algebra.Order.Algebra

/-!
# Tensor API Implementation

User-facing tensor API for TorchLean.

This module implements the constructors and notation exported through the public `NN.Tensor`
umbrella. The tensor type and its mathematical operations remain those of `NN.Spec`; this file
only supplies convenient construction and display functions.

Typical responsibilities here:

- arbitrary-rank shapes represented by `Spec.Shape`,
- friendly constructors from lists and literals,
- small syntax helpers for examples/tests,
- and printing support for runtime-friendly dtypes.

Notation policy:
- keep compact literal constructors (`shape!`, `tensor!`, `tensorOfList!`, `tensorF!`, `tensor32!`,
  `fin!`) in this module so they are easy to find and do not leak into unrelated imports,
- keep more semantic glyphs scoped when possible (as in `NN.IR` and `NN.Floats.IEEEExec`),
- and prefer namespace-local notation over reusing the same unscoped token in multiple layers.

Lean is statically typed, so the element type usually plays the role of “dtype”:

  `let x := NN.Tensor.vector (α := Float) [0.1, 0.2]`
  `let q := NN.Tensor.vector (α := ℚ) [1, 2]`

If you ever see `Tensor Float _` in code: the `_` is just “let Lean infer the shape from the RHS”.
In examples we prefer to omit the type annotation entirely unless it helps readability.

For convenience, this module also provides constructors that start from `Float` literals and cast
into common backends such as `IEEE32Exec`.

Printing: for `Tensor ℝ` we intentionally refuse to pretty-print, since that backend is meant for
proof-oriented mathematics rather than runtime IO. Use `Float`, `IEEE32Exec`, `ℚ`, etc. for values
you actually want to display.
-/

@[expose] public section


namespace NN.Tensor

open _root_.Spec

/-! ## Construction -/

/-- Create a 1-D tensor from a Lean `List`.

Notes:
- The shape is computed from `xs.length`, so the type remembers the length.
- This is total and does not perform any runtime checks.

PyTorch analogy: `torch.tensor(xs)` producing a 1D tensor.
-/
def vector (α : Type := Float) (xs : List α) :
    Spec.Tensor α (.dim xs.length .scalar) :=
  Spec.vectorFromList xs

/-! ### One-hot vectors -/

/-- One-hot vector of length `n`, with a single `1` at index `k`. -/
def oneHot {α : Type} [Zero α] [One α] (n : Nat) (k : Fin n) :
    Spec.Tensor α (.dim n .scalar) :=
  Spec.Tensor.dim (fun i => Spec.Tensor.scalar (if decide (i = k) then (1 : α) else 0))

/-- One-hot vector using a raw `Nat` index, or the zero vector when the index is out of range.

If $k\ge n$, we return the all-zeros vector instead of failing.
This is convenient in data-conversion code where carrying a `Fin n` would obscure the caller.
-/
def oneHotNatOrZero {α : Type} [Zero α] [One α] (n k : Nat) :
    Spec.Tensor α (.dim n .scalar) :=
  if h : k < n then
    oneHot (α := α) n ⟨k, h⟩
  else
    Spec.fill (0 : α) (.dim n .scalar)

/--
One-hot encode every entry of a natural-number tensor along a new final axis.

An index smaller than `n` selects the corresponding coordinate. An out-of-range index produces an
all-zero row, matching `oneHotNatOrZero`. The source shape is otherwise arbitrary.
-/
def oneHotIndicesOrZero {α : Type} [Zero α] [One α] (n : Nat) :
    {s : Spec.Shape} → Spec.Tensor Nat s → Spec.Tensor α (s.appendDim n)
  | .scalar, .scalar k => oneHotNatOrZero (α := α) n k
  | .dim _ _, .dim values =>
      Spec.Tensor.dim fun i => oneHotIndicesOrZero (α := α) n (values i)

/-- 2-D tensor from nested lists, returning `none` for ragged input.

Empty input denotes a `0 × 0` tensor. Empty rows are accepted when every row is empty.

PyTorch analogy: `torch.tensor(xss)` also refuses ragged inputs.
-/
def matrix? (α : Type := Float) (xss : List (List α)) :
    Option (Spec.Tensor α (.dim xss.length
      (.dim (Option.getD (xss.head?.map List.length) 0) .scalar)))
      :=
  Spec.matrixFromRows? xss

/-- 2-D tensor from nested lists, with a clear error message on failure. -/
def matrix (α : Type := Float) (xss : List (List α)) :
    Except String (Spec.Tensor α (.dim xss.length
      (.dim (Option.getD (xss.head?.map List.length) 0) .scalar))) :=
  match matrix? (α := α) xss with
  | some t => .ok t
  | none => .error "matrix: rows have different lengths"

/-! ### Rectangularizing two-dimensional data -/

/-- Resize every row to `columnCount`, padding with `default` or dropping trailing entries.

Unlike `matrix`, this constructor accepts ragged input. It is intended for boundaries where the
caller has deliberately chosen a fixed width, such as a padded sequence batch.

PyTorch analogy: closer to `pad_sequence(..., batch_first=True)` followed by `torch.tensor`.
-/
def matrixResize (α : Type := Float) [Inhabited α]
    (columnCount : Nat) (rows : List (List α)) :
    Spec.Tensor α (.dim rows.length (.dim columnCount .scalar)) :=
  Spec.matrixFromRowsResize (columnCount := columnCount) rows

/-- 2-D tensor from nested lists, padding to the maximum row length (`0` if empty).

This is convenient when you just want a rectangular tensor without precomputing the width. -/
def matrixPadRight (α : Type := Float) [Inhabited α] (xss : List (List α)) :
    Spec.Tensor α (.dim xss.length (.dim (Spec.maxRowLength xss) .scalar)) :=
  Spec.matrixFromRowsPadRight xss

/-! ### General N-D tensors from a flat list -/

/-!
The “real” spec constructors are shape-indexed, i.e. you usually build tensors by recursion on a
`Shape`. In example and data-loading code, though, it’s common to have:

- a runtime dims list (`List Nat`), and
- a flat list of scalars.

So we do a small amount of reshaping here.

`ofListOfLength` performs the structural recursion locally, so the implementation does not add a
second tensor-construction entry point to the public API. `ofList` supplies the same length proof
after checking it at runtime.
-/
/-- N-D tensor from a runtime `dims` list and a flat `xs`, given a proof of matching length.

This is the “static / proof-carrying” version: if you can prove the length match, you avoid any
runtime checks and you keep a precise shape in the type.
-/
def ofListOfLength {α : Type} (dims : List Nat) (xs : List α)
    (h : xs.length = dims.prod) : Spec.Tensor α (Spec.Shape.ofList dims) :=
  let rec build :
      (s : Spec.Shape) → (ys : List α) → ys.length = Spec.Shape.size s → Spec.Tensor α s
    | .scalar, ys, hSize => by
        have hy : ys.length = 1 := by
          simpa [Spec.Shape.size] using hSize
        have hNonempty : 0 < ys.length := by simp [hy]
        exact .scalar (ys.get ⟨0, hNonempty⟩)
    | .dim n tail, ys, hSize => by
        let chunkSize := Spec.Shape.size tail
        have hy : ys.length = n * chunkSize := by
          simpa [Spec.Shape.size, chunkSize] using hSize
        refine .dim fun i => ?_
        let start := i.val * chunkSize
        let chunk := (ys.drop start).take chunkSize
        have hEnd : start + chunkSize ≤ ys.length := by
          have hMul : (i.val + 1) * chunkSize ≤ n * chunkSize :=
            Nat.mul_le_mul_right chunkSize (Nat.succ_le_of_lt i.2)
          simpa [hy, start, Nat.add_mul] using hMul
        have hStart : start ≤ ys.length :=
          Nat.le_trans (Nat.le_add_right start chunkSize) hEnd
        have hChunkLe : chunkSize ≤ ys.length - start := by
          apply (Nat.le_sub_iff_add_le hStart).2
          simpa [Nat.add_comm] using hEnd
        have hChunk : chunk.length = Spec.Shape.size tail := by
          simp [chunk, List.length_take, List.length_drop, Nat.min_eq_left hChunkLe, chunkSize]
        exact build tail chunk hChunk
  build (Spec.Shape.ofList dims) xs (by simpa using h)

/-- N-D tensor from a runtime `dims` list and a flat `xs`, with a runtime length check.

This is the “dynamic / user-friendly” version: it fails with a descriptive message if the number
of provided scalars doesn’t match the implied `numel`.
-/
def ofList {α : Type} (dims : List Nat) (xs : List α) :
    Except String (Spec.Tensor α (Spec.Shape.ofList dims)) :=
  let expected := dims.prod
  if h : xs.length = expected then
    .ok (ofListOfLength (α := α) (dims := dims) (xs := xs) h)
  else
    .error s!"ofList: expected {expected} elements for dims={dims}, got {xs.length}"

/-! ### Fill, zeros, and ones from runtime dimensions -/

/-- Fill a tensor from a runtime dimension list.

The `OfDims` suffix distinguishes this constructor from `Spec.fill`, whose second argument is an
already-typed `Shape`.
-/
def fillOfDims {α : Type} (value : α) (dims : List Nat) :
    Spec.Tensor α (Spec.Shape.ofList dims) :=
  Spec.fill value (Spec.Shape.ofList dims)

/-- All-zeros tensor, from a runtime `dims` list. -/
def zerosOfDims {α : Type} [Zero α] (dims : List Nat) :
    Spec.Tensor α (Spec.Shape.ofList dims) :=
  fillOfDims (α := α) 0 dims

/-- All-ones tensor, from a runtime `dims` list. -/
def onesOfDims {α : Type} [One α] (dims : List Nat) :
    Spec.Tensor α (Spec.Shape.ofList dims) :=
  fillOfDims (α := α) 1 dims

/-! ## Tactics -/

/-!
`ofListOfLength` needs a proof that your flat list has the right length.

For the common “all dimensions and lists are literals/abbrevs” case, this tactic
usually closes that goal automatically.

Usage:
  `ofListOfLength ... (by tensor_len)`
-/
macro "tensor_len" : tactic =>
  `(tactic| first
    | (simp [List.prod]; dsimp; decide)
    | decide
    | simp [List.prod])

/-!
`tensorOfList!` is the checked literal constructor for constants whose length proof should be solved
by `tensor_len`.

It expands to `ofListOfLength ... (by tensor_len)`, so you usually don’t have to write the proof.
If the proof can’t be solved (for example, when `dims` is computed at runtime), elaboration fails;
use `ofList` in that
  case.
-/
macro "tensorOfList!" dims:term:max xs:term:max : term =>
  `(NN.Tensor.ofListOfLength (dims := $dims) (xs := $xs) (by tensor_len))

/--
Typed variant of `tensorOfList!`.

This is the same constructor as `tensorOfList! dims xs`, but lets you explicitly specify the element
type when numeric literals would otherwise default to an undesired type.

Example:
`def x : Tensor ℚ (shape![2, 2]) := tensorOfList! (ty := ℚ) [2, 2] [1, 2, 3, 4]`

Implementation note: we avoid reserving a common identifier as a syntax keyword (Lean would then
treat it as a keyword in downstream files). Instead, we parse an `ident` and check that it is
`ty`.
-/
macro "tensorOfList!" "(" name:ident ":=" elemTy:term ")" dims:term:max xs:term:max : term => do
  if name.getId != `ty then
    Lean.Macro.throwErrorAt name "tensorOfList!: expected `(ty := <type>)`"
  `(NN.Tensor.ofListOfLength (α := $elemTy) (dims := $dims) (xs := $xs) (by tensor_len))

/-!
`shape!` is a compact convenience macro for writing a statically known `Shape`. It expands directly
to `Shape.dim` constructors, so structural shape evidence remains available to typeclass search.

Example:
  `def s : Shape := shape![2, 3, 4]`
-/
syntax (name := shapeBang) "shape!" "[" term,* "]" : term

macro_rules
  | `(shape![ $ds:term,* ]) => do
      let mut shape ← `(Spec.Shape.scalar)
      for dim in ds.getElems.reverse do
        shape ← `(Spec.Shape.dim $dim $shape)
      pure shape

/-!
`tensor!` is the "nested brackets" constructor, similar to writing nested Python lists in PyTorch.

Examples:

```lean
-- shape (2, 3, 4)
def x :=
  tensor! [
    [ [0,1,2,3],   [4,5,6,7],   [8,9,10,11] ],
    [ [12,13,14,15],[16,17,18,19],[20,21,22,23] ]
  ]
```

It is fully general over rank: the nesting depth determines the rank.
Internally, it computes the dims from list lengths and flattens in row-major order
(last dimension changes fastest), then calls `tensorOfList!`.

If you need runtime/ragged handling, use `ofList`/`someTensorOfList` instead.
-/

open Lean

namespace TensorLit

-- Parse a nested list literal into (dims, flat-leaf-terms).
private meta partial def parseNested (t : Syntax) : MacroM (List Nat × List Syntax) := do
  match t with
  | `([$ts,*]) =>
      -- `ts` is a `TSepArray`, i.e. `#[el1, ",", el2, ",", ...]`.
      let es := ts.elemsAndSeps
      let mut elems : Array Syntax := #[]
      for i in List.finRange es.size do
        if i.val % 2 = 0 then
          elems := elems.push (es[i.val]!)
      if elems.isEmpty then
        pure ([0], [])
      else
        let parsed : List (List Nat × List Syntax) ← elems.toList.mapM parseNested
        match parsed with
        | [] => pure ([0], [])
        | (dims0, flat0) :: rest =>
            -- Refuse mixed depths like `[1, [2]]` and ragged shapes.
            for (dims, _) in rest do
              if dims != dims0 then
                Macro.throwErrorAt t "tensor!: ragged or mixed-depth nested lists are not supported"
            let flat := rest.foldl (fun acc p => acc ++ p.snd) flat0
            pure (elems.size :: dims0, flat)
  | _ =>
      -- Leaf scalar term.
      pure ([], [t])

end TensorLit

/-!
Typed variant of `tensor!`.

This is useful when the leaf literals don’t uniquely determine the element type.

Example:
```lean
def x : Tensor ℚ (shape![2, 2]) :=
  tensor! (ty := ℚ) [[1, 2], [3, 4]]
```
-/

macro "tensor!" "(" name:ident ":=" elemTy:term ")" xs:term:max : term => do
  if name.getId != `ty then
    Lean.Macro.throwErrorAt name "tensor!: expected `(ty := <type>)`"
  let (dims, flat) ← TensorLit.parseNested xs
  let dimsElems : Array (TSyntax `term) :=
    dims.toArray.map (fun n => ⟨Syntax.mkNumLit (toString n)⟩)
  let flatElems : Array (TSyntax `term) :=
    flat.toArray.map (fun stx => ⟨stx⟩)
  let dimsSep := Syntax.TSepArray.ofElems (sep := ",") dimsElems
  let flatSep := Syntax.TSepArray.ofElems (sep := ",") flatElems
  let dimsTerm ← `(term| [$(dimsSep),*])
  let flatTerm ← `(term| [$(flatSep),*])
  `(NN.Tensor.ofListOfLength (α := $elemTy) (dims := $dimsTerm) (xs := $flatTerm) (by tensor_len))

/--
Untyped variant of `tensor!`.

This is the common case: Lean infers the element type from the literal leaves (e.g. `Nat`, `Int`,
`Float`, `ℚ`, ...). If inference picks the wrong type, use the typed form
`tensor! (ty := α) ...` instead.
-/
macro "tensor!" xs:term:max : term => do
  let (dims, flat) ← TensorLit.parseNested xs
  let dimsElems : Array (TSyntax `term) :=
    dims.toArray.map (fun n => ⟨Syntax.mkNumLit (toString n)⟩)
  let flatElems : Array (TSyntax `term) :=
    flat.toArray.map (fun stx => ⟨stx⟩)
  let dimsSep := Syntax.TSepArray.ofElems (sep := ",") dimsElems
  let flatSep := Syntax.TSepArray.ofElems (sep := ",") flatElems
  let dimsTerm ← `(term| [$(dimsSep),*])
  let flatTerm ← `(term| [$(flatSep),*])
  `(NN.Tensor.ofListOfLength (dims := $dimsTerm) (xs := $flatTerm) (by tensor_len))

/-! ## Small index helpers -/

/-!
Writing `⟨0, by decide⟩` everywhere is noisy.

These macros are intended for examples/tests where the dimension is a literal or abbreviation, so
`by decide` can close the bounds proof. They are not a replacement for carrying a real
`Fin n` in library code.

Examples:
  `Tensor.vecGet v fin0!`
  `Tensor.vecGet v fin1!`
  `Tensor.vecGet v (fin! 5 3)`
-/

macro "fin0!" : term =>
  `(⟨0, by decide⟩)

/-- Convenience index `⟨1, by decide⟩` for example code where the bound proof is decidable. -/
macro "fin1!" : term =>
  `(⟨1, by decide⟩)

/-- Convenience index `⟨i, by decide⟩` packaged as `Fin n`, for examples/tests with literal bounds. -/
macro "fin!" n:term:max i:num : term =>
  `(show Fin $n from ⟨$i, by decide⟩)

/-!
`tensorF!` is a convenience macro for constants: build from **Float literals**, then cast into your
chosen runtime scalar backend using `cast : Float → α`.

Example:
  `let w : Tensor α (.dim 3 (.dim 2 .scalar)) := tensorF! cast [3, 2] [0.1, 0.2, 0.3, 0.4, 0.5,
    0.6]`

This avoids noisy lists like `[cast 0.1, cast 0.2, ...]`.
-/
macro "tensorF!" cast:term:max dims:term:max xs:term:max : term =>
  `(Spec.mapTensor $cast
      (NN.Tensor.ofListOfLength (α := Float) (dims := $dims) (xs := $xs) (by tensor_len)))

/-!
`tensor32!` is the `tensor!` macro specialized to the executable IEEE-754 binary32 backend.

It builds a `Tensor Float _` from a nested bracket literal, then casts elementwise via
`IEEE32Exec.ofFloat`.

Example:
```lean
def w : Tensor TorchLean.Floats.IEEE32Exec (shape![2, 2]) :=
  tensor32! [[0.1, 0.2], [0.3, 0.4]]
```
-/
macro "tensor32!" xs:term:max : term =>
  `(Spec.mapTensor TorchLean.Floats.IEEE754.IEEE32Exec.ofFloat (tensor! $xs))

/-! ## Existentially shaped tensors -/

/-- A tensor paired with its shape, where the shape is stored as data instead of being in the type.

This is useful when:
- you need to accept arbitrary tensors at runtime (CLI / file formats / interoperability),
- but you still want to carry a `Shape` around so downstream code can inspect it.
-/
structure SomeTensor (α : Type) where
  /-- Runtime shape tag carried alongside the tensor. -/
  s : Spec.Shape
  /-- Shape-indexed tensor whose type is tied to `s`. -/
  t : Spec.Tensor α s

/-- Pair runtime dimensions with a tensor after checking the number of flat entries. -/
def someTensorOfList {α : Type} (dims : List Nat) (xs : List α) : Except String (SomeTensor α) := do
  let t ← ofList (α := α) dims xs
  pure { s := Spec.Shape.ofList dims, t := t }

/-! ## Printing -/

/-- A compact “dtype name” class used by `NN.Tensor.print`.

The class only records the display name needed by human-facing tensor output.
-/
class DTypeName (α : Type) where
  name : String

/-- Display name for `Float` tensors in `NN.Tensor.print`. -/
instance : DTypeName Float where
  name := "Float"

/-- Display name for native binary32 tensors. -/
instance : DTypeName Float32 where
  name := "Float32"
/-- Display name for rational tensors in `NN.Tensor.print`. -/
instance : DTypeName ℚ where
  name := "ℚ"
/-- Display name for integer tensors in `NN.Tensor.print`. -/
instance : DTypeName Int where
  name := "Int"
/-- Display name for proof-level `FP32` rounding-model tensors in `NN.Tensor.print`. -/
instance : DTypeName TorchLean.Floats.FP32 where
  name := "FP32"
/-- Display name for executable IEEE-754 FP32 tensors in `NN.Tensor.print`. -/
instance : DTypeName TorchLean.Floats.IEEE32Exec where
  name := "IEEE32Exec"
/-- Display name for proof-level real-valued tensors in `NN.Tensor.print`. -/
instance : DTypeName ℝ where
  name := "ℝ"

/-- Display name for TorchLean complex scalars in `NN.Tensor.print`. -/
instance {α : Type} [DTypeName α] : DTypeName (TorchLean.Complex α) where
  name := s!"Complex[{DTypeName.name (α := α)}]"

/-- A “pretty-printer with failure”.

We model printing as `Except String String` because some tensor element types are intentionally
non-printable (e.g. `ℝ` or proof-only floating-point models).
-/
class TensorPrintable (α : Type) where
  pretty : {s : Spec.Shape} → Spec.Tensor α s → Except String String

/-- Default printing for element types that support `ToString`. -/
instance (priority := 10) {α : Type} [ToString α] : TensorPrintable α where
  pretty := fun {_s} t => .ok (Spec.pretty t)

/-- Printing is disabled by design for proof-level `ℝ` tensors. -/
instance (priority := 100) : TensorPrintable ℝ where
  pretty := fun {_s} _ =>
    .error
      "Refusing to print `Tensor ℝ` (proof-level); cast to `Float`/`Float32`/`IEEE32Exec`/`ℚ` to display."

/-- Printing is disabled by design for the proof-only rounding model `FP32`. -/
instance (priority := 100) : TensorPrintable TorchLean.Floats.FP32 where
  pretty := fun {_s} _ =>
    .error
      ("Refusing to print `Tensor FP32` (proof-only rounding model); " ++
        "use `IEEE32Exec`/`Float` for runtime printing.")

/-- Print a tensor with a dtype tag, or throw an `IO.userError` if the dtype refuses to print. -/
def print {α : Type} [DTypeName α] [TensorPrintable α] {s : Spec.Shape}
    (t : Spec.Tensor α s) : IO Unit := do
  match (TensorPrintable.pretty (α := α) t) with
  | .ok str => IO.println s!"[{DTypeName.name (α := α)}] {str}"
  | .error msg => throw <| IO.userError msg

end NN.Tensor
