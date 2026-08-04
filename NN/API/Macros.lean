/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Lean

/-!
# Small Convenience Macros

This file contains only **general-purpose** syntactic sugar:
- `seq! a, b, c` for composing TorchLean `Seq` models without chaining `>>>` manually.
- `tensorpack! x, y, ...` for building `TorchLean.TensorPack` values without
  `.cons ... .nil` boilerplate.

The sequential macro expands to the composition helpers defined by `NN.API.Neural.Builders`.
The tensor-pack macro expands to constructors under `TorchLean.tensorpack`.

We avoid layer-specific "proof-eliding" macros here; prefer the named-field APIs in `NN.API.Neural`
for clarity and stable documentation.
-/

@[expose] public section


namespace TorchLean

/-- Compose `Seq` models without chaining `>>>` manually. -/
syntax (name := seqLit) "seq!" term,+ : term

/-!
## Sequential Literals

TorchLean sequential models are *shape-indexed* (`Seq σ τ`), so we cannot use a plain `List` of
layers like PyTorch does (a `List` would require every element to have the same type).

Instead we provide macros that expand to ordinary `Seq` composition while still letting users
write “list-shaped” model definitions.
-/

-- NOTE: This must *not* reserve the keyword `nn.Sequential`, because that breaks parsing of
-- expressions like `nn.Sequential σ τ` where `nn.Sequential` is used as a constant/type name.
--
-- So we provide the `...!` form (with `!`): `nn.Sequential![...]`.
syntax (name := nnSequentialBangLit) "nn.Sequential!" "[" term,+ "]" : term

private meta def mkGlobalIdent (val : Lean.Name) : Lean.Ident :=
  -- Use a *macro-scope-free* identifier, so expansions refer to the actual constant name
  -- (e.g. `...compAny`) instead of a macro-scoped one (`...compAny✝`).
  ⟨Lean.Syntax.ident Lean.SourceInfo.none (toString val).toRawSubstring val []⟩

macro_rules (kind := nnSequentialBangLit)
  | `(nn.Sequential![$a:term]) =>
      let f := mkGlobalIdent `_root_.TorchLean.nn.Internal.AsSequential.asSequential
      `(do
        let a ← ($a)
        pure ($f a))
  | `(nn.Sequential![$a:term, $b:term]) =>
      let f := mkGlobalIdent `_root_.TorchLean.nn.Internal.compose
      `(do
        let a ← ($a)
        let b ← ($b)
        pure ($f a b))
  | `(nn.Sequential![$a:term, $b:term, $rest:term,*]) =>
      let f := mkGlobalIdent `_root_.TorchLean.nn.Internal.compose
      `(do
        let a ← ($a)
        let bc ← (nn.Sequential![$b, $rest,*])
        pure ($f a bc))

macro_rules (kind := seqLit)
  | `(seq! $a:term) => `($a)
  | `(seq! $a:term, $b:term) =>
      let f := mkGlobalIdent `_root_.TorchLean.nn.Internal.compose
      `($f $a $b)
  | `(seq! $a:term, $b:term, $rest:term,*) =>
      let f := mkGlobalIdent `_root_.TorchLean.nn.Internal.compose
      `($f $a (seq! $b, $rest,*))

/-- Build a `TorchLean.TensorPack` from comma-separated tensors. -/
syntax (name := tensorpackLit) "tensorpack!" term,+ : term

macro_rules (kind := tensorpackLit)
  | `(tensorpack! $x:term) =>
      let f := mkGlobalIdent `_root_.TorchLean.tensorpack.singleton
      `($f $x)
  | `(tensorpack! $x:term, $y:term) =>
      let f := mkGlobalIdent `_root_.TorchLean.tensorpack.pair
      `($f $x $y)
  | `(tensorpack! $x:term, $y:term, $rest:term,*) =>
      `(.cons $x (tensorpack! $y, $rest,*))

end TorchLean
