/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Torch.Core
public import NN.Tensor.API
import Mathlib.Algebra.Order.Algebra

/-!
# Torch Utils

Helpers for writing PyTorch-style training loops on top of `Runtime.Autograd.Torch`.

This file focuses on training-loop ergonomics:
- extract scalar values,
- build short `TList`s,
- run simple SGD loops for `Torch.ScalarTrainer`.

Stateful optimizer loops live in `Runtime.Autograd.TorchLean`, because those depend on
`TorchLean.Optim`. Keeping that dependency out of this low-level utility module prevents the
session/ref layer from depending upward on the model/optimizer API surface.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace Torch

open Spec
open Tensor
open Proofs.Autograd.Algebra

/-! ## Initialization helpers (Float constants) -/

namespace Init

/--
Deterministic initialization schemes stored as `Float` constants.

PyTorch comparison:
- `.zeros` / `.ones` correspond to `torch.nn.init.zeros_` / `torch.nn.init.ones_`
- `.uniform lo hi` corresponds to `torch.nn.init.uniform_` (with explicit `a=lo`, `b=hi`)
- `.xavierUniform fanIn fanOut` corresponds to `torch.nn.init.xavier_uniform_`
- `.kaimingUniform fanIn` corresponds to `torch.nn.init.kaiming_uniform_` with `nonlinearity="relu"`

References:
- https://pytorch.org/docs/stable/nn.init.html
- https://pytorch.org/docs/stable/generated/torch.nn.init.xavier_uniform_.html
- https://pytorch.org/docs/stable/generated/torch.nn.init.kaiming_uniform_.html
-/
inductive Scheme where
  | zeros
  | ones
  | uniform (lo hi : Float)
  | normal (mean std : Float)
  | xavierUniform (fanIn fanOut : Nat)
  | kaimingUniform (fanIn : Nat)
  deriving Repr

namespace Internal

/-- SplitMix64 mixing used by indexed parameter initialization. -/
def splitmix64 (x : UInt64) : UInt64 :=
  let z1 := x + 0x9e3779b97f4a7c15
  let z2 := (z1 ^^^ (z1 >>> 30)) * 0xbf58476d1ce4e5b9
  let z3 := (z2 ^^^ (z2 >>> 27)) * 0x94d049bb133111eb
  z3 ^^^ (z3 >>> 31)

/--
Deterministic `U[0,1)` sampler derived independently from a seed and scalar index.

Indexing is constant-time, so constructing an `n`-element tensor takes `O(n)` sampler work. The
same key/index formula is used by TorchLean's storage-first runtime initializer.
-/
def rand01 (seed idx : Nat) : Float :=
  let key := splitmix64 (UInt64.ofNat seed)
  let z := splitmix64 (key + UInt64.ofNat idx)
  (Float.ofNat z.toUInt32.toNat) / 4294967296.0

/--
Sample the `idx`-th scalar of a tensor initialized using `Scheme`.

This is the scalar-level primitive used by `Init.tensor`.
-/
def sampleAt (sch : Scheme) (seed idx : Nat) : Float :=
  match sch with
  | .zeros => 0.0
  | .ones => 1.0
  | .uniform lo hi =>
      lo + rand01 seed idx * (hi - lo)
  | .normal mean std =>
      let u1 := (Float.ofNat (splitmix64 (splitmix64 (UInt64.ofNat seed)) +
        UInt64.ofNat (2 * idx)).toUInt32.toNat + 1) / 4294967297.0
      let u2 := rand01 seed (2 * idx + 1)
      mean + std * Float.sqrt (-2.0 * Float.log u1) * Float.cos (6.283185307179586 * u2)
  | .xavierUniform fanIn fanOut =>
      let denom := (Float.ofNat fanIn) + (Float.ofNat fanOut)
      let limit := Float.sqrt (6.0 / denom)
      (-limit) + rand01 seed idx * (2.0 * limit)
  | .kaimingUniform fanIn =>
      let limit := Float.sqrt (6.0 / (Float.ofNat fanIn))
      (-limit) + rand01 seed idx * (2.0 * limit)

end Internal

open _root_.Runtime.Autograd.Torch.Init.Internal

/--
Create a `Tensor Float s` by sampling a `Scheme` deterministically.

This pure initializer is convenient for model definitions and reproducible examples. Runtime
initialization paths can provide more specialized allocation strategies for very large tensors.

PyTorch comparison: this mimics using `torch.nn.init.*` routines on freshly allocated parameters,
but here we work with *pure* `Tensor Float s` values (no mutation) and use a deterministic
seeded sampler for reproducibility.
-/
def tensor (sch : Scheme) (seed : Nat := 0) : {s : Shape} → Tensor Float s
  | .scalar =>
      Tensor.scalar (sampleAt sch seed 0)
  | .dim _n s' =>
      let chunk := Spec.Shape.size s'
      Tensor.dim (fun i =>
        -- offset-by-block so different blocks get different samples
        let seed' := seed
        let idxBase := i.val * chunk
        -- build a sub-tensor whose scalars use indices `idxBase + k`
        let rec build : {t : Shape} → Nat → Tensor Float t
          | .scalar, k => Tensor.scalar (sampleAt sch seed' (idxBase + k))
          | .dim _m t', k =>
              let chunk' := Spec.Shape.size t'
              Tensor.dim (fun j => build (t := t') (k + j.val * chunk'))
        build (t := s') 0)

/--
Xavier/Glorot-uniform initializer for 2D weight matrices.

PyTorch comparison: `torch.nn.init.xavier_uniform_` with `gain=1`.
-/
def xavierW (outDim inDim : Nat) (seed : Nat := 0) :
    Tensor Float (.dim outDim (.dim inDim .scalar)) :=
  tensor (s := .dim outDim (.dim inDim .scalar)) (sch := .xavierUniform inDim outDim) (seed := seed)

/--
Kaiming/He-uniform initializer for 2D weight matrices.

PyTorch comparison: `torch.nn.init.kaiming_uniform_` with `nonlinearity="relu"` and default
parameters (so the bound is `sqrt(6/fan_in)`).
-/
def kaimingW (outDim inDim : Nat) (seed : Nat := 0) :
    Tensor Float (.dim outDim (.dim inDim .scalar)) :=
  tensor (s := .dim outDim (.dim inDim .scalar)) (sch := .kaimingUniform inDim) (seed := seed)

end Init

/-! ## Conveniences for scalar training loops -/

/--
Extract the scalar value from a scalar-shaped tensor.

PyTorch comparison: like `t.item()` for a 0-dim tensor.
-/
abbrev scalarOf {α : Type} (t : Tensor α Shape.scalar) : α :=
  t.item

/-- Build a one-element `TList` (useful for curried trainer APIs). -/
def tlistSingleton {α : Type} {s₁ : Shape} (x₁ : Tensor α s₁) : TList α [s₁] :=
  .cons x₁ .nil

/-! ## `TList` syntax sugar -/

/--
Build a `TList` from a comma-separated list of terms.

This is meant for training code where `tlistSingleton`/`tlistPair`/… becomes tedious.

Example:

```lean
let xs : TList Float [.dim 2 .scalar, .dim 1 .scalar] :=
  tlist![x, y]
```
-/
syntax (name := tlistBang) "tlist!" "[" term,* "]" : term

macro_rules
  | `(tlist![ $xs:term,* ]) => do
      let xs := xs.getElems.toList
      let rec go : List (Lean.TSyntax `term) → Lean.MacroM (Lean.TSyntax `term)
        | [] => `(Proofs.Autograd.Algebra.TList.nil)
        | x :: xs => do
            let tail ← go xs
            `(Proofs.Autograd.Algebra.TList.cons $x $tail)
      go xs

/-- Build a two-element `TList` (useful for curried trainer APIs). -/
def tlistPair {α : Type} {s₁ s₂ : Shape} (x₁ : Tensor α s₁) (x₂ : Tensor α s₂) : TList α [s₁, s₂] :=
  .cons x₁ (.cons x₂ .nil)

/-- Build a three-element `TList` (useful for curried trainer APIs). -/
def tlistTriple {α : Type} {s₁ s₂ s₃ : Shape}
    (x₁ : Tensor α s₁) (x₂ : Tensor α s₂) (x₃ : Tensor α s₃) : TList α [s₁, s₂, s₃] :=
  .cons x₁ (.cons x₂ (.cons x₃ .nil))

/-- Build a four-element `TList` (useful for curried trainer APIs). -/
def tlistQuad {α : Type} {s₁ s₂ s₃ s₄ : Shape}
    (x₁ : Tensor α s₁) (x₂ : Tensor α s₂) (x₃ : Tensor α s₃) (x₄ : Tensor α s₄) : TList α [s₁, s₂,
      s₃, s₄] :=
  .cons x₁ (.cons x₂ (.cons x₃ (.cons x₄ .nil)))

namespace ScalarTrainer

/--
Uncurried forward pass for `ScalarTrainer`.

`ScalarTrainer.forward` is stored as a curried function over the input shapes; this helper lets you
pass a `TList` (like a tuple of tensors).
-/
def forwardT {α : Type} {paramShapes inputShapes natInputShapes : List Shape}
    (tr : ScalarTrainer α paramShapes inputShapes natInputShapes)
    (xs : TList α inputShapes) (natInputs : TList Nat natInputShapes) :
    IO (Tensor α Shape.scalar) :=
  let withNat := Curried.uncurry (α := α) (ss := inputShapes)
    (β := Curried.Fn Nat natInputShapes (IO (Tensor α Shape.scalar))) tr.forward xs
  Curried.uncurry (α := Nat) (ss := natInputShapes)
    (β := IO (Tensor α Shape.scalar)) withNat natInputs

/--
Uncurried loss-and-gradient pass for `ScalarTrainer`.

The loss and gradients come from the same tape. Use this instead of calling `forwardT` followed by
`backwardT` when both results are needed.
-/
def lossAndBackwardT {α : Type} {paramShapes inputShapes natInputShapes : List Shape}
    (tr : ScalarTrainer α paramShapes inputShapes natInputShapes)
    (xs : TList α inputShapes) (natInputs : TList Nat natInputShapes) :
    IO (Tensor α Shape.scalar × TList α paramShapes) :=
  let withNat := Curried.uncurry (α := α) (ss := inputShapes)
    (β := Curried.Fn Nat natInputShapes
      (IO (Tensor α Shape.scalar × TList α paramShapes))) tr.lossAndBackward xs
  Curried.uncurry (α := Nat) (ss := natInputShapes)
    (β := IO (Tensor α Shape.scalar × TList α paramShapes)) withNat natInputs

/--
Uncurried backward pass for `ScalarTrainer`.

Returns per-parameter gradients (aligned with `paramShapes`).
-/
def backwardT {α : Type} {paramShapes inputShapes natInputShapes : List Shape}
    (tr : ScalarTrainer α paramShapes inputShapes natInputShapes)
    (xs : TList α inputShapes) (natInputs : TList Nat natInputShapes) :
    IO (TList α paramShapes) :=
  let withNat := Curried.uncurry (α := α) (ss := inputShapes)
    (β := Curried.Fn Nat natInputShapes (IO (TList α paramShapes))) tr.backward xs
  Curried.uncurry (α := Nat) (ss := natInputShapes)
    (β := IO (TList α paramShapes)) withNat natInputs

/--
Uncurried SGD step for `ScalarTrainer`.

PyTorch comparison: analogous to `loss.backward(); optimizer.step()` for a fixed SGD optimizer,
except here the trainer bundles the update rule.
-/
def stepT {α : Type} {paramShapes inputShapes natInputShapes : List Shape}
    (tr : ScalarTrainer α paramShapes inputShapes natInputShapes)
    (lr : α) (xs : TList α inputShapes) (natInputs : TList Nat natInputShapes) : IO Unit :=
  let withNat := Curried.uncurry (α := α) (ss := inputShapes)
    (β := Curried.Fn Nat natInputShapes (IO Unit)) (tr.step lr) xs
  Curried.uncurry (α := Nat) (ss := natInputShapes) (β := IO Unit) withNat natInputs

/-- Uncurried SGD step that returns the loss used to compute the update. -/
def stepWithLossT {α : Type} {paramShapes inputShapes natInputShapes : List Shape}
    (tr : ScalarTrainer α paramShapes inputShapes natInputShapes)
    (lr : α) (xs : TList α inputShapes) (natInputs : TList Nat natInputShapes) :
    IO (Tensor α Shape.scalar) :=
  let withNat := Curried.uncurry (α := α) (ss := inputShapes)
    (β := Curried.Fn Nat natInputShapes (IO (Tensor α Shape.scalar)))
    (tr.stepWithLoss lr) xs
  Curried.uncurry (α := Nat) (ss := natInputShapes)
    (β := IO (Tensor α Shape.scalar)) withNat natInputs

end ScalarTrainer

/--
Train `steps` SGD updates, cycling through `samples`.

PyTorch comparison: this matches the common eager training skeleton:

```
for step in range(steps):
  batch = dataset[step % len(dataset)]
  loss = forward(batch)
  step(lr, batch)   # typically: loss.backward(); optimizer.step()
```

Note: `ScalarTrainer.step` is the "bundled SGD optimizer" for the trainer. Stateful optimizers
(Adam, RMSProp, ...) are exposed from `Runtime.Autograd.TorchLean`.
-/
def trainCycleSGD
    {α : Type} [ToString α] {paramShapes inputShapes : List Shape}
    (tr : ScalarTrainer α paramShapes inputShapes)
    (lr : α) (steps : Nat) (samples : List (TList α inputShapes)) (logEvery : Nat := 1) : IO Unit :=
      do
  match samples with
  | [] =>
      throw <| IO.userError "trainCycleSGD: empty dataset"
  | hd :: _tl =>
      for step in [0:steps] do
        let xs := samples.getD (step % samples.length) hd
        let lossT ← ScalarTrainer.forwardT (α := α) (paramShapes := paramShapes) (inputShapes :=
          inputShapes) tr xs .nil
        if logEvery != 0 && step % logEvery = 0 then
          IO.println s!"step {step}: loss={scalarOf lossT}"
        ScalarTrainer.stepT (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes) tr lr
          xs .nil

/--
Evaluate mean loss over a dataset.

PyTorch comparison: like running a model in `torch.no_grad()` over a dataloader and averaging
the scalar loss values, except here we call `ScalarTrainer.forwardT` directly.
-/
def meanLoss
    {α : Type} [ToString α] [Add α] [Div α] [Zero α] [Coe Nat α]
    {paramShapes inputShapes : List Shape}
    (tr : ScalarTrainer α paramShapes inputShapes)
    (samples : List (TList α inputShapes)) : IO α := do
  match samples with
  | [] =>
      throw <| IO.userError "meanLoss: empty dataset"
  | _ =>
      let mut acc : α := 0
      for xs in samples do
        let lossT ← ScalarTrainer.forwardT (α := α) (paramShapes := paramShapes) (inputShapes :=
          inputShapes) tr xs .nil
        acc := acc + scalarOf lossT
      pure (acc / (samples.length : α))

end Torch
end Autograd
end Runtime
