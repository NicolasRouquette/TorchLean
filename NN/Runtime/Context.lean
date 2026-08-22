/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Packed

/-!
# Runtime context

This module defines the runtime environment for executing TorchLean computations.

The main challenge is that tensors are dependently typed by their `Shape`, but at runtime we want
to store a heterogeneous map from names to values. `Spec.PackedTensor` carries each runtime shape
alongside its tensor. Lookup functions check that shape and use the resulting equality proof to
recover the requested typed tensor.

## Main declarations

- `Spec.PackedTensor` is the shape-erased wrapper used by the registries.
- `RuntimeContext` is the mutable-looking record that stores values, gradients, and a fresh-id
  counter.
- `register_variable` / `get_variable` are the core value registry operations.
- `register_gradient` / `get_gradient` do the same on the gradient side.
-/

@[expose] public section


namespace Runtime

open Spec
open Tensor

/--
Runtime context for tracking named variables and their gradients.

We keep two registries:
- `bindings` for values, and
- `gradients` for accumulated gradients (typically produced by backprop).
-/
structure RuntimeContext (α : Type) where
  /--
  Registry of named variable values.

  The registry is ordered by shadowing priority: lookup returns the first matching name.
  -/
  bindings : List (String × PackedTensor α)
  /--
  Registry of named gradients.

  The gradient registry uses the same ordering convention as `bindings`. Higher-level training
  code decides whether a gradient should shadow, replace, or accumulate with an existing entry.
  -/
  gradients : List (String × PackedTensor α)
  /--
  Fresh id counter for allocating new variables/parameters.

  This is incremented by `registerVariable` and used by higher-level runtime layers.
  -/
  nextId : Nat

/-- An empty context with no variables and no gradients. -/
def emptyContext {α : Type} : RuntimeContext α :=
  { bindings := [], gradients := [], nextId := 0 }

/--
Register a new variable in the context.

The value is packed with its shape so we can store it in a heterogeneous registry.
-/
def registerVariable {α : Type} {s : Shape} (ctx : RuntimeContext α) (name : String) (value : Tensor
  α s) : RuntimeContext α :=
  let packed := PackedTensor.ofTensor value
  { ctx with
    bindings := (name, packed) :: ctx.bindings,
    nextId := ctx.nextId + 1 }

/--
Lookup a variable by name and requested shape.

If the name exists and the stored `Shape` matches `s`, we cast the stored tensor to `Tensor α s`.
-/
def getVariable {α : Type} {s : Shape} [DecidableEq Shape]
  (ctx : RuntimeContext α) (name : String) : Option (Tensor α s) :=
  match ctx.bindings.find? (fun (n, _) => n == name) with
  | none => none
  | some (_, packed) =>
    if h : packed.shape = s then
      some (packed.cast h)
    else
      none

/--
Update the value of an existing variable (or do nothing if the name is absent).

The registry order is preserved. Entries with the requested name receive the new shape-tagged
tensor; other entries are left unchanged.
-/
def setVariable {α : Type} {s : Shape} (ctx : RuntimeContext α) (name : String) (value : Tensor α s)
  : RuntimeContext α :=
  let packed := PackedTensor.ofTensor value
  let updatedBindings := ctx.bindings.map (fun (n, v) => if n == name then (n, packed) else (n,
    v))
  { ctx with bindings := updatedBindings }

/--
Register (prepend) a gradient entry in the context.

If you want to *accumulate* gradients under the same name, do that at a higher layer before
calling this helper.
-/
def registerGradient {α : Type} {s : Shape} (ctx : RuntimeContext α) (name : String) (grad : Tensor
  α s) : RuntimeContext α :=
  let packed := PackedTensor.ofTensor grad
  { ctx with gradients := (name, packed) :: ctx.gradients }

/-- Lookup a gradient by name and requested shape. -/
def getGradient {α : Type} {s : Shape} [DecidableEq Shape] (ctx : RuntimeContext α) (name : String)
  : Option (Tensor α s) :=
  match ctx.gradients.find? (fun (n, _) => n == name) |>.map (fun (_, packed) => packed)
    with
  | none => none
  | some packed =>
    if h : packed.shape = s then
      some (packed.cast h)
    else
      none

/-- Update gradient entries with a matching name while preserving registry order. -/
def setGradient {α : Type} {s : Shape} (ctx : RuntimeContext α) (name : String) (grad : Tensor α s)
  : RuntimeContext α :=
  let packed := PackedTensor.ofTensor grad
  let updatedGradients := ctx.gradients.map (fun (n, g) => if n == name then (n, packed) else (n,
    g))
  { ctx with gradients := updatedGradients }

/-- Remove all gradient entries (analogue of PyTorch `optimizer.zero_grad()`). -/
def clearGradients {α : Type} (ctx : RuntimeContext α) : RuntimeContext α :=
  { ctx with gradients := [] }

/-- Return `true` iff the context contains a variable named `name`. -/
def hasVariable {α : Type} (ctx : RuntimeContext α) (name : String) : Bool :=
  ctx.bindings.any (fun (n, _) => n == name)

/-- List all variable names stored in `ctx`. -/
def variableNames {α : Type} (ctx : RuntimeContext α) : List String :=
  ctx.bindings.map (fun (n, _) => n)

/-- List all gradient names stored in `ctx`. -/
def gradientNames {α : Type} (ctx : RuntimeContext α) : List String :=
  ctx.gradients.map (fun (n, _) => n)

/-- Number of registered variables. -/
def contextSize {α : Type} (ctx : RuntimeContext α) : Nat :=
  ctx.bindings.length

/-- Number of registered gradient entries. -/
def gradientCount {α : Type} (ctx : RuntimeContext α) : Nat :=
  ctx.gradients.length

/--
Check a simple invariant: every gradient entry refers to an existing variable name.

This invariant checks name presence. Shape agreement is checked by the typed lookup functions.
-/
def isValidContext {α : Type} (ctx : RuntimeContext α) : Bool :=
  ctx.gradients.all (fun (name, _) => hasVariable ctx name)

/-- Render the context contents as a string (for debugging). -/
def contextToString {α : Type} [ToString α] (ctx : RuntimeContext α) : String :=
  let variablesString := String.intercalate ", " (ctx.bindings.map (fun (n, packed) =>
    s!"{n}: {pretty packed.tensor}"))
  let gradientsString := String.intercalate ", " (ctx.gradients.map (fun (n, packed) =>
    s!"{n}: {pretty packed.tensor}"))
  s!"Context(vars: [{variablesString}], grads: [{gradientsString}])"
end Runtime
