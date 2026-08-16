/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.Program
public import NN.Runtime.Autograd.TorchLean.Dual

import Mathlib.Algebra.Order.Algebra

/-!
# Autodiff

Autodiff utilities beyond basic `.backward()`:

- `hvpParams`: Hessian-vector product for scalar losses w.r.t. parameters, using
  forward-over-reverse via `Dual` scalars.

This is runtime/executable functionality intended for TorchLean ergonomics; it is separate from
the `fderiv` proof developments.
-/

@[expose] public section


namespace Runtime
namespace Autograd
namespace TorchLean

open Spec
open Tensor
open Proofs.Autograd.Algebra

namespace Autodiff

namespace Internal

/--
Unwrap a runtime `Result` into `IO`, throwing a user error on failure.

This is used throughout this module because lowering and backpropagation utilities return an
`Autograd.Result` with a structured error message.
-/
def okOrThrow {α : Type} : Runtime.Autograd.Result α → IO α
  | .ok a => pure a
  | .error e => throw <| IO.userError e

/--
Basis-direction tensors for a shape `s`.

For a scalar shape this is `[1]`. For a tensor shape it returns one tensor per basis direction
in the flattened coordinate system, with a `1` in that position and `0` elsewhere.

These basis tensors are used to compute Jacobians by repeated VJP/JVP calls.
-/
def basisTensors {α : Type} [Context α] : (s : Shape) → Array (Tensor α s)
  | .scalar =>
      #[Tensor.scalar (1 : α)]
  | .dim n s =>
      let z : Tensor α s := Spec.zeros (α := α) s
      let sub : Array (Tensor α s) := basisTensors (α := α) s
      let subL : List (Tensor α s) := sub.toList
      let all : List (Tensor α (.dim n s)) :=
        (List.finRange n).flatMap (fun i =>
          subL.map (fun b => Tensor.dim (fun j => if j = i then b else z)))
      all.toArray

/--
Append two typed tensor lists.

This is a small helper for building `args : TList α (paramShapes ++ inputShapes)` in a way that
keeps the shape indices explicit and type-correct.
-/
abbrev tlistAppend {α : Type} :
    {ss₁ ss₂ : List Shape} → TList α ss₁ → TList α ss₂ → TList α (ss₁ ++ ss₂) :=
  _root_.Proofs.Autograd.Algebra.TList.append (α := α)

/--
Split a typed list indexed by `ss₁ ++ ss₂` back into `(ss₁, ss₂)` pieces.

This is the inverse of `tlistAppend`.
-/
abbrev tlistSplitAppend {α : Type} :
    {ss₁ ss₂ : List Shape} → TList α (ss₁ ++ ss₂) → TList α ss₁ × TList α ss₂ :=
  _root_.Proofs.Autograd.Algebra.TList.splitAppend (α := α)

/--
Cast a prefix of a dense gradient array into a typed list `TList α ss`.

This is used when we ask the typed graph engine for a dense list of gradients with respect to all inputs and
need to recover a shape-typed view.
-/
def gradsPrefix {α : Type} [DecidableEq Shape] :
    {ss : List Shape} → Array (Runtime.AnyTensor α) → Nat → IO (TList α ss)
  | [], _grads, _off => pure .nil
  | s :: ss, grads, off => do
      let any ← match grads[off]? with
        | some v => pure v
        | none => throw <| IO.userError "torchlean(hvp): gradient array too small"
        if h : any.s = s then
          let g : Tensor α s := Tensor.castShape any.t h
          let gs ← gradsPrefix (α := α) (ss := ss) grads (off + 1)
          pure (.cons g gs)
        else
          throw <| IO.userError <|
            s!"torchlean(hvp): gradient shape mismatch at idx={off} (expected "
              ++ s!"{Shape.pretty s}, got "
              ++ s!"{Shape.pretty any.s})"

end Internal

open Internal

/-- Lower a scalar-valued TorchLean program to a reusable typed graph. -/
def lowerScalarToTypedGraph {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes inputShapes : List Shape}
    (program :
      ∀ {β : Type}, [Context β] → [DecidableEq Shape] →
        TorchLean.Program β (paramShapes ++ inputShapes) Shape.scalar) :
    IO (_root_.Runtime.Autograd.Torch.TypedScalarGraph α (paramShapes ++ inputShapes)) := do
  let Γ : List Shape := paramShapes ++ inputShapes
  let build : Runtime.Autograd.TypedGraph.GraphM.M α Γ (Runtime.Autograd.TypedGraph.GraphM.Var
    Shape.scalar) := do
    let vs ← Runtime.Autograd.TypedGraph.GraphM.args (α := α) (Γ := Γ)
    CurriedRef.applyVarList (Γ := Γ)
      (β := Runtime.Autograd.TypedGraph.GraphM.M α Γ (Runtime.Autograd.TypedGraph.GraphM.Var
        Shape.scalar))
      (program (β := α) (m := Runtime.Autograd.TypedGraph.GraphM.M α Γ)) vs
  okOrThrow (_root_.Runtime.Autograd.Torch.lowerScalarToTypedGraph (α := α) (Γ := Γ) build)

/--
Lower a TorchLean program to a reusable `TypedGraph`.

The graph retains a typed reference to the value returned by the program, which may be an input or
any recorded node. This is the tensor-output analogue of `lowerScalarToTypedGraph`; it is
used by `jacrevOut*` and `vjpOut*`.
-/
def lowerToTypedGraph {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes inputShapes : List Shape} {τ : Shape}
    (f :
      ∀ {β : Type}, [Context β] → [DecidableEq Shape] →
        TorchLean.Program β (paramShapes ++ inputShapes) τ) :
    IO (_root_.Runtime.Autograd.Torch.TypedGraph α (paramShapes ++ inputShapes) τ) := do
  let Γ : List Shape := paramShapes ++ inputShapes
  let build : Runtime.Autograd.TypedGraph.GraphM.M α Γ (Runtime.Autograd.TypedGraph.GraphM.Var τ) := do
    let vs ← Runtime.Autograd.TypedGraph.GraphM.args (α := α) (Γ := Γ)
    CurriedRef.applyVarList (Γ := Γ)
      (β := Runtime.Autograd.TypedGraph.GraphM.M α Γ (Runtime.Autograd.TypedGraph.GraphM.Var τ))
      (f (β := α) (m := Runtime.Autograd.TypedGraph.GraphM.M α Γ)) vs
  okOrThrow (_root_.Runtime.Autograd.Torch.lowerToTypedGraph (α := α) (Γ := Γ) (τ := τ) build)

/-- Jacobian (reverse-mode) of a tensor output w.r.t. parameters, as an array of VJPs. -/
def jacrevOutParams {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes inputShapes : List Shape} {τ : Shape}
    (f :
      ∀ {β : Type}, [Context β] → [DecidableEq Shape] →
        TorchLean.Program β (paramShapes ++ inputShapes) τ)
    (params : TList α paramShapes)
    (xs : TList α inputShapes) :
    IO (Array (TList α paramShapes)) := do
  let c ← lowerToTypedGraph (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes) (τ := τ) f
  let Γ : List Shape := paramShapes ++ inputShapes
  let args : TList α Γ := tlistAppend (α := α) (ss₁ := paramShapes) (ss₂ := inputShapes) params xs
  let seeds : Array (Tensor α τ) := basisTensors (α := α) τ
  let rows : Array (TList α paramShapes) :=
    seeds.map (fun seedOut =>
      let gAll : TList α Γ :=
        _root_.Runtime.Autograd.Torch.TypedGraph.vjpWithSeed (α := α) (Γ := Γ) (τ := τ) c args
          seedOut
      (tlistSplitAppend (α := α) (ss₁ := paramShapes) (ss₂ := inputShapes) gAll).1)
  pure rows

/-- Jacobian (reverse-mode) of a tensor output w.r.t. inputs, as an array of VJPs. -/
def jacrevOutInputs {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes inputShapes : List Shape} {τ : Shape}
    (f :
      ∀ {β : Type}, [Context β] → [DecidableEq Shape] →
        TorchLean.Program β (paramShapes ++ inputShapes) τ)
    (params : TList α paramShapes)
    (xs : TList α inputShapes) :
    IO (Array (TList α inputShapes)) := do
  let c ← lowerToTypedGraph (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes) (τ := τ) f
  let Γ : List Shape := paramShapes ++ inputShapes
  let args : TList α Γ := tlistAppend (α := α) (ss₁ := paramShapes) (ss₂ := inputShapes) params xs
  let seeds : Array (Tensor α τ) := basisTensors (α := α) τ
  let rows : Array (TList α inputShapes) :=
    seeds.map (fun seedOut =>
      let gAll : TList α Γ :=
        _root_.Runtime.Autograd.Torch.TypedGraph.vjpWithSeed (α := α) (Γ := Γ) (τ := τ) c args
          seedOut
      (tlistSplitAppend (α := α) (ss₁ := paramShapes) (ss₂ := inputShapes) gAll).2)
  pure rows

/--
Jacobian (forward-mode) of a tensor output for a *single* tensor input.

Returns the Jacobian columns as `Array (Tensor α τ)`, one column per input basis direction.

Implementation note: this uses **dual-number forward evaluation** (record and run under `Dual α`)
instead of graph-level `jvp`, because the typed graph provides VJPs broadly but not
JVP rules for every op.
-/
def jacfwdInput {α : Type} [Context α] [DecidableEq Shape]
    {σ τ : Shape}
    (f :
      ∀ {β : Type}, [Context β] → [DecidableEq Shape] →
        TorchLean.Program β [σ] τ)
    (x : Tensor α σ) :
    IO (Array (Tensor α τ)) := do
  let αD := Dual α
  let c ← lowerToTypedGraph (α := αD)
    (paramShapes := ([] : List Shape)) (inputShapes := [σ]) (τ := τ)
    (fun {β} _ _ => f (β := β))
  let dirs : Array (Tensor α σ) := basisTensors (α := α) σ
  pure <| dirs.map (fun dx =>
    let xD : Tensor αD σ := DualTensor.withTangents (α := α) (s := σ) x dx
    let outD : Tensor αD τ :=
      _root_.Runtime.Autograd.Torch.TypedGraph.forward (α := αD) (Γ := [σ]) (τ := τ) c (.cons xD
        .nil)
    DualTensor.tangent (α := α) (s := τ) outD)

/-- Gradient of scalar loss w.r.t. parameters (reverse-mode). -/
def gradParams {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes inputShapes : List Shape}
    (loss :
      ∀ {β : Type}, [Context β] → [DecidableEq Shape] →
        TorchLean.Program β (paramShapes ++ inputShapes) Shape.scalar)
    (params : TList α paramShapes)
    (xs : TList α inputShapes) :
    IO (TList α paramShapes) := do
  let c ← lowerScalarToTypedGraph (α := α)
    (paramShapes := paramShapes) (inputShapes := inputShapes) loss
  let Γ : List Shape := paramShapes ++ inputShapes
  let args : TList α Γ := tlistAppend (α := α) (ss₁ := paramShapes) (ss₂ := inputShapes) params xs
  let gAll : TList α Γ := _root_.Runtime.Autograd.Torch.TypedScalarGraph.backward (α := α) (Γ := Γ) c
    args
  pure (tlistSplitAppend (α := α) (ss₁ := paramShapes) (ss₂ := inputShapes) gAll).1

/-- Gradient of scalar loss w.r.t. inputs (reverse-mode). -/
def gradInputs {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes inputShapes : List Shape}
    (loss :
      ∀ {β : Type}, [Context β] → [DecidableEq Shape] →
        TorchLean.Program β (paramShapes ++ inputShapes) Shape.scalar)
    (params : TList α paramShapes)
    (xs : TList α inputShapes) :
    IO (TList α inputShapes) := do
  let c ← lowerScalarToTypedGraph (α := α)
    (paramShapes := paramShapes) (inputShapes := inputShapes) loss
  let Γ : List Shape := paramShapes ++ inputShapes
  let args : TList α Γ := tlistAppend (α := α) (ss₁ := paramShapes) (ss₂ := inputShapes) params xs
  let gAll : TList α Γ := _root_.Runtime.Autograd.Torch.TypedScalarGraph.backward (α := α) (Γ := Γ) c
    args
  pure (tlistSplitAppend (α := α) (ss₁ := paramShapes) (ss₂ := inputShapes) gAll).2

/-- VJP of a tensor output w.r.t. parameters. -/
def vjpOutParams {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes inputShapes : List Shape} {τ : Shape}
    (f :
      ∀ {β : Type}, [Context β] → [DecidableEq Shape] →
        TorchLean.Program β (paramShapes ++ inputShapes) τ)
    (params : TList α paramShapes)
    (xs : TList α inputShapes)
    (seedOut : Tensor α τ) :
    IO (TList α paramShapes) := do
  let c ← lowerToTypedGraph (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes) (τ := τ) f
  let Γ : List Shape := paramShapes ++ inputShapes
  let args : TList α Γ := tlistAppend (α := α) (ss₁ := paramShapes) (ss₂ := inputShapes) params xs
  let gAll : TList α Γ := _root_.Runtime.Autograd.Torch.TypedGraph.vjpWithSeed (α := α) (Γ := Γ) (τ
    := τ) c args seedOut
  pure (tlistSplitAppend (α := α) (ss₁ := paramShapes) (ss₂ := inputShapes) gAll).1

/-- VJP of a tensor output w.r.t. inputs. -/
def vjpOutInputs {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes inputShapes : List Shape} {τ : Shape}
    (f :
      ∀ {β : Type}, [Context β] → [DecidableEq Shape] →
        TorchLean.Program β (paramShapes ++ inputShapes) τ)
    (params : TList α paramShapes)
    (xs : TList α inputShapes)
    (seedOut : Tensor α τ) :
    IO (TList α inputShapes) := do
  let c ← lowerToTypedGraph (α := α) (paramShapes := paramShapes) (inputShapes := inputShapes) (τ := τ) f
  let Γ : List Shape := paramShapes ++ inputShapes
  let args : TList α Γ := tlistAppend (α := α) (ss₁ := paramShapes) (ss₂ := inputShapes) params xs
  let gAll : TList α Γ := _root_.Runtime.Autograd.Torch.TypedGraph.vjpWithSeed (α := α) (Γ := Γ) (τ
    := τ) c args seedOut
  pure (tlistSplitAppend (α := α) (ss₁ := paramShapes) (ss₂ := inputShapes) gAll).2

/-- Directional derivative of scalar loss along `vparams` (forward-mode JVP). -/
def jvpLossParams {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes inputShapes : List Shape}
    (loss :
      ∀ {β : Type}, [Context β] → [DecidableEq Shape] →
        TorchLean.Program β (paramShapes ++ inputShapes) Shape.scalar)
    (params : TList α paramShapes)
    (xs : TList α inputShapes)
    (vparams : TList α paramShapes) :
    IO α := do
  let c ← lowerScalarToTypedGraph (α := α)
    (paramShapes := paramShapes) (inputShapes := inputShapes) loss
  let Γ : List Shape := paramShapes ++ inputShapes
  let args : TList α Γ := tlistAppend (α := α) (ss₁ := paramShapes) (ss₂ := inputShapes) params xs
  let zerosX : TList α inputShapes := TList.zero (α := α) (ss := inputShapes)
  let dargs : TList α Γ := tlistAppend (α := α) (ss₁ := paramShapes) (ss₂ := inputShapes) vparams
    zerosX
  let dl : Tensor α Shape.scalar := _root_.Runtime.Autograd.Torch.TypedScalarGraph.jvp (α := α) (Γ :=
    Γ) c args dargs
  match dl with
  | .scalar a => pure a

/-- Directional derivative of scalar loss along `vxs` (forward-mode JVP). -/
def jvpLossInputs {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes inputShapes : List Shape}
    (loss :
      ∀ {β : Type}, [Context β] → [DecidableEq Shape] →
        TorchLean.Program β (paramShapes ++ inputShapes) Shape.scalar)
    (params : TList α paramShapes)
    (xs : TList α inputShapes)
    (vxs : TList α inputShapes) :
    IO α := do
  let c ← lowerScalarToTypedGraph (α := α)
    (paramShapes := paramShapes) (inputShapes := inputShapes) loss
  let Γ : List Shape := paramShapes ++ inputShapes
  let args : TList α Γ := tlistAppend (α := α) (ss₁ := paramShapes) (ss₂ := inputShapes) params xs
  let zerosP : TList α paramShapes := TList.zero (α := α) (ss := paramShapes)
  let dargs : TList α Γ := tlistAppend (α := α) (ss₁ := paramShapes) (ss₂ := inputShapes) zerosP vxs
  let dl : Tensor α Shape.scalar := _root_.Runtime.Autograd.Torch.TypedScalarGraph.jvp (α := α) (Γ :=
    Γ) c args dargs
  match dl with
  | .scalar a => pure a

/--
Hessian-vector product (HVP) for a scalar loss w.r.t. *parameters*.

This computes `d/dε (∇_params loss(params + ε*vparams)) |_{ε=0}` and returns a `TList` aligned
with `paramShapes`.

Implementation: run reverse-mode AD over dual scalars (`Dual`), with parameter tangents set to
`vparams` and input tangents set to `0`. The tangent part of the resulting gradients is the HVP.
-/
def hvpParams {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes inputShapes : List Shape}
    (loss :
      ∀ {β : Type}, [Context β] → [DecidableEq Shape] →
        TorchLean.Program β (paramShapes ++ inputShapes) Shape.scalar)
    (params : TList α paramShapes)
    (xs : TList α inputShapes)
    (vparams : TList α paramShapes) :
    IO (TList α paramShapes) := do
  let αD := Dual α

  let paramsD : TList αD paramShapes :=
    DualTensor.withTangentsTList (α := α) (ss := paramShapes) params vparams
  let xsD : TList αD inputShapes :=
    DualTensor.ofPrimalTList (α := α) (ss := inputShapes) xs

  let Γ : List Shape := paramShapes ++ inputShapes
  let argsD : TList αD Γ :=
    tlistAppend (α := αD) (ss₁ := paramShapes) (ss₂ := inputShapes) paramsD xsD

  let build : Runtime.Autograd.TypedGraph.GraphM.M αD Γ (Runtime.Autograd.TypedGraph.GraphM.Var
    Shape.scalar) := do
    let vs ← Runtime.Autograd.TypedGraph.GraphM.args (α := αD) (Γ := Γ)
    CurriedRef.applyVarList (Γ := Γ)
      (β := Runtime.Autograd.TypedGraph.GraphM.M αD Γ (Runtime.Autograd.TypedGraph.GraphM.Var
        Shape.scalar))
      (loss (β := αD) (m := Runtime.Autograd.TypedGraph.GraphM.M αD Γ)) vs

  let graph ← okOrThrow (_root_.Runtime.Autograd.Torch.lowerScalarToTypedGraph (α := αD) (Γ := Γ) build)
  let ssFull : List Shape := graph.nodeShapes
  let fullGraph : Proofs.Autograd.Algebra.GraphData αD Unit Γ ssFull :=
    graph.data

  let (tape, _ctx) := Runtime.Autograd.TypedGraph.lowerToTape (α := αD) (Γ := Γ) (ss := ssFull) fullGraph
    argsD
  let gradsAny ← okOrThrow (Runtime.Autograd.TypedGraph.backwardDenseAllFrom (α := αD) (Γ := Γ)
    (ss := ssFull) tape graph.output (Tensor.scalar (1 : αD)))
  let gradsD : TList αD Γ := ← gradsPrefix (α := αD) (ss := Γ) gradsAny 0
  let gradsParamsD : TList αD paramShapes :=
    (tlistSplitAppend (α := αD) (ss₁ := paramShapes) (ss₂ := inputShapes) gradsD).1

  pure (DualTensor.tangentTList (α := α) (ss := paramShapes) gradsParamsD)

/--
Hessian-vector product (HVP) for a scalar loss w.r.t. *inputs*.

This computes `d/dε (∇_xs loss(xs + ε*vxs)) |_{ε=0}` and returns a `TList` aligned with
`inputShapes`.

Implementation: the same forward-over-reverse trick as `hvpParams`, but we attach tangents to
inputs instead of parameters.
-/
def hvpInputs {α : Type} [Context α] [DecidableEq Shape]
    {paramShapes inputShapes : List Shape}
    (loss :
      ∀ {β : Type}, [Context β] → [DecidableEq Shape] →
        TorchLean.Program β (paramShapes ++ inputShapes) Shape.scalar)
    (params : TList α paramShapes)
    (xs : TList α inputShapes)
    (vxs : TList α inputShapes) :
    IO (TList α inputShapes) := do
  let αD := Dual α

  let paramsD : TList αD paramShapes :=
    DualTensor.ofPrimalTList (α := α) (ss := paramShapes) params
  let xsD : TList αD inputShapes :=
    DualTensor.withTangentsTList (α := α) (ss := inputShapes) xs vxs

  let Γ : List Shape := paramShapes ++ inputShapes
  let argsD : TList αD Γ :=
    tlistAppend (α := αD) (ss₁ := paramShapes) (ss₂ := inputShapes) paramsD xsD

  let build : Runtime.Autograd.TypedGraph.GraphM.M αD Γ (Runtime.Autograd.TypedGraph.GraphM.Var
    Shape.scalar) := do
    let vs ← Runtime.Autograd.TypedGraph.GraphM.args (α := αD) (Γ := Γ)
    CurriedRef.applyVarList (Γ := Γ)
      (β := Runtime.Autograd.TypedGraph.GraphM.M αD Γ (Runtime.Autograd.TypedGraph.GraphM.Var
        Shape.scalar))
      (loss (β := αD) (m := Runtime.Autograd.TypedGraph.GraphM.M αD Γ)) vs

  let graph ← okOrThrow (_root_.Runtime.Autograd.Torch.lowerScalarToTypedGraph (α := αD) (Γ := Γ) build)
  let ssFull : List Shape := graph.nodeShapes
  let fullGraph : Proofs.Autograd.Algebra.GraphData αD Unit Γ ssFull :=
    graph.data

  let (tape, _ctx) := Runtime.Autograd.TypedGraph.lowerToTape (α := αD) (Γ := Γ) (ss := ssFull) fullGraph
    argsD
  let gradsAny ← okOrThrow (Runtime.Autograd.TypedGraph.backwardDenseAllFrom (α := αD) (Γ := Γ)
    (ss := ssFull) tape graph.output (Tensor.scalar (1 : αD)))
  let gradsD : TList αD Γ := ← gradsPrefix (α := αD) (ss := Γ) gradsAny 0
  let gradsInputsD : TList αD inputShapes :=
    (tlistSplitAppend (α := αD) (ss₁ := paramShapes) (ss₂ := inputShapes) gradsD).2

  pure (DualTensor.tangentTList (α := α) (ss := inputShapes) gradsInputsD)

/--
Full Hessian (as an array of columns) for a scalar function of a *single* tensor input.

Returns `Array (Tensor α σ)` where each element is `H * e_i` in the flattened coordinate basis of
  `σ`.
-/
def hessianInput {α : Type} [Context α] [DecidableEq Shape]
    {σ : Shape}
    (f :
      ∀ {β : Type}, [Context β] → [DecidableEq Shape] →
        TorchLean.Program β [σ] Shape.scalar)
    (x : Tensor α σ) :
    IO (Array (Tensor α σ)) := do
  let dirs : Array (Tensor α σ) := basisTensors (α := α) σ
  let cols : Array (Tensor α σ) ←
    dirs.mapM (fun dx => do
      let hvp : TList α [σ] ←
        hvpInputs (α := α)
          (paramShapes := ([] : List Shape)) (inputShapes := [σ])
          (fun {β} _ _ => f (β := β)) .nil (.cons x .nil) (.cons dx .nil)
      let .cons col .nil := hvp
      pure col)
  pure cols

end Autodiff

end TorchLean
end Autograd
end Runtime
