/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.API.Neural.VisionLayers
public import NN.API.TensorPack

/-!
Reusable neural-network blocks.

This module defines public block constructors such as residual, convolutional, and MLP-style
compositions built from the public layer-building API.
-/

@[expose] public section

namespace TorchLean
namespace nn
namespace Internal
namespace blocks


/--
Small set of activation choices for block builders.

PyTorch module analogues:
- `relu`    <-> `torch.nn.ReLU`
- `gelu`    <-> `torch.nn.GELU`
- `silu`    <-> `torch.nn.SiLU`
- `tanh`    <-> `torch.nn.Tanh`
- `sigmoid` <-> `torch.nn.Sigmoid`
-/
inductive Activation where
  | relu
  | gelu
  | silu
  | tanh
  | sigmoid
deriving Repr, DecidableEq

/-- Interpret an `Activation` as a TorchLean layer. -/
def activation {s : Spec.Shape} : Activation → Sequential s s
  | .relu => relu (s := s)
  | .gelu => gelu (s := s)
  | .silu => silu (s := s)
  | .tanh => tanh (s := s)
  | .sigmoid => sigmoid (s := s)

/--
MLP (multi-layer perceptron) configuration.

This builder produces a sequential stack of linear layers with activations and optional dropout.

PyTorch analogue: a hand-written `nn.Sequential(Linear(...), ReLU(), ..., Linear(...))`.
-/
structure MLP where
  /-- Hidden layer widths (each entry creates a `Linear -> Activation` stage). -/
  hidden : List Nat := []
  /-- Activation used after each hidden linear layer. -/
  activation : Activation := .relu
  /-- Optional dropout probability after each activation. -/
  dropout? : Option Float := none
  /-- Base seed used to deterministically initialize all linear layers (and dropout if present). -/
  seedBase : Nat := 0

/--
Internal recursion for `mlp`.

This builds the sequential stack stage-by-stage, threading a seed so each linear (and optional
dropout) layer gets a deterministic initialization key.
-/
def mlpGo (act : Activation) (dropout? : Option Float) :
    (inDim : Nat) → (hidden : List Nat) → (outDim : Nat) → (seed : Nat) →
      Sequential (.dim inDim .scalar) (.dim outDim .scalar)
  | inDim, [], outDim, seed =>
      linear inDim outDim seed (seed + 1)
  | inDim, h :: hs, outDim, seed =>
      let lin : Sequential (.dim inDim .scalar) (.dim h .scalar) :=
        linear inDim h seed (seed + 1)
      let seed' := seed + 2
      let actLayer : Sequential (.dim h .scalar) (.dim h .scalar) :=
        activation (s := .dim h .scalar) act
      let mid : Sequential (.dim h .scalar) (.dim h .scalar) × Nat :=
        match dropout? with
        | none => (actLayer, seed')
        | some p =>
            ((seq! actLayer, dropout (s := .dim h .scalar) p (seed := seed')), seed' + 1)
      let rest :=
        mlpGo act dropout? h hs outDim mid.snd
      seq! lin, mid.fst, rest

/--
Build an MLP as a sequential stack of linear layers and activations.

This is a small PyTorch-shaped constructor: a typical call looks like:
`TorchLean.nn.blocks.mlp 784 10 { hidden := [128, 128], activation := .relu }`.
-/
def mlp (inDim outDim : Nat) (cfg : MLP := {}) :
    Sequential (.dim inDim .scalar) (.dim outDim .scalar) :=
  mlpGo cfg.activation cfg.dropout? inDim cfg.hidden outDim cfg.seedBase

/-- Convolution followed by an activation and optional dropout. -/
structure ConvAct (d : Nat) where
  conv : Conv d
  activation : Activation := .relu
  dropout? : Option Float := none
  seedDropout : Nat := 0

/-- Build a rank-polymorphic convolution/activation block. -/
def convAct (leading : Spec.Shape := .scalar) {d inChannels : Nat}
    (spatial : Vector Nat d) (cfg : ConvAct d) [NeZero inChannels] :
    Sequential
      (leading.concat (Spec.Shape.ofList (inChannels :: spatial.toList)))
      (leading.concat (Spec.Shape.ofList
        (cfg.conv.outChannels ::
          (Spec.convOutSpatial spatial cfg.conv.kernel cfg.conv.stride cfg.conv.padding).toList))) :=
  let outShape := leading.concat (Spec.Shape.ofList
    (cfg.conv.outChannels ::
      (Spec.convOutSpatial spatial cfg.conv.kernel cfg.conv.stride cfg.conv.padding).toList))
  let core := seq!
    conv leading spatial cfg.conv,
    activation (s := outShape) cfg.activation
  match cfg.dropout? with
  | none => core
  | some p => seq! core, dropout (s := outShape) p (seed := cfg.seedDropout)

/-- Convolution/activation followed by max pooling. -/
structure ConvActPool (d : Nat) where
  block : ConvAct d
  pool : Pool d

/-- Build a rank-polymorphic convolution/activation/max-pooling block. -/
def convActPool (leading : Spec.Shape := .scalar) {d inChannels : Nat}
    (spatial : Vector Nat d) (cfg : ConvActPool d) [NeZero inChannels] :
    Sequential
      (leading.concat (Spec.Shape.ofList (inChannels :: spatial.toList)))
      (leading.concat (Spec.Shape.ofList
        (cfg.block.conv.outChannels ::
          (Spec.poolOutSpatialPad
            (Spec.convOutSpatial spatial cfg.block.conv.kernel cfg.block.conv.stride
              cfg.block.conv.padding)
            cfg.pool.kernel cfg.pool.stride cfg.pool.padding).toList))) :=
  let afterConv := Spec.convOutSpatial spatial cfg.block.conv.kernel cfg.block.conv.stride
    cfg.block.conv.padding
  seq!
    convAct leading spatial cfg.block,
    maxPool leading afterConv cfg.pool

/--
Residual/skip-connection layer as a single `Layer`.

Given `inner : Seq s s`, this builds a layer that computes
$x \mapsto \operatorname{inner}(x) + x$.

PyTorch analogue: $x + f(x)$ blocks used throughout ResNets and Transformers.
-/
def residualLayer {s : Spec.Shape} (inner : Sequential s s) : Layer s s :=
  let ps := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes inner
  { kind := "Residual"
    stateShapes := ps
    initState := _root_.Runtime.Autograd.TorchLean.NN.Seq.initState inner
    runtimeInit := _root_.Runtime.Autograd.TorchLean.NN.Seq.runtimeInit? inner
    requiresGrad := _root_.Runtime.Autograd.TorchLean.NN.Seq.requiresGrad inner
    updateBuffers := some (fun mode {α} _ _ ps x =>
      _root_.Runtime.Autograd.TorchLean.NN.Seq.updateBuffers (α := α) (model := inner) mode ps x)
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        _root_.Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := fun sh => _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) sh)
          (ss := ps ++ [s])
          (β := m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) s))
          (fun args => do
            let (_psRefs, xRef) :=
              _root_.Runtime.Autograd.Torch.RefList.splitLast
                (Ref := fun sh => _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) sh)
                (ss := ps) (τ := s) args
            let y ←
              _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
                (Ref := fun sh => _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) sh)
                (ss := ps ++ [s])
                (β := m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) s))
                (_root_.Runtime.Autograd.TorchLean.NN.Seq.forward inner (mode := mode) (α := α))
                args
            _root_.Runtime.Autograd.Torch.add (m := m) (α := α) (s := s) y xRef)
  }

/-- Lift `residualLayer` into a sequential model. -/
def residual {s : Spec.Shape} (inner : Sequential s s) : Sequential s s :=
  nn.of (residualLayer inner)

/-!
## Branching (skip connections)

`Seq` is linear, but we sometimes want a PyTorch-like $x \mapsto f(x) + g(x)$ block.

We expose this as a single `Layer` whose parameter list is `params(f) ++ params(g)` and whose
forward pass runs both programs and adds their outputs.
-/

/--
Combine two sequential branches into a single layer that adds their outputs.

The resulting layer runs both `f` and `g` on the same input $x$ and returns $f(x) + g(x)$.
Parameters are concatenated as `params(f) ++ params(g)`.
-/
def addBranchesLayer {σ τ : Spec.Shape} (f g : Sequential σ τ) : Layer σ τ :=
  let psF := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes f
  let psG := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes g
  { kind := "AddBranches"
    stateShapes := psF ++ psG
    initState :=
      _root_.TorchLean.tensorpack.append (α := Float) (ss₁ := psF) (ss₂ := psG)
        (_root_.Runtime.Autograd.TorchLean.NN.Seq.initState f) (_root_.Runtime.Autograd.TorchLean.NN.Seq.initState g)
    runtimeInit :=
      match _root_.Runtime.Autograd.TorchLean.NN.Seq.runtimeInit? f, _root_.Runtime.Autograd.TorchLean.NN.Seq.runtimeInit? g with
      | some fPlan, some gPlan => some (fPlan.append gPlan)
      | _, _ => none
    requiresGrad := _root_.Runtime.Autograd.TorchLean.NN.Seq.requiresGrad f ++ _root_.Runtime.Autograd.TorchLean.NN.Seq.requiresGrad
      g
    updateBuffers := some (fun mode {α} _ _ ps x => do
      let (psFv, psGv) := _root_.TorchLean.tensorpack.split (α := α) (ss₁ := psF) (ss₂ := psG) ps
      let psFv' ← _root_.Runtime.Autograd.TorchLean.NN.Seq.updateBuffers (α := α) (model := f) mode psFv x
      let psGv' ← _root_.Runtime.Autograd.TorchLean.NN.Seq.updateBuffers (α := α) (model := g) mode psGv x
      pure <| _root_.TorchLean.tensorpack.append (α := α) (ss₁ := psF) (ss₂ := psG) psFv' psGv'
    )
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        _root_.Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := fun sh => _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) sh)
          (ss := psF ++ psG ++ [σ])
          (β := m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) τ))
          (fun args => do
            let (psAll, xRef) :=
              _root_.Runtime.Autograd.Torch.RefList.splitLast
                (Ref := fun sh => _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) sh)
                (ss := psF ++ psG) (τ := σ) args
            let (psFrefs, psGrefs) :=
              _root_.Runtime.Autograd.Torch.RefList.split
                (Ref := fun sh => _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) sh)
                (ss₁ := psF) (ss₂ := psG) psAll
            let yF ←
              _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
                (Ref := fun sh => _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) sh)
                (ss := psF ++ [σ])
                (β := m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) τ))
                (_root_.Runtime.Autograd.TorchLean.NN.Seq.forward f (mode := mode) (α := α))
                (_root_.Runtime.Autograd.Torch.RefList.append psFrefs (.cons xRef .nil))
            let yG ←
              _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
                (Ref := fun sh => _root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) sh)
                (ss := psG ++ [σ])
                (β := m (_root_.Runtime.Autograd.TorchLean.RefTy (m := m) (α := α) τ))
                (_root_.Runtime.Autograd.TorchLean.NN.Seq.forward g (mode := mode) (α := α))
                (_root_.Runtime.Autograd.Torch.RefList.append psGrefs (.cons xRef .nil))
            _root_.Runtime.Autograd.Torch.add (m := m) (α := α) (s := τ) yF yG)
  }

/--
Combine two models with the same input/output shapes by summing their outputs.

This is a typed residual-add block: `addBranches f g` represents the model
$x \mapsto f(x) + g(x)$,
and its parameter list is the concatenation of the two branches’ parameter lists.
-/
def addBranches {σ τ : Spec.Shape} (f g : Sequential σ τ) : Sequential σ τ :=
  nn.of (addBranchesLayer f g)

/-- Apply an activation after adding two branches with the same output shape. -/
def residualBlock {input output : Spec.Shape}
    (main skip : Sequential input output) (act : Activation := .relu) :
    Sequential input output :=
  seq! addBranches main skip, activation (s := output) act
