/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/


module

public import NN.API.Neural.Layers
public import NN.API.Runtime
public import NN.API.Sample

/-!
# Reusable Neural-Network Blocks

Residual, convolutional, branching, and MLP compositions built from the checked layer API.
-/

@[expose] public section

namespace TorchLean
namespace nn
namespace Internal
namespace blocks


/-- Interpret an activation kind as a TorchLean layer. -/
def activation {s : Spec.Shape} : _root_.Activation.Kind → Sequential s s
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
structure MlpConfig where
  /-- Hidden layer widths (each entry creates a `Linear -> Activation` stage). -/
  hidden : List Nat := []
  /-- Activation used after each hidden linear layer. -/
  activation : _root_.Activation.Kind := .relu
  /-- Optional dropout probability after each activation. -/
  dropout? : Option Float := none

/-- Build the hidden stages of an MLP while assigning a distinct seed to each stateful layer. -/
def mlpStages (leading : List Nat) (act : _root_.Activation.Kind)
    (dropout? : Option Float) :
    (inDim : Nat) → (hidden : List Nat) → (outDim : Nat) → (seed : Nat) →
      Sequential (leading ++ [inDim]) (leading ++ [outDim])
  | inDim, [], outDim, seed =>
      linear inDim outDim seed (seed + 1) leading
  | inDim, h :: hs, outDim, seed =>
      let hiddenShape : Spec.Shape := leading ++ [h]
      let lin : Sequential (leading ++ [inDim]) hiddenShape :=
        linear inDim h seed (seed + 1) leading
      let seed' := seed + 2
      let actLayer : Sequential hiddenShape hiddenShape :=
        activation (s := hiddenShape) act
      let mid : Sequential hiddenShape hiddenShape × Nat :=
        match dropout? with
        | none => (actLayer, seed')
        | some p =>
            ((seq! actLayer, dropout (s := hiddenShape) p (seed := seed')), seed' + 1)
      let rest :=
        mlpStages leading act dropout? h hs outDim mid.snd
      seq! lin, mid.fst, rest

/-- Assemble an MLP from an explicit base seed for the public seeded builder. -/
def mlpWithSeed (inDim outDim seed : Nat) (cfg : MlpConfig := {})
    (leading : List Nat := []) :
    Sequential (leading ++ [inDim]) (leading ++ [outDim]) :=
  mlpStages leading cfg.activation cfg.dropout? inDim cfg.hidden outDim seed

/-- Convolution followed by an activation and optional dropout. -/
structure ConvAct (d : Nat) where
  conv : Conv d
  activation : _root_.Activation.Kind := .relu
  dropout? : Option Float := none
  seedDropout : Nat := 0

/-- Build a rank-polymorphic convolution/activation block. -/
def convAct (leading : List Nat := []) {d inChannels : Nat}
    (spatial : Tensor Nat [d]) (cfg : ConvAct d) [NeZero inChannels] :
    Sequential
      (leading ++ inChannels :: spatial.toList)
      (leading ++ cfg.conv.outChannels ::
        (Spec.convOutSpatial spatial cfg.conv.kernel cfg.conv.stride cfg.conv.padding).toList) := by
  let outShape := leading ++ cfg.conv.outChannels ::
    (Spec.convOutSpatial spatial cfg.conv.kernel cfg.conv.stride cfg.conv.padding).toList
  let actLayer : Sequential outShape outShape :=
    activation (s := Spec.Shape.ofList outShape) cfg.activation
  let core : Sequential (leading ++ inChannels :: spatial.toList) outShape := seq!
    conv leading spatial cfg.conv,
    actLayer
  exact match cfg.dropout? with
  | none => core
  | some p =>
      let dropoutLayer : Sequential outShape outShape :=
        dropout (s := Spec.Shape.ofList outShape) p (seed := cfg.seedDropout)
      seq! core, dropoutLayer

/-- Convolution/activation followed by max pooling. -/
structure ConvActPool (d : Nat) where
  block : ConvAct d
  pool : Pool d

/-- Build a rank-polymorphic convolution/activation/max-pooling block. -/
def convActPool (leading : List Nat := []) {d inChannels : Nat}
    (spatial : Tensor Nat [d]) (cfg : ConvActPool d) [NeZero inChannels] :
    Sequential
      (leading ++ inChannels :: spatial.toList)
      (leading ++ cfg.block.conv.outChannels ::
        (Spec.poolOutSpatialPad
          (Spec.convOutSpatial spatial cfg.block.conv.kernel cfg.block.conv.stride
            cfg.block.conv.padding)
          cfg.pool.kernel cfg.pool.stride cfg.pool.padding).toList) :=
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
          (Ref := fun sh => _root_.TorchLean.Runtime.ValueRef (m := m) (α := α) sh)
          (ss := ps ++ [s])
          (β := m (_root_.TorchLean.Runtime.ValueRef (m := m) (α := α) s))
          (fun args => do
            let (_psRefs, xRef) :=
              _root_.Runtime.Autograd.Torch.RefList.splitLast
                (Ref := fun sh => _root_.TorchLean.Runtime.ValueRef (m := m) (α := α) sh)
                (ss := ps) (τ := s) args
            let y ←
              _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
                (Ref := fun sh => _root_.TorchLean.Runtime.ValueRef (m := m) (α := α) sh)
                (ss := ps ++ [s])
                (β := m (_root_.TorchLean.Runtime.ValueRef (m := m) (α := α) s))
                (_root_.Runtime.Autograd.TorchLean.NN.Seq.forward inner (mode := mode) (α := α))
                args
            _root_.Runtime.Autograd.Torch.add (m := m) (α := α) (s := s) y xRef)
  }

/-- Lift `residualLayer` into a sequential model. -/
def residual {s : Spec.Shape} (inner : Sequential s s) : Sequential s s :=
  nn.of (residualLayer inner)

/-!
## Branching (skip connections)

`Seq` is linear, but skip connections require two computations to consume the same input. The
generic constructor below owns the shared state plumbing; public blocks choose how to combine the
two outputs.
-/

/--
Run two sequential branches on the same input and combine their outputs.

Parameters and persistent buffers are stored as `state(f) ++ state(g)`. The combining operation is
polymorphic in the runtime, so eager execution and graph lowering share the same branch structure.
-/
def combineBranchesLayer {σ τ₁ τ₂ υ : Spec.Shape} (kind : String)
    (f : Sequential σ τ₁) (g : Sequential σ τ₂)
    (combine : ∀ {α : Type}, [_root_.Context α] → [DecidableEq Spec.Shape] →
      ∀ {m : Type → Type}, [Monad m] →
        [_root_.Runtime.Autograd.Torch.Ops (m := m) (α := α)] →
        _root_.TorchLean.Runtime.ValueRef (m := m) (α := α) τ₁ →
        _root_.TorchLean.Runtime.ValueRef (m := m) (α := α) τ₂ →
        m (_root_.TorchLean.Runtime.ValueRef (m := m) (α := α) υ)) : Layer σ υ :=
  let psF := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes f
  let psG := _root_.Runtime.Autograd.TorchLean.NN.Seq.stateShapes g
  { kind := kind
    stateShapes := psF ++ psG
    initState :=
      _root_.TorchLean.TensorPack.append (α := Float) (ss₁ := psF) (ss₂ := psG)
        (_root_.Runtime.Autograd.TorchLean.NN.Seq.initState f) (_root_.Runtime.Autograd.TorchLean.NN.Seq.initState g)
    runtimeInit :=
      match _root_.Runtime.Autograd.TorchLean.NN.Seq.runtimeInit? f, _root_.Runtime.Autograd.TorchLean.NN.Seq.runtimeInit? g with
      | some fPlan, some gPlan => some (fPlan.append gPlan)
      | _, _ => none
    requiresGrad := _root_.Runtime.Autograd.TorchLean.NN.Seq.requiresGrad f ++ _root_.Runtime.Autograd.TorchLean.NN.Seq.requiresGrad
      g
    updateBuffers := some (fun mode {α} _ _ ps x => do
      let (psFv, psGv) := _root_.TorchLean.TensorPack.split (α := α) (ss₁ := psF) (ss₂ := psG) ps
      let psFv' ← _root_.Runtime.Autograd.TorchLean.NN.Seq.updateBuffers (α := α) (model := f) mode psFv x
      let psGv' ← _root_.Runtime.Autograd.TorchLean.NN.Seq.updateBuffers (α := α) (model := g) mode psGv x
      pure <| _root_.TorchLean.TensorPack.append (α := α) (ss₁ := psF) (ss₂ := psG) psFv' psGv'
    )
    forward := fun mode {α} _ _ =>
      fun {m} _ _ =>
        _root_.Runtime.Autograd.Torch.CurriedRef.curry
          (Ref := fun sh => _root_.TorchLean.Runtime.ValueRef (m := m) (α := α) sh)
          (ss := psF ++ psG ++ [σ])
          (β := m (_root_.TorchLean.Runtime.ValueRef (m := m) (α := α) υ))
          (fun args => do
            let (psAll, xRef) :=
              _root_.Runtime.Autograd.Torch.RefList.splitLast
                (Ref := fun sh => _root_.TorchLean.Runtime.ValueRef (m := m) (α := α) sh)
                (ss := psF ++ psG) (τ := σ) args
            let (psFrefs, psGrefs) :=
              _root_.Runtime.Autograd.Torch.RefList.split
                (Ref := fun sh => _root_.TorchLean.Runtime.ValueRef (m := m) (α := α) sh)
                (ss₁ := psF) (ss₂ := psG) psAll
            let yF ←
              _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
                (Ref := fun sh => _root_.TorchLean.Runtime.ValueRef (m := m) (α := α) sh)
                (ss := psF ++ [σ])
                (β := m (_root_.TorchLean.Runtime.ValueRef (m := m) (α := α) τ₁))
                (_root_.Runtime.Autograd.TorchLean.NN.Seq.forward f (mode := mode) (α := α))
                (_root_.Runtime.Autograd.Torch.RefList.append psFrefs (.cons xRef .nil))
            let yG ←
              _root_.Runtime.Autograd.Torch.CurriedRef.uncurry
                (Ref := fun sh => _root_.TorchLean.Runtime.ValueRef (m := m) (α := α) sh)
                (ss := psG ++ [σ])
                (β := m (_root_.TorchLean.Runtime.ValueRef (m := m) (α := α) τ₂))
                (_root_.Runtime.Autograd.TorchLean.NN.Seq.forward g (mode := mode) (α := α))
                (_root_.Runtime.Autograd.Torch.RefList.append psGrefs (.cons xRef .nil))
            combine yF yG)
  }

/-- Combine two sequential branches into a single layer that adds their outputs. -/
def addBranchesLayer {σ τ : Spec.Shape} (f g : Sequential σ τ) : Layer σ τ :=
  combineBranchesLayer "AddBranches" f g fun {α} _ _ {m} _ _ yF yG =>
    _root_.Runtime.Autograd.Torch.add (m := m) (α := α) (s := τ) yF yG

/--
Combine two models with the same input/output shapes by summing their outputs.

This is a typed residual-add block: `addBranches f g` represents the model
$x \mapsto f(x) + g(x)$,
and its parameter list is the concatenation of the two branches’ parameter lists.
-/
def addBranches {σ τ : Spec.Shape} (f g : Sequential σ τ) : Sequential σ τ :=
  nn.of (addBranchesLayer f g)

/--
Concatenate two branch outputs along their leading axis.

Both branches consume the same input. Their outputs must agree below the leading dimension, and
the result records the sum of their leading extents in its type. This is the general typed skip
connection needed by encoder-decoder models; arbitrary outer batch axes can be added with
`mapEach`.
-/
def concatBranchesLayer {σ s : Spec.Shape} {n m : Nat}
    (f : Sequential σ (s.prependDim n)) (g : Sequential σ (s.prependDim m)) :
    Layer σ (s.prependDim (n + m)) :=
  combineBranchesLayer "ConcatBranches" f g fun {α} _ _ {mRuntime} _ _ yF yG =>
    _root_.Runtime.Autograd.Torch.concatLeadingAxis
      (m := mRuntime) (α := α) (nDim := n) (mDim := m) (s := s) yF yG

/-- Concatenate two typed branches along their leading output axis. -/
def concatBranches {σ s : Spec.Shape} {n m : Nat}
    (f : Sequential σ (s.prependDim n)) (g : Sequential σ (s.prependDim m)) :
    Sequential σ (s.prependDim (n + m)) :=
  nn.of (concatBranchesLayer f g)

/-- Apply an activation after adding two branches with the same output shape. -/
def residualBlock {input output : Spec.Shape}
    (main skip : Sequential input output) (act : _root_.Activation.Kind := .relu) :
    Sequential input output :=
  seq! addBranches main skip, activation (s := output) act
