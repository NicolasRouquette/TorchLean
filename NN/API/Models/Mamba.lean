/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Seeded
public import NN.Spec.Models.Mamba

/-!
# Mamba Models

Reusable configuration, model constructors, and text helpers for Mamba-style sequence models.

The trainable model path uses TorchLean autograd layers and therefore runs on the CPU and CUDA
backends.  The spec-backed deterministic helpers below are kept as small mathematical reference
utilities; runnable training examples use the autograd constructor.
-/

@[expose] public section

namespace TorchLean


open Spec Tensor

namespace nn
namespace models
namespace Mamba

/-- Configuration for byte-level Mamba-style language models. -/
structure Config where
  vocab : Nat
  stateDim : Nat
  ssmStateDim : Nat
  convWidth : Nat
deriving Repr

/-- One-hot token shape `leading ++ (seqLen × vocab)`. -/
abbrev inputShape (cfg : Config) (seqLen : Nat)
    (leading : Spec.Shape := .scalar) : Spec.Shape :=
  leading.concat (.dim seqLen (.dim cfg.vocab .scalar))

/-- Output-logit shape `leading ++ (seqLen × vocab)`. -/
abbrev outputShape (cfg : Config) (seqLen : Nat)
    (leading : Spec.Shape := .scalar) : Spec.Shape :=
  leading.concat (.dim seqLen (.dim cfg.vocab .scalar))

/--
Trainable Mamba-style causal language model over one-hot token inputs.

Architecture:

`mamba(seqLen, vocab, stateDim) → linear(stateDim → vocab)` applied at every time step.

The recurrent core is a gated diagonal state-space update implemented with autograd-covered
TorchLean ops. Passing `--device cuda` to a runner that instantiates this model trains the same
parameters on the CUDA backend.
-/
def textLM (cfg : Config) (seqLen : Nat) (leading : Spec.Shape := .scalar) :
    nn.Builder (nn.Sequential (inputShape cfg seqLen leading) (outputShape cfg seqLen leading)) := do
  let recurrent ← nn.mamba seqLen cfg.vocab cfg.stateDim (leading := leading)
  let headRaw ← linear cfg.stateDim cfg.vocab
    (leading := leading.concat (.dim seqLen .scalar))
  let head : nn.Sequential
      (leading.concat (.dim seqLen (.dim cfg.stateDim .scalar)))
      (leading.concat (.dim seqLen (.dim cfg.vocab .scalar))) := by
    simpa only [Spec.Shape.concat_appendDim, Spec.Shape.appendDim] using headRaw
  pure (recurrent >>> head)

namespace Reference

/-- Small deterministic initializer for spec-level reference blocks. -/
def centeredHash (seed modulus : Nat) : Float :=
  (Float.ofNat (seed % modulus) - Float.ofNat (modulus / 2)) / Float.ofNat modulus

/-- Compact diagonal Mamba-style block for spec-level reference evaluation. -/
def compact (cfg : Config) :
    _root_.Models.MambaBlockSpec Float cfg.vocab cfg.stateDim cfg.vocab :=
  { inProj := _root_.Spec.Tensor.matrix
      (fun i j => centeredHash (i.val * 17 + j.val * 31 + 3) 47 / 4.0)
    gateProj := _root_.Spec.Tensor.matrix
      (fun i j => centeredHash (i.val * 13 + j.val * 19 + 7) 43 / 5.0)
    outProj := _root_.Spec.Tensor.matrix
      (fun i j => centeredHash (i.val * 29 + j.val * 11 + 5) 53 / 3.0)
    ssm :=
      { A := _root_.Spec.Tensor.vector (fun i => 0.82 + Float.ofNat (i.val % 5) * 0.025)
        B := _root_.Spec.Tensor.vector (fun i => 0.12 + Float.ofNat (i.val % 3) * 0.015)
        C := _root_.Spec.Tensor.vector (fun i => 0.90 - Float.ofNat (i.val % 4) * 0.03)
        D := _root_.Spec.Tensor.vector (fun i => 0.08 + Float.ofNat (i.val % 2) * 0.02) } }

/--
Full selective Mamba-style block with causal depthwise convolution and token-dependent scan
parameters.  This deterministic initializer is meant for reference evaluation rather than
checkpoint-quality training.
-/
def selective (cfg : Config) :
    _root_.Models.SelectiveMambaBlockSpec Float
      cfg.vocab cfg.stateDim cfg.ssmStateDim cfg.vocab cfg.convWidth :=
  { xProj := _root_.Spec.Tensor.matrix
      (fun i j => centeredHash (i.val * 17 + j.val * 31 + 3) 47 * 2.0)
    zProj := _root_.Spec.Tensor.matrix
      (fun i j => centeredHash (i.val * 13 + j.val * 19 + 7) 43 * 2.0)
    convKernel := _root_.Spec.Tensor.matrix
      (fun tap i => 0.4 + centeredHash (tap.val * 23 + i.val * 7 + 11) 41 * 0.2)
    convBias := _root_.Spec.Tensor.vector
      (fun i => centeredHash (i.val * 5 + 3) 37 * 0.2)
    dtProj := _root_.Spec.Tensor.matrix
      (fun i j => centeredHash (i.val * 19 + j.val * 17 + 5) 47 * 0.2)
    dtBias := _root_.Spec.Tensor.vector
      (fun i => -1.5 + Float.ofNat (i.val % 5) * 0.1)
    A := _root_.Spec.Tensor.matrix
      (fun i n => 0.2 + Float.ofNat ((i.val + n.val) % 7) * 0.03)
    bProj := _root_.Spec.Tensor.matrix
      (fun i n => centeredHash (i.val * 11 + n.val * 29 + 13) 53)
    cProj := _root_.Spec.Tensor.matrix
      (fun i n => centeredHash (i.val * 31 + n.val * 7 + 17) 59)
    dSkip := _root_.Spec.Tensor.vector
      (fun i => 0.35 + Float.ofNat (i.val % 3) * 0.03)
    outProj := _root_.Spec.Tensor.matrix
      (fun i j => centeredHash (i.val * 29 + j.val * 11 + 5) 53 / 3.0) }

end Reference

end Mamba
end models
end nn

end TorchLean
