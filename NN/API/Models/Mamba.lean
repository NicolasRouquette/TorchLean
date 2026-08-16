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

/-- Sequence-major one-hot token matrix shape. -/
abbrev inputShape (cfg : Config) (seqLen : Nat) : Spec.Shape :=
  .dim seqLen (.dim cfg.vocab .scalar)

/-- Output logits shape for byte-level causal language modeling. -/
abbrev outputShape (cfg : Config) (seqLen : Nat) : Spec.Shape :=
  .dim seqLen (.dim cfg.vocab .scalar)

/--
Trainable Mamba-style causal language model over one-hot token inputs.

Architecture:

`mamba(seqLen, vocab, stateDim) → linear(stateDim → vocab)` applied at every time step.

The recurrent core is a gated diagonal state-space update implemented with autograd-covered
TorchLean ops. Passing `--device cuda` to a runner that instantiates this model trains the same
parameters on the CUDA backend.
-/
def textLM (cfg : Config) (seqLen : Nat) :
    nn.Builder (nn.Sequential (inputShape cfg seqLen) (outputShape cfg seqLen)) :=
  nn.Sequential![
    nn.mamba seqLen cfg.vocab cfg.stateDim,
    linear cfg.stateDim cfg.vocab (pfx := .dim seqLen .scalar)
  ]

namespace Reference
namespace Internal

/-- Small deterministic initializer for spec-level reference blocks. -/
def centeredHash (seed modulus : Nat) : Float :=
  (Float.ofNat (seed % modulus) - Float.ofNat (modulus / 2)) / Float.ofNat modulus

/-- Build a vector tensor from an index function. -/
def vector {n : Nat} (f : Fin n → Float) : _root_.Spec.Tensor Float (shape![n]) :=
  _root_.Spec.Tensor.dim (fun i => _root_.Spec.Tensor.scalar (f i))

/-- Build a matrix tensor from an index function. -/
def matrix {m n : Nat} (f : Fin m → Fin n → Float) :
    _root_.Spec.Tensor Float (shape![m, n]) :=
  _root_.Spec.Tensor.dim (fun i =>
    _root_.Spec.Tensor.dim (fun j => _root_.Spec.Tensor.scalar (f i j)))

end Internal

/-- Compact diagonal Mamba-style block for spec-level reference evaluation. -/
def compact (cfg : Config) :
    _root_.Models.MambaBlockSpec Float cfg.vocab cfg.stateDim cfg.vocab :=
  { inProj := Internal.matrix
      (fun i j => Internal.centeredHash (i.val * 17 + j.val * 31 + 3) 47 / 4.0)
    gateProj := Internal.matrix
      (fun i j => Internal.centeredHash (i.val * 13 + j.val * 19 + 7) 43 / 5.0)
    outProj := Internal.matrix
      (fun i j => Internal.centeredHash (i.val * 29 + j.val * 11 + 5) 53 / 3.0)
    ssm :=
      { A := Internal.vector (fun i => 0.82 + Float.ofNat (i.val % 5) * 0.025)
        B := Internal.vector (fun i => 0.12 + Float.ofNat (i.val % 3) * 0.015)
        C := Internal.vector (fun i => 0.90 - Float.ofNat (i.val % 4) * 0.03)
        D := Internal.vector (fun i => 0.08 + Float.ofNat (i.val % 2) * 0.02) } }

/--
Full selective Mamba-style block with causal depthwise convolution and token-dependent scan
parameters.  This deterministic initializer is meant for reference evaluation rather than
checkpoint-quality training.
-/
def selective (cfg : Config) :
    _root_.Models.SelectiveMambaBlockSpec Float
      cfg.vocab cfg.stateDim cfg.ssmStateDim cfg.vocab cfg.convWidth :=
  { xProj := Internal.matrix
      (fun i j => Internal.centeredHash (i.val * 17 + j.val * 31 + 3) 47 * 2.0)
    zProj := Internal.matrix
      (fun i j => Internal.centeredHash (i.val * 13 + j.val * 19 + 7) 43 * 2.0)
    convKernel := Internal.matrix
      (fun tap i => 0.4 + Internal.centeredHash (tap.val * 23 + i.val * 7 + 11) 41 * 0.2)
    convBias := Internal.vector
      (fun i => Internal.centeredHash (i.val * 5 + 3) 37 * 0.2)
    dtProj := Internal.matrix
      (fun i j => Internal.centeredHash (i.val * 19 + j.val * 17 + 5) 47 * 0.2)
    dtBias := Internal.vector
      (fun i => -1.5 + Float.ofNat (i.val % 5) * 0.1)
    A := Internal.matrix
      (fun i n => 0.2 + Float.ofNat ((i.val + n.val) % 7) * 0.03)
    bProj := Internal.matrix
      (fun i n => Internal.centeredHash (i.val * 11 + n.val * 29 + 13) 53)
    cProj := Internal.matrix
      (fun i n => Internal.centeredHash (i.val * 31 + n.val * 7 + 17) 59)
    dSkip := Internal.vector
      (fun i => 0.35 + Float.ofNat (i.val % 3) * 0.03)
    outProj := Internal.matrix
      (fun i j => Internal.centeredHash (i.val * 29 + j.val * 11 + 5) 53 / 3.0) }

end Reference

/-- Deterministic starting offsets for fixed-width language-model training windows. -/
def trainingOffsets (tokenCount seqLen windows : Nat) : List Nat :=
  let usable := if tokenCount > seqLen + 1 then tokenCount - seqLen - 1 else 1
  let stride := Nat.max 1 (usable / Nat.max 1 windows)
  (List.range windows).map (fun i => (i * stride) % usable)

end Mamba
end models
end nn

end TorchLean
