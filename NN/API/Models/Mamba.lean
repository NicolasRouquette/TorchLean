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

/-- Configuration for the trainable byte-level Mamba-style language model. -/
structure Config where
  vocab : Nat
  stateDim : Nat
deriving Repr

/-- One-hot token shape `leading ++ (seqLen × vocab)`. -/
abbrev inputShape (cfg : Config) (seqLen : Nat)
    (leading : List Nat := []) : List Nat :=
  leading ++ [seqLen, cfg.vocab]

/-- Output-logit shape `leading ++ (seqLen × vocab)`. -/
abbrev outputShape (cfg : Config) (seqLen : Nat)
    (leading : List Nat := []) : List Nat :=
  leading ++ [seqLen, cfg.vocab]

/--
Trainable Mamba-style causal language model over one-hot token inputs.

Architecture:

`mamba(seqLen, vocab, stateDim) → linear(stateDim → vocab)` applied at every time step.

The recurrent core is a gated diagonal state-space update implemented with autograd-covered
TorchLean ops. Passing `--device cuda` to a runner that instantiates this model trains the same
parameters on the CUDA backend.
-/
def textLM (cfg : Config) (seqLen : Nat) (leading : List Nat := []) :
    nn.Builder (nn.Sequential (inputShape cfg seqLen leading) (outputShape cfg seqLen leading)) := do
  let recurrent ← nn.mamba seqLen cfg.vocab cfg.stateDim (leading := leading)
  let headRaw ← linear cfg.stateDim cfg.vocab
    (leading := leading ++ [seqLen])
  let head : nn.Sequential
      (leading ++ [seqLen, cfg.stateDim])
      (leading ++ [seqLen, cfg.vocab]) := by
    simpa [List.append_assoc] using headRaw
  pure (recurrent >>> head)

namespace Reference

/-- Dimensions used by the full selective reference block. -/
structure SelectiveConfig extends Config where
  ssmStateDim : Nat
  convWidth : Nat
deriving Repr

namespace Internal

/-- Deterministic scalar initializer shared by the reference Mamba blocks. -/
def centeredHash (seed modulus : Nat) : Float :=
  (Float.ofNat (seed % modulus) - Float.ofNat (modulus / 2)) / Float.ofNat modulus

end Internal

/-- Compact diagonal Mamba-style block for spec-level reference evaluation. -/
def compact (cfg : Config) :
    _root_.Models.MambaBlockSpec Float cfg.vocab cfg.stateDim cfg.vocab :=
  { inProj := Tensor.generate [cfg.vocab, cfg.stateDim] fun coordinates =>
      Internal.centeredHash
        (coordinates.getD 0 0 * 17 + coordinates.getD 1 0 * 31 + 3) 47 / 4.0
    gateProj := Tensor.generate [cfg.vocab, cfg.stateDim] fun coordinates =>
      Internal.centeredHash
        (coordinates.getD 0 0 * 13 + coordinates.getD 1 0 * 19 + 7) 43 / 5.0
    outProj := Tensor.generate [cfg.stateDim, cfg.vocab] fun coordinates =>
      Internal.centeredHash
        (coordinates.getD 0 0 * 29 + coordinates.getD 1 0 * 11 + 5) 53 / 3.0
    ssm :=
      { A := Tensor.generate [cfg.stateDim] fun coordinates =>
          0.82 + Float.ofNat (coordinates.getD 0 0 % 5) * 0.025
        B := Tensor.generate [cfg.stateDim] fun coordinates =>
          0.12 + Float.ofNat (coordinates.getD 0 0 % 3) * 0.015
        C := Tensor.generate [cfg.stateDim] fun coordinates =>
          0.90 - Float.ofNat (coordinates.getD 0 0 % 4) * 0.03
        D := Tensor.generate [cfg.stateDim] fun coordinates =>
          0.08 + Float.ofNat (coordinates.getD 0 0 % 2) * 0.02 } }

/--
Full selective Mamba-style block with causal depthwise convolution and token-dependent scan
parameters.  This deterministic initializer is meant for reference evaluation rather than
checkpoint-quality training.
-/
def selective (cfg : SelectiveConfig) :
    _root_.Models.SelectiveMambaBlockSpec Float
      cfg.vocab cfg.stateDim cfg.ssmStateDim cfg.vocab cfg.convWidth :=
  { xProj := Tensor.generate [cfg.vocab, cfg.stateDim] fun coordinates =>
      Internal.centeredHash
        (coordinates.getD 0 0 * 17 + coordinates.getD 1 0 * 31 + 3) 47 * 2.0
    zProj := Tensor.generate [cfg.vocab, cfg.stateDim] fun coordinates =>
      Internal.centeredHash
        (coordinates.getD 0 0 * 13 + coordinates.getD 1 0 * 19 + 7) 43 * 2.0
    convKernel := Tensor.generate [cfg.convWidth, cfg.stateDim] fun coordinates =>
      0.4 + Internal.centeredHash
        (coordinates.getD 0 0 * 23 + coordinates.getD 1 0 * 7 + 11) 41 * 0.2
    convBias := Tensor.generate [cfg.stateDim] fun coordinates =>
      Internal.centeredHash (coordinates.getD 0 0 * 5 + 3) 37 * 0.2
    dtProj := Tensor.generate [cfg.stateDim, cfg.stateDim] fun coordinates =>
      Internal.centeredHash
        (coordinates.getD 0 0 * 19 + coordinates.getD 1 0 * 17 + 5) 47 * 0.2
    dtBias := Tensor.generate [cfg.stateDim] fun coordinates =>
      -1.5 + Float.ofNat (coordinates.getD 0 0 % 5) * 0.1
    A := Tensor.generate [cfg.stateDim, cfg.ssmStateDim] fun coordinates =>
      0.2 + Float.ofNat ((coordinates.getD 0 0 + coordinates.getD 1 0) % 7) * 0.03
    bProj := Tensor.generate [cfg.stateDim, cfg.ssmStateDim] fun coordinates =>
      Internal.centeredHash
        (coordinates.getD 0 0 * 11 + coordinates.getD 1 0 * 29 + 13) 53
    cProj := Tensor.generate [cfg.stateDim, cfg.ssmStateDim] fun coordinates =>
      Internal.centeredHash
        (coordinates.getD 0 0 * 31 + coordinates.getD 1 0 * 7 + 17) 59
    dSkip := Tensor.generate [cfg.stateDim] fun coordinates =>
      0.35 + Float.ofNat (coordinates.getD 0 0 % 3) * 0.03
    outProj := Tensor.generate [cfg.stateDim, cfg.vocab] fun coordinates =>
      Internal.centeredHash
        (coordinates.getD 0 0 * 29 + coordinates.getD 1 0 * 11 + 5) 53 / 3.0 }

end Reference

end Mamba
end models
end nn

end TorchLean
