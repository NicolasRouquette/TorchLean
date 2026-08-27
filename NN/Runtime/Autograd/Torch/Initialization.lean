/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tensor
public import NN.Runtime.Autograd.TorchLean.Random

/-!
# Deterministic parameter initialization

Pure, reproducible initializers for `Spec.Tensor Float`. These definitions are used when building
model parameters before they enter a runtime session. Large runtime backends may provide more
specialized allocation paths, but they should implement the same initialization scheme.

The formulas follow the corresponding PyTorch initializers:

* `Scheme.xavierUniform` uses the Glorot bound `sqrt (6 / (fanIn + fanOut))`;
* `Scheme.kaimingUniform` uses the ReLU-oriented He bound `sqrt (6 / fanIn)`.

References:

* Glorot and Bengio, *Understanding the difficulty of training deep feedforward neural networks*,
  AISTATS 2010.
* He et al., *Delving Deep into Rectifiers*, ICCV 2015.
* PyTorch initialization reference: https://pytorch.org/docs/stable/nn.init.html
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Torch
namespace Init

open Spec
open Tensor

/-- A deterministic scheme for initializing a tensor of `Float` values. -/
inductive Scheme where
  | zeros
  | ones
  | uniform (lo hi : Float)
  | normal (mean std : Float)
  | xavierUniform (fanIn fanOut : Nat)
  | kaimingUniform (fanIn : Nat)
  deriving Repr

/-- Return sample `idx` from `sch`, using the counter-based stream determined by `seed`. -/
def sampleAt (sch : Scheme) (seed idx : Nat) : Float :=
  let key := _root_.Runtime.Autograd.TorchLean.Random.keyOf seed 0
  let denominator : Nat := (2 : Nat) ^ 32
  let unit := _root_.Runtime.Autograd.TorchLean.Random.sampleUnit (α := Float)
    (_root_.Runtime.Autograd.TorchLean.Random.sampleNat key idx denominator) denominator
  match sch with
  | .zeros => 0.0
  | .ones => 1.0
  | .uniform lo hi =>
      lo + unit * (hi - lo)
  | .normal mean std =>
      mean + std * _root_.Runtime.Autograd.TorchLean.Random.normalScalar key idx
  | .xavierUniform fanIn fanOut =>
      let denominator := (Float.ofNat fanIn) + (Float.ofNat fanOut)
      let limit := Float.sqrt (6.0 / denominator)
      (-limit) + unit * (2.0 * limit)
  | .kaimingUniform fanIn =>
      let limit := Float.sqrt (6.0 / (Float.ofNat fanIn))
      (-limit) + unit * (2.0 * limit)

/-- Initialize a tensor by assigning sample `i` to the scalar at flat index `i`. -/
def tensor (sch : Scheme) (seed : Nat := 0) : {s : Shape} → Tensor Float s
  | .scalar =>
      Tensor.scalar (sampleAt sch seed 0)
  | .dim _ s =>
      let blockSize := Spec.Shape.size s
      Tensor.dim fun i =>
        let offset := i.val * blockSize
        let rec build : {t : Shape} → Nat → Tensor Float t
          | .scalar, j => Tensor.scalar (sampleAt sch seed (offset + j))
          | .dim _ t, j =>
              let childSize := Spec.Shape.size t
              Tensor.dim fun k => build (t := t) (j + k.val * childSize)
        build (t := s) 0

/-- Initialize a matrix with the Xavier/Glorot uniform distribution and gain `1`. -/
def xavierW (outDim inDim : Nat) (seed : Nat := 0) :
    Tensor Float [outDim, inDim] :=
  tensor (s := .dim outDim (.dim inDim .scalar)) (.xavierUniform inDim outDim) seed

/-- Initialize a matrix with the Kaiming/He uniform distribution for ReLU networks. -/
def kaimingW (outDim inDim : Nat) (seed : Nat := 0) :
    Tensor Float [outDim, inDim] :=
  tensor (s := .dim outDim (.dim inDim .scalar)) (.kaimingUniform inDim) seed

end Init
end Torch
end Autograd
end Runtime
