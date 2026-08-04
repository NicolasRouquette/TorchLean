/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.Spec.Models.Mlp
public import NN.Tests.Runtime.Floats.Utils
public import Std

/-!
# TorchLeanSpecMlpEquivCheck

Runtime check: TorchLean MLP forward agrees with Spec MLP forward.

Goal: make the "Spec vs TorchLean" relationship concrete with an executable check:
given the *same* initialized parameters, both front-ends produce the same output tensor.

This is not a performance test; it is a regression guard for:
- parameter ordering conventions (W,b pairs),
- shape conventions (Vec/Mat layout),
- and the meaning of `Linear → ReLU → Linear`.
-/

@[expose] public section


open Spec
open Tensor
open Tests.Floats.Utils

namespace Tests
namespace Floats
namespace TorchLeanSpecMLPEquivCheck

def run : IO Unit := do
  IO.println "torchlean_spec_mlp_equiv_check: begin"

  let inDim : Nat := 2
  let hidDim : Nat := 3
  let outDim : Nat := 1
  let xShape : Shape := .dim inDim .scalar
  let yShape : Shape := .dim outDim .scalar

  -- TorchLean MLP (deterministic init via explicit seeds).
  let model :=
    NN.GraphSpec.Models.TorchLean.mlp
      (inDim := inDim) (hidDim := hidDim) (outDim := outDim)
      (seedW1 := 0) (seedB1 := 1) (seedW2 := 2) (seedB2 := 3)

  -- One input vector.
  let x : Tensor Float xShape :=
    Tensor.dim (fun i => Tensor.scalar ([0.5, 0.8][i.val]!))

  -- Extract TorchLean parameters and reinterpret them as Spec `LinearSpec`s.
  let ps := Runtime.Autograd.TorchLean.NN.Seq.initParams (m := model)
  let (w1, b1, w2, b2) :=
    match ps with
    | .cons w1 (.cons b1 (.cons w2 (.cons b2 .nil))) => (w1, b1, w2, b2)

  let l1 : Spec.LinearSpec Float inDim hidDim := { weights := w1, bias := b1 }
  let l2 : Spec.LinearSpec Float hidDim outDim := { weights := w2, bias := b2 }

  -- Spec forward reference.
  let ySpec : Tensor Float yShape := Examples.mlpForward (α := Float) l1 l2 x

  -- TorchLean forward reference (compiled-out evaluation of the TorchLean forwardProgram).
  let compiled ← Runtime.Autograd.TorchLean.Autodiff.compileGraph (α := Float)
    (paramShapes := Runtime.Autograd.TorchLean.NN.Seq.paramShapes model)
    (inputShapes := [xShape]) (τ := yShape)
    (fun {β} _ _ => Runtime.Autograd.TorchLean.NN.Seq.forwardProgram (model := model) (α := β))

  let args : TorchLean.TensorPack Float (Runtime.Autograd.TorchLean.NN.Seq.paramShapes model ++ [xShape])
    :=
    tensorpack! w1, b1, w2, b2, x

  let yTorch : Tensor Float yShape :=
    _root_.Runtime.Autograd.Torch.CompiledGraph.forward compiled args

  -- Since `outDim = 1`, check the single coordinate. (Kept structured so it scales to `outDim >
  -- 1`.)
  for i in List.finRange outDim do
    assertApprox s!"mlp forward[{i.val}] spec/torchlean" (vecVal ySpec i) (vecVal yTorch i) 1e-6

  IO.println "torchlean_spec_mlp_equiv_check: ok"

end TorchLeanSpecMLPEquivCheck
end Floats
end Tests
