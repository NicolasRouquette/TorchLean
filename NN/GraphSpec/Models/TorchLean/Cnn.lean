/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.TorchLean.NN
import Mathlib.Algebra.Order.Algebra

/-!
# TorchLean-executable model: CNN

This file provides a small CNN constructor using the TorchLean `Seq` builder.

We keep this model “PyTorch-shaped”: it is a literal chain of conv/pool/flatten/linear.

For residual / DAG-style CNNs, see GraphSpec-backed models like `resnet18`.
-/

@[expose] public section


namespace NN
namespace GraphSpec
namespace Models
namespace TorchLean

open NN.Tensor

/--
2-conv CNN in TorchLean:

`Conv2D → ReLU → MaxPool2D → Conv2D → ReLU → MaxPool2D → Flatten → Linear`.

Initialization is deterministic and matches the current GraphSpec primitive convention:
- each “parameterized layer occurrence” gets an index `i = 0,1,2,...`,
- and seeds are `seedW/seedK = 2*i`, `seedB = 2*i + 1`.

So for this CNN:
- Conv1 uses `(seedK,seedB) = (0,1)`
- Conv2 uses `(2,3)`
- Linear head uses `(seedW,seedB) = (4,5)`
-/
def twoConvCnn
    (inC c1 c2 outDim inH inW kH kW stride1 padding1 stride2 padding2 poolKH poolKW poolStride1
      poolStride2 : Nat)
    {h_inC : inC ≠ 0} {h_c1 : c1 ≠ 0} {_h_c2 : c2 ≠ 0}
    {h_kH : kH ≠ 0} {h_kW : kW ≠ 0}
    {h_poolKH : poolKH ≠ 0} {h_poolKW : poolKW ≠ 0} :
    _root_.Runtime.Autograd.TorchLean.NN.Seq (.dim inC (.dim inH (.dim inW .scalar))) (.dim outDim .scalar) :=
  let firstConvOutHeight : Nat :=
    Spec.Shape.slidingWindowOutDim inH kH stride1 padding1
  let firstConvOutWidth : Nat :=
    Spec.Shape.slidingWindowOutDim inW kW stride1 padding1
  let firstPoolOutHeight : Nat :=
    Spec.poolOutDim firstConvOutHeight poolKH poolStride1 0
  let firstPoolOutWidth : Nat :=
    Spec.poolOutDim firstConvOutWidth poolKW poolStride1 0
  let secondConvOutHeight : Nat :=
    Spec.Shape.slidingWindowOutDim firstPoolOutHeight kH stride2 padding2
  let secondConvOutWidth : Nat :=
    Spec.Shape.slidingWindowOutDim firstPoolOutWidth kW stride2 padding2
  let secondPoolOutHeight : Nat :=
    Spec.poolOutDim secondConvOutHeight poolKH poolStride2 0
  let secondPoolOutWidth : Nat :=
    Spec.poolOutDim secondConvOutWidth poolKW poolStride2 0
  let featSize : Nat :=
    Spec.Shape.size (.dim c2 (.dim secondPoolOutHeight (.dim secondPoolOutWidth .scalar)))
  _root_.Runtime.Autograd.TorchLean.NN.singleLayer
      (_root_.Runtime.Autograd.TorchLean.NN.conv2d
        (inC := inC) (outC := c1) (kH := kH) (kW := kW) (stride := stride1) (padding := padding1)
        (inH := inH) (inW := inW) (h1 := h_inC) (h2 := h_kH) (h3 := h_kW)
        (seedK := 0) (seedB := 1))
    >>>
    _root_.Runtime.Autograd.TorchLean.NN.singleLayer _root_.Runtime.Autograd.TorchLean.NN.relu
    >>>
    _root_.Runtime.Autograd.TorchLean.NN.singleLayer
      (_root_.Runtime.Autograd.TorchLean.NN.maxPool2d
        (kH := poolKH) (kW := poolKW) (inH := firstConvOutHeight) (inW := firstConvOutWidth) (inC := c1) (stride :=
          poolStride1)
        (h1 := h_poolKH) (h2 := h_poolKW))
    >>>
    _root_.Runtime.Autograd.TorchLean.NN.singleLayer
      (_root_.Runtime.Autograd.TorchLean.NN.conv2d
        (inC := c1) (outC := c2) (kH := kH) (kW := kW) (stride := stride2) (padding := padding2)
        (inH := firstPoolOutHeight) (inW := firstPoolOutWidth) (h1 := h_c1) (h2 := h_kH) (h3 := h_kW)
        (seedK := 2) (seedB := 3))
    >>>
    _root_.Runtime.Autograd.TorchLean.NN.singleLayer _root_.Runtime.Autograd.TorchLean.NN.relu
    >>>
    _root_.Runtime.Autograd.TorchLean.NN.singleLayer
      (_root_.Runtime.Autograd.TorchLean.NN.maxPool2d
        (kH := poolKH) (kW := poolKW) (inH := secondConvOutHeight) (inW := secondConvOutWidth) (inC := c2) (stride :=
          poolStride2)
        (h1 := h_poolKH) (h2 := h_poolKW))
    >>>
    _root_.Runtime.Autograd.TorchLean.NN.singleLayer
      (_root_.Runtime.Autograd.TorchLean.NN.flatten (s := .dim c2 (.dim secondPoolOutHeight (.dim secondPoolOutWidth .scalar))))
    >>>
    _root_.Runtime.Autograd.TorchLean.NN.singleLayer
      (_root_.Runtime.Autograd.TorchLean.NN.linear featSize outDim (seedW := 4) (seedB := 5))

end TorchLean
end Models
end GraphSpec
end NN
