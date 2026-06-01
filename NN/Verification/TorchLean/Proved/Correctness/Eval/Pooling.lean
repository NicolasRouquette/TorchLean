/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Verification.TorchLean.Proved.Correctness.Eval.PayloadOps

/-!
# Pooling IR Evaluation

Local semantics for the CHW 2D pooling nodes in the shared IR.
-/

@[expose] public section

namespace NN.Verification.TorchLean.Proved

open _root_.Spec
open _root_.Spec.Tensor
open NN.IR

namespace Correctness

namespace IRStep

/-- Local IR semantics for unpadded max-pooling over CHW tensors. -/
theorem evalAt_maxPool2d_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {inC inH inW kH kW stride : Nat}
    (x : Tensor α (.dim inC (.dim inH (.dim inW .scalar))))
    (hkH : kH ≠ 0) (hkW : kW ≠ 0) (hs : stride ≠ 0)
    (hHeight : OpContracts.checkWindowFits "max_pool2d" "height" inH kH 0 = .ok ())
    (hWidth : OpContracts.checkWindowFits "max_pool2d" "width" inW kW 0 = .ok ()) :
    Graph.evalAt (α := α)
        (g := unaryGraphOut (.maxPool2d kH kW stride)
          (.dim inC (.dim inH (.dim inW .scalar)))
          (Spec.pool2dMultiOutShape inC inH inW kH kW stride))
        (payload := {})
        (input := DVal.mk (α := α) (.dim inC (.dim inH (.dim inW .scalar))) x)
        (vals := #[DVal.mk (α := α) (.dim inC (.dim inH (.dim inW .scalar))) x]) (i := 1)
      =
      Except.ok
        (DVal.mk (α := α) (Spec.pool2dMultiOutShape inC inH inW kH kW stride)
          (Spec.maxPool2dMultiSpec (α := α) (kH := kH) (kW := kW)
            (inH := inH) (inW := inW) (inC := inC) (stride := stride)
            (layer := ({} : Spec.MaxPool2DSpec kH kW stride hkH hkW hs)) (input := x))) := by
  simp [Graph.evalAt, unaryGraphOut, unaryNodeOut, Graph.getNode, Graph.getNode?,
    Graph.expectShape, hkH, hkW, hs, hHeight, hWidth, Bind.bind, Except.bind, Pure.pure,
    Except.pure]

/-- Local IR semantics for padded max-pooling over CHW tensors. -/
theorem evalAt_maxPool2dPad_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {inC inH inW kH kW stride padding : Nat}
    (x : Tensor α (.dim inC (.dim inH (.dim inW .scalar))))
    (hkH : kH ≠ 0) (hkW : kW ≠ 0) (hs : stride ≠ 0)
    (hHeight : OpContracts.checkWindowFits "max_pool2d_pad" "height" inH kH padding = .ok ())
    (hWidth : OpContracts.checkWindowFits "max_pool2d_pad" "width" inW kW padding = .ok ()) :
    Graph.evalAt (α := α)
        (g := unaryGraphOut (.maxPool2dPad kH kW stride padding)
          (.dim inC (.dim inH (.dim inW .scalar)))
          (Spec.pool2dMultiOutShapePad inC inH inW kH kW stride padding))
        (payload := {})
        (input := DVal.mk (α := α) (.dim inC (.dim inH (.dim inW .scalar))) x)
        (vals := #[DVal.mk (α := α) (.dim inC (.dim inH (.dim inW .scalar))) x]) (i := 1)
      =
      Except.ok
        (DVal.mk (α := α) (Spec.pool2dMultiOutShapePad inC inH inW kH kW stride padding)
          (Spec.maxPool2dMultiSpecPad (α := α) (kH := kH) (kW := kW)
            (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding := padding)
            (layer := ({} : Spec.MaxPool2DSpec kH kW stride hkH hkW hs)) (input := x))) := by
  simp [Graph.evalAt, unaryGraphOut, unaryNodeOut, Graph.getNode, Graph.getNode?,
    Graph.expectShape, hkH, hkW, hs, hHeight, hWidth, Bind.bind, Except.bind, Pure.pure,
    Except.pure]

/-- Local IR semantics for unpadded average-pooling over CHW tensors. -/
theorem evalAt_avgPool2d_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {inC inH inW kH kW stride : Nat}
    (x : Tensor α (.dim inC (.dim inH (.dim inW .scalar))))
    (hkH : kH ≠ 0) (hkW : kW ≠ 0) (hs : stride ≠ 0)
    (hHeight : OpContracts.checkWindowFits "avg_pool2d" "height" inH kH 0 = .ok ())
    (hWidth : OpContracts.checkWindowFits "avg_pool2d" "width" inW kW 0 = .ok ()) :
    Graph.evalAt (α := α)
        (g := unaryGraphOut (.avgPool2d kH kW stride)
          (.dim inC (.dim inH (.dim inW .scalar)))
          (Spec.pool2dMultiOutShape inC inH inW kH kW stride))
        (payload := {})
        (input := DVal.mk (α := α) (.dim inC (.dim inH (.dim inW .scalar))) x)
        (vals := #[DVal.mk (α := α) (.dim inC (.dim inH (.dim inW .scalar))) x]) (i := 1)
      =
      Except.ok
        (DVal.mk (α := α) (Spec.pool2dMultiOutShape inC inH inW kH kW stride)
          (Spec.avgPool2dMultiSpec (α := α) (kH := kH) (kW := kW)
            (inH := inH) (inW := inW) (inC := inC) (stride := stride)
            (h1 := hkH) (h2 := hkW)
            (layer := ({} : Spec.AvgPool2DSpec kH kW stride hkH hkW hs)) (input := x))) := by
  simp [Graph.evalAt, unaryGraphOut, unaryNodeOut, Graph.getNode, Graph.getNode?,
    Graph.expectShape, hkH, hkW, hs, hHeight, hWidth, Bind.bind, Except.bind, Pure.pure,
    Except.pure]

/-- Local IR semantics for padded average-pooling over CHW tensors. -/
theorem evalAt_avgPool2dPad_eq
    {α : Type} [Context α] [DecidableEq Shape]
    {inC inH inW kH kW stride padding : Nat}
    (x : Tensor α (.dim inC (.dim inH (.dim inW .scalar))))
    (hkH : kH ≠ 0) (hkW : kW ≠ 0) (hs : stride ≠ 0)
    (hHeight : OpContracts.checkWindowFits "avg_pool2d_pad" "height" inH kH padding = .ok ())
    (hWidth : OpContracts.checkWindowFits "avg_pool2d_pad" "width" inW kW padding = .ok ()) :
    Graph.evalAt (α := α)
        (g := unaryGraphOut (.avgPool2dPad kH kW stride padding)
          (.dim inC (.dim inH (.dim inW .scalar)))
          (Spec.pool2dMultiOutShapePad inC inH inW kH kW stride padding))
        (payload := {})
        (input := DVal.mk (α := α) (.dim inC (.dim inH (.dim inW .scalar))) x)
        (vals := #[DVal.mk (α := α) (.dim inC (.dim inH (.dim inW .scalar))) x]) (i := 1)
      =
      Except.ok
        (DVal.mk (α := α) (Spec.pool2dMultiOutShapePad inC inH inW kH kW stride padding)
          (Spec.avgPool2dMultiSpecPad (α := α) (kH := kH) (kW := kW)
            (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding := padding)
            (h1 := hkH) (h2 := hkW)
            (layer := ({} : Spec.AvgPool2DSpec kH kW stride hkH hkW hs)) (input := x))) := by
  simp [Graph.evalAt, unaryGraphOut, unaryNodeOut, Graph.getNode, Graph.getNode?,
    Graph.expectShape, hkH, hkW, hs, hHeight, hWidth, Bind.bind, Except.bind, Pure.pure,
    Except.pure]

end IRStep

end Correctness

end NN.Verification.TorchLean.Proved
