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

/-- Pooling operations whose local evaluator proofs share the same graph and shape contract. -/
inductive Pool2DOperation where
  /-- Unpadded maximum pooling. -/
  | maximum
  /-- Padded maximum pooling. -/
  | maximumPadded (padding : Nat)
  /-- Unpadded average pooling. -/
  | average
  /-- Padded average pooling. -/
  | averagePadded (padding : Nat)

/-- Padding used by a pooling operation's window-fit contract. -/
def Pool2DOperation.padding : Pool2DOperation → Nat
  | .maximum | .average => 0
  | .maximumPadded padding | .averagePadded padding => padding

/-- Diagnostic operation name used by the evaluator's window-fit contract. -/
def Pool2DOperation.contractName : Pool2DOperation → String
  | .maximum => "max_pool2d"
  | .maximumPadded _ => "max_pool2d_pad"
  | .average => "avg_pool2d"
  | .averagePadded _ => "avg_pool2d_pad"

/-- IR opcode corresponding to a pooling operation. -/
def Pool2DOperation.toOpKind (op : Pool2DOperation) (kH kW stride : Nat) : OpKind :=
  match op with
  | .maximum => .maxPool2d kH kW stride
  | .maximumPadded padding => .maxPool2dPad kH kW stride padding
  | .average => .avgPool2d kH kW stride
  | .averagePadded padding => .avgPool2dPad kH kW stride padding

/-- Output shape computed by a pooling operation. -/
def Pool2DOperation.outShape (op : Pool2DOperation)
    (inC inH inW kH kW stride : Nat) : Shape :=
  match op with
  | .maximum | .average => Spec.pool2dMultiOutShape inC inH inW kH kW stride
  | .maximumPadded padding | .averagePadded padding =>
      Spec.pool2dMultiOutShapePad inC inH inW kH kW stride padding

/-- Typed denotation shared by the four pooling evaluator specializations. -/
def Pool2DOperation.denote
    {α : Type} [Context α]
    (op : Pool2DOperation) {inC inH inW kH kW stride : Nat}
    (hkH : kH ≠ 0) (hkW : kW ≠ 0) (hs : stride ≠ 0)
    (x : Tensor α (.dim inC (.dim inH (.dim inW .scalar)))) :
    Tensor α (op.outShape inC inH inW kH kW stride) :=
  match op with
  | .maximum =>
      Spec.maxPool2dMultiSpec (α := α) (kH := kH) (kW := kW)
        (inH := inH) (inW := inW) (inC := inC) (stride := stride)
        (layer := ({} : Spec.MaxPool2DSpec kH kW stride hkH hkW hs)) (input := x)
  | .maximumPadded padding =>
      Spec.maxPool2dMultiSpecPad (α := α) (kH := kH) (kW := kW)
        (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding := padding)
        (layer := ({} : Spec.MaxPool2DSpec kH kW stride hkH hkW hs)) (input := x)
  | .average =>
      Spec.avgPool2dMultiSpec (α := α) (kH := kH) (kW := kW)
        (inH := inH) (inW := inW) (inC := inC) (stride := stride)
        (h1 := hkH) (h2 := hkW)
        (layer := ({} : Spec.AvgPool2DSpec kH kW stride hkH hkW hs)) (input := x)
  | .averagePadded padding =>
      Spec.avgPool2dMultiSpecPad (α := α) (kH := kH) (kW := kW)
        (inH := inH) (inW := inW) (inC := inC) (stride := stride) (padding := padding)
        (h1 := hkH) (h2 := hkW)
        (layer := ({} : Spec.AvgPool2DSpec kH kW stride hkH hkW hs)) (input := x)

/-- Evaluate any supported CHW pooling operation in its canonical two-node graph. -/
theorem evalAt_pool2d_eq
    {α : Type} [Context α] [DecidableEq Shape]
    (op : Pool2DOperation) {inC inH inW kH kW stride : Nat}
    (x : Tensor α (.dim inC (.dim inH (.dim inW .scalar))))
    (hkH : kH ≠ 0) (hkW : kW ≠ 0) (hs : stride ≠ 0)
    (hHeight : OpContracts.checkWindowFits op.contractName "height" inH kH op.padding = .ok ())
    (hWidth : OpContracts.checkWindowFits op.contractName "width" inW kW op.padding = .ok ()) :
    Graph.evalAt (α := α)
        (g := unaryGraphOut (op.toOpKind kH kW stride)
          (.dim inC (.dim inH (.dim inW .scalar)))
          (op.outShape inC inH inW kH kW stride))
        (payload := {})
        (input := DVal.mk (α := α) (.dim inC (.dim inH (.dim inW .scalar))) x)
        (vals := #[DVal.mk (α := α) (.dim inC (.dim inH (.dim inW .scalar))) x]) (i := 1)
      =
      Except.ok
        (DVal.mk (α := α) (op.outShape inC inH inW kH kW stride)
          (op.denote hkH hkW hs x)) := by
  cases op <;>
    simp [Pool2DOperation.contractName, Pool2DOperation.padding] at hHeight hWidth <;>
    simp [Pool2DOperation.toOpKind, Pool2DOperation.outShape, Pool2DOperation.denote,
      Graph.evalAt, unaryGraphOut, unaryNodeOut, Graph.getNode, Graph.getNode?, Graph.expectShape,
      hkH, hkW, hs, hHeight, hWidth, Bind.bind, Except.bind, Pure.pure, Except.pure]

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
  exact evalAt_pool2d_eq .maximum x hkH hkW hs hHeight hWidth

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
  exact evalAt_pool2d_eq (.maximumPadded padding) x hkH hkW hs hHeight hWidth

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
  exact evalAt_pool2d_eq .average x hkH hkW hs hHeight hWidth

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
  exact evalAt_pool2d_eq (.averagePadded padding) x hkH hkW hs hHeight hWidth

end IRStep

end Correctness

end NN.Verification.TorchLean.Proved
