/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Infer
public meta import NN.IR.Infer

/-!
# IR Shape Contract Facts

Closed facts for the graph-level shape contracts used by `NN.IR.Infer`.

These used to live as an IO regression test. They are pure Lean computations, so the useful
artifact is a theorem: the same shape checker that protects runtime IR paths rejects malformed
nodes and accepts the intended scalar-broadcast case.
-/

@[expose] public section

namespace NN.Proofs.IR.ShapeContracts

open NN
open NN.IR
open _root_.NN.Spec

def chwSmall : Spec.Shape := .dim 1 (.dim 2 (.dim 2 .scalar))

def node (kind : OpKind) (outShape : Spec.Shape) : Node :=
  { id := 0, parents := [0], kind := kind, outShape := outShape }

def rejects (x : Except String Spec.Shape) : Bool :=
  match x with
  | .error _ => true
  | .ok _ => false

/-- A 3x3 convolution window is rejected for a 2x2 CHW input. -/
example :
    rejects (Infer.inferNodeOutShape (node (.conv2d 1 1 3 3 1 0) chwSmall) [chwSmall]) = true := by
  decide

/-- A 3x3 max-pool window is rejected for a 2x2 CHW input. -/
example :
    rejects (Infer.inferNodeOutShape (node (.maxPool2d 3 3 1) chwSmall) [chwSmall]) = true := by
  decide

/-- Broadcast declarations must agree with the parent shape. -/
example :
    rejects
      (Infer.inferNodeOutShape
        (node (.broadcastTo (.dim 2 .scalar) (.dim 3 .scalar)) (.dim 3 .scalar))
        [.dim 2 .scalar]) = true := by
  decide

/-- Scalar-to-vector broadcast is accepted with the declared output shape. -/
example :
    Infer.inferNodeOutShape
        (node (.broadcastTo .scalar (.dim 3 .scalar)) (.dim 3 .scalar))
        [.scalar] =
      .ok (.dim 3 .scalar) := by
  decide

/-- LayerNorm rejects an empty normalized suffix. -/
example :
    rejects (Infer.inferNodeOutShape (node (.layernorm 1) (.dim 0 .scalar)) [.dim 0 .scalar]) =
      true := by
  decide

end NN.Proofs.IR.ShapeContracts
