/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Infer

/-!
# IR Shape Contract Checks

These tests pin the graph-level contracts that protect executable IR paths from accepting malformed
shape declarations.
-/

@[expose] public section

namespace Tests
namespace Floats
namespace IRShapeContracts

open NN
open NN.IR
open Spec

def assertRejects (label : String) (x : Except String Shape) : IO Unit := do
  match x with
  | .error _ => pure ()
  | .ok s => throw (IO.userError s!"ir_shape_contracts: {label} unexpectedly accepted {repr s}")

def assertAccepts (label : String) (x : Except String Shape) (expected : Shape) : IO Unit := do
  match x with
  | .ok s =>
      unless s = expected do
        throw (IO.userError s!"ir_shape_contracts: {label}: expected {repr expected}, got {repr s}")
  | .error msg => throw (IO.userError s!"ir_shape_contracts: {label} rejected: {msg}")

def node (kind : OpKind) (outShape : Shape) : Node :=
  { id := 0, parents := [0], kind := kind, outShape := outShape }

def run : IO Unit := do
  IO.println "ir_shape_contracts: begin"
  let chwSmall : Shape := .dim 1 (.dim 2 (.dim 2 .scalar))
  assertRejects "conv window larger than input"
    (Infer.inferNodeOutShape (node (.conv2d 1 1 3 3 1 0) chwSmall) [chwSmall])
  assertRejects "pool window larger than input"
    (Infer.inferNodeOutShape (node (.maxPool2d 3 3 1) chwSmall) [chwSmall])
  assertRejects "incompatible declared broadcast"
    (Infer.inferNodeOutShape
      (node (.broadcastTo (.dim 2 .scalar) (.dim 3 .scalar)) (.dim 3 .scalar))
      [.dim 2 .scalar])
  assertAccepts "scalar broadcast"
    (Infer.inferNodeOutShape
      (node (.broadcastTo .scalar (.dim 3 .scalar)) (.dim 3 .scalar))
      [.scalar])
    (.dim 3 .scalar)
  assertRejects "empty layernorm suffix"
    (Infer.inferNodeOutShape (node (.layernorm 1) (.dim 0 .scalar)) [.dim 0 .scalar])
  IO.println "ir_shape_contracts: ok"

end IRShapeContracts
end Floats
end Tests

