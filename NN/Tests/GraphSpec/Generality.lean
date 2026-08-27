/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.GraphSpec.DAG

/-!
# GraphSpec Generality Tests

Compile-time regressions for broadcasted DAG matrix multiplication, vector contractions expressed
through reshape, and attention over more than one leading axis.
-/

@[expose] public section

namespace NN.Tests.GraphSpec.Generality

open _root_.Spec
open NN.GraphSpec.DAG

/-- A shared matrix broadcasts across both result batch axes. -/
def sharedMatrixMatmul (outer inner m n p : Nat) :
    PrimOp [[m, n], [outer, inner, n, p]] [outer, inner, m, p] :=
  PrimOp.matmul .scalar [outer, inner] [outer, inner] m n p
    (Shape.CanBroadcastTo.scalarTo [outer, inner])
    (Shape.CanBroadcastTo.refl [outer, inner])

/-- A shared vector-matrix contraction is a row reshape, general matmul, and result reshape. -/
def sharedVectorMatmul (batch rows columns : Nat) :
    Term [[rows], [batch, rows, columns]] [batch, columns] :=
  let vector : Term [[rows], [batch, rows, columns]] [rows] := Term.var .head
  let matrices : Term [[rows], [batch, rows, columns]] [batch, rows, columns] :=
    Term.var (.tail .head)
  let row : Term [[rows], [batch, rows, columns]] [1, rows] :=
    Term.op (PrimOp.reshape [rows] [1, rows] (by simp [Shape.size])) (.cons vector .nil)
  let product : Term [[rows], [batch, rows, columns]] [batch, 1, columns] :=
    Term.op
      (PrimOp.matmul .scalar [batch] [batch] 1 rows columns
        (Shape.CanBroadcastTo.scalarTo [batch]) (Shape.CanBroadcastTo.refl [batch]))
      (.cons row (.cons matrices .nil))
  Term.op (PrimOp.reshape [batch, 1, columns] [batch, columns] (by simp [Shape.size]))
    (.cons product .nil)

/-- Pairwise batched vector-matrix multiplication uses the same two reshapes. -/
def batchedVectorMatmul (batch rows columns : Nat) :
    Term [[batch, rows], [batch, rows, columns]] [batch, columns] :=
  let vectors : Term [[batch, rows], [batch, rows, columns]] [batch, rows] := Term.var .head
  let matrices : Term [[batch, rows], [batch, rows, columns]] [batch, rows, columns] :=
    Term.var (.tail .head)
  let rows' : Term [[batch, rows], [batch, rows, columns]] [batch, 1, rows] :=
    Term.op (PrimOp.reshape [batch, rows] [batch, 1, rows] (by simp [Shape.size]))
      (.cons vectors .nil)
  let product : Term [[batch, rows], [batch, rows, columns]] [batch, 1, columns] :=
    Term.op
      (PrimOp.matmul [batch] [batch] [batch] 1 rows columns
        (Shape.CanBroadcastTo.refl [batch]) (Shape.CanBroadcastTo.refl [batch]))
      (.cons rows' (.cons matrices .nil))
  Term.op (PrimOp.reshape [batch, 1, columns] [batch, columns] (by simp [Shape.size]))
    (.cons product .nil)

/-- Attention accepts an arbitrary prefix rather than one distinguished batch axis. -/
def attentionWithTwoLeadingAxes
    (outer inner n numHeads dModel headDim : Nat) (hN : 0 < n) :
    PrimOp
      [ [dModel, numHeads * headDim], [dModel, numHeads * headDim],
        [dModel, numHeads * headDim], [numHeads * headDim, dModel],
        [outer, inner, n, dModel] ]
      [outer, inner, n, dModel] :=
  PrimOp.multiHeadAttention [outer, inner] n numHeads dModel headDim hN

end NN.Tests.GraphSpec.Generality
