/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN
public import NN.Tests.Runtime.Floats.Utils
public import Std

/-!
# TorchLeanIndexShapeCheck

Runtime checks for TorchLean indexing and shape helpers over floats.

These checks focus on shape-manipulating ops and indexing helpers that are easy to break when
refactoring tensor APIs.
-/

@[expose] public section

open Spec
open Tensor
open Tests.Floats.Utils

namespace Tests
namespace Floats
namespace TorchLeanIndexShapeCheck

/-- A task head preserves every leading axis rather than treating only axis zero as a batch. -/
def multiLeadingClassifier :
    TorchLean.nn.Sequential (shape![2, 3, 4, 5]) (shape![2, 3, 7]) :=
  TorchLean.nn.heads.classifier (leading := shape![2, 3]) (s := shape![4, 5]) 7

/-- Check that prefix flattening preserves every leading slice and row-major suffix order. -/
def checkFlattenPrefix : IO Unit := do
  let x : Tensor Float (shape![2, 2, 2, 2]) :=
    Tensor.dim (fun i => Tensor.dim (fun j => Tensor.dim (fun k => Tensor.dim (fun l =>
      Tensor.scalar (Float.ofNat (100 * i.val + 10 * j.val + 2 * k.val + l.val))))))
  let y : Tensor Float (shape![2, 2, 3]) :=
    TorchLean.Tensor.flattenPrefix shape![2, 2] 3 (by decide) x
  for i in List.finRange 2 do
    for j in List.finRange 2 do
      let row := Spec.getAtSpec (Spec.getAtSpec y i) j
      for k in List.finRange 3 do
        assertApprox s!"flattenPrefix[{i.val},{j.val},{k.val}]"
          (vecVal row k) (Float.ofNat (100 * i.val + 10 * j.val + k.val))

def gradGatherVec (execution : _root_.Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float (.dim 3 .scalar)) := do
  let sess ← _root_.Runtime.Autograd.TorchLean.Session.new (α := Float)
    (opts := { execution := execution })
  let xVal : Tensor Float (.dim 3 .scalar) := NN.Tensor.vector (α := Float) [1.0, 2.0, 3.0]
  let x ← _root_.Runtime.Autograd.TorchLean.Session.input sess xVal (name := some "x") (requiresGrad := true)
  let y ← _root_.Runtime.Autograd.TorchLean.Session.gatherScalar sess (n := 3) x ⟨1, by decide⟩
  let grads ← _root_.Runtime.Autograd.TorchLean.Session.backwardScalarDenseAll sess y
  _root_.Runtime.Autograd.TorchLean.Session.grad grads x

def gradGatherScalarNat (execution : _root_.Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float (.dim 3 .scalar)) := do
  let sess ← _root_.Runtime.Autograd.TorchLean.Session.new (α := Float)
    (opts := { execution := execution })
  let xVal : Tensor Float (.dim 3 .scalar) := NN.Tensor.vector (α := Float) [1.0, 2.0, 3.0]
  let x ← _root_.Runtime.Autograd.TorchLean.Session.input sess xVal (name := some "x") (requiresGrad := true)
  let y ← _root_.Runtime.Autograd.TorchLean.Session.gatherScalarNatOrZero sess (n := 3) x 1
  let grads ← _root_.Runtime.Autograd.TorchLean.Session.backwardScalarDenseAll sess y
  _root_.Runtime.Autograd.TorchLean.Session.grad grads x

def gradGatherVecNat (execution : _root_.Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float (.dim 3 .scalar)) := do
  let sess ← _root_.Runtime.Autograd.TorchLean.Session.new (α := Float)
    (opts := { execution := execution })
  let xVal : Tensor Float (.dim 3 .scalar) := NN.Tensor.vector (α := Float) [1.0, 2.0, 3.0]
  let idx : Tensor Nat (.dim 4 .scalar) := NN.Tensor.vector (α := Nat) [2, 0, 2, 10]
  let x ← _root_.Runtime.Autograd.TorchLean.Session.input sess xVal (name := some "x") (requiresGrad := true)
  let y ← _root_.Runtime.Autograd.TorchLean.Session.gatherVecNatOrZero sess (n := 3) (k := 4) x idx
  let total ← _root_.Runtime.Autograd.TorchLean.Session.sum sess (sh := .dim 4 .scalar) y
  let grads ← _root_.Runtime.Autograd.TorchLean.Session.backwardScalarDenseAll sess total
  _root_.Runtime.Autograd.TorchLean.Session.grad grads x

def gradGatherRowsNat (execution : _root_.Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float (.dim 3 (.dim 2 .scalar))) :=
  do
  let sess ← _root_.Runtime.Autograd.TorchLean.Session.new (α := Float)
    (opts := { execution := execution })
  let xVal : Tensor Float (.dim 3 (.dim 2 .scalar)) :=
    Tensor.dim (fun r => Tensor.dim (fun c => Tensor.scalar (Float.ofNat (r.val * 10 + c.val + 1))))
  let idx : Tensor Nat (.dim 3 .scalar) := NN.Tensor.vector (α := Nat) [2, 10, 2]
  let x ← _root_.Runtime.Autograd.TorchLean.Session.input sess xVal (name := some "x") (requiresGrad := true)
  let y ← _root_.Runtime.Autograd.TorchLean.Session.gatherRowsNatOrZero sess (rows := 3) (cols := 2) (k := 3) x idx
  let total ← _root_.Runtime.Autograd.TorchLean.Session.sum sess (sh := .dim 3 (.dim 2 .scalar)) y
  let grads ← _root_.Runtime.Autograd.TorchLean.Session.backwardScalarDenseAll sess total
  _root_.Runtime.Autograd.TorchLean.Session.grad grads x

def gradBroadcastScalar (execution : _root_.Runtime.Autograd.Torch.ExecutionMode) : IO Float := do
  let sess ← _root_.Runtime.Autograd.TorchLean.Session.new (α := Float)
    (opts := { execution := execution })
  let sVal : Tensor Float Shape.scalar := Tensor.scalar 2.0
  let sRef ← _root_.Runtime.Autograd.TorchLean.Session.input sess sVal (name := some "s") (requiresGrad := true)
  let cb : Shape.CanBroadcastTo Shape.scalar (.dim 4 .scalar) :=
    Shape.CanBroadcastTo.scalarTo (.dim 4 .scalar)
  let v ← _root_.Runtime.Autograd.TorchLean.Session.broadcastTo sess (sh1 := Shape.scalar) (sh2 := .dim 4 .scalar) cb sRef
  let total ← _root_.Runtime.Autograd.TorchLean.Session.sum sess (sh := .dim 4 .scalar) v
  let grads ← _root_.Runtime.Autograd.TorchLean.Session.backwardScalarDenseAll sess total
  let dsT ← _root_.Runtime.Autograd.TorchLean.Session.grad grads sRef
  pure (scalarVal dsT)

def gradReshapeMat (execution : _root_.Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float (.dim 2 (.dim 3 .scalar))) := do
  let sess ← _root_.Runtime.Autograd.TorchLean.Session.new (α := Float)
    (opts := { execution := execution })
  let xVal : Tensor Float (.dim 2 (.dim 3 .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (Float.ofNat (i.val * 10 + j.val))))
  let x ← _root_.Runtime.Autograd.TorchLean.Session.input sess xVal (name := some "x") (requiresGrad := true)
  let h : Spec.Shape.size (.dim 2 (.dim 3 .scalar)) = Spec.Shape.size (.dim 6 .scalar) := by decide
  let y ← _root_.Runtime.Autograd.TorchLean.Session.reshape sess (sh1 := .dim 2 (.dim 3 .scalar)) (sh2 := .dim 6 .scalar) x
    h
  let total ← _root_.Runtime.Autograd.TorchLean.Session.sum sess (sh := .dim 6 .scalar) y
  let grads ← _root_.Runtime.Autograd.TorchLean.Session.backwardScalarDenseAll sess total
  _root_.Runtime.Autograd.TorchLean.Session.grad grads x

def gradTransposeMat (execution : _root_.Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float (.dim 2 (.dim 3 .scalar))) :=
  do
  let sess ← _root_.Runtime.Autograd.TorchLean.Session.new (α := Float)
    (opts := { execution := execution })
  let xVal : Tensor Float (.dim 2 (.dim 3 .scalar)) :=
    Tensor.dim (fun i => Tensor.dim (fun j => Tensor.scalar (Float.ofNat (i.val + j.val + 1))))
  let x ← _root_.Runtime.Autograd.TorchLean.Session.input sess xVal (name := some "x") (requiresGrad := true)
  let xt ← _root_.Runtime.Autograd.TorchLean.Session.transpose2d sess (m := 2) (n := 3) x
  let total ← _root_.Runtime.Autograd.TorchLean.Session.sum sess (sh := .dim 3 (.dim 2 .scalar)) xt
  let grads ← _root_.Runtime.Autograd.TorchLean.Session.backwardScalarDenseAll sess total
  _root_.Runtime.Autograd.TorchLean.Session.grad grads x

def gradReduceMeanVec (execution : _root_.Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float (.dim 3 .scalar)) := do
  let sess ← _root_.Runtime.Autograd.TorchLean.Session.new (α := Float)
    (opts := { execution := execution })
  let xVal : Tensor Float (.dim 3 .scalar) := NN.Tensor.vector (α := Float) [1.0, 2.0, 3.0]
  let x ← _root_.Runtime.Autograd.TorchLean.Session.input sess xVal (name := some "x") (requiresGrad := true)
  let m ← _root_.Runtime.Autograd.TorchLean.Session.reduceMean sess (sh := .dim 3 .scalar) 0 x
  let grads ← _root_.Runtime.Autograd.TorchLean.Session.backwardScalarDenseAll sess m
  _root_.Runtime.Autograd.TorchLean.Session.grad grads x

def gradScatterAddVec (execution : _root_.Runtime.Autograd.Torch.ExecutionMode) :
    IO (Tensor Float (.dim 3 .scalar) × Float) :=
  do
  let sess ← _root_.Runtime.Autograd.TorchLean.Session.new (α := Float)
    (opts := { execution := execution })
  let xVal : Tensor Float (.dim 3 .scalar) := NN.Tensor.vector (α := Float) [1.0, 2.0, 3.0]
  let vVal : Tensor Float Shape.scalar := Tensor.scalar 5.0
  let x ← _root_.Runtime.Autograd.TorchLean.Session.input sess xVal (name := some "x") (requiresGrad := true)
  let v ← _root_.Runtime.Autograd.TorchLean.Session.input sess vVal (name := some "v") (requiresGrad := true)
  let y ← _root_.Runtime.Autograd.TorchLean.Session.scatterAddVec sess (n := 3) x v ⟨2, by decide⟩
  let total ← _root_.Runtime.Autograd.TorchLean.Session.sum sess (sh := .dim 3 .scalar) y
  let grads ← _root_.Runtime.Autograd.TorchLean.Session.backwardScalarDenseAll sess total
  let dx ← _root_.Runtime.Autograd.TorchLean.Session.grad grads x
  let dvT ← _root_.Runtime.Autograd.TorchLean.Session.grad grads v
  pure (dx, scalarVal dvT)

/-- Check forward-mode derivatives for gathers whose indices are runtime inputs. -/
def checkRuntimeGatherJvps : IO Unit := do
  let scalarSession ← _root_.Runtime.Autograd.TorchLean.Session.new (α := Float)
    (opts := { execution := .typedGraph })
  let scalarIndex ← _root_.Runtime.Autograd.TorchLean.Session.inputNat scalarSession 1
  let scalarInput ← _root_.Runtime.Autograd.TorchLean.Session.input scalarSession
    (NN.Tensor.vector (α := Float) [10.0, 20.0, 30.0])
  let scalarOutput ← _root_.Runtime.Autograd.TorchLean.Session.gatherScalarRefOrZero scalarSession
    (n := 3) scalarInput scalarIndex
  let scalarTangent ← _root_.Runtime.Autograd.TorchLean.Session.jvpLeaf
    (α := Float) (shOut := Shape.scalar) (shX := .dim 3 .scalar) scalarSession
    scalarOutput scalarInput (NN.Tensor.vector (α := Float) [2.0, 3.0, 5.0])
  assertApprox "gather_scalar_ref JVP" (scalarVal scalarTangent) 3.0

  let rowSession ← _root_.Runtime.Autograd.TorchLean.Session.new (α := Float)
    (opts := { execution := .typedGraph })
  let rowIndex ← _root_.Runtime.Autograd.TorchLean.Session.inputNat rowSession 2
  let rowValue : Tensor Float (.dim 3 (.dim 2 .scalar)) :=
    Tensor.dim (fun r => Tensor.dim (fun c =>
      Tensor.scalar (Float.ofNat (r.val * 2 + c.val))))
  let rowInput ← _root_.Runtime.Autograd.TorchLean.Session.input rowSession rowValue
  let rowOutput ← _root_.Runtime.Autograd.TorchLean.Session.gatherRowRefOrZero rowSession
    (rows := 3) (cols := 2) rowInput rowIndex
  let rowDirection : Tensor Float (.dim 3 (.dim 2 .scalar)) :=
    Tensor.dim (fun r => Tensor.dim (fun c =>
      Tensor.scalar (Float.ofNat (10 + r.val * 2 + c.val))))
  let rowTangent ← _root_.Runtime.Autograd.TorchLean.Session.jvpLeaf
    (α := Float) (shOut := .dim 2 .scalar) (shX := .dim 3 (.dim 2 .scalar)) rowSession
    rowOutput rowInput rowDirection
  assertApprox "gather_row_ref JVP[0]" (vecVal rowTangent ⟨0, by decide⟩) 14.0
  assertApprox "gather_row_ref JVP[1]" (vecVal rowTangent ⟨1, by decide⟩) 15.0

  let vectorSession ← _root_.Runtime.Autograd.TorchLean.Session.new (α := Float)
    (opts := { execution := .typedGraph })
  let vectorIndices ← _root_.Runtime.Autograd.TorchLean.Session.inputNatVec vectorSession
    (NN.Tensor.vector (α := Nat) [2, 0, 2, 10])
  let vectorInput ← _root_.Runtime.Autograd.TorchLean.Session.input vectorSession
    (NN.Tensor.vector (α := Float) [10.0, 20.0, 30.0])
  let vectorOutput ← _root_.Runtime.Autograd.TorchLean.Session.gatherVecRefOrZero vectorSession
    (n := 3) (k := 4) vectorInput vectorIndices
  let vectorTangent ← _root_.Runtime.Autograd.TorchLean.Session.jvpLeaf
    (α := Float) (shOut := .dim 4 .scalar) (shX := .dim 3 .scalar) vectorSession
    vectorOutput vectorInput (NN.Tensor.vector (α := Float) [5.0, 7.0, 11.0])
  let vectorExpected : List Float := [11.0, 5.0, 11.0, 0.0]
  for i in List.finRange 4 do
    assertApprox s!"gather_vec_ref JVP[{i.val}]" (vecVal vectorTangent i)
      vectorExpected[i.val]!

  let rowsSession ← _root_.Runtime.Autograd.TorchLean.Session.new (α := Float)
    (opts := { execution := .typedGraph })
  let rowsIndices ← _root_.Runtime.Autograd.TorchLean.Session.inputNatVec rowsSession
    (NN.Tensor.vector (α := Nat) [2, 10, 0])
  let rowsInput ← _root_.Runtime.Autograd.TorchLean.Session.input rowsSession rowValue
  let rowsOutput ← _root_.Runtime.Autograd.TorchLean.Session.gatherRowsRefOrZero rowsSession
    (rows := 3) (cols := 2) (k := 3) rowsInput rowsIndices
  let rowsTangent ← _root_.Runtime.Autograd.TorchLean.Session.jvpLeaf
    (α := Float) (shOut := .dim 3 (.dim 2 .scalar))
    (shX := .dim 3 (.dim 2 .scalar)) rowsSession rowsOutput rowsInput rowDirection
  let expectedRows : List (List Float) := [[14.0, 15.0], [0.0, 0.0], [10.0, 11.0]]
  for i in List.finRange 3 do
    for j in List.finRange 2 do
      assertApprox s!"gather_rows_ref JVP[{i.val},{j.val}]" (matVal rowsTangent i j)
        ((expectedRows[i.val]!)[j.val]!)

/-- Check exact-weight construction, freezing, lookup, and vocabulary validation. -/
def checkEmbeddingModule : IO Unit := do
  let weight : Tensor Float (.dim 4 (.dim 2 .scalar)) :=
    Tensor.dim (fun row => Tensor.dim (fun col =>
      Tensor.scalar (Float.ofNat (10 * row.val + col.val))))
  let table := TorchLean.nn.Embedding.ofWeight weight
  let model := table.model (.dim 3 .scalar)
  let module ← TorchLean.nn.IndexedModule.instantiate model {}

  let frozen := TorchLean.nn.Embedding.ofWeight weight true
  unless frozen.requiresGrad == false && (frozen.model (.dim 3 .scalar)).requiresGrad == [false] do
    throw <| IO.userError "frozen embedding still requests parameter gradients"

  let output ← module.predict (NN.Tensor.vector (α := Nat) [2, 0, 2])
  let output := Spec.mapTensor Float32.toFloat output
  let expected : List (List Float) := [[20.0, 21.0], [0.0, 1.0], [20.0, 21.0]]

  let wasRejected ←
    try
      let _ ← module.predict (NN.Tensor.vector (α := Nat) [0, 4, 1])
      pure false
    catch _ => pure true
  unless wasRejected do
    throw <| IO.userError "embedding accepted a token outside its vocabulary"

  unless (← module.isTraining) do
    throw <| IO.userError "embedding prediction did not restore the module's training mode"

  for i in List.finRange 3 do
    for j in List.finRange 2 do
      assertApprox s!"embedding[{i.val},{j.val}]" (matVal output i j)
        ((expected[i.val]!)[j.val]!)

/-- Check that an indexed model can be paired with a loss and trained without casting its IDs. -/
def embeddingTrainingLoss (execution : _root_.Runtime.Autograd.Torch.ExecutionMode) :
    IO (Float × Float) := do
  let table := TorchLean.nn.build 19 <| TorchLean.nn.embedding 3 2
  let model := table.model (.dim 2 .scalar)
  let objective := TorchLean.nn.IndexedModel.Objective.mse model
  let module ← TorchLean.Module.instantiate { execution := execution } objective
  let table : Tensor Float (.dim 3 (.dim 2 .scalar)) :=
    Tensor.dim (fun row => Tensor.dim (fun col =>
      let values : List (List Float) := [[0.0, 0.0], [2.0, -2.0], [1.0, 1.0]]
      Tensor.scalar ((values[row.val]!)[col.val]!)))
  module.loadState (TensorPack! table)

  let target : Tensor Float (.dim 2 (.dim 2 .scalar)) := Spec.fill 0.0 _
  let tokenIds : Tensor Nat (.dim 2 .scalar) := NN.Tensor.vector (α := Nat) [1, 1]
  let inputs := TensorPack! target
  let indices := TensorPack! tokenIds
  let before ← TorchLean.Module.loss module inputs indices
  let _ ← TorchLean.Module.sgdStepWithLoss module 0.1 inputs indices
  let after ← TorchLean.Module.loss module inputs indices
  pure (scalarVal before, scalarVal after)

/-- Check indexed embedding training under both execution modes. -/
def checkEmbeddingTraining : IO Unit := do
  let (eagerBefore, eagerAfter) ← embeddingTrainingLoss .eager
  let (graphBefore, graphAfter) ← embeddingTrainingLoss .typedGraph
  assertApprox "embedding training initial loss eager/typed-graph" eagerBefore graphBefore
  assertApprox "embedding training final loss eager/typed-graph" eagerAfter graphAfter
  unless eagerAfter < eagerBefore do
    throw <| IO.userError
      s!"embedding training did not reduce loss ({eagerBefore} -> {eagerAfter})"

  for execution in [_root_.Runtime.Autograd.Torch.ExecutionMode.eager,
      _root_.Runtime.Autograd.Torch.ExecutionMode.typedGraph] do
    let table := TorchLean.nn.build 19 <| TorchLean.nn.embedding 3 2
    let model := table.model (.dim 2 .scalar)
    let objective := TorchLean.nn.IndexedModel.Objective.mse model
    let module ← TorchLean.Module.instantiate { execution := execution } objective
    let target : Tensor Float (.dim 2 (.dim 2 .scalar)) := Spec.fill 0.0 _
    let invalidIds : Tensor Nat (.dim 2 .scalar) := NN.Tensor.vector (α := Nat) [0, 3]
    let rejected ←
      try
        let _ ← TorchLean.Module.loss module (TensorPack! target) (TensorPack! invalidIds)
        pure false
      catch _ => pure true
    unless rejected do
      throw <| IO.userError
        "embedding objective accepted a token outside its vocabulary"

def run : IO Unit := do
  IO.println "torchlean_index_shape_check: begin"

  let gE ← gradGatherVec .eager
  let gC ← gradGatherVec .typedGraph
  for i in List.finRange 3 do
    assertApprox s!"gather grad[{i.val}] eager/typed-graph" (vecVal gE i) (vecVal gC i)
  for i in List.finRange 3 do
    assertApprox s!"gather grad[{i.val}] expected" (vecVal gE i) (if i.val = 1 then 1.0 else 0.0)

  let gnE ← gradGatherScalarNat .eager
  let gnC ← gradGatherScalarNat .typedGraph
  for i in List.finRange 3 do
    assertApprox s!"gather_scalar_nat_or_zero grad[{i.val}] eager/typed-graph" (vecVal gnE i) (vecVal gnC i)
    assertApprox s!"gather_scalar_nat_or_zero grad[{i.val}] expected" (vecVal gnE i) (if i.val = 1 then 1.0
      else 0.0)

  let gvE ← gradGatherVecNat .eager
  let gvC ← gradGatherVecNat .typedGraph
  for i in List.finRange 3 do
    assertApprox s!"gather_vec_nat_or_zero dx[{i.val}] eager/typed-graph" (vecVal gvE i) (vecVal gvC i)
  assertApprox "gather_vec_nat_or_zero dx[0] expected" (vecVal gvE ⟨0, by decide⟩) 1.0
  assertApprox "gather_vec_nat_or_zero dx[1] expected" (vecVal gvE ⟨1, by decide⟩) 0.0
  assertApprox "gather_vec_nat_or_zero dx[2] expected" (vecVal gvE ⟨2, by decide⟩) 2.0

  let grE ← gradGatherRowsNat .eager
  let grC ← gradGatherRowsNat .typedGraph
  for i in List.finRange 3 do
    for j in List.finRange 2 do
      assertApprox s!"gather_rows_nat_or_zero dx[{i.val},{j.val}] eager/typed-graph" (matVal grE i j) (matVal
        grC i j)
  for j in List.finRange 2 do
    assertApprox s!"gather_rows_nat_or_zero dx[0,{j.val}] expected" (matVal grE ⟨0, by decide⟩ j) 0.0
    assertApprox s!"gather_rows_nat_or_zero dx[1,{j.val}] expected" (matVal grE ⟨1, by decide⟩ j) 0.0
    assertApprox s!"gather_rows_nat_or_zero dx[2,{j.val}] expected" (matVal grE ⟨2, by decide⟩ j) 2.0

  let bsE ← gradBroadcastScalar .eager
  let bsC ← gradBroadcastScalar .typedGraph
  assertApprox "broadcast scalar grad eager/typed-graph" bsE bsC
  assertApprox "broadcast scalar grad expected" bsE 4.0

  let rE ← gradReshapeMat .eager
  let rC ← gradReshapeMat .typedGraph
  for i in List.finRange 2 do
    for j in List.finRange 3 do
      assertApprox s!"reshape grad[{i.val},{j.val}] eager/typed-graph" (matVal rE i j) (matVal rC i j)
      assertApprox s!"reshape grad[{i.val},{j.val}] expected" (matVal rE i j) 1.0

  let tE ← gradTransposeMat .eager
  let tC ← gradTransposeMat .typedGraph
  for i in List.finRange 2 do
    for j in List.finRange 3 do
      assertApprox s!"transpose grad[{i.val},{j.val}] eager/typed-graph" (matVal tE i j) (matVal tC i
        j)
      assertApprox s!"transpose grad[{i.val},{j.val}] expected" (matVal tE i j) 1.0

  let mE ← gradReduceMeanVec .eager
  let mC ← gradReduceMeanVec .typedGraph
  for i in List.finRange 3 do
    assertApprox s!"reduce_mean grad[{i.val}] eager/typed-graph" (vecVal mE i) (vecVal mC i)
    assertApprox s!"reduce_mean grad[{i.val}] expected" (vecVal mE i) (1.0 / 3.0) 1e-6

  let (sxE, svE) ← gradScatterAddVec .eager
  let (sxC, svC) ← gradScatterAddVec .typedGraph
  for i in List.finRange 3 do
    assertApprox s!"scatter_add_vec dx[{i.val}] eager/typed-graph" (vecVal sxE i) (vecVal sxC i)
    assertApprox s!"scatter_add_vec dx[{i.val}] expected" (vecVal sxE i) 1.0
  assertApprox "scatter_add_vec dv eager/typed-graph" svE svC
  assertApprox "scatter_add_vec dv expected" svE 1.0

  checkFlattenPrefix
  checkRuntimeGatherJvps
  checkEmbeddingModule
  checkEmbeddingTraining

  IO.println "torchlean_index_shape_check: ok"

end TorchLeanIndexShapeCheck
end Floats
end Tests
