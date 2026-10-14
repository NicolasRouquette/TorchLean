/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.IR.Semantics
public import NN.Proofs.Tensor.Basic.LinearAlgebra
public import NN.Runtime.JAX
public import NN.Spec.Core.Context.Real
public import NN.Tensor
public import NN.Verification.Builtin.Proved.Correctness.WellFormed

/-!
# From Jaxpr to a permutation-invariance proof

The Python companion captures `sum(x * x)` and several other tensor computations.
`run` loads its artifacts through the shared IR checker and compares the native Lean
outputs with the recorded JAX outputs. Theorems below concern the real-valued IR
computation, for every vector length and every permutation, not just those samples.

Neither numerical agreement nor these theorems certify the Python adapter or XLA.
Permutation invariance uses exact addition and does not assert bitwise invariance
of a floating-point reduction.
-/

@[expose] public section

namespace NN.Examples.Interop.JAX

open TorchLean
open NN.IR
open NN.Verification.Builtin.Proved.Correctness (wellFormed_push)
open scoped BigOperators

/-- The three IR nodes emitted for `jax.numpy.sum(x * x)`. -/
def sumSquaresGraph (shape : Shape) : Graph :=
  { nodes := #[
      { id := 0, kind := .input, parents := #[], outShape := shape },
      { id := 1, kind := .mulElem, parents := #[0, 0], outShape := shape },
      { id := 2, kind := .sum, parents := #[1], outShape := [] }
    ] }

/-- The tensor calculation represented by the captured graph. -/
noncomputable def sumSquares {shape : Shape} (x : Tensor ℝ shape) : ℝ :=
  Tensor.sum (x * x)

/-- Evaluate the complete IR graph, rather than proving only a separate tensor formula. -/
theorem sumSquaresGraph_denote {shape : Shape} (x : Tensor ℝ shape) :
    (sumSquaresGraph shape).denote {} (Spec.SomeTensor.ofTensor x) 2 =
      .ok (Spec.SomeTensor.ofTensor (Tensor.scalar (sumSquares x))) := by
  have hWellFormed : (sumSquaresGraph shape).wellFormed = true := by
    let input : Node := { id := 0, kind := .input, parents := #[], outShape := shape }
    let product : Node := { id := 1, kind := .mulElem, parents := #[0, 0], outShape := shape }
    let total : Node := { id := 2, kind := .sum, parents := #[1], outShape := [] }
    have hInput := wellFormed_push
      ({ nodes := #[] } : Graph) input (by with_unfolding_all rfl)
      (by rfl) (by with_unfolding_all rfl) (by simp [Node.parentsBelow, input])
    have hProduct := wellFormed_push
      ({ nodes := #[input] } : Graph) product hInput
      (by rfl) (by with_unfolding_all rfl) (by simp [Node.parentsBelow, product])
    have hTotal := wellFormed_push
      ({ nodes := #[input, product] } : Graph) total hProduct
      (by rfl) (by with_unfolding_all rfl) (by simp [Node.parentsBelow, total])
    exact hTotal
  unfold Graph.denote Graph.denoteAll
  rw [hWellFormed]
  simp [Graph.denoteAllFrom, Graph.evalAt, Graph.getNode,
    Graph.evalNode, Graph.normalizeNodeOutput, Graph.expectShape,
    Graph.binaryParentIds, Graph.unaryParentId, binaryParents?, unaryParent?, sumSquaresGraph,
    Graph.getParentValue, sumSquares, Bind.bind, Except.bind, Pure.pure, Except.pure]
  rfl

/-- Reordering vector entries leaves the exact real sum of their squares unchanged. -/
theorem sumSquares_permutation {n : Nat} (permutation : Equiv.Perm (Fin n))
    (x y : Tensor ℝ [n]) (h : ∀ i, y[i] = x[permutation i]) :
    sumSquares y = sumSquares x := by
  unfold sumSquares Tensor.sum
  rw [Spec.sum_spec_vec, Spec.sum_spec_vec]
  have hSquare (z : Tensor ℝ [n]) (i : Fin n) :
      Tensor.getScalar (z * z) i = z[i] * z[i] := Spec.getScalar_mul_spec z z i
  simp_rw [hSquare]
  calc
    (∑ i : Fin n, y[i] * y[i]) = ∑ i : Fin n, x[permutation i] * x[permutation i] := by
      apply Finset.sum_congr rfl
      intro i _
      rw [h i]
    _ = ∑ i, x[i] * x[i] := Equiv.sum_comp permutation (fun i : Fin n => x[i] * x[i])

/-- The captured graph itself is permutation invariant under its real-valued semantics. -/
theorem sumSquaresGraph_permutation {n : Nat} (permutation : Equiv.Perm (Fin n))
    (x y : Tensor ℝ [n]) (h : ∀ i, y[i] = x[permutation i]) :
    (sumSquaresGraph [n]).denote {} (Spec.SomeTensor.ofTensor y) 2 =
      (sumSquaresGraph [n]).denote {} (Spec.SomeTensor.ofTensor x) 2 := by
  rw [sumSquaresGraph_denote, sumSquaresGraph_denote, sumSquares_permutation permutation x y h]

/-- Compare a mixed-precision example using exact binary words at every declared precision.

This runner checks the recorded example, not all inputs or XLA's reduction schedule.
The companion chooses exactly representable reduction inputs for these comparisons.
-/
def checkBinary (artifact : Lean.Json) : IO Unit := do
  let program ← IO.ofExcept (Import.JAX.parseProgram artifact)
  let payload ← IO.ofExcept (Import.JAX.parseBinaryPayload artifact)
  let words ← if program.inputs.size == 1 then
    pure #[← IO.ofExcept (artifact.getObjVal? "example_input_bits")]
  else
    IO.ofExcept ((← IO.ofExcept (artifact.getObjVal? "example_inputs_bits")).getArr?)
  unless words.size == program.inputs.size do
    throw (IO.userError "Jaxpr example: argument count mismatch")
  let inputs ← program.inputs.zip words |>.mapM fun (id, json) => do
    let node ← IO.ofExcept (program.graph.getNode id)
    let dtype ← IO.ofExcept (program.dtype id)
    let .binary format := dtype
      | throw (IO.userError "Jaxpr example: expected a binary input")
    let words ← IO.ofExcept (Import.PyTorch.TorchExport.Internal.nats "input.bits" json)
    IO.ofExcept (Interpreter.Value.binary format node.outShape words)
  let (outputs, flags) ← IO.ofExcept (Interpreter.binary.eval program payload inputs)
  let expected ← IO.ofExcept
    ((← IO.ofExcept (artifact.getObjVal? "example_outputs_bits")).getArr?)
  unless outputs.size == expected.size do
    throw (IO.userError "Jaxpr example: output count mismatch")
  for i in [:outputs.size] do
    let some output := outputs[i]?
      | throw (IO.userError "Jaxpr example: missing output")
    let some referenceJson := expected[i]?
      | throw (IO.userError "Jaxpr example: missing reference")
    let actual ← IO.ofExcept output.words
    let reference ← IO.ofExcept
      (Import.PyTorch.TorchExport.Internal.nats "output.bits" referenceJson)
    unless actual == reference do
      throw (IO.userError s!"Jaxpr example: result {i} differs in its interchange words")
    IO.println s!"  output {i}: {output.dtype}, bit-for-bit match for this input"
  IO.println s!"  conversion flags: {repr flags}"
  IO.println s!"  checked {program.graph.nodes.size} typed IR nodes"

/-- Check an artifact and compare all designated outputs using the binary64 diagnostic evaluator. -/
def check (artifact : Lean.Json) : IO Unit := do
  let program ← IO.ofExcept (Import.JAX.parseProgram artifact)
  let some dtype := program.dtypes[0]?
    | throw (IO.userError "Jaxpr example: empty type assignment")
  let mixed := !(program.dtypes.all (· == dtype))
  let diagnosticTypes := #[DType.float16, DType.bfloat16, DType.float32, DType.float64]
  if program.inputs.size != 1 || mixed || !diagnosticTypes.contains dtype then
    checkBinary artifact
    return
  let captured ← IO.ofExcept (Import.JAX.parseGraph artifact)
  let payload ← IO.ofExcept (Import.JAX.parsePayload artifact)
  let inputNode ← IO.ofExcept (captured.graph.getNode captured.inputId)
  let inputJson ← IO.ofExcept (artifact.getObjVal? "example_input")
  let some input := Import.PyTorch.parseTensor inputNode.outShape inputJson
    | throw (IO.userError "Jaxpr example: malformed input tensor")
  let expected ← IO.ofExcept ((← IO.ofExcept (artifact.getObjVal? "example_outputs")).getArr?)
  unless expected.size == captured.outputIds.size do
    throw (IO.userError "Jaxpr example: output count mismatch")
  for i in [:captured.outputIds.size] do
    let some outputId := captured.outputIds[i]?
      | throw (IO.userError "Jaxpr example: missing output id")
    let node ← IO.ofExcept (captured.graph.getNode outputId)
    let actual ← IO.ofExcept
      (captured.graph.denote payload (Spec.SomeTensor.ofTensor input) outputId)
    let actual ← IO.ofExcept (Graph.expectShape node.outShape actual)
    let some referenceJson := expected[i]?
      | throw (IO.userError "Jaxpr example: missing reference output")
    let some reference := Import.PyTorch.parseTensor node.outShape referenceJson
      | throw (IO.userError "Jaxpr example: malformed reference output")
    let error := Tensor.maxAbsDiff actual reference
    unless error <= 2e-5 do
      throw (IO.userError s!"Jaxpr example: output {i} differs by {error}")
    IO.println s!"  output {i}: shape {repr node.outShape}, maximum error {error}"
  -- Matching the complete graph connects this artifact to the general theorem above.
  let expectedNodes := (sumSquaresGraph inputNode.outShape).nodes
  let sameGraph := captured.graph.nodes.size == 3 && captured.inputId == 0 &&
    captured.outputIds == #[2] && (captured.graph.nodes.zip expectedNodes).all fun (a, b) =>
      a.id == b.id && a.parents == b.parents && a.outShape == b.outShape &&
        a.kind.opTag == b.kind.opTag
  if sameGraph && inputNode.outShape.rank == 1 then
    IO.println "  matches the graph covered by sumSquaresGraph_permutation"
  IO.println s!"  checked {captured.graph.nodes.size} IR nodes"

/-- Read one or more artifacts produced by the Python companion. -/
def run (paths : List String) : IO Unit := do
  if paths.isEmpty then
    throw (IO.userError "usage: torchlean jaxpr <artifact.json> ...")
  for path in paths do
    IO.println path
    let artifact ← IO.ofExcept (Lean.Json.parse (← IO.FS.readFile path))
    check artifact

end NN.Examples.Interop.JAX
