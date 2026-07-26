import Verso
import VersoManual
import VersoBlueprint
import NN.Runtime.Autograd.Engine
import NN.Runtime.Autograd.Compiled
import NN.Runtime.Autograd.Compiled.IRExec.Correctness.SemanticEquivalence
import NN.Runtime.Autograd.Torch
import NN.Runtime.Autograd.Torch.LinkedSession.Public
import NN.Runtime.Autograd.TorchLean
import NN.Runtime.Autograd.TorchLean.CompileExec
import NN.Proofs.Autograd.Runtime.Link.BackwardGraphData
import NN.Verification.TorchLean.Proved.Public

open Verso.Genre
open Verso.Genre.Manual
open Informal

#doc (Manual) "Autograd and Execution" =>

The runtime has a dynamic tape for eager execution and a typed SSA builder for compiled programs.
Both sit below the layer API. A separate bridge executes the supported fragment of `NN.IR.Graph`;
its correctness theorem states the fragment explicitly.

:::group "autograd_execution"
Programs, tapes, and compiled execution.
:::

:::definition "runtime_ops_program" (parent := "autograd_execution") (lean := "Runtime.Autograd.TorchLean.Program")
The operation interface describes programs over
{uses "shape_indexed_tensors"}[shape-indexed tensors] and
{uses "scalar_context"}[scalar operations] without choosing eager or compiled execution at the
call site.
:::

:::definition "runtime_autograd_tape" (parent := "autograd_execution") (lean := "Runtime.Autograd.Tape")
The CPU tape is a dynamic computation DAG. It stores shape-erased values and accumulates
reverse-mode gradients by node index.
:::

:::definition "runtime_compiled_autograd" (parent := "autograd_execution") (lean := "Runtime.Autograd.Compiled.compile")
`compile` takes executable typed graph data and its input context, then returns a
{uses "runtime_autograd_tape"}[runtime tape] together with the typed list of inputs and intermediate
values.
:::

:::theorem "compiled_graph_backward_agreement" (parent := "autograd_execution") (lean := "Proofs.Autograd.Algebra.Graph.backwardDenseFrom_compileAuxData_eq_backpropAllCtx")
For executable typed graph data over a commutative semiring, dense reverse accumulation on the
{uses "runtime_autograd_tape"}[compiled tape] equals proof-level graph backpropagation after both
contexts are converted to the tape's value array.
:::

:::proof "compiled_graph_backward_agreement"
Induction over the typed graph keeps the forward context and tape indices aligned while the reverse
loop accumulates each node's contribution.
:::

:::theorem "runtime_compiled_backprop_link" (parent := "autograd_execution") (lean := "Runtime.Autograd.Torch.backwardDenseFrom_compileAuxData_eq_backpropAllCtx")
For a linked-session snapshot, dense reverse accumulation on
{uses "runtime_autograd_tape"}[the compiled tape] agrees with proof-level graph backpropagation.
:::

:::proof "runtime_compiled_backprop_link"
The linked-session hook specializes
{uses "compiled_graph_backward_agreement"}[the graph-data backward theorem] to the session's graph,
inputs, and auxiliary index environment.
:::

:::definition "runtime_layer_model" (parent := "autograd_execution") (lean := "Runtime.Autograd.TorchLean.NN.Seq")
Layer definitions carry parameter and buffer initialization, training or evaluation mode, and
shape-checked sequential composition. They build their computations through
{uses "runtime_ops_program"}[the runtime operation interface].
:::

:::group "shared_ir_runtime"
The executable path from the shared graph IR.
:::

:::definition "shared_ir_execution" (parent := "shared_ir_runtime") (lean := "Runtime.Autograd.Compiled.execGraphOfIR")
After {uses "ir_structural_validation"}[structural validation], the supported shared IR operations
are lowered to a typed executable graph for forward evaluation.
:::

:::theorem "shared_ir_execution_correctness" (parent := "shared_ir_runtime") (lean := "Runtime.Autograd.Compiled.execGraphOfIR_semantics_eq")
{uses "shared_ir_execution"}[Successful compilation] agrees with
{uses "ir_denotation"}[IR denotation] when the graph satisfies `NoMSELoss`, `NoRawLog`, and
`NoConcat`.
:::

:::proof "shared_ir_execution_correctness"
After unfolding {uses "shared_ir_execution"}[the IR compiler], the proof peels off
{uses "ir_structural_validation"}[the structural check] and input node. The recursive lowering
invariant then matches each compiled value with {uses "ir_denotation"}[the corresponding
denotational value].
:::
