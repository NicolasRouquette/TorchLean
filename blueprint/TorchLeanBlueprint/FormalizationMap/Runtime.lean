import Verso
import VersoManual
import VersoBlueprint
import NN.Runtime.Autograd.Engine
import NN.Runtime.Autograd.TypedGraph
import NN.Runtime.Autograd.IRExec.Correctness.SemanticEquivalence
import NN.Runtime.Autograd.Torch
import NN.Runtime.Autograd.Torch.TypedGraphSession
import NN.Runtime.Autograd.TorchLean
import NN.Proofs.Autograd.Runtime.Link.BackwardGraphData
import NN.Verification.TorchLean.Proved.Correctness

open Verso.Genre
open Verso.Genre.Manual
open Informal

#doc (Manual) "Autograd and Execution" =>

The runtime has a dynamic tape for eager execution and a typed SSA builder for typed graph execution.
Both sit below the layer API. A separate bridge executes the supported fragment of `NN.IR.Graph`;
its correctness theorem states the fragment explicitly.

:::group "autograd_execution"
Programs, tapes, and typed graph execution.
:::

:::definition "runtime_ops_program" (parent := "autograd_execution") (lean := "Runtime.Autograd.TorchLean.Program")
The operation interface describes programs over
{uses "shape_indexed_tensors"}[shape-indexed tensors] and
{uses "scalar_context"}[scalar operations] without choosing eager or typed graph execution at the
call site.
:::

:::definition "runtime_autograd_tape" (parent := "autograd_execution") (lean := "Runtime.Autograd.Tape")
The CPU tape is a dynamic computation DAG. It stores shape-erased values and accumulates
reverse-mode gradients by node index.
:::

:::definition "runtime_typed_graph_autograd" (parent := "autograd_execution") (lean := "Runtime.Autograd.TypedGraph.lowerToTape")
`lowerToTape` takes executable typed graph data and its input context, then returns a
{uses "runtime_autograd_tape"}[runtime tape] together with the typed list of inputs and intermediate
values.
:::

:::theorem "typed_graph_backward_agreement" (parent := "autograd_execution") (lean := "Proofs.Autograd.Algebra.Graph.backwardDenseFrom_lowerGraphDataToTape_eq_backpropAllCtx")
For executable typed graph data over a commutative semiring, dense reverse accumulation on the
{uses "runtime_autograd_tape"}[lowered tape] equals proof-level graph backpropagation after both
contexts are converted to the tape's value array.
:::

:::proof "typed_graph_backward_agreement"
Induction over the typed graph keeps the forward context and tape indices aligned while the reverse
loop accumulates each node's contribution.
:::

:::theorem "typed_graph_output_backward_agreement" (parent := "autograd_execution") (lean := "Runtime.Autograd.TypedGraph.backwardDenseAllFrom_lowerToTape_eq_backpropAllCtx")
For any typed output reference, including an input or intermediate node, lowering to the runtime
tape and running reverse mode agrees with seeding that same output in executable graph
backpropagation.
:::

:::proof "typed_graph_output_backward_agreement"
The general graph-data lowering theorem is instantiated with a seed context containing the supplied
cotangent at exactly the selected output reference.
:::

:::theorem "runtime_typed_graph_backprop_link" (parent := "autograd_execution") (lean := "Runtime.Autograd.Torch.Internal.TypedGraphSession.backwardDenseFrom_lowerGraphDataToTape_eq_backpropAllCtx")
For the executable graph data in a typed-graph session snapshot, dense reverse accumulation on
{uses "runtime_autograd_tape"}[the lowered tape] agrees with graph backpropagation. Runtime leaf
names and `requiresGrad` masks are attached after this raw graph-data lowering theorem.
:::

:::proof "runtime_typed_graph_backprop_link"
The typed-graph session theorem specializes
{uses "typed_graph_backward_agreement"}[the graph-data backward theorem] to the session's graph,
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

:::definition "shared_ir_execution" (parent := "shared_ir_runtime") (lean := "Runtime.Autograd.IRExec.lowerToForwardGraph")
After {uses "ir_structural_validation"}[structural validation], the supported shared IR operations
are lowered to a shape-indexed `ForwardGraph` for forward evaluation. Its `ForwardData` contains
only forward closures: there are no JVP or VJP fields to call. It is distinct from the reusable
autograd `Torch.TypedGraph`.
:::

:::theorem "shared_ir_execution_correctness" (parent := "shared_ir_runtime") (lean := "Runtime.Autograd.IRExec.denoteAll_eq_of_lowerToForwardGraph")
{uses "shared_ir_execution"}[Successful lowering] agrees with
{uses "ir_denotation"}[IR denotation] when the graph satisfies `NoMSELoss`, `NoRawLog`, and
`NoConcat`.
:::

:::proof "shared_ir_execution_correctness"
After unfolding {uses "shared_ir_execution"}[IR lowering], the proof peels off
{uses "ir_structural_validation"}[the structural check] and input node. The recursive lowering
invariant then matches each forward-graph value with {uses "ir_denotation"}[the corresponding
denotational value].
:::
