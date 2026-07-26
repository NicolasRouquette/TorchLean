import Verso
import VersoManual
import VersoBlueprint
import NN.Spec.Core.Context
import NN.Spec.Core.Tensor
import NN.Spec.Core.TensorReductionShape
import NN.Proofs.Tensor.Algebra
import NN.GraphSpec
import NN.GraphSpec.Models.MlpDeterministicInit
import NN.GraphSpec.Models.MlpSpecEquivalence
import NN.IR

open Verso.Genre
open Verso.Genre.Manual
open Informal

#doc (Manual) "Tensors and Graphs" =>

TorchLean starts with shape-indexed tensors and then offers three ways to describe a computation.
`GraphSpec` is a typed sequential architecture, `GraphSpec.DAG` adds sharing, and `NN.IR.Graph` is
the ordinary op-tagged graph used at runtime boundaries. Keeping those representations distinct
makes the claims attached to each one easier to read.

:::group "tensor_foundations"
Scalar operations, tensor shapes, and the algebra used by model specifications.
:::

:::definition "scalar_context" (parent := "tensor_foundations") (lean := "Context")
`Context α` collects the arithmetic, order, constants, and transcendental operations needed by
scalar-polymorphic model code.
:::

:::definition "shape_indexed_tensors" (parent := "tensor_foundations") (lean := "Spec.Tensor")
`Spec.Shape` indexes `Spec.Tensor α s`, so the dimensions of a pure tensor are present in its type.
:::

:::definition "shape_well_formedness" (parent := "tensor_foundations") (lean := "Spec.Shape.wellFormed")
A shape is well formed when every dimension in its tree is positive.
:::

:::definition "broadcast_compatibility" (parent := "tensor_foundations") (lean := "Spec.Shape.CanBroadcastTo")
`CanBroadcastTo s₁ s₂` records when values with shape `s₁` can be expanded to shape `s₂`.
:::

:::theorem "well_formed_shape_has_elements" (parent := "tensor_foundations") (lean := "Spec.Shape.size_pos_of_well_formed")
A {uses "shape_well_formedness"}[well-formed shape] has positive total size.
:::

:::proof "well_formed_shape_has_elements"
The proof follows the {uses "shape_well_formedness"}[shape tree] and uses positivity of every
dimension.
:::

:::theorem "flatten_round_trip" (parent := "tensor_foundations") (lean := "Spec.Tensor.flatten_unflatten_inverse")
Rebuilding a flattened {uses "shape_indexed_tensors"}[shape-indexed tensor] at its original shape
returns that tensor.
:::

:::proof "flatten_round_trip"
Induction on {uses "shape_indexed_tensors"}[the tensor shape] follows the same scalar and dimension
structure used by flattening and rebuilding.
:::

:::theorem "tensor_linear_adjointness" (parent := "tensor_foundations") (lean := "Proofs.TensorAlgebra.dot_mat_linear_adjoint")
Matrix-vector multiplication satisfies the dot-product adjoint identity used by the linear-layer
gradient rule for {uses "shape_indexed_tensors"}[typed tensors] over a commutative semiring.
:::

:::proof "tensor_linear_adjointness"
The proof expands {uses "shape_indexed_tensors"}[the typed tensor operations] into finite sums and
rearranges those sums.
:::

:::group "graph_representations"
Typed architecture descriptions and the shared runtime graph.
:::

:::definition "graphspec_syntax" (parent := "graph_representations") (lean := "NN.GraphSpec.Graph")
A sequential `GraphSpec` records its parameter shapes, input shape, and output shape in its type.
Graphs are built from identity, primitive, and sequential-composition nodes.
:::

:::definition "graphspec_pure_semantics" (parent := "graph_representations") (lean := "NN.GraphSpec.Interp.spec")
The pure interpreter evaluates {uses "graphspec_syntax"}[a sequential graph] on
{uses "shape_indexed_tensors"}[typed parameter and input tensors] using
{uses "scalar_context"}[scalar-polymorphic operations].
:::

:::definition "graphspec_runtime_translation" (parent := "graph_representations") (lean := "NN.GraphSpec.Compile.torchProgram")
The runtime translation turns {uses "graphspec_syntax"}[a sequential graph] into a
backend-polymorphic TorchLean program over {uses "scalar_context"}[the same scalar operations].
:::

:::definition "typed_dag_syntax" (parent := "graph_representations") (lean := "NN.GraphSpec.DAG.Model")
A typed DAG model pairs initialized parameters with a term whose environment contains the parameter
and input shapes. Terms may name arguments and share intermediate results.
:::

:::definition "typed_dag_pure_semantics" (parent := "graph_representations") (lean := "NN.GraphSpec.DAG.Model.specFwd")
The DAG interpreter evaluates {uses "typed_dag_syntax"}[the model body] from typed parameter and
input lists under {uses "scalar_context"}[the scalar context].
:::

:::definition "typed_dag_runtime_translation" (parent := "graph_representations") (lean := "NN.GraphSpec.DAG.Model.torchProgram")
The DAG compiler turns {uses "typed_dag_syntax"}[the same model body] into a backend-polymorphic
TorchLean program over {uses "scalar_context"}[the same scalar operations].
:::

:::definition "graphspec_mlp_model" (parent := "graph_representations") (lean := "NN.GraphSpec.Models.mlp")
The {uses "graphspec_syntax"}[sequential GraphSpec] MLP composes two linear maps with an intervening
ReLU. Its type fixes the order and shapes of both weights and biases.
:::

:::theorem "graphspec_mlp_spec_alignment" (parent := "graph_representations") (lean := "NN.GraphSpec.Models.mlp_interp_eq_spec_mlp_forward")
Interpreting the {uses "graphspec_mlp_model"}[GraphSpec MLP] gives the same tensor as the
hand-written two-layer MLP specification.
:::

:::proof "graphspec_mlp_spec_alignment"
The proof unpacks the four-tensor parameter list, unfolds
{uses "graphspec_pure_semantics"}[the pure interpreter] for
{uses "graphspec_mlp_model"}[the MLP], and reduces both sides to the same two linear maps with an
intervening ReLU.
:::

:::theorem "graphspec_mlp_initialization_alignment" (parent := "graph_representations") (lean := "NN.GraphSpec.Models.mlp_detInitParams_eq_torchlean_linear_inits")
Deterministic initialization for the {uses "graphspec_mlp_model"}[GraphSpec MLP] produces the same
typed parameter list, in the same order, as the two TorchLean linear-layer initializers.
:::

:::proof "graphspec_mlp_initialization_alignment"
For {uses "graphspec_mlp_model"}[the two-layer graph], the occurrence-indexed seed calculation
reduces to seeds `0, 1` for the first layer and `2, 3` for the second.
:::

:::definition "ir_structural_predicate" (parent := "graph_representations") (lean := "NN.IR.Graph.wellFormed")
The Boolean structural predicate checks that node identifiers match their array positions,
operation arities are valid, and every parent points to an earlier node.
:::

:::definition "ir_structural_validation" (parent := "graph_representations") (lean := "NN.IR.Graph.checkWellFormed")
The diagnostic checker enforces the same conditions as the
{bpref "ir_structural_predicate"}[Boolean structural predicate], but returns the first useful error
message instead of a bare `false`. The two implementations are kept separate; no equivalence
theorem currently connects them.
:::

:::definition "ir_shape_validation" (parent := "graph_representations") (lean := "NN.IR.Graph.checkShapes")
After {uses "ir_structural_validation"}[the structural check], shape validation infers every node's
output shape and compares it with the shape stored in the graph.
:::

:::definition "ir_denotation" (parent := "graph_representations") (lean := "NN.IR.Graph.denoteAll")
After {uses "ir_structural_validation"}[the structural check], IR denotation evaluates every node
into a table of {uses "shape_indexed_tensors"}[shape-tagged tensor values] using
{uses "scalar_context"}[scalar-polymorphic operations] and the supplied external payload.
:::
