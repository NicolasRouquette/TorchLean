import VersoManual

open Verso.Genre Manual

#doc (Manual) "The Canonical Graph IR" =>
%%%
tag := "graphs-and-ir"
%%%

GraphSpec preserves an architecture as a typed Lean term. A backend pass needs a more ordinary
piece of data: an array of operation nodes that can be imported, traversed, serialized, checked, and
assigned to kernels. That representation is `NN.IR.Graph`.

The easiest way to understand it is to build one. Consider

$$`
G(x)=\tanh\!\left(
  \sum_k
  \left[W_2\operatorname{ReLU}(W_1x+b_1)+b_2\right]_k
\right).
`

TorchLean stores this as six nodes:

```
0  input   []    [4]
1  linear  [0]   [5]
2  relu    [1]   [5]
3  linear  [2]   [3]
4  sum     [3]   scalar
5  tanh    [4]   scalar
```

The parent list gives data dependencies. Because every parent ID is smaller than the node ID, the
array order is already a topological execution order.

# Build The Six Nodes

The executable deep dive constructs the graph directly:

```
import NN.IR.Graph

open NN.IR
open Spec

def xShape : Shape := shape![4]
def hShape : Shape := shape![5]
def yShape : Shape := shape![3]

def graph : Graph :=
  let n0 : Node :=
    { id := 0, parents := [],  kind := .input,  outShape := xShape }
  let n1 : Node :=
    { id := 1, parents := [0], kind := .linear, outShape := hShape }
  let n2 : Node :=
    { id := 2, parents := [1], kind := .relu,   outShape := hShape }
  let n3 : Node :=
    { id := 3, parents := [2], kind := .linear, outShape := yShape }
  let n4 : Node :=
    { id := 4, parents := [3], kind := .sum,    outShape := .scalar }
  let n5 : Node :=
    { id := 5, parents := [4], kind := .tanh,   outShape := .scalar }
  { nodes := #[n0, n1, n2, n3, n4, n5] }
```

Unlike GraphSpec, this datatype does not make every edge shape-correct by construction. Node IDs,
raw axis numbers, and declared output shapes are ordinary data. That is deliberate: an importer
must be able to construct a candidate graph from an external document before Lean knows it is
valid.

The price of an import-friendly representation is an explicit validation phase.

# Structure And Shape Are Separate Checks

`Graph.checkWellFormed` checks graph structure:

- node ID equals its array position;
- parents occur earlier in topological order;
- the operation has an admissible number of parents;
- designated input and constant nodes have the required arity.

`Graph.checkShapes` follows with operation-specific shape rules. For the graph above it checks, among
other things, that:

- the payload for node 1 accepts `[4]` and produces `[5]`;
- ReLU preserves `[5]`;
- the payload for node 3 accepts `[5]` and produces `[3]`;
- `sum` produces a scalar;
- `tanh` preserves that scalar.

Try changing only node 2's declared output from `[5]` to `[4]`. The graph remains topologically
well formed, but the shape checker rejects the edge into the second linear layer. This is why a
claim that an imported graph is “validated” should name both checks.

The distinction also matters in the backend adapter:
`NN.Backend.IR.checkedPlanGraphNodesWithRegistry` currently calls `checkWellFormed` before planning,
but it does not replace an importer's shape check. A caller accepting untrusted graph data must not
infer shape validity from a successful plan alone.

# Parameters Live In A Payload

The two `.linear` nodes mention only their activation parent. Their weights and biases are stored in
a payload keyed by node ID:

```
def payload {α : Type} [Context α] (p : Params α) : Payload α :=
  { linear? := fun id =>
      if id = 1 then
        some { outDim := 5, inDim := 4,
               W := p.hiddenWeight, b := p.hiddenBias }
      else if id = 3 then
        some { outDim := 3, inDim := 5,
               W := p.outputWeight, b := p.outputBias }
      else
        none }
```

Separating structure from values has practical consequences:

- one graph can be reused with initial, trained, or bounded parameters;
- checkpoint loading changes the payload without rebuilding the node array;
- a verifier can replace concrete parameters by interval metadata;
- an exporter can emit graph nodes and initializers through different channels.

It also creates an ABI obligation. If the payload at node 1 contains a `[3, 4]` weight while the
node claims output shape `[5]`, the graph structure is unchanged but evaluation must fail. A
payload is not trusted merely because its key exists.

The shared `NN.IR.Payload` currently has typed records for constants, linear weights and bias,
convolution parameters, and eval-mode NCHW BatchNorm parameters. Other operations obtain their
values from parent edges. Adding a new payload-backed operation requires coordinated changes to the
payload type, shape inference, denotation, import/export adapters, and every runtime or verifier
that claims to support it.

# A Heterogeneous Value Table

During evaluation, node 0 stores a vector, nodes 1 and 2 store `[5]`, node 3 stores `[3]`, and nodes
4 and 5 store scalars. One homogeneous Lean array cannot directly contain all those tensor types.

The evaluator uses

```
DVal α = Σ s : Shape, Spec.Tensor α s
```

a dependent pair of a runtime shape tag and a tensor with exactly that shape. The table can hold
`DVal α` values of different shapes, while `Graph.expectShape` recovers a statically typed tensor
after checking the tag.

For node 1, evaluation performs:

1. fetch parent 0 from the value table;
2. fetch the linear payload keyed by `1`;
3. check the parent tag equals `[4]`;
4. check the declared output equals `[5]`;
5. call the pure `linearSpec`;
6. store the result as `DVal α`.

Failures are reported as `Except String`; malformed imported data does not receive a fabricated
proof cast.

# Run One Graph Under Several Semantics

The full example is executable:

```
lake exe torchlean one_semantic_universe --samples 50
```

It prints:

```
== One semantic universe tutorial ==
graph nodes = 6
[eval IEEE32Exec] y(x0) = 0.027713
[IBP IEEE endpoints] lo = 0.020772
[IBP IEEE endpoints] hi = 0.035625
consistency: 50/50 samples satisfied evalIEEE(x) ∈ IBP(B)
checker theorem: `NN.MLTheory.CROWN.Box.containsDecBool_sound`
```

This command demonstrates three different statements:

1. the graph evaluator produced one binary32 result at the center input;
2. interval bound propagation produced one output interval for an input box;
3. fifty sampled evaluations happened to fall inside that interval.

Only the named checker theorem turns a successful Boolean containment check into a proposition
about that checked point. Fifty samples are a regression experiment, not the universal IBP
soundness theorem. The verification chapters identify the additional theorem needed to conclude
that every input in the box is enclosed.

# Axis Operations Expose Partial Coverage

Run:

```
lake exe torchlean ir_axis_ops
```

The example checks softmax, layer normalization, and concatenation on rank-three tensors. For
concatenation it reports the same output shape and leading values from both the pure spec evaluator
and the compiled IR path:

```
concat axis=1: [2,3,4] ++ [2,5,4] -> [2,8,4]
spec outShape:     [2,8,4]
compiled outShape: [2,8,4]
```

For the current middle-axis softmax and layer-normalization cases, the spec evaluator runs while
the compiled path explicitly reports that the case is unsupported. This is preferable to silently
changing the axis or falling back to a different meaning.

Change `concat axis=1` to an out-of-range axis in the source
[`IRAxisOps.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/DeepDives/IRAxisOps.lean).
Shape inference rejects the node before evaluation.

# Executable IR Coverage And Proof Coverage

`Runtime.Autograd.Compiled.IRExec` lowers the current IR vocabulary operation by operation,
including elementwise arithmetic, seeded masks, broadcasting, reductions, rank-two and limited
rank-three matmul, linear and convolution payloads, pooling, normalization, reshape/permutation,
concat, and scalar MSE. Lowering remains allowed to reject a shape or axis that its runtime builder
cannot represent.

The recursive semantic-equivalence theorem is narrower than the executable lowering. Its named
side conditions are `NoRawLog`, `NoMSELoss`, and `NoConcat`. Per-operation lemmas live under
`NN.Runtime.Autograd.Compiled.IRExec.Correctness.Ops`; the end-to-end theorem is in
`Correctness.SemanticEquivalence`. This separation lets the executable path grow without silently
expanding the theorem statement.

# Pure Denotation

`NN.IR.Graph.denote` folds over the node array using the spec operations and a scalar `Context α`.
The same structural graph can therefore be interpreted at:

- `α := ℝ`, for exact-real theorem statements;
- `α := FP32`, for finite rounded-real analysis;
- `α := IEEE32Exec`, for executable binary32 behavior;
- interval endpoints, for bound propagation.

The graph is the same data, but the meaning of arithmetic changes with `α`. The equality of two
interpretations is never automatic.

`DVal α` varies the shape while retaining the scalar contract introduced in *Tensors And Shapes*.
The PyTorch comparison discusses the separate mixed-precision question.

For the six-node example:

```
Graph.denote (α := ℝ)          graph payloadReal inputReal
Graph.denote (α := FP32)       graph payloadFP32 inputFP32
Graph.denote (α := IEEE32Exec) graph payloadIEEE inputIEEE
```

have the same node structure and different scalar semantics. A runtime-approximation theorem must
relate their inputs, parameters, and operations before it can bound the final outputs.

# Compiler Claims Have A Fragment

The proof-bearing compiler under `NN.Verification.TorchLean.Proved` relates supported compiled
forward evaluation to IR denotation. Its theorem is not a wildcard over every `OpKind`.

Other compiled bridges have side conditions such as excluding raw logarithm or MSE nodes. Those
conditions are mathematically meaningful:

- `log` needs a domain and numerical policy;
- an MSE node may combine reduction and loss conventions not yet covered by a compiler proof.

When a compiler returns an executable object, ask two separate questions:

1. did lowering succeed for this concrete graph?
2. which theorem covers the operations and side conditions in that graph?

Successful compilation without the second answer is an execution result, not a semantic proof.

# Backend Planning Does Not Execute

The backend adapter maps operation tags to backend operations:

```
.linear  ↦ BackendOp.linear
.relu    ↦ BackendOp.relu
.sum     ↦ BackendOp.reduceSum
.tanh    ↦ BackendOp.tanh
```

It then chooses an admissible kernel capsule for each runtime-relevant node. A plan preserves node
IDs and records capsule names in graph order.

This is useful audit data, but a `KernelCapsule` is a contract descriptor, not a closure containing
machine code. Planning node 1 for `nativeCuda` does not call a CUDA kernel. Provider-specific eager
or compiled runtime code must interpret that choice.

The distinction prevents a common architecture mistake:

```
registered     ≠ selectable
selectable     ≠ executable
executable     ≠ proved correct
```

Each arrow has its own availability check, dispatcher, and evidence.

# How This Differs From The Autograd Tape

The canonical IR records a persistent model computation. An eager autograd tape records one
execution:

- concrete runtime tensor handles;
- which values require gradients;
- saved forward values needed by VJPs;
- the actual order in which wrappers ran.

The tape may contain enough information to reconstruct an IR-like graph, but it is not
`NN.IR.Graph`. Conversely, the canonical IR does not own mutable gradient buffers or optimizer
state.

This difference is why LibTorch forward can still participate in a TorchLean-owned backward path:
the TorchLean wrapper records a local tape node even when an external provider computes the forward
value. The canonical semantic graph and the execution tape remain distinct objects connected by
the operation contract.

# Reading A Graph Result

For any graph-based claim, locate:

1. the exact graph representation;
2. its concrete parameter payload;
3. the structural and shape checks that ran;
4. the scalar context used by denotation;
5. the compiler theorem and fragment, if lowering was used;
6. the capsule selected for each native operation;
7. the provider branch that actually executed it.

The source map is:

- [`NN.IR.Graph`](https://github.com/lean-dojo/TorchLean/blob/main/NN/IR/Graph.lean) for nodes and
  operation tags;
- [`NN.IR.Check`](https://github.com/lean-dojo/TorchLean/blob/main/NN/IR/Check.lean) for structural
  and shape validation;
- [`NN.IR.Semantics`](https://github.com/lean-dojo/TorchLean/blob/main/NN/IR/Semantics.lean) for
  pure denotation;
- [`NN.Backend.IR`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Backend/IR.lean) for capsule
  planning;
- [`OneSemanticUniverse.lean`](https://github.com/lean-dojo/TorchLean/blob/main/NN/Examples/DeepDives/OneSemanticUniverse.lean)
  for the complete six-node experiment.
