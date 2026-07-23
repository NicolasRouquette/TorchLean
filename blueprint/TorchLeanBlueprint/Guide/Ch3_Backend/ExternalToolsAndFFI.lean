import VersoManual

open Verso.Genre Manual

#doc (Manual) "Crossing Lean's Boundary" =>
%%%
tag := "external-tools-and-ffi"
%%%

PyTorch can capture a model, an interval library can propose a numerical enclosure, and a CUDA
kernel can compute a tensor. TorchLean records what Lean learns when each external program returns.

There are two main boundaries:

- a *subprocess* exchanges files, standard output, or JSON with another executable;
- an *FFI call* invokes a linked native symbol and may exchange opaque memory handles.

Both can be used with explicit validation and named trust boundaries. They have different failure
modes and support different proof stories.

# A Subprocess Is An Untrusted Producer

The common process helper is small:

```
def runJsonStdoutChecked
    (ctx : String)
    (cmd : String)
    (args : Array String)
    (cwd : Option String := some ".") :
    IO Json := do
  let stdout ← runStdoutChecked ctx cmd args cwd
  match Json.parse stdout with
  | .ok value => pure value
  | .error message =>
      throw <| IO.userError
        s!"{ctx}: JSON parse error: {message}\nstdout:\n{stdout}"
```

`runStdoutChecked` starts the process, captures its streams, and rejects a nonzero exit code with the
command, arguments, status, and standard error in the diagnostic. `runJsonStdoutChecked` then
requires all of standard output to be one JSON document.

Suppose Python prints:

```
{"format":"torchlean.bound.v1","lower":0.12,"upper":0.31}
```

Successful parsing establishes only that this text is valid JSON. A useful checker must still:

1. require the exact `format` string;
2. require finite numeric fields;
3. establish `lower ≤ upper`;
4. connect the interval to a particular graph, payload, input set, and output;
5. invoke a sound acceptance theorem.

Changing `upper` to `1e999` is a good boundary test. JSON syntax accepts the number, but converting
it to a machine `Float` can produce infinity. Verification parsers therefore use
`expectFiniteFloatE` or `expectFieldFiniteFloatE`, not the permissive float parser, whenever the
certificate schema promises finite claims.

# A Complete PyTorch Capture

Run:

```
lake exe torchlean pytorch_export_check
```

The command asks Python and `torch.export` to capture several small `nn.Module`s, emits
`torchlean.ir.v1` JSON, parses each document in Lean, lowers supported values to `NN.IR.Graph`, and
runs the graph validators.

The current run accepts:

```
TinyAddRelu                 nodes=3
TinyMLP                     nodes=4
TinyCheckpointMLP           nodes=4
TinyCNN                     nodes=4
TinyCNNHead                 nodes=6
TinyBatchNorm2d             nodes=2
TinyNormSoftmax             nodes=3
TinyTransformerishBlock     nodes=7
TinySelfAttentionOps        nodes=5
TinySingleHeadMHA           nodes=11
```

Every accepted line reports:

```
guarantee: WellShaped via parseGraph_wellShaped
```

The wording is precise. The parser theorem says that successful parsing yields a graph satisfying
the executable well-shaped predicate. It does not say that Python captured the intended module or
that every PyTorch operator has been translated correctly.

# Inspect The Exchange Format

A minimal captured graph looks like:

```
{
  "format": "torchlean.ir.v1",
  "input_id": 0,
  "output_id": 2,
  "nodes": [
    {"id": 0, "kind": "input", "parents": [], "shape": [1, 4]},
    {"id": 1, "kind": "relu",  "parents": [0], "shape": [1, 4]},
    {"id": 2, "kind": "sum",   "parents": [1], "shape": []}
  ]
}
```

The Python producer is responsible for translating raw FX or ATen names into stable TorchLean tags.
The Lean parser does not accept arbitrary operator strings and guess their meaning. It recognizes a
conservative list, parses operation-specific fields, constructs candidate nodes, and validates the
result.

This division is intentional:

```
PyTorch module
   ↓ external capture, treated as an untrusted producer
FX/value graph
   ↓ explicit JSON schema
Lean parser
   ↓ checked structural lowering
NN.IR.Graph
   ↓ denotation / verifier / exporter
TorchLean analysis
```

The container-valued FX layer matters. `nn.MultiheadAttention` returns a tuple of attention output
and attention weights. Treating every FX node as a tensor loses that fact. TorchLean first retains
tuple shape metadata, then lowers only supported tensor projections. The producer is free to do a
complicated capture; the Lean side believes only the smaller document that its parser and graph
checks actually recognize.

# Rejection Is Part Of The Interface

The same command deliberately tries unsupported models.

`torch.sort(...).values` is rejected because the current producer has no lowering rule for that
tuple-valued operation. A two-head `nn.MultiheadAttention` is captured as a tuple-valued node but
then rejected with:

```
PyTorch graph import: node[2]:
`nn.MultiheadAttention` lowering supports only num_heads=1, got 2
```

This is useful behavior. Replacing the unsupported node by an identity or silently dropping the
attention weights would produce a valid-looking graph for the wrong model.

Try adding `torch.sort` to the Python example without changing the Lean parser. The expected result
is an explicit unsupported-operation failure, not partial import.

# A State Dictionary Is Not A Graph

A `state_dict` supplies named tensors. It does not describe data flow. A graph capture supplies
operations and edges. It may refer to parameters without carrying the full checkpoint provenance.

Round-trip import therefore has two obligations:

:::table +header
*
  * Artifact
  * Checks
*
  * graph
  * operator subset, IDs, parents, shapes, attributes, input/output IDs
*
  * state dictionary
  * key mapping, tensor shape, flat length, layout, finite values
:::

Run:

```
lake exe torchlean pytorch_roundtrip
```

This writes the generated MLP PyTorch artifacts under
`NN/Examples/Interop/PyTorch/MLP/`. Open the generated model and parameter files together: neither
one is a complete description of the executable network by itself.

# From Import Success To Semantic Equality

Three propositions are easy to conflate:

1. Python produced a document and exited successfully.
2. Lean parsed the document into a well-shaped supported graph.
3. The graph's denotation equals the original PyTorch module on all admissible inputs.

The capture experiment and `parseGraph_wellShaped` establish the second proposition, conditional on
the bytes received. The third needs a translation theorem for the supported producer or an explicit
trust assumption about capture and lowering.

Parity tests on random inputs are excellent engineering evidence for that assumption. They can
find transposes, axis errors, missing biases, and mask inversions. They do not universally quantify
over every parameter and input.

# ONNX Uses The Same Checked Ingress

`NN.Runtime.PyTorch.Export.ONNX` emits a Python adapter for a conservative static-shape ONNX
fragment. The producer handles elementwise operations, rank-two/limited batched matmul, reductions,
softmax, reshape/flatten, concat, selected transposes, `Gemm`, inference BatchNorm, and ungrouped,
undilated convolution with the layouts represented by the current IR. It writes the same
`torchlean.ir.v1` document consumed by `Import.PyTorch.TorchExport.parseGraph`.

ONNX protobuf parsing and shape inference therefore remain outside Lean. Lean checks the smaller IR
artifact it receives. Graph initializers and payload-backed nodes still need a matching payload
import; a well-shaped convolution node by itself does not contain authenticated weights.

# Native FFI Calls Have A Different Risk

A subprocess returns copied bytes owned by Lean after parsing. A native symbol can allocate,
mutate, alias, or free memory behind an opaque Lean value.

The CUDA buffer boundary contains declarations such as:

```
@[extern "torchlean_cuda_buffer_of_float_array_with_token"]
opaque ofFloatArrayWithToken
    (values : @& FloatArray) (token : UInt32) : Buffer

@[extern "torchlean_cuda_buffer_to_float_array_io"]
opaque toFloatArrayIO (buffer : @& Buffer) : IO FloatArray

@[extern "torchlean_cuda_buffer_release_with_token"]
opaque releaseWithToken
    (buffer : @& Buffer) (token : UInt32) : UInt32
```

The type `Buffer` is opaque. Lean code cannot forge its internal pointer, but the C implementation
still determines whether allocation, copying, finalization, and release are correct.

The shape-erased tape therefore does not trust a shape tag by itself. Before a buffer reaches a
shape-indexed kernel, `AnyBuffer.validate` checks that every axis and the total element count fit the
CUDA `UInt32` ABI and that the native buffer has exactly that many elements. Convolution and pooling
wrappers additionally reject zero strides and oversized dimension arrays, validate input, kernel,
and output spatial products independently, and check the returned native output length. Runtime
natural-number gather indices accept the full `UInt32` range and reject larger values before
conversion. These checks turn malformed runtime values into ordinary Lean errors; they do not prove
that a correctly sized native kernel implements its specification.

The `@&` annotation marks a borrowed Lean object. Native code may inspect such an argument during
the call but must not retain or decrement it. Native translation units repeat critical length and
geometry checks because an FFI declaration is still a trusted boundary. The file-to-symbol map is
documented in `NN.Runtime.Autograd.Engine.Cuda.NativeSources`.

# Why The IO Token Exists

An external declaration that appears pure may be common-subexpression eliminated or reused by Lean
as if equal arguments always denote the same value. Allocation does not have that semantics: two
uploads of the same host array should produce independently owned buffers.

The effectful wrapper obtains a changing monotonic-time token:

```
def ofFloatArrayIO (values : @& FloatArray) : IO Buffer := do
  let timestamp ← IO.monoNanosNow
  pure <| ofFloatArrayWithToken values (UInt32.ofNat timestamp)
```

The native function ignores the token numerically. Its presence makes each allocation depend on the
surrounding `IO` sequence, preventing Lean from treating repeated uploads as one pure object.

Release has the same issue. `releaseIO` uses a token and is called only at an ownership boundary
where no alias will be used again. The native finalizer remains safe after explicit release because
the implementation nulls the pointer.

A stale alias after release is a memory-safety bug that the tensor's shape type cannot detect.

# Workspaces And Backward

Some forward kernels produce an output plus intermediates needed by their VJP. TorchLean represents
that ownership as:

```
structure WithWorkspace where
  value : Buffer
  workspace : List Buffer := []
```

The tape node retains the workspace until backward has consumed it. Afterwards
`releaseWorkspaceThen` or `releaseAllThen` threads cleanup through a retained result, so the native
release cannot be erased as dead pure code.

For long training runs this prevents two forms of growth:

- GPU allocations waiting for Lean external-object finalizers;
- tape closures retaining workspaces after their VJP has run.

Allocator counters report live and peak bytes, allocation/free counts, wrapper counts, and device
free memory. They are observability tools, not a proof that no native leak exists.

# External Oracles And Certificates

TorchLean also calls arbitrary-precision or interval tools to propose bounds. A typical workflow is:

```
Lean writes an exact query
        ↓
Arb / python-flint computes an enclosure
        ↓
Lean parses rational midpoint-radius data
        ↓
a checker validates the enclosure or treats it as oracle evidence
```

Parsing exact rationals avoids an extra binary64 conversion at ingestion. It still does not verify
the external interval algorithm. If the result is replayed by a proved checker, checker acceptance
supports the checker's proposition. If it is only compared in a test, it remains an oracle.

The same producer/checker distinction applies to α,β-CROWN leaf dumps, PINN residual artifacts,
ODE enclosures, and geometry certificates. Search can be large and external; the accepted schema
and soundness theorem should stay small enough to audit.

# Evidence Is Field-Specific

A useful boundary report may say:

- *shape:* proved by a typed constructor;
- *layout:* guarded by length and row-major checks;
- *value:* compared by a regression suite;
- *VJP:* delegated to a trusted external provider;
- *provenance:* native symbol `torchlean_cuda_buffer_matmul`.

One strong field does not upgrade the others. In particular, a proof of shape safety is not a proof
of arithmetic, and a source-file link is provenance rather than evidence.

# Reproduce And Break The Boundary

The two most useful experiments are:

```
lake exe torchlean pytorch_export_check
lake -R -K cuda=true exe torchlean quickstart_mlp \
  --device cuda --steps 2 --show-backend
```

Then deliberately break one condition:

1. add an unsupported PyTorch operation and observe import rejection;
2. change a JSON shape and observe `checkShapes` reject it;
3. request CUDA from a stub build and observe runtime availability rejection;
4. pass a wrong-size Q buffer to the LibTorch SDPA test and observe the Lean/native guard reject it.

These failure paths are part of the contract. A boundary that reports only success is difficult to
audit and easy to overstate.
