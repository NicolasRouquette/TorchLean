# `NN/IR`

`NN.IR` is TorchLean's op tagged SSA/DAG intermediate representation. It is the small shared graph
language that model compilers, verification passes, exporters, widgets, and compiled-runtime
correctness proofs can all point at without each inventing a private graph format.

For public use, prefer the broad library import or the IR entrypoint. Internal code that only needs
one IR component should import the focused leaf directly.

```lean
import NN
-- or, if you only want this subsystem:
import NN.IR
-- or a focused internal leaf, for example:
import NN.IR.Graph
import NN.IR.Semantics
```

There is intentionally no extra top-level `NN.IR` umbrella. Use `NN` for the broad library import,
`NN.IR` for the IR subsystem, and the individual `NN.IR.*` files when a module needs a
precise internal dependency.

## What Belongs Here

- `Graph.lean`: graph syntax, node ids, op tags, arity conventions, and topological well formedness.
- `OpContracts.lean`: shared shape arithmetic for ops such as concat, matmul, pooling, conv, and
  axis-moving utilities. Concat rank checks and leading-axis output inference are list-indexed
  rather than duplicated by parent count.
- `Infer.lean`: the single source of truth for declared-output-shape validation.
- `Check.lean`: public validation wrappers and proposition-level `WellFormed` / `WellShaped` names.
- `Semantics.lean`: denotational evaluator into spec-layer tensor operations with explicit payloads,
  including the checked `Graph.expectLeadingAxisInput` boundary used by list-indexed concat, plus
  the scoped `IR` notation for graph denotation.
- `Pretty.lean`: readable text and GraphViz renderers for debugging.

## Relationship To `NN.GraphSpec`

`NN.GraphSpec` is a typed authoring DSL for model architectures, with a pure semantics and lowering
to TorchLean runtime programs. `NN.IR` is the lower-level op tagged graph target that
verification/export/runtime tooling can consume after a model has been compiled or traced.

In PyTorch terms, `NN.GraphSpec` is closer to a typed model construction DSL; `NN.IR` is closer to
an FX/TorchScript-style graph with explicit shapes and external parameter payloads.

## How A Model Reaches IR

There are several routes into the same graph language:

- TorchLean model code can lower supported fragments into IR for execution, inspection, or
  verification.
- GraphSpec models can be compiled or lowered when the architecture needs explicit sharing and
  named parameter layouts.
- PyTorch `torch.export` and ONNX adapters can write `torchlean.ir.v1` JSON, which Lean then parses
  and validates.
- Verification examples can build small graphs directly when the graph itself is the artifact under
  study.

Those routes are intentionally different producers with one consumer contract. Once a graph reaches
`NN.IR.Graph`, downstream code should be able to ask the same questions: are node ids topological,
are shapes inferred by the shared op contracts, which payloads are required, and what denotation does
the graph have in the spec layer?

## Role And Scope

The IR gives the runtime, checkers, exporters, and compiler passes one graph object to share. Write
ordinary models through `TorchLean.nn`, `Trainer`, or `GraphSpec`, then lower them. Construct `Node`
arrays directly only when testing an IR consumer.

Each runtime backend retains its own proof status. Proofs, tests, and trust-boundary statements say
how a particular runtime, compiler fragment, or certificate checker relates to the shared graph.

## Current Consumers

| Consumer | How it uses IR |
| --- | --- |
| Compiled runtime | Executes graph-shaped programs through runtime tensor values. |
| Verification | Runs IBP/CROWN-style passes, margin checks, and certificate replay over node ids and payloads. |
| TorchLean compiler fragments | Prove that supported source fragments compile to IR with the same denotation. |
| PyTorch/ONNX/export paths | Use a small graph format to make parameter order and tensor shapes explicit at the boundary. |
| Widgets and graph pages | Render graphs, inferred shapes, execution traces, and dependency structure for debugging. |

## Proof And Runtime Status

The IR is the object shared by proofs and runtime code, but proof coverage is still named
fragment-by-fragment. Current theorem work covers supported evaluator bridges, graph well-formedness
conditions, selected compiled-runtime fragments, and verification-oriented bound propagation. A new
operator should therefore add three things in the right places:

- its shape contract in `OpContracts`/`Infer`;
- its semantics in `Semantics` or a documented payload-backed evaluator;
- the runtime/proof/checker coverage that makes the operator usable in the intended workflow.

Adding a tag without these follow-up pieces creates a graph that can be printed but not responsibly
used for verified claims.

## Payload Discipline

The graph syntax stores operation structure. It does not smuggle learned tensors into node fields.
Weights, constants, and input boxes live in payload stores keyed by node id. This is slightly more
ceremonial than embedding everything in the node, but it is much easier to audit:

- graph topology can be checked independently of parameter values,
- payload shape mismatches are explicit errors,
- certificate and verifier code can cite the node id that owns each parameter or input box,
- imported weights can be treated as artifacts rather than trusted syntax.

## Release Invariants

- Node ids are array indices: `g.nodes[i].id = i`.
- Parents always point backward: every parent id is smaller than the child id.
- Parameter tensors are not embedded in `Graph`; `const`, `linear`, and `conv2d` use external
  payload stores keyed by node id.
- Shape checking is centralized through `Infer.inferNodeOutShape`; `Graph.checkShapes` delegates to
  that implementation to avoid duplicate op-contract logic.
