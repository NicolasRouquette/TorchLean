# GraphSpec

GraphSpec is TorchLean's typed language for neural-network architectures. Parameter shapes, input
shapes, output shapes, and shared intermediate values are represented in Lean before the model is
executed.

```lean
import NN.GraphSpec
```

TorchLean has three graph-facing interfaces:

| Interface | Use |
| --- | --- |
| `TorchLean.nn` | direct model construction and training |
| `NN.GraphSpec` | typed architecture definitions with pure and executable interpretations |
| `NN.IR.Graph` | op-tagged graph artifacts for verification, export, and runtime tooling |

GraphSpec is useful for residual connections, recurrent cells, shared subgraphs, explicit parameter
ABIs, and model families whose architecture is itself part of a theorem.

## Sequential Graphs

`Chain ps σ τ` represents a chain from input shape `σ` to output shape `τ`. The list `ps`
records parameter tensor shapes in ABI order.

```lean
import NN.GraphSpec

open NN
open NN.GraphSpec

def g (inDim hidDim outDim : Nat) :=
  GraphSpec.Models.mlp inDim hidDim outDim

#check GraphSpec.Interp.spec (g 4 8 2)
#check GraphSpec.Chain.toProgram (g 4 8 2)
#check GraphSpec.LowerToDAG.Chain.toDAGModelZeroInit (g 4 8 2)
```

Sequential graphs work well for MLPs and feed-forward pipelines. `>>>` composes layers while the
type checker verifies adjacent shapes and concatenates parameter lists.

## Typed DAGs

The DAG language represents sharing and multi-input operations without storing untyped node
positions.

- `DAG.Var Γ s` selects a value of shape `s` from environment `Γ`.
- `DAG.Term Γ s` computes one tensor of shape `s`.
- `DAG.Args Γ shapes` stores one term for each shape in `shapes`.
- `DAG.Block Γ outputs` computes several outputs while preserving shared `let1` bindings.
- `DAG.Model ps ins out` packages parameters, inputs, and one output.
- `DAG.MultiModel ps ins outs` packages a shared computation with several typed outputs.

The shape index on `Var` prevents a variable from being read at the wrong tensor shape. Numeric
positions are converted through `Var.ofFin` only when a programmatic lowering discovers an index at
runtime.

`Term.rename`, `Term.substitute`, and `Term.instantiate` provide the usual operations for open
terms. Their types preserve every tensor shape. `Block.andThen` feeds all outputs of one block into
another block without duplicating shared intermediates.

## Semantics And Lowering

`Term.eval` and `Block.eval` give pure tensor semantics for any scalar `Context`.
`Term.lower` and `Block.lower` produce programs for any execution monad implementing TorchLean's
runtime operations.

The library proves that:

- evaluation commutes with variable renaming and typed substitution;
- instantiating a term evaluates as supplying its argument environment;
- `Block.andThen` evaluates as typed block composition;
- inlining a `Model` preserves its pure forward function;
- inlining a `MultiModel` preserves every output and shared intermediate.

These theorems let a large model be assembled from proved blocks. They do not require unfolding the
entire architecture for every later result.

## Files

| File | Contents |
| --- | --- |
| `Core.lean` | sequential chain syntax and composition |
| `DAG/Core.lean` | typed variables, terms, blocks, substitutions, semantics, and lowering |
| `DAG/Term.lean` | reusable term combinators |
| `DAG/Primitives/Core.lean` | primitive operation interface and basic operations |
| `DAG/Primitives/LinearAlgebra.lean` | matrix and batched linear algebra |
| `DAG/Primitives/Nonlinear.lean` | activations and elementwise nonlinearities |
| `DAG/Primitives/Normalization.lean` | normalization operations |
| `DAG/Primitives/Shape.lean` | reshape, broadcast, concat, slicing, and axis operations |
| `ToSequential.lean` | conversion of the supported layer-stack subset to `TorchLean.NN.Seq` |
| `Models/` | MLP, CNN, residual, and TorchLean lowering examples |

## Adding A Primitive

A primitive supplies a pure tensor function and an executable TorchLean program:

```lean
namespace NN.GraphSpec.Primitive

open Spec
open Tensor
open NN.Tensor

def myOp (s : Shape) : Primitive [] s s :=
  { name := "myOp"
    specFwd := fun {α} _ctx _params x => x
    program := fun {α} _ctx _deq =>
      fun {m} _ _ => fun x => pure x
    toLayerM? := none
    countsAsLayer := false }

end NN.GraphSpec.Primitive
```

Unary sequential primitives can be embedded in DAG syntax with
`LowerToDAG.Primitive.toDAGPrimOp`. A genuinely multi-input operation should define a
`DAG.PrimOp inputs output` directly.

Add semantic lemmas next to the primitive. Model-specific proofs can then simplify through the
primitive interface rather than unfold its implementation.

## Runtime Data

GraphSpec records architecture and parameter order. Optimizer state, device buffers, checkpoint
bytes, imported weights, and certificate files belong to the runtime, interoperability, or
verification modules. `NN.IR.Graph` is the lower-level artifact used when a verifier or exporter
needs an op-tagged graph.

## References

- Kaiming He et al., [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), 2016.
- Ron Cytron et al., [Efficiently Computing Static Single Assignment Form and the Control Dependence Graph](https://doi.org/10.1145/115372.115320), 1991.
- Atılım Güneş Baydin et al., [Automatic Differentiation in Machine Learning: a Survey](https://jmlr.org/papers/v18/17-468.html), 2018.
